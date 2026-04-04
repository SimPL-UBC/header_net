from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

HEADER_NET_ROOT = Path(__file__).resolve().parents[1]
if str(HEADER_NET_ROOT) not in sys.path:
    sys.path.append(str(HEADER_NET_ROOT))

from training.config import merge_cli_args
import training.cli_train_header_parquet_train as parquet_train_cli
import training.engine.supervised_trainer as supervised_trainer


class RecordingAutocast:
    def __init__(self, calls, kwargs):
        self.calls = calls
        self.kwargs = kwargs

    def __enter__(self):
        self.calls.append(("enter", self.kwargs))
        return None

    def __exit__(self, exc_type, exc, tb):
        self.calls.append(("exit", self.kwargs))
        return False


class _ScaledLoss:
    def __init__(self, loss, scaler):
        self.loss = loss
        self.scaler = scaler

    def backward(self):
        self.scaler.backward_calls += 1
        self.loss.backward()


class FakeGradScaler:
    def __init__(self, device_type, enabled=False):
        self.device_type = device_type
        self.enabled = enabled
        self.scale_calls = 0
        self.backward_calls = 0
        self.step_calls = 0
        self.update_calls = 0

    def scale(self, loss):
        self.scale_calls += 1
        return _ScaledLoss(loss, self)

    def step(self, optimizer):
        self.step_calls += 1
        optimizer.step()

    def update(self):
        self.update_calls += 1

    def state_dict(self):
        return {"enabled": self.enabled}

    def load_state_dict(self, state_dict):
        self.enabled = bool(state_dict.get("enabled", self.enabled))


@pytest.mark.parametrize(
    ("cli_flags", "expected_amp", "expected_gradient_checkpointing"),
    [
        (["--amp", "--gradient_checkpointing"], True, True),
        (["--no-amp", "--no-gradient_checkpointing"], False, False),
    ],
)
def test_parse_args_and_merge_cli_for_runtime_toggles(
    monkeypatch,
    cli_flags,
    expected_amp,
    expected_gradient_checkpointing,
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--train_parquet",
            "train.parquet",
            "--run_name",
            "giant_run",
            *cli_flags,
        ],
    )

    args = parquet_train_cli.parse_args()
    config = merge_cli_args(args)

    assert args.amp is expected_amp
    assert args.gradient_checkpointing is expected_gradient_checkpointing
    assert config.amp is expected_amp
    assert config.gradient_checkpointing is expected_gradient_checkpointing


def test_train_one_epoch_uses_amp_autocast_and_scaler(monkeypatch):
    autocast_calls = []
    scaler_instances = []

    def fake_autocast(**kwargs):
        return RecordingAutocast(autocast_calls, kwargs)

    def fake_grad_scaler(device_type, enabled=False):
        scaler = FakeGradScaler(device_type, enabled=enabled)
        scaler_instances.append(scaler)
        return scaler

    monkeypatch.setattr(supervised_trainer.torch.amp, "autocast", fake_autocast)
    monkeypatch.setattr(supervised_trainer.torch.amp, "GradScaler", fake_grad_scaler)
    monkeypatch.setattr(torch.Tensor, "to", lambda self, *_args, **_kwargs: self, raising=False)

    trainer = supervised_trainer.Trainer(
        SimpleNamespace(loss_type="ce", amp=True),
        torch.device("cuda:0"),
    )
    model = nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loader = [(torch.randn(2, 4), torch.tensor([0, 1]), None)]

    metrics = trainer.train_one_epoch(model, loader, optimizer, epoch=1)

    assert trainer.amp_enabled is True
    assert scaler_instances[0].device_type == "cuda"
    assert scaler_instances[0].enabled is True
    assert scaler_instances[0].scale_calls == 1
    assert scaler_instances[0].backward_calls == 1
    assert scaler_instances[0].step_calls == 1
    assert scaler_instances[0].update_calls == 1
    assert autocast_calls[0][0] == "enter"
    assert autocast_calls[0][1]["device_type"] == "cuda"
    assert autocast_calls[0][1]["dtype"] == torch.float16
    assert metrics["train_loss"] >= 0.0


def test_train_one_epoch_without_amp_uses_full_precision():
    trainer = supervised_trainer.Trainer(
        SimpleNamespace(loss_type="ce", amp=False),
        torch.device("cpu"),
    )
    model = nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loader = [(torch.randn(2, 4), torch.tensor([0, 1]), None)]

    metrics = trainer.train_one_epoch(model, loader, optimizer, epoch=1)

    assert trainer.amp_enabled is False
    assert trainer.scaler is None
    assert metrics["train_loss"] >= 0.0
