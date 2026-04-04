from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

HEADER_NET_ROOT = Path(__file__).resolve().parents[1]
if str(HEADER_NET_ROOT) not in sys.path:
    sys.path.append(str(HEADER_NET_ROOT))

import training.models.backbones as backbones


BASE_CKPT = HEADER_NET_ROOT / "checkpoints" / "VideoMAEv2-Base"
GIANT_CKPT = HEADER_NET_ROOT / "checkpoints" / "VideoMAEv2-giant"


class FakeVisionTransformer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.init_kwargs = dict(kwargs)
        self.loaded_state_dict = None
        self.strict_arg = None

    def forward_features(self, x):
        return x

    def load_state_dict(self, state_dict, strict=False):
        self.loaded_state_dict = dict(state_dict)
        self.strict_arg = strict
        return [], []


def test_base_constructor_kwargs_match_checkpoint_config():
    model_config = backbones._load_vmae_model_config(BASE_CKPT)

    kwargs = backbones._build_vmae_constructor_kwargs(model_config, num_frames=16)

    assert kwargs["patch_size"] == 16
    assert kwargs["embed_dim"] == 768
    assert kwargs["depth"] == 12
    assert kwargs["use_mean_pooling"] is True
    assert kwargs["num_frames"] == 16
    assert kwargs["num_classes"] == 0


def test_giant_constructor_kwargs_match_checkpoint_config():
    model_config = backbones._load_vmae_model_config(GIANT_CKPT)

    kwargs = backbones._build_vmae_constructor_kwargs(model_config, num_frames=16)

    assert kwargs["patch_size"] == 14
    assert kwargs["embed_dim"] == 1408
    assert kwargs["depth"] == 40
    assert kwargs["use_mean_pooling"] is False
    assert kwargs["num_frames"] == 16
    assert kwargs["num_classes"] == 0


@pytest.mark.parametrize(
    ("ckpt_dir", "expected_patch_size", "expected_embed_dim", "expected_depth", "expected_use_mean_pooling"),
    [
        (BASE_CKPT, 16, 768, 12, True),
        (GIANT_CKPT, 14, 1408, 40, False),
    ],
)
def test_build_vmae_backbone_uses_checkpoint_specific_kwargs(
    monkeypatch,
    ckpt_dir,
    expected_patch_size,
    expected_embed_dim,
    expected_depth,
    expected_use_mean_pooling,
):
    fake_module = SimpleNamespace(VisionTransformer=FakeVisionTransformer)

    monkeypatch.setattr(backbones, "_load_vmae_checkpoint_module", lambda _: fake_module)
    monkeypatch.setattr(
        backbones,
        "load_file",
        lambda _path: {"model.fake_weight": torch.tensor([1.0])},
    )

    cfg = SimpleNamespace(backbone_ckpt=str(ckpt_dir), num_frames=16)

    backbone = backbones.build_vmae_backbone(cfg)

    assert isinstance(backbone, backbones.VideoMAEBackbone)
    assert backbone.hidden_dim == expected_embed_dim
    assert backbone.vmae.init_kwargs["patch_size"] == expected_patch_size
    assert backbone.vmae.init_kwargs["embed_dim"] == expected_embed_dim
    assert backbone.vmae.init_kwargs["depth"] == expected_depth
    assert backbone.vmae.init_kwargs["use_mean_pooling"] is expected_use_mean_pooling
    assert backbone.vmae.init_kwargs["num_frames"] == 16
    assert backbone.vmae.init_kwargs["num_classes"] == 0
    assert backbone.vmae.strict_arg is False
    assert "fake_weight" in backbone.vmae.loaded_state_dict
    assert "model.fake_weight" not in backbone.vmae.loaded_state_dict


def test_load_vmae_model_config_requires_model_config(tmp_path):
    ckpt_dir = tmp_path / "VideoMAEv2-invalid"
    ckpt_dir.mkdir()
    (ckpt_dir / "config.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="model_config"):
        backbones._load_vmae_model_config(ckpt_dir)


def test_build_vmae_constructor_kwargs_requires_required_fields():
    with pytest.raises(ValueError, match="patch_size"):
        backbones._build_vmae_constructor_kwargs({"img_size": 224}, num_frames=16)
