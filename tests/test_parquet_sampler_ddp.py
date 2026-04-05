from pathlib import Path
import importlib.machinery
import sys
import types

import numpy as np

HEADER_NET_ROOT = Path(__file__).resolve().parents[1]
if str(HEADER_NET_ROOT) not in sys.path:
    sys.path.append(str(HEADER_NET_ROOT))


if "decord" not in sys.modules:
    fake_decord = types.ModuleType("decord")
    fake_decord.__spec__ = importlib.machinery.ModuleSpec("decord", loader=None)
    fake_decord.bridge = types.SimpleNamespace(set_bridge=lambda *_args, **_kwargs: None)
    fake_decord.cpu = lambda: None
    fake_decord.VideoReader = object
    sys.modules["decord"] = fake_decord

if "inference" not in sys.modules:
    inference_module = types.ModuleType("inference")
    inference_module.__spec__ = importlib.machinery.ModuleSpec("inference", loader=None)
    preprocessing_module = types.ModuleType("inference.preprocessing")
    preprocessing_module.__spec__ = importlib.machinery.ModuleSpec(
        "inference.preprocessing", loader=None
    )
    frame_cropper_module = types.ModuleType("inference.preprocessing.frame_cropper")
    frame_cropper_module.__spec__ = importlib.machinery.ModuleSpec(
        "inference.preprocessing.frame_cropper", loader=None
    )

    class FrameCropper:
        def __init__(self, *args, **kwargs):
            pass

    frame_cropper_module.FrameCropper = FrameCropper
    preprocessing_module.frame_cropper = frame_cropper_module
    inference_module.preprocessing = preprocessing_module
    sys.modules["inference"] = inference_module
    sys.modules["inference.preprocessing"] = preprocessing_module
    sys.modules["inference.preprocessing.frame_cropper"] = frame_cropper_module


from training.data.parquet_header_dataset import DeterministicRatioSampler


def test_deterministic_ratio_sampler_ddp_preserves_global_epoch_plan():
    positive_indices = np.array([0, 1], dtype=np.int64)
    negative_indices = np.array([2, 3, 4], dtype=np.int64)
    group_codes = np.array([0, 0, 1, 1, 2], dtype=np.int64)
    order_values = np.array([10, 20, 30, 40, 50], dtype=np.int64)

    single_process_sampler = DeterministicRatioSampler(
        positive_indices=positive_indices,
        negative_indices=negative_indices,
        neg_pos_ratio=1,
        seed=123,
        shuffle=True,
        group_codes=group_codes,
        order_values=order_values,
    )
    distributed_rank0 = DeterministicRatioSampler(
        positive_indices=positive_indices,
        negative_indices=negative_indices,
        neg_pos_ratio=1,
        seed=123,
        shuffle=True,
        group_codes=group_codes,
        order_values=order_values,
        num_replicas=2,
        rank=0,
    )
    distributed_rank1 = DeterministicRatioSampler(
        positive_indices=positive_indices,
        negative_indices=negative_indices,
        neg_pos_ratio=1,
        seed=123,
        shuffle=True,
        group_codes=group_codes,
        order_values=order_values,
        num_replicas=2,
        rank=1,
    )

    single_process_sampler.set_epoch(3)
    distributed_rank0.set_epoch(3)
    distributed_rank1.set_epoch(3)

    np.testing.assert_array_equal(
        distributed_rank0.get_global_indices(),
        single_process_sampler.get_global_indices(),
    )
    np.testing.assert_array_equal(
        distributed_rank1.get_global_indices(),
        single_process_sampler.get_global_indices(),
    )
    assert distributed_rank0.get_counts() == single_process_sampler.get_counts()
    assert distributed_rank1.get_counts() == single_process_sampler.get_counts()
    assert len(distributed_rank0) == len(distributed_rank1)


def test_deterministic_ratio_sampler_ddp_pads_evenly_across_ranks():
    sampler_rank0 = DeterministicRatioSampler(
        positive_indices=np.array([0], dtype=np.int64),
        negative_indices=np.array([1, 2], dtype=np.int64),
        neg_pos_ratio=1,
        seed=7,
        shuffle=False,
        num_replicas=2,
        rank=0,
    )
    sampler_rank1 = DeterministicRatioSampler(
        positive_indices=np.array([0], dtype=np.int64),
        negative_indices=np.array([1, 2], dtype=np.int64),
        neg_pos_ratio=1,
        seed=7,
        shuffle=False,
        num_replicas=2,
        rank=1,
    )

    sampler_rank0.set_epoch(0)
    sampler_rank1.set_epoch(0)

    assert sampler_rank0.get_counts() == {
        "samples": 2,
        "positives": 1,
        "negatives": 1,
    }
    assert len(sampler_rank0.get_indices()) == len(sampler_rank1.get_indices()) == 1
