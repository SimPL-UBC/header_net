"""RF-DETR wrapper built on the official `rfdetr` package."""
from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

try:
    import rfdetr
    from rfdetr import (
        RFDETRBase,
        RFDETRLarge,
        RFDETRMedium,
        RFDETRNano,
        RFDETRSmall,
    )
except ImportError as exc:  # pragma: no cover - enforced at runtime
    raise ImportError(
        "The `rfdetr` package is required. Install it with `pip install rfdetr`."
    ) from exc

try:
    from supervision.detection.core import Detections
except ImportError as exc:  # pragma: no cover - installed via rfdetr dependency tree
    raise ImportError(
        "The `supervision` package is required. It is installed automatically with `rfdetr`."
    ) from exc

# Canonical 80-class COCO label list without the background class.
COCO80_CLASS_NAMES: Tuple[str, ...] = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)

CLASS_NAME_TO_ID: Dict[str, int] = {name: idx for idx, name in enumerate(COCO80_CLASS_NAMES)}


def _instantiate_quietly(factory, **kwargs):
    """Call `factory(**kwargs)` while suppressing noisy stdout, surfacing failures only."""

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        instance = factory(**kwargs)
    for line in buffer.getvalue().splitlines():
        text = line.strip()
        if not text:
            continue
        lowered = text.lower()
        if "fail" in lowered or "error" in lowered:
            print(text)
    return instance


@dataclass
class RFDetrConfig:
    """Configuration for RF-DETR inference."""

    variant: str = "medium"
    weights_path: Optional[str] = None
    device: Optional[str] = None
    target_class_names: Sequence[str] = ("sports ball",)
    target_class_ids: Optional[Sequence[int]] = None
    optimize: bool = False
    optimize_batch_size: int = 1
    optimize_compile: bool = False

    def resolved_device(self) -> str:
        if self.device:
            return self.device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def rfdetr_device(self) -> str:
        dev = self.resolved_device()
        dev_lower = dev.lower()
        if dev_lower.startswith("cuda"):
            return "cuda"
        if dev_lower.startswith("cpu"):
            return "cpu"
        if dev_lower.startswith("mps"):
            return "mps"
        return dev


_VARIANT_MAP = {
    "nano": RFDETRNano,
    "small": RFDETRSmall,
    "medium": RFDETRMedium,
    "base": RFDETRBase,
    "large": RFDETRLarge,
}


def _normalise_variant(name: str) -> str:
    return name.lower().replace("_", "-")


def build_rf_detr(config: Optional[RFDetrConfig] = None):
    """Instantiate an RF-DETR model from the official package."""
    if config is None:
        config = RFDetrConfig()
    variant_key = _normalise_variant(config.variant)
    if variant_key not in _VARIANT_MAP:
        available = ", ".join(sorted(_VARIANT_MAP))
        raise ValueError(f"Unknown RF-DETR variant '{config.variant}'. Options: {available}")

    cls = _VARIANT_MAP[variant_key]
    kwargs = {
        "device": config.rfdetr_device(),
    }
    if config.weights_path:
        kwargs["pretrain_weights"] = str(config.weights_path)
    try:
        model = _instantiate_quietly(cls, **kwargs)
    except RuntimeError as exc:
        message = str(exc)
        requested_device = config.device or kwargs.get("device", "")
        wants_cuda = str(requested_device).lower().startswith("cuda")
        if wants_cuda and "cudaGetDeviceCount" in message:
            print("RF-DETR could not access CUDA; falling back to CPU.")
            config.device = "cpu"
            kwargs["device"] = "cpu"
            model = _instantiate_quietly(cls, **kwargs)
        else:
            raise
    return model


class RFDetrInference:
    """Thin wrapper around `rfdetr` predict API returning header_net-style detections."""

    def __init__(self, model, config: Optional[RFDetrConfig] = None):
        if config is None:
            config = RFDetrConfig()
        self.model = model
        self.config = config
        self.device = config.resolved_device()
        self._target_class_ids = self._resolve_target_classes(config)

        if config.optimize:
            # JIT compilation can take a while and is optional. Use with caution.
            self.model.optimize_for_inference(
                compile=config.optimize_compile,
                batch_size=config.optimize_batch_size,
            )

    @staticmethod
    def _resolve_target_classes(config: RFDetrConfig) -> Optional[set[int]]:
        if config.target_class_ids is not None:
            return {int(idx) for idx in config.target_class_ids}

        ids: set[int] = set()
        for name in config.target_class_names:
            key = name.lower().strip()
            if key in CLASS_NAME_TO_ID:
                ids.add(CLASS_NAME_TO_ID[key])
        return ids or None

    def _ensure_numpy(self, image: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                return np.stack([image] * 3, axis=-1)
            if image.ndim == 3 and image.shape[2] == 3:
                return image
            if image.ndim == 3 and image.shape[0] == 3:
                return np.transpose(image, (1, 2, 0))
            raise ValueError(f"Unsupported numpy image shape {image.shape}")

        if isinstance(image, torch.Tensor):
            tensor = image.detach().cpu()
            if tensor.ndim == 3 and tensor.shape[0] == 3:
                tensor = tensor.permute(1, 2, 0)
            elif tensor.ndim == 3 and tensor.shape[2] == 3:
                tensor = tensor
            else:
                raise ValueError(f"Unsupported tensor shape {tuple(tensor.shape)}")
            return tensor.numpy()

        raise TypeError(f"Unsupported image type: {type(image)!r}")

    def _detections_to_dicts(
        self,
        detections: Detections,
        topk: int,
    ) -> List[Dict[str, float]]:
        if len(detections) == 0:
            return []

        boxes_xyxy = detections.xyxy
        confidences = detections.confidence
        class_ids = detections.class_id

        records: List[Dict[str, float]] = []
        for box, score, class_id in zip(boxes_xyxy, confidences, class_ids):
            class_int = int(class_id)
            if self._target_class_ids is not None and class_int not in self._target_class_ids:
                continue
            x1, y1, x2, y2 = box.tolist()
            width = float(max(0.0, x2 - x1))
            height = float(max(0.0, y2 - y1))
            records.append(
                {
                    "box": [float(x1), float(y1), width, height],
                    "confidence": float(score),
                    "class_id": class_int,
                }
            )

        if not records:
            return []

        records.sort(key=lambda rec: rec.get("confidence", 0.0), reverse=True)
        return records[:topk]

    def __call__(
        self,
        images: Iterable[Union[np.ndarray, torch.Tensor]],
        score_threshold: float = 0.3,
        topk: int = 10,
    ) -> List[List[Dict[str, float]]]:
        prepared: List[np.ndarray] = []
        for img in images:
            arr = self._ensure_numpy(img)
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            prepared.append(arr)

        if not prepared:
            return []

        predictions = self.model.predict(prepared, threshold=score_threshold)
        if isinstance(predictions, Detections):
            predictions = [predictions]

        results: List[List[Dict[str, float]]] = []
        for det in predictions:
            results.append(self._detections_to_dicts(det, topk=topk))
        return results

    def load_weights(self, path: str, strict: bool = False) -> None:
        """Reload weights from disk using the RF-DETR helper."""
        # Standard RF-DETR constructors load weights automatically. This helper allows
        # swapping checkpoints post-construction without re-instantiating the model.
        weights = torch.load(path, map_location="cpu", weights_only=False)
        if "model" in weights:
            weights = weights["model"]
        missing, unexpected = self.model.model.load_state_dict(weights, strict=strict)
        if missing or unexpected:
            raise RuntimeError(
                f"Incompatible checkpoint when loading {path}: missing={missing}, unexpected={unexpected}"
            )


__all__ = [
    "RFDetrConfig",
    "RFDetrInference",
    "build_rf_detr",
]
