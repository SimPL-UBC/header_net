import hashlib
import importlib.util
import json
import sys
import torch
import torch.nn as nn
from pathlib import Path
from safetensors.torch import load_file

from .resnet3d_csn_local import ResNet3dCSN


_DEFAULT_VMAE_CKPT = Path("checkpoints/VideoMAEv2-Base")
_VMAE_REQUIRED_MODEL_CONFIG_KEYS = (
    "img_size",
    "patch_size",
    "in_chans",
    "embed_dim",
    "depth",
    "num_heads",
    "mlp_ratio",
    "qkv_bias",
    "qk_scale",
    "drop_rate",
    "attn_drop_rate",
    "drop_path_rate",
    "norm_layer",
    "layer_norm_eps",
    "init_values",
    "use_learnable_pos_emb",
    "tubelet_size",
    "use_mean_pooling",
    "with_cp",
    "cos_attn",
)

def build_csn_backbone(config):
        
    # Constructs model: num_classes=2, depth 50 (as per train_header.py)
    model = ResNet3dCSN(
        pretrained2d=False,
        pretrained=None,
        depth=50,
        with_pool2=False,
        bottleneck_mode='ir',
        norm_eval=False,
        zero_init_residual=False,
        bn_frozen=False,
        num_classes=2
    )
    return model


class VideoMAEBackbone(nn.Module):
    """Wrapper for VideoMAE model to use as a feature extractor."""
    
    def __init__(self, vmae_model, hidden_dim):
        super().__init__()
        self.vmae = vmae_model
        self.hidden_dim = hidden_dim
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, C, T, H, W)
        Returns:
            features: Tensor of shape (B, hidden_dim)
        """
        # VideoMAE expects (B, C, T, H, W) which matches our format
        features = self.vmae.forward_features(x)
        return features


def _resolve_vmae_checkpoint_dir(config) -> Path:
    ckpt_dir = Path(config.backbone_ckpt) if config.backbone_ckpt is not None else _DEFAULT_VMAE_CKPT
    if not ckpt_dir.exists():
        raise ValueError(f"Checkpoint directory not found: {ckpt_dir}")
    return ckpt_dir


def _load_vmae_model_config(ckpt_dir: Path) -> dict:
    config_path = ckpt_dir / "config.json"
    if not config_path.exists():
        raise ValueError(f"VideoMAE checkpoint config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        checkpoint_config = json.load(f)

    model_config = checkpoint_config.get("model_config")
    if not isinstance(model_config, dict):
        raise ValueError(
            f"VideoMAE checkpoint config is missing a valid model_config: {config_path}"
        )

    return dict(model_config)


def _build_vmae_constructor_kwargs(model_config: dict, num_frames: int) -> dict:
    missing = [key for key in _VMAE_REQUIRED_MODEL_CONFIG_KEYS if key not in model_config]
    if missing:
        raise ValueError(
            "VideoMAE checkpoint model_config is missing required field(s): "
            + ", ".join(sorted(missing))
        )

    kwargs = {key: model_config[key] for key in _VMAE_REQUIRED_MODEL_CONFIG_KEYS}
    kwargs["num_classes"] = 0
    kwargs["num_frames"] = num_frames
    return kwargs


def _apply_vmae_runtime_overrides(model_kwargs: dict, config) -> dict:
    override = getattr(config, "gradient_checkpointing", None)
    if override is not None:
        model_kwargs["with_cp"] = bool(override)
    return model_kwargs


def _should_log(config) -> bool:
    return bool(getattr(config, "is_main_process", True))


def _module_cache_key(prefix: str, path: Path) -> str:
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:12]
    return f"_header_net_{prefix}_{digest}"


def _load_module_from_path(module_name: str, module_path: Path):
    cached_module = sys.modules.get(module_name)
    if cached_module is not None:
        return cached_module

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to create import spec for {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def _load_vmae_checkpoint_module(ckpt_dir: Path):
    config_module_path = ckpt_dir / "modeling_config.py"
    model_module_path = ckpt_dir / "modeling_videomaev2.py"
    if not config_module_path.exists():
        raise ValueError(f"VideoMAE checkpoint modeling config not found: {config_module_path}")
    if not model_module_path.exists():
        raise ValueError(f"VideoMAE checkpoint model implementation not found: {model_module_path}")

    config_module = _load_module_from_path(
        _module_cache_key("videomaev2_config", config_module_path),
        config_module_path,
    )

    previous_modeling_config = sys.modules.get("modeling_config")
    sys.modules["modeling_config"] = config_module
    try:
        return _load_module_from_path(
            _module_cache_key("videomaev2_model", model_module_path),
            model_module_path,
        )
    finally:
        if previous_modeling_config is None:
            sys.modules.pop("modeling_config", None)
        else:
            sys.modules["modeling_config"] = previous_modeling_config


def build_vmae_backbone(config):
    """
    Build VideoMAE v2 backbone from checkpoint.
    
    Args:
        config: Config object with backbone_ckpt path
        
    Returns:
        VideoMAEBackbone module
    """
    ckpt_dir = _resolve_vmae_checkpoint_dir(config)
    model_config = _load_vmae_model_config(ckpt_dir)
    model_kwargs = _build_vmae_constructor_kwargs(model_config, config.num_frames)
    model_kwargs = _apply_vmae_runtime_overrides(model_kwargs, config)
    modeling_videomaev2 = _load_vmae_checkpoint_module(ckpt_dir)

    vision_transformer_cls = getattr(modeling_videomaev2, "VisionTransformer", None)
    if vision_transformer_cls is None:
        raise ValueError(
            f"VideoMAE checkpoint module does not define VisionTransformer: {ckpt_dir / 'modeling_videomaev2.py'}"
        )

    hidden_dim = int(model_config["embed_dim"])
    if _should_log(config):
        print(
            "Resolved VideoMAE backbone: "
            f"checkpoint={ckpt_dir}, "
            f"patch_size={model_kwargs['patch_size']}, "
            f"embed_dim={hidden_dim}, "
            f"depth={model_kwargs['depth']}, "
            f"num_heads={model_kwargs['num_heads']}, "
            f"use_mean_pooling={model_kwargs['use_mean_pooling']}, "
            f"with_cp={model_kwargs['with_cp']}, "
            f"num_frames={model_kwargs['num_frames']}"
        )

    model = vision_transformer_cls(**model_kwargs)
    
    # Load pretrained weights from safetensors
    safetensors_path = ckpt_dir / "model.safetensors"
    if safetensors_path.exists():
        state_dict = load_file(str(safetensors_path))
        
        # Remove 'model.' prefix if present
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace("model.", "") if k.startswith("model.") else k
            new_state_dict[new_key] = v
        
        # Load with strict=False to handle missing head weights
        load_result = model.load_state_dict(new_state_dict, strict=False)
        if isinstance(load_result, tuple):
            missing, unexpected = load_result
        else:
            missing = getattr(load_result, "missing_keys", [])
            unexpected = getattr(load_result, "unexpected_keys", [])
        if _should_log(config):
            print(f"Loaded VideoMAE checkpoint from {safetensors_path}")
            if missing:
                print(f"Missing keys: {len(missing)} total")
            if unexpected:
                print(f"Unexpected keys: {len(unexpected)} total")
    else:
        if _should_log(config):
            print(
                f"Warning: No safetensors checkpoint found at {safetensors_path}, "
                "using random initialization"
            )
    
    return VideoMAEBackbone(model, hidden_dim)
