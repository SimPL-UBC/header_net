import sys
from pathlib import Path
from typing import Type


def _import_resnet3d_csn() -> Type:
    """Dynamically import ResNet3dCSN from the legacy NFL codebase."""
    current_path = Path(__file__).resolve()
    repo_root = current_path.parents[3]
    csn_model_path = repo_root / "1st_place_kaggle_player_contact_detection" / "cnn" / "models"
    csn_root_path = repo_root / "1st_place_kaggle_player_contact_detection" / "cnn"

    for path in (csn_model_path, csn_root_path):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.append(path_str)

    from resnet3d_csn import ResNet3dCSN  # type: ignore

    return ResNet3dCSN


def build_csn_backbone(config):
    """
    Construct the CSN backbone using the same parameters as train_header.py.
    """
    ResNet3dCSN = _import_resnet3d_csn()
    model = ResNet3dCSN(
        pretrained2d=False,
        pretrained=None,
        depth=50,
        with_pool2=False,
        bottleneck_mode="ir",
        norm_eval=False,
        zero_init_residual=False,
        bn_frozen=False,
        num_classes=2,
    )
    return model
