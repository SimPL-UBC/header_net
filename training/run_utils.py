import logging
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import yaml

from .config import Config, config_to_dict


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_run_dir(output_root: str, run_name: str) -> Path:
    """
    Create (or recreate) the run directory under output_root.
    Existing contents are removed to guarantee a clean slate.
    """
    root_path = Path(output_root)
    run_dir = root_path / run_name
    try:
        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        logging.error("Failed to create run directory %s: %s", run_dir, exc)
        raise
    return run_dir


def save_config(config: Config, run_dir: Path) -> None:
    """Persist the Config dataclass to config.yaml inside the run directory."""
    config_path = run_dir / "config.yaml"
    try:
        with config_path.open("w") as f:
            yaml.safe_dump(config_to_dict(config), f, default_flow_style=False)
    except Exception as exc:
        logging.error("Failed to write config to %s: %s", config_path, exc)
        raise
