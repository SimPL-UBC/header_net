import os
import random
import numpy as np
import torch
import yaml
from pathlib import Path
from dataclasses import asdict
from .config import Config

def set_seed(seed: int):
    """
    Sets seed for Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_run_dir(output_root: str, run_name: str) -> Path:
    """
    Creates or overwrites output_root/run_name and returns its Path.
    """
    run_dir = Path(output_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Create checkpoints directory inside run_dir
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    
    return run_dir

def save_config(config: Config, run_dir: Path):
    """
    Writes the config dataclass to config.yaml inside the run directory.
    """
    config_dict = asdict(config)
    config_path = run_dir / "config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)
