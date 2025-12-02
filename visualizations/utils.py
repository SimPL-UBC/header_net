import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


# Default location of the latest run outputs; override via function arguments if needed.
DEFAULT_RUN_DIR = Path(__file__).resolve().parents[1] / "scratch_output" / "csn_16frames_test"
CLASS_NAMES = ["not_header", "header"]
COLORS = {
    "train": "#1f77b4",
    "val": "#ff7f0e",
    "positive": "#d62728",
    "negative": "#2ca02c",
    "baseline": "#7f7f7f",
}


def get_run_dir(run_dir: Optional[Path] = None) -> Path:
    """Resolve the directory that contains metrics and predictions."""
    resolved = Path(run_dir) if run_dir is not None else DEFAULT_RUN_DIR
    return resolved.expanduser().resolve()


def load_epoch_metrics(run_dir: Optional[Path] = None) -> Dict[str, np.ndarray]:
    """
    Load per-epoch metrics from metrics_epoch.csv.

    Returns a dict with epochs, train_loss, val_loss, val_f1, val_acc.
    Note: training accuracy/F1 are not logged in the current training script,
    so callers should handle missing train curve gracefully.
    """
    run_path = get_run_dir(run_dir)
    metrics_path = run_path / "metrics_epoch.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics file not found at {metrics_path}")

    df = pd.read_csv(metrics_path)
    epochs = df["epoch"].to_numpy()
    train_loss = df["train_loss"].to_numpy()
    val_loss = df["val_loss"].to_numpy()
    val_f1 = df["val_f1"].to_numpy()
    val_acc = df["val_acc"].to_numpy()
    train_f1 = df["train_f1"].to_numpy() if "train_f1" in df.columns else None

    # train_f1 may be None for older runs.
    return {
        "epochs": epochs,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_f1": val_f1,
        "val_acc": val_acc,
        "train_f1": train_f1,
    }


def load_predictions(run_dir: Optional[Path] = None) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Load validation predictions (labels + probabilities) from val_predictions.csv.

    Returns (y_true, y_pred_proba, dataframe).
    """
    run_path = get_run_dir(run_dir)
    preds_path = run_path / "val_predictions.csv"
    if not preds_path.exists():
        raise FileNotFoundError(f"predictions file not found at {preds_path}")

    df = pd.read_csv(preds_path)
    y_true = df["label"].to_numpy()
    y_pred_proba = df["prob_header"].to_numpy()
    return y_true, y_pred_proba, df


def load_best_checkpoint_path(run_dir: Optional[Path] = None) -> Optional[Path]:
    """Return the path to the best checkpoint stored in best_metrics.json, if present."""
    run_path = get_run_dir(run_dir)
    best_path = run_path / "best_metrics.json"
    if not best_path.exists():
        return None

    with open(best_path, "r") as f:
        data = json.load(f)
    checkpoint_rel = data.get("checkpoint")
    if checkpoint_rel is None:
        return None
    return (run_path / checkpoint_rel).resolve()


def threshold_predictions(y_pred_proba: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Convert probabilities into binary predictions at a given threshold."""
    return (y_pred_proba >= threshold).astype(int)


def apply_global_style() -> None:
    """Apply a lightweight global matplotlib style to keep plots consistent."""
    plt.rcParams.update(
        {
            "axes.grid": True,
            "grid.alpha": 0.3,
            "figure.figsize": (10, 6),
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "legend.fontsize": 14,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "font.size": 13,
        }
    )


def to_device_if_tensor(x: Any, device: torch.device) -> Any:
    """Send a tensor to device if applicable; otherwise return unchanged."""
    if torch.is_tensor(x):
        return x.to(device)
    return x


def even_frame_indices(num_frames: int, num_samples: int) -> np.ndarray:
    """Return evenly spaced frame indices for a video."""
    num_samples = max(1, num_samples)
    num_frames = max(1, num_frames)
    return np.linspace(0, num_frames - 1, num_samples, dtype=int)
