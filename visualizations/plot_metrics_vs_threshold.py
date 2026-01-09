from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from .utils import COLORS, apply_global_style, load_predictions


def _safe_metric(fn, y_true, y_pred) -> float:
    """Compute metric with zero_division handling."""
    return fn(y_true, y_pred, zero_division=0)


def plot_metrics_vs_threshold(
    y_true: Sequence[int],
    y_pred_proba: Sequence[float],
    save_path: Optional[Path] = None,
    thresholds: Optional[np.ndarray] = None,
) -> None:
    """Plot precision, recall, and F1 across probability thresholds."""
    apply_global_style()
    thresholds = thresholds if thresholds is not None else np.linspace(0.0, 1.0, 101)

    precisions = []
    recalls = []
    f1s = []
    for thr in thresholds:
        preds = (np.array(y_pred_proba) >= thr).astype(int)
        precisions.append(_safe_metric(precision_score, y_true, preds))
        recalls.append(_safe_metric(recall_score, y_true, preds))
        f1s.append(_safe_metric(f1_score, y_true, preds))

    fig, ax = plt.subplots()
    ax.plot(thresholds, precisions, label="Precision", color=COLORS["positive"], linewidth=2)
    ax.plot(thresholds, recalls, label="Recall", color=COLORS["train"], linewidth=2)
    ax.plot(thresholds, f1s, label="F1", color=COLORS["val"], linewidth=2)
    ax.axvline(0.5, color=COLORS["baseline"], linestyle="--", linewidth=1.5, label="Threshold = 0.5")

    # Annotate metrics at threshold 0.5
    idx_05 = np.argmin(np.abs(thresholds - 0.5))
    text = f"@0.5 → P={precisions[idx_05]:.2f}, R={recalls[idx_05]:.2f}, F1={f1s[idx_05]:.2f}"
    ax.text(0.52, 0.95, text, transform=ax.transAxes, fontsize=9, va="top")

    ax.set_xlabel("Threshold")
    ax.set_ylabel("Metric value")
    ax.set_title("Precision, Recall, F1 vs Threshold (Header Class)")
    ax.set_ylim(0, 1.05)
    ax.legend()
    plt.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)


if __name__ == "__main__":
    y_true, y_pred_proba, _ = load_predictions()
    plot_metrics_vs_threshold(y_true, y_pred_proba)
    plt.show()
