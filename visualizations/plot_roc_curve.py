from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve

from .utils import COLORS, apply_global_style, load_predictions


def plot_roc_curve(
    y_true: Sequence[int],
    y_pred_proba: Sequence[float],
    save_path: Optional[Path] = None,
) -> None:
    """Plot ROC curve for the header class (positive class)."""
    apply_global_style()
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc_value = auc(fpr, tpr)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color=COLORS["positive"], linewidth=2, label=f"AUC = {auc_value:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color=COLORS["baseline"], label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve – Header vs Not Header")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)


if __name__ == "__main__":
    y_true, y_pred_proba, _ = load_predictions()
    plot_roc_curve(y_true, y_pred_proba)
    plt.show()
