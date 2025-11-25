from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve

from .utils import COLORS, apply_global_style, load_predictions


def plot_precision_recall_curve(
    y_true: Sequence[int],
    y_pred_proba: Sequence[float],
    save_path: Optional[Path] = None,
) -> None:
    """Plot Precision–Recall curve for the header class."""
    apply_global_style()
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    avg_precision = average_precision_score(y_true, y_pred_proba)

    fig, ax = plt.subplots()
    ax.plot(
        recall,
        precision,
        color=COLORS["positive"],
        linewidth=2,
        label=f"Avg Precision = {avg_precision:.3f}",
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall Curve – Header Class")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)


if __name__ == "__main__":
    y_true, y_pred_proba, _ = load_predictions()
    plot_precision_recall_curve(y_true, y_pred_proba)
    plt.show()
