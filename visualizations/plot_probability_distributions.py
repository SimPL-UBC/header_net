from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .utils import COLORS, apply_global_style, load_predictions


def plot_probability_distributions(
    y_true: Sequence[int],
    y_pred_proba: Sequence[float],
    save_path: Optional[Path] = None,
    bins: int = 25,
) -> None:
    """Plot overlapping predicted probability histograms for each class."""
    apply_global_style()
    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)

    fig, ax = plt.subplots()
    ax.hist(
        y_pred_proba[y_true == 1],
        bins=bins,
        range=(0, 1),
        density=True,
        alpha=0.5,
        color=COLORS["positive"],
        label="True header",
    )
    ax.hist(
        y_pred_proba[y_true == 0],
        bins=bins,
        range=(0, 1),
        density=True,
        alpha=0.5,
        color=COLORS["negative"],
        label="True not_header",
    )
    ax.axvline(0.5, color=COLORS["baseline"], linestyle="--", linewidth=1.5, label="Threshold = 0.5")
    ax.set_xlabel("Predicted probability p(header)")
    ax.set_ylabel("Density")
    ax.set_title("Predicted Probability Distributions – Header vs Not Header")
    ax.legend()
    plt.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)


if __name__ == "__main__":
    y_true, y_pred_proba, _ = load_predictions()
    plot_probability_distributions(y_true, y_pred_proba)
    plt.show()
