from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .utils import COLORS, apply_global_style, load_epoch_metrics


def plot_f1_curves(
    epochs: Sequence[float],
    train_f1: Optional[Sequence[float]],
    val_f1: Sequence[float],
    save_path: Optional[Path] = None,
) -> None:
    """
    Plot training vs validation F1 (or accuracy).

    train_f1 can be None if training F1/accuracy was not logged; in that case
    only the validation curve is shown with a note.
    """
    apply_global_style()
    fig, ax = plt.subplots()
    has_train = train_f1 is not None and len(train_f1) == len(epochs)
    if has_train:
        ax.plot(epochs, train_f1, label="Train F1/Acc", color=COLORS["train"], linewidth=2)
    else:
        ax.text(
            0.02,
            0.02,
            "Training F1/accuracy not logged; showing validation only.",
            transform=ax.transAxes,
            fontsize=9,
            alpha=0.8,
            ha="left",
            va="bottom",
        )

    ax.plot(epochs, val_f1, label="Val F1", color=COLORS["val"], linewidth=2, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("F1 Score")
    ax.set_title("Training and Validation F1 Score")
    ax.set_ylim(0, 1.05)
    ax.legend()
    plt.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)


if __name__ == "__main__":
    metrics = load_epoch_metrics()
    plot_f1_curves(metrics["epochs"], metrics["train_f1"], metrics["val_f1"])
    plt.show()
