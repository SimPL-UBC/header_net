from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .utils import COLORS, apply_global_style, load_epoch_metrics


def plot_loss_curves(
    epochs: Sequence[float],
    train_loss: Sequence[float],
    val_loss: Sequence[float],
    save_path: Optional[Path] = None,
) -> None:
    """Plot training vs validation loss curves."""
    apply_global_style()
    fig, ax = plt.subplots()
    ax.plot(epochs, train_loss, label="Train loss", color=COLORS["train"], linewidth=2)
    ax.plot(epochs, val_loss, label="Val loss", color=COLORS["val"], linewidth=2, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Loss (3D CNN Header Classifier)")
    ax.legend()
    plt.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)


if __name__ == "__main__":
    metrics = load_epoch_metrics()
    plot_loss_curves(metrics["epochs"], metrics["train_loss"], metrics["val_loss"])
    plt.show()
