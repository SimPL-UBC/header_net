from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

from .utils import CLASS_NAMES, COLORS, apply_global_style, load_predictions, threshold_predictions


def plot_confusion_matrix(
    y_true: Sequence[int],
    y_pred_proba: Sequence[float],
    class_names: Sequence[str] = CLASS_NAMES,
    threshold: float = 0.5,
    save_path: Optional[Path] = None,
) -> None:
    """Plot a 2x2 confusion matrix heatmap at a fixed threshold."""
    apply_global_style()
    preds = threshold_predictions(np.array(y_pred_proba), threshold=threshold)
    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    cm_percent = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1e-9)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="YlGnBu")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels([f"Pred {name}" for name in class_names])
    ax.set_yticklabels([f"True {name}" for name in class_names])
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(f"Confusion Matrix at Threshold {threshold}")

    # Annotate counts and row percentages.
    text_contrast_threshold = cm.max() * 0.45 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            cell_value = cm[i, j]
            text_color = "white" if cell_value > text_contrast_threshold else "black"
            ax.text(
                j,
                i,
                f"{cell_value}\n{cm_percent[i, j] * 100:.1f}%",
                ha="center",
                va="center",
                color=text_color,
                fontsize=13,
                )

    plt.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)


if __name__ == "__main__":
    y_true, y_pred_proba, _ = load_predictions()
    plot_confusion_matrix(y_true, y_pred_proba)
    plt.show()
