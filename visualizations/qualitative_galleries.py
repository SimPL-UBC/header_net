from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .utils import CLASS_NAMES, apply_global_style, even_frame_indices


def sample_keyframes(video_frames: np.ndarray, num_frames: int = 4) -> np.ndarray:
    """Return indices of evenly spaced keyframes."""
    return even_frame_indices(len(video_frames), num_frames)


def plot_video_gallery(
    videos: Sequence[np.ndarray],
    y_true: Sequence[int],
    y_pred_proba: Sequence[float],
    class_names: Sequence[str],
    indices: Sequence[int],
    title: str,
    num_frames_per_video: int = 4,
    max_videos: int = 8,
    save_path: Optional[Path] = None,
) -> None:
    """Plot a grid of sampled frames per video."""
    apply_global_style()
    indices = list(indices)[:max_videos]
    if not indices:
        print("No videos to display for gallery.")
        return

    n_rows = len(indices)
    fig_width = num_frames_per_video * 4.5
    fig_height = n_rows * 4.0
    fig, axes = plt.subplots(n_rows, num_frames_per_video, figsize=(fig_width, fig_height))
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, idx in enumerate(indices):
        frames = np.asarray(videos[idx])
        if frames.dtype != np.float32 and frames.dtype != np.float64:
            frames = frames.astype(np.float32) / 255.0
        frame_ids = sample_keyframes(frames, num_frames_per_video)

        true_label = class_names[y_true[idx]]
        prob_header = float(y_pred_proba[idx])
        pred_class = int(prob_header >= 0.5)
        pred_label = class_names[pred_class]
        pred_prob = prob_header if pred_class == 1 else 1 - prob_header
        row_label = f"True: {true_label} | Pred: {pred_label} ({pred_prob*100:.1f}%)"

        for col, frame_idx in enumerate(frame_ids):
            ax = axes[row, col]
            ax.imshow(frames[frame_idx])
            ax.axis("off")
            if col == 0:
                ax.text(
                    -0.12,
                    0.5,
                    row_label,
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    fontsize=14,
                    fontweight="bold",
                    wrap=True,
                )
            ax.set_title(f"t={frame_idx}", fontsize=12)

    fig.suptitle(title, fontsize=18)
    fig.subplots_adjust(left=0.26, right=0.98, top=0.92, wspace=0.08, hspace=0.35)
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")


def plot_error_gallery(
    test_videos_subset: Sequence[np.ndarray],
    y_true_subset: Sequence[int],
    y_pred_proba_subset: Sequence[float],
    class_names: Sequence[str] = CLASS_NAMES,
    max_videos: int = 8,
    save_path: Optional[Path] = None,
) -> None:
    """Select misclassified videos and visualize them."""
    probs = np.array(y_pred_proba_subset)
    y_true = np.array(y_true_subset)
    preds = (probs >= 0.5).astype(int)
    misclassified = np.where(preds != y_true)[0]
    title = "Qualitative Error Gallery – Misclassified Videos"
    plot_video_gallery(
        test_videos_subset,
        y_true_subset,
        y_pred_proba_subset,
        class_names,
        indices=misclassified,
        title=title,
        max_videos=max_videos,
        save_path=save_path,
    )


def plot_correct_gallery(
    test_videos_subset: Sequence[np.ndarray],
    y_true_subset: Sequence[int],
    y_pred_proba_subset: Sequence[float],
    class_names: Sequence[str] = CLASS_NAMES,
    max_videos: int = 8,
    save_path: Optional[Path] = None,
) -> None:
    """Select correctly classified videos (mix of both classes if available)."""
    probs = np.array(y_pred_proba_subset)
    y_true = np.array(y_true_subset)
    preds = (probs >= 0.5).astype(int)

    correct_header = list(np.where((preds == y_true) & (y_true == 1))[0])
    correct_non_header = list(np.where((preds == y_true) & (y_true == 0))[0])

    selected = []
    while len(selected) < max_videos and (correct_header or correct_non_header):
        if correct_header:
            selected.append(correct_header.pop(0))
        if len(selected) < max_videos and correct_non_header:
            selected.append(correct_non_header.pop(0))

    title = "Qualitative Correct Gallery – Correctly Classified Videos"
    plot_video_gallery(
        test_videos_subset,
        y_true_subset,
        y_pred_proba_subset,
        class_names,
        indices=selected,
        title=title,
        max_videos=max_videos,
        save_path=save_path,
    )
