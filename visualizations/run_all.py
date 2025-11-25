import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from .embeddings_2d import extract_features, plot_embedding_2d
from .gradcam_video import GradCAM3D, demo_gradcam_on_subset
from .plot_confusion_matrix import plot_confusion_matrix
from .plot_f1_curves import plot_f1_curves
from .plot_loss_curves import plot_loss_curves
from .plot_metrics_vs_threshold import plot_metrics_vs_threshold
from .plot_precision_recall_curve import plot_precision_recall_curve
from .plot_probability_distributions import plot_probability_distributions
from .plot_roc_curve import plot_roc_curve
from .qualitative_galleries import plot_correct_gallery, plot_error_gallery
from .utils import (
    CLASS_NAMES,
    DEFAULT_RUN_DIR,
    apply_global_style,
    load_epoch_metrics,
    load_predictions,
)

# Optional context file where you can define: model, device, gradcam_target_layer,
# test_loader, test_videos_subset, y_true_subset, y_pred_proba_subset.
try:
    from . import user_context  # type: ignore
except ImportError:
    user_context = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all header classifier visualizations.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR, help="Path to run directory with metrics/predictions.")
    parser.add_argument("--save-dir", type=Path, default=None, help="Directory to save figures (optional).")
    parser.add_argument("--no-show", action="store_true", help="Do not call plt.show() at the end.")
    parser.add_argument("--enable-gradcam", action="store_true", help="Run Grad-CAM visualizations (requires user_context).")
    parser.add_argument("--enable-embedding", action="store_true", help="Run embedding visualization (requires user_context.test_loader).")
    parser.add_argument("--enable-galleries", action="store_true", help="Run qualitative galleries (requires user_context test videos).")
    parser.add_argument(
        "--feature-layer-name",
        type=str,
        default=None,
        help="Named layer to hook for embeddings (e.g., 'backbone.layer4').",
    )
    parser.add_argument(
        "--embedding-method",
        type=str,
        default="umap",
        choices=["umap", "tsne"],
        help="Dimensionality reduction method for embeddings.",
    )
    return parser.parse_args()


def maybe_save_path(save_dir: Path, name: str) -> Path:
    return save_dir / f"{name}.png"


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    save_dir = args.save_dir
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    apply_global_style()

    metrics = load_epoch_metrics(run_dir)
    y_true, y_pred_proba, _ = load_predictions(run_dir)

    plot_loss_curves(metrics["epochs"], metrics["train_loss"], metrics["val_loss"], save_path=maybe_save_path(save_dir, "loss") if save_dir else None)
    plot_f1_curves(metrics["epochs"], metrics["train_f1"], metrics["val_f1"], save_path=maybe_save_path(save_dir, "f1") if save_dir else None)
    plot_roc_curve(y_true, y_pred_proba, save_path=maybe_save_path(save_dir, "roc") if save_dir else None)
    plot_precision_recall_curve(y_true, y_pred_proba, save_path=maybe_save_path(save_dir, "precision_recall") if save_dir else None)
    plot_metrics_vs_threshold(y_true, y_pred_proba, save_path=maybe_save_path(save_dir, "metrics_vs_threshold") if save_dir else None)
    plot_probability_distributions(y_true, y_pred_proba, save_path=maybe_save_path(save_dir, "probability_distributions") if save_dir else None)
    plot_confusion_matrix(y_true, y_pred_proba, class_names=CLASS_NAMES, threshold=0.5, save_path=maybe_save_path(save_dir, "confusion_matrix") if save_dir else None)

    # Optional: Grad-CAM on subset
    if args.enable_gradcam:
        if user_context is None:
            print("Grad-CAM skipped: create visualizations/user_context.py with model, gradcam_target_layer, test_videos_subset, y_true_subset, y_pred_proba_subset.")
        else:
            model = getattr(user_context, "model", None)
            gradcam_target_layer = getattr(user_context, "gradcam_target_layer", None)
            videos = getattr(user_context, "test_videos_subset", None)
            y_true_subset = getattr(user_context, "y_true_subset", None)
            y_pred_proba_subset = getattr(user_context, "y_pred_proba_subset", None)
            device = getattr(user_context, "device", None)
            preprocess_fn = getattr(user_context, "preprocess_fn", None)
            if model is None or gradcam_target_layer is None or videos is None or y_true_subset is None or y_pred_proba_subset is None:
                print("Grad-CAM skipped: missing model/target layer or test subset in user_context.")
            else:
                gradcam = GradCAM3D(model, gradcam_target_layer)
                demo_gradcam_on_subset(
                    model,
                    videos,
                    y_true_subset,
                    y_pred_proba_subset,
                    CLASS_NAMES,
                    gradcam,
                    num_examples=3,
                    device=device,
                    preprocess_fn=preprocess_fn,
                    save_dir=save_dir,
                )

    # Optional: Embedding projection
    if args.enable_embedding:
        if user_context is None:
            print("Embedding visualization skipped: create visualizations/user_context.py with model and test_loader.")
        else:
            model = getattr(user_context, "model", None)
            test_loader = getattr(user_context, "test_loader", None)
            device = getattr(user_context, "device", None)
            if model is None or test_loader is None:
                print("Embedding visualization skipped: missing model or test_loader in user_context.")
            else:
                device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model.to(device)
                features, labels, probs = extract_features(model, test_loader, device, feature_layer_name=args.feature_layer_name)
                plot_embedding_2d(
                    features,
                    labels,
                    probs,
                    method=args.embedding_method,
                    class_names=CLASS_NAMES,
                    save_path=maybe_save_path(save_dir, "embedding") if save_dir else None,
                )

    # Optional: qualitative galleries
    if args.enable_galleries:
        if user_context is None:
            print("Galleries skipped: create visualizations/user_context.py with test_videos_subset, y_true_subset, y_pred_proba_subset.")
        else:
            videos = getattr(user_context, "test_videos_subset", None)
            y_true_subset = getattr(user_context, "y_true_subset", None)
            y_pred_proba_subset = getattr(user_context, "y_pred_proba_subset", None)
            if videos is None or y_true_subset is None or y_pred_proba_subset is None:
                print("Galleries skipped: missing required arrays in user_context.")
            else:
                plot_error_gallery(
                    videos,
                    y_true_subset,
                    y_pred_proba_subset,
                    class_names=CLASS_NAMES,
                    max_videos=8,
                    save_path=maybe_save_path(save_dir, "error_gallery") if save_dir else None,
                )
                plot_correct_gallery(
                    videos,
                    y_true_subset,
                    y_pred_proba_subset,
                    class_names=CLASS_NAMES,
                    max_videos=8,
                    save_path=maybe_save_path(save_dir, "correct_gallery") if save_dir else None,
                )

    if not args.no_show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
