from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc


CLASS_NAMES = ["not_header", "header"]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _save_fig(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_loss(metrics: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots()
    ax.plot(metrics["epoch"], metrics["train_loss"], label="train")
    ax.plot(metrics["epoch"], metrics["val_loss"], label="val")
    ax.set_title("Train vs Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    _save_fig(fig, output_path)


def _plot_f1_recall(metrics: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots()
    ax.plot(metrics["epoch"], metrics["val_f1"], label="val_f1")
    ax.plot(metrics["epoch"], metrics["val_recall"], label="val_recall")
    ax.set_title("Validation F1/Recall")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.legend()
    _save_fig(fig, output_path)


def _plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, output_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion Matrix (Best Epoch)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticklabels(CLASS_NAMES)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save_fig(fig, output_path)


def _plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots()
    if len(np.unique(y_true)) < 2:
        ax.text(0.5, 0.5, "ROC undefined: only one class present", ha="center", va="center")
        ax.axis("off")
    else:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        ax.plot([0, 1], [0, 1], "k--", linewidth=1)
        ax.set_title("ROC Curve")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right")
    _save_fig(fig, output_path)


def _plot_pr_curve(y_true: np.ndarray, y_prob: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots()
    if len(np.unique(y_true)) < 2:
        ax.text(0.5, 0.5, "PR undefined: only one class present", ha="center", va="center")
        ax.axis("off")
    else:
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ax.plot(recall, precision)
        ax.set_title("Precision-Recall Curve")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
    _save_fig(fig, output_path)


def generate_all_plots(run_dir: Path, metrics_path: Path, predictions_path: Path) -> None:
    run_dir = Path(run_dir)
    metrics_path = Path(metrics_path)
    predictions_path = Path(predictions_path)

    plots_dir = run_dir / "plots"
    _ensure_dir(plots_dir)

    if not metrics_path.exists():
        print(f"Metrics file not found: {metrics_path}")
        return

    metrics = pd.read_csv(metrics_path)

    _plot_loss(metrics, plots_dir / "loss_train_val.png")
    _plot_f1_recall(metrics, plots_dir / "f1_recall_val.png")

    if not predictions_path.exists():
        print(f"Predictions file not found: {predictions_path}")
        return

    preds = pd.read_csv(predictions_path)
    if "label" not in preds.columns:
        print(f"Predictions file missing 'label' column: {predictions_path}")
        return

    y_true = preds["label"].to_numpy()
    if "pred_label" in preds.columns:
        y_pred = preds["pred_label"].to_numpy()
    else:
        y_pred = (preds["prob_header"].to_numpy() >= 0.5).astype(int)

    if "prob_header" in preds.columns:
        y_prob = preds["prob_header"].to_numpy()
    else:
        y_prob = y_pred.astype(float)

    _plot_confusion_matrix(y_true, y_pred, plots_dir / "confusion_matrix_best.png")
    _plot_roc_curve(y_true, y_prob, plots_dir / "roc_curve.png")
    _plot_pr_curve(y_true, y_prob, plots_dir / "pr_curve.png")
