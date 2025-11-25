from pathlib import Path
from typing import Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE

from .utils import CLASS_NAMES, COLORS, apply_global_style

try:
    import umap  # type: ignore
except ImportError:
    umap = None


def _resolve_layer(model: torch.nn.Module, layer_name: str) -> torch.nn.Module:
    """Find a submodule by dotted name."""
    named_modules = dict(model.named_modules())
    if layer_name not in named_modules:
        raise ValueError(f"Layer '{layer_name}' not found in model. Available: {list(named_modules.keys())[:20]}...")
    return named_modules[layer_name]


def extract_features(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    feature_layer_name: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract penultimate features + labels + probabilities from a loader.

    feature_layer_name: dotted module name to hook (e.g., "backbone.layer4").
    If None, falls back to using logits as features.
    """
    model.eval()
    hook_handle = None
    feature_buffer = {}

    target_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    if feature_layer_name is not None:
        target_layer = _resolve_layer(target_model, feature_layer_name)

        def hook(_, __, output):
            feature_buffer["feat"] = output.detach()

        hook_handle = target_layer.register_forward_hook(hook)

    all_features = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                videos, labels, _ = batch
            else:
                videos, labels = batch
            videos = videos.to(device)
            labels = labels.to(device)

            logits = model(videos)
            probs = torch.softmax(logits, dim=1)[:, 1]

            if feature_layer_name is not None:
                if "feat" not in feature_buffer:
                    raise RuntimeError("Feature hook did not capture output. Check feature_layer_name.")
                feat = feature_buffer.pop("feat")
                if feat.dim() == 5:  # (B, C, T, H, W)
                    feat = feat.mean(dim=(2, 3, 4))
            else:
                feat = logits

            all_features.append(feat.cpu())
            all_labels.append(labels.cpu())
            all_probs.append(probs.cpu())

    if hook_handle is not None:
        hook_handle.remove()

    features = torch.cat(all_features).numpy()
    labels = torch.cat(all_labels).numpy()
    probs = torch.cat(all_probs).numpy()
    return features, labels, probs


def plot_embedding_2d(
    features: np.ndarray,
    labels: Sequence[int],
    probs: Sequence[float],
    method: str = "umap",
    class_names: Sequence[str] = CLASS_NAMES,
    save_path: Optional[Path] = None,
) -> None:
    """Project features to 2D using UMAP (if available) or t-SNE and plot."""
    apply_global_style()
    labels = np.array(labels)
    probs = np.array(probs)
    preds = (probs >= 0.5).astype(int)

    if method == "umap" and umap is not None:
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        embedding = reducer.fit_transform(features)
        title_suffix = "UMAP"
    else:
        n_samples = features.shape[0]
        if n_samples < 2:
            raise ValueError("Need at least 2 samples to plot embeddings.")
        perplexity = max(2, min(30, n_samples - 1))
        tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate="auto", init="random", random_state=42)
        embedding = tsne.fit_transform(features)
        title_suffix = "t-SNE"

    fig, ax = plt.subplots()
    class_colors = {0: COLORS["negative"], 1: COLORS["positive"]}

    for label in [0, 1]:
        mask = labels == label
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=30,
            alpha=0.7,
            c=class_colors[label],
            label=class_names[label],
        )

    misclassified = np.where(preds != labels)[0]
    if misclassified.size > 0:
        ax.scatter(
            embedding[misclassified, 0],
            embedding[misclassified, 1],
            s=60,
            facecolors="none",
            edgecolors="black",
            linewidths=1.5,
            label="Misclassified",
        )

    ax.set_title(f"2D Embedding of Video Features ({title_suffix})")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)
