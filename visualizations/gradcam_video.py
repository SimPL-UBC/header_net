from pathlib import Path
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from .utils import CLASS_NAMES, apply_global_style, even_frame_indices


class GradCAM3D:
    """
    Grad-CAM for 3D CNNs.

    target_layer should be a convolutional block (e.g., model.backbone.layer4) whose
    output shape is (N, C, T, H, W). For DataParallel models, pass model.module layers.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.fwd_handle = None
        self.bwd_handle = None
        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(_, __, output):
            self.activations = output.detach()

        def backward_hook(_, grad_input, grad_output):
            # grad_output is a tuple with one element for Conv/activation layers.
            self.gradients = grad_output[0].detach()

        self.fwd_handle = self.target_layer.register_forward_hook(forward_hook)
        # register_full_backward_hook is preferred for newer PyTorch; fallback for older versions.
        if hasattr(self.target_layer, "register_full_backward_hook"):
            self.bwd_handle = self.target_layer.register_full_backward_hook(backward_hook)
        else:
            self.bwd_handle = self.target_layer.register_backward_hook(backward_hook)  # type: ignore

    def remove_hooks(self) -> None:
        if self.fwd_handle is not None:
            self.fwd_handle.remove()
        if self.bwd_handle is not None:
            self.bwd_handle.remove()

    def generate(
        self, input_tensor: torch.Tensor, target_class: Optional[int] = None
    ) -> tuple[np.ndarray, float, int]:
        """
        Compute Grad-CAM heatmap.

        Args:
            input_tensor: shape (1, C, T, H, W) on the correct device.
            target_class: optional int (0 or 1); if None, uses predicted class.
        Returns:
            heatmap: numpy array (T, H, W) normalized to [0, 1]
            pred_prob: predicted probability for class 1 (header)
            chosen_class: class index used for backprop
        """
        self.model.eval()
        self.model.zero_grad()
        logits = self.model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_prob = probs[0, 1].item()
        if target_class is None:
            target_class = int(torch.argmax(logits, dim=1).item())

        score = logits[:, target_class].sum()
        score.backward(retain_graph=True)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

        # Handle different output shapes
        # 3D CNN: (B, C, T, H, W)
        # Transformer: (B, N, D) where N is tokens, D is embedding dim
        
        if self.gradients.ndim == 5:
            # CNN case: (B, C, T, H, W)
            # Average over spatial-temporal dims (2, 3, 4)
            weights = self.gradients.mean(dim=(2, 3, 4), keepdim=True)  # (B, C, 1, 1, 1)
            cam = (weights * self.activations).sum(dim=1)  # (B, T, H, W)
            
        elif self.gradients.ndim == 3:
            # Transformer case: (B, N, D)
            # Average over tokens (dim 1) to get weight per channel/embedding dim
            weights = self.gradients.mean(dim=1, keepdim=True)  # (B, 1, D)
            
            # Weighted sum over embedding dim (dim 2)
            cam = (weights * self.activations).sum(dim=2)  # (B, N)
            
            # Reshape N tokens back to (T', H', W')
            # Infer dimensions from input_tensor (B, C, T, H, W)
            # VideoMAE v2: tubelet=2, patch=16
            B, _, T, H, W = input_tensor.shape
            t_down = 2
            h_down = 16
            w_down = 16
            
            T_prime = T // t_down
            H_prime = H // h_down
            W_prime = W // w_down
            
            # Verify shape matches
            if cam.shape[1] == T_prime * H_prime * W_prime:
                cam = cam.reshape(B, T_prime, H_prime, W_prime)
            else:
                # Fallback: try to infer square spatial
                # Assuming T_prime is 8 for 16 frames
                N = cam.shape[1]
                # Try T_prime = 8
                if N % 8 == 0:
                    HW = N // 8
                    S = int(np.sqrt(HW))
                    if S * S == HW:
                        cam = cam.reshape(B, 8, S, S)
                    else:
                        raise ValueError(f"Could not reshape CAM with {N} tokens to spatial dimensions.")
                else:
                    raise ValueError(f"Could not reshape CAM with {N} tokens to spatial dimensions.")
        else:
            raise ValueError(f"Unsupported gradient shape: {self.gradients.shape}")

        cam = torch.relu(cam).squeeze(0)  # (T, H, W)

        cam_np = cam.cpu().numpy()
        cam_np -= cam_np.min()
        if cam_np.max() > 0:
            cam_np /= cam_np.max()
        return cam_np, float(pred_prob), int(target_class)


def visualize_gradcam_video(
    model: torch.nn.Module,
    gradcam: GradCAM3D,
    video_frames: np.ndarray,
    title_prefix: str = "",
    num_frames_to_show: int = 6,
    class_names: Sequence[str] = CLASS_NAMES,
    device: Optional[torch.device] = None,
    preprocess_fn: Optional[Callable[[np.ndarray], torch.Tensor]] = None,
    save_path: Optional[Path] = None,
) -> None:
    """
    Compute Grad-CAM for a video and plot overlays on sampled frames.
    video_frames: numpy array (T, H, W, 3) with values in [0,1] or [0,255].
    preprocess_fn: optional callable that takes a single frame (H, W, 3) np.uint8/float
                   and returns a normalized torch.Tensor of shape (C, H, W).
    """
    apply_global_style()
    frames = np.asarray(video_frames)
    if frames.dtype != np.float32 and frames.dtype != np.float64:
        frames = frames.astype(np.float32) / 255.0
    frames = np.clip(frames, 0.0, 1.0)
    T, H, W, _ = frames.shape

    if preprocess_fn is None:
        tensor = torch.from_numpy(frames).permute(3, 0, 1, 2).unsqueeze(0).float()
    else:
        processed = [preprocess_fn((frames[t] * 255).astype(np.uint8)) for t in range(T)]
        tensor = torch.stack(processed, dim=0).permute(1, 0, 2, 3).unsqueeze(0)  # (1, C, T, H, W)

    device = device or next(model.parameters()).device
    tensor = tensor.to(device)

    heatmap, pred_prob, pred_class = gradcam.generate(tensor)

    # Resize heatmap to match frame size for overlay.
    heatmap_tensor = torch.from_numpy(heatmap).unsqueeze(0).unsqueeze(0)  # (1, 1, T, Hc, Wc)
    heatmap_resized = (
        F.interpolate(
            heatmap_tensor,
            size=(T, H, W),
            mode="trilinear",
            align_corners=False,
        )
        .squeeze()
        .cpu()
        .numpy()
    )

    indices = even_frame_indices(T, num_frames_to_show)
    fig, axes = plt.subplots(1, len(indices), figsize=(3 * len(indices), 3))
    if len(indices) == 1:
        axes = [axes]

    for ax, idx in zip(axes, indices):
        ax.imshow(frames[idx])
        ax.imshow(heatmap_resized[idx], cmap="jet", alpha=0.45)
        ax.set_title(f"t={idx}")
        ax.axis("off")

    pred_label = class_names[pred_class]
    title = f"{title_prefix}Grad-CAM – Pred: {pred_label}, p(header)={pred_prob:.2f}"
    fig.suptitle(title)
    plt.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)


def demo_gradcam_on_subset(
    model: torch.nn.Module,
    test_videos_subset: Sequence[np.ndarray],
    y_true_subset: Sequence[int],
    y_pred_proba_subset: Sequence[float],
    class_names: Sequence[str],
    gradcam: GradCAM3D,
    num_examples: int = 3,
    device: Optional[torch.device] = None,
    preprocess_fn: Optional[Callable[[np.ndarray], torch.Tensor]] = None,
    save_dir: Optional[Path] = None,
) -> None:
    """
    Visualize Grad-CAM on a few correct headers and misclassified examples.
    """
    y_true = np.array(y_true_subset)
    probs = np.array(y_pred_proba_subset)
    preds = (probs >= 0.5).astype(int)

    correct_headers = np.where((y_true == 1) & (preds == 1))[0][:num_examples]
    misclassified = np.where(y_true != preds)[0][:num_examples]

    correct_counter = 0
    error_counter = 0

    for idx in correct_headers:
        visualize_gradcam_video(
            model,
            gradcam,
            test_videos_subset[idx],
            title_prefix="Correct header – ",
            num_frames_to_show=6,
            class_names=class_names,
            device=device,
            preprocess_fn=preprocess_fn,
            save_path=(save_dir / f"gradcam_correct_{correct_counter}.png") if save_dir else None,
        )
        correct_counter += 1

    for idx in misclassified:
        visualize_gradcam_video(
            model,
            gradcam,
            test_videos_subset[idx],
            title_prefix=f"Misclassified (true {class_names[y_true[idx]]}) – ",
            num_frames_to_show=6,
            class_names=class_names,
            device=device,
            preprocess_fn=preprocess_fn,
            save_path=(save_dir / f"gradcam_error_{error_counter}.png") if save_dir else None,
        )
        error_counter += 1
