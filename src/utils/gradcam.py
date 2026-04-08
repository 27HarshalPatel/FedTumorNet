"""Grad-CAM explainability for Brain Tumor MRI classification."""
from pathlib import Path
from typing import List, Optional
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PIL import Image
from src.data.preprocessing import get_inverse_normalize
from src.data.dataset import CLASS_NAMES

def generate_gradcam(model: nn.Module, image_tensor: torch.Tensor,
                     target_class: int, target_layer=None):
    """Generate Grad-CAM heatmap for one image."""
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        from pytorch_grad_cam.utils.image import show_cam_on_image

        # Default: last ResNet layer
        if target_layer is None:
            if hasattr(model, "layer4"):
                target_layer = model.layer4[-1]
            elif hasattr(model, "features"):
                target_layer = model.features[-1]
            else:
                return None, None

        model.eval()
        cam = GradCAM(model=model, target_layers=[target_layer])
        targets = [ClassifierOutputTarget(target_class)]

        grayscale_cam = cam(input_tensor=image_tensor.unsqueeze(0), targets=targets)
        grayscale_cam = grayscale_cam[0]

        # Convert tensor to RGB image for overlay
        inv_norm = get_inverse_normalize()
        img_rgb = inv_norm(image_tensor).permute(1, 2, 0).numpy()
        img_rgb = np.clip(img_rgb, 0, 1).astype(np.float32)
        visualization = show_cam_on_image(img_rgb, grayscale_cam, use_rgb=True)
        return grayscale_cam, visualization

    except ImportError:
        print("grad-cam library not installed. Run: pip install grad-cam")
        return None, None

def generate_gradcam_grid(model: nn.Module, dataloader, class_names=CLASS_NAMES,
                           num_per_class: int = 2, save_path: str = None):
    """4-class × num_per_class grid of Grad-CAM visualizations."""
    model.eval()
    device = next(model.parameters()).device
    collected = {c: [] for c in range(len(class_names))}

    for imgs, labels in dataloader:
        for img, lbl in zip(imgs, labels):
            c = lbl.item()
            if len(collected[c]) < num_per_class:
                collected[c].append(img)
        if all(len(v) >= num_per_class for v in collected.values()): break

    nrows = len(class_names); ncols = num_per_class * 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))

    inv_norm = get_inverse_normalize()

    for row, cls_name in enumerate(class_names):
        images = collected[row]
        col = 0
        for img in images[:num_per_class]:
            _, vis = generate_gradcam(model, img.to(device), target_class=row)
            # Original
            rgb = inv_norm(img).permute(1, 2, 0).numpy()
            rgb = np.clip(rgb, 0, 1)
            axes[row, col].imshow(rgb); axes[row, col].axis("off")
            if col == 0: axes[row, col].set_ylabel(cls_name, fontsize=10, fontweight="bold")
            axes[row, col].set_title("Original" if row == 0 else "")
            col += 1
            # Grad-CAM
            if vis is not None:
                axes[row, col].imshow(vis)
            else:
                axes[row, col].imshow(rgb)
            axes[row, col].axis("off")
            axes[row, col].set_title("Grad-CAM" if row == 0 else "")
            col += 1

    plt.suptitle("Grad-CAM Attention Maps by Tumor Class", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Grad-CAM grid → {save_path}")
    plt.close(fig)
