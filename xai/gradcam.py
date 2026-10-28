"""
Grad-CAM and Grad-CAM++ implementations for PyTorch.
====================================================
Compatible with:
  - Single-logit binary classifiers (Branch A Spatial ResNet50, BCEWithLogits)
  - Two-logit multi-class classifiers (Branch B Spectral ResNet50, CrossEntropy)
  - Any standard PyTorch CNN backbone (ResNet, EfficientNet, ConvNeXt, etc.)
"""

from __future__ import annotations
from typing import List, Optional, Tuple, Union, Any
import numpy as np
from PIL import Image

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    torch = None
    nn = None
    F = None
    HAS_TORCH = False


class GradCAM:
    """Gradient-weighted Class Activation Mapping (Grad-CAM).

    Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via
    Gradient-based Localization", ICCV 2017.
    """

    def __init__(
        self,
        model,
        target_layer = None,
        use_cuda: bool = False,
    ):
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for GradCAM. Please run 'pip install torch'.")
        self.model = model
        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        """
        Args:
            model: PyTorch model in eval mode.
            target_layer: The convolutional layer to inspect. If None, defaults to model.layer4[-1].
            use_cuda: Whether to compute on CUDA device.
        """
        self.model = model
        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        if target_layer is None:
            # Default to the final residual block of ResNet
            if hasattr(self.model, "layer4"):
                target_layer = self.model.layer4[-1]
            else:
                raise ValueError("Target layer must be specified if model is not a standard ResNet.")

        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            # grad_output[0] is the gradient of target score w.r.t layer output
            self.gradients = grad_output[0].detach()

        self._hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self._hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def __del__(self):
        self.remove_hooks()

    def _extract_target_score(self, logits: torch.Tensor, target_class: int) -> torch.Tensor:
        """Extract scalar score for backpropagation.

        Handles:
          1) Single-logit output (Branch A, shape [B, 1] or [B]):
             - target_class == 1 (AI): score = logit
             - target_class == 0 (Real): score = -logit
          2) Multi-logit output (Branch B, shape [B, 2]):
             - score = logits[:, target_class]
        """
        if logits.dim() == 1 or (logits.dim() == 2 and logits.size(1) == 1):
            logit = logits.view(-1)[0]
            return logit if target_class == 1 else -logit
        else:
            return logits[0, target_class]

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: int = 1,
    ) -> Tuple[np.ndarray, float]:
        """Generate Grad-CAM activation map for a single input image tensor.

        Args:
            input_tensor: Normalized image tensor of shape [1, 3, H, W] or [3, H, W].
            target_class: 1 for AI / Fake, 0 for Nature / Real.

        Returns:
            cam: Normalized heatmap as float32 np.ndarray of shape (H, W) in range [0, 1].
            predicted_prob: Probability score of the target class.
        """
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)

        input_tensor = input_tensor.to(self.device).requires_grad_(True)
        self.model.zero_grad()

        # Forward pass
        logits = self.model(input_tensor)

        # Compute probability
        if logits.dim() == 1 or (logits.dim() == 2 and logits.size(1) == 1):
            prob = torch.sigmoid(logits.view(-1)[0]).item()
            pred_prob = prob if target_class == 1 else (1.0 - prob)
        else:
            probs = F.softmax(logits, dim=-1)
            pred_prob = probs[0, target_class].item()

        # Backward pass for the target score
        target_score = self._extract_target_score(logits, target_class)
        self.model.zero_grad()
        target_score.backward(retain_graph=False)

        # Gradients: [1, C, H_feat, W_feat]
        # Activations: [1, C, H_feat, W_feat]
        grads = self.gradients
        acts = self.activations

        if grads is None or acts is None:
            raise RuntimeError("Failed to capture activations or gradients. Check target layer.")

        # Global average pooling of gradients -> alpha_k
        weights = torch.mean(grads, dim=(2, 3), keepdim=True)  # [1, C, 1, 1]

        # Linear combination of feature maps weighted by alpha
        cam = torch.sum(weights * acts, dim=1, keepdim=True)  # [1, 1, H_feat, W_feat]
        cam = F.relu(cam)  # Only features positively contributing to target class

        # Upsample to input resolution
        _, _, in_h, in_w = input_tensor.shape
        cam = F.interpolate(cam, size=(in_h, in_w), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Min-max normalization
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam.astype(np.float32), float(pred_prob)


class GradCAMPlusPlus(GradCAM):
    """Grad-CAM++: Improved visual explanations for deep convolutional networks.

    Chattopadhyay et al., "Grad-CAM++: Generalized Gradient-based Visual Explanations
    for Deep Convolutional Networks", WACV 2018.
    Better at localizing multiple object instances and fine-grained synthesis artifacts.
    """

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: int = 1,
    ) -> Tuple[np.ndarray, float]:
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)

        input_tensor = input_tensor.to(self.device).requires_grad_(True)
        self.model.zero_grad()

        logits = self.model(input_tensor)

        if logits.dim() == 1 or (logits.dim() == 2 and logits.size(1) == 1):
            prob = torch.sigmoid(logits.view(-1)[0]).item()
            pred_prob = prob if target_class == 1 else (1.0 - prob)
        else:
            probs = F.softmax(logits, dim=-1)
            pred_prob = probs[0, target_class].item()

        target_score = self._extract_target_score(logits, target_class)
        self.model.zero_grad()
        target_score.backward(retain_graph=False)

        grads = self.gradients
        acts = self.activations

        if grads is None or acts is None:
            raise RuntimeError("Failed to capture activations or gradients. Check target layer.")

        # Compute Grad-CAM++ higher order weighting
        # grads: [1, C, H, W], acts: [1, C, H, W]
        grads_power_2 = grads.pow(2)
        grads_power_3 = grads.pow(3)

        sum_acts = acts.sum(dim=(2, 3), keepdim=True)
        eps = 1e-7

        aij = grads_power_2 / (2 * grads_power_2 + sum_acts * grads_power_3 + eps)
        aij = torch.where(grads != 0, aij, torch.zeros_like(aij))

        weights = torch.clamp(grads, min=0) * aij
        alphas = weights.sum(dim=(2, 3), keepdim=True)

        cam = torch.sum(alphas * acts, dim=1, keepdim=True)
        cam = F.relu(cam)

        _, _, in_h, in_w = input_tensor.shape
        cam = F.interpolate(cam, size=(in_h, in_w), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam.astype(np.float32), float(pred_prob)


def apply_colormap_on_image(
    original_image: Union[np.ndarray, Image.Image],
    activation_map: np.ndarray,
    alpha: float = 0.5,
    colormap_name: str = "jet",
) -> np.ndarray:
    """Superimpose an activation heatmap onto an image.

    Args:
        original_image: PIL Image or np.ndarray of shape (H, W, 3) in range [0, 255] or [0, 1].
        activation_map: 2D np.ndarray of shape (H, W) in range [0, 1].
        alpha: Heatmap blend weight (0 = original image only, 1 = heatmap only).
        colormap_name: Matplotlib colormap ('jet', 'viridis', 'magma', 'turbo').

    Returns:
        blended_image: RGB np.ndarray of shape (H, W, 3) with dtype uint8.
    """
    import matplotlib.cm as cm

    # Convert PIL to numpy if needed
    if isinstance(original_image, Image.Image):
        img_np = np.asarray(original_image.convert("RGB"), dtype=np.float32) / 255.0
    else:
        img_np = original_image.copy()
        if img_np.max() > 1.0:
            img_np = img_np.astype(np.float32) / 255.0

    # Ensure shape matches activation map
    h, w = activation_map.shape[:2]
    if img_np.shape[:2] != (h, w):
        pil_img = Image.fromarray((img_np * 255).astype(np.uint8))
        img_np = np.asarray(pil_img.resize((w, h)), dtype=np.float32) / 255.0

    # Get colormap compatible with all matplotlib versions (including 3.9+)
    import matplotlib as mpl
    if hasattr(mpl, "colormaps"):
        cmap = mpl.colormaps[colormap_name]
    else:
        import matplotlib.cm as cm
        cmap = cm.get_cmap(colormap_name)
    heatmap = cmap(activation_map)[:, :, :3]  # Drop alpha channel -> shape (H, W, 3), range [0, 1]

    # Weighted blend
    blended = alpha * heatmap + (1.0 - alpha) * img_np
    blended = np.clip(blended, 0.0, 1.0)
    return (blended * 255).astype(np.uint8)
