"""
Inference & Explainability Engine for Backend REST API.
======================================================
Loads Spatial and Frequency backbones (when available) or runs the
high-efficiency physics & surface forensics engine, producing unified
predictions along with base64-encoded diagnostic visualizations.
"""

import base64
import gc
import io
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from xai.visualizer import XAIVisualizer


class MultimodalInferenceEngine:
    def __init__(
        self,
        branch_a_checkpoint: Optional[str] = "checkpoints/imagenet_ai_0419_biggan/best_model.pt",
        branch_b_checkpoint: Optional[str] = "branch_b/models/frequency/best_model.pt",
        alpha: float = 0.51,
        device: Optional[str] = None,
    ):
        self.alpha = alpha

        self.has_branch_a_ckpt = bool(branch_a_checkpoint and Path(branch_a_checkpoint).is_file())
        self.has_branch_b_ckpt = bool(branch_b_checkpoint and Path(branch_b_checkpoint).is_file())

        self.spatial_model = None
        self.spectral_model = None
        self.spatial_cam = None
        self.spectral_xai = None
        self.transform = None

        # Only import and initialize heavy neural weights if checkpoints actually exist on disk
        if self.has_branch_a_ckpt or self.has_branch_b_ckpt:
            import torch
            import torch.nn as nn
            from torchvision import transforms
            from torchvision.models import resnet50

            self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

            from xai.gradcam import GradCAM
            from xai.spectral_xai import SpectralXAI

            if self.has_branch_a_ckpt:
                self.spatial_model = self._load_spatial_model(branch_a_checkpoint)
                self.spatial_cam = GradCAM(self.spatial_model, use_cuda=(self.device.type == "cuda"))

            if self.has_branch_b_ckpt:
                self.spectral_model = self._load_spectral_model(branch_b_checkpoint)
                self.spectral_xai = SpectralXAI(self.spectral_model, use_cuda=(self.device.type == "cuda"))

    def _load_spatial_model(self, checkpoint_path: str):
        import torch
        import torch.nn as nn
        from torchvision.models import resnet50

        model = resnet50(weights=None)
        model.fc = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(model.fc.in_features, 1))
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        try:
            model.load_state_dict(state_dict)
        except Exception:
            model.load_state_dict(state_dict, strict=False)
        model.to(self.device).eval()
        return model

    def _load_spectral_model(self, checkpoint_path: str):
        import torch
        import torch.nn as nn
        from torchvision.models import resnet50

        model = resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        try:
            model.load_state_dict(state_dict)
        except Exception:
            model.load_state_dict(state_dict, strict=False)
        model.to(self.device).eval()
        return model

    def predict_and_explain(self, pil_image: Image.Image) -> dict:
        """Run classification and generate base64-encoded visual explanations."""
        from xai.spectral_xai import (
            image_to_fft_spectrum,
            compute_radial_profile,
            detect_spectral_physics_anomaly,
        )

        pil_image = pil_image.convert("RGB")

        # Branch 1: If full neural checkpoints are loaded
        if self.has_branch_a_ckpt and self.has_branch_b_ckpt:
            from xai.gradcam import apply_colormap_on_image

            in_tensor = self.transform(pil_image)
            spatial_cam, spatial_prob = self.spatial_cam.generate(in_tensor, target_class=1)
            spatial_overlay = apply_colormap_on_image(pil_image, spatial_cam, alpha=0.55)

            spectral_res = self.spectral_xai.explain(pil_image, self.transform, target_class=1)
            spectral_overlay = spectral_res["overlay"]
            fft_pil = spectral_res["fft_image"]
            band_attr = spectral_res["band_attribution"]
            radii = spectral_res["radial_radii"]
            profile = spectral_res["radial_profile"]
            spectral_prob = spectral_res["pred_prob"]
        else:
            # Branch 2: High-efficiency, zero-RAM physics & surface forensics engine (< 200MB memory)
            fft_pil, magnitude_raw = image_to_fft_spectrum(pil_image, size=224)
            radii, profile = compute_radial_profile(magnitude_raw, num_bins=100)
            physics_res = detect_spectral_physics_anomaly(
                raw_magnitude=magnitude_raw,
                radii=radii,
                profile=profile,
                original_image=pil_image,
            )

            spectral_prob = physics_res["prob_ai"]
            spatial_prob = spectral_prob

            # Generate surface texture anomaly overlay
            img_np = np.asarray(pil_image.resize((224, 224)), dtype=np.float32) / 255.0
            gray = 0.299 * img_np[:, :, 0] + 0.587 * img_np[:, :, 1] + 0.114 * img_np[:, :, 2]
            gy, gx = np.gradient(gray)
            grad_mag = np.sqrt(gx**2 + gy**2)

            from scipy.ndimage import gaussian_filter
            smoothed = gaussian_filter(grad_mag, sigma=2.0)
            variance = np.abs(grad_mag - smoothed)
            heatmap = gaussian_filter(variance, sigma=3.0)
            h_range = heatmap.max() - heatmap.min()
            heatmap = (heatmap - heatmap.min()) / max(h_range, 1e-6)

            cm = plt.get_cmap("jet")
            colored = cm(heatmap)[:, :, :3]
            spatial_overlay = (0.55 * colored + 0.45 * img_np).clip(0, 1)

            # Spectral overlay
            fft_np = np.asarray(fft_pil, dtype=np.float32) / 255.0
            mag_norm = magnitude_raw / max(float(magnitude_raw.max()), 1e-6)
            spectral_overlay = (0.55 * cm(mag_norm)[:, :, :3] + 0.45 * fft_np).clip(0, 1)

            # Band attribution based on radial profile
            low_energy = float(profile[:33].mean()) if len(profile) >= 33 else 1.0
            mid_energy = float(profile[33:66].mean()) if len(profile) >= 66 else 1.0
            high_energy = float(profile[66:].mean()) if len(profile) > 66 else 1.0
            total_e = max(low_energy + mid_energy + high_energy, 1e-6)

            band_attr = {
                "low_frequency_attribution": round(low_energy / total_e, 4),
                "mid_frequency_attribution": round(mid_energy / total_e, 4),
                "high_frequency_attribution": round(high_energy / total_e, 4),
            }

        # Calibrated late fusion
        fusion_prob = (self.alpha * spatial_prob) + ((1.0 - self.alpha) * spectral_prob)
        prediction = "AI" if fusion_prob >= 0.5 else "REAL"

        # Generate panel image in-memory as PNG buffer
        fig = XAIVisualizer.create_multimodal_diagnostic_panel(
            original_image=pil_image,
            spatial_overlay=spatial_overlay,
            spatial_prob=spatial_prob,
            fft_image=fft_pil,
            spectral_overlay=spectral_overlay,
            spectral_prob=spectral_prob,
            radial_radii=radii,
            radial_profile=profile,
            band_attribution=band_attr,
            fusion_prob=fusion_prob,
            save_path=None,
        )

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=90, bbox_inches="tight")
        plt.close(fig)
        plt.close("all")
        del fig
        gc.collect()

        buf.seek(0)
        panel_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        return {
            "prediction": prediction,
            "fusion_prob_ai": round(fusion_prob, 4),
            "spatial_prob_ai": round(spatial_prob, 4),
            "spectral_prob_ai": round(spectral_prob, 4),
            "confidence_percent": round(max(fusion_prob, 1.0 - fusion_prob) * 100, 2),
            "band_attribution": band_attr,
            "diagnostic_panel_base64": f"data:image/png;base64,{panel_b64}",
        }
