"""
Inference & Explainability Engine for Backend REST API.
======================================================
Loads Spatial and Frequency backbones and produces unified predictions
along with base64-encoded Grad-CAM and Spectral diagnostic visualizations.
"""

import base64
import io
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights

from xai.gradcam import GradCAM, apply_colormap_on_image
from xai.spectral_xai import SpectralXAI
from xai.visualizer import XAIVisualizer


class MultimodalInferenceEngine:
    def __init__(
        self,
        branch_a_checkpoint: Optional[str] = "checkpoints/imagenet_ai_0419_biggan/best_model.pt",
        branch_b_checkpoint: Optional[str] = "branch_b/models/frequency/best_model.pt",
        alpha: float = 0.51,
        device: Optional[str] = None,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.alpha = alpha

        self.has_branch_a_ckpt = bool(branch_a_checkpoint and Path(branch_a_checkpoint).is_file())
        self.has_branch_b_ckpt = bool(branch_b_checkpoint and Path(branch_b_checkpoint).is_file())

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        self.spatial_model = self._load_spatial_model(branch_a_checkpoint)
        self.spectral_model = self._load_spectral_model(branch_b_checkpoint)

        self.spatial_cam = GradCAM(self.spatial_model, use_cuda=(self.device.type == "cuda"))
        self.spectral_xai = SpectralXAI(self.spectral_model, use_cuda=(self.device.type == "cuda"))

    def _load_spatial_model(self, checkpoint_path: Optional[str]) -> nn.Module:
        has_ckpt = checkpoint_path and Path(checkpoint_path).is_file()
        hub_dir = Path(torch.hub.get_dir()) / "checkpoints"
        has_cached = bool(list(hub_dir.glob("resnet50*.pth"))) if hub_dir.exists() else False

        weights = ResNet50_Weights.DEFAULT if (not has_ckpt and has_cached) else None
        model = resnet50(weights=weights)
        model.fc = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(model.fc.in_features, 1))

        if has_ckpt:
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            try:
                model.load_state_dict(state_dict)
            except Exception:
                model.load_state_dict(state_dict, strict=False)

        model.to(self.device).eval()
        return model

    def _load_spectral_model(self, checkpoint_path: Optional[str]) -> nn.Module:
        has_ckpt = checkpoint_path and Path(checkpoint_path).is_file()
        hub_dir = Path(torch.hub.get_dir()) / "checkpoints"
        has_cached = bool(list(hub_dir.glob("resnet50*.pth"))) if hub_dir.exists() else False

        weights = ResNet50_Weights.DEFAULT if (not has_ckpt and has_cached) else None
        model = resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, 2)

        if has_ckpt:
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
        pil_image = pil_image.convert("RGB")
        in_tensor = self.transform(pil_image)

        # Spatial prediction & Grad-CAM
        spatial_cam, spatial_prob = self.spatial_cam.generate(in_tensor, target_class=1)
        spatial_overlay = apply_colormap_on_image(pil_image, spatial_cam, alpha=0.55)

        # Spectral prediction & Spectral Grad-CAM
        spectral_res = self.spectral_xai.explain(pil_image, self.transform, target_class=1)
        spectral_overlay = spectral_res["overlay"]
        fft_pil = spectral_res["fft_image"]
        band_attr = spectral_res["band_attribution"]

        # Physics anomaly detection when running without neural checkpoints
        from xai.spectral_xai import detect_spectral_physics_anomaly
        physics_res = detect_spectral_physics_anomaly(
            raw_magnitude=spectral_res["raw_magnitude"],
            radii=spectral_res["radial_radii"],
            profile=spectral_res["radial_profile"],
            original_image=pil_image,
        )

        if not self.has_branch_b_ckpt:
            spectral_prob = physics_res["prob_ai"]
        else:
            spectral_prob = spectral_res["pred_prob"]

        if not self.has_branch_a_ckpt:
            spatial_prob = spectral_prob

        # Late fusion
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
            radial_radii=spectral_res["radial_radii"],
            radial_profile=spectral_res["radial_profile"],
            band_attribution=band_attr,
            fusion_prob=fusion_prob,
            save_path=None,
        )

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
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
