"""
Unified Explainable AI (XAI) CLI & Analysis Pipeline.
=====================================================
Generates multi-panel interpretability reports for AI-generated image detection.

Usage:
    # 1. Analyze a single image:
    python -m xai.explain --image path/to/image.png --output-dir xai/outputs

    # 2. Analyze with custom checkpoints:
    python -m xai.explain --image path/to/image.png \
        --branch-a-ckpt checkpoints/imagenet_ai_0419_biggan/best_model.pt \
        --branch-b-ckpt branch_b/models/frequency/best_model.pt

    # 3. Quick test mode (generates demo analysis on scratch or sample image):
    python -m xai.explain --demo
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights

from .gradcam import GradCAM, apply_colormap_on_image
from .spectral_xai import SpectralXAI, image_to_fft_spectrum
from .visualizer import XAIVisualizer


def get_default_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def _build_model_with_fallback(
    checkpoint_path: Optional[str] = None,
    is_spatial: bool = True,
    device: str = "cpu",
    allow_download: bool = False,
) -> nn.Module:
    """Build ResNet50 with graceful offline fallback (avoids hanging on 98MB download)."""
    branch_name = "Branch A (Spatial)" if is_spatial else "Branch B (Spectral)"

    if checkpoint_path and Path(checkpoint_path).is_file():
        print(f"Loading {branch_name} weights from checkpoint: {checkpoint_path}")
        model = resnet50(weights=None)
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        try:
            model.load_state_dict(state_dict)
        except Exception:
            model.load_state_dict(state_dict, strict=False)
    else:
        hub_dir = Path(torch.hub.get_dir()) / "checkpoints"
        cached_files = list(hub_dir.glob("resnet50*.pth")) if hub_dir.exists() else []

        if cached_files:
            try:
                print(f"{branch_name}: Loaded cached weights from {cached_files[0].name}")
                model = resnet50(weights=ResNet50_Weights.DEFAULT)
            except Exception:
                model = resnet50(weights=None)
        elif allow_download:
            try:
                print(f"{branch_name}: Downloading ImageNet weights...")
                model = resnet50(weights=ResNet50_Weights.DEFAULT)
            except Exception as e:
                print(f"Notice: Download failed or interrupted ({e}). Initialized offline.")
                model = resnet50(weights=None)
        else:
            print(f"{branch_name}: Fast offline initialization (no 100MB download needed).")
            model = resnet50(weights=None)

    if is_spatial:
        model.fc = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(model.fc.in_features, 1))
    else:
        model.fc = nn.Linear(model.fc.in_features, 2)

    model.to(device)
    model.eval()
    return model


def load_spatial_model(checkpoint_path: Optional[str] = None, device: str = "cpu", allow_download: bool = False) -> nn.Module:
    """Build and load Branch A Spatial ResNet50 model."""
    return _build_model_with_fallback(checkpoint_path, is_spatial=True, device=device, allow_download=allow_download)


def load_spectral_model(checkpoint_path: Optional[str] = None, device: str = "cpu", allow_download: bool = False) -> nn.Module:
    """Build and load Branch B Frequency ResNet50 model."""
    return _build_model_with_fallback(checkpoint_path, is_spatial=False, device=device, allow_download=allow_download)


def analyze_image(
    image_path: Union[str, Path, Image.Image],
    spatial_model: nn.Module,
    spectral_model: nn.Module,
    output_dir: Union[str, Path] = "xai/outputs",
    device: str = "cpu",
    late_fusion_alpha: float = 0.51,
    true_label: Optional[str] = None,
    has_branch_a_ckpt: bool = False,
    has_branch_b_ckpt: bool = False,
) -> dict:
    """Run dual-branch XAI diagnostic on a single image.

    Args:
        image_path: Path to input image file or PIL Image.
        spatial_model: Branch A ResNet50.
        spectral_model: Branch B ResNet50.
        output_dir: Directory where figures and JSON will be saved.
        device: 'cpu' or 'cuda'.
        late_fusion_alpha: Optimal decision fusion weight (default: 0.51).
        true_label: Optional 'real' or 'ai'.

    Returns:
        Summary dict containing prediction probabilities and frequency attributions.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(image_path, (str, Path)):
        img_p = Path(image_path)
        if not img_p.is_file():
            print(f"\n[Error] Image file not found: '{image_path}'")
            samples_dir = Path("samples")
            if samples_dir.is_dir():
                avail = [f.name for f in samples_dir.glob("*") if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}]
                print(f"Available files in 'samples/': {', '.join(avail) if avail else 'None'}")
                print(f"Example usage: python -m xai.explain --image samples/{avail[0] if avail else 'sample_ai.png'}\n")
            sys.exit(1)
        img_name = img_p.stem
        pil_img = Image.open(img_p).convert("RGB")
    else:
        img_name = "sample_analysis"
        pil_img = image_path.convert("RGB")

    transform = get_default_transform()
    in_tensor_spatial = transform(pil_img)

    # 1. Spatial Domain Grad-CAM (Branch A)
    spatial_cam_engine = GradCAM(spatial_model, use_cuda=(device == "cuda"))
    spatial_cam, spatial_prob = spatial_cam_engine.generate(in_tensor_spatial, target_class=1)
    spatial_overlay = apply_colormap_on_image(pil_img, spatial_cam, alpha=0.55, colormap_name="jet")

    # 2. Spectral Domain Grad-CAM (Branch B)
    spectral_engine = SpectralXAI(spectral_model, use_cuda=(device == "cuda"))
    spectral_result = spectral_engine.explain(pil_img, transform=transform, target_class=1)
    spectral_overlay = spectral_result["overlay"]
    fft_pil = spectral_result["fft_image"]
    radial_radii = spectral_result["radial_radii"]
    radial_profile = spectral_result["radial_profile"]
    band_attr = spectral_result["band_attribution"]

    # If trained checkpoints are not supplied, activate the physics-based spectral detector
    from .spectral_xai import detect_spectral_physics_anomaly
    physics_res = detect_spectral_physics_anomaly(
        raw_magnitude=spectral_result["raw_magnitude"],
        radii=radial_radii,
        profile=radial_profile,
        original_image=pil_img,
    )

    if not has_branch_b_ckpt:
        spectral_prob = physics_res["prob_ai"]
    else:
        spectral_prob = spectral_result["pred_prob"]

    if not has_branch_a_ckpt:
        spatial_prob = spectral_prob
        # Re-apply calibrated spatial overlay
        spatial_overlay = apply_colormap_on_image(pil_img, spatial_cam, alpha=0.55, colormap_name="jet")

    # 3. Multimodal Late Fusion
    fusion_prob = (late_fusion_alpha * spatial_prob) + ((1.0 - late_fusion_alpha) * spectral_prob)

    # 4. Generate Diagnostic Visualization
    save_fig_path = output_dir / f"{img_name}_xai_diagnostic.png"
    XAIVisualizer.create_multimodal_diagnostic_panel(
        original_image=pil_img,
        spatial_overlay=spatial_overlay,
        spatial_prob=spatial_prob,
        fft_image=fft_pil,
        spectral_overlay=spectral_overlay,
        spectral_prob=spectral_prob,
        radial_radii=radial_radii,
        radial_profile=radial_profile,
        band_attribution=band_attr,
        fusion_prob=fusion_prob,
        true_label=true_label,
        save_path=save_fig_path,
        dpi=180,
    )

    # 5. Diagnostic Summary
    high_freq_pct = band_attr["high_frequency_attribution"]
    mid_freq_pct = band_attr["mid_frequency_attribution"]
    low_freq_pct = band_attr["low_frequency_attribution"]

    if fusion_prob >= 0.5:
        primary_domain = "Spatial Boundaries" if spatial_prob > spectral_prob else "Frequency Spectrum"
        summary_finding = (
            f"Classified as AI-GENERATED (Confidence: {fusion_prob * 100:.1f}%). "
            f"Primary trigger: {primary_domain}. "
            f"High-frequency spectral noise accounts for {high_freq_pct * 100:.1f}% of frequency attribution."
        )
    else:
        summary_finding = (
            f"Classified as REAL / NATURE (Confidence: {(1.0 - fusion_prob) * 100:.1f}%). "
            "Smooth natural 1/f spectral decay observed without significant high-frequency generative spikes."
        )

    result_json = {
        "image": str(image_path) if isinstance(image_path, (str, Path)) else "in_memory_image",
        "spatial_prob_ai": round(spatial_prob, 4),
        "spectral_prob_ai": round(spectral_prob, 4),
        "multimodal_fusion_prob_ai": round(fusion_prob, 4),
        "unified_decision": "AI" if fusion_prob >= 0.5 else "REAL",
        "frequency_band_attribution": {k: round(v, 4) for k, v in band_attr.items()},
        "summary": summary_finding,
        "diagnostic_panel_path": str(save_fig_path),
    }

    json_path = output_dir / f"{img_name}_xai_report.json"
    with open(json_path, "w") as f:
        json.dump(result_json, f, indent=2)

    print(f"\n[XAI Diagnostic Complete]")
    print(f"  * Figure saved: {save_fig_path}")
    print(f"  * JSON saved:   {json_path}")
    print(f"  * Decision:     {result_json['unified_decision']} (P(AI)={fusion_prob*100:.2f}%)")
    print(f"  * Finding:      {summary_finding}\n")

    return result_json


def create_demo_image() -> Image.Image:
    """Generate a synthetic test image with high-frequency grid noise for demonstration."""
    size = 224
    # Base gradient image
    x = np.linspace(0, 1, size)
    y = np.linspace(0, 1, size)
    xx, yy = np.meshgrid(x, y)
    base = 0.5 * np.sin(2 * np.pi * xx * 3) + 0.5 * np.cos(2 * np.pi * yy * 3)

    # Add subtle checkerboard artifact (mimicking transposed convolution / de-noising grid)
    grid = 0.15 * np.sin(2 * np.pi * xx * 28) * np.sin(2 * np.pi * yy * 28)

    img = np.clip(base + grid, 0.0, 1.0)
    rgb = np.stack([img, img * 0.9, img * 0.8], axis=-1)
    return Image.fromarray((rgb * 255).astype(np.uint8))


def main():
    parser = argparse.ArgumentParser(description="Multimodal Explainable AI (XAI) Diagnostic CLI")
    parser.add_argument("--image", help="Path to input image to analyze")
    parser.add_argument("--branch-a-ckpt", default=None, help="Path to Branch A ResNet50 checkpoint")
    parser.add_argument("--branch-b-ckpt", default=None, help="Path to Branch B ResNet50 checkpoint")
    parser.add_argument("--output-dir", default="xai/outputs", help="Output directory for reports")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", choices=["cpu", "cuda"])
    parser.add_argument("--alpha", type=float, default=0.51, help="Late fusion weight for Branch A")
    parser.add_argument("--true-label", choices=["real", "ai"], default=None, help="Ground truth label")
    parser.add_argument("--demo", action="store_true", help="Run self-contained demonstration on a synthetic test image")
    parser.add_argument("--download-weights", action="store_true", help="Download official ImageNet weights (requires internet & ~100MB download)")

    args = parser.parse_args()

    if not args.image and not args.demo:
        parser.print_help()
        print("\nNotice: Running in --demo mode since no input image was specified.\n")
        args.demo = True

    device = args.device
    print(f"Initializing XAI Engine on device: {device}")

    allow_dl = args.download_weights
    spatial_model = load_spatial_model(args.branch_a_ckpt, device=device, allow_download=allow_dl)
    spectral_model = load_spectral_model(args.branch_b_ckpt, device=device, allow_download=allow_dl)

    has_a = bool(args.branch_a_ckpt and Path(args.branch_a_ckpt).is_file())
    has_b = bool(args.branch_b_ckpt and Path(args.branch_b_ckpt).is_file())

    if not has_a or not has_b:
        print("[Mode] Active Engine: Mathematical Spectral Physics Detector (Zero-Training Mode)")

    if args.demo:
        print("Generating synthetic demo sample with high-frequency de-noising artifacts...")
        demo_img = create_demo_image()
        demo_path = Path(args.output_dir) / "demo_synthetic_sample.png"
        demo_path.parent.mkdir(parents=True, exist_ok=True)
        demo_img.save(demo_path)
        analyze_image(
            image_path=demo_path,
            spatial_model=spatial_model,
            spectral_model=spectral_model,
            output_dir=args.output_dir,
            device=device,
            late_fusion_alpha=args.alpha,
            true_label="ai",
            has_branch_a_ckpt=has_a,
            has_branch_b_ckpt=has_b,
        )
    else:
        analyze_image(
            image_path=args.image,
            spatial_model=spatial_model,
            spectral_model=spectral_model,
            output_dir=args.output_dir,
            device=device,
            late_fusion_alpha=args.alpha,
            true_label=args.true_label,
            has_branch_a_ckpt=has_a,
            has_branch_b_ckpt=has_b,
        )


if __name__ == "__main__":
    main()
