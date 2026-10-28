"""
Spectral Explainable AI (Spectral-XAI) for Frequency-Domain Analysis.
=====================================================================
Explains the predictions of Branch B (2D FFT ResNet50) and provides
frequency-domain diagnostics:
  1. 2D FFT Log-Magnitude Spectrum extraction
  2. Spectral Grad-CAM overlay on the frequency spectrum
  3. Azimuthal Radial Frequency Energy Profiling (1/f power law analysis)
  4. High-frequency anomaly detection for generative artifacts
"""

from __future__ import annotations
from typing import Optional, Tuple, Union, Any
import numpy as np
from PIL import Image

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    torch = None
    nn = None
    HAS_TORCH = False

from .gradcam import GradCAM, apply_colormap_on_image


def image_to_fft_spectrum(
    image: Union[Image.Image, np.ndarray],
    size: int = 224,
) -> Tuple[Image.Image, np.ndarray]:
    """Convert an image to a 2D FFT log-magnitude spectrum.

    Matches the exact preprocessing used in Branch B (fft_utils.py).

    Args:
        image: PIL Image or RGB numpy array.
        size: Target image dimensions.

    Returns:
        fft_pil: 3-channel RGB PIL Image of the normalized spectrum.
        magnitude_raw: 2D numpy array (size, size) of raw log-magnitude spectrum.
    """
    if isinstance(image, np.ndarray):
        if image.dtype != np.uint8 and image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        image = Image.fromarray(image)

    image = image.convert("RGB").resize((size, size))
    pixels = np.asarray(image, dtype=np.float32) / 255.0

    # ITU-R 601-2 luminance conversion
    gray = (
        0.299 * pixels[:, :, 0]
        + 0.587 * pixels[:, :, 1]
        + 0.114 * pixels[:, :, 2]
    )

    # 2D Fast Fourier Transform + Shift DC component to center
    spectrum = np.fft.fftshift(np.fft.fft2(gray))
    magnitude = np.log1p(np.abs(spectrum))

    # Normalize by 1st and 99th percentile
    low, high = np.percentile(magnitude, [1, 99])
    denom = max(float(high - low), 1e-8)
    magnitude_norm = np.clip((magnitude - low) / denom, 0.0, 1.0)

    fft_uint8 = (magnitude_norm * 255).astype(np.uint8)
    fft_rgb = np.repeat(fft_uint8[:, :, None], 3, axis=2)

    return Image.fromarray(fft_rgb), magnitude


def compute_radial_profile(
    magnitude_spectrum: np.ndarray,
    num_bins: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the 1D Azimuthally Integrated Radial Frequency Energy Profile.

    Natural images follow an empirical 1/f^alpha power-law distribution.
    Generative models exhibit high-frequency spikes and periodic deviations.

    Args:
        magnitude_spectrum: 2D numpy array representing the centered FFT magnitude.
        num_bins: Number of radial frequency bins from center to Nyquist frequency.

    Returns:
        radii: Radial distances from center (normalized [0, 1]).
        radial_energy: Mean spectral magnitude at each radius.
    """
    h, w = magnitude_spectrum.shape
    center_y, center_x = h // 2, w // 2

    # Coordinate grids
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

    max_r = min(center_x, center_y)
    bin_edges = np.linspace(0, max_r, num_bins + 1)
    radial_energy = np.zeros(num_bins, dtype=np.float32)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:]) / max_r

    for i in range(num_bins):
        mask = (r >= bin_edges[i]) & (r < bin_edges[i + 1])
        if np.any(mask):
            radial_energy[i] = magnitude_spectrum[mask].mean()
        else:
            radial_energy[i] = 0.0

    return bin_centers, radial_energy


class SpectralXAI:
    """End-to-End Spectral Explainable AI Engine."""

    def __init__(
        self,
        frequency_model: nn.Module,
        target_layer: Optional[nn.Module] = None,
        use_cuda: bool = False,
    ):
        self.model = frequency_model
        self.gradcam = GradCAM(
            model=frequency_model,
            target_layer=target_layer,
            use_cuda=use_cuda,
        )

    def explain(
        self,
        image: Union[Image.Image, np.ndarray],
        transform,
        target_class: int = 1,
    ) -> dict:
        """Run complete frequency-domain explainability analysis.

        Args:
            image: Original input RGB image.
            transform: PyTorch torchvision transform used for Branch B inputs.
            target_class: 1 for AI / Fake, 0 for Real / Nature.

        Returns:
            Dictionary with:
              - 'fft_image': PIL Image of 2D FFT spectrum.
              - 'raw_magnitude': 2D numpy array of raw FFT log-magnitude.
              - 'cam': 2D numpy array of activation heatmap on the spectrum.
              - 'overlay': Blended visualization on the FFT spectrum.
              - 'pred_prob': Prediction probability for target_class.
              - 'radial_radii': 1D normalized radius array [0, 1].
              - 'radial_profile': 1D radial energy values.
        """
        fft_pil, raw_mag = image_to_fft_spectrum(image, size=224)
        input_tensor = transform(fft_pil)

        cam, pred_prob = self.gradcam.generate(input_tensor, target_class=target_class)
        overlay = apply_colormap_on_image(fft_pil, cam, alpha=0.55, colormap_name="jet")

        radii, profile = compute_radial_profile(raw_mag)

        # Quantify frequency band attribution (Low, Mid, High)
        h, w = cam.shape
        cy, cx = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        max_r = min(cx, cy)

        low_mask = r < (max_r * 0.33)
        mid_mask = (r >= (max_r * 0.33)) & (r < (max_r * 0.66))
        high_mask = r >= (max_r * 0.66)

        band_attribution = {
            "low_frequency_attribution": float(cam[low_mask].mean()) if np.any(low_mask) else 0.0,
            "mid_frequency_attribution": float(cam[mid_mask].mean()) if np.any(mid_mask) else 0.0,
            "high_frequency_attribution": float(cam[high_mask].mean()) if np.any(high_mask) else 0.0,
        }

        return {
            "fft_image": fft_pil,
            "raw_magnitude": raw_mag,
            "cam": cam,
            "overlay": overlay,
            "pred_prob": pred_prob,
            "radial_radii": radii,
            "radial_profile": profile,
            "band_attribution": band_attribution,
        }


def detect_spectral_physics_anomaly(
    raw_magnitude: np.ndarray,
    radii: np.ndarray,
    profile: np.ndarray,
    original_image: Optional[Union[Image.Image, np.ndarray]] = None,
) -> dict:
    """Zero-training physics-based detector for generative spectral and chromatic anomalies.

    Examines:
      1. Natural 1/f power-law decay slope (steeper in real photography, flatter in diffusion).
      2. High-to-Low frequency energy distribution ratio.
      3. Local 2D High-Frequency Periodicity Residual spikes (GAN transposed conv & diffusion upsampler harmonics).
      4. Chromatic Saturation Anomaly (characteristic of diffusion synthesis).

    Returns:
        Dict with 'prob_ai', 'decision', 'slope', 'max_spike', 'finding'.
    """
    from scipy.ndimage import uniform_filter

    # 1. Slope of log(E) vs log(r) in mid-to-high frequencies
    fit_mask = (radii >= 0.08) & (radii <= 0.85)
    log_r = np.log(radii[fit_mask] + 1e-8)
    log_e = np.log(np.maximum(profile[fit_mask], 1e-8))
    slope, _ = np.polyfit(log_r, log_e, 1)

    # 2. High-to-low frequency ratio
    low_e = np.mean(profile[radii < 0.30])
    high_e = np.mean(profile[radii > 0.65])
    hf_ratio = float(high_e / (low_e + 1e-8))

    # 3. Local 2D High-Frequency Periodicity Residual
    h, w = raw_magnitude.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    outer_mask = r > (0.35 * min(cx, cy))

    smoothed = uniform_filter(raw_magnitude, size=9)
    residual = np.abs(raw_magnitude - smoothed)
    high_freq_max_spike = float(np.max(residual[outer_mask]))

    # 4. Chromatic saturation analysis
    high_sat_ratio = 0.05
    if original_image is not None:
        if isinstance(original_image, Image.Image):
            arr = np.asarray(original_image.convert("RGB"), dtype=np.float32) / 255.0
        else:
            arr = original_image.astype(np.float32)
            if arr.max() > 1.0:
                arr = arr / 255.0
        max_c = np.max(arr, axis=-1)
        min_c = np.min(arr, axis=-1)
        sat = np.where(max_c > 0, (max_c - min_c) / (max_c + 1e-6), 0.0)
        high_sat_ratio = float(np.mean(sat > 0.55))

    # Calibrated Multimodal Physics Scoring:
    # Real baseline: slope ~ -0.48 to -0.65, hf_ratio < 0.38, high_sat_ratio < 0.08, max_spike < 2.8
    # AI characteristics: slope > -0.41, hf_ratio > 0.42, high_sat_ratio > 0.25, max_spike > 3.0
    z_slope = (slope - (-0.44)) / 0.06
    z_hf = (hf_ratio - 0.39) / 0.04
    z_sat = (high_sat_ratio - 0.15) / 0.15
    z_spike = (high_freq_max_spike - 3.0) / 0.5

    logit = 1.8 * z_slope + 1.2 * z_hf + 1.8 * z_sat + 1.2 * z_spike
    prob_ai = float(1.0 / (1.0 + np.exp(-np.clip(logit, -10.0, 10.0))))
    decision = "AI" if prob_ai >= 0.5 else "REAL"

    if prob_ai >= 0.5:
        finding = (
            f"Anomalous spectral decay slope ({slope:.3f}), elevated high-frequency ratio ({hf_ratio:.2f}), "
            f"and synthesis artifacts detected (Confidence: {prob_ai * 100:.1f}%)."
        )
    else:
        finding = (
            f"Natural 1/f power-law decay slope ({slope:.3f}) and low high-frequency residual observed, "
            f"consistent with authentic optical camera capture."
        )

    return {
        "prob_ai": prob_ai,
        "decision": decision,
        "slope": float(slope),
        "hf_ratio": hf_ratio,
        "max_spike": high_freq_max_spike,
        "high_sat_ratio": high_sat_ratio,
        "finding": finding,
    }
