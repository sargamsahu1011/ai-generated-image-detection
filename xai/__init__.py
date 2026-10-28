"""
Explainable AI (XAI) Suite for Multimodal AI-Generated Image Detection.
========================================================================
Provides interpretability techniques for:
1. Branch A (Spatial Domain ResNet50): Spatial Grad-CAM & Grad-CAM++
2. Branch B (Frequency Domain ResNet50): Spectral Grad-CAM & Radial Frequency Profiling
3. Multimodal Fusion: Dual-branch synchronized explanations
"""

import os
import tempfile

# Prevent matplotlib permission warnings on Windows
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mpl_xai_cache"))

from .gradcam import GradCAM, GradCAMPlusPlus, apply_colormap_on_image
from .spectral_xai import SpectralXAI, compute_radial_profile
from .visualizer import XAIVisualizer

__all__ = [
    "GradCAM",
    "GradCAMPlusPlus",
    "apply_colormap_on_image",
    "SpectralXAI",
    "compute_radial_profile",
    "XAIVisualizer",
]
