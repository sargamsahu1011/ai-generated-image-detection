"""
Unified Visualizer for Explainable AI (XAI) Multimodal Diagnostics.
===================================================================
Produces presentation-ready diagnostic panels displaying:
  1. Input RGB Image + Spatial Grad-CAM
  2. 2D FFT Log-Magnitude Spectrum + Spectral Grad-CAM
  3. Frequency Band Attribution & Radial Energy Decay Curve
"""

from pathlib import Path
from typing import Optional, Union
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image


class XAIVisualizer:
    """Diagnostic Plotter and Visualizer for Dual-Branch Image Detection."""

    def __init__(self, style: str = "seaborn-v0_8-whitegrid"):
        try:
            plt.style.use(style)
        except Exception:
            plt.style.use("default")

    @staticmethod
    def create_multimodal_diagnostic_panel(
        original_image: Union[Image.Image, np.ndarray],
        spatial_overlay: np.ndarray,
        spatial_prob: float,
        fft_image: Union[Image.Image, np.ndarray],
        spectral_overlay: np.ndarray,
        spectral_prob: float,
        radial_radii: np.ndarray,
        radial_profile: np.ndarray,
        band_attribution: Optional[dict] = None,
        fusion_prob: Optional[float] = None,
        true_label: Optional[str] = None,
        save_path: Optional[Union[str, Path]] = None,
        dpi: int = 200,
    ) -> plt.Figure:
        """Create a 5-panel comprehensive multimodal diagnostic figure.

        Args:
            original_image: Input RGB image.
            spatial_overlay: Spatial Grad-CAM blended on RGB image.
            spatial_prob: P(AI) from Spatial Branch A.
            fft_image: 2D FFT Log-Magnitude Spectrum.
            spectral_overlay: Spectral Grad-CAM blended on FFT spectrum.
            spectral_prob: P(AI) from Frequency Branch B.
            radial_radii: Normalized radius bins [0, 1].
            radial_profile: Mean energy profile curve.
            band_attribution: Dict with 'low_frequency_attribution', etc.
            fusion_prob: Optional P(AI) from Multimodal Fusion.
            true_label: Optional ground truth string ('Real' or 'AI').
            save_path: Optional path to save the generated figure.
            dpi: Dots per inch for saved image.

        Returns:
            fig: The Matplotlib Figure object.
        """
        fig = plt.figure(figsize=(18, 10), constrained_layout=True)
        gs = gridspec.GridSpec(2, 3, figure=fig, height_ratios=[1.2, 1.0])

        # Color cues
        decision_spatial = "AI Generated" if spatial_prob >= 0.5 else "Real / Nature"
        decision_spectral = "AI Generated" if spectral_prob >= 0.5 else "Real / Nature"

        # -------------------------------------------------------------
        # Panel 1: Original RGB Image
        # -------------------------------------------------------------
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(original_image)
        title_str = "1. Original Input Image (Spatial Domain)"
        if true_label:
            title_str += f"\n[Ground Truth: {true_label.upper()}]"
        ax1.set_title(title_str, fontsize=12, fontweight="bold", pad=8)
        ax1.axis("off")

        # -------------------------------------------------------------
        # Panel 2: Spatial Domain Grad-CAM (Branch A)
        # -------------------------------------------------------------
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(spatial_overlay)
        ax2.set_title(
            f"2. Spatial Grad-CAM (Branch A ResNet50)\n"
            f"P(AI) = {spatial_prob * 100:.1f}%  [{decision_spatial}]",
            fontsize=12,
            fontweight="bold",
            pad=8,
            color="#991B1B" if spatial_prob >= 0.5 else "#166534",
        )
        ax2.axis("off")

        # -------------------------------------------------------------
        # Panel 3: 2D FFT Log-Magnitude Spectrum
        # -------------------------------------------------------------
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(fft_image)
        ax3.set_title(
            "3. 2D FFT Log-Magnitude Spectrum\n"
            "(Centered DC: Low freq -> Outer: High freq)",
            fontsize=12,
            fontweight="bold",
            pad=8,
        )
        ax3.axis("off")

        # -------------------------------------------------------------
        # Panel 4: Spectral Domain Grad-CAM (Branch B)
        # -------------------------------------------------------------
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.imshow(spectral_overlay)
        ax4.set_title(
            f"4. Spectral Grad-CAM (Branch B ResNet50)\n"
            f"P(AI) = {spectral_prob * 100:.1f}%  [{decision_spectral}]",
            fontsize=12,
            fontweight="bold",
            pad=8,
            color="#991B1B" if spectral_prob >= 0.5 else "#166534",
        )
        ax4.axis("off")

        # -------------------------------------------------------------
        # Panel 5: Radial Frequency Energy Profile
        # -------------------------------------------------------------
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.plot(
            radial_radii,
            radial_profile,
            color="#2563EB",
            linewidth=2.4,
            label="Sample Radial Profile",
        )
        ax5.set_title(
            "5. Azimuthal Radial Frequency Profile\n(1/f Power-Law Analysis)",
            fontsize=12,
            fontweight="bold",
            pad=8,
        )
        ax5.set_xlabel("Normalized Frequency ($f / f_{Nyquist}$)", fontsize=10)
        ax5.set_ylabel("Log Magnitude Energy", fontsize=10)
        ax5.grid(True, linestyle="--", alpha=0.5)

        # Highlight frequency regions
        ax5.axvspan(0.0, 0.33, color="#86EFAC", alpha=0.18, label="Low Freq (Coarse Shape)")
        ax5.axvspan(0.33, 0.66, color="#FDE047", alpha=0.18, label="Mid Freq (Textures)")
        ax5.axvspan(0.66, 1.0, color="#FCA5A5", alpha=0.18, label="High Freq (Generative Noise)")
        ax5.legend(loc="upper right", fontsize=8)

        # -------------------------------------------------------------
        # Panel 6: Frequency Band Attribution & Multimodal Summary
        # -------------------------------------------------------------
        ax6 = fig.add_subplot(gs[1, 2])
        if band_attribution:
            categories = ["Low Freq\n(0-33%)", "Mid Freq\n(33-66%)", "High Freq\n(66-100%)"]
            values = [
                band_attribution.get("low_frequency_attribution", 0.0),
                band_attribution.get("mid_frequency_attribution", 0.0),
                band_attribution.get("high_frequency_attribution", 0.0),
            ]
            total = sum(values) if sum(values) > 0 else 1.0
            pcts = [(v / total) * 100 for v in values]
            bars = ax6.bar(categories, pcts, color=["#10B981", "#F59E0B", "#EF4444"], width=0.55)
            for bar in bars:
                height = bar.get_height()
                ax6.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 1.5,
                    f"{height:.1f}%",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                    fontsize=9,
                )
            ax6.set_ylim(0, max(max(pcts) * 1.25 if pcts else 100, 10.0))
            ax6.set_ylabel("Attribution Weight (%)", fontsize=10)
        else:
            ax6.text(0.5, 0.5, "Band Attribution\nN/A", ha="center", va="center")

        summary_title = "6. Spectral Frequency Attribution"
        if fusion_prob is not None:
            summary_title += f"\n[Multimodal Fusion P(AI) = {fusion_prob * 100:.1f}%]"
        ax6.set_title(summary_title, fontsize=12, fontweight="bold", pad=8)
        ax6.grid(axis="y", linestyle="--", alpha=0.5)

        # Main Super Title
        suptitle_text = "Multimodal Explainable AI (XAI) Diagnostic Analysis"
        if fusion_prob is not None:
            f_decision = "AI GENERATED" if fusion_prob >= 0.5 else "REAL / NATURE"
            suptitle_text += f" | Unified Decision: {f_decision} (Confidence: {max(fusion_prob, 1 - fusion_prob) * 100:.1f}%)"
        fig.suptitle(suptitle_text, fontsize=15, fontweight="bold", y=0.99)

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)

        return fig
