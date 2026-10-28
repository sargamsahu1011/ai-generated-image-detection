# Multimodal AI-Generated Image Detector & XAI Suite

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0+-000000.svg)](https://flask.palletsprojects.com/)
[![TailwindCSS](https://img.shields.io/badge/TailwindCSS-v3-38bdf8.svg)](https://tailwindcss.com/)
[![JUET Minor Project](https://img.shields.io/badge/JUET-Minor%20Project-success.svg)](https://www.juet.ac.in/)

An end-to-end deep learning and physics-grounded framework for detecting AI-generated synthetic images across diverse generative architectures (Diffusion Models and GANs). The system combines **spatial-domain pixel analysis** and **frequency-domain (2D FFT) spectral fingerprinting** into a unified multimodal fusion detector with **Explainable AI (XAI)** and a **real-time web dashboard**.

---

## 📌 Project Overview

Generative models (Stable Diffusion, Midjourney, DALL-E, BigGAN, etc.) achieve photorealistic synthesis that evades human perception. However, synthetic image generators leave distinct, verifiable traces across orthogonal domains:
* **Spatial Domain**: Subtle local boundary artifacts, texture blending inconsistencies, and unnatural gradient smoothness.
* **Frequency Domain**: Periodic spectral spikes, high-frequency energy anomalies, and grid artifacts caused by transposed convolutions and iterative latent de-noising upsamplers.
* **Physics Law Violation**: Natural physical photographs strictly adhere to the empirical **$1/f^\beta$ power-law decay** in azimuthal radial power spectra; generative models violate this decay curve.

This project implements a **dual-branch multimodal detection architecture** that exploits both representations for in-distribution precision, cross-generator generalization, and transparent visual explanations.

---

## 🏗️ Model Architectures

```
                          ┌────────────────────────┐
                          │      Input Image       │
                          └───────────┬────────────┘
                                      │
            ┌─────────────────────────┴─────────────────────────┐
            ▼                                                   ▼
 ┌──────────────────────┐                           ┌───────────────────────┐
 │   Spatial Domain     │                           │   Frequency Domain    │
 │   RGB Image (224x224)│                           │   2D FFT Log-Magnitude│
 └──────────┬───────────┘                           └───────────┬───────────┘
            ▼                                                   ▼
 ┌──────────────────────┐                           ┌───────────────────────┐
 │   Branch A: ResNet50 │                           │   Branch B: ResNet50  │
 │  Spatial Classifier  │                           │  Spectral Classifier  │
 └──────────┬───────────┘                           └───────────┬───────────┘
            │                                                   │
            ├─────────────────────────┬─────────────────────────┤
            ▼                                                   ▼
 ┌──────────────────────┐                           ┌───────────────────────┐
 │     Late Fusion      │                           │    Feature Fusion     │
 │ Decision Ensemble    │                           │ Concatenated Latent   │
 │ (Optimal alpha=0.51) │                           │ Vectors (4096-d) + MLP│
 └──────────────────────┘                           └───────────────────────┘
```

1. **Branch A (Spatial ResNet50)**:
   * Analyzes standard 3-channel RGB image tensors $(224 \times 224)$.
   * Detects pixel-level structural inconsistencies, edge blur, and synthesis artifacts.
2. **Branch B (Frequency ResNet50)**:
   * Converts grayscale luminance into the frequency domain via 2D Fast Fourier Transform:
     $$\text{Spectrum} = \text{fftshift}(\text{fft2}(\text{Grayscale}))$$
   * Computes normalized log-magnitude spectral maps replicated across 3 channels.
   * Exposes structural frequency grids characteristic of GANs and diffusion upsamplers.
3. **Multimodal Fusion**:
   * **Late Fusion (Decision Ensemble)**: Convex combination of branch posteriors:
     $$P_{\text{fusion}}(\text{AI}) = 0.51 \cdot P_{\text{spatial}}(\text{AI}) + 0.49 \cdot P_{\text{freq}}(\text{AI})$$
   * **Feature-Level Fusion (MLP)**: Concatenates frozen bottleneck embeddings ($2048 + 2048 = 4096\text{-d}$) forwarded through a projection MLP head with LayerNorm, ReLU, and Dropout.

---

## 📊 Benchmark Results

Complete report and numerical raw logs:
* 📄 Detailed Report: [`FINAL_MODEL_COMPARISON.md`](./FINAL_MODEL_COMPARISON.md)
* 📈 Raw Metrics Table: [`final_model_comparison.csv`](./final_model_comparison.csv)

### 1. In-Distribution Validation (6 Seen Generators, $N = 6,000$)
Evaluated across `vqdm`, `sdv5`, `wukong`, `adm`, `glide`, and `midjourney`:

| Model Architecture | Accuracy | Macro F1 | ROC-AUC | Real Prec / Rec | Fake Prec / Rec |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Branch A (Spatial)** | 95.83% | 0.9583 | 0.9933 | 96.52% / 95.10% | 95.17% / 96.57% |
| **Branch B (Frequency FFT)** | 80.42% | 0.8041 | 0.8905 | 79.71% / 81.60% | 81.15% / 79.23% |
| **Late Fusion ($\alpha = 0.51$)** | 96.50% | 0.9650 | 0.9890 | 97.19% / 95.77% | 95.83% / 97.23% |
| **Feature Fusion (MLP)** | **96.73%** | **0.9673** | **0.9953** | **96.95% / 96.50%** | **96.52% / 96.97%** |

> **Key Finding**: Feature Fusion delivers the highest in-distribution performance with **96.73% accuracy** and **0.9953 ROC-AUC**.

### 2. Out-of-Distribution Generalization (Held-out BigGAN, $N = 5,000$)
Evaluated on an unseen generator family held out entirely from training:

| Model Architecture | Accuracy | Macro F1 | ROC-AUC | Real Prec / Rec | Fake Prec / Rec |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Branch A (Spatial)** | **85.08%** | **0.8493** | **0.9576** | **79.23% / 95.08%** | **93.85% / 75.08%** |
| **Branch B (Frequency FFT)** | 58.40% | 0.5647 | 0.5820 | 55.91% / 79.48% | 64.52% / 37.32% |
| **Late Fusion ($\alpha = 0.51$)** | 80.52% | 0.8008 | 0.8958 | 73.53% / 95.36% | 93.40% / 65.68% |
| **Feature Fusion (MLP)** | 72.68% | 0.7105 | 0.9128 | 65.37% / 96.44% | 93.22% / 48.92% |

> **Key Finding**: Spatial representations generalize strongly to unseen GAN architectures (85.08% accuracy, 0.9576 ROC-AUC), while frequency spectra are generator-specific.

---

## 🔬 Explainable AI (XAI) Suite

The XAI engine produces a **synchronized 6-panel multimodal diagnostic visualization** for every evaluated image:

| Panel | Description | Purpose |
| :--- | :--- | :--- |
| **1. Original Image** | Input RGB image | Visual reference |
| **2. Spatial Grad-CAM** | Spatial attention heatmap over pixels | Identifies boundaries, textures, and unnatural edge blending |
| **3. 2D FFT Spectrum** | Centered log-magnitude frequency map | Highlights high-frequency noise and transposed convolution grid dots |
| **4. Spectral Grad-CAM** | Attention heatmap overlaid on FFT spectrum | Reveals which frequency bands influenced the neural model's prediction |
| **5. Radial Frequency Decay** | Azimuthal integration curve: $\log_{10} E(r)$ vs $\log_{10} r$ | Verifies natural $1/f^\beta$ physical camera decay vs synthetic distortion |
| **6. Multimodal Diagnostic Verdict** | Bar chart comparing low/mid/high-frequency energy | Shows final decision with confidence breakdown |

*Full mathematical derivations and viva defense details: [`xai/README.md`](./xai/README.md)*

---

## 📁 Repository Structure

```text
ai-generated-image-detection/
├── backend/                    # Flask REST API and inference engine
│   ├── app.py                  # Endpoints: /, /health, /api/samples, /predict, /explain
│   └── inference.py            # Multimodal inference & base64 XAI generator
├── frontend/                   # Web user interface
│   ├── index.html              # Modern dark glassmorphic dashboard (Tailwind CSS)
│   └── README.md               # Frontend setup and launch instructions
├── branch_a/                   # Spatial domain (ResNet50)
│   ├── models/                 # Model definitions & training logic
│   └── results/                # LOGO & per-generator evaluation results
├── branch_b/                   # Frequency domain (2D FFT + ResNet50)
│   ├── fft_utils.py            # 2D FFT extraction & log-magnitude scaling
│   ├── frequency_model.py      # Frequency ResNet50 backbone
│   └── results/                # Validation & BigGAN test results
├── fusion/                     # Multimodal fusion architecture
│   ├── fusion_model.py         # Latent concatenation + MLP head
│   ├── evaluate.py             # Dual-branch comparative evaluation
│   └── results/                # Comparative benchmark JSONs & predictions
├── xai/                        # Explainable AI (Grad-CAM, Spectral-XAI & Diagnostics)
│   ├── gradcam.py              # Pure PyTorch Grad-CAM & Grad-CAM++
│   ├── spectral_xai.py         # 2D FFT & Azimuthal Radial Energy Profiling
│   ├── visualizer.py           # 6-panel publication-quality diagnostic figure generator
│   ├── explain.py              # Unified CLI diagnostic tool
│   └── README.md               # Mathematical formulations & viva defense Q&A
├── metadata/                   # Train/Val/Test CSV splits and LOGO manifests
├── samples/                    # Preloaded real and synthetic test images
├── FINAL_MODEL_COMPARISON.md   # Complete experimental benchmark report
├── final_model_comparison.csv  # Machine-readable tabular results
└── requirements.txt            # Project dependencies
```

---

## 🚀 Quickstart & Usage

### 1. Environment Setup

```bash
git clone https://github.com/shubhi1401/ai-generated-image-detection.git
cd ai-generated-image-detection

# Create and activate virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

### 2. Launch the Interactive Web Dashboard

Start the Flask application:

```bash
python -m backend.app
```

Then open your browser at:
👉 **`http://localhost:5000`**

#### Dashboard Features:
* **Drag-and-Drop Image Upload**: Test any custom JPG, PNG, or WEBP image.
* **One-Click Quick Test Gallery**: Pre-loaded real photos and AI-generated samples (Diffusion & GAN) for instant testing.
* **Live Decision Badge**: Clear verdict tag (`AI-Generated` vs `Real Natural Photo`) with confidence percentage.
* **Metrics Grid**: Individual confidence scores for Spatial $P(\text{AI})$, Spectral $P(\text{AI})$, and Multimodal Fusion.
* **6-Panel Multimodal XAI Viewer**: Synchronized Grad-CAM and 2D FFT spectral diagnostics with a one-click **"Save Figure"** PNG export.

---

### 3. REST API Endpoints

The Flask server provides REST endpoints for automated evaluation and integration:

| Endpoint | Method | Description |
| :--- | :---: | :--- |
| `/` | `GET` | Serves the interactive web dashboard |
| `/health` | `GET` | Service health status |
| `/api/samples` | `GET` | Lists available pre-loaded sample images |
| `/samples/<filename>` | `GET` | Serves a sample image file |
| `/predict` | `POST` | Lightweight classification probabilities (`image` form-data) |
| `/explain` | `POST` | Full prediction + base64-encoded 6-panel XAI diagnostic figure |

#### Example API Request (Python):

```python
import requests

with open("samples/uploaded_check.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:5000/explain",
        files={"image": f}
    )
data = response.json()
print("Prediction:", data["prediction"])
print("Confidence:", data["confidence_percent"], "%")
print("Fusion Probability:", data["fusion_prob_ai"])
```

---

### 4. Run CLI Diagnostics

You can also run explanations directly from your terminal:

```bash
# Run self-contained demonstration:
python -m xai.explain --demo --output-dir xai/outputs

# Run on any specific image:
python -m xai.explain --image samples/sample_real.jpg --output-dir xai/outputs
```

---

### 5. Evaluate Individual Branches & Fusion

```bash
# Evaluate Branch B (Frequency Domain)
python -m branch_b.evaluate \
  --csv metadata/branch_b_splits/val.csv \
  --checkpoint branch_b/models/frequency/best_model.pt \
  --output-json branch_b/results/val_metrics.json

# Evaluate Multimodal Fusion
python -m fusion.evaluate \
  --csv metadata/branch_b_splits/val.csv \
  --fusion-checkpoint fusion/checkpoints/best_fusion_model.pt \
  --output-json fusion/results/val_metrics.json
```

---

## 👥 Authors & Institutional Affiliation

* **Course**: Minor Project, B.Tech Computer Science & Engineering
* **Institution**: [Jaypee University of Engineering and Technology (JUET), Guna, India](https://www.juet.ac.in/)
* **Project Contributors**:
  * **Shubhi Jain** ([@shubhi1401](https://github.com/shubhi1401))
  * **Eklavya Verma** ([@EklavyaVerma](https://github.com/EklavyaVerma))
  * **Sargam Sahu**
