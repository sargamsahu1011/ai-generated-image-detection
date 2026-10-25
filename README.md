# Multimodal AI-Generated Image Detector

An end-to-end deep learning framework for detecting AI-generated synthetic images across diverse generative models (Diffusion Models and GANs). The system combines **spatial-domain pixel analysis** and **frequency-domain (2D FFT) spectral fingerprinting** into a unified multimodal fusion detector.

---

## 📌 Project Overview

Generative models (Stable Diffusion, Midjourney, BigGAN, etc.) have achieved photorealistic synthesis that easily evades human perception. However, synthetic image generators leave distinct traces in different domains:
* **Spatial Domain**: Subtle local boundary artifacts, texture inconsistencies, and blending discrepancies.
* **Frequency Domain**: Periodic spectral anomalies, high-frequency energy attenuation, and grid patterns caused by transposed convolutions and de-noising upsamplers.

This project proposes a **dual-branch multimodal detection architecture** that exploits both orthogonal representations for robust in-distribution detection and cross-generator generalization.

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
   * Analyzes standard 3-channel RGB image tensors.
   * Fine-tuned with ImageNet weights to detect pixel-level structural inconsistencies.
2. **Branch B (Frequency ResNet50)**:
   * Converts luminance to the frequency domain via 2D Fast Fourier Transform:
     $$\text{Spectrum} = \text{fftshift}(\text{fft2}(\text{Grayscale}))$$
   * Computes normalized log-magnitude spectral maps replicating 3-channel inputs.
   * Exposes structural frequency grids characteristic of GANs and diffusion upsamplers.
3. **Multimodal Fusion**:
   * **Late Fusion (Decision Ensemble)**: Convex combination of branch posteriors:
     $$P_{\text{fusion}}(\text{AI}) = 0.51 \cdot P_{\text{spatial}}(\text{AI}) + 0.49 \cdot P_{\text{freq}}(\text{AI})$$
   * **Feature-Level Fusion (MLP)**: Concatenates frozen bottleneck embeddings ($2048 + 2048 = 4096\text{-d}$) forwarded through a non-linear projection MLP head with LayerNorm, ReLU, and Dropout.

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

> **Key Finding**: Feature Fusion delivers the best overall performance, surpassing both individual branches with **96.73% accuracy** and **0.9953 ROC-AUC**.

### 2. Out-of-Distribution Generalization (Held-out BigGAN, $N = 5,000$)
Evaluated on an unseen generator family held out entirely from training:

| Model Architecture | Accuracy | Macro F1 | ROC-AUC | Real Prec / Rec | Fake Prec / Rec |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Branch A (Spatial)** | **85.08%** | **0.8493** | **0.9576** | **79.23% / 95.08%** | **93.85% / 75.08%** |
| **Branch B (Frequency FFT)** | 58.40% | 0.5647 | 0.5820 | 55.91% / 79.48% | 64.52% / 37.32% |
| **Late Fusion ($\alpha = 0.51$)** | 80.52% | 0.8008 | 0.8958 | 73.53% / 95.36% | 93.40% / 65.68% |
| **Feature Fusion (MLP)** | 72.68% | 0.7105 | 0.9128 | 65.37% / 96.44% | 93.22% / 48.92% |

> **Key Finding**: Spatial representations generalize with high resilience to unseen GAN architectures (85.08% accuracy, 0.9576 ROC-AUC), while frequency spectra are generator-specific.

---

## 📁 Repository Structure

```text
ai-generated-image-detection/
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
├── metadata/                   # Train/Val/Test CSV splits and LOGO manifests
├── backend/                    # Flask REST API for inference deployment
├── frontend/                   # Web user interface (Vercel deployment)
├── FINAL_MODEL_COMPARISON.md   # Complete experimental benchmark report
└── final_model_comparison.csv  # Machine-readable tabular results
```

---

## 🚀 Usage & Evaluation

### 1. Environment Setup
```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Evaluate Branch B (Frequency Domain)
```bash
python -m branch_b.evaluate \
  --csv metadata/branch_b_splits/val.csv \
  --checkpoint branch_b/models/frequency/best_model.pt \
  --output-json branch_b/results/val_metrics.json
```

### 3. Evaluate Multimodal Fusion
```bash
python -m fusion.evaluate \
  --csv metadata/branch_b_splits/val.csv \
  --fusion-checkpoint fusion/checkpoints/best_fusion_model.pt \
  --output-json fusion/results/val_metrics.json
```

---

## 👥 Authors & Acknowledgments

* **Minor Project**: AI-Generated Image Detection
* **Techniques**: Spatial Convolutional Networks, 2D Fast Fourier Transform (FFT), Multimodal Latent Fusion, Leave-One-Generator-Out (LOGO) Cross-Validation.
