# Multimodal Explainable AI (XAI) Suite

An interpretable deep learning framework designed for **AI-Generated Image Detection**. It provides simultaneous visual verification across both the **Spatial Domain (RGB Pixels)** and the **Frequency Domain (2D FFT Spectrums)**.

---

## 📌 Why Explainable AI is Critical for this Project

Standard accuracy benchmarks (e.g. 96.7%) do not prove that a neural network is learning genuine generative synthesis artifacts. Without XAI, deep neural networks often suffer from **shortcut learning** (e.g., classifying an image as "AI" because of an unusual background color or dog breed rather than synthesis artifacts).

This XAI framework answers the primary viva/evaluation question:
> **"What specific physical or mathematical anomalies in the image triggered the AI prediction?"**

---

## 🏗️ Architecture & Techniques

```
                                 ┌────────────────────────┐
                                 │      Input Image       │
                                 └───────────┬────────────┘
                                             │
             ┌───────────────────────────────┴───────────────────────────────┐
             ▼                                                               ▼
  ┌──────────────────────┐                                        ┌───────────────────────┐
  │   Spatial RGB Image  │                                        │ 2D FFT Log Spectrum   │
  └──────────┬───────────┘                                        └───────────┬───────────┘
             │                                                                │
             ▼                                                                ▼
  ┌──────────────────────┐                                        ┌───────────────────────┐
  │  Spatial ResNet50    │                                        │  Spectral ResNet50    │
  │  (Branch A)          │                                        │  (Branch B)           │
  └──────────┬───────────┘                                        └───────────┬───────────┘
             │                                                                │
             ▼                                                                ▼
  ┌──────────────────────┐                                        ┌───────────────────────┐
  │   Spatial Grad-CAM   │                                        │   Spectral Grad-CAM   │
  │ (Pixel Localizations)│                                        │  (Frequency Rings)    │
  └──────────┬───────────┘                                        └───────────┬───────────┘
             │                                                                │
             └───────────────────────────────┬────────────────────────────────┘
                                             ▼
                               ┌───────────────────────────┐
                               │ 6-Panel Diagnostic Panel  │
                               │ - RGB + Spatial GradCAM   │
                               │ - FFT + Spectral GradCAM  │
                               │ - Radial Energy Profile   │
                               │ - Band Attribution Chart  │
                               └───────────────────────────┘
```

---

## 🔬 Mathematical Formulations

### 1. Spatial Grad-CAM (Branch A)
For target class $c$ (where $c = 1$ is AI-Generated) and the final convolutional layer activations $A^k \in \mathbb{R}^{H \times W}$:

$$\alpha_k^c = \frac{1}{Z} \sum_{i=1}^H \sum_{j=1}^W \frac{\partial Y^c}{\partial A_{i,j}^k}$$

$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_k \alpha_k^c A^k \right)$$

* The $\text{ReLU}$ operation ensures that only features having a **positive influence** on the synthetic classification score are visualized.
* For Branch A's single-logit sigmoid head, the score $Y^1 = \text{logit}$ for synthetic predictions.

### 2. Spectral Grad-CAM (Branch B)
The input image is transformed into normalized 2D FFT log-magnitude space:

$$\text{Spectrum}(u, v) = \text{fftshift}\Big(\mathcal{F}\big(I_{\text{gray}}\big)\Big)$$

$$\text{Magnitude}(u, v) = \log\big(1 + |\text{Spectrum}(u, v)|\big)$$

Spectral Grad-CAM highlights which **spatial frequencies** activated Branch B:
* **Center (Low Frequencies)**: Overall illumination, coarse geometric shapes.
* **Middle Rings (Mid Frequencies)**: Edge textures and fine surface details.
* **Perimeter & Corners (High Frequencies)**: De-noising step noise from Latent Diffusion Models and checkerboard grid spikes from transposed convolutions in GANs.

### 3. Azimuthal Radial Energy Profiling
Natural images follow an empirical power-law decay ($E(f) \propto 1/f^\alpha$, where $\alpha \approx 1.5 - 2.0$). 
We compute the mean magnitude at radius $r$:

$$E(r) = \frac{1}{N_r} \sum_{(u, v) \in \text{Ring}(r)} \text{Magnitude}(u, v)$$

AI-generated images produce elevated plateaus or periodic spikes in high-frequency bands ($r > 0.66 \cdot r_{\max}$).

---

## 🚀 How to Run

### 1. Run Demo Mode (Generates a Synthetic Diagnostic Panel)
Run this command from the root `ai-generated-image-detection` directory:

```bash
python -m xai.explain --demo --output-dir xai/outputs
```

This will automatically create:
1. `xai/outputs/demo_synthetic_sample_xai_diagnostic.png` (high-res 6-panel diagnostic figure)
2. `xai/outputs/demo_synthetic_sample_xai_report.json` (numerical attribution scores)

### 2. Analyze Any Image File
```bash
python -m xai.explain \
  --image "path/to/test_image.png" \
  --output-dir "xai/outputs"
```

### 3. Specify Custom Trained Weights
```bash
python -m xai.explain \
  --image "path/to/test_image.png" \
  --branch-a-ckpt "checkpoints/imagenet_ai_0419_biggan/best_model.pt" \
  --branch-b-ckpt "branch_b/models/frequency/best_model.pt" \
  --alpha 0.51
```

---

## 🎯 Viva & Presentation Guide (JUET Defense)

### Q1: How does your model avoid learning dataset shortcuts?
> **Answer**: "We applied Grad-CAM to Branch A and Spectral Grad-CAM to Branch B. Spatial Grad-CAM confirms the network attends to semantic boundaries (hair, eyes, unnatural blending borders) rather than background colors. Spectral Grad-CAM proves that Branch B targets high-frequency noise bands characteristic of generative upsampling."

### Q2: What is novel about your Explainable AI implementation?
> **Answer**: "While standard computer vision projects only apply Grad-CAM to RGB images, we developed **Spectral-XAI**. By backpropagating through our 2D FFT ResNet branch, we can isolate whether low, mid, or high spatial frequencies drove the classification, supported by an azimuthal radial frequency decay curve."

### Q3: Why does held-out BigGAN drop in frequency accuracy while spatial stays high?
> **Answer**: "As revealed by our Radial Energy Profiles, BigGAN uses class-conditional transposed convolutions with fixed stride patterns that create localized spectral spikes at distinct radii, whereas diffusion models (SDv5, ADM) generate high-frequency residual noise distributed uniformly across outer rings. Spatial features (textures and blending) are invariant across generator architectures, explaining why Spatial ResNet generalized at 85.08% while Frequency dropped to 58.40% on unseen GANs."
