# AI-Generated Image Detection: Final Model Comparison Report

This report provides the unified benchmark comparison across all detector architectures developed in the project:
* **Branch A (Spatial Domain)**: ResNet50 analyzing pixel-level generative artifacts.
* **Branch B (Frequency Domain)**: ResNet50 analyzing 2D Fast Fourier Transform (FFT) log-magnitude spectrums.
* **Late Fusion (Decision Level)**: Convex probability ensemble ($\alpha \cdot P_A + (1 - \alpha) \cdot P_B$ with optimal $\alpha^* = 0.51$).
* **Feature Fusion (Representation Level)**: Frozen backbones with concatenated latent embeddings ($2048 + 2048 = 4096 \text{ dims}$) trained with an MLP classifier.

---

## 1. Executive Summary Benchmark Table

### A. In-Distribution Validation Performance (6 Seen Generators)
Evaluated on **6,000 samples** (3,000 Real / Nature, 3,000 AI / Fake) across `vqdm`, `sdv5`, `wukong`, `adm`, `glide`, and `midjourney`.

| Model Architecture | Accuracy | Macro F1 | ROC-AUC | Real Prec / Rec | Fake Prec / Rec |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Branch A (Spatial ResNet50)** | 95.83% | 0.9583 | 0.9933 | 96.52% / 95.10% | 95.17% / 96.57% |
| **Branch B (Frequency FFT ResNet50)** | 80.42% | 0.8041 | 0.8905 | 79.71% / 81.60% | 81.15% / 79.23% |
| **Late Fusion ($\alpha = 0.51$)** | 96.50% | 0.9650 | 0.9890 | 97.19% / 95.77% | 95.83% / 97.23% |
| **Feature Fusion (MLP Backbone)** | **96.73%** | **0.9673** | **0.9953** | **96.95% / 96.50%** | **96.52% / 96.97%** |

* **Winner (In-Distribution)**: **Feature Fusion** achieves the highest overall accuracy (**96.73%**) and ROC-AUC (**0.9953**), improving upon Branch A alone by reducing both false positives and false negatives.

---

### B. Out-of-Distribution (OOD) Generalization Performance (Held-out BigGAN)
Evaluated on **5,000 samples** (2,500 Real / Nature, 2,500 AI / Fake from `imagenet_ai_0419_biggan`) strictly held out during training.

| Model Architecture | Accuracy | Macro F1 | ROC-AUC | Real Prec / Rec | Fake Prec / Rec |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Branch A (Spatial ResNet50)** | **85.08%** | **0.8493** | **0.9576** | **79.23% / 95.08%** | **93.85% / 75.08%** |
| **Branch B (Frequency FFT ResNet50)** | 58.40% | 0.5647 | 0.5820 | 55.91% / 79.48% | 64.52% / 37.32% |
| **Late Fusion ($\alpha = 0.51$)** | 80.52% | 0.8008 | 0.8958 | 73.53% / 95.36% | 93.40% / 65.68% |
| **Feature Fusion (MLP Backbone)** | 72.68% | 0.7105 | 0.9128 | 65.37% / 96.44% | 93.22% / 48.92% |

* **Winner (Out-of-Distribution)**: **Branch A (Spatial)** maintains the highest robust generalization (**85.08%** accuracy and **0.9576** ROC-AUC).
* **Observation**: Because BigGAN produces distinct frequency spectral patterns differing drastically from diffusion models (SDv5, Glide, ADM), Branch B experiences domain shift on unseen GAN frequencies (58.40%), which slightly drags down uncalibrated multimodal fusion on pure GAN held-out evaluations.

---

## 2. Confusion Matrices (Detailed Breakdown)

### In-Distribution Validation Set ($N = 6,000$)
```
Branch A (Spatial):
                Pred Real    Pred AI
  Actual Real      2,853        147
  Actual AI          103      2,897

Branch B (Frequency):
                Pred Real    Pred AI
  Actual Real      2,448        552
  Actual AI          623      2,377

Late Fusion (alpha = 0.51):
                Pred Real    Pred AI
  Actual Real      2,873        127
  Actual AI           83      2,917

Feature Fusion (MLP):
                Pred Real    Pred AI
  Actual Real      2,895        105
  Actual AI           91      2,909
```

### Held-Out BigGAN Test Set ($N = 5,000$)
```
Branch A (Spatial):
                Pred Real    Pred AI
  Actual Real      2,377        123
  Actual AI          623      1,877

Branch B (Frequency):
                Pred Real    Pred AI
  Actual Real      1,987        513
  Actual AI        1,567        933

Late Fusion (alpha = 0.51):
                Pred Real    Pred AI
  Actual Real      2,384        116
  Actual AI          858      1,642

Feature Fusion (MLP):
                Pred Real    Pred AI
  Actual Real      2,411         89
  Actual AI        1,277      1,223
```

---

## 3. Leave-One-Generator-Out (LOGO) Cross-Validation (Branch A)

Evaluates model robustness across all generators when withholding each generator entirely from training (evaluated across 5,000 samples per fold with 0.5 threshold).

### A. Final 10-Epoch LOGO Benchmark (Latest `test_metrics.json` Results)
*Evaluated using the best converged checkpoint for each generator:*

| Fold | Held-Out Generator | Generator Family | Accuracy | Precision | Recall | F1 Score | ROC-AUC | Best Epoch |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | `imagenet_glide` | Diffusion | **97.18%** | 96.57% | 97.84% | 97.20% | 0.9964 | Epoch 3 |
| 2 | `imagenet_ai_0424_sdv5` | Diffusion | **95.02%** | 94.47% | 95.64% | 95.05% | 0.9905 | Epoch 5 |
| 3 | `imagenet_ai_0424_wukong` | Diffusion | **89.12%** | 95.92% | 81.72% | 88.25% | 0.9707 | Epoch 2 |
| 4 | `imagenet_ai_0419_biggan` | **GAN** | **85.00%** | 93.79% | 74.96% | 83.33% | 0.9575 | Epoch 8 |
| 5 | `imagenet_midjourney` | Diffusion | **72.70%** | 92.77% | 49.24% | 64.33% | 0.8744 | Epoch 2 |
| 6 | `imagenet_ai_0508_adm` | Diffusion | **72.25%** | 95.88% | 46.52% | 62.64% | 0.8941 | Epoch 6 |
| 7 | `imagenet_ai_0419_vqdm` | **VQ-Diffusion** | **61.60%** | 88.98% | 26.48% | 40.81% | 0.8160 | Epoch 9 |
| **Mean** | **Overall Cross-Generator** | — | **81.84%** | **94.05%** | **67.49%** | **75.94%** | **0.9285** | — |

> **VQDM Analysis (61.60% Accuracy, 81.60% ROC-AUC)**: VQDM utilizes vector-quantized discrete latent tokens rather than continuous diffusion noise schedules, producing distinct boundary representations. Despite lower recall at default 0.5 threshold, precision remains high (88.98%) and ROC-AUC is strong (0.8160).

### B. Initial 3-Epoch Baseline Reference (`LOGO_results.csv`)
*Historical baseline from Week 7 experimentation:*
* Mean Accuracy: **80.36%**, Mean ROC-AUC: **0.8798**
* Folds: BigGAN (82.70%), VQDM (55.26%), Wukong (86.76%), Glide (84.92%), Midjourney (82.80%), SDv5 (88.50%), ADM (81.56%).

---

## 4. Key Takeaways & Defense Talking Points

1. **Orthogonal Information**: Spatial domain and frequency domain capture fundamentally different signals. Spatial captures texture and structural artifacts, whereas FFT captures periodic grid fingerprints caused by upsampling and deconvolution operations.
2. **In-Distribution Superiority**: When test generators share structural similarities with training generators, **Feature Fusion** is the superior architecture, boasting **96.73% accuracy** and **0.9953 ROC-AUC**.
3. **Out-of-Distribution Generalization**: When facing a completely unfamiliar generative family (e.g. GAN vs Diffusion), pixel spatial artifacts generalize with higher fidelity (85.08%) than spectral patterns alone.
4. **Optimal Practical Deployment**: For a real-time web deployment or inference API, an adaptive fusion scheme (e.g., dynamically adjusting confidence or defaulting to late fusion with uncertainty calibration) yields the safest production behavior.
