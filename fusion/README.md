# AI Image Detector: Fusion Architecture (Spatial + Frequency)

The Fusion module integrates **Branch A (Spatial-Domain ResNet50)** and **Branch B (Frequency-Domain ResNet50)** into a unified detector to achieve superior detection accuracy and generalization against unseen image generators.

---

## 1. Motivation

* **Branch A (Spatial Domain)**: Focuses on pixel-level generative artifacts (textures, semantic inconsistencies, blurriness, local edge anomalies).
* **Branch B (Frequency Domain)**: Focuses on spectral signatures in the 2D FFT magnitude spectrum (periodic grid patterns, high-frequency spectral roll-offs, GAN/diffusion frequency fingerprints).
* **Fusion Synergy**: Combining orthogonal representations overcomes blind spots inherent to single-domain detectors.

---

## 2. Fusion Methodologies

### Method 1: Late Fusion (Probability Decision Ensemble)
Combines the calibrated posterior probabilities from both branches via a convex combination:
$$P_{\text{fusion}}(\text{AI}) = \alpha \cdot P_A(\text{AI}) + (1 - \alpha) \cdot P_B(\text{AI})$$
* **$\alpha$ Optimization**: Evaluated over 101 candidate thresholds on the clean validation set.
* **Empirical Optimal Alpha**: $\alpha^* = 0.51$ (near-equal weighting between spatial and spectral domains).
* **Validation Accuracy**: Reached **96.50%** (outperforming Branch A's 95.83% and Branch B's 80.42%).

### Method 2: Feature-Level Fusion (Concatenation + Multi-Layer Perceptron)
Fuses deep bottleneck representations extracted from frozen backbones:
1. **Spatial Representation**: 2048-dimensional average-pooled feature vector $f_{\text{spatial}} \in \mathbb{R}^{2048}$ from Branch A.
2. **Frequency Representation**: 2048-dimensional average-pooled feature vector $f_{\text{freq}} \in \mathbb{R}^{2048}$ from Branch B.
3. **Joint Embedding**: Concatenated feature vector $f_{\text{joint}} = [f_{\text{spatial}}, f_{\text{freq}}] \in \mathbb{R}^{4096}$.
4. **Classification MLP**:
   $$\mathbb{R}^{4096} \xrightarrow{\text{Linear}} \mathbb{R}^{512} \xrightarrow{\text{LayerNorm, ReLU, Dropout}} \mathbb{R}^{128} \xrightarrow{\text{LayerNorm, ReLU, Dropout}} \mathbb{R}^2$$

---

## 3. Training & Evaluation Protocol

* **Backbones**: Frozen to prevent catastrophic forgetting and preserve Branch A & B weights.
* **Dataset**: Same 6 non-held-out generators used in Branch A and B training.
* **Held-out Unseen Generator**: BigGAN (`imagenet_ai_0419_biggan`).

---

## 4. Usage Commands

### Train the Fusion Model:
```powershell
.\venv\Scripts\python.exe -m fusion.train --epochs 3 --batch-size 32 --device cuda
```

### Evaluate on Validation Split:
```powershell
.\venv\Scripts\python.exe -m fusion.evaluate `
  --csv metadata/branch_b_splits/val.csv `
  --exclude-generators imagenet_ai_0419_biggan `
  --fusion-checkpoint fusion/checkpoints/best_fusion_model.pt `
  --output-json fusion/results/val_metrics.json `
  --output-predictions fusion/results/val_predictions.csv
```

### Evaluate on Unseen Held-out BigGAN Test Split:
```powershell
.\venv\Scripts\python.exe -m fusion.evaluate `
  --csv metadata/logo_splits/imagenet_ai_0419_biggan_test.csv `
  --fusion-checkpoint fusion/checkpoints/best_fusion_model.pt `
  --output-json fusion/results/biggan_test_metrics.json `
  --output-predictions fusion/results/biggan_test_predictions.csv
```
