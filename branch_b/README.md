# Branch B: Frequency-Domain (FFT) AI Image Detector

Branch B implements a frequency-domain AI image detector based on 2D Fast Fourier Transform (FFT) magnitude spectrum analysis coupled with a deep convolutional neural network (ResNet50).

---

## 1. Environment & Dependencies

- **OS**: Windows (x86_64)
- **Python**: 3.10.11 (`.\venv\Scripts\python.exe`)
- **PyTorch**: 2.5.1+cu121 (CUDA 12.3)
- **Torchvision**: 0.20.1+cu121
- **Hardware**: NVIDIA GeForce RTX 4050 Laptop GPU (6.0 GB VRAM)
- **Key Libraries**: Pillow 12.3.0, Pandas 2.3.3, NumPy 2.2.6, scikit-learn 1.7.2, tqdm 4.67.1

---

## 2. Methodology & FFT Pipeline

The frequency-domain pipeline processes each input image through the following ten stages:
1. **Safe Loading**: Read image file safely via PIL with context manager.
2. **Color Space Conversion**: Convert image to RGB format.
3. **Standard Resizing**: Bilinear resize to fixed dimensions ($224 \times 224$).
4. **Luminance Conversion**: Standard ITU-R BT.601 conversion to grayscale:
   $$\text{gray} = 0.299 \cdot R + 0.587 \cdot G + 0.114 \cdot B$$
5. **2D FFT & Shift**: Compute 2D discrete Fourier transform and center low-frequency components:
   $$\text{spectrum} = \text{fftshift}(\text{fft2}(\text{gray}))$$
6. **Log-Magnitude Spectrum**: Compute numerically stable log-magnitude:
   $$\text{magnitude} = \log(1 + |\text{spectrum}|)$$
7. **Percentile Normalization**: Robust min-max scaling between 1st and 99th percentiles clipped to $[0.0, 1.0]$.
8. **3-Channel Conversion**: Replicate the single-channel magnitude spectrum across 3 channels to form a $(224, 224, 3)$ array.
9. **Tensor Transformation**: Convert to PyTorch float32 tensor in $[0, 1]$ and apply ImageNet normalization:
   $$\text{mean} = [0.485, 0.456, 0.406], \quad \text{std} = [0.229, 0.224, 0.225]$$
10. **Classification Backbone**: Forward through a ResNet50 classifier with linear head $\mathbb{R}^{2048} \to \mathbb{R}^2$ outputting logits for real/nature ($0$) and AI/fake ($1$).

---

## 3. Dataset & Split Verification

### Training Generators (Seen in Training & Validation)
Six distinct image generators were used:
- `imagenet_ai_0419_vqdm`
- `imagenet_ai_0424_sdv5`
- `imagenet_ai_0424_wukong`
- `imagenet_ai_0508_adm`
- `imagenet_glide`
- `imagenet_midjourney`

### Held-Out Generator (Unseen Evaluation)
- `imagenet_ai_0419_biggan` was strictly excluded from training and validation splits to test generalizability to unseen GAN architectures.

### Split Statistics (After BigGAN Exclusion and Exact Path Deduplication)
- **Train Set**:
  - Total deduplicated samples: **23,999**
  - Real / Nature ($0$): 11,999
  - AI / Fake ($1$): 12,000 (2,000 per generator across 6 generators)
  - Train/Val overlap: **0**
- **Validation Set**:
  - Total deduplicated samples: **6,000**
  - Real / Nature ($0$): 3,000
  - AI / Fake ($1$): 3,000 (500 per generator across 6 generators)
- **Held-Out BigGAN Test Set**:
  - Total samples: **5,000**
  - Real / Nature ($0$): 2,500
  - AI / Fake ($1$): 2,500
  - Overlap with training/validation: **0**

---

## 4. Verification and Pre-Training Tests

All modules were compiled and tested through the automated test suite (`scratch/test_branch_b.py`):
1. **Module Compilation**: `dataset.py`, `evaluate.py`, `fft_utils.py`, `frequency_model.py`, `train.py`, `fusion_interface.py` all compiled with zero syntax or import errors.
2. **FFT Numerical Check**: Output shape $(224, 224, 3)$, range $[0, 255]$, no NaNs or Infs.
3. **Dataset Output**: Validated tensor shape $(3, 224, 224)$, dtype `torch.float32`, labels $\in \{0, 1\}$.
4. **Model Forward Pass**: Validated forward pass on CUDA with output tensor $(B, 2)$.
5. **Training Smoke Test**: Gradient computation and optimizer step passed without anomaly.
6. **Validation Smoke Test**: Loss computation and accuracy metrics evaluated cleanly.
7. **Checkpoint Integrity**: Verified saving and reload round-trip with weight fidelity.

---

## 5. Training Protocol

- **Architecture**: ResNet50 (ImageNet pretrained initialization)
- **Loss Function**: `CrossEntropyLoss`
- **Optimizer**: `AdamW` (learning rate = $1 \times 10^{-4}$)
- **Batch Size**: 16
- **Epochs**: 10
- **Device**: CUDA (NVIDIA GeForce RTX 4050 Laptop GPU)
- **Workers**: 0 (safe single-process data loading on Windows)

### Training History Summary
| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Notes |
|:-----:|:----------:|:---------:|:--------:|:-------:|:------|
| 1 | 0.5480 | 71.06% | 0.4751 | 74.80% | Warmup |
| **2** | **0.4355** | **78.87%** | **0.4167** | **80.42%** | **Best Checkpoint** |
| 3 | 0.3241 | 85.53% | 0.4412 | 80.05% | |
| 4 | 0.2018 | 91.65% | 0.4826 | 78.33% | |
| 5 | 0.1189 | 95.43% | 0.5713 | 80.20% | |
| 6 | 0.0881 | 96.55% | 0.6322 | 79.15% | |
| 7 | 0.0709 | 97.31% | 0.6795 | 79.62% | |
| 8 | 0.0638 | 97.76% | 0.6482 | 79.12% | |
| 9 | 0.0597 | 97.83% | 0.7553 | 79.85% | Overfitting beyond Epoch 2 |

---

## 6. Actual Evaluation Metrics

### Validation Set (6 Non-Held-Out Generators)
- **Checkpoint Evaluated**: `branch_b/models/frequency/best_model.pt` (Epoch 2)
- **Evaluated Samples**: 6,000
- **Cross-Entropy Loss**: 0.4167
- **Accuracy**: **80.42%** (0.804167)
- **Macro F1 Score**: **80.41%** (0.804139)
- **ROC-AUC**: **89.05%** (0.890476)
- **Confusion Matrix**:
  ```
                 Predicted Real    Predicted AI
  Actual Real         2,448             552
  Actual AI             623           2,377
  ```
- **Class-wise Metrics**:
  - **Real / Nature ($0$)**: Precision = 79.71%, Recall = 81.60%, F1 = 80.65% (Support = 3,000)
  - **AI / Fake ($1$)**: Precision = 81.15%, Recall = 79.23%, F1 = 80.18% (Support = 3,000)

---

### Held-Out Unseen Generator (BigGAN Test Set)
- **Evaluated Samples**: 5,000
- **Cross-Entropy Loss**: 1.0156
- **Accuracy**: **58.40%** (0.584000)
- **Macro F1 Score**: **56.47%** (0.564655)
- **ROC-AUC**: **58.20%** (0.582013)
- **Confusion Matrix**:
  ```
                 Predicted Real    Predicted AI
  Actual Real         1,987             513
  Actual AI           1,567             933
  ```
- **Class-wise Metrics**:
  - **Real / Nature ($0$)**: Precision = 55.91%, Recall = 79.48%, F1 = 65.64% (Support = 2,500)
  - **AI / Fake ($1$)**: Precision = 64.52%, Recall = 37.32%, F1 = 47.29% (Support = 2,500)

---

## 7. Artifact Locations

- **Best Model Checkpoint**: `branch_b/models/frequency/best_model.pt`
- **Last Model State**: `branch_b/models/frequency/last_model_state.pt`
- **Training History**: `branch_b/models/frequency/history.json`
- **Validation Metrics**: `branch_b/results/val_metrics.json`
- **Validation Predictions**: `branch_b/results/val_predictions.csv`
- **BigGAN Test Metrics**: `branch_b/results/biggan_test_metrics.json`
- **BigGAN Test Predictions**: `branch_b/results/biggan_test_predictions.csv`

---

## 8. Standalone Inference & Fusion Interface

Branch B provides an isolated interface for downstream fusion in `branch_b/fusion_interface.py`:

```python
from PIL import Image
from branch_b.fusion_interface import BranchBFrequencyDetector

# 1. Initialize detector
detector = BranchBFrequencyDetector(
    checkpoint_path="branch_b/models/frequency/best_model.pt",
    device="cuda",
)

# 2. Predict on an arbitrary PIL Image
image = Image.open("path/to/test.jpg")
result = detector.predict_image(image)
print(result)
# Output:
# {
#   "prob_ai": 0.942,
#   "prob_real": 0.058,
#   "pred_label": 1,
#   "logits": [-1.42, 1.38]
# }

# 3. Extract 2048-dimensional bottleneck features for fusion
features = detector.extract_features(image)  # torch.Tensor of shape (2048,)
```

---

## 9. Preparation for Fusion with Branch A

| Property | Branch A (Spatial Domain) | Branch B (Frequency Domain) |
| :--- | :--- | :--- |
| **Backbone** | ResNet50 | ResNet50 |
| **Input Domain** | Spatial RGB image | 2D FFT Log-Magnitude Spectrum |
| **Input Shape** | $(3, 224, 224)$ | $(3, 224, 224)$ |
| **Classification Head** | `Linear(2048, 1)` with Dropout | `Linear(2048, 2)` |
| **Output** | Single logit (BCE) | 2 logits (Cross-Entropy) |
| **Class Mapping** | 0 = Real, 1 = AI | 0 = Real, 1 = AI |
| **Bottleneck Feature** | 2048-dim avgpool vector | 2048-dim avgpool vector |

### Recommended Fusion Strategies:
1. **Decision / Probability Fusion (Late Fusion)**:
   $$P_{\text{fusion}}(\text{AI}) = \alpha \cdot P_{\text{Branch A}}(\text{AI}) + (1 - \alpha) \cdot P_{\text{Branch B}}(\text{AI})$$
   where $\alpha \in [0, 1]$ is optimized on the validation split.
2. **Feature Concatenation (Early/Mid Fusion)**:
   Concatenate the 2048-dim Branch A spatial feature vector with the 2048-dim Branch B frequency feature vector to form a 4096-dim vector, trained via an MLP or linear probe.

---

## 10. Protection of Branch A Confirmation

- **Branch A files**: Not modified, renamed, moved, or overwritten (`branch_a/` remained strictly untouched).
- **Branch A checkpoints**: `checkpoints/` directory preserved completely intact.
- **Branch A historical results**: `results/` directory preserved completely intact.
- **Git Branch**: Maintained on `branch-b-fft` throughout execution.
