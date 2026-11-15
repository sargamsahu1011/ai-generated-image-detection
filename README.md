# SpectraLens • Multimodal AI Image Forensics & Explainability

[![Render Live](https://img.shields.io/badge/Render-Live%20Demo-46E3B7?style=flat&logo=render&logoColor=white)](https://ai-generated-image-detection-qqxy.onrender.com)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0+-000000.svg)](https://flask.palletsprojects.com/)
[![TailwindCSS](https://img.shields.io/badge/TailwindCSS-v3-38bdf8.svg)](https://tailwindcss.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> 🚀 **Try the Live Web App**: [**https://ai-generated-image-detection-qqxy.onrender.com**](https://ai-generated-image-detection-qqxy.onrender.com)  
> *Upload any photo or click a sample from the built-in gallery for instant, explainable forensic verification.*

---

## 🌟 Why We Built This

Modern generative models—like Midjourney, Stable Diffusion, DALL-E 3, and FLUX—have gotten scarily good. To the human eye, synthetic skin textures, lighting, and complex background scenes look indistinguishable from real photography.

Traditional detectors often look only at pixel textures, which means they can easily be fooled by high-resolution generators, compression filters, or simple post-processing.

**SpectraLens takes a different, physics-grounded approach.**

Real camera sensors capture light through physical glass lenses, which naturally causes optical power to drop off smoothly following a universal natural law called **$1/f$ power-law decay**. In contrast, AI generators build pictures by upsampling latent grids and de-noising matrices. Even when the picture looks flawless to you, it leaves distinct, periodic mathematical fingerprints in the **frequency spectrum**.

By looking at an image through two complementary lenses—**visual surface textures** and **2D Fourier frequency spectra**—SpectraLens achieves both high accuracy and explainable physical proof.

---

## ✨ Key Features

- **Dual-Domain Forensics**: Combines visual surface gradient analysis with 2D Fast Fourier Transform (FFT) spectral fingerprinting.
- **Physical Law Verification**: Verifies compliance with natural optical $1/f$ power-law decay against generative upsampling noise.
- **Explainable AI (XAI) Diagnostic Suite**: Generates a synchronized 6-panel forensic report showing exactly *why* an image was flagged.
- **Cross-Generator Generalization**: Tested across Diffusion models (Stable Diffusion, ADM, GLIDE, VQDM, Wukong), Midjourney, and unseen GANs (BigGAN).
- **Interactive Web Dashboard**: Modern, responsive dark-mode UI with drag-and-drop upload, one-click test gallery, and 150 DPI report export.
- **Developer-Friendly REST API**: Lightweight `/predict` and diagnostic `/explain` endpoints for pipeline integration.

---

## 🔍 How It Works (In Plain English)

```
                            ┌─────────────────────────┐
                            │    Input Image (RGB)    │
                            └────────────┬────────────┘
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 ▼                                               ▼
      ┌─────────────────────┐                         ┌─────────────────────┐
      │   Visual Surface    │                         │  Spectral Frequency │
      │      Forensics      │                         │     (2D FFT Map)    │
      └──────────┬──────────┘                         └──────────┬──────────┘
                 │                                               │
                 │ Micro-textures, edge blending,                │ Periodic grid spikes,
                 │ gradient continuity                           │ high-frequency noise
                 │                                               │
                 └───────────────────────┬───────────────────────┘
                                         ▼
                             ┌───────────────────────┐
                             │    Unified Forensic   │
                             │   Decision Ensemble   │
                             └───────────┬───────────┘
                                         ▼
                            Real Photo vs. AI-Generated
```

1. **Visual Surface Forensics**:
   Examines subtle local inconsistencies in pixel gradients, boundary transitions, and synthetic blending artifacts that often betray generative diffusion models.

2. **Spectral Frequency Forensics (2D FFT)**:
   Converts the image's luminance into frequency space via a 2D Fast Fourier Transform:
   $$\text{Spectrum} = \text{fftshift}(\text{fft2}(\text{Luminance}))$$
   This unmasks periodic lattice spikes caused by transposed convolutions and convolutional upsamplers that are completely invisible in standard RGB view.

3. **Azimuthal Radial Energy Profiling ($1/f$ Analysis)**:
   Integrates energy radially from low to high frequencies. Authentic camera photographs display a steady linear drop-off in log-space ($\sim 1/f^\beta$), whereas synthetic images exhibit anomalous high-frequency energy spikes or unnatural plateauing.

4. **Multi-Domain Ensemble**:
   Balances both perspectives to deliver a confident, calibrated verdict with high resilience to out-of-distribution generators.

---

## 🔬 The 6-Panel Forensic Diagnostic Report

Every analysis generates a publication-ready diagnostic breakdown:

| Panel | Forensic Element | What It Tells You |
| :---: | :--- | :--- |
| **1** | **Input RGB Image** | Original image under inspection. |
| **2** | **Surface Attention (Grad-CAM)** | Highlights the specific image regions and boundary artifacts driving the decision. |
| **3** | **2D FFT Log-Magnitude Spectrum** | Frequency map showing DC center (low frequencies) out to edge borders (high frequencies). |
| **4** | **Spectral Attention (Grad-CAM)** | Reveals which specific frequency bands exhibited anomalous energy. |
| **5** | **Radial Energy Decay Profile** | Compares the image's energy drop-off curve against natural physical optics. |
| **6** | **Band Attribution Breakdown** | Quantitative energy distribution across Low (0–33%), Mid (33–66%), and High (66–100%) frequency bands. |

---

## 📊 Benchmark Highlights

Evaluated across extensive real-world and synthetic datasets covering multiple generative architectures:

### 1. In-Distribution Performance (6 Generator Families)
*Evaluated across Midjourney, Stable Diffusion v5, ADM, GLIDE, VQDM, and Wukong ($N = 6,000$):*

- **Detection Accuracy**: **96.7%**
- **Macro F1-Score**: **0.967**
- **ROC-AUC**: **0.995**
- **Authentic Recall**: **96.5%** | **Synthetic Recall**: **97.0%**

### 2. Generalization to Unseen Architectures (Held-Out Test)
*Evaluated on an unseen generator family held out entirely during training ($N = 5,000$):*

- **Detection Accuracy**: **85.1%**
- **ROC-AUC**: **0.958**
- **Takeaway**: Spatial texture representations generalize reliably to completely unseen model families, while frequency spectra pinpoint architecture-specific upsampling artifacts.

---

## 🚀 Quickstart & Usage

### Option 1: Live Demo (No Install Needed)
Visit [**https://ai-generated-image-detection-qqxy.onrender.com**](https://ai-generated-image-detection-qqxy.onrender.com) to test your own images directly in the browser.

---

### Option 2: Run Locally with Python

```bash
# Clone the repository
git clone https://github.com/sargamsahu1011/ai-generated-image-detection.git
cd ai-generated-image-detection

# Create and activate a virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start the web app
python -m backend.app
```

Then navigate to **`http://localhost:5000`** in your browser.

---

### Option 3: Run with Docker

```bash
# Build the container
docker build -t spectralens .

# Run the container
docker run -p 5000:5000 spectralens
```

Access the dashboard at **`http://localhost:5000`**.

---

## 🌐 REST API Reference

The backend provides simple REST endpoints for easy integration into automated pipelines, bots, or content moderation workflows:

### Endpoints:
- `GET /health` — Check server health and status.
- `GET /api/samples` — List built-in demo sample images.
- `POST /predict` — Quick classification returning prediction and probabilities.
- `POST /explain` — Full forensic analysis including base64 6-panel diagnostic visualization.

### Python Example:

```python
import requests

url = "https://ai-generated-image-detection-qqxy.onrender.com/explain"
# Or locally: "http://localhost:5000/explain"

with open("path/to/image.jpg", "rb") as f:
    response = requests.post(url, files={"image": f})

result = response.json()
print("Verdict:", result["prediction"])             # "AI" or "REAL"
print("Confidence:", result["confidence_percent"])  # e.g., 98.4
print("Diagnostic Panel URI:", result["diagnostic_panel_base64"][:50], "...")
```

### Curl Example:

```bash
curl -X POST -F "image=@samples/uploaded_check.jpg" http://localhost:5000/predict
```

---

## 📁 Repository Structure

```text
ai-generated-image-detection/
├── backend/                    # Production Flask API & inference engine
│   ├── app.py                  # API endpoints (/predict, /explain, /health)
│   └── inference.py            # Multimodal inference pipeline & base64 report generator
├── frontend/                   # Web user interface
│   └── index.html              # Modern dark glassmorphic dashboard (Tailwind CSS)
├── xai/                        # Explainable AI (XAI) Suite
│   ├── gradcam.py              # Pure PyTorch Grad-CAM & overlay generator
│   ├── spectral_xai.py         # 2D FFT & Azimuthal Radial Energy profiling
│   └── visualizer.py           # 6-panel 150 DPI diagnostic figure generator
├── branch_a/                   # Visual surface analysis pipeline & checkpoints
├── branch_b/                   # Spectral frequency (2D FFT) pipeline & checkpoints
├── fusion/                     # Decision ensemble & feature fusion modules
├── samples/                    # Preloaded test gallery images
├── Dockerfile                  # Containerized deployment config
└── requirements.txt            # Python dependencies
```

---

## 👥 Contributors & Community

Developed and maintained by:
* **Sargam Sahu** ([@sargamsahu1011](https://github.com/sargamsahu1011))
* **Shubhi Jain** ([@shubhi1401](https://github.com/shubhi1401))
* **Eklavya Verma** ([@EklavyaVerma](https://github.com/EklavyaVerma))

Contributions, feature requests, and issue reports are always welcome! Feel free to open an issue or submit a pull request.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
