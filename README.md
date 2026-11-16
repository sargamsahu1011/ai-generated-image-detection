# AI-Generated Image Detection

[![Render Live](https://img.shields.io/badge/Render-Live%20Demo-46E3B7?style=flat&logo=render&logoColor=white)](https://ai-generated-image-detection-qqxy.onrender.com)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0+-000000.svg)](https://flask.palletsprojects.com/)
[![TailwindCSS](https://img.shields.io/badge/TailwindCSS-v3-38bdf8.svg)](https://tailwindcss.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> 🚀 **Try the Live Web App**: [**https://ai-generated-image-detection-qqxy.onrender.com**](https://ai-generated-image-detection-qqxy.onrender.com)  
> *Upload any image or test built-in samples to see real-time AI detection and visual forensic heatmaps.*

---

## 🌟 Why We Built This

Modern AI image generators—like Midjourney, Stable Diffusion, DALL-E, and FLUX—have gotten remarkably good. To our eyes, skin textures, natural lighting, and background details look almost indistinguishable from real photography.

Most standard detectors look only at surface pixels. The problem? As image generators get better, those surface pixels look increasingly realistic, and traditional detectors easily get fooled by high resolution or basic compression.

**This project takes a different approach: looking at camera physics.**

When a physical camera takes a photo, light passes through glass lenses onto a silicon sensor. This natural optical process causes light energy to fall off smoothly from low to high frequencies. In contrast, generative AI builds an image by upsampling numerical matrices and iteratively de-noising latent grids. Even when the picture looks completely real to human eyes, it leaves telltale periodic patterns and high-frequency noise spikes in the **frequency spectrum**.

By looking at an image through two complementary angles—**surface visual details** and **2D Fourier frequency patterns**—we can spot synthetic media with high confidence and verifiable visual proof.

---

## ✨ Highlights

- **Dual-Domain Detection**: Analyzes both pixel-level visual boundaries and 2D Fast Fourier Transform (FFT) frequency signatures.
- **Grounded in Optical Physics**: Checks if the image follows natural camera lens light decay or exhibits artificial upsampling noise.
- **Explainable AI (XAI)**: Generates a clear 6-panel visual diagnostic report showing exactly where and why an image was flagged.
- **Tested Across Models**: Benchmarked against major generator families (Stable Diffusion, Midjourney, ADM, GLIDE, VQDM, Wukong) and unseen GANs.
- **Interactive Web App**: Responsive dark-mode dashboard with drag-and-drop upload, instant sample gallery, and downloadable diagnostic figures.
- **Ready-to-Use REST API**: Clean `/predict` and `/explain` endpoints for pipeline integration.

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
      │      Analysis       │                         │     (2D FFT Map)    │
      └──────────┬──────────┘                         └──────────┬──────────┘
                 │                                               │
                 │ Checks micro-textures, edge blending,         │ Unmasks periodic grid spikes
                 │ and boundary consistency                      │ and high-frequency noise
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

1. **Surface Texture Analysis**:
   Scans for local inconsistencies in pixel gradients, unnatural skin/hair blending, and edge smoothing that often betray synthetic images.

2. **Spectral Frequency Analysis (2D FFT)**:
   Converts image luminance into frequency space using a 2D Fast Fourier Transform. This instantly reveals periodic lattice dots and grid patterns left behind by neural upsampling layers that are invisible in standard RGB view.

3. **Natural Optical Physics Check**:
   Measures energy from low frequencies (broad shapes) to high frequencies (fine details). Real camera captures follow a smooth, steady downward decay curve; synthetic images routinely break this curve with abnormal high-frequency plateaus.

4. **Unified Decision**:
   Combines both perspectives to produce a reliable, calibrated verdict with strong generalization.

---

## 🔬 The 6-Panel Diagnostic Report

When you analyze an image, the system generates a synchronized 6-panel forensic report:

| Panel | Name | What It Shows |
| :---: | :--- | :--- |
| **1** | **Input RGB Image** | Original image used as the reference baseline. |
| **2** | **Surface Attention Heatmap** | Grad-CAM saliency highlighting the exact image regions and boundaries driving the prediction. |
| **3** | **2D FFT Frequency Spectrum** | Centered frequency map (center = broad shapes, outer edges = fine details & noise). |
| **4** | **Spectral Attention Heatmap** | Shows which specific frequency bands triggered the detector. |
| **5** | **Radial Energy Decay Curve** | Compares the image's energy drop-off curve against natural physical camera optics. |
| **6** | **Band Attribution Breakdown** | Energy percentages across Low (0–33%), Mid (33–66%), and High (66–100%) frequency bands. |

---

## 📊 Performance & Results

We evaluated the system across thousands of real and synthetic images spanning diverse generator families:

* **In-Distribution Accuracy**: **96.7%** across 6 known generator families (Midjourney, Stable Diffusion v5, ADM, GLIDE, VQDM, Wukong).
* **Cross-Architecture Generalization**: **85.1%** accuracy on completely unseen model architectures (held-out BigGAN test set).
* **Key Takeaway**: Visual surface cues generalize reliably to unseen models, while frequency signatures excel at catching architecture-specific upsampling patterns.

---

## 🚀 Quickstart & Usage

### Option 1: Try the Live Web App (No Installation)
Open [**https://ai-generated-image-detection-qqxy.onrender.com**](https://ai-generated-image-detection-qqxy.onrender.com) in your browser to test your own images immediately.

---

### Option 2: Run Locally with Python

```bash
# 1. Clone the repository
git clone https://github.com/sargamsahu1011/ai-generated-image-detection.git
cd ai-generated-image-detection

# 2. Create and activate a virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the app
python -m backend.app
```

Then open your browser and go to: **`http://localhost:5000`**

---

### Option 3: Run with Docker

```bash
# Build the container image
docker build -t ai-image-detector .

# Run the container
docker run -p 5000:5000 ai-image-detector
```

Open **`http://localhost:5000`** in your browser.

---

## 🌐 Simple REST API

You can easily integrate detection into your own apps, bots, or scripts:

### Endpoints
* `GET /health` — Service health check.
* `GET /api/samples` — List built-in demo sample images.
* `POST /predict` — Fast prediction returning classification and probabilities.
* `POST /explain` — Full analysis including the base64-encoded 6-panel diagnostic visualization.

### Python Example:

```python
import requests

url = "https://ai-generated-image-detection-qqxy.onrender.com/explain"
# Or locally: "http://localhost:5000/explain"

with open("path/to/image.jpg", "rb") as f:
    response = requests.post(url, files={"image": f})

result = response.json()
print("Verdict:", result["prediction"])             # "AI" or "REAL"
print("Confidence:", result["confidence_percent"])  # e.g., 98.4%
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
├── backend/                    # Flask REST API & inference logic
│   ├── app.py                  # API routes (/predict, /explain, /health)
│   └── inference.py            # Multimodal inference & report generator
├── frontend/                   # Interactive web dashboard
│   └── index.html              # Modern dark-mode UI (Tailwind CSS)
├── xai/                        # Explainable AI (XAI) modules
│   ├── gradcam.py              # Pure PyTorch Grad-CAM heatmaps
│   ├── spectral_xai.py         # 2D FFT & radial energy profiling
│   └── visualizer.py           # 6-panel diagnostic figure generator
├── branch_a/                   # Visual surface analysis pipeline & checkpoints
├── branch_b/                   # Spectral frequency (2D FFT) pipeline & checkpoints
├── fusion/                     # Decision ensemble & feature fusion modules
├── samples/                    # Curated sample images for testing
├── Dockerfile                  # Containerized deployment config
└── requirements.txt            # Python dependencies
```

---

## 👥 Contributors

Developed and maintained by:
* **Sargam Sahu** ([@sargamsahu1011](https://github.com/sargamsahu1011))
* **Shubhi Jain** ([@shubhi1401](https://github.com/shubhi1401))
* **Eklavya Verma** ([@EklavyaVerma](https://github.com/EklavyaVerma))

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
