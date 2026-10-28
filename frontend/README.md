# Multimodal AI Detector - Web Frontend

A responsive web dashboard for testing AI-generated images with real-time Explainable AI (XAI) diagnostics.

---

## 🚀 How to Run the Frontend

The web dashboard is served directly through Flask from `backend/app.py`.

### 1. Start the Server:
From the root `ai-generated-image-detection/` folder, run:

```bash
python -m backend.app
```

### 2. Open in Your Browser:
Visit:
```text
http://localhost:5000
```

---

## ✨ Features

* **Drag-and-Drop Image Uploader**: Supports any `.jpg`, `.png`, or `.webp` file.
* **One-Click Quick Test Gallery**: Instantly test built-in sample images (Real vs. AI).
* **Live Explainable AI Visualization**: Displays the 6-panel multimodal diagnostic panel (Spatial Grad-CAM, 2D FFT, Spectral Grad-CAM, Radial Energy Curve).
* **Dual-Branch Forensic Metrics**: Shows Spatial probability, Spectral probability, and Multimodal Fusion confidence scores.
* **Dark Mode Glassmorphic UI**: Styled using Tailwind CSS.
