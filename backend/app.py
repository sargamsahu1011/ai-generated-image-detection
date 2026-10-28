"""
Flask REST API for AI-Generated Image Detection and XAI Visualizations.
========================================================================
Endpoints:
  GET  /health   - API status and model load health check
  POST /predict  - Quick prediction (AI vs Nature probabilities)
  POST /explain  - Full prediction + Base64 Diagnostic XAI Panel (Grad-CAM & FFT)
"""

import io
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image

app = Flask(__name__)
CORS(app)

# Lazy initialized engine
_engine = None


def get_engine():
    global _engine
    if _engine is None:
        from backend.inference import MultimodalInferenceEngine
        _engine = MultimodalInferenceEngine()
    return _engine


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "online",
        "service": "Multimodal AI-Generated Image Detector & XAI Suite",
        "version": "1.0.0",
    })


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided in multipart 'image' field."}), 400

    file = request.files["image"]
    try:
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
        engine = get_engine()
        result = engine.predict_and_explain(img)

        # Omit large base64 image for lightweight predict endpoint
        result.pop("diagnostic_panel_base64", None)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/explain", methods=["POST"])
def explain():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided in multipart 'image' field."}), 400

    file = request.files["image"]
    try:
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
        engine = get_engine()
        result = engine.predict_and_explain(img)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
