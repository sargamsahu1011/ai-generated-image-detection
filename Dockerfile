FROM python:3.10-slim

WORKDIR /app

# Install system dependencies required by Pillow, Matplotlib, and OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU-only build first (keeps container small: ~1.2GB instead of ~7GB)
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Hugging Face Spaces listens on port 7860
ENV PORT=7860
EXPOSE 7860

# Run Gunicorn bound to 0.0.0.0:7860
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "2", "--timeout", "120", "backend.app:app"]
