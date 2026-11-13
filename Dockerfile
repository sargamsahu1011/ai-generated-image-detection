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

# Expose port (Render automatically sets $PORT, defaults to 5000)
ENV PORT=5000
EXPOSE 5000

# 1 Gunicorn worker with 2 threads keeps memory at ~300MB, safely inside Render's 512MB free RAM limit
CMD exec gunicorn --bind 0.0.0.0:${PORT} --workers 1 --threads 2 --timeout 120 "backend.app:app"
