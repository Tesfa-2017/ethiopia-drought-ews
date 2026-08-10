# Dockerfile for Prithvi WxC Drought Prediction Pipeline (Ethiopia AOI)
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# Non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

# Install system dependencies & GDAL development libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin \
    libgdal-dev \
    python3-gdal \
    curl \
    git \
    wget \
    zip \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip and set up wheels
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install Python requirements with CUDA wheel repository
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu121 -r /app/requirements.txt

# Copy source code into container
COPY . /app

# Ensure storage directories exist
RUN mkdir -p /app/data /app/credentials /app/mlruns /app/checkpoints

# Default command: run data pipeline then fine-tuning
CMD ["python", "src/train.py"]
