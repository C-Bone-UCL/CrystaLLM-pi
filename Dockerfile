# Stage 1: Build stage with Python slim
FROM python:3.10-slim AS builder

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Copy application code
COPY . .

# Install the application package
RUN pip install --no-cache-dir --prefix=/install .

# Stage 2: Runtime stage with CUDA
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Install Python 3.10
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python-is-python3 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3.10 /usr/local/bin/python3.10

# Set working directory
WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY --from=builder /app /app

# Create directories for data and outputs
RUN mkdir -p /app/data /app/outputs /tmp/.cache

ENV MPLCONFIGDIR=/app
ENV HF_HOME=/tmp/.cache
ENV PYTHONPATH=/usr/local/lib/python3.10/site-packages
ENV TRITON_CACHE_DIR=/tmp/triton-cache

# Expose API port
EXPOSE 8000

# Entrypoint modifies token file
ENTRYPOINT ["/app/entrypoint.sh"]

# Run the API
CMD ["uvicorn", "crystallm_api:app", "--host", "0.0.0.0", "--port", "8000"]
