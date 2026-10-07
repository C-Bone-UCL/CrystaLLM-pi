# Stage 1: Build environment
FROM ghcr.io/astral-sh/uv:0.8-bookworm-slim AS builder

ENV UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_PYTHON_PREFERENCE=only-managed \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --locked --extra all --no-install-project --python 3.12

COPY . .
RUN uv sync --locked --extra all --python 3.12

# Stage 2: Runtime
FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04

COPY --from=builder /opt/python /opt/python
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app /app

WORKDIR /app

# Create directories for data and outputs
RUN mkdir -p /app/data /app/outputs /tmp/.cache /tmp/triton-cache

# Ensure docker entrypoint is executable
RUN chmod +x /app/docker/entrypoint.sh

ENV PATH=/opt/venv/bin:$PATH
ENV MPLCONFIGDIR=/app
ENV HF_HOME=/tmp/.cache
ENV TRITON_CACHE_DIR=/tmp/triton-cache

# Expose API port
EXPOSE 8000

# Entrypoint modifies token file
ENTRYPOINT ["/app/docker/entrypoint.sh"]

# Run the API
CMD ["uvicorn", "_api:app", "--host", "0.0.0.0", "--port", "8000"]
