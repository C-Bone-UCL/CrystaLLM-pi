FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install FastAPI and Uvicorn for the API
RUN pip install --no-cache-dir fastapi uvicorn[standard]

# Copy application code
COPY . .

# Install the application package
RUN pip install --no-cache-dir .

# Create directories for data and outputs
RUN mkdir -p /app/data /app/outputs /tmp/.cache

# Create non-root user with UID:GID 1000
RUN groupadd -g 1000 appuser && \
    useradd -r -u 1000 -g appuser appuser && \
    chown -R appuser:appuser /app && \
    chmod +x /app/entrypoint.sh && \
    chmod a+w /app && \
    chmod a+w /tmp/.cache

# Switch to non-root user
USER appuser

ENV MPLCONFIGDIR=/app
ENV HF_HOME=/tmp/.cache

# Expose API port
EXPOSE 8000

# Entrypoint modifies token file
ENTRYPOINT ["/app/entrypoint.sh"]
# Run the API
CMD ["uvicorn", "crystallm_api:app", "--host", "0.0.0.0", "--port", "8000"]
