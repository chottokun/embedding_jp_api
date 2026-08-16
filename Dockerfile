# Stage 1: Builder
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/opt/uv/python

RUN apt-get update && \
    apt-get install -y git curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Copy uv binary from official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock README.md ./

# Create virtualenv using uv managed Python 3.11 and install dependencies with CUDA torch
RUN uv python install 3.11 && \
    uv sync --no-dev --no-install-project --python 3.11 && \
    uv pip install --python /app/.venv/bin/python torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Stage 2: Final Image
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    APP_PORT=8000 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HOME=/home/appuser/.cache/huggingface \
    UV_PYTHON_INSTALL_DIR=/opt/uv/python \
    PATH="/app/.venv/bin:/opt/uv/python/cpython-3.11-linux-x86_64-gnu/bin:$PATH"

RUN apt-get update && \
    apt-get install -y ca-certificates && \
    useradd -m -u 1000 appuser && \
    mkdir -p /home/appuser/.cache/huggingface && \
    chown -R appuser:appuser /home/appuser && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy python runtime and virtualenv from builder
COPY --from=builder /opt/uv/python /opt/uv/python
COPY --chown=appuser:appuser --from=builder /app/.venv /app/.venv

# Copy source code and config
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser config/ ./config/
COPY --chown=appuser:appuser README.md ./

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

ENV GUNICORN_WORKERS=2

HEALTHCHECK --interval=10s --timeout=5s --start-period=20s --retries=3 \
  CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1

# Command to run the application using Gunicorn from the virtual environment
CMD ["sh", "-c", "gunicorn --workers ${GUNICORN_WORKERS} --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000 --timeout 600 --worker-tmp-dir /dev/shm --keep-alive 5 src.app.main:app"]
