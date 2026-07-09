# Stage 1: Builder
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

RUN apt-get update && \
    apt-get install -y python3.11 python3.11-venv python3-pip git && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    rm -rf /var/lib/apt/lists/*

# Copy uv binary from official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock README.md ./

# Create virtualenv and install production dependencies only (no dev dependencies)
RUN uv sync --no-dev --no-install-project

# Stage 2: Final Image
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    APP_PORT=8000 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HOME=/home/appuser/.cache/huggingface \
    PATH="/app/.venv/bin:$PATH"

RUN apt-get update && \
    apt-get install -y python3.11 python3.11-venv && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    # Create a non-root user
    useradd -m -u 1000 appuser && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy virtualenv containing production packages from builder
COPY --from=builder /app/.venv /app/.venv

# Copy source code and config
COPY src/ ./src/
COPY config/ ./config/
COPY README.md ./

# Ensure the appuser owns the application directory and HF_HOME
RUN chown -R appuser:appuser /app /home/appuser

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

ENV GUNICORN_WORKERS=2

# Command to run the application using Gunicorn from the virtual environment
CMD ["sh", "-c", "gunicorn --workers ${GUNICORN_WORKERS} --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000 --timeout 600 --worker-tmp-dir /dev/shm --keep-alive 5 src.app.main:app"]
