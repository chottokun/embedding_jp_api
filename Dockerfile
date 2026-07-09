# Stage 1: Builder
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1

RUN apt-get update && \
    apt-get install -y python3.11 python3-pip git && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install uv

WORKDIR /app

# Copy only dependency manifests
COPY pyproject.toml poetry.lock* README.md ./

# Create a dummy structure to allow installing dependencies
RUN mkdir -p src/app && touch src/app/__init__.py

# Install dependencies only (cached if pyproject.toml is unchanged)
# Install dependencies only (cached if pyproject.toml is unchanged)
RUN uv pip install --python /usr/bin/python3.11 . --system protobuf sentencepiece

# Stage 2: Final Image
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    APP_PORT=8000 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HOME=/home/appuser/.cache/huggingface

RUN apt-get update && \
    apt-get install -y python3.11 python3-pip && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    # Create a non-root user
    useradd -m -u 1000 appuser && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/dist-packages /usr/local/lib/python3.11/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy source code and other necessary files
COPY src/ ./src/
COPY config/ ./config/
COPY README.md locustfile.py ./

# Ensure the appuser owns the application directory and HF_HOME
RUN chown -R appuser:appuser /app /home/appuser

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

ENV GUNICORN_WORKERS=2

# Command to run the application
CMD ["sh", "-c", "python -m gunicorn --workers ${GUNICORN_WORKERS} --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000 --timeout 600 --worker-tmp-dir /dev/shm --keep-alive 5 src.app.main:app"]
