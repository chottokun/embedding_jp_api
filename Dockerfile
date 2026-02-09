# Use NVIDIA's CUDA base image for GPU support
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Set environment variables to prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV APP_PORT=8000

# Install Python 3.11, pip, git, and set it as the default python
RUN apt-get update && \
    apt-get install -y python3.11 python3-pip git && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    rm -rf /var/lib/apt/lists/*

# Install uv using pip
RUN pip3 install uv

# Set up the working directory
WORKDIR /app

# Copy dependency manifests first to leverage Docker cache
COPY pyproject.toml poetry.lock* ./

# Install dependencies using uv into the system environment
# Note: we install dependencies without the project itself first
RUN uv pip install . --system protobuf sentencepiece

# Copy the rest of the project files
COPY README.md ./
COPY src/ ./src/
COPY config/ ./config/
COPY locustfile.py ./

# Install the project in editable mode (or just install it)
RUN uv pip install -e . --system

# Expose the application port
EXPOSE 8000

ENV GUNICORN_WORKERS=2

# Command to run the application using python -m gunicorn
CMD ["sh", "-c", "python -m gunicorn --workers ${GUNICORN_WORKERS} --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000 --timeout 300 src.app.main:app"]
