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

# Copy all project files first
COPY pyproject.toml poetry.lock* ./
COPY README.md ./
COPY src/ ./src/
COPY config/ ./config/
COPY locustfile.py ./

# Add potential binary paths to PATH
ENV PATH="/root/.local/bin:/usr/local/bin:${PATH}"

# Install dependencies using uv into the system environment
# We install all dependencies including dev for locust
RUN uv pip install -e .[dev] --system protobuf sentencepiece

# Expose the application port
EXPOSE 8000

# Command to run the application using python -m gunicorn
CMD ["python", "-m", "gunicorn", "--workers", "1", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--timeout", "300", "src.app.main:app"]
