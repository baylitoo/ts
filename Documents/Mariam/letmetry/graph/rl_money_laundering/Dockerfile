# Dockerfile for RL Money Laundering Detection
# Optimized for H100 GPUs with CUDA 12.4 + uv for fast dependency management

FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast Python package installer)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set Python version for uv
ENV UV_PYTHON=python3.11

# Copy project files
COPY pyproject.toml uv.lock ./
COPY src ./src
COPY scripts ./scripts
COPY README.md ./

# Install dependencies with uv (uses lockfile for reproducibility)
RUN uv sync --frozen --no-dev

# Create directories for data and outputs
RUN mkdir -p /app/data/amlnet /app/data/elliptic /app/data/processed /app/outputs /app/checkpoints

# Set environment variables
ENV PYTHONPATH=/app/src
ENV CUDA_VISIBLE_DEVICES=0

# Expose ports for TensorBoard and Jupyter
EXPOSE 6006
EXPOSE 8888

# Default command (can be overridden in docker-compose)
CMD ["uv", "run", "python", "scripts/train_agent_v2.py", "--config", "quick_test"]
