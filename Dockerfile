FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.10 and base packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
 && add-apt-repository ppa:deadsnakes/ppa \
 && apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3.10-distutils python3-pip \
    ca-certificates git wget vim curl \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
 && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
 && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Install PyTorch for CUDA 11.8
RUN python3 -m pip install --extra-index-url https://download.pytorch.org/whl/cu118 \
    torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0+cu118

# Install IrisPupilNet core dependencies
RUN python3 -m pip install \
    matplotlib \
    numpy>=1.24.0 \
    pandas>=2.0.0 \
    opencv-python>=4.8.0 \
    albumentations>=1.3.0 \
    PyYAML>=6.0 \
    tqdm>=4.65.0 \
    mediapipe \
    scipy

# Install optional dependencies for ONNX export (Python 3.10 compatible)
RUN python3 -m pip install \
    onnx>=1.14.0 \
    onnxruntime>=1.16.0

# Install TensorBoard for logging
RUN python3 -m pip install tensorboard>=2.14.0

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Set environment variable to disable albumentations update checks
ENV NO_ALBUMENTATIONS_UPDATE=1

# Default command
CMD ["/bin/bash"]
