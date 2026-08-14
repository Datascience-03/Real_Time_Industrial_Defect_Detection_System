# GPU container (CUDA + optional TensorRT)

This repository includes `Dockerfile.gpu` and `docker-compose.gpu.yml` to run the project inside a GPU-enabled container.

Prerequisites
- NVIDIA GPU with recent driver
- NVIDIA Container Toolkit installed (https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- Docker Engine >= 20.10

Build and run with docker-compose (GPU)

```bash
docker compose -f docker-compose.gpu.yml build --pull
docker compose -f docker-compose.gpu.yml up -d
```

Notes
- The image is based on `nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04`. If you need TensorRT inside the container, use an NVIDIA NGC TensorRT image or install TensorRT packages from NVIDIA inside the container.
- For best performance with PyTorch and GPU inference, install a CUDA-enabled `torch` wheel matching your CUDA driver inside the container (the default `requirements.txt` may install CPU-only `torch`).

Quick test (after container starts)

```bash
# list running containers
docker ps

# view logs
docker logs -f defect-detection-gpu
```
