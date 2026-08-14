# Dependencies and Environment Notes

This file summarizes the project's dependencies and guidance for CPU vs GPU setups.

Core Python packages
- `ultralytics` — YOLOv8 training/inference and model export
- `opencv-python` — image/video I/O and visualization
- `numpy`, `matplotlib`, `Pillow` — numerical and image utilities
- `fastapi`, `uvicorn` — API server
- `onnx`, `onnxruntime` — ONNX model runtime (use `onnxruntime-gpu` for GPU)

PyTorch / CUDA
- Install `torch` and `torchvision` appropriate for your CUDA driver.
- Example install (adjust CUDA version):
  - CPU: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`
  - CUDA 12.1: `pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision`

ONNX Runtime
- For better GPU performance, use `onnxruntime-gpu` matched to your CUDA/cuDNN.

TensorRT
- TensorRT Python bindings are platform-specific and not typically installed via pip.
- To use TensorRT you can either:
  1. Install TensorRT and the Python wheel from NVIDIA on the host.
  2. Use an NVIDIA NGC container image that already includes TensorRT.

Docker / Containers
- `Dockerfile.gpu` is provided to run inside an NVIDIA-enabled environment; you must
  install the correct CUDA/PyTorch/ONNX wheels inside the container for GPU inference.

CI Guidance
- CI workflow intentionally avoids installing heavy ML packages. Use the GPU image
  and a self-hosted runner with a GPU to run end-to-end GPU tests.
