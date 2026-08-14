#!/usr/bin/env bash
set -euo pipefail

# Build a TensorRT engine inside an NVIDIA TensorRT container using trtexec.
# Requires Docker and NVIDIA Container Toolkit. The NGC TensorRT image may require
# authentication. Adjust IMAGE if you have a different TensorRT container.

IMAGE=${1:-nvcr.io/nvidia/tensorrt:23.09-py3}
ONNX_PATH=${2:-runs/detect/train/weights/best.onnx}
ENGINE_OUT=${3:-model.engine}

if [ ! -f "$ONNX_PATH" ]; then
  echo "ONNX file not found: $ONNX_PATH"
  exit 2
fi

PWD_HOST=$(pwd)

echo "Running trtexec inside container $IMAGE"
docker run --rm --gpus all -v "$PWD_HOST":/workspace -w /workspace \
  $IMAGE trtexec --onnx=$ONNX_PATH --saveEngine=$ENGINE_OUT --workspace=4096 --fp16

echo "Engine saved to: $ENGINE_OUT"
