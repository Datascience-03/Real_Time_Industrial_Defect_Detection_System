#!/bin/bash
# Simple helper to build TensorRT engine using trtexec (inside NVIDIA container or host with TensorRT)

set -euo pipefail

ONNX=${1:-runs/detect/train/weights/best.onnx}
ENGINE=${2:-model.engine}

if ! command -v trtexec >/dev/null 2>&1; then
  echo "trtexec not found in PATH. Run inside an NVIDIA container or install TensorRT tools."
  exit 2
fi

echo "Building engine from $ONNX -> $ENGINE"
trtexec --onnx=$ONNX --saveEngine=$ENGINE --workspace=4096 --fp16
echo "Engine build complete: $ENGINE"
