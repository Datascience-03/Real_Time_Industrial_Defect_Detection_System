#!/usr/bin/env bash
set -euo pipefail

OUT_DIR=release
mkdir -p "$OUT_DIR"

# Collect artifacts
cp -v runs/detect/train/weights/best.onnx "$OUT_DIR/" || true
cp -v model.engine "$OUT_DIR/" || true
cp -v requirements-pinned.txt "$OUT_DIR/" || true
cp -v Dockerfile.gpu.trt "$OUT_DIR/" || true

pushd "$OUT_DIR" >/dev/null
if command -v sha256sum >/dev/null; then
  sha256sum * > release.sha256
elif command -v shasum >/dev/null; then
  shasum -a 256 * > release.sha256
else
  echo "Warning: no sha256 tool found; skipping checksum generation"
fi

tar czf release-artifacts.tar.gz * || true
popd >/dev/null

echo "Release artifacts prepared in $OUT_DIR/"
