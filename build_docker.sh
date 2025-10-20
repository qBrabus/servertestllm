#!/usr/bin/env bash
set -euo pipefail

IMAGE_TAG=${1:-unified-inference:latest}

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is required to build the image." >&2
  exit 1
fi

echo "Building Docker image ${IMAGE_TAG}..."
docker build -t "${IMAGE_TAG}" .

echo "Image ${IMAGE_TAG} built successfully."
