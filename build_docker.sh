#!/usr/bin/env bash

set -euo pipefail

readonly IMAGE_TAG=${1:-unified-inference:latest}
readonly CACHE_DIR=${CACHE_DIR:-.docker-cache}
readonly PROGRESS_MODE=${PROGRESS_MODE:-plain}

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is required to build the image." >&2
  exit 1
fi

echo "[build] Docker CLI: $(docker --version)"
echo "[build] Target image tag: ${IMAGE_TAG}"
echo "[build] Cache directory: ${CACHE_DIR}"

SECONDS=0

run_buildx() {
  echo "[build] Using docker buildx with local cache"
  mkdir -p "${CACHE_DIR}"
  DOCKER_BUILDKIT=1 BUILDKIT_PROGRESS="${PROGRESS_MODE}" \
    docker buildx build \
    --progress "${PROGRESS_MODE}" \
    --cache-from type=local,src="${CACHE_DIR}" \
    --cache-to type=local,dest="${CACHE_DIR}",mode=max \
    --tag "${IMAGE_TAG}" \
    --load \
    .
}

run_classic() {
  echo "[build] docker buildx not available, falling back to classic docker build"
  echo "[build] Disabling BuildKit to avoid buildx dependency"
  echo "[build] Build cache optimizations unavailable in classic mode"
  DOCKER_BUILDKIT=0 docker build \
    --tag "${IMAGE_TAG}" \
    .
}

if docker buildx version >/dev/null 2>&1; then
  run_buildx
else
  run_classic
fi

echo "[build] Image ${IMAGE_TAG} built successfully in ${SECONDS}s"
