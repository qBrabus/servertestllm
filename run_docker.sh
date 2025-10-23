#!/usr/bin/env bash
set -euo pipefail

IMAGE_TAG=${1:-unified-inference:latest}
if [[ $# -gt 0 ]]; then
  shift
fi
CONTAINER_NAME=${CONTAINER_NAME:-unified-inference}
HOST_PORT=${HOST_PORT:-8000}
MODEL_CACHE_DIR=${MODEL_CACHE_DIR:-$(pwd)/model_cache}
HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN:-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}
OPENAI_KEYS=${OPENAI_KEYS:-}

mkdir -p "${MODEL_CACHE_DIR}"

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is required to run the container." >&2
  exit 1
fi

if docker ps -a --format '{{.Names}}' | grep -Eq "^${CONTAINER_NAME}$"; then
  echo "Stopping existing container ${CONTAINER_NAME}..."
  docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
fi

echo "Launching container ${CONTAINER_NAME} from image ${IMAGE_TAG}..."

docker_args=(
  --name
  "${CONTAINER_NAME}"
  --gpus
  all
  -p
  "${HOST_PORT}:8000"
  -e
  "HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN}"
  -e
  "MODEL_CACHE_DIR=/models"
  -v
  "${MODEL_CACHE_DIR}:/models"
)

if [[ -n "${OPENAI_KEYS}" ]]; then
  docker_args+=(
    -e
    "OPENAI_API_KEYS=${OPENAI_KEYS}"
  )
fi

docker run -d \
  "${docker_args[@]}" \
  "${IMAGE_TAG}" \
  "$@"

echo "Container ${CONTAINER_NAME} is running. Access the API at http://localhost:${HOST_PORT}".
