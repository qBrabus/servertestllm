#!/usr/bin/env bash
set -euo pipefail

IMAGE_TAG=${1:-unified-inference:latest}
if [[ $# -gt 0 ]]; then
  shift
fi
CONTAINER_NAME=${CONTAINER_NAME:-unified-inference}
HOST_PORT=${HOST_PORT:-8000}
HOST_BIND_ADDRESS=${HOST_BIND_ADDRESS:-0.0.0.0}
ADVERTISED_HOSTS=${ADVERTISED_HOSTS:-}
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
  "${HOST_BIND_ADDRESS}:${HOST_PORT}:8000"
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

detect_host_ips() {
  local ips=()
  local manual_ips=()

  if [[ -n "${ADVERTISED_HOSTS}" ]]; then
    while IFS= read -r entry; do
      [[ -z "${entry}" ]] && continue
      manual_ips+=("${entry}")
    done < <(printf '%s' "${ADVERTISED_HOSTS}" | tr ',;' '\n' | tr ' ' '\n')
  fi

  if [[ "${HOST_BIND_ADDRESS}" != "0.0.0.0" && "${HOST_BIND_ADDRESS}" != "*" ]]; then
    manual_ips+=("${HOST_BIND_ADDRESS}")
  fi

  if [[ ${#manual_ips[@]} -gt 0 ]]; then
    ips+=("${manual_ips[@]}")
  fi

  if command -v hostname >/dev/null 2>&1; then
    while IFS= read -r addr; do
      [[ -z "${addr}" ]] && continue
      [[ "${addr}" == *:* ]] && continue
      ips+=("${addr}")
    done < <(hostname -I 2>/dev/null | tr ' ' '\n')
  fi

  if command -v ip >/dev/null 2>&1; then
    while IFS= read -r addr; do
      [[ -z "${addr}" ]] && continue
      [[ "${addr}" == *:* ]] && continue
      ips+=("${addr}")
    done < <(ip -4 addr show scope global | awk '/inet / {print $2}' | cut -d/ -f1)
  fi

  ips+=("localhost")

  printf '%s\n' "${ips[@]}" | awk '!x[$0]++'
}

mapfile -t available_ips < <(detect_host_ips)

echo "Container ${CONTAINER_NAME} is running. Access the API at:"
for addr in "${available_ips[@]}"; do
  echo "  http://${addr}:${HOST_PORT}"
done
