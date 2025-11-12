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
LOG_BASE_DIR=${LOG_DIR:-$(pwd)/logs}
CONTAINER_LOG_FILE=${CONTAINER_LOG_FILE:-${LOG_BASE_DIR}/container.log}
RUN_LOG_FILE=${RUN_LOG_FILE:-${LOG_BASE_DIR}/docker_run.log}
FOLLOW_CONTAINER_LOGS=${FOLLOW_CONTAINER_LOGS:-true}

mkdir -p "${MODEL_CACHE_DIR}"
mkdir -p "${LOG_BASE_DIR}"

exec > >(stdbuf -oL tee -a "${RUN_LOG_FILE}") 2>&1

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

log() {
  local level=$1
  shift
  printf '%s %s %s\n' "$(timestamp)" "${level}" "$*"
}

log_info() {
  log "[INFO]" "$@"
}

log_warn() {
  log "[WARN]" "$@"
}

log_error() {
  log "[ERROR]" "$@"
}

if ! command -v docker >/dev/null 2>&1; then
  log_error "Docker is required to run the container."
  exit 1
fi

if docker ps -a --format '{{.Names}}' | grep -Eq "^${CONTAINER_NAME}$"; then
  log_warn "Stopping existing container ${CONTAINER_NAME}"
  docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
fi

log_info "Launching container ${CONTAINER_NAME} from image ${IMAGE_TAG}"
log_info "Model cache directory: ${MODEL_CACHE_DIR}"
log_info "Container log file: ${CONTAINER_LOG_FILE}"
log_info "Run log file: ${RUN_LOG_FILE}"

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
  -e
  "LOG_DIR=/logs"
  -v
  "${MODEL_CACHE_DIR}:/models"
  -v
  "${LOG_BASE_DIR}:/logs"
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
  local addrs=()

  if [[ -n "${ADVERTISED_HOSTS}" ]]; then
    while IFS= read -r entry; do
      [[ -z "${entry}" ]] && continue
      addrs+=("${entry}")
    done < <(printf '%s' "${ADVERTISED_HOSTS}" | tr ',;' '\n' | tr ' ' '\n')
  fi

  if [[ ${#addrs[@]} -eq 0 ]]; then
    if [[ "${HOST_BIND_ADDRESS}" != "0.0.0.0" && "${HOST_BIND_ADDRESS}" != "*" ]]; then
      addrs+=("${HOST_BIND_ADDRESS}")
    fi

    if command -v ip >/dev/null 2>&1; then
      local primary
      primary=$(ip route get 1.1.1.1 2>/dev/null | awk '/src/ {for (i=1; i<=NF; ++i) if ($i == "src") {print $(i+1); exit}}')
      if [[ -n "${primary}" ]]; then
        addrs+=("${primary}")
      fi
    fi

    if [[ ${#addrs[@]} -eq 0 ]] && command -v hostname >/dev/null 2>&1; then
      read -r -a addrs <<<"$(hostname -I 2>/dev/null)"
    fi
  fi

  if [[ "${HOST_BIND_ADDRESS}" == "0.0.0.0" || "${HOST_BIND_ADDRESS}" == "*" ]]; then
    addrs+=("localhost")
  fi

  for addr in "${addrs[@]}"; do
    [[ -z "${addr}" ]] && continue
    [[ "${addr}" == *:* ]] && continue
    # Filter docker bridge addresses which are rarely reachable from outside the host.
    if [[ "${addr}" == 172.17.* ]]; then
      continue
    fi
    ips+=("${addr}")
  done

  printf '%s\n' "${ips[@]}" | awk '!x[$0]++'
}

mapfile -t available_ips < <(detect_host_ips)

log_info "Container ${CONTAINER_NAME} is running. Access the API at:"
for addr in "${available_ips[@]}"; do
  log_info "  http://${addr}:${HOST_PORT}"
done

if [[ "${FOLLOW_CONTAINER_LOGS}" == "true" ]]; then
  log_info "Streaming container logs (Ctrl+C to stop). Output appended to ${CONTAINER_LOG_FILE}"
  mkdir -p "$(dirname "${CONTAINER_LOG_FILE}")"
  set +e
  docker logs -f "${CONTAINER_NAME}" | stdbuf -oL tee -a "${CONTAINER_LOG_FILE}"
  set -e
else
  log_info "Container log streaming disabled (FOLLOW_CONTAINER_LOGS=${FOLLOW_CONTAINER_LOGS})."
fi
