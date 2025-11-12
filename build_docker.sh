#!/usr/bin/env bash

set -euo pipefail

readonly IMAGE_TAG=${1:-unified-inference:latest}
readonly CACHE_DIR=${CACHE_DIR:-.docker-cache}
readonly PROGRESS_MODE=${PROGRESS_MODE:-plain}
readonly LOG_BASE_DIR=${LOG_DIR:-$(pwd)/logs}
readonly BUILD_LOG_FILE=${BUILD_LOG_FILE:-${LOG_BASE_DIR}/docker_build.log}

mkdir -p "${LOG_BASE_DIR}"
mkdir -p "${CACHE_DIR}"

exec > >(stdbuf -oL tee -a "${BUILD_LOG_FILE}") 2>&1

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
  log_error "Docker is required to build the image."
  exit 1
fi

log_info "Docker CLI: $(docker --version)"
log_info "Target image tag: ${IMAGE_TAG}"
log_info "Cache directory: ${CACHE_DIR}"
log_info "Build log file: ${BUILD_LOG_FILE}"

SECONDS=0

run_buildx() {
  log_info "Using docker buildx with local cache"
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
  log_warn "docker buildx not available, falling back to classic docker build"
  log_warn "Disabling BuildKit to avoid buildx dependency"
  log_warn "Build cache optimizations unavailable in classic mode"
  DOCKER_BUILDKIT=0 docker build \
    --tag "${IMAGE_TAG}" \
    .
}

if docker buildx version >/dev/null 2>&1; then
  run_buildx
else
  run_classic
fi

log_info "Image ${IMAGE_TAG} built successfully in ${SECONDS}s"
