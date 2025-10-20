#!/usr/bin/env bash
# deploy_ai_stack.sh
# Déploie l'AI stack (vLLM, ASR, Diarizer, Admin) avec logs complets.

set -Eeuo pipefail

#-----------------------------#
#           Logging           #
#-----------------------------#
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p logs
LOG_FILE="logs/deploy_$(date +'%Y%m%d_%H%M%S').log"

# Rediriger TOUT vers le log + console
exec > >(tee -a "$LOG_FILE") 2>&1

echo "========== $(date -Is) :: DÉBUT DU DÉPLOIEMENT =========="
echo "Script : $0"
echo "CWD    : $(pwd)"
echo "Log    : $LOG_FILE"
echo

# Trap pour noter la fin/erreur
trap 'ec=$?; echo "========== $(date -Is) :: FIN (exit $ec) =========="; exit $ec' EXIT

# Petite fonction d'étape
step() {
  echo
  echo "---- $(date -Is) :: $* ----"
}

#-----------------------------#
#     Prérequis & contexte    #
#-----------------------------#
step "Contexte système"
uname -a || true
which docker && docker --version || { echo "Docker introuvable. Installe-le."; exit 1; }
which nvidia-smi && nvidia-smi || echo "nvidia-smi indisponible (ok si pas de GPU visible)."

#-----------------------------#
# 0) Charger .env dans shell  #
#-----------------------------#
step "0) Chargement des variables d'environnement depuis .env"
if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
else
  echo "Fichier .env introuvable. Je continue, mais tes docker run vont t'insulter si des vars manquent."
fi

# Afficher quelques variables utiles si présentes
echo "VLLM_PORT=${VLLM_PORT:-non défini}"
echo "VLLM_MODEL=${VLLM_MODEL:-non défini}"
echo "VLLM_TP=${VLLM_TP:-non défini}"
echo "ASR_PORT=${ASR_PORT:-non défini}"
echo "DIAR_PORT=${DIAR_PORT:-non défini}"
echo "ADMIN_PORT=${ADMIN_PORT:-non défini}"
echo "HUGGINGFACE_HUB_TOKEN: $( [[ -n "${HUGGINGFACE_HUB_TOKEN:-}" ]] && echo 'défini' || echo 'absent' )"

#-----------------------------#
# 1) Réseau Docker dédié      #
#-----------------------------#
step "1) Création du réseau Docker 'ai_stack' (si absent)"
if ! docker network inspect ai_stack >/dev/null 2>&1; then
  docker network create ai_stack
else
  echo "Réseau 'ai_stack' déjà présent, on garde."
fi

#-----------------------------#
# 2) Build des images locales #
#-----------------------------#
step "2) Build des images locales (ASR, Diarizer, Admin)"
docker build -t ai_asr        ./services/asr-canary
docker build -t ai_diarizer   ./services/diarizer
docker build -t ai_admin      ./services/admin

#-----------------------------#
# 3) Lancer vLLM              #
#-----------------------------#
step "3) Lancement de vLLM (OpenAI-compatible)"

# Préparer le flag tensor-parallel si défini
VLLM_TP_FLAG=""
if [[ -n "${VLLM_TP:-}" ]]; then
  VLLM_TP_FLAG="--tensor-parallel-size ${VLLM_TP}"
fi

# Préparer args extra optionnels
VLLM_EXTRA="${VLLM_EXTRA_ARGS:-}"

# Nettoyer un conteneur éventuel précédent
docker rm -f vllm >/dev/null 2>&1 || true

docker run -d --name vllm \
  --network ai_stack \
  --gpus all \
  --shm-size=16g \
  --env-file .env \
  -v "$PWD/models":/models \
  -v "$PWD/huggingface":/root/.cache/huggingface \
  -p "${VLLM_PORT}:${VLLM_PORT}" \
  vllm/vllm-openai:latest \
  bash -lc "python -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 \
    --port ${VLLM_PORT} \
    --model ${VLLM_MODEL} \
    --download-dir /models \
    ${VLLM_EXTRA} \
    ${VLLM_TP_FLAG}"

#-----------------------------#
# 4) Lancer l’ASR Canary      #
#-----------------------------#
step "4) Lancement ASR (Canary 1B v2, NeMo)"

docker rm -f asr-canary >/dev/null 2>&1 || true

docker run -d --name asr-canary \
  --network ai_stack \
  --gpus all \
  --shm-size=8g \
  --env-file .env \
  -v "$PWD/models":/models \
  -v "$PWD/huggingface":/root/.cache/huggingface \
  -p "${ASR_PORT}:6000" \
  ai_asr

#-----------------------------#
# 5) Lancer la diarisation    #
#-----------------------------#
step "5) Lancement Diarizer (pyannote community-1)"

docker rm -f diarizer >/dev/null 2>&1 || true

docker run -d --name diarizer \
  --network ai_stack \
  --gpus all \
  --shm-size=8g \
  --env-file .env \
  -v "$PWD/models":/models \
  -v "$PWD/huggingface":/root/.cache/huggingface \
  -p "${DIAR_PORT}:7000" \
  ai_diarizer

#-----------------------------#
# 6) Lancer l’Admin + UI      #
#-----------------------------#
step "6) Lancement Admin + UI (dashboard GPU + santé services)"

docker rm -f admin >/dev/null 2>&1 || true

docker run -d --name admin \
  --network ai_stack \
  --gpus all \
  -p "${ADMIN_PORT}:8080" \
  -e "VLLM_BASE=http://vllm:${VLLM_PORT}" \
  -e "ASR_BASE=http://asr-canary:6000" \
  -e "DIAR_BASE=http://diarizer:7000" \
  -v "$PWD/models":/models:ro \
  -v "$PWD/huggingface":/root/.cache/huggingface:ro \
  -v /usr/bin/nvidia-smi:/usr/bin/nvidia-smi:ro \
  ai_admin

#-----------------------------#
#      Récap & diagnostics    #
#-----------------------------#
step "Récap containers"
docker ps -a

step "Derniers logs conteneurs (50 lignes chacun)"
for c in vllm asr-canary diarizer admin; do
  echo
  echo "=== docker logs --tail=50 $c ==="
  docker logs --tail=50 "$c" || true
done

step "vLLM /health (si curl dispo et port exposé)"
if command -v curl >/dev/null 2>&1; then
  curl -s "http://127.0.0.1:${VLLM_PORT}/health" || true
fi

echo
echo "Tout est loggé ici : $LOG_FILE"
echo "========== $(date -Is) :: DÉPLOIEMENT TERMINÉ =========="
