#!/bin/bash
# Requires vLLM / vLLM-Omni 0.24+.
set -e

MODEL="${1:?Usage: run_server.sh <model_dir> [port]}"
PORT="${2:-8091}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_CONFIG="${EASYMAGPIE_DEPLOY_CONFIG:-${SCRIPT_DIR}/../deploy/easymagpie.yaml}"

echo "Starting EasyMagpieTTS: model=${MODEL} deploy=${DEPLOY_CONFIG} port=${PORT}"

VLLM_PLUGINS=easymagpie_omni vllm serve "$MODEL" \
    --deploy-config "$DEPLOY_CONFIG" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --trust-remote-code \
    --disable-log-stats \
    --disable-uvicorn-access-log \
    --uvicorn-log-level warning \
    --omni
