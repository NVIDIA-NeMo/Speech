#!/bin/bash
# Launch bot_server.py in AGENT role (uses server_configs/agent.yaml).

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export SERVER_CONFIG_PATH="server_configs/agent.yaml"
export WEBSOCKET_PORT=8765
export FASTAPI_PORT=7860

python "$SCRIPT_DIR/bot_server.py"
