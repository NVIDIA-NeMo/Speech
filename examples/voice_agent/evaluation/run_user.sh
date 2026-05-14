#!/bin/bash
# Launch bot_server.py in USER role (uses server_configs/user.yaml).

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export SERVER_CONFIG_PATH="server_configs/user.yaml"
export WEBSOCKET_PORT=8766
export FASTAPI_PORT=7861

python "$SCRIPT_DIR/bot_server.py"
