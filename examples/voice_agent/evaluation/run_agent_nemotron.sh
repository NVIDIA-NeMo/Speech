#!/bin/bash
# Launch bot_server_nemotron.py in AGENT role.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export WEBSOCKET_PORT=8765
export FASTAPI_PORT=7860
export LOG_FILE="bot_agent_nemotron.log"
export VAD_PROFILE="ASR"
export VAD_STOP_SECS=1.0
export TTS_VOICE_ID="Magpie-Multilingual.EN-US.Aria"
export TALK_FIRST="false"

export NVIDIA_LLM_MODEL="nvidia/nemotron-3-super-120b-a12b"
# export NVIDIA_LLM_MODEL="nvidia/nemotron-3-nano-30b-a3b"
export ENABLE_TOOL_CALLING="true"
export ENABLE_THINKING="false"
export THINKING_BUDGET=1500
export TEMPERATURE=1.0
export TOP_P=0.95
export MAX_TOKENS=2048

python "$SCRIPT_DIR/bot_server_nemotron.py"
