#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Quick start script for dynamic voice agent evaluation

NEMO_PATH=/home/heh/github/NeMo-main
export PYTHONPATH=$NEMO_PATH:$PYTHONPATH

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
USER_PORT=8765
AGENT_PORT=8766
DURATION=120
SCENARIOS=""
OUTPUT_DIR="./eval_results"

# Function to print colored messages
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --user-port)
            USER_PORT="$2"
            shift 2
            ;;
        --agent-port)
            AGENT_PORT="$2"
            shift 2
            ;;
        --duration)
            DURATION="$2"
            shift 2
            ;;
        --scenarios)
            SCENARIOS="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --user-port PORT        User (simulated user) port (default: 8765)"
            echo "  --agent-port PORT       Agent (being tested) port (default: 8766)"
            echo "  --duration SECONDS      Duration per scenario (default: 60)"
            echo "  --scenarios FILE        Scenarios JSON file (default: built-in scenarios)"
            echo "  --output-dir DIR        Output directory (default: ./eval_results)"
            echo "  -h, --help              Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0"
            echo "  $0 --duration 120 --scenarios scenarios/customer_service.json"
            echo "  $0 --output-dir ./results/my_test"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

print_info "Dynamic Voice Agent Evaluation"
echo "========================================"
echo ""

# Check if agents are running
print_info "Checking if agents are running..."

if ! nc -z localhost $USER_PORT 2>/dev/null; then
    print_error "User agent not running on port $USER_PORT"
    echo ""
    echo "Start the user agent (simulated user) in another terminal:"
    echo "  cd examples/voice_agent"
    echo "  export SERVER_CONFIG_PATH=\"evaluation/configs/user_config.yaml\""
    echo "  export WEBSOCKET_PORT=$USER_PORT"
    echo "  python server/bot_websocket.py"
    exit 1
fi
print_info "User agent is running on port $USER_PORT"

if ! nc -z localhost $AGENT_PORT 2>/dev/null; then
    print_error "Agent not running on port $AGENT_PORT"
    echo ""
    echo "Start the agent (being tested) in another terminal:"
    echo "  cd examples/voice_agent"
    echo "  export SERVER_CONFIG_PATH=\"evaluation/configs/agent_config.yaml\""
    echo "  export WEBSOCKET_PORT=$AGENT_PORT"
    echo "  python server/bot_websocket.py"
    exit 1
fi
print_info "Agent is running on port $AGENT_PORT"

echo ""
print_info "Configuration:"
echo "  User URL:      ws://localhost:$USER_PORT"
echo "  Agent URL:     ws://localhost:$AGENT_PORT"
echo "  Duration:      ${DURATION}s per scenario"
echo "  Output dir:    $OUTPUT_DIR"
if [ -n "$SCENARIOS" ]; then
    echo "  Scenarios:     $SCENARIOS"
else
    echo "  Scenarios:     Default (3 scenarios)"
fi
echo ""

# Build command
CMD="python dynamic_evaluation_runner.py"
CMD="$CMD --user-url ws://localhost:$USER_PORT"
CMD="$CMD --agent-url ws://localhost:$AGENT_PORT"
CMD="$CMD --duration $DURATION"
CMD="$CMD --output-dir $OUTPUT_DIR"

if [ -n "$SCENARIOS" ]; then
    if [ ! -f "$SCENARIOS" ]; then
        print_error "Scenarios file not found: $SCENARIOS"
        exit 1
    fi
    CMD="$CMD --scenarios-file $SCENARIOS"
fi

print_info "Starting evaluation..."
echo ""
echo "Press Ctrl+C to stop"
echo "========================================"
echo ""

# Run evaluation
eval $CMD

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    print_info "Evaluation completed successfully!"
    print_info "Results saved to: $OUTPUT_DIR"
    echo ""
    print_info "View results:"
    echo "  Summary:       cat $OUTPUT_DIR/summary_*.txt"
    echo "  Conversation:  cat $OUTPUT_DIR/conversation_*.log"
    echo "  Latencies:     cat $OUTPUT_DIR/latencies_*.csv"
else
    echo ""
    print_error "Evaluation failed!"
    exit 1
fi
