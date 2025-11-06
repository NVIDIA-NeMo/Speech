#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

################################################################################
# Agent-to-Agent Communication Demo
#
# This script demonstrates connecting two voice agents together.
# It starts two agent servers and bridges them with the WebsocketBridge.
################################################################################

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored messages
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to cleanup background processes on exit
cleanup() {
    print_info "Cleaning up processes..."
    
    if [ ! -z "$AGENT1_PID" ]; then
        print_info "Stopping Agent 1 (PID: $AGENT1_PID)"
        kill $AGENT1_PID 2>/dev/null || true
    fi
    
    if [ ! -z "$AGENT2_PID" ]; then
        print_info "Stopping Agent 2 (PID: $AGENT2_PID)"
        kill $AGENT2_PID 2>/dev/null || true
    fi
    
    if [ ! -z "$BRIDGE_PID" ]; then
        print_info "Stopping Bridge (PID: $BRIDGE_PID)"
        kill $BRIDGE_PID 2>/dev/null || true
    fi
    
    print_success "Cleanup complete"
}

# Register cleanup function
trap cleanup EXIT INT TERM

# Configuration
AGENT1_PORT=8765
AGENT2_PORT=8766
AGENT1_NAME="Alice"
AGENT2_NAME="Bob"
STARTUP_DELAY=5

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --agent1-port)
            AGENT1_PORT="$2"
            shift 2
            ;;
        --agent2-port)
            AGENT2_PORT="$2"
            shift 2
            ;;
        --agent1-name)
            AGENT1_NAME="$2"
            shift 2
            ;;
        --agent2-name)
            AGENT2_NAME="$2"
            shift 2
            ;;
        --startup-delay)
            STARTUP_DELAY="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --agent1-port PORT      Port for first agent (default: 8765)"
            echo "  --agent2-port PORT      Port for second agent (default: 8766)"
            echo "  --agent1-name NAME      Name for first agent (default: Alice)"
            echo "  --agent2-name NAME      Name for second agent (default: Bob)"
            echo "  --startup-delay SECS    Delay before starting bridge (default: 5)"
            echo "  --help                  Show this help message"
            echo ""
            echo "Example:"
            echo "  $0 --agent1-port 8765 --agent2-port 8766 --startup-delay 10"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Print configuration
echo ""
echo "=============================================================================="
echo "  Voice Agent-to-Agent Communication Demo"
echo "=============================================================================="
echo "  Agent 1: $AGENT1_NAME (Port $AGENT1_PORT)"
echo "  Agent 2: $AGENT2_NAME (Port $AGENT2_PORT)"
echo "  Startup Delay: ${STARTUP_DELAY}s"
echo "=============================================================================="
echo ""

# Check if required files exist
if [ ! -f "bot_websocket_server.py" ]; then
    print_error "bot_websocket_server.py not found in current directory"
    exit 1
fi

if [ ! -f "bot_websocket_server_alt.py" ]; then
    print_error "bot_websocket_server_alt.py not found in current directory"
    exit 1
fi

if [ ! -f "connect_two_agents.py" ]; then
    print_error "connect_two_agents.py not found in current directory"
    exit 1
fi

# Start Agent 1
print_info "Starting $AGENT1_NAME on port $AGENT1_PORT..."
python bot_websocket_server.py > agent1_output.log 2>&1 &
AGENT1_PID=$!
print_success "$AGENT1_NAME started (PID: $AGENT1_PID)"

# Wait a moment for Agent 1 to initialize
sleep 2

# Start Agent 2
print_info "Starting $AGENT2_NAME on port $AGENT2_PORT..."
python bot_websocket_server_alt.py --port $AGENT2_PORT > agent2_output.log 2>&1 &
AGENT2_PID=$!
print_success "$AGENT2_NAME started (PID: $AGENT2_PID)"

# Wait for both agents to fully initialize
print_info "Waiting ${STARTUP_DELAY}s for agents to initialize..."
for ((i=$STARTUP_DELAY; i>0; i--)); do
    echo -n "."
    sleep 1
done
echo ""
print_success "Agents initialized"

# Check if agents are still running
if ! kill -0 $AGENT1_PID 2>/dev/null; then
    print_error "$AGENT1_NAME failed to start. Check agent1_output.log for details."
    exit 1
fi

if ! kill -0 $AGENT2_PID 2>/dev/null; then
    print_error "$AGENT2_NAME failed to start. Check agent2_output.log for details."
    exit 1
fi

# Start the bridge
print_info "Starting WebSocket bridge..."
python connect_two_agents.py \
    --agent1-url "ws://localhost:$AGENT1_PORT" \
    --agent2-url "ws://localhost:$AGENT2_PORT" \
    --agent1-name "$AGENT1_NAME" \
    --agent2-name "$AGENT2_NAME" \
    --filter-audio &
BRIDGE_PID=$!
print_success "Bridge started (PID: $BRIDGE_PID)"

echo ""
print_success "Demo is now running!"
echo ""
echo "The two agents are now connected and can communicate with each other."
echo ""
echo "Logs:"
echo "  - Agent 1 output: agent1_output.log"
echo "  - Agent 2 output: agent2_output.log"
echo "  - Bridge output: agent_bridge.log"
echo ""
echo "Press Ctrl+C to stop all processes"
echo ""

# Wait for the bridge process (main loop)
wait $BRIDGE_PID

