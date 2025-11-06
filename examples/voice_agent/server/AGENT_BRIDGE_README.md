# Voice Agent Bridge - Connecting Two Agents

This directory contains tools to connect two voice agents directly so they can communicate with each other.

## Files

- **`websocket_bridge.py`**: The `WebsocketBridge` class that manages bidirectional communication between two websocket endpoints
- **`connect_two_agents.py`**: A script that uses the bridge to connect two voice agent servers
- **`AGENT_BRIDGE_README.md`**: This documentation file

## Overview

The WebsocketBridge acts as a relay between two voice agent servers, forwarding audio and control messages bidirectionally. This allows two AI agents to have a conversation with each other.

### Architecture

```
Agent 1 (Server 1)  <----->  WebsocketBridge  <----->  Agent 2 (Server 2)
  Port 8765                                              Port 8766
```

## Quick Start

### Step 1: Start Two Voice Agent Servers

First, you need two running voice agent servers on different ports.

**Terminal 1 - Start Agent 1:**
```bash
# Default port 8765
python bot_websocket_server.py
```

**Terminal 2 - Start Agent 2:**
```bash
# On port 8766 (you'll need to modify bot_websocket_server.py to use a different port)
# Or start a second instance with modified configuration
python bot_websocket_server.py
```

### Step 2: Connect the Agents

**Terminal 3 - Run the Bridge:**
```bash
python connect_two_agents.py \
    --agent1-url ws://localhost:8765 \
    --agent2-url ws://localhost:8766 \
    --agent1-name "Alice" \
    --agent2-name "Bob"
```

## Usage Examples

### Basic Connection
```bash
python connect_two_agents.py \
    --agent1-url ws://localhost:8765 \
    --agent2-url ws://localhost:8766
```

### With Custom Agent Names
```bash
python connect_two_agents.py \
    --agent1-url ws://localhost:8765 \
    --agent1-name "Assistant" \
    --agent2-url ws://localhost:8766 \
    --agent2-name "Customer"
```

### With Audio Filtering (Less Verbose Logs)
```bash
python connect_two_agents.py \
    --agent1-url ws://localhost:8765 \
    --agent2-url ws://localhost:8766 \
    --filter-audio
```

### Minimal Logging
```bash
python connect_two_agents.py \
    --agent1-url ws://localhost:8765 \
    --agent2-url ws://localhost:8766 \
    --no-log-messages
```

## Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--agent1-url` | `ws://localhost:8765` | WebSocket URL for the first agent |
| `--agent2-url` | `ws://localhost:8766` | WebSocket URL for the second agent |
| `--agent1-name` | `Agent1` | Name for the first agent (for logging) |
| `--agent2-name` | `Agent2` | Name for the second agent (for logging) |
| `--log-messages` | `True` | Log message traffic |
| `--no-log-messages` | - | Disable message traffic logging |
| `--filter-audio` | `False` | Filter audio frames from logging |
| `--init-delay` | `1.0` | Delay in seconds between agent initializations |

## Using the WebsocketBridge Class Programmatically

You can also use the `WebsocketBridge` class directly in your own code:

```python
import asyncio
from websocket_bridge import WebsocketBridge

async def main():
    # Create bridge
    bridge = WebsocketBridge(
        agent1_url="ws://localhost:8765",
        agent2_url="ws://localhost:8766",
        agent1_name="Alice",
        agent2_name="Bob",
        log_messages=True,
        filter_audio=True
    )
    
    # Connect and run
    try:
        await bridge.connect()
        await bridge.run()
    finally:
        await bridge.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
```

## Features

### WebsocketBridge Class

- **Bidirectional Message Forwarding**: Automatically forwards all messages between two agents
- **Connection Management**: Handles connection setup and teardown
- **Message Logging**: Optionally log all traffic with filtering options
- **Audio Filtering**: Filter binary audio data from logs to reduce verbosity
- **Error Handling**: Graceful handling of connection errors and disconnections
- **Async/Await**: Built on modern Python asyncio

### Key Methods

- `connect()`: Establish connections to both agents
- `disconnect()`: Close both connections gracefully
- `run()`: Start bidirectional message forwarding (runs until connection closes)
- `start()`: Convenience method that calls connect(), run(), and disconnect()
- `stop()`: Stop the bridge gracefully

## Configuration Tips

### Running Multiple Agent Servers

To test agent-to-agent communication, you need to run two separate voice agent servers. Here are some approaches:

1. **Modify port in code**: Edit `bot_websocket_server.py` line 151 to use different ports
2. **Use environment variables**: Set different configurations for each server
3. **Docker containers**: Run each agent in a separate container with different ports

### Agent Personalities

To make the conversation more interesting, configure each agent with different:
- System prompts (in `server_configs/`)
- TTS voices (different voice models)
- Response styles (adjust LLM parameters)

Example: Set Agent 1 as a customer and Agent 2 as a support representative by using different prompts.

## Logging

The bridge creates two log outputs:
- **Console output**: Timestamped, colored logs to stderr
- **File output**: `agent_bridge.log` with daily rotation

Logs include:
- Connection events (connect, disconnect)
- Message forwarding (with previews)
- Errors and warnings

## Troubleshooting

### Connection Refused
- Ensure both agent servers are running
- Verify the correct ports in the URLs
- Check firewall settings

### Agents Not Responding
- Check that agents are properly initialized
- Review agent server logs for errors
- Ensure agents have proper LLM, STT, and TTS configurations

### High CPU/Memory Usage
- Use `--filter-audio` to reduce logging overhead
- Use `--no-log-messages` for production use
- Monitor agent server resource usage separately

## Advanced Usage

### Custom Message Filtering

Extend the `WebsocketBridge` class to add custom message filtering:

```python
class CustomBridge(WebsocketBridge):
    def _should_log_message(self, message) -> bool:
        # Custom filtering logic
        if self._is_sensitive_data(message):
            return False
        return super()._should_log_message(message)
```

### Message Transformation

Override the `_forward_messages` method to transform messages:

```python
class TransformingBridge(WebsocketBridge):
    async def _forward_messages(self, source_ws, dest_ws, source_name, dest_name):
        async for message in source_ws:
            # Transform message
            transformed = self._transform(message)
            await dest_ws.send(transformed)
```

## Requirements

- Python 3.8+
- websockets library
- loguru library
- Running NeMo voice agent servers

## License

Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

Licensed under the Apache License, Version 2.0.

