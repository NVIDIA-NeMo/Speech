# Quick Start: Connecting Two Voice Agents

This guide will help you quickly set up two voice agents that can communicate with each other.

## Prerequisites

- Two NeMo voice agent servers configured and ready to run
- Python 3.8+
- Required dependencies installed (websockets, loguru)

## Option 1: Automated Demo (Easiest)

### Using Python (Cross-platform)
```bash
python run_agent_demo.py
```

### Using Bash (Linux/Mac)
```bash
./run_agent_demo.sh
```

This will automatically:
1. Start Agent 1 on port 8765
2. Start Agent 2 on port 8766
3. Connect them with the WebSocket bridge
4. Log all communication

**Stop the demo:** Press `Ctrl+C`

## Option 2: Manual Setup (More Control)

### Step 1: Start First Agent
```bash
# Terminal 1
python bot_websocket_server.py
```

### Step 2: Start Second Agent
```bash
# Terminal 2
python bot_websocket_server_alt.py --port 8766
```

### Step 3: Connect Agents
```bash
# Terminal 3
python connect_two_agents.py \
    --agent1-url ws://localhost:8765 \
    --agent2-url ws://localhost:8766 \
    --agent1-name "Alice" \
    --agent2-name "Bob"
```

## Configuration Options

### Demo Script Options
```bash
python run_agent_demo.py \
    --agent1-port 8765 \
    --agent2-port 8766 \
    --agent1-name "Alice" \
    --agent2-name "Bob" \
    --startup-delay 10
```

### Bridge Options
```bash
python connect_two_agents.py \
    --agent1-url ws://localhost:8765 \
    --agent2-url ws://localhost:8766 \
    --agent1-name "Alice" \
    --agent2-name "Bob" \
    --filter-audio        # Reduce logging verbosity
    --no-log-messages     # Disable message logging
```

## Customizing Agent Personalities

To make the conversation more interesting, configure each agent with different personalities:

1. Create custom prompt files in `example_prompts/`
2. Create custom configs in `server_configs/`
3. Set environment variables:

```bash
# For Agent 1 (Customer)
export SERVER_CONFIG_PATH=server_configs/customer_config.yaml
python bot_websocket_server.py

# For Agent 2 (Support)
export SERVER_CONFIG_PATH=server_configs/support_config.yaml
python bot_websocket_server_alt.py --port 8766
```

Example custom prompt for customer agent:
```text
You are a customer calling technical support. You have a problem with your 
internet connection. Be polite but frustrated. Ask questions and respond 
to the support agent's suggestions.
```

Example custom prompt for support agent:
```text
You are a technical support agent. Help customers solve their problems 
with patience and clear instructions. Ask diagnostic questions and provide 
step-by-step solutions.
```

## Monitoring

### Real-time Logs
```bash
# Agent 1 logs
tail -f bot_server.log

# Agent 2 logs
tail -f bot_server_alt_8766.log

# Bridge logs
tail -f agent_bridge.log
```

### Demo Logs (if using run_agent_demo.py/sh)
```bash
tail -f agent1_output.log
tail -f agent2_output.log
tail -f agent_bridge.log
```

## Troubleshooting

### "Connection refused"
- Ensure both agent servers are running
- Check that ports 8765 and 8766 are not in use
- Verify firewall settings

### Agents not responding
- Check agent server logs for initialization errors
- Ensure models (STT, TTS, LLM) are properly configured
- Verify sufficient GPU memory is available

### High CPU usage
- Use `--filter-audio` flag to reduce logging
- Use `--no-log-messages` for production
- Consider using lighter models

### Agents talking over each other
- Adjust `TURN_TAKING_BOT_STOP_DELAY` in config
- Modify VAD parameters for better turn detection
- Enable diarization if not already enabled

## Architecture

```
┌─────────────────┐          ┌─────────────────┐          ┌─────────────────┐
│   Agent 1       │          │  WebSocket      │          │   Agent 2       │
│   (Port 8765)   │◄────────►│  Bridge         │◄────────►│   (Port 8766)   │
│                 │          │                 │          │                 │
│  STT → LLM → TTS│          │  Bidirectional  │          │  STT → LLM → TTS│
└─────────────────┘          │  Message Relay  │          └─────────────────┘
                             └─────────────────┘
```

## Example Use Cases

1. **Testing Conversational AI**: Test how agents handle multi-turn conversations
2. **Dialogue Research**: Study natural conversation patterns between AI agents
3. **Quality Assurance**: Automated testing of agent responses
4. **Training Data Generation**: Generate synthetic conversational datasets
5. **Agent Comparison**: Compare different LLM/TTS configurations side-by-side

## Next Steps

- Review full documentation in `AGENT_BRIDGE_README.md`
- Explore `WebsocketBridge` class API in `websocket_bridge.py`
- Create custom agent configurations
- Implement message filtering or transformation

## Support

For issues or questions:
1. Check the full README: `AGENT_BRIDGE_README.md`
2. Review agent server logs
3. Enable verbose logging: remove `--filter-audio` flag

