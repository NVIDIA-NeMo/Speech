# Voice Agent Evaluation System

Evaluate the performance of a voice agent by using a simulated user to interact with the agent. The user and agent will be running a number of scenarios to evaluate the agent's task success rate and response latency.

## Architecture

```
┌─────────────────┐                    ┌─────────────────┐
│   Evaluator     │◄──────────────────►│   Evaluation    │
│   Agent         │    Audio + RTVI    │   Bridge        │
│ (Simulated User)│    Control Msgs    │  (Monitor +     │
│                 │    (Bidirectional) │   Controller)   │
└─────────────────┘                    └─────────────────┘
                                               ▲
                                               │
                                               │ Audio + RTVI
                                               │ Control Msgs
                                               │ (Bidirectional)
                                               ▼
                                       ┌─────────────────┐
                                       │   Target        │
                                       │   Agent         │
                                       │ (Being Tested)  │
                                       └─────────────────┘

Audio Flow:
1. Evaluator speaks → Bridge monitors & forwards → Target receives
2. Target responds → Bridge monitors & forwards → Evaluator receives
3. Bridge measures latency: Time from (1) stops to (2) starts

Metrics Collected:
- Response latency (evaluator stop → target start)
- Turn counts and transcripts
- Latency statistics (mean, median, P95, min, max)
- Task success rate (the percentage of scenarios that the agent yieds the expected result)
```

## Quick Start

### 1. Start the User and Agent

**Terminal 1 - Simulated User:**
```bash
cd examples/voice_agent/evaluation
export SERVER_CONFIG_PATH="server_configs/user.yaml"
export WEBSOCKET_PORT=8766
export PYTHONPATH=/path/to/NeMo:$PYTHONPATH
python bot_websocket_user.py
```

**Terminal 2 - Target Agent:**
```bash
cd examples/voice_agent/evaluation
export SERVER_CONFIG_PATH="server_configs/agent.yaml"
export WEBSOCKET_PORT=8765
export PYTHONPATH=/path/to/NeMo:$PYTHONPATH
python bot_websocket_agent.py
```

### 2. Run Evaluation

**Terminal 3 - Evaluation Bridge:**
```bash
cd examples/voice_agent/evaluation
python run_evaluation.py \
    --agent-url ws://localhost:8765 \
    --user-url ws://localhost:8766 \
    --domain restaurant
```
