# Voice Agent Evaluation System

Evaluate a voice agent by having a simulated user (another voice agent) talk to it through a live audio connection. The bridge routes audio, measures latency, captures the agent's final structured response, and scores success against a reference answer.

## Architecture

```
┌──────────────────────┐        Audio +        ┌──────────────────────┐       Audio +       ┌──────────────────────┐
│   User Bot Server    │        RTVI           │                      │       RTVI          │   Agent Bot Server   │
│  (Simulated User)    │◄─────────────────────►│        Bridge        │◄───────────────────►│  (Agent Under Test)  │
│                      │                       │                      │                     │                      │
│  ASR → LLM → TTS     │                       │  Audio routing       │                     │  ASR → LLM → TTS     │
│  WebSocket on 8766   │                       │  Latency metrics     │                     │  WebSocket on 8765   │
└──────────────────────┘                       │  Transcript capture  │                     └──────────────────────┘
                                               │  <final_response>    │
                                               │  <exit> detection    │
                                               │  RTVI prompt updates │
                                               └──────────────────────┘
```

- **Two independent WebSocket bot servers.** Each runs its own Pipecat pipeline (NeMo ASR → LLM → TTS) and speaks RTVI.
- **Bridge process.** Opens a WebSocket client to each bot, runs two threads (one per bot), and shuttles audio between them via thread-safe queues. Resamples audio at the source to match each bot's sample rate. Monitors RTVI events for transcripts, turn timing, `<final_response>` (structured result), and `<exit>` (early termination signal).
- **Control plane.** The bridge uses RTVI `update_system_prompt` (inject scenario prompts and tool configs), `reset` (clear context between scenarios), and `get_context_history` (retrieve LLM context + server logs at scenario end).

## Quick Start

### 1. Start the two bot servers

**Terminal 1 — Simulated User**

```bash
cd examples/voice_agent/evaluation
export PYTHONPATH=/path/to/NeMo:$PYTHONPATH
export SERVER_CONFIG_PATH=server_configs/user.yaml
export WEBSOCKET_PORT=8766
python bot_websocket_user.py
```

**Terminal 2 — Agent Under Test**

```bash
cd examples/voice_agent/evaluation
export PYTHONPATH=/path/to/NeMo:$PYTHONPATH
export SERVER_CONFIG_PATH=server_configs/agent.yaml
export WEBSOCKET_PORT=8765
python bot_websocket_agent.py
```

### 2. Run an evaluation

**Terminal 3 — Bridge**

```bash
cd examples/voice_agent/evaluation
python run_evaluation.py \
    --user-url ws://localhost:8766 \
    --agent-url ws://localhost:8765 \
    --domain restaurant
```

### 3. Scoring

By default, each scenario is scored by **strict dictionary comparison** between its `reference_answer` and the agent's `<final_response>` payload — every key/value in the reference must be present and matching in the prediction (extra keys in the prediction are allowed). Pass `--judge-url`, `--judge-model`, and `--judge-api-key` to additionally run an **LLM judge** that scores each scenario 0–1 using the full conversation and tool context. Both results are saved in `metrics.json` and `judge_result.json` respectively. See [Evaluation Methods](#evaluation-methods) for details.

## CLI Reference

### `run_evaluation.py` flags

| Flag | Description |
|------|-------------|
| `--user-url` | WebSocket URL of the user bot (default: `ws://localhost:8766`) |
| `--agent-url` | WebSocket URL of the agent bot (default: `ws://localhost:8765`) |
| `--scenarios <name …>` | Run specific scenarios by name |
| `--domain <name>` | Run all scenarios in a domain (matches `{domain}__*` prefix) |
| `--list` | List all registered scenarios and exit |
| `--list-domains` | List available domains and exit |
| `--duration <seconds>` | Default max duration per scenario (default: 120). Overridden by scenario's own `max_duration` if set. |
| `--pause <seconds>` | Pause between scenarios (default: 0.5) |
| `--output-dir <path>` | Output directory root (default: `./eval_results`) |
| `--output-sample-rate <hz>` | Sample rate for recorded stereo WAV (default: 16000) |
| `--judge-url <url>` | LLM judge endpoint (OpenAI-compatible chat completions) |
| `--judge-model <model>` | Judge model name |
| `--judge-api-key <key>` | Judge API key (defaults to env var if set) |
| `--judge-threshold <threshold>` | Threshold for the LLM judge score if binary result is desired (default: 0.95) |

If neither `--scenarios` nor `--domain` is given, all registered scenarios run.

### Bot server environment variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `SERVER_CONFIG_PATH` | Path to the YAML server config | `server_configs/agent.yaml` / `server_configs/user.yaml` |
| `SERVER_HOST` | Host to bind | `0.0.0.0` |
| `WEBSOCKET_PORT` | Pipecat WebSocket port | `8765` (agent) / `8766` (user) |

## Available Scenarios

| Domain | Count | Summary tool | Description |
|--------|-------|--------------|-------------|
| `restaurant` | 11 | `PlaceOrderTool`, `JoinWaitListTool` / `DropWaitListTool` | Ordering food at pizza, burger, and deli restaurants, plus a waitlist join/drop scenario (demonstrates shared state across tools). |
| `customer_service` | 10 | `ResolveTicketTool` | TechCorp customer service — billing disputes, order delays, defective returns, plan upgrades, account access, warranty claims, subscription cancellations, wrong items, and service outages. |
| `qa` | 10 | `SaveQuestionAnswerTool` | Single-turn Q&A — geography, math, science, history, literature, weather (uses `GetCityWeatherTool`), and general knowledge. |
| *legacy (no domain)* | 4 | — | `fastbite`, `simple_qa_1`, `simple_qa_2`, `simple_qa_3` — original scenarios kept for backward compatibility. |

Run `python run_evaluation.py --list` for the full list of scenario names, or `--list-domains` for just the domain summary.

## Evaluation Methods

### 1. Strict dictionary comparison (default)

`check_if_task_success` performs a recursive comparison between each scenario's `reference_answer` and the agent's `<final_response>` payload:

- **Dict vs. Dict** — every key/value in the reference must be present and match in the prediction. Extra keys in the prediction are allowed.
- **Dict vs. List-of-Dicts** — the reference dict must match the **last** dict in the prediction list.
- **List-of-Dicts vs. List-of-Dicts** — every dict in the reference must find a matching dict in the prediction (order-independent, each prediction can match at most one reference).

String matching respects the scenario's `ignore_capitalization`, `ignore_punctuation`, and `clean_text` flags. Numeric values are compared with `np.isclose`.

The boolean result is saved as `is_successful` in `metrics.json`. If LLM judge is enabled, the result is overwritten by the judge's score.

### 2. LLM judge (optional)

When `--judge-url` and `--judge-model` are provided, an additional `LLMJudge` scores each scenario on a 0–1 scale. It receives the `reference_answer`, the `final_agent_response`, the full conversation turn list, and the LLM context history, then returns a score with a short reasoning string.

The judge is robust to extra conversational content the agent produces (apologies, pleasantries, paraphrased information) that would fail strict matching but still correctly accomplishes the task.

Output: `judge_result.json` per scenario with `{score: float, reason: str}`.

## Output Structure

Each run creates a timestamped session directory. Within it, each scenario has its own subdirectory.

```
eval_results/eval_YYYYMMDD_HHMMSS/
├── evaluation_log.txt              # Top-level runner log
├── all_metrics.json                # Aggregated metrics across all scenarios
├── all_latencies.csv               # Every latency measurement as CSV rows
├── all_summary.txt                 # Human-readable summary (per-scenario + overall stats)
└── <scenario_name>/                # One directory per scenario
    ├── conversation_log.txt        # Timestamped transcript with latency annotations
    ├── conversation_log.seglst.json  # segLST-format speaker segments (for ASR evaluation)
    ├── conversation_log.wav        # Stereo audio: L=user→agent, R=agent→user
    ├── bridge_log.txt              # Bridge debug/info log
    ├── final_agent_response.json   # All <final_response> payloads captured from the agent
    ├── metrics.json                # Per-scenario metrics + is_successful flag
    ├── judge_result.json           # LLM judge output (present only if judge was enabled)
    ├── scenario_config/            # Snapshot of the scenario definition used for this run
    │   ├── metadata.json           # name, description, max_duration, matching flags, noise config
    │   ├── reference_answer.json   # The expected answer that was compared against
    │   ├── user_prompt.txt         # Rendered system prompt for the user bot
    │   ├── user_tools.json         # Tool config sent to the user bot
    │   ├── agent_prompt.txt        # Rendered system prompt for the agent bot
    │   └── agent_tools.json        # Tool config sent to the agent bot
    ├── bot_logs_user/
    │   └── llm_context.json        # Full LLM context history retrieved from the user bot
    └── bot_logs_agent/
        └── llm_context.json        # Full LLM context history retrieved from the agent bot
```

Key files to inspect:
- **`metrics.json`** — turn count, duration, latency stats (mean/P50/P95/min/max), individual latencies, `is_successful`.
- **`final_agent_response.json`** — what the agent actually produced (a list of all summary tool payloads in order).
- **`conversation_log.wav`** — listen to the actual conversation.
- **`bot_logs_{user,agent}/llm_context.json`** — full LLM conversation including tool calls and results, useful for debugging agent behavior.

## Extending the System

Three extension points, in increasing order of scope. Each links to the detailed reference below.

1. **[New scenario](#scenario-structure)** — subclass an existing domain base and override the properties that differ. ~30–60 lines.
2. **[New tool](#tool-system)** — subclass `StandardSchemaTool` or `SendScenarioSummaryTool`, register with `@register_schema_tool_for_eval`, import the module in `tools/__init__.py`.
3. **[New domain](#creating-a-new-scenario)** — create a `{Domain}BaseScenario` in `scenarios/data/{domain}.py` that sets domain defaults (persona, guidelines, tool defaults including a domain summary tool + `EndConversationTool`), add a `tools/{domain}_tools.py` if the domain needs its own tools, and import the new module in `scenarios/data/__init__.py`.

## Scenario Structure

A scenario fully specifies what both the user and the agent do during one evaluation run. Each is a Python class with 8 properties plus some scenario-level fields.

### The 8 properties (per side: user and agent)

| Property | Type | Purpose |
|----------|------|---------|
| `{side}_persona` | `Persona` | `role`, `name`, `background`, `personality`, optional `language`/`accent`. Rendered as the opening lines of the system prompt. |
| `{side}_task` | `Task` | `goal` and `background`. The single objective this side is trying to achieve. |
| `{side}_actions` | `Actions` | Ordered `instructions` (step-by-step script) and persistent `guidelines` (always-apply rules). |
| `{side}_resources` | `Resources` | `tools` dict (tool class name → constructor kwargs), `documents`, free-form `information` strings. |

### Scenario-level fields

| Field | Purpose |
|-------|---------|
| `name` | Unique scenario ID. Convention: `{domain}__{scenario_name}` (e.g., `restaurant__pizza_pepperoni`). |
| `description` | Short human-readable summary. |
| `max_duration` | Max scenario duration in seconds. Overrides the CLI default. |
| `reference_answer` | The expected `<final_response>` payload. Dict or list-of-dicts. |
| `ignore_capitalization` | String matching: case-insensitive. |
| `ignore_punctuation` | String matching: strip punctuation. |
| `clean_text` | String matching: apply ASR text cleaning. |
| `noise_config` | Optional `NoiseConfig` to inject background noise into the user→agent channel. |

### Domain organization

Scenarios are organized by domain using a **base class pattern**:

- A domain base class (e.g., `RestaurantBaseScenario`, `CustomerServiceBaseScenario`, `QABaseScenario`) implements all 8 properties with domain-level defaults. It is **not** registered.
- Concrete scenarios inherit the base and override only the properties that differ — typically `user_persona`, `user_task`, `user_actions`, `agent_actions`, `agent_resources`, and `reference_answer`.
- Each domain lives in one file under `nemo/agents/voice_agent/evaluation/scenarios/data/` (e.g., `restaurant.py`, `customer_service.py`, `qa.py`).

## Creating a New Scenario

### 1. Pick or create a domain

Existing domains: `restaurant`, `customer_service`, `qa`. Add new scenarios to the matching file. For a brand-new domain, create a new file with a `{Domain}BaseScenario` base class and register an import in `scenarios/data/__init__.py`.

### 2. Subclass the domain base

Override only what's specific to your scenario. Inherited properties come from the base.

```python
from nemo.agents.voice_agent.evaluation.scenarios import register_eval_scenario
from nemo.agents.voice_agent.evaluation.scenarios.classes import Actions, Persona, Resources, Task
from nemo.agents.voice_agent.evaluation.scenarios.data.restaurant import RestaurantBaseScenario

PIZZA_PALACE_MENU = """
## Pizza Palace Menu
Pepperoni Pizza - $9.99
Extra Cheese - $1.50
"""

@register_eval_scenario
class PizzaPepperoni(RestaurantBaseScenario):
    name = "restaurant__pizza_pepperoni"
    description = "Order a pepperoni pizza with extra cheese at Pizza Palace"
    reference_answer = {
        "items": [
            {"name": "Pepperoni Pizza", "unit_price": "9.99", "quantity": "1"},
            {"name": "Extra Cheese", "unit_price": "1.50", "quantity": "1"},
        ],
        "customer_name": "Charlie",
        "customer_phone": "314-527-8960",
        "total_price": "11.49",
    }

    @property
    def user_persona(self) -> Persona:
        return Persona(
            role="human user",
            name="Charlie",
            background="You work as a teacher. Your phone number is 314-527-8960.",
            personality="Communicative, friendly, decisive.",
        )

    @property
    def user_task(self) -> Task:
        return Task(goal="Order a pepperoni pizza with extra cheese.")

    @property
    def user_actions(self) -> Actions:
        return Actions(
            instructions=[
                "Ask for pizza options.",
                "Order one pepperoni pizza.",
                "Ask if extra cheese is available and add it.",
                "Finish the order and ask for the price.",
            ],
        )

    @property
    def agent_resources(self) -> Resources:
        return Resources(
            tools={
                "GetMenuTool": {"menu": PIZZA_PALACE_MENU},
                "PlaceOrderTool": {"auto_validate": "False"},
                "EndConversationTool": {},
            },
        )
```

### 3. Verify

```bash
python run_evaluation.py --list
# → restaurant__pizza_pepperoni should appear

python run_evaluation.py --scenarios restaurant__pizza_pepperoni
```

## Tool System

### Tool configuration

Tools are referenced by class name in `Resources.tools` as `{tool_name: constructor_kwargs}`. The bridge serializes this to JSON and the bot server instantiates each tool by calling `TheTool(**tool_args)`. Different scenarios can pass different kwargs to the same tool class — e.g., `GetMenuTool({"menu": PIZZA_PALACE_MENU})` vs. `GetMenuTool({"menu": BURGER_BARN_MENU})`.

### Shared state

Tools in the same scenario can share mutable state via a `shared_state` dict that is injected into their constructors if they declare it. The bridge creates one fresh dict per scenario and passes the same reference to every tool that accepts it.

Example: `JoinWaitListTool`, `DropWaitListTool`, and `GetWaitlistTool` all read and write `shared_state["waitlist"]`, so when the agent joins a customer via one tool, checking the list via another returns the updated data.

### Mandatory tools per scenario

Every scenario must include:

1. **A summary tool** that inherits `SendScenarioSummaryTool`. This tool wraps the agent's final structured result in `<final_response>` tags so the bridge captures it in `final_agent_response.json`. Examples: `PlaceOrderTool` (restaurant), `ResolveTicketTool` (customer service), `SaveQuestionAnswerTool` (QA), `JoinWaitListTool` / `DropWaitListTool` (waitlist). Without a summary tool, scoring has nothing to evaluate against.

2. **`EndConversationTool`** — sends an `<exit>` tag that triggers the bridge to stop the scenario early. Without it, the bridge waits for the full `max_duration`, which can cause server-side WebSocket keepalive timeouts during idle periods.

The domain base class should always include both in `agent_resources.tools` and instruct the agent (via `agent_actions.guidelines`) to call them.

### Creating a new tool

Subclass `StandardSchemaTool` (or `SendScenarioSummaryTool` for summary tools) and register with `@register_schema_tool_for_eval`:

```python
from nemo.agents.voice_agent.evaluation.tools import register_schema_tool_for_eval
from nemo.agents.voice_agent.utils.tool_calling import StandardSchemaTool

@register_schema_tool_for_eval
class GetMenuTool(StandardSchemaTool):
    def __init__(self, *, menu: str = "", description: Optional[str] = None):
        super().__init__(description=description or "Get the restaurant menu.")
        self.menu = menu

    @property
    def properties(self):
        return {}

    @property
    def required_properties(self):
        return []

    async def _execute(self, params):
        await params.result_callback({"menu": self.menu})
```

Add the module to `tools/__init__.py` so its `@register_schema_tool_for_eval` decorator fires. The constructor can accept:
- Any number of data kwargs (e.g., `menu`, `accounts`, `orders`)
- `shared_state: Optional[dict]` — auto-injected if declared
- `rtvi: Optional[RTVIProcessor]` — auto-injected if declared (needed for summary tools)
