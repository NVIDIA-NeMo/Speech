# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this directory is

A self-contained example of a real-time voice agent built on **Pipecat** (`pipecat-ai==0.0.98`) wired together with NeMo speech models and either a HuggingFace or vLLM LLM backend. The example has its **own `pyproject.toml` + `uv.lock`** and is decoupled from the parent NeMo repo's install:

- Python **3.12–3.13** (not 3.10+ like the rest of NeMo).
- Default install pulls **CUDA 13.0** PyTorch/vLLM wheels (`torch-backend = "cu130"` in `pyproject.toml`). Override via `pyproject.toml` if you need cu128/cu124/cpu.
- The actual implementation lives in `nemo/agents/voice_agent/` at the repo root — this directory only contains the example entry-point (`server/server.py`), YAML configs, the browser client, evaluation harness, and tests. Source-code edits typically belong under `nemo/agents/voice_agent/`.

## Install & run

```bash
# One-shot install (apt deps + uv + venv): bash install.sh
uv sync
source .venv/bin/activate

# Server (terminal 1)
export PYTHONPATH=/path/to/NeMo:$PYTHONPATH      # path to the repo root containing nemo/
# export SERVER_CONFIG_PATH=server/server_configs/default.yaml   # override the default config
# export HF_TOKEN=hf_...                          # for gated HF models
python ./server/server.py

# Client (terminal 2)
cd client && npm install && npm run dev          # vite on http://localhost:5173
```

The server binds two ports: a **WebSocket** for the audio pipeline (`WEBSOCKET_PORT`, default `8765`) and a **FastAPI** control plane (`FASTAPI_PORT`, default `7860`). Bind addresses come from `SERVER_HOST` (default `0.0.0.0`).

Browsers block mic access on plain HTTP — Chrome users must allow-list `http://<host>:5173/` via `chrome://flags/#unsafely-treat-insecure-origin-as-secure`.

## Tests

```bash
pytest tests/ -v                                          # all
pytest tests/test_config_manager.py -v                    # config loader
pytest tests/test_reasoning_budget_logits_processor.py    # vLLM reasoning-budget processor (needs CUDA + tokenizer download)
```

`tests/test_*.py` insert the repo root into `sys.path` so they always test the working-tree NeMo, not whatever is `pip install`ed.

## Server architecture

`server/server.py:run_bot_websocket()` is the whole show — it loads a YAML config and assembles a Pipecat pipeline:

```
ws.input → RTVI → STT → [Diar?] → TurnTaking → UserAggregator → LLM → TTS → ws.output → AssistantAggregator
```

Components are constructed via the **builder pattern** in `nemo/agents/voice_agent/pipecat/services/nemo/builders.py` (`build_stt`, `build_diar`, `build_llm`, `build_tts`, `build_turn_taking`, `build_vad_analyzer`, `build_ws_transport`, `build_audio_logger`, `build_context_and_aggregators`). The example file rarely needs editing — most behavioral changes happen in YAML or in the builders/services under `nemo/agents/voice_agent/`.

Key cross-cutting concepts:

- **`ConfigManager`** (`nemo/agents/voice_agent/utils/config_manager.py`) loads `server/server_configs/default.yaml`, then merges in the model-specific YAML referenced by each component's `model_config:` field (or auto-resolves via `server/model_registry.yaml` when `server.use_model_registry: true`). Configs use OmegaConf interpolation (e.g. `${llm.temperature}`) — be aware when adding new keys.
- **LLM backend selection.** `llm.type` is `auto | hf | vllm`. `auto` tries vLLM first and falls back to HF. vLLM is required for tool calling. When `start_vllm_on_init: true` the server spawns vLLM via `vllm serve` with the flags in `vllm_server_params`; otherwise you must start vLLM in another terminal (see README for the Nemotron-Nano-3 30B example).
- **Reasoning / thinking mode.** Off by default. Enable via `llm.enable_reasoning: true` (which switches to the sibling `*_think.yaml` config). `tts.think_tokens=["<think>","</think>"]` causes TTS to skip the reasoning span, so the user only hears the final answer. For vLLM, `--reasoning-parser` filters reasoning out of the OpenAI response entirely (see `server/parsers/nano_v3_reasoning_parser.py`).
- **Backchannels.** `turn_taking.backchannel_phrases_path` (or an inline list) prevents short utterances like "uh-huh" from interrupting the bot. Set to `null` to make any speech interrupt.
- **Single-connection server.** A new WebSocket connection disconnects the previous one (LLM context is preserved). Don't add multi-tenant logic here; this example is single-user by design.

## Config layout (`server/server_configs/`)

```
default.yaml                          # top-level: server/transport/vad/stt/diar/turn_taking/llm/tts
llm_configs/<model>.yaml              # per-model llm sub-config (HF + vLLM params)
llm_configs/<model>_think.yaml        # reasoning-mode variant of the same model
tts_configs/<model>.yaml              # kokoro / fastpitch-hifigan / magpie
stt_configs/nemo_cache_aware_streaming.yaml
NVIDIA_NeMo_models.yaml               # extra NeMo-hosted model defs
```

`server/example_prompts/*.txt` holds reusable system prompts referenceable from `llm.system_prompt` (path-or-literal).

## Tool calling

Two extension points (only works with `llm.type: vllm` + a model whose vLLM tool parser is configured):

1. **Direct functions** — write an async function and pass it to `register_direct_tools_to_llm(...)` in `server.py`. Example: `tool_get_city_weather` from `nemo/agents/voice_agent/utils/tool_calling/basic_tools.py`.
2. **Component-owned tools** — mix `ToolCallingMixin` into a service (STT/TTS/Diar/LLM/TurnTaking) and implement `setup_tool_calling()`. The mixin lives at `nemo/agents/voice_agent/utils/tool_calling/mixins.py`; `KokoroTTSService` in `pipecat/services/nemo/tts.py` is the canonical example (e.g. "speak faster", "switch to British accent").

## Evaluation harness (`evaluation/`)

A separate two-bot system: a **simulated user bot** talks to the **agent under test** via a bridge that shuttles audio between two WebSocket Pipecat servers, captures `<final_response>` payloads, and scores them. See `evaluation/README.md` for the full architecture, scenario authoring guide, and tool-system reference. Quick run:

```bash
# Three terminals: user bot (8766), agent bot (8765), bridge
python evaluation/bot_websocket_user.py     # WEBSOCKET_PORT=8766, SERVER_CONFIG_PATH=server_configs/user.yaml
python evaluation/bot_websocket_agent.py    # WEBSOCKET_PORT=8765, SERVER_CONFIG_PATH=server_configs/agent.yaml
python evaluation/run_evaluation.py --domain restaurant
```

Scenario classes live under `nemo/agents/voice_agent/evaluation/scenarios/data/` (one file per domain: `restaurant.py`, `customer_service.py`, `qa.py`, …) and tools under `nemo/agents/voice_agent/evaluation/tools/`. Adding a scenario: subclass the domain's `*BaseScenario`, decorate with `@register_eval_scenario`, override only what differs.

### Eval framework key concepts (read before editing)

The eval framework has evolved beyond a simple `<final_response>` capture. The pieces below are easy to miss if you only read `run_evaluation.py`.

**Shared state via RTVI.** Each side (user / agent) holds a per-scenario `shared_state` dict that the bridge seeds at scenario start. The handle is a `SharedStateRef` dataclass (mirrors `TaskRef`) published by `create_update_system_prompt_action(...)` in `nemo/agents/voice_agent/pipecat/processors/frameworks/rtvi_actions.py` — it gives later RTVI actions a mutable view of the same dict. The bridge passes initial state via the `shared_state_init` JSON-string argument of `update_system_prompt`. Scenarios populate state by overriding `Scenario.setup_shared_state(self, state, side)` (in `scenarios/classes.py`) — same method called twice with `side="user"` / `side="agent"`.

**Bridge-pull summary (not LLM-callable).** End-of-scenario state is **pulled** by the bridge after `<exit>`, not pushed by an LLM tool call. The bridge calls `_retrieve_scenario_summary(ws)` in `nemo/agents/voice_agent/evaluation/bridge.py`, which sends an RTVI `get_scenario_summary` action; the handler (`create_get_scenario_summary_action`) returns `{"actions": [...], "db": {...}}` read straight from `shared_state`. This eliminates the previous double-emit / forgot-to-call / mid-conversation-call class of bugs. **Don't reintroduce a `SubmitTransactionSummaryTool`-style LLM-callable summary.**

**DB-state hash matching (primary signal).** When a scenario sets `expected_scenario_db` (a `cached_property` on the class), the runner ignores the action-list comparator and instead hashes the agent's final `shared_state["db"]` via `get_dict_hash` (`nemo/agents/voice_agent/evaluation/db_hash.py`, adapted from eva 0.1.3 / tau-2-bench style). The hash normalizes floats (`1.0 → 1`), `"none" → None`, and uses `ORDER_INDEPENDENT_LIST_FIELDS` for set-like fields; `HASH_EXCLUDED_KEYS = {"session"}` skips per-run noise. On mismatch the runner writes a structured `db_state_diff` (tables → records → fields) via `compute_db_diff` for debugging. The action-list (`reference_answer`) remains as a secondary signal. Aggregate: `db_state_success_rate` printed by the runner.

**Auto-aggregated action records.** Each write tool extends `WriteAirlineTool` (in `nemo/agents/voice_agent/evaluation/tools/eva_airline_tools.py`) and calls `self._record_action({...})` on success — the record is appended to `shared_state["actions"]` so the bridge picks it up via the pull. The action `type` must come from the locked `AIRLINE_ACTION_TYPES` vocabulary (1:1 with eva tool names). Read tools don't record.

**Symmetric DB transfer.** The bridge sends the full original DB content (not a path) to the agent via `shared_state_init`. The agent mutates its in-memory copy through tools; the bridge pulls the full mutated DB back at end-of-scenario. There is also a `db_path` fallback for legacy paths — see the `state["db_path"]` branch in the action handler.

### `eva_airline` domain layout

```
nemo/agents/voice_agent/evaluation/
├── scenarios/data/eva_airline/      # package, not a single file
│   ├── __init__.py                  # re-exports EvaAirlineBaseScenario; imports group_Nx
│   ├── base.py                      # EvaAirlineBaseScenario + 5 hand-authored seeds
│   │                                # (1.1.2, 2.1.1, 3.1.3, 5.1.1, 7.2.1)
│   └── group_{1..7}x.py             # auto-scaffolded scenarios per eva sub-flow
├── tools/eva_airline_tools.py       # 15 ported tools + WriteAirlineTool base
├── tools/eva_airline_params.py      # Pydantic schemas for tool args
└── db_hash.py                       # eva-compatible normalize + hash
```

`EvaAirlineBaseScenario` derives everything from a single class attribute `eva_id` (e.g. `"1.1.2"`) via `cached_property`: `current_date`, `_scenario_db`, `expected_scenario_db`. The dataset metadata is read once per process via `_load_eva_airline_dataset_index()` (cached at module level). Subclasses only declare `name`, `eva_id`, `description`, `user_persona`, `user_task`, `user_actions`.

Voice-readability rule on `EvaAirlineBaseScenario.VOICE_ALPHANUMERIC_RULE`: alphanumerics are spelled **canonical-first**: `EPXYEK (spelled out as E, P, X, Y, E, K)`. Use this constant in both agent and user guidelines.

Fixtures live in `examples/voice_agent/evaluation/data/` (resolved by `get_eval_data_root()`, override via `EVAL_DATA_ROOT`). The directory has a `README.md` recording upstream source + license for each domain — append a section when adding a new source. The `get_eval_data_root()` helper is at `nemo/agents/voice_agent/evaluation/__init__.py` and uses `parents[4]` to walk from `nemo/agents/voice_agent/evaluation/__init__.py` to the repo root.

### Scaffolding more eva scenarios

The 5 seed scenarios in `base.py` are hand-authored; the rest are scaffolded from `eva_airline_dataset.jsonl` via:

```bash
python examples/voice_agent/nemo_experiments/generate_eva_airline_scaffolds.py --major 4 \
    >> nemo/agents/voice_agent/evaluation/scenarios/data/eva_airline/group_4x.py
```

The generator (in `nemo_experiments/`, gitignored personal-scratch dir) emits one `@register_eval_scenario` class per dataset entry, applies the alphanumeric voice rule, and reads `must_have_criteria` / `negotiation_behavior` / `edge_cases` into guidelines. **The output is a starting point, not final** — hand-review prose and prune negotiation/edge-case bullets before committing.

### Running a single eva_airline scenario

```bash
# After both bots are running (see Quick Start above):
python evaluation/run_evaluation.py \
    --scenarios eva_airline__1_1_2 \
    --duration 900                    # bump from default 600s — voice round-trips are ~10× slower than text
```

Scenario names map from eva ids: `"1.1.2" → "eva_airline__1_1_2"`, class names `"1.1.2" → "EvaAirline112"`.

### Known limitations

- **Parakeet STT misrecognizes spelled alphanumerics.** Letter sequences and digit-words (`"for"` vs `"four"`, `"B Z I W"`) frequently get mangled. Diagnose by checking `bot_logs_user/llm_context.json` to confirm the user simulator emitted the correct text before blaming the user-side LLM.
- **Action list lookups are case-sensitive.** Tool action `type` strings must match `AIRLINE_ACTION_TYPES` exactly.
- **DB diff isn't shown unless `expected_scenario_db` is set.** Scenarios without a ground-truth DB fall back to action-list comparison only.

## Code style

The parent repo's style rules (line length 119, black with `skip_string_normalization`, isort `profile=black`) apply. Most of this directory is **excluded from black's auto-format scope** in the parent `pyproject.toml`'s `extend-exclude` — only reformat files you're actively changing, and don't bulk-reformat unrelated code. Run lint via the repo-root command: `python setup.py style --scope examples/voice_agent --fix`.

## Gotchas

- **Don't run `uv sync` inside an active conda env** — `install.sh` exits early in that case because conda's gcc + system Python headers break C extensions like `cdifflib`. Run `conda deactivate` first.
- The egg-info dir (`nemo_voice_agent.egg-info/`), `.venv/`, `nemo_experiments/` (personal scratch + `.env`), and `*.log` files are local artifacts — don't commit changes to them.
- `server/parsers/*.py` are vLLM **plugins** loaded by path (`--tool-parser-plugin`, `--reasoning-parser-plugin`). They run inside the vLLM process, not the bot server, so logging/imports there have a different runtime than the rest of the codebase.
- `bot_server.log` saves the logs from the pipecat pipeline, by default it's rotated every day. Recent failures: check the newest `bot_server.<timestamp>.log`, not just `bot_server.log` (which may be from an in-flight run).
