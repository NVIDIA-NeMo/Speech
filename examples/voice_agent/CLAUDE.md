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

## Code style

The parent repo's style rules (line length 119, black with `skip_string_normalization`, isort `profile=black`) apply. Most of this directory is **excluded from black's auto-format scope** in the parent `pyproject.toml`'s `extend-exclude` — only reformat files you're actively changing, and don't bulk-reformat unrelated code. Run lint via the repo-root command: `python setup.py style --scope examples/voice_agent --fix`.

## Gotchas

- **Don't run `uv sync` inside an active conda env** — `install.sh` exits early in that case because conda's gcc + system Python headers break C extensions like `cdifflib`. Run `conda deactivate` first.
- The egg-info dir (`nemo_voice_agent.egg-info/`), `.venv/`, `nemo_experiments/` (personal scratch + `.env`), and `*.log` files are local artifacts — don't commit changes to them.
- `server/parsers/*.py` are vLLM **plugins** loaded by path (`--tool-parser-plugin`, `--reasoning-parser-plugin`). They run inside the vLLM process, not the bot server, so logging/imports there have a different runtime than the rest of the codebase.
- `bot_server.log` saves the logs from the pipecat pipeline, by default it's rotated every day. Recent failures: check the newest `bot_server.<timestamp>.log`, not just `bot_server.log` (which may be from an in-flight run).
