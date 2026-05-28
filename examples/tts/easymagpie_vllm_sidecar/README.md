# EasyMagpie vLLM Sidecar

vLLM-backed TTS sidecar for the NemotronH SmallMamba checkpoint. Runs the AR
loop (backbone + Local Transformer + sampler) in a separate process from the
voice-agent server because vllm-omni 0.19.1's torch/triton/nccl pins cannot
coexist with NeMo's `nemo_virtual_environment`.

The voice-agent talks to this sidecar over HTTP (NDJSON streaming on
`/tts/stream`); see
[`nemo/agents/voice_agent/pipecat/services/nemo/easymagpie_vllm_client.py`](../../../nemo/agents/voice_agent/pipecat/services/nemo/easymagpie_vllm_client.py)
for the client and
[`server_configs/tts_configs/easy_magpie_smallmamba_vllm.yaml`](../../voice_agent/server/server_configs/tts_configs/easy_magpie_smallmamba_vllm.yaml)
for the yaml the voice-agent reads.

## Install

> **Do NOT install into `nemo_virtual_environment`.** This package targets a
> separate venv (`vllm_omni_env`) that has vllm-omni 0.19.1 + NemotronH-
> compatible torch / triton / nccl + a working `mamba-ssm`. Installing into
> the NeMo env will break NeMo.

```bash
# From an already-bootstrapped vllm_omni_env:
source /path/to/vllm_omni_env/bin/activate
pip install -e .
```

This registers `EasyMagpieSmallMamba` in vLLM's ModelRegistry via the
`vllm.general_plugins` entry point. The legacy `EasyMagpieSmallMambaV2` arch
name is also registered (alias) so existing checkpoints still resolve.

## Run

```bash
export TMPDIR=/mnt/n1_mount/personal/tmp_nemo
python -m easymagpie_server.server --port 18765

# or, via the entry point:
easymagpie-sidecar --port 18765

# Smoke test:
curl -s http://127.0.0.1:18765/healthz   # → {"status":"ok"}
```

## Package layout

- `easymagpie_vllm/` — model + streaming head + LT backends (compile +
  max-autotune FP16 is the default; trt_fused available behind an env knob).
- `easymagpie_server/` — FastAPI sidecar (`/healthz`, `/tts/wire_tokenizer`,
  `/tts/stream`).
- `vllm_plugin_easymagpie/` — `vllm.general_plugins` entry-point package that
  registers `EasyMagpieSmallMamba` (and the `EasyMagpieSmallMambaV2` alias)
  with vLLM's ModelRegistry.
- `dev/` — workstation-only tooling: `refactor_harness.py` (A/B
  benchmarking — WER + UTMOS + RTF), `say.py` (CLI wrapper), `smoke_client.py`
  (curl-equivalent for `/tts/stream`), integration tests.

## Notes

- First request after sidecar boot takes 30–90s on cold Inductor/Triton
  cache. Subsequent runs reuse `~/.cache/vllm`, `~/.triton/cache`, and
  `$TMPDIR/torchinductor_subhankarg`.
- `max_num_seqs=1` today — single-request serving. Productionizing batched
  serving requires per-request state isolation (see
  [`dev_scripts/easymagpietts/plan/phase6_voice_agent_tts_service.md`](../../../dev_scripts/easymagpietts/plan/phase6_voice_agent_tts_service.md)
  Workstream C1).
- vLLM 0.19.1 in-tree patch required at `vllm/v1/worker/gpu_model_runner.py`
  (PR #37679 — assertion removal at line 6675).
