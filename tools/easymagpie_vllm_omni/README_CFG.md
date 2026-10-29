## EasyMagpieTTS — V2 classifier-free guidance (CFG)

For checkpoint conversion, serving-environment setup, and standard synthesis,
see the [main README](README.md). This guide covers V2 CFG synthesis, cache
pairing, deployment profiles, CUDA graphs, benchmarks, and validation.

### V2 CFG cache-pairing diagnostic

The first CFG milestone provides an explicit diagnostic pipeline,
[`deploy/easymagpie_cfg_pairing.yaml`](deploy/easymagpie_cfg_pairing.yaml).
It allocates two physical Stage-0 sequences per logical request, each with its
own backbone attention, Mamba, and temporal acoustic-predictor cache state.
Both sequences advance together; only the public sequence's output reaches the
codec or client. Completion and cancellation free both histories.

The cache diagnostic uses the existing conditional inputs on both branches. It accepts the internal
`additional_information._cfg_pairing_probe: true` marker through direct `AsyncOmni`
requests. Select the synthesis profile below for `use_cfg: true`, including
HTTP/WebSocket requests as described below.

In the pinned GPU serving environment, run the completion/abort/reuse smoke test:

```bash
VLLM_PLUGINS=easymagpie_omni python tools/easymagpie_vllm_omni/scripts/probe_cfg_pairing.py \
  --model /path/to/converted_v2 --speaker eng
```

The diagnostic supports one non-resumable known-speaker request at a time,
eager execution, synchronous scheduling, and disabled prefix caching and chunked
prefill. Stage-0 capacity counts physical sequences: one logical request needs
two slots and a token budget covering both prompts. Split scheduling or preemption
aborts the diagnostic before the worker executes the unsafe step. Use this profile
for development; it deliberately fails on unsupported scheduling rather than
recovering or serving other requests concurrently.

The scheduler tests exercise the real vLLM/Omni 0.24 classes and hybrid cache-block
allocator on CPU. The smoke test additionally requires GPU execution with a V2
checkpoint; CPU allocation bookkeeping alone does not establish numerical cache
correctness or audio parity.

### V2 CFG synthesis

[`deploy/easymagpie_cfg.yaml`](deploy/easymagpie_cfg.yaml) adds guided synthesis
for known-speaker requests with complete text or bounded incremental text. Its two physical
Stage-0 slots fit one active CFG request; additional requests queue. Guided and
conditional-only requests use the same explicit pipeline, with `use_cfg` selecting
whether the scheduler creates a private branch. The pairing-only diagnostic keeps
its separate single-request scheduler.

The corrected profile passed the GPU completion/abort/reuse probe at CFG 2.5,
acoustic temperature 0.7, top-k 80 before the mixed-admission extension. Mixed
concurrency also passed the mixed GPU cancellation/reuse probe;
see the profile and probe below. Raw reference/user audio,
retained turns, preemption, asynchronous scheduling, chunked prefill,
and prefix caching remain unsupported in this CFG path. The default profiles keep CUDA graphs
disabled; an experimental decode-only graph profile is described below. Scale 1 did not match
conditional-only greedy inference on the tested prompt; numerical parity remains open.

Re-run checkpoint conversion before using CFG. V2 conversion now exports
`cfg_context_embedding` using the exact one-token context CAS setting from NeMo,
the `cfg_unk_token_id`, and versioned `cfg_capability` metadata. Older artifacts
remain usable for conditional synthesis; CFG rejects them with a reconversion error.
An advertised capability with missing or malformed embedding weights fails loading.

For direct `AsyncOmni` calls, load plugins before constructing the engine so the
custom pipeline is registered when Omni resolves the deployment:

```python
from vllm.plugins import load_general_plugins
from vllm_omni import AsyncOmni

load_general_plugins()
omni = AsyncOmni(model="/path/to/reconverted_v2", deploy_config="tools/easymagpie_vllm_omni/deploy/easymagpie_cfg.yaml")
# Add these to an ordinary complete-text known-speaker prompt's additional_information:
prompt["additional_information"].update(use_cfg=True, cfg_scale=2.5, temperature=0.7, top_k=80)
```

`use_cfg` is a strict boolean. `cfg_scale` must be finite; zero and one are valid.
The serving default is 2.5, matching NeMo's high-level inference default (the
low-level streaming API defaults to 1.0). Acoustic temperature is finite, with
values at or below zero selecting actual greedy argmax. `top_k` is a positive
integer and retains all logits tied at the kth cutoff. V2 acoustic sampling settings
belong to each request and do not affect the engine's dummy continue/stop sampler.

Both branches advance their own temporal caches. The unconditional branch uses
CFG_UNK across the full task/speaker/context span, zeros during target warmup,
and shared audio inputs during decode. Phonemes come from the conditional branch.
Each predictor block combines logits before sanitization and sampling, then feeds
the one sampled code into both branches. Only one logical code stream reaches the codec.
The CFG profiles reserve 2048 positions for the codec's cumulative prompt history,
matching the Stage-0 generation budget. Keep those capacities aligned when raising
`max_tokens`; a small codec chunk size does not bound accumulated prompt history.

Run the same completion/abort/reuse check at the default acoustic settings:

```bash
VLLM_PLUGINS=easymagpie_omni python tools/easymagpie_vllm_omni/scripts/probe_cfg_pairing.py \
  --model /path/to/reconverted_v2 --speaker eng --use-cfg --cfg-scale 2.5 --temperature 0.7
```

CPU tests cover export semantics, metadata/weight validation, branch inputs,
logit guidance, cutoff/greedy sampling, shared feedback, and engine bookkeeping.
They do not establish waveform quality, numerical cache parity, or measured CFG overhead.

### Precision shared with standard serving

All CFG profiles, including the pairing diagnostic and experimental graph profile,
use the same precision settings as `deploy/easymagpie.yaml` and `deploy/easymagpie_lm.yaml`:

| Component | Deployment dtype |
| --- | --- |
| Stage-0 talker/backbone and acoustic predictor | `float16` |
| Stage-0 Mamba SSM cache | `float32` |
| Stage-1 codec (two-stage profiles) | `float32` |

Other cache dtype fields remain unspecified, matching the standard profiles and
inheriting the same runtime defaults. FP32 sampling/noise buffers and guidance
intermediate arithmetic remain intentional; model weights and embeddings follow
the configured model dtype. Integer code/index buffers and boolean masks retain
their existing types.

Earlier registered-speaker V2 CFG listening/acceptance used BF16 talker execution
to match the reference inference setup.
BF16 is not required by CFG or CUDA graphs. The profiles now use FP16 for consistency
with standard serving; those historical results do not validate the new FP16 runtime.
FP16 GPU lifecycle and frontend checks have since passed, and the LibriTTS benchmarks
below use FP16. The user accepted the selected post-fix FP16 listening samples.

### Experimental V2 CFG decode graphs

`deploy/easymagpie_cfg_graph.yaml` opts into Stage-0 `FULL_DECODE_ONLY` CUDA
capture without `torch.compile` (`compilation_config.mode: 0`). It keeps the same
FP16 talker, FP32 Mamba SSM cache, two physical talker slots, synchronous scheduling,
and eager FP32 codec as `easymagpie_cfg.yaml`. Prefill remains eager. Existing eager profiles remain the
default and provide the fallback by selecting their deployment YAML on restart.

**The zero-audio completion race is fixed and post-fix GPU revalidation passed.**
On 2026-10-30, the pinned FP16 runtime passed graph streaming completion/cancellation/reuse,
mixed CFG/conditional traffic, and actual replay validation (169 GPU graph replays across
four captured shapes). Those checks preceded the completion fix. After the fix, an
uninstrumented graph run passed all 13 HTTP/WebSocket acceptance cases and 80 alternating
CFG/conditional requests. The final text-input sentinel now leaves the codec live until
its ordered terminal connector payload is consumed; frontend completion waits for the
selected output stages to acknowledge termination. This prevents zero audio and truncated
tails when the codec lags behind the talker.

After GPU access recovered, the full serving suite passed **323 tests / 1 intentional
CI-only skip**, including CUDA replay and packed codec. Default graph lifecycle and
six-slot mixed pressure probes passed again after the fix, including acoustic tails,
waiting cancellation, and reuse. Keep the profile experimental: the benchmark comparison
below covers 100 LibriTTS texts with one registered speaker at concurrency one, and
long outputs still need stopping/repetition review.
The separate pairing-only diagnostic still requires eager execution. The CFG scheduler
also accepts the experimental compiled-backbone profile below; other graph and compilation
combinations are rejected.

To try the experimental profile after verifying GPU access, use the existing model
and [serving environment](README.md#setup-the-serving-environment):

```bash
EASYMAGPIE_DEPLOY_CONFIG=tools/easymagpie_vllm_omni/deploy/easymagpie_cfg_graph.yaml \
  bash tools/easymagpie_vllm_omni/scripts/run_server.sh /path/to/converted_v2 8091
```

Check startup logs for actual decode graph capture; a successful server launch or
`enforce_eager: false` alone is insufficient evidence of replay. Run lifecycle and
incremental-text probes, including cancellation/reuse, before relying on this profile.
For a direct incremental lifecycle probe, select it with `probe_cfg_streaming.py
--deploy-config tools/easymagpie_vllm_omni/deploy/easymagpie_cfg_graph.yaml` (plus
`--model` and `--speaker`). Mixed pressure needs a profile with sufficient physical
slots and capture sizes; the two-slot profile deliberately queues extra guided requests.

The runner stages owner/partner indices and padding masks into persistent GPU buffers
after request reordering. Captured predictor operations use fixed physical batch
shapes and copy the public sampled frame to both histories. Random draws are fresh
on replay, including draws for unused private/padded rows. Tensor-valued greedy
settings still select argmax but consume draws on this graph path; exact seeded
sample sequences and RNG advancement need not match compact eager CFG sampling.
CFG scale, temperature, and top-k remain runtime tensor inputs. Checkpoint storage dtype
is separate from deployment dtype; an existing BF16 artifact is cast to FP16 on loading.
Recasting a saved BF16 artifact cannot recover precision lost in its earlier conversion.

Compare eager and graph deployments using identical artifacts, FP16 runtime settings,
warmups, and traffic, without a profiler. Account for generated frame counts and
actual audio duration as well as latency. Report startup/capture cost and GPU memory
separately; do not infer a speedup from graph enablement alone.

### LibriTTS 100 benchmark results

Completed 2026-10-30 on the same 100 LibriTTS texts. Each of the four benchmarks
completed **100 successful measured requests with zero failures**, plus one
excluded warmup. All 200 CFG WAVs passed format, ID, nonempty, and nonzero checks.

Both modes used the same server command and byte-identical text files, the existing
`demon` model/speaker, `deploy/easymagpie_cfg_graph.yaml`, FP16 talker, FP32 Mamba
SSM cache/codec, concurrency one, 32 kHz PCM, and a maximum of 512 output tokens.
CFG requests used scale 2.5, acoustic temperature 0.7, and top-k 80. Incremental
input used five tokens per chunk with zero send delay. Script hash checks passed.
The conditional baseline used this same CFG graph pipeline with CFG disabled.
These results compare CFG and conditional synthesis on the decode-only graph
profile; they do not measure the compiled-prefill profile described below.

Throughput is generated audio seconds divided by benchmark batch wall seconds
(RTF), including batch/process-pool overhead and excluding warmup; higher is faster.

| Metric | Conditional HTTP | CFG HTTP | Conditional incremental | CFG incremental |
| --- | ---: | ---: | ---: | ---: |
| Throughput, times realtime | 8.64x | 6.64x | 8.56x | 7.10x |
| CFG throughput decrease | — | **23.15%** | — | **17.06%** |
| Mean TTFA, ms | 155.9 | 160.8 | 166.9 | 174.7 |
| p95 TTFA, ms | 194.4 | 188.3 | 193.6 | 205.0 |
| Mean latency, s | 1.14 | 0.68 | 0.94 | 0.68 |
| p95 latency, s | 3.63 | 1.57 | 1.88 | 1.10 |
| Total generated audio, s | 985.76 | 451.36 | 811.60 | 481.44 |
| Batch wall, s | 114.12 | 68.00 | 94.76 | 67.85 |
| Cumulative playback underruns | 0 | 0 | 0 | 0 |
| Outputs at least 40 s | 3 | 0 | 2 | 1 |

Percentage decreases use the printed RTF values:
`(conditional RTF − CFG RTF) / conditional RTF × 100`.
Mean time to first audio (TTFA) increased by **4.9 ms** over HTTP and **7.8 ms**
over incremental input. CFG generated 54.2% less audio over HTTP and 40.7% less
over incremental input, so shorter request latency does not establish faster
synthesis. These are single runs with stochastic outputs and independently
randomized request order; output-length differences and suspected repetition
limit a controlled performance comparison.

One CFG incremental output, `237_126133_000051_000002.wav`, lasted **40.64 s**.
The conditional baseline had five similarly long outputs. These need listening
for repetition/stopping issues; duration alone does not confirm a bound hit.
Clients did not save per-request stop metadata or elapsed times, so filtered RTF
is unavailable. All outputs remain included in throughput. No listening or
text-fidelity pass is claimed for this workload.

The local source report is `work/cfg_benchmark_report.md`; detailed results and
audio audits are under `work/cfg_graph_validation/legacy_cfg_libritts_100_02/`
and `work/cfg_graph_validation/legacy_libritts_100_02/`. These evidence files are
Git-ignored workspace artifacts.

### Experimental compiled backbone prefill

[`deploy/easymagpie_cfg_compile.yaml`](deploy/easymagpie_cfg_compile.yaml) adds
`torch.compile` for the Nemotron-H backbone (`compilation_config.mode: 3`,
`VLLM_COMPILE`) and selects `FULL_AND_PIECEWISE` CUDA graphs. This restores the
standard non-CFG profile's backbone optimization policy while retaining synchronous,
unchunked paired scheduling and separate branch caches. During prefill, the outer
EasyMagpie logic and V2 acoustic-predictor cache advancement stay eager; eligible
backbone regions use piecewise graphs, with attention and Mamba operations outside
those graph regions. Decode retains full graph capture and the same persistent
pairing buffers. The codec remains eager FP32.

The profile captures physical token counts up to 512, including both branches'
prefill spans, with padding to a configured capture size. Larger prefills use the
compiled backbone without CUDA graph replay. This capture limit is separate from
the request's output-token bound. Model precision, cache limits, and the two-slot
capacity match the decode-only graph profile.

```bash
EASYMAGPIE_DEPLOY_CONFIG=tools/easymagpie_vllm_omni/deploy/easymagpie_cfg_compile.yaml \
  bash tools/easymagpie_vllm_omni/scripts/run_server.sh /path/to/converted_v2 8091
```

Tests check the opt-in guards, real vLLM dispatch to piecewise prefill/full
decode, fallback beyond the capture range, and profile/cache contracts. On
2026-10-31, the restored runtime passed all 124 selected CFG/profile tests,
including the six previously skipped CUDA cases. The compiled profile passed all
13 HTTP/WebSocket acceptance cases and the separate streaming completion,
waiting-cancellation, and reuse probe. Completed streams produced acoustic tails
and finished codec output. A validation-only worker recorded **54 actual PIECEWISE
backbone replays** at 256 physical tokens and **78 FULL decode replays** at two
physical tokens, with paired request IDs; this instrumentation was excluded from
the benchmark timings below.

Keep this profile experimental: validation covers one registered speaker and the
two-slot profile. Listening/text fidelity and broader concurrency remain open.
Re-run acceptance and lifecycle checks with your model before relying on it.
Compilation adds startup cost; the measured TTFA improvement below does not
establish equal-output synthesis speed or quality.

### CFG comparison with 100 warmup requests

Completed 2026-10-31. This additional comparison preserves the earlier results above.
Each protocol/profile uses **100 excluded warmup requests followed by 100 measured
requests**. Both profiles passed all 400 synthesis requests
and all 13 HTTP/WebSocket acceptance cases. The compiled run completed after
GPU access recovered; the earlier blocked preflight remains preserved.

A fresh conversion uses the same source checkpoint, five seconds of
`demon_voice_01.wav`, and registered speaker `demon`, saved under
`work/demon_cfg_reconverted_20261031/model`. Its weights, codec, and tokenizer
match the historical artifact byte for byte; the speaker tensor is also identical.
Storage remains BF16, with FP16 talker execution and FP32 SSM cache/codec.
Both profiles use the same 100-text TSV and unchanged CFG benchmark scripts,
CFG 2.5 / temperature 0.7 / top-k 80, concurrency one, 512 output tokens,
and 32 kHz PCM. Incremental input uses five tokens per chunk and zero send delay.
All columns below use CFG. RTF is audio seconds / batch wall seconds, including
process-pool overhead and excluding warmup; higher is faster.

| Metric | Decode-only HTTP | Compiled HTTP | Decode-only incremental | Compiled incremental |
| --- | ---: | ---: | ---: | ---: |
| Warmup successful / failed | 100 / 0 | 100 / 0 | 100 / 0 | 100 / 0 |
| Measured successful / failed | 100 / 0 | 100 / 0 | 100 / 0 | 100 / 0 |
| Throughput, times realtime | 7.70x | 7.75x | 7.14x | 7.26x |
| Mean TTFA, ms | 161.1 | 149.8 | 176.4 | 165.7 |
| p95 TTFA, ms | 196.3 | 179.6 | 209.0 | 195.3 |
| Mean latency, s | 0.65 | 0.81 | 0.72 | 0.84 |
| p95 latency, s | 1.19 | 1.50 | 1.23 | 1.50 |
| Total generated audio, s | 504.08 | 633.28 | 512.88 | 612.64 |
| Batch wall, s | 65.48 | 81.68 | 71.87 | 84.39 |
| Cumulative playback underruns | 0 | 0 | 0 | 0 |
| Outputs at least 40 s | 0 | 0 | 1 | 0 |

Compiled mean TTFA fell by 11.3 ms (7.0%) over HTTP and 10.7 ms (6.1%) over
incremental input. Recomputed throughput increased by 0.7% and 1.7%, respectively.
Generated audio increased by 25.6% and 19.5%, and mean request latency increased;
these single stochastic runs do not establish equal-output synthesis speed or quality.

Decode-only service startup to health took 184.16 s; graph capture reported
4 s / 0.07 GiB. Compiled startup took 235.17 s, including 12.78 s of backbone
compilation; graph capture reported 9 s / 0.17 GiB. GPU 0 used 26,219 / 26,221 MiB
at readiness, with sampled startup peaks of 47,584.56 / 47,586.56 MiB, respectively.
Observed compiled inference settled near 26,907 MiB (26.28 GiB); its monitoring
covered the full run without errors. These allocations are not minimum hardware
requirements. The decode-only run lost NVML monitoring late in inference, after
which the first compiled preflight failed. The resumed compiled run reported no OOM.

All 200 measured and 200 warmup WAVs per profile passed format, ID, nonempty,
and nonzero checks. The longest compiled measured outputs were 13.04 s (HTTP)
and 13.12 s (incremental). One decode-only measured incremental output,
`908_157963_000010_000001.wav`, lasted 40.64 s and needs listening for
repetition/stopping; duration does not establish a bound hit. It remains included
in RTF. Output sampling and request order are stochastic, and the earlier table
used only one warmup per protocol, so differences from historical numbers do not
isolate a warmup benefit or establish audio quality. Listening/text-fidelity
acceptance remains pending for the compiled outputs.

Report: `work/cfg_compiled_benchmark_report.md`. Logs, commands, WAVs, configuration,
memory samples, and audits are in
`work/cfg_graph_validation/fresh_decode_cfg_libritts_100_warm100_01/` and
`work/cfg_graph_validation/fresh_compiled_cfg_libritts_100_warm100_02/`.
The earlier blocked compiled preflight remains in the `_01` directory.
These are Git-ignored workspace artifacts.

### CFG benchmark scripts

Use `scripts/benchmark_server_cfg.py` for complete-text HTTP requests and
`scripts/benchmark_incremental_server_cfg.py` for token-chunked WebSocket requests.
These variants enable CFG by default and send acoustic settings in `extra_params`;
the original benchmark scripts retain their conditional-only defaults.

Both variants default to guidance scale **2.5**, acoustic temperature **0.7**,
top-k **80**, concurrency **1**, maximum **512** output tokens, and **32 kHz** PCM.
Incremental input defaults to five tokens per chunk with no send delay. Each script
prints the effective acoustic settings and uses them for warmup and measured
requests. Throughput remains generated audio seconds divided by benchmark batch
wall seconds (higher is faster), including process-pool overhead.

Start a server with a CFG deployment as described below, then run:

```bash
python tools/easymagpie_vllm_omni/scripts/benchmark_server_cfg.py \
  --text-file texts.tsv -n 100 --speaker-id demon --output-dir cfg_http_audio

PYTHONPATH=tools/easymagpie_vllm_omni \
python tools/easymagpie_vllm_omni/scripts/benchmark_incremental_server_cfg.py \
  --model /path/to/reconverted_v2 --text-file texts.tsv -n 100 \
  --speaker-id demon --tokens-per-chunk 5 --send-delay-ms 0 \
  --output-dir cfg_incremental_audio
```

Choose a speaker registered in your converted model (`demon` is an example).
Override sampling with `--cfg-scale`, `--temperature`, and `--top-k`; use
`--no-use-cfg` for conditional inference with the same sampling settings.
Nonpositive temperature selects greedy decoding. Guided requests require a CFG
pipeline and registered speaker; HTTP `--reference-audio` requires `--no-use-cfg`.
The single-request graph/eager CFG profiles support the default concurrency one;
use the concurrent deployment and its documented capacity for higher concurrency.
The 512-token bound includes stream warmup, EOS, and acoustic tail, so inspect long
outputs for repetition or truncation before interpreting throughput as quality.
These clients do not persist per-request frame counts or stop reasons.

### HTTP and WebSocket CFG selection

Launch the server with a CFG deployment and a reconverted V2 checkpoint containing
registered speaker embeddings:

```bash
EASYMAGPIE_DEPLOY_CONFIG=tools/easymagpie_vllm_omni/deploy/easymagpie_cfg.yaml \
  bash tools/easymagpie_vllm_omni/scripts/run_server.sh /path/to/reconverted_v2 8091
```

For complete text, send acoustic settings in `extra_params`:

```bash
curl http://localhost:8091/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"Hello from the speech server.","voice":"eng","response_format":"wav",
       "extra_params":{"use_cfg":true,"cfg_scale":2.5,"temperature":0.7,"top_k":80}}' \
  --output speech.wav
```

For incremental text, connect to `ws://localhost:8091/v1/audio/speech/stream`
and send these JSON messages in order. Receive binary PCM audio concurrently
with sending text; completion is reported by `audio.done` and `session.done`.

```json
{"type":"session.config","voice":"eng","stream_audio":true,"response_format":"pcm","max_new_tokens":512,"extra_params":{"use_cfg":true,"cfg_scale":2.5,"temperature":0.7,"top_k":80}}
{"type":"input.text","text":"Hello from "}
{"type":"input.text","text":"the speech server."}
{"type":"input.done"}
```

CFG is opt-in: omitting `use_cfg` or setting it to `false` selects conditional-only
inference. Guidance defaults to 2.5, acoustic temperature to 0.7, and top-k to 80.
Settings use the same strict validation as direct backend requests: `use_cfg` must
be a boolean, scale and temperature must be finite numbers, and top-k must be a
positive integer. Nonpositive temperature selects greedy acoustic decoding; scales
zero and one are supported. These are acoustic settings, separate from the engine's
dummy-token sampling settings.

Guided requests require the `easymagpie_cfg` pipeline (either CFG deployment profile),
valid V2 capability metadata, and a registered speaker. Unsupported deployments,
older artifacts, and raw-reference CFG requests are rejected before generation.
Incremental sessions require registered speakers for conditional-only inference too.
On the CFG pipeline, `max_new_tokens` bounds the whole stream, including text warmup,
EOS, and acoustic tail, for guided and conditional requests. The bound must fit the
deployment's cache and codec capacities. Settings are fixed for the session.

Live GPU acceptance passed complete-text HTTP and incremental WebSocket synthesis
for guided and conditional requests, invalid-setting and unsupported-profile rejection,
and disconnect after audio delivery followed by successful reuse. These smoke checks
do not establish speech quality, numerical parity, throughput, or long-run leak freedom.

### Incremental text with paired V2 CFG

Use the direct `AsyncOmni.generate()` streaming-input API with the explicit
`easymagpie_cfg` pipeline and a registered speaker. Speaker embeddings must be
prepared offline with the served checkpoint. The pairing-only diagnostic still
rejects streaming. HTTP/WebSocket clients can select CFG through `extra_params` as described above.

[`EasyMagpieInputStream`](easymagpie_vllm_omni/serving_stream.py) builds the input
iterator, buffers the initial text warmup, assigns absolute `text_token_start`
offsets, appends text EOS, and reserves the remaining budget for the acoustic tail.
Pass `use_cfg=True`, `cfg_scale=2.5`, `temperature=0.7`, `top_k=80`, and `speaker_id`
in its prefill prompt's `additional_information`. Size `prompt_token_ids` for the
registered speaker context **plus** `arch.text_prefill_num`, and pass that same
`text_prefill_num` to the input stream. Feed tokenizer IDs through `put_tokens()`
and call `finish()` when text is complete. For every engine output call
`stream.observe_output(output)` so the next text segment can resume. Run the input
producer and output consumer concurrently; abort the request on disconnect or error.

The stream's `max_new_tokens` becomes the immutable `max_stream_tokens` request
bound. It covers the sum of all segment generation limits, including warmup, text
EOS, and tail; it is not a per-chunk limit. The initial prompt plus that bound must
fit Stage 0's `max_model_len`, and the codec history must cover the bound (2048 in
the supplied profiles). Text that leaves no room for EOS/tail is rejected. Use
the checkpoint's phoneme-BOS warmup; a too-short initial text input cannot start
generation. Requests keep their original CFG mode, scale, speaker, and acoustic
sampling settings throughout the stream.

Both physical histories pause and resume together, using separate engine-managed
cache blocks and one shared acoustic sample. A suspended stream retains its full
reservation and resumes before fresh admissions that may be waiting for its
blocks. Exact duplicate text chunks are ignored; gaps, conflicts, partial overlaps,
and text after the acoustic tail are rejected. Only the public branch reaches the
codec and frontend. Text chunks belong to one synthesis request; retained dialogue
turns and request-time audio conditioning are outside this contract.

Run a delayed-chunk completion, cancellation-while-waiting, and reuse probe:

```bash
VLLM_PLUGINS=easymagpie_omni python tools/easymagpie_vllm_omni/scripts/probe_cfg_streaming.py \
  --model /path/to/reconverted_v2 --speaker eng --output /tmp/cfg-streaming.json
```

Add `--mixed --deploy-config tools/easymagpie_vllm_omni/deploy/easymagpie_cfg_concurrent.yaml`
to test streaming CFG/conditional requests alongside complete-text traffic under
sequence pressure. The probe uses CFG 2.5 / temperature 0.7 / top-k 80 and eager,
synchronous scheduling. Successful completion checks output routing and lifecycle;
it does not establish waveform/text quality, numerical parity, or throughput.

### Experimental mixed CFG/conditional admission

[`deploy/easymagpie_cfg_concurrent.yaml`](deploy/easymagpie_cfg_concurrent.yaml)
selects the same `easymagpie_cfg` pipeline with six physical Stage-0 slots and four
public Stage-1 slots. CFG consumes two physical slots, conditional-only consumes
one, and private branches never transfer codec payloads. Set `use_cfg=False` for
conditional-only requests in this pipeline, including HTTP/WebSocket requests.

`EasyMagpieCFGScheduler` admits logical units in FCFS order. Before admission it
reserves engine-managed cache for the entire bounded prompt plus generation
history of every branch. A pair waits together if sequence slots, the per-step
token budget, or cache capacity are unavailable. A unit that cannot fit the
total cache at all is rejected. Both branches run at aligned positions, without
preemption or Mamba replay. Request-wide reservation trades memory utilization
for this scheduling guarantee; lowering the output bound can increase admission
capacity. No throughput or memory-efficiency claims have been measured.

The mixed scheduler requires FCFS, Mamba cache mode `none`, and a token budget
covering every active physical decode sequence. Keep eager synchronous execution
and disable prefix caching/chunked prefill. CPU tests use the real hybrid cache
allocator to check pair isolation, independent stopping, queueing under pressure,
abort/late outputs, failure cleanup, and slot reuse. They also check multiple
guided pairs mixed with a conditional predictor row and private connector filtering.

The GPU probe submits six logical requests needing nine physical slots against
the six-slot profile, cancels one guided request, then checks another mixed wave
using the same engine. It requires public request IDs, no reported engine errors,
and completed codec output for every non-cancelled request:

```bash
VLLM_PLUGINS=easymagpie_omni python tools/easymagpie_vllm_omni/scripts/probe_cfg_concurrency.py \
  --model /path/to/reconverted_v2 --speaker eng --cfg-scale 2.5 --temperature 0.7 --top-k 80 \
  --output mixed_probe_results.json
```

GPU execution of this mixed profile remains pending because CUDA/NVML became
unavailable again. Do not interpret the earlier single-request lifecycle pass
as validation of this extension or of exact waveform parity across batch sizes.

