## EasyMagpieTTS — V2 classifier-free guidance (CFG)

For checkpoint conversion, serving-environment setup, and standard synthesis,
see the [main README](README.md). This guide covers V2 CFG synthesis, cache
pairing, the compiled deployment profile, CUDA graphs, benchmarks, and validation.

### Shared default sampling and per-request fallback

V2 batches using acoustic temperature **0.7**, top-k **80**, and CFG scale
**2.5** (when CFG is enabled) select exactly
`min(80, vocabulary_size)` logits and draw one FP32 Gumbel tensor for the entire
frame before iterating through codebooks. Temperature division happens in the
selected logits' dtype. CFG uses the shared scalar 2.5 before sampling;
conditional-only rows bypass guidance. Both branches still consume the public
branch's sampled feedback.

The runner validates settings on the CPU during each request's preprocessing.
If any request in a batch uses another temperature, top-k, or enabled CFG scale,
the **entire batch** uses the existing per-request sampler and bypasses CUDA
graph replay for that forward. This includes greedy requests. The next batch
can return to the captured default path. Custom values remain supported, with
an eager-execution performance cost. Changing `cfg_scale` while CFG is disabled
does not affect dispatch or apply guidance. The local-transformer V1 path is
unchanged.

The default path selects exactly k candidates; the per-request sampler includes
all logits tied at the cutoff. The paths also differ in sampling arithmetic and
RNG layout. Eager CFG samples logical prediction rows, while graph execution
samples a fixed physical shape, including private/padded rows, then shares owner
results and masks padding. Seeded audio can differ across samplers, execution
modes, and batch shapes.

### V2 CFG synthesis

[`deploy/easymagpie_cfg.yaml`](deploy/easymagpie_cfg.yaml) adds guided synthesis
for known-speaker requests with complete text or bounded incremental text. This is the single
CFG deployment profile, using a compiled backbone and full decode CUDA graphs. It matches
the standard `easymagpie.yaml` profile's configured capacity for 32 logical CFG requests.
Stage 0 has 64 physical slots because each guided request owns two histories; Stage 1 has
32 slots because only the public branch produces codec input. Guided and
conditional-only requests use the same explicit pipeline, with `use_cfg` selecting
whether the scheduler creates a private branch. Each guided request has separate backbone
attention, Mamba, and temporal acoustic-predictor cache state for its two branches.
Completion and cancellation free both histories.

The batching limits account for both physical CFG branches:

| Setting | Standard `easymagpie.yaml` | CFG synthesis profile |
| --- | --- | --- |
| Stage-0 `max_num_seqs` | 32 | 64 (32 guided logical requests) |
| Stage-0 `max_num_batched_tokens` | 4096 | 8192 (both branch spans) |
| Stage-0 `max_model_len` | 4096 | 4096 per physical history |
| Stage-1 `max_num_seqs` | 32 | 32 public streams |
| Stage-1 `max_num_batched_tokens` / `max_model_len` | 512 / 512 | 2048 / 2048 |
| Stage-0 async scheduling / chunked prefill | enabled / enabled | disabled / disabled |

The codec retains cumulative prompt history across chunks, so its 2048-position capacity must
cover the Stage-0 output bound. Unchunked scheduling also requires its per-step token budget to
be at least that history limit; copying the standard codec's 512-token budget would reject this
configuration. The synchronous, unchunked CFG restriction remains in force. Thus sequence capacity
and the Stage-0 logical token budget match the baseline, while those execution/history requirements
remain explicit differences. Cache availability and the request bound can reduce achieved admission.

Raw reference/user audio, retained turns, preemption, asynchronous scheduling,
chunked prefill, and prefix caching remain unsupported in this CFG path.
Scale 1 numerical parity with conditional-only greedy inference is not established.

CFG requires a V2 checkpoint converted with
`cfg_context_embedding`, `cfg_unk_token_id`, and versioned `cfg_capability` metadata.
The embedding uses NeMo's one-token context CAS setting. Checkpoints without this
metadata support conditional synthesis but require reconversion for CFG. Missing
or malformed embedding weights fail loading when CFG capability is advertised.

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
integer. The default sampler selects exactly k logits; the per-request fallback retains
all logits tied at the kth cutoff. V2 acoustic sampling settings
belong to each request and do not affect the engine's dummy continue/stop sampler.

Both branches advance their own temporal caches. The unconditional branch uses
CFG_UNK across the full task/speaker/context span, zeros during target warmup,
and shared audio inputs during decode. Phonemes come from the conditional branch.
Each predictor block combines logits before sanitization and sampling, then feeds
the one sampled code into both branches. Only one logical code stream reaches the codec.
The CFG profile reserves 2048 positions for the codec's cumulative prompt history,
matching the Stage-0 generation budget. Keep those capacities aligned when raising
`max_tokens`; a small codec chunk size does not bound accumulated prompt history.

CPU tests cover export semantics, metadata/weight validation, branch inputs,
logit guidance, cutoff/greedy sampling, shared feedback, and engine bookkeeping.
They do not establish waveform quality, numerical cache parity, or measured CFG overhead.

### Precision shared with standard serving

The CFG profile uses the same precision settings as `deploy/easymagpie.yaml` and `deploy/easymagpie_lm.yaml`:

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

Checkpoint storage dtype is independent of serving dtype. A BF16 export is cast
to FP16 when loaded with the supplied profile. BF16 is not required for CFG or
CUDA graphs. The benchmark results below use FP16 talker execution.

### Compiled backbone prefill and decode graphs

[`deploy/easymagpie_cfg.yaml`](deploy/easymagpie_cfg.yaml) enables
`torch.compile` for the Nemotron-H backbone (`compilation_config.mode: 3`,
`VLLM_COMPILE`) and selects `FULL_AND_PIECEWISE` CUDA graphs. It uses the standard
non-CFG profile's backbone optimization policy with synchronous, unchunked paired
scheduling and separate branch caches. During prefill, the outer
EasyMagpie logic and V2 acoustic-predictor cache advancement stay eager; eligible
backbone regions use piecewise graphs, with attention and Mamba operations outside
those graph regions. Decode retains full graph capture and the same persistent
pairing buffers. The codec remains eager FP32.

Fully eager Stage-0 execution remains supported by setting `enforce_eager: true`
and disabling backbone compilation (`compilation_config.mode: 0`) and CUDA graphs
(`compilation_config.cudagraph_mode: NONE`) in a deployment configuration. Decode-only
graph execution also remains supported with mode 0 and `FULL_DECODE_ONLY` when
`enforce_eager: false`. These execution options use the same CFG scheduler.
Custom sampling settings retain the automatic eager fallback described above.

The profile captures physical token counts up to 512, including both branches'
prefill spans, with padding to a configured capture size. Larger prefills use the
compiled backbone without CUDA graph replay. This capture limit is separate from
the request's output-token bound. Full decode capture covers physical batch sizes
through 64 rows, including both branches of all 32 guided logical requests.

```bash
EASYMAGPIE_DEPLOY_CONFIG=tools/easymagpie_vllm_omni/deploy/easymagpie_cfg.yaml \
  bash tools/easymagpie_vllm_omni/scripts/run_server.sh /path/to/converted_v2 8091
```

The runner stages owner/partner indices and padding masks into persistent GPU buffers
after request reordering. Captured predictor operations use fixed physical batch
shapes and copy the public sampled frame to both histories. Random draws are fresh
on replay, including draws for unused private/padded rows. Exact seeded sample
sequences and RNG advancement need not match compact eager CFG sampling.
Custom-settings batches retain tensor-valued greedy argmax semantics in the eager
fallback; the tensor sampler may consume unused random draws. CFG scale,
temperature, and top-k remain runtime tensor inputs for those batches.
Default-settings graphs capture the shared constants described above.

Check startup logs for actual graph capture; a successful server launch alone is
insufficient evidence of replay. Report compilation/capture startup costs separately
from warmed-up request timings.

Tests cover configuration guards, dispatch to piecewise prefill/full decode,
fallback beyond the capture range, and cache/graph contracts. CPU tests do not
establish actual GPU replay, waveform quality, or performance.

### Benchmarks

The benchmark covers 10 configurations across 11 runs, including a legacy HTTP c32
repeat, with **1,100/1,100 successful measured requests**. Legacy incremental c1/c32
are excluded because of a prefill-boundary engine bug. An incomplete CFG HTTP c32
attempt stopped before reuse and is excluded from the tables.

#### Setup and measurement

- Legacy source: `fa1009753cc353702ae0ac701b2aab10d7d4136a`;
  current non-CFG and CFG: `2c69a87f2dbc083af91bc39df2a7ad75846ea11b`.
  Results apply to these revisions; subsequent source changes have not been
  benchmarked here.
- GPU0: NVIDIA RTX 5880 Ada Generation, 49,140 MiB. Python 3.12,
  torch 2.11.0+cu130, vLLM/vLLM-Omni 0.24.0. BF16 exports, FP16 talker,
  FP32 SSM cache, eager FP32 codec.
- Same 100 texts/IDs and registered speaker `test_voice_01`, prepared from the
  first five seconds of `test_voice_01.wav`. Temperature 0.7, top-k 80, CFG scale 2.5,
  max 512 new tokens, client ordering seed 0, mono PCM16 at 32 kHz.
- HTTP supplies complete text. WS supplies five text tokens per chunk with zero
  intentional delay; it does not simulate paced human input.
- Each invocation uses fresh isolated compiler caches and a startup compilation
  gate, then a new case server, 2 c1 smoke requests, 100 excluded warmups at the
  selected concurrency, 100 measured requests, and 2 c1 reuse requests.
  Startup, compilation, smoke, warmup, and reuse are excluded from measured wall time.
- Non-CFG uses the standard asynchronous/chunked profile; CFG uses synchronous,
  unchunked scheduling. Slot counts and token/cache budgets differ as shown above.
  This compares complete deployments, not guidance alone or conditional requests
  within the CFG deployment.

#### Throughput and latency

Each row contains 100/100 successful measured requests. **Audio/wall** is generated
audio seconds divided by client wall seconds (higher is better); conventional
**RTF** is wall/audio (lower is better). The client's field named `rtf` actually
reports audio/wall. TTFA includes queueing, latency ends at stream completion,
and ITL is the interval between received audio chunks. Values are mean/p95.

| Configuration | Audio s | Wall s | Requests/s | Audio/wall | RTF | TTFA mean/p95 ms | Latency mean/p95 s | ITL mean/p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Legacy HTTP c1 | 788.56 | 80.161 | 1.2475 | 9.8372× | 0.10165 | 155.9/191.1 | 0.799/1.483 | 52.0/68.0 |
| Legacy HTTP c32 | 745.60 | 17.200 | 5.8139 | 43.3484× | 0.02307 | 639.1/917.1 | 4.944/9.553 | 371.9/581.9 |
| Legacy HTTP c32 repeat | 766.40 | 18.087 | 5.5288 | 42.3724× | 0.02360 | 851.3/1259.8 | 5.124/10.536 | 357.4/543.5 |
| Current HTTP c1 | 788.72 | 80.963 | 1.2351 | 9.7417× | 0.10265 | 156.5/190.5 | 0.807/1.598 | 52.8/69.0 |
| Current HTTP c32 | 774.24 | 18.380 | 5.4408 | 42.1251× | 0.02374 | 628.0/848.5 | 5.311/10.101 | 388.9/609.3 |
| Current WS c1 | 755.28 | 79.493 | 1.2580 | 9.5013× | 0.10525 | 166.5/198.0 | 0.792/1.505 | 53.0/72.0 |
| Current WS c32 | 750.40 | 18.215 | 5.4900 | 41.1973× | 0.02427 | 895.7/1648.6 | 5.184/8.162 | 365.5/614.2 |
| CFG HTTP c1 | 637.76 | 74.428 | 1.3436 | 8.5689× | 0.11670 | 142.7/167.3 | 0.742/1.336 | 60.4/78.7 |
| CFG HTTP c32 | 627.12 | 19.223 | 5.2020 | 32.6228× | 0.03065 | 953.3/1372.9 | 5.576/11.062 | 475.4/716.4 |
| CFG WS c1 | 602.24 | 75.415 | 1.3260 | 7.9857× | 0.12522 | 154.6/190.6 | 0.752/1.376 | 63.5/81.4 |
| CFG WS c32 | 614.32 | 19.156 | 5.2203 | 32.0696× | 0.03118 | 847.3/1195.5 | 5.482/9.363 | 473.7/717.8 |

Current non-CFG HTTP audio throughput is 0.97% below legacy at c1, and 2.82% below
the first legacy c32 run or 0.58% below its repeat. Both legacy c32 observations
are shown separately; the repeat includes compiler activity. The observed differences
are descriptive, without confidence intervals or a controlled regression estimate.

Relative to current non-CFG, CFG audio throughput is 12.04%/15.95% lower at c1
HTTP/WS and 22.56%/22.16% lower at c32 HTTP/WS. CFG generates 18–20% less total
audio, which contributes to its higher requests/s at c1. Since the generated outputs
differ, these ratios do not measure equal-output inference speed or speech quality.

#### Playback and measured compilation

An underrun estimates playback-buffer exhaustion when playback starts on first
audio. A deadline miss is a gap exceeding the preceding chunk's audio duration;
previously buffered audio can cover it. These are playback estimates, not failed
synthesis requests. Affected requests count streams with at least one underrun.

| Configuration | Underruns | Requests affected / 100 | Deadline misses | Measured compiler/cache activity |
| --- | ---: | ---: | ---: | --- |
| Legacy HTTP c1 | 0 | 0 | 0 | None observed |
| Legacy HTTP c32 | 15 | 11 | 59 | None observed |
| Legacy HTTP c32 repeat | 3 | 3 | 21 | 16 files; 2 fused-MoE specializations |
| Current HTTP c1 | 0 | 0 | 0 | None observed |
| Current HTTP c32 | 27 | 20 | 74 | None observed |
| Current WS c1 | 0 | 0 | 0 | None observed |
| Current WS c32 | 17 | 14 | 84 | 160 files; 18 gather + 2 fused-MoE specializations |
| CFG HTTP c1 | 0 | 0 | 0 | None observed |
| CFG HTTP c32 | 114 | 58 | 260 | 16 files; 2 fused-MoE specializations |
| CFG WS c1 | 0 | 0 | 0 | None observed |
| CFG WS c32 | 94 | 76 | 166 | 104 files; 13 gather specializations |

All c1 rows have zero playback gaps. Every c32 row has underruns; CFG affects
58/100 HTTP and 76/100 WS requests. Client concurrency must account for playback
latency as well as aggregate throughput; the maximum gap-free concurrency was not measured.

Cache snapshots show additional compiled specializations in four runs after the
fixed warmup. Their timings include compilation overhead and should not be treated
as steady-state measurements. The other seven runs show no measured cache activity.
No run logged a measured JIT warning or detected compiler processes at phase
boundaries, so logs and boundary checks alone are insufficient to detect compilation.
The compilation cost cannot be separated from the recorded timings.

#### Startup, memory, and validation limits

CFG fresh-cache startup to health took 279.32–281.57 s; measurement-server startup
using those caches took 87.81–89.83 s. These are separate startup wall times,
not pure compiler timings or part of request throughput.

GPU memory was sampled every 15 seconds across whole server lifetimes. CFG case
maxima were 26,549–30,631 MiB; earlier CFG compilation gates reached 47,049 MiB.
These are sampled observations, not instantaneous peaks or minimum requirements.
The lower CFG case memory is confounded by codec cache budgets: a recorded
non-CFG/CFG pair used 12.40/1.03 GiB. It does not establish lower intrinsic CFG memory.

Validation recomputed metrics and checked request IDs, PCM format, frame counts,
and chunk durations for all 2,244 WAVs across smoke, warmup, measured, and reuse
phases. Each included run passed readiness, integrity, postflight, and cleanup
checks. Two 40.80-second c1 outputs remain in the totals without filtering.
Speech intelligibility, speaker similarity, and stopping quality were not evaluated.

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
  --text-file texts.tsv -n 100 --speaker-id test_voice_01 --output-dir cfg_http_audio

PYTHONPATH=tools/easymagpie_vllm_omni \
python tools/easymagpie_vllm_omni/scripts/benchmark_incremental_server_cfg.py \
  --model /path/to/reconverted_v2 --text-file texts.tsv -n 100 \
  --speaker-id test_voice_01 --tokens-per-chunk 5 --send-delay-ms 0 \
  --output-dir cfg_incremental_audio
```

Choose a speaker registered in your converted model (`test_voice_01` in these examples).
Override sampling with `--cfg-scale`, `--temperature`, and `--top-k`; use
`--no-use-cfg` for conditional inference with the same sampling settings.
Nonpositive temperature selects greedy decoding. Guided requests require a CFG
pipeline and registered speaker; HTTP `--reference-audio` requires `--no-use-cfg`.
Use `--concurrency` to change client load. The same CFG deployment supports up to
32 active guided logical requests, subject to cache availability; excess load queues.
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
  -d '{"input":"Hello from the speech server.","voice":"test_voice_01","response_format":"wav",
       "extra_params":{"use_cfg":true,"cfg_scale":2.5,"temperature":0.7,"top_k":80}}' \
  --output speech.wav
```

For incremental text, connect to `ws://localhost:8091/v1/audio/speech/stream`
and send these JSON messages in order. Receive binary PCM audio concurrently
with sending text; completion is reported by `audio.done` and `session.done`.

```json
{"type":"session.config","voice":"test_voice_01","stream_audio":true,"response_format":"pcm","max_new_tokens":512,"extra_params":{"use_cfg":true,"cfg_scale":2.5,"temperature":0.7,"top_k":80}}
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

Guided requests require the `easymagpie_cfg` pipeline,
valid V2 capability metadata, and a registered speaker. Unsupported deployments,
older artifacts, and raw-reference CFG requests are rejected before generation.
Incremental sessions require registered speakers for conditional-only inference too.
On the CFG pipeline, `max_new_tokens` bounds the whole stream, including text warmup,
EOS, and acoustic tail, for guided and conditional requests. The bound must fit the
deployment's cache and codec capacities. Settings are fixed for the session.

### Incremental text with paired V2 CFG

Use the direct `AsyncOmni.generate()` streaming-input API with the explicit
`easymagpie_cfg` pipeline and a registered speaker. Speaker embeddings must be
prepared offline with the served checkpoint. HTTP/WebSocket clients can select CFG
through `extra_params` as described above.

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
the supplied profile). Text that leaves no room for EOS/tail is rejected. Use
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

### Mixed CFG/conditional admission

The same `easymagpie_cfg.yaml` deployment supports mixed CFG/conditional traffic
with 64 physical Stage-0 slots and 32 public Stage-1 slots. CFG consumes two physical
slots, conditional-only consumes one, and private branches never transfer codec
payloads. Set `use_cfg=False` for conditional-only direct requests or `use_cfg: false`
in HTTP/WebSocket `extra_params`.

The native codec scheduler bounds active streams by Stage 1's `max_num_seqs`
(or a smaller configured `active_stream_window`). Excess streams wait while
retaining their cached codec state; they must not re-enter as new requests and
replay cumulative frame placeholders against the latest chunk. Downstream
prewarm requests remain empty until their first acoustic payload arrives.
Client concurrency above the active capacity therefore includes queue wait.
The incremental segment watchdog allows 67 seconds, including queue wait;
client deadlines may expire sooner. An empty final input sentinel arriving after
stream cleanup is ignored, including its engine-injected `global_request_id` and
`omni_final_stage_id` routing metadata, so timeout cleanup cannot become a new synthesis request.
When its pending text queue fills, incremental input coalesces small token chunks
so WebSocket reads and heartbeat replies continue during engine queue wait.
The buffer retains capacity for eight 256-token chunks; excess pending input is
rejected with a generation error instead of blocking the receive loop.
`input.done` also completes without waiting for queue space. If the token queue is full,
completion is recorded separately and recognized after pending tokens drain; otherwise
a completion marker wakes a waiting consumer. This avoids hanging error/abort cleanup
when the Stage-0 consumer times out or exits while input is still queued.
Connector completion can arrive before the final downstream prewarm update. A
completed incremental stream ignores those late updates until its input iterator
closes, so a terminal prewarm cannot reopen it and block subsequent requests.
Cancellation also clears this pending terminal tracking.

`EasyMagpieCFGScheduler` admits logical units in FCFS order. Before admission it
reserves engine-managed cache for the entire bounded prompt plus generation
history of every branch. A pair waits together if sequence slots, the per-step
token budget, or cache capacity are unavailable. A unit that cannot fit the
total cache at all is rejected. Both branches run at aligned positions, without
preemption or Mamba replay. Request-wide reservation trades memory utilization
for this scheduling guarantee; lowering the output bound can increase admission
capacity. The benchmarks do not isolate the throughput or memory effect of this
reservation policy.

The mixed scheduler requires FCFS, Mamba cache mode `none`, and a token budget
covering every active physical decode sequence. Scheduling remains synchronous
with prefix caching and chunked prefill disabled, including when CUDA graphs are enabled.
CPU tests use the real hybrid cache allocator to check pair isolation, independent stopping, queueing under pressure,
abort/late outputs, failure cleanup, and slot reuse. They also check multiple
guided pairs mixed with a conditional predictor row and private connector filtering.

The benchmark measures guided traffic rather than saturated mixed CFG/conditional
traffic. GPU cancellation and lifecycle behavior under saturated mixed load, and
waveform parity across batch sizes, require separate validation.
