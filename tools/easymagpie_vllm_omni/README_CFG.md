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
for known-speaker requests with complete text or bounded incremental text. This is the
synchronous CFG deployment profile, using a compiled backbone and full decode
CUDA graphs. It matches the standard `easymagpie.yaml` profile's configured capacity for
32 logical CFG requests. The recommended asynchronous CFG profile is described below; this
synchronous profile remains available as an alternative during the move to asynchronous scheduling.
Stage 0 has 64 physical slots because each guided request owns two histories; Stage 1 has
32 slots because only the public branch produces codec input. Guided and
conditional-only requests use the same explicit pipeline, with `use_cfg` selecting
whether the scheduler creates a private branch. Each guided request has separate backbone
attention, Mamba, and temporal acoustic-predictor cache state for its two branches.
Completion and cancellation free both histories.

The batching limits account for both physical CFG branches:

| Setting | Standard `easymagpie.yaml` | CFG synthesis profile | Async CFG profile |
| --- | --- | --- | --- |
| Stage-0 `max_num_seqs` | 32 | 64 (32 guided logical requests) | 64 |
| Stage-0 `max_num_batched_tokens` | 4096 | 8192 (both branch spans) | 8192 |
| Stage-0 `max_model_len` | 4096 | 4096 per physical history | 4096 |
| Stage-1 `max_num_seqs` | 32 | 32 public streams | 32 |
| Stage-1 `max_num_batched_tokens` / `max_model_len` | 512 / 512 | 2048 / 2048 | 2048 / 2048 |
| Stage-0 async scheduling / chunked prefill | enabled / enabled | disabled / disabled | enabled / disabled |

The codec retains cumulative prompt history across chunks, so its 2048-position capacity must
cover the Stage-0 output bound. Unchunked scheduling also requires its per-step token budget to
be at least that history limit; copying the standard codec's 512-token budget would reject this
configuration. The unchunked CFG restriction remains in force. Thus sequence capacity
and the Stage-0 logical token budget match the baseline, while those execution/history requirements
remain explicit differences. Cache availability and the request bound can reduce achieved admission.

Asynchronous scheduling is provided by the recommended CFG profile
[`deploy/easymagpie_cfg_async.yaml`](deploy/easymagpie_cfg_async.yaml) (compiled Stage 0, eager codec).
Selecting it enables both complete-text (HTTP) and incremental (WebSocket) input; no other switch is needed.
For an eager Stage 0, set `enforce_eager: true` and `compilation_config: {mode: 0, cudagraph_mode: NONE}`
on stage 0 of a copy of this profile.

Raw reference/user audio, retained turns, preemption, chunked prefill, and prefix caching
remain unsupported in this CFG path.
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
omni = AsyncOmni(
    model="/path/to/reconverted_v2", deploy_config="tools/easymagpie_vllm_omni/deploy/easymagpie_cfg_async.yaml"
)
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

### Synchronous CFG benchmarks

The benchmark covers 10 configurations across 11 runs, including a legacy HTTP c32
repeat, with **1,100/1,100 successful measured requests**. Legacy incremental c1/c32
are excluded because of a prefill-boundary engine bug. An incomplete CFG HTTP c32
attempt stopped before reuse and is excluded from the tables.

#### Setup and measurement

- Legacy source: `028cc3807ff82dd12083055533c5870031a5f621`;
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

- **Underrun:** estimates playback-buffer exhaustion when playback starts on first audio.
- **Deadline miss:** a gap exceeding the preceding chunk's audio duration; previously
  buffered audio can cover it.
- **Requests affected:** streams with at least one underrun.
- **Measured compiler/cache activity:** files added or changed in the run's isolated compiler
  caches between snapshots taken before and after the measured phase. Specializations count
  new Triton kernel variants by kernel name (for example, fused-MoE or gather). "None observed"
  does not prove that no compilation occurred.

Underruns and deadline misses are playback estimates, not failed synthesis requests.

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

### Asynchronous CFG benchmarks

This benchmark covers 8 configurations in 8 runs, one run each, with **800/800 successful
measured requests**. It measures the opt-in asynchronous CFG profile
`deploy/easymagpie_cfg_async.yaml` (named `easymagpie_cfg_async_compiled.yaml` at the benchmarked revision) together with the standard non-CFG profile
at the same revision. Two attempts stopped at NVML preflight before server start and are
excluded. Legacy was not rerun: the legacy rows below are copied unchanged from the
synchronous benchmark (source `028cc3807ff82dd12083055533c5870031a5f621`) for reference.

#### Setup and measurement

- Non-CFG and CFG async source: `630260754595d5803f118fd98aa080c1e88a86a0`.
  Results apply to this revision.
- GPU0: NVIDIA RTX 5880 Ada Generation, 49,140 MiB. Python 3.12,
  torch 2.11.0+cu130, vLLM/vLLM-Omni 0.24.0. BF16 exports, FP16 talker,
  FP32 SSM cache, eager FP32 codec.
- The 100-text corpus, speaker, sampling settings, client ordering seed, PCM format, and
  WS chunking are the same as in the synchronous benchmark.
- Each run uses fresh isolated compiler caches and one server: startup, 2 c1 smoke requests,
  100 excluded warmups at the selected concurrency, 100 measured requests, and 2 c1 reuse
  requests. Unlike the synchronous benchmark, there is no separate startup compilation gate;
  the measured server compiles from empty caches during startup.
- Non-CFG uses the standard asynchronous/chunked `easymagpie.yaml`. CFG async uses Stage-0
  asynchronous scheduling with unchunked prefill and the CFG slot and token budgets shown above.
  WS runs set `EASYMAGPIE_ASYNC_STREAMING=1`, which the benchmarked revision required for incremental input;
  the variable has since been removed and the async CFG profile accepts incremental input directly.

#### Throughput and latency

Columns and definitions match the synchronous tables. Each row contains 100/100 successful
measured requests. Values are mean/p95.

| Configuration | Audio s | Wall s | Requests/s | Audio/wall | RTF | TTFA mean/p95 ms | Latency mean/p95 s | ITL mean/p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Legacy HTTP c1 | 788.56 | 80.161 | 1.2475 | 9.8372× | 0.10165 | 155.9/191.1 | 0.799/1.483 | 52.0/68.0 |
| Legacy HTTP c32 | 745.60 | 17.200 | 5.8139 | 43.3484× | 0.02307 | 639.1/917.1 | 4.944/9.553 | 371.9/581.9 |
| Legacy HTTP c32 repeat | 766.40 | 18.087 | 5.5288 | 42.3724× | 0.02360 | 851.3/1259.8 | 5.124/10.536 | 357.4/543.5 |
| Non-CFG HTTP c1 | 755.28 | 76.737 | 1.3032 | 9.8425× | 0.10160 | 162.2/207.7 | 0.765/1.332 | 51.3/68.3 |
| Non-CFG HTTP c32 | 791.60 | 17.688 | 5.6535 | 44.7530× | 0.02234 | 988.6/1559.6 | 5.271/10.822 | 345.0/866.2 |
| Non-CFG WS c1 | 755.28 | 78.522 | 1.2735 | 9.6187× | 0.10396 | 173.5/213.6 | 0.782/1.329 | 51.8/69.6 |
| Non-CFG WS c32 | 818.08 | 17.418 | 5.7411 | 46.9664× | 0.02129 | 766.9/1495.4 | 4.665/7.537 | 303.0/510.2 |
| CFG async HTTP c1 | 628.72 | 76.611 | 1.3053 | 8.2066× | 0.12185 | 157.8/188.7 | 0.764/1.332 | 61.8/83.0 |
| CFG async HTTP c32 | 626.88 | 14.354 | 6.9667 | 43.6729× | 0.02290 | 656.4/820.3 | 4.153/7.135 | 358.4/538.9 |
| CFG async HTTP c32 repeat | 631.60 | 14.794 | 6.7593 | 42.6916× | 0.02342 | 707.8/802.7 | 4.306/7.256 | 365.6/562.8 |
| CFG async WS c1 | 631.20 | 77.447 | 1.2912 | 8.1501× | 0.12270 | 161.8/190.9 | 0.772/1.305 | 61.9/84.9 |
| CFG async WS c32 | 622.96 | 16.021 | 6.2419 | 38.8843× | 0.02572 | 1025.9/1543.8 | 4.626/8.269 | 369.7/609.0 |
| CFG async WS c32 repeat | 676.24 | 18.175 | 5.5020 | 37.2070× | 0.02688 | 1050.2/1648.3 | 4.949/8.540 | 368.2/593.2 |

Rows marked "repeat" rerun the CFG async c32 cells at `c12046773037bd0b6a08467652b587a5b5b304a1`, which
contains only behaviour-neutral refactoring of the source above, with the same setup and procedure
(measured before this README section and one error message were edited).
The repeat is within 2.25%/4.31% (HTTP/WS) of the original audio throughput. In the same session, non-CFG
c32 measured 11.06%/7.44% above its rows above (HTTP/WS), so these differences are within one-run variation.

Relative to non-CFG at the same revision, CFG async audio throughput is 16.62%/15.27% lower
at c1 HTTP/WS and 2.41%/17.21% lower at c32 HTTP/WS. CFG async generates 16–24% less total
audio. As in the synchronous benchmark, the generated outputs differ, so these ratios do
not measure equal-output inference speed or speech quality.

Non-CFG HTTP audio throughput is 0.05% above legacy at c1, and 3.24% above the first
legacy c32 run or 5.62% above its repeat. CFG async HTTP c32 is 0.75%/3.07% above the two
legacy c32 runs; CFG async HTTP c1 is 16.58% below legacy c1.

Compared with the synchronous CFG rows above, CFG async audio throughput is 4.23% lower at
c1 HTTP, 2.06% higher at c1 WS, and 33.87%/21.25% higher at c32 HTTP/WS. The synchronous CFG
and legacy rows come from different revisions and benchmark runs, and each new configuration
has one run, so all these differences are descriptive, without confidence intervals or a
controlled regression estimate.

#### Playback and measured compilation

Definitions match the synchronous playback table.

| Configuration | Underruns | Requests affected / 100 | Deadline misses | Measured compiler/cache activity |
| --- | ---: | ---: | ---: | --- |
| Legacy HTTP c1 | 0 | 0 | 0 | None observed |
| Legacy HTTP c32 | 15 | 11 | 59 | None observed |
| Legacy HTTP c32 repeat | 3 | 3 | 21 | 16 files; 2 fused-MoE specializations |
| Non-CFG HTTP c1 | 0 | 0 | 0 | None observed |
| Non-CFG HTTP c32 | 10 | 9 | 103 | 64 files; 8 fused-MoE specializations |
| Non-CFG WS c1 | 0 | 0 | 0 | None observed |
| Non-CFG WS c32 | 5 | 3 | 38 | 16 files; 2 fused-MoE specializations |
| CFG async HTTP c1 | 0 | 0 | 0 | None observed |
| CFG async HTTP c32 | 0 | 0 | 25 | None observed |
| CFG async HTTP c32 repeat | 3 | 3 | 37 | None observed |
| CFG async WS c1 | 0 | 0 | 0 | None observed |
| CFG async WS c32 | 12 | 9 | 60 | 16 files; 2 fused-MoE specializations |
| CFG async WS c32 repeat | 12 | 11 | 56 | 16 files; 2 fused-MoE specializations |

All c1 rows have zero playback gaps. CFG async HTTP c32 has no underruns; CFG async WS c32
affects 9/100 requests (synchronous CFG: 58/100 HTTP and 76/100 WS).

Among the eight new runs, cache snapshots show additional compiled specializations in three
c32 runs after the fixed warmup. Their timings include compilation overhead and should not be
treated as steady-state measurements. The other five new runs show no measured cache activity.
No new run logged a measured JIT warning or detected compiler processes at phase boundaries.

#### Startup, memory, and validation limits

Fresh-cache startup to health took 266.80–270.81 s across all eight runs (CFG async
269.15–270.81 s). This is startup wall time, not part of request throughput.

GPU memory was sampled every 15 seconds across whole server lifetimes. CFG async maxima were
26,551 MiB at c1 and 29,129–31,601 MiB at c32; non-CFG maxima were 38,443 MiB at c1 and
40,671–42,679 MiB at c32. As in the synchronous benchmark, codec cache budgets differ between
the profiles, so this does not establish lower intrinsic CFG memory.

Validation checked request IDs, PCM format, frame counts, and chunk durations for all 1,632
WAVs across smoke, warmup, measured, and reuse phases. Each run passed readiness, integrity,
GPU health, and cleanup checks. Two 40.80-second non-CFG WS c32 outputs remain in the totals
without filtering. Speech intelligibility, speaker similarity, and stopping quality were not evaluated.

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
EASYMAGPIE_DEPLOY_CONFIG=tools/easymagpie_vllm_omni/deploy/easymagpie_cfg_async.yaml \
  bash tools/easymagpie_vllm_omni/scripts/run_server.sh /path/to/reconverted_v2 8091
```

The synchronous `deploy/easymagpie_cfg.yaml` is an alternative with the same requests and settings below;
it remains available during the move to asynchronous scheduling.

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

Guided requests require a CFG pipeline (`easymagpie_cfg_async` or `easymagpie_cfg`),
valid V2 capability metadata, and a registered speaker. Unsupported deployments,
older artifacts, and raw-reference CFG requests are rejected before generation.
Incremental sessions require registered speakers for conditional-only inference too.
On both CFG pipelines, `max_new_tokens` bounds the whole stream, including text warmup,
EOS, and acoustic tail, for guided and conditional requests. The bound must fit the
deployment's cache and codec capacities. Settings are fixed for the session.

### Incremental text with paired V2 CFG

Use the direct `AsyncOmni.generate()` streaming-input API with the explicit
`easymagpie_cfg_async` pipeline (or the synchronous alternative `easymagpie_cfg`) and a
registered speaker. Speaker embeddings must be
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
both supplied CFG profiles). Text that leaves no room for EOS/tail is rejected. Use
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

The `easymagpie_cfg_async.yaml` deployment, like its synchronous alternative `easymagpie_cfg.yaml`,
supports mixed CFG/conditional traffic with 64 physical Stage-0 slots and 32 public Stage-1 slots. CFG consumes two physical
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

`EasyMagpieCFGScheduler` admits logical units in FCFS order; the async profile's
`EasyMagpieCFGAsyncScheduler` extends it and keeps the same admission rules. Before admission it
reserves engine-managed cache for the entire bounded prompt plus generation
history of every branch. A pair waits together if sequence slots, the per-step
token budget, or cache capacity are unavailable. A unit that cannot fit the
total cache at all is rejected. Both branches run at aligned positions, without
preemption or Mamba replay. Request-wide reservation trades memory utilization
for this scheduling guarantee; lowering the output bound can increase admission
capacity. The benchmarks do not isolate the throughput or memory effect of this
reservation policy.

The mixed scheduler requires FCFS, Mamba cache mode `none`, and a token budget
covering every active physical decode sequence. Both profiles keep prefix caching and
chunked prefill disabled, including when CUDA graphs are enabled. The async profile schedules the
next step while the current one runs; the synchronous profile schedules one step at a time.
CPU tests use the real hybrid cache allocator to check pair isolation, independent stopping, queueing under pressure,
abort/late outputs, failure cleanup, and slot reuse. They also check multiple
guided pairs mixed with a conditional predictor row and private connector filtering.
Async scheduler tests also admit two guided pairs with a conditional request within the physical budget.

The benchmark measures guided traffic rather than saturated mixed CFG/conditional
traffic. GPU cancellation and lifecycle behavior under saturated mixed load, and
waveform parity across batch sizes, require separate validation.
