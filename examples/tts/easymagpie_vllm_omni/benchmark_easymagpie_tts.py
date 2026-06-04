# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Benchmark the EasyMagpieTTS talker via a single-stage AsyncOmni engine.

Runs the EasyMagpie talker (``EasyMagpieTTSForConditionalGeneration``) only —
no codec / code2wav — producing stacked audio codes as output. It mirrors the
reference ``qwen3-tts`` talker benchmark and the
``easymagpie_inference_demo.ipynb`` engine setup.

Metrics measured under configurable concurrency:

* **TTFT** — time to first decoded frame (first engine token).
* **ITL**  — per-token inter-token latency (excluding the first token).
* **E2E**  — end-to-end latency per request (up to the audio-EOS frame).
* **RTX**  — real-time factor (generated audio seconds / wall time). Both the
  per-request RTX and an overall (concurrency-aware) RTX are reported.
* **Throughput** — frames/s and requests/s.

The decode loop stops at the audio-EOS frame (the EasyMagpie model signals
end-of-speech inside codebook 0 of the codes, not via the vLLM token stream),
so E2E / RTX reflect the real synthesized length rather than the full token
budget. Audio duration is derived from the number of decoded frames:
``audio_seconds = (frames - speech_delay) * frame_stacking_factor / codec_fps``.

Reads texts from a file (one utterance per line, optionally tab-separated with
the text in the second column) or uses a small built-in default set.

Usage:
    # Basic benchmark with default prompts
    python benchmark_easymagpie_tts.py \\
        --model ./easymp_vllm_model \\
        --num-requests 50

    # From a text file with a concurrency sweep
    python benchmark_easymagpie_tts.py \\
        --model ./easymp_vllm_model \\
        --text-file texts.txt \\
        --num-requests 100 \\
        --concurrency 1 4 8

    # With torch profiler on the run
    python benchmark_easymagpie_tts.py \\
        --model ./easymp_vllm_model \\
        --num-requests 20 --concurrency 1 --profile

    # Save JSON results
    python benchmark_easymagpie_tts.py \\
        --model ./easymp_vllm_model \\
        --text-file texts.txt \\
        --num-requests 100 --concurrency 1 4 \\
        --result-dir results/
"""

import os

# Keep spawn semantics consistent with the qwen3-tts / eartts demos in case the
# executor backend is switched to a multiproc one.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import argparse
import asyncio
import json
import logging
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_PROMPTS = [
    "Hello, welcome to the voice synthesis benchmark test.",
    "She said she would be here by noon, but nobody showed up.",
    "The quick brown fox jumps over the lazy dog near the riverbank.",
    "I can't believe how beautiful the sunset looks from up here on the mountain.",
    "Please remember to bring your identification documents to the appointment tomorrow morning.",
    "Have you ever wondered what it would be like to travel through time and visit ancient civilizations?",
    "The restaurant on the corner serves the best pasta I have ever tasted in my entire life.",
    "After the meeting, we should discuss the quarterly results and plan for the next phase.",
    "Learning a new language takes patience, practice, and a genuine curiosity about other cultures.",
    "The train leaves at half past seven, so we need to arrive at the station before then.",
    "Could you please turn down the music a little bit, I'm trying to concentrate on my work.",
    "It was a dark and stormy night when the old lighthouse keeper heard a knock at the door.",
]


# ---------------------------------------------------------------------------
#  Stage config generation
# ---------------------------------------------------------------------------


def _build_easymagpie_stage_config(
    max_num_seqs: int = 1,
    profile: bool = False,
    torch_profiler_dir: str = "./profiler_traces",
    with_stack: bool = False,
    record_shapes: bool = False,
    gpu_memory_utilization: float = 0.8,
    max_model_len: int = 1024,
    max_num_batched_tokens: int = 1024,
    enforce_eager: bool = False,
    max_new_tokens: int = 256,
    dtype: str = "float16",
    distributed_executor_backend: str = "uni",
    cudagraph_mode: Optional[str] = None,
) -> dict:
    """Build a single-stage YAML dict containing only the EasyMagpie talker.

    Mirrors the engine_args used in ``easymagpie_inference_demo.ipynb``.

    ``cudagraph_mode`` (when set and ``enforce_eager`` is False) selects the
    vLLM CUDA-graph capture strategy via ``compilation_config.cudagraph_mode``:

    * ``FULL_AND_PIECEWISE`` (vLLM default) — a single full graph over the whole
      forward for uniform/decode-only batches, piecewise (per compile group:
      backbone vs local transformer) for mixed/prefill batches.
    * ``PIECEWISE`` — always piecewise, so the backbone and local transformer are
      captured as *separate* graphs even during decode. This re-introduces a
      launch boundary between them (so decode is a touch slower than FULL), but
      makes the backbone-vs-LT split visible as two distinct ``cudaGraphLaunch``
      events in a profiler.
    * ``FULL`` / ``FULL_DECODE_ONLY`` — full graph (decode only) capture.
    * ``NONE`` — no CUDA graphs (equivalent to ``--enforce-eager``).
    """
    engine_args: dict[str, Any] = {
        "model_stage": "easymagpie",
        "max_num_seqs": max_num_seqs,
        "model_arch": "EasyMagpieTTSForConditionalGeneration",
        "worker_type": "ar",
        "scheduler_cls": "vllm_omni.core.sched.omni_ar_scheduler.OmniARAsyncScheduler",
        "enforce_eager": enforce_eager,
        "trust_remote_code": True,
        "async_scheduling": True,
        "enable_prefix_caching": False,
        "engine_output_type": "audio",
        "gpu_memory_utilization": gpu_memory_utilization,
        # "uni" runs the worker in-process (no shm_broadcast IPC); use "mp"
        # only when TP/PP > 1 or you actually need a separate worker process.
        "distributed_executor_backend": distributed_executor_backend,
        "max_num_batched_tokens": max_num_batched_tokens,
        "max_model_len": max_model_len,
        # bf16/fp16 (not fp32): the Nemotron-H fused-MoE Triton kernel's block
        # sizes are tuned for 16-bit and overflow shared memory in fp32.
        "dtype": dtype,
        "mamba_ssm_cache_dtype": "float32",
        "attention_backend": "TRITON_ATTN",
        # We feed prompt_token_ids directly; the model loads the bundled
        # AutoTokenizer from the model dir to tokenize context_text + text.
        "skip_tokenizer_init": True,
    }

    # CUDA-graph capture strategy. ``enforce_eager`` already disables graphs, so
    # only set compilation_config when graphs are enabled (mirrors the sidecar
    # server). Passed as a plain dict so it survives YAML serialization; vLLM
    # parses it into a CompilationConfig.
    if cudagraph_mode is not None and not enforce_eager:
        engine_args["compilation_config"] = {"cudagraph_mode": cudagraph_mode}

    if profile:
        engine_args["profiler_config"] = {
            "profiler": "torch",
            "torch_profiler_dir": os.path.abspath(torch_profiler_dir),
            "torch_profiler_with_stack": with_stack,
            "torch_profiler_record_shapes": record_shapes,
        }

    cfg = {
        "stage_args": [
            {
                "stage_id": 0,
                "stage_type": "llm",
                "is_comprehension": True,
                "final_output": True,
                # "audio" (not "latent") is required for a single-stage AR TTS
                # model: it makes the AR model runner attach the per-step
                # multimodal payload ("audio_codes") to the output so the codes
                # reach the client.
                "final_output_type": "audio",
                "runtime": {"devices": "0"},
                "engine_args": engine_args,
                "default_sampling_params": {
                    # The backbone token sampler is a no-op (audio is sampled in
                    # the local transformer); the audio temperature/top-k are
                    # forwarded per-request via additional_information.
                    "temperature": 0.0,
                    "max_tokens": max_new_tokens,
                    "detokenize": False,
                    # Audio EOS lives in the codes, not the vLLM token stream, so
                    # let the budget run and stop client-side at the EOS frame.
                    "ignore_eos": True,
                },
            }
        ],
    }
    return cfg


def _write_temp_stage_config(cfg: dict) -> str:
    """Write stage config dict to a temp YAML file, return its path."""
    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".yaml",
        prefix="easymagpie_bench_",
        delete=False,
    )
    yaml.dump(cfg, tmp, default_flow_style=False, sort_keys=False)
    tmp.close()
    logger.info("Wrote single-stage config to %s", tmp.name)
    return tmp.name


# ---------------------------------------------------------------------------
#  Model metadata (arch scalars + tokenizer + speaker embedding)
# ---------------------------------------------------------------------------


@dataclass
class ModelMeta:
    """Scalars + assets needed to build prompts and interpret outputs."""

    arch: Any
    tokenizer: Any
    speaker_embedding: Any  # torch.Tensor (T_audio, embedding_dim)
    prompt_len: int
    audio_eos_id: int
    speech_delay: int
    frame_stacking_factor: int
    stop_token_id: int  # backbone stop token the model emits at the audio-EOS frame


def _load_model_meta(
    model_dir: str,
    speaker: str,
    speaker_embedding_path: Optional[str],
    context_text: str,
) -> ModelMeta:
    """Read config.json, tokenizer, and the speaker embedding from the model dir.

    Mirrors the prompt-prep cells of ``easymagpie_inference_demo.ipynb``: the
    arch scalars come from ``config.json``, the speaker embedding from
    ``speaker_embeddings/<name>.pt``, and the prefill placeholder length from
    ``EasyMagpieTTSForConditionalGeneration.estimate_prompt_len(...)``.
    """
    import torch
    from transformers import AutoTokenizer

    from easymagpie_vllm_omni.config import EasyMagpieOmniArch
    from easymagpie_vllm_omni.easymagpie import EasyMagpieTTSForConditionalGeneration

    model_path = Path(model_dir)
    config = json.loads((model_path / "config.json").read_text())
    arch = EasyMagpieOmniArch.from_hf_config(type("Cfg", (), config))

    # Speaker-encoded context audio (audio branch of prepare_context_tensors).
    if speaker_embedding_path is not None:
        emb_path = Path(speaker_embedding_path)
    else:
        emb_path = model_path / "speaker_embeddings" / f"{speaker}.pt"
    if not emb_path.exists():
        raise FileNotFoundError(f"Speaker embedding not found: {emb_path}")
    loaded = torch.load(emb_path, map_location="cpu")
    speaker_embedding = loaded["speaker_encoding"] if isinstance(loaded, dict) else loaded
    speaker_embedding = speaker_embedding.to(torch.float32)

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    # prompt_len depends only on the speaker embedding + context_text (+ task
    # embedding) — NOT on the target text (which is streamed in-engine), so we
    # size it once.
    prompt_len = EasyMagpieTTSForConditionalGeneration.estimate_prompt_len(
        speaker_embedding,
        tokenize=lambda t: tokenizer.encode(t),
        context_text=context_text,
        has_task_embedding=arch.num_task_embeddings > 0,
    )

    return ModelMeta(
        arch=arch,
        tokenizer=tokenizer,
        speaker_embedding=speaker_embedding,
        prompt_len=int(prompt_len),
        audio_eos_id=int(arch.audio_eos_id),
        speech_delay=int(getattr(arch, "streaming_speech_delay", 0) or 0),
        frame_stacking_factor=int(arch.frame_stacking_factor),
        stop_token_id=EasyMagpieTTSForConditionalGeneration.audio_eos_stop_token_id(type("Cfg", (), config)),
    )


def build_prompt(
    text: str,
    meta: ModelMeta,
    context_text: str,
    lt_temperature: float,
    lt_topk: int,
) -> dict:
    """Build an engine input dict from a target sentence + the shared assets."""
    additional_information = {
        "speaker_embedding": meta.speaker_embedding,  # (T_audio, embedding_dim)
        "context_text": context_text,  # plain string, tokenized in-model
        "text": text,  # plain target sentence, tokenized in-model
        "temperature": lt_temperature,  # audio sampling temperature (local transformer)
        "top_k": lt_topk,  # audio sampling top-k (local transformer)
    }
    return {
        "prompt_token_ids": [0] * meta.prompt_len,
        "additional_information": additional_information,
    }


# ---------------------------------------------------------------------------
#  Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class RequestResult:
    success: bool = False
    text: str = ""
    prompt_len: int = 0
    num_generated: int = 0  # decoded frames (engine tokens) up to EOS
    audio_frames: int = 0  # codec frames of real audio (post speech-delay, pre-EOS)
    audio_s: float = 0.0  # synthesized audio duration in seconds
    steps: int = 0
    eos_reached: bool = False
    ttft_s: float = 0.0
    e2e_s: float = 0.0
    rtx: float = 0.0  # audio_s / e2e_s
    inter_token_latencies: list = field(default_factory=list)
    error: str = ""


@dataclass
class BenchmarkResult:
    config_name: str = ""
    concurrency: int = 0
    num_requests: int = 0
    completed: int = 0
    failed: int = 0
    duration_s: float = 0.0
    # TTFT
    mean_ttft_ms: float = 0.0
    median_ttft_ms: float = 0.0
    p95_ttft_ms: float = 0.0
    p99_ttft_ms: float = 0.0
    # E2E
    mean_e2e_ms: float = 0.0
    median_e2e_ms: float = 0.0
    p95_e2e_ms: float = 0.0
    p99_e2e_ms: float = 0.0
    # ITL (inter-token latency, excluding first token)
    mean_itl_ms: float = 0.0
    median_itl_ms: float = 0.0
    p95_itl_ms: float = 0.0
    p99_itl_ms: float = 0.0
    # RTX (real-time factor: synthesized audio seconds / generation seconds)
    mean_rtx: float = 0.0
    median_rtx: float = 0.0
    overall_rtx: float = 0.0  # total_audio_s / wall_clock_duration (concurrency-aware)
    # Throughput
    total_tokens: int = 0
    total_audio_s: float = 0.0
    mean_tokens_per_request: float = 0.0
    token_throughput: float = 0.0
    request_throughput: float = 0.0
    per_request: list = field(default_factory=list)


# ---------------------------------------------------------------------------
#  Inference
# ---------------------------------------------------------------------------


def _extract_request_output(stage_output):
    """Return the RequestOutput-like object from a yielded stage output.

    AsyncOmni stages may yield either a wrapper carrying ``.request_output``
    (qwen3-tts style) or the RequestOutput directly (easymagpie demo style).
    """
    return getattr(stage_output, "request_output", stage_output)


async def run_one_request(
    omni,
    prompt: dict,
    sampling_params,
    request_id: str,
    meta: ModelMeta,
    codec_fps: float,
    stop_on_eos: bool,
) -> RequestResult:
    """Submit one TTS request, collect per-token timing and audio length.

    Each engine step yields one decoded frame (one layer-0 token). We time the
    first token (TTFT) and the gaps between subsequent tokens (ITL). The audio
    EOS lives in codebook 0 of the accumulated ``audio_codes`` (not in the vLLM
    token stream), so we watch the newest decoded frame and stop at the EOS
    frame to recover the real synthesized length.
    """
    import torch

    result = RequestResult()
    t_start = time.perf_counter()
    t_last_token = None
    prev_num_tokens = 0
    eos_decode_idx = None  # 0-based decode-frame index where audio EOS appears

    try:
        gen = omni.generate(
            prompt,
            sampling_params_list=[sampling_params],
            request_id=request_id,
        )
        async for stage_output in gen:
            now = time.perf_counter()
            ro = _extract_request_output(stage_output)
            result.steps += 1

            cur_num_tokens = prev_num_tokens
            if hasattr(ro, "outputs") and ro.outputs:
                out0 = ro.outputs[0]
                cum_ids = getattr(out0, "cumulative_token_ids", None)
                if cum_ids is not None:
                    cur_num_tokens = len(cum_ids)
                else:
                    cur_num_tokens = len(getattr(out0, "token_ids", []) or [])

            if cur_num_tokens > prev_num_tokens:
                if t_last_token is None:
                    result.ttft_s = now - t_start
                else:
                    result.inter_token_latencies.append(now - t_last_token)
                t_last_token = now

                # Audio-EOS detection on the newest decoded frame. The accumulated
                # audio_codes hold (T_ctx prefill + decode) rows; the last row is
                # the newest decoded frame. Only meaningful past the speech delay.
                mm = getattr(stage_output, "multimodal_output", None) or {}
                audio_codes = mm.get("audio_codes")
                newest_frame_idx = cur_num_tokens - 1  # 0-based decode-frame index
                if (
                    eos_decode_idx is None
                    and newest_frame_idx >= meta.speech_delay
                    and isinstance(audio_codes, torch.Tensor)
                    and audio_codes.numel() > 0
                ):
                    # audio EOS in ANY codebook (not just codebook 0) — mirrors the
                    # reference EOS check and the model's own stop signal.
                    if bool((audio_codes[-1] == meta.audio_eos_id).any()):
                        eos_decode_idx = newest_frame_idx
                        result.eos_reached = True

                prev_num_tokens = cur_num_tokens

                if eos_decode_idx is not None and stop_on_eos:
                    break

        t_end = time.perf_counter()
        result.e2e_s = t_end - t_start
        result.num_generated = prev_num_tokens
        result.success = True

        if result.ttft_s == 0.0 and result.steps > 0:
            result.ttft_s = t_end - t_start

        # Real audio length: frames between the start of speech (speech_delay)
        # and the EOS frame (or the full decode if no EOS was emitted).
        last_audio_frame = eos_decode_idx if eos_decode_idx is not None else prev_num_tokens
        result.audio_frames = max(0, last_audio_frame - meta.speech_delay)
        if codec_fps > 0:
            result.audio_s = result.audio_frames * meta.frame_stacking_factor / codec_fps
        result.rtx = result.audio_s / result.e2e_s if result.e2e_s > 0 else 0.0

    except Exception as exc:
        result.e2e_s = time.perf_counter() - t_start
        result.error = str(exc)
        logger.error("Request %s failed: %s", request_id, exc)
    finally:
        # Make sure the async generator is closed (aborts the request in the
        # engine when we broke out early on EOS).
        try:
            await gen.aclose()
        except Exception:
            pass

    return result


# ---------------------------------------------------------------------------
#  Worker / concurrency
# ---------------------------------------------------------------------------


async def worker(
    worker_id: int,
    omni,
    texts: list,
    meta: ModelMeta,
    context_text: str,
    lt_temperature: float,
    lt_topk: int,
    sampling_params,
    codec_fps: float,
    stop_on_eos: bool,
    results: list,
    counter: dict,
    lock: asyncio.Lock,
):
    """Persistent async worker that picks texts until the quota is exhausted."""
    while True:
        async with lock:
            if counter["remaining"] <= 0:
                break
            counter["remaining"] -= 1
            idx = counter["issued"]
            counter["issued"] += 1

        text = texts[idx % len(texts)]
        request_id = f"bench-easymp-w{worker_id}-{uuid.uuid4().hex[:8]}"

        prompt = build_prompt(
            text=text,
            meta=meta,
            context_text=context_text,
            lt_temperature=lt_temperature,
            lt_topk=lt_topk,
        )

        result = await run_one_request(
            omni,
            prompt,
            sampling_params,
            request_id,
            meta,
            codec_fps,
            stop_on_eos,
        )
        result.text = text
        result.prompt_len = len(prompt["prompt_token_ids"])

        async with lock:
            results.append(result)
            done = len(results)

        if done % 10 == 0 or done == counter["total"]:
            logger.info("  progress: %d / %d", done, counter["total"])


# ---------------------------------------------------------------------------
#  Metrics
# ---------------------------------------------------------------------------


def _pct(arr, p):
    return float(np.percentile(arr, p)) if len(arr) > 0 else 0.0


def compute_and_print_metrics(
    results: list,
    duration: float,
    concurrency: int,
    num_requests: int,
) -> BenchmarkResult:
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    bench = BenchmarkResult(
        concurrency=concurrency,
        num_requests=num_requests,
        completed=len(successful),
        failed=len(failed),
        duration_s=duration,
    )

    if not successful:
        print("ERROR: No requests completed successfully.")
        return bench

    ttfts = [r.ttft_s * 1000 for r in successful]
    e2es = [r.e2e_s * 1000 for r in successful]
    rtxs = [r.rtx for r in successful]
    all_itls = []
    for r in successful:
        all_itls.extend([t * 1000 for t in r.inter_token_latencies])
    gen_tokens = [r.num_generated for r in successful]

    bench.mean_ttft_ms = float(np.mean(ttfts))
    bench.median_ttft_ms = float(np.median(ttfts))
    bench.p95_ttft_ms = _pct(ttfts, 95)
    bench.p99_ttft_ms = _pct(ttfts, 99)

    bench.mean_e2e_ms = float(np.mean(e2es))
    bench.median_e2e_ms = float(np.median(e2es))
    bench.p95_e2e_ms = _pct(e2es, 95)
    bench.p99_e2e_ms = _pct(e2es, 99)

    if all_itls:
        bench.mean_itl_ms = float(np.mean(all_itls))
        bench.median_itl_ms = float(np.median(all_itls))
        bench.p95_itl_ms = _pct(all_itls, 95)
        bench.p99_itl_ms = _pct(all_itls, 99)

    bench.mean_rtx = float(np.mean(rtxs))
    bench.median_rtx = float(np.median(rtxs))

    bench.total_tokens = int(sum(gen_tokens))
    bench.total_audio_s = float(sum(r.audio_s for r in successful))
    bench.mean_tokens_per_request = float(np.mean(gen_tokens))
    bench.token_throughput = bench.total_tokens / duration if duration > 0 else 0.0
    bench.request_throughput = len(successful) / duration if duration > 0 else 0.0
    bench.overall_rtx = bench.total_audio_s / duration if duration > 0 else 0.0

    bench.per_request = [
        {
            "ttft_ms": r.ttft_s * 1000,
            "e2e_ms": r.e2e_s * 1000,
            "rtx": r.rtx,
            "num_generated": r.num_generated,
            "audio_frames": r.audio_frames,
            "audio_s": r.audio_s,
            "eos_reached": r.eos_reached,
            "steps": r.steps,
            "prompt_len": r.prompt_len,
            "mean_itl_ms": float(np.mean([t * 1000 for t in r.inter_token_latencies]))
            if r.inter_token_latencies
            else 0.0,
            "text": r.text,
        }
        for r in successful
    ]

    eos_hits = sum(1 for r in successful if r.eos_reached)

    W = 56
    print(f"\n{'=' * W}")
    print(f"{'Benchmark Result':^{W}}")
    print(f"{'=' * W}")
    print(f"{'Successful requests:':<42}{bench.completed}")
    print(f"{'Failed requests:':<42}{bench.failed}")
    print(f"{'Reached audio EOS:':<42}{eos_hits} / {bench.completed}")
    print(f"{'Concurrency:':<42}{concurrency}")
    print(f"{'Wall-clock duration (s):':<42}{duration:.2f}")
    print(f"{'Request throughput (req/s):':<42}{bench.request_throughput:.2f}")

    print(f"\n{'-' * W}")
    print(f"{'Time to First Token (TTFT)':^{W}}")
    print(f"{'-' * W}")
    print(f"{'Mean  (ms):':<42}{bench.mean_ttft_ms:.2f}")
    print(f"{'Median (ms):':<42}{bench.median_ttft_ms:.2f}")
    print(f"{'P95   (ms):':<42}{bench.p95_ttft_ms:.2f}")
    print(f"{'P99   (ms):':<42}{bench.p99_ttft_ms:.2f}")

    print(f"\n{'-' * W}")
    print(f"{'End-to-End Latency (E2E)':^{W}}")
    print(f"{'-' * W}")
    print(f"{'Mean  (ms):':<42}{bench.mean_e2e_ms:.2f}")
    print(f"{'Median (ms):':<42}{bench.median_e2e_ms:.2f}")
    print(f"{'P95   (ms):':<42}{bench.p95_e2e_ms:.2f}")
    print(f"{'P99   (ms):':<42}{bench.p99_e2e_ms:.2f}")

    print(f"\n{'-' * W}")
    print(f"{'Inter-Token Latency (ITL)':^{W}}")
    print(f"{'-' * W}")
    if all_itls:
        print(f"{'Mean  (ms):':<42}{bench.mean_itl_ms:.2f}")
        print(f"{'Median (ms):':<42}{bench.median_itl_ms:.2f}")
        print(f"{'P95   (ms):':<42}{bench.p95_itl_ms:.2f}")
        print(f"{'P99   (ms):':<42}{bench.p99_itl_ms:.2f}")
    else:
        print(f"{'(no inter-token data)':^{W}}")

    print(f"\n{'-' * W}")
    print(f"{'Real-Time Factor (RTX = audio_s / gen_s)':^{W}}")
    print(f"{'-' * W}")
    print(f"{'Mean RTX (per request):':<42}{bench.mean_rtx:.2f}x")
    print(f"{'Median RTX (per request):':<42}{bench.median_rtx:.2f}x")
    print(f"{'Overall RTX (total audio / wall):':<42}{bench.overall_rtx:.2f}x")

    print(f"\n{'-' * W}")
    print(f"{'Throughput':^{W}}")
    print(f"{'-' * W}")
    print(f"{'Total frames generated:':<42}{bench.total_tokens}")
    print(f"{'Total audio generated (s):':<42}{bench.total_audio_s:.2f}")
    print(f"{'Mean frames / request:':<42}{bench.mean_tokens_per_request:.1f}")
    print(f"{'Frame throughput (frames/s):':<42}{bench.token_throughput:.2f}")
    print(f"{'=' * W}\n")

    if failed:
        print(f"  First {min(3, len(failed))} errors:")
        for r in failed[:3]:
            print(f"    {r.error[:200]}")

    return bench


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------


async def main(args):
    from vllm import SamplingParams
    from vllm_omni import AsyncOmni

    model_name = args.model

    # ── Load texts ────────────────────────────────────────────────────────
    if args.text_file:
        path = Path(args.text_file)
        if not path.exists():
            print(f"ERROR: text file not found: {path}")
            return
        raw_lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
        texts = []
        for line in raw_lines:
            if "\t" in line:
                texts.append(line.split("\t", 1)[1].strip())
            else:
                texts.append(line)
        texts = [t for t in texts if t]
        logger.info("Loaded %d texts from %s", len(texts), path)
    else:
        texts = DEFAULT_PROMPTS
        logger.info("Using %d default prompts", len(texts))

    if not texts:
        print("ERROR: no texts available.")
        return

    # ── Read arch scalars + tokenizer + speaker embedding ─────────────────
    logger.info("Reading model metadata from %s ...", model_name)
    meta = _load_model_meta(
        model_dir=model_name,
        speaker=args.speaker,
        speaker_embedding_path=args.speaker_embedding,
        context_text=args.context_text,
    )
    logger.info(
        "prompt_len=%d  audio_eos_id=%d  speech_delay=%d  frame_stacking=%d",
        meta.prompt_len,
        meta.audio_eos_id,
        meta.speech_delay,
        meta.frame_stacking_factor,
    )
    if meta.prompt_len + args.max_new_tokens > args.max_model_len:
        logger.warning(
            "prompt_len (%d) + max_new_tokens (%d) exceeds max_model_len (%d); raise --max-model-len.",
            meta.prompt_len,
            args.max_new_tokens,
            args.max_model_len,
        )

    max_concurrency = max(args.concurrency)

    # ── Build stage config ────────────────────────────────────────────────
    stage_cfg = _build_easymagpie_stage_config(
        max_num_seqs=max_concurrency,
        profile=args.profile,
        torch_profiler_dir=args.torch_profiler_dir,
        with_stack=args.with_stack,
        record_shapes=args.record_shapes,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        enforce_eager=args.enforce_eager,
        max_new_tokens=args.max_new_tokens,
        dtype=args.dtype,
        distributed_executor_backend=args.distributed_executor_backend,
        cudagraph_mode=args.cudagraph_mode,
    )
    if args.cudagraph_mode is not None and args.enforce_eager:
        logger.warning(
            "--cudagraph-mode %s is ignored because --enforce-eager disables CUDA graphs.",
            args.cudagraph_mode,
        )
    elif args.cudagraph_mode is not None:
        logger.info("CUDA-graph mode: %s", args.cudagraph_mode)
    tmp_config_path = _write_temp_stage_config(stage_cfg)

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_new_tokens,
        detokenize=False,
        ignore_eos=True,
        # The model emits this backbone token at the audio-EOS frame (audio EOS in
        # any codebook), so vLLM stops the request there instead of decoding the
        # full budget. stop_token_ids is honored even with ignore_eos.
        stop_token_ids=[meta.stop_token_id],
    )

    try:
        logger.info("Creating AsyncOmni engine (EasyMagpie talker only) for %s ...", model_name)
        omni = AsyncOmni(
            model=model_name,
            stage_configs_path=tmp_config_path,
            log_stats=args.log_stats,
            stage_init_timeout=args.stage_init_timeout,
        )
        logger.info("Engine ready (single stage: EasyMagpie talker).")

        all_bench_results = []

        for concurrency in args.concurrency:
            logger.info(
                "=== concurrency=%d  requests=%d ===",
                concurrency,
                args.num_requests,
            )

            # ── Warmup ────────────────────────────────────────────────────
            warmup_count = 0 if args.no_warmup else args.num_warmups * concurrency
            if warmup_count > 0:
                logger.info("Warming up with %d requests (concurrency=%d)...", warmup_count, concurrency)
                warmup_results: list = []
                warmup_counter = {
                    "remaining": warmup_count,
                    "issued": 0,
                    "total": warmup_count,
                }
                warmup_lock = asyncio.Lock()
                warmup_tasks = [
                    asyncio.create_task(
                        worker(
                            worker_id=i,
                            omni=omni,
                            texts=texts,
                            meta=meta,
                            context_text=args.context_text,
                            lt_temperature=args.lt_temperature,
                            lt_topk=args.lt_topk,
                            sampling_params=sampling_params,
                            codec_fps=args.codec_frame_rate,
                            stop_on_eos=not args.no_stop_on_eos,
                            results=warmup_results,
                            counter=warmup_counter,
                            lock=warmup_lock,
                        )
                    )
                    for i in range(concurrency)
                ]
                await asyncio.gather(*warmup_tasks)
                warmup_ok = sum(1 for r in warmup_results if r.success)
                logger.info("Warmup done: %d / %d succeeded.", warmup_ok, warmup_count)

            # ── Benchmark run ─────────────────────────────────────────────
            logger.info("Starting benchmark run (%d requests, concurrency=%d)...", args.num_requests, concurrency)

            bench_results: list = []
            counter = {
                "remaining": args.num_requests,
                "issued": 0,
                "total": args.num_requests,
            }
            lock = asyncio.Lock()

            if args.profile:
                logger.info("Starting profiler ...")
                await omni.start_profile(
                    profile_prefix=args.profile_prefix,
                    stages=[0],
                )

            start_time = time.perf_counter()
            try:
                tasks = [
                    asyncio.create_task(
                        worker(
                            worker_id=i,
                            omni=omni,
                            texts=texts,
                            meta=meta,
                            context_text=args.context_text,
                            lt_temperature=args.lt_temperature,
                            lt_topk=args.lt_topk,
                            sampling_params=sampling_params,
                            codec_fps=args.codec_frame_rate,
                            stop_on_eos=not args.no_stop_on_eos,
                            results=bench_results,
                            counter=counter,
                            lock=lock,
                        )
                    )
                    for i in range(concurrency)
                ]
                await asyncio.gather(*tasks)
            finally:
                if args.profile:
                    logger.info("Stopping profiler ...")
                    await omni.stop_profile(stages=[0])

            duration = time.perf_counter() - start_time

            bench = compute_and_print_metrics(
                bench_results,
                duration,
                concurrency,
                args.num_requests,
            )
            bench.config_name = args.config_name
            all_bench_results.append(asdict(bench))

        # ── Save results ──────────────────────────────────────────────────
        if args.result_dir:
            result_dir = Path(args.result_dir)
            result_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_file = result_dir / f"bench_easymagpie_{args.config_name}_{timestamp}.json"
            with open(result_file, "w") as f:
                json.dump(all_bench_results, f, indent=2)
            logger.info("Results saved to %s", result_file)

        omni.shutdown()
    finally:
        os.unlink(tmp_config_path)

    logger.info("Done.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark the EasyMagpieTTS talker (AR stage only) via AsyncOmni",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    model = parser.add_argument_group("model / input")
    model.add_argument(
        "--model",
        type=str,
        default="./easymp_vllm_model",
        help="Converted EasyMagpie model directory (output of easy_magpietts_convert_to_vllm.py)",
    )
    model.add_argument(
        "--text-file",
        type=str,
        default=None,
        help="Path to text file (one utterance per line, optionally tab-separated with text in 2nd column)",
    )
    model.add_argument(
        "--speaker",
        type=str,
        default="eng",
        help="Speaker embedding name under <model>/speaker_embeddings/<name>.pt",
    )
    model.add_argument(
        "--speaker-embedding",
        type=str,
        default=None,
        help="Explicit path to a speaker embedding .pt (overrides --speaker)",
    )
    model.add_argument(
        "--context-text",
        type=str,
        default="[EN]",
        help="Conditioning string tokenized + embedded in-engine (e.g. '[EN]')",
    )
    model.add_argument(
        "--lt-temperature",
        type=float,
        default=0.0,
        help="Audio (local-transformer) sampling temperature (0.0 == argmax)",
    )
    model.add_argument(
        "--lt-topk",
        type=int,
        default=80,
        help="Audio (local-transformer) sampling top-k",
    )
    model.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Max decode frames per request (decode budget; trimmed at audio EOS)",
    )
    model.add_argument(
        "--codec-frame-rate",
        type=float,
        default=25.0,
        help="Codec frame rate (Hz) used to convert decoded frames to audio seconds "
        "(default 25 for the 25fps spectral codec)",
    )

    bench = parser.add_argument_group("benchmark")
    bench.add_argument(
        "-c",
        "--concurrency",
        type=int,
        nargs="+",
        default=[1],
        help="Concurrency levels to test (space-separated, default: 1)",
    )
    bench.add_argument(
        "-n",
        "--num-requests",
        type=int,
        default=50,
        help="Total number of requests per concurrency level (default: 50)",
    )
    bench.add_argument(
        "--num-warmups",
        type=int,
        default=3,
        help="Warmup rounds per concurrency level (total warmup = concurrency * this, default: 3)",
    )
    bench.add_argument("--no-warmup", action="store_true", help="Skip warmup")
    bench.add_argument(
        "--no-stop-on-eos",
        action="store_true",
        help="Do not stop at the audio-EOS frame; run the full decode budget every request",
    )
    bench.add_argument(
        "--config-name",
        type=str,
        default="easymagpie",
        help="Label for this run (used in result filenames)",
    )
    bench.add_argument(
        "--result-dir",
        type=str,
        default=None,
        help="Directory to save JSON results",
    )

    engine = parser.add_argument_group("engine")
    engine.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    engine.add_argument("--max-model-len", type=int, default=1024)
    engine.add_argument("--max-num-batched-tokens", type=int, default=1024)
    engine.add_argument("--dtype", type=str, default="float16", help="Model dtype (float16 / bfloat16)")
    engine.add_argument("--enforce-eager", action="store_true")
    engine.add_argument(
        "--cudagraph-mode",
        type=str,
        default=None,
        choices=["NONE", "PIECEWISE", "FULL", "FULL_DECODE_ONLY", "FULL_AND_PIECEWISE"],
        help="vLLM CUDA-graph capture strategy (compilation_config.cudagraph_mode). "
        "Default: unset (vLLM default, FULL_AND_PIECEWISE). Use PIECEWISE to capture the "
        "backbone and local transformer as separate graphs during decode so their split is "
        "visible in a profiler (slightly slower than the default full decode graph). "
        "Ignored when --enforce-eager is set.",
    )
    engine.add_argument("--stage-init-timeout", type=int, default=300)
    engine.add_argument("--log-stats", action="store_true", default=False)
    engine.add_argument(
        "--distributed-executor-backend",
        type=str,
        default="uni",
        choices=["uni", "mp", "ray"],
        help="vLLM executor backend. 'uni' runs the worker in-process and "
        "avoids shm_broadcast IPC round-trips (recommended for TP=1, single "
        "GPU). Default: uni.",
    )

    prof = parser.add_argument_group("profiling")
    prof.add_argument(
        "--profile",
        action="store_true",
        help="Enable torch profiler during the benchmark run",
    )
    prof.add_argument("--profile-prefix", type=str, default=None, help="Prefix for profiler trace filenames")
    prof.add_argument(
        "--torch-profiler-dir", type=str, default="./profiler_traces", help="Directory for torch profiler traces"
    )
    prof.add_argument("--with-stack", action="store_true", help="Record Python call stacks in profiler")
    prof.add_argument("--record-shapes", action="store_true", help="Record tensor shapes in profiler")

    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
