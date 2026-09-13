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
"""Benchmark EasyMagpie LM via a single-stage AsyncOmni engine.

Measures acoustic-token prediction only (no in-engine Code2Wav). Uses the
``easymagpie_lm`` pipeline so LM throughput can be tracked separately from
the two-stage codec path.

Two input modes, selectable with ``--streaming``:

* whole-text (default) — the full target text is handed to the engine up front.
* streaming-text — subword ids are pushed as the model decodes, ``--tokens-per-chunk``
  ids at a time (prefill chunk, then one ``StreamingInput`` per chunk carrying a
  ``list[int]`` of ids with ``max_tokens == len(chunk)`` so the engine free-runs
  that many frames off one message, then a free-running acoustic tail).

Both run on the same engine config. Reports throughput, TTFT, ITL (mean + p95),
EOS hit rate and overall RTF (estimated from codec frame rate, not decoded audio).

Usage:
    python benchmark_model.py --model ./converted_model_multiturn --num-requests 50
    python benchmark_model.py --model ./converted_model_multiturn -n 50 --streaming
    python benchmark_model.py --model ./converted_model_multiturn -n 50 --streaming --tokens-per-chunk 3
    python benchmark_model.py --model ./converted_model_multiturn -n 50 -c 1 4 8
"""

import os

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import argparse
import asyncio
import copy
import json
import logging
import tempfile
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

_SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DEPLOY_CONFIG = _SCRIPT_DIR.parent / "deploy" / "easymagpie_lm.yaml"

# ── Hardcoded run settings ─────────────────────────────────────────────────
SPEAKER = "eng"
CONTEXT_TEXT = "[EN]"
LT_TEMPERATURE = 0.7  # audio (local-transformer) sampling temperature
LT_TOPK = 80  # audio sampling top-k
CODEC_FRAME_RATE = 25.0  # Hz, used to convert decoded frames -> audio seconds (RTF)
GPU_MEMORY_UTILIZATION = 0.5
DISTRIBUTED_EXECUTOR_BACKEND = "uni"
ENFORCE_EAGER = False
DTYPE = "float16"
STAGE_INIT_TIMEOUT = 300
# vLLM CUDA-graph capture strategy; None == vLLM default (FULL_AND_PIECEWISE).
CUDAGRAPH_MODE: Optional[str] = None

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
]


# ---------------------------------------------------------------------------
#  Deploy config
# ---------------------------------------------------------------------------


def _build_deploy_config(
    deploy_config: str,
    max_num_seqs: int,
    max_model_len: int,
    max_num_batched_tokens: int,
    max_new_tokens: int,
    profile: bool,
    torch_profiler_dir: str,
    load_format: Optional[str],
) -> dict:
    """Load the EasyMagpie LM deploy YAML and apply benchmark runtime overrides."""
    config_path = Path(deploy_config)
    cfg = yaml.safe_load(config_path.read_text())
    if cfg.get("pipeline") != "easymagpie_lm":
        raise ValueError(f"{config_path} must set pipeline: easymagpie_lm")
    stages = cfg.get("stages", [])
    if len(stages) != 1:
        raise ValueError(f"{config_path} must define exactly one EasyMagpie LM stage")

    cfg = copy.deepcopy(cfg)
    stage: dict[str, Any] = cfg["stages"][0]
    stage.update(
        {
            "max_num_seqs": max_num_seqs,
            "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
            "enforce_eager": ENFORCE_EAGER,
            "max_num_batched_tokens": max_num_batched_tokens,
            "max_model_len": max_model_len,
        }
    )
    sampling = stage.setdefault("default_sampling_params", {})
    sampling.update({"max_tokens": max_new_tokens, "ignore_eos": True})
    if load_format is not None:
        stage["load_format"] = load_format
    if CUDAGRAPH_MODE is not None and not ENFORCE_EAGER:
        stage["compilation_config"] = {"cudagraph_mode": CUDAGRAPH_MODE}
    if profile:
        stage["profiler_config"] = {
            "profiler": "torch",
            "torch_profiler_dir": os.path.abspath(torch_profiler_dir),
            "torch_profiler_with_stack": True,
            "torch_profiler_record_shapes": True,
        }

    cfg["dtype"] = DTYPE
    cfg["distributed_executor_backend"] = DISTRIBUTED_EXECUTOR_BACKEND
    return cfg


def _write_temp_deploy_config(cfg: dict) -> str:
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", prefix="easymagpie_bench_", delete=False)
    yaml.dump(cfg, tmp, default_flow_style=False, sort_keys=False)
    tmp.close()
    return tmp.name


# ---------------------------------------------------------------------------
#  Model metadata
# ---------------------------------------------------------------------------


@dataclass
class ModelMeta:
    tokenizer: Any
    speaker_embedding: Any  # torch.Tensor (T_audio, embedding_dim); None in speaker_id mode
    speaker_id: Optional[str]  # known-speaker id (None => pass raw speaker_embedding)
    prompt_len: int
    audio_eos_id: int
    speech_delay: int
    frame_stacking_factor: int
    text_prefill_num: int
    stop_token_id: int  # backbone token emitted at the audio-EOS frame
    text_eos_id: int  # appended to streamed subword ids


def _load_model_meta(
    model_dir: str,
    lim_prefill: Optional[int] = None,
    speaker_id: str = SPEAKER,
    use_spkr_emb: bool = False,
) -> ModelMeta:
    import torch
    from easymagpie_vllm_omni.config import EasyMagpieOmniArch
    from easymagpie_vllm_omni.easymagpie import EasyMagpieTTSForConditionalGeneration
    from easymagpie_vllm_omni.tokenizer import EasyMagpieTextTokenizer

    model_path = Path(model_dir)
    config = json.loads((model_path / "config.json").read_text())
    arch = EasyMagpieOmniArch.from_hf_config(type("Cfg", (), config))

    use_id = not (use_spkr_emb or lim_prefill is not None)
    tokenizer = EasyMagpieTextTokenizer.from_pretrained(model_dir)

    if use_id:
        speaker_embedding = None
        prompt_len = EasyMagpieTTSForConditionalGeneration.get_prompt_len(
            speaker_id,
            model_dir,
            tokenize=tokenizer.encode_context,
        )
    else:
        emb_path = model_path / "speaker_embeddings" / f"{speaker_id}.pt"
        if not emb_path.exists():
            raise FileNotFoundError(f"Speaker embedding not found: {emb_path}")
        loaded = torch.load(emb_path, map_location="cpu")
        speaker_embedding = (loaded["speaker_encoding"] if isinstance(loaded, dict) else loaded).to(torch.float32)
        if lim_prefill is not None:
            orig_frames = int(speaker_embedding.shape[0])
            speaker_embedding = speaker_embedding[: max(1, int(lim_prefill))].contiguous()
            logger.info("Limiting speaker-embedding prefill: %d -> %d frames", orig_frames, speaker_embedding.shape[0])
        prompt_len = EasyMagpieTTSForConditionalGeneration.estimate_prompt_len(
            speaker_embedding,
            tokenize=tokenizer.encode_context,
            context_text=CONTEXT_TEXT,
            has_task_embedding=arch.num_task_embeddings > 0,
        )

    text_prefill_num = arch.text_prefill_num
    prompt_len += text_prefill_num

    return ModelMeta(
        tokenizer=tokenizer,
        speaker_embedding=speaker_embedding,
        speaker_id=speaker_id if use_id else None,
        prompt_len=int(prompt_len),
        audio_eos_id=int(arch.audio_eos_id),
        speech_delay=int(getattr(arch, "streaming_speech_delay", 0) or 0),
        frame_stacking_factor=int(arch.frame_stacking_factor),
        text_prefill_num=text_prefill_num,
        stop_token_id=EasyMagpieTTSForConditionalGeneration.audio_eos_stop_token_id(type("Cfg", (), config)),
        text_eos_id=int(arch.resolved_text_eos_id(int(config.get("text_vocab_size", config.get("vocab_size", 0))))),
    )


def build_prompt(text: str, meta: ModelMeta) -> dict:
    text_prefill_num = getattr(meta, "text_prefill_num", 0)
    text_tokens = list(meta.tokenizer.encode(text, add_special_tokens=False)) + [meta.text_eos_id]
    info: dict = {
        "context_text": CONTEXT_TEXT,
        "text_tokens": text_tokens,
        "prefill_text_tokens": text_tokens[:text_prefill_num],
        "text_prefill_num": text_prefill_num,
        "temperature": LT_TEMPERATURE,
        "top_k": LT_TOPK,
    }
    info.update(_speaker_info(meta))
    return {"prompt_token_ids": [0] * meta.prompt_len, "additional_information": info}


def _speaker_info(meta: ModelMeta) -> dict:
    if meta.speaker_id is not None:
        return {"speaker_id": meta.speaker_id}
    return {"speaker_embedding": meta.speaker_embedding}


# ---------------------------------------------------------------------------
#  Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class RequestResult:
    success: bool = False
    audio_s: float = 0.0
    generated_tokens: int = 0
    audio_frames: int = 0
    eos_reached: bool = False
    finish_reason: Optional[str] = None
    ttft_s: float = 0.0
    ttfa_s: float = 0.0
    inter_token_latencies: list = field(default_factory=list)
    error: str = ""
    request_index: int = -1
    text: str = ""
    text_tokens: int = 0
    audio_codes: Any = field(default=None, repr=False)


# ---------------------------------------------------------------------------
#  Inference
# ---------------------------------------------------------------------------


def _extract_request_output(stage_output):
    return getattr(stage_output, "request_output", stage_output)


def _extract_step_audio_codes(stage_output):
    """Find this step's audio-code payload as one ``[T, Q]`` delta tensor.

    The single-stage EasyMagpie LM emits its codes under the ``model_outputs`` key,
    which vLLM-Omni remaps to the drainable ``audio`` modality; in DELTA mode
    that key is drained every step, so each per-step snapshot carries only the
    codes produced this step (a delta), NOT the cumulative sequence. Callers
    accumulate these deltas in a list and concatenate once to reconstruct the
    full ``[prompt_len + generated, Q]`` tensor — no per-step growth on the wire.

    The ``codes.audio`` / ``audio_codes`` keys are read only as a fallback for
    non-drainable configs.
    """
    import torch

    def reduce_payload(payload):
        if isinstance(payload, list):
            parts = [part for part in payload if isinstance(part, torch.Tensor) and part.numel() > 0]
            return torch.cat(parts, dim=0) if parts else None
        if isinstance(payload, torch.Tensor) and payload.numel() > 0:
            return payload
        return None

    def inspect_output(output):
        if isinstance(output, (list, tuple)):
            for item in reversed(output):
                codes = inspect_output(item)
                if codes is not None:
                    return codes
            return None

        multimodal_output = getattr(output, "multimodal_output", None)
        if isinstance(multimodal_output, Mapping):
            # Drainable single-stage key (remapped from "model_outputs"), then
            # the non-drainable inter-stage keys as a fallback.
            payload = multimodal_output.get("audio")
            if payload is None:
                payload = multimodal_output.get("model_outputs")
            if payload is None:
                payload = multimodal_output.get("audio_codes")
            if payload is None:
                nested_codes = multimodal_output.get("codes")
                if isinstance(nested_codes, Mapping):
                    payload = nested_codes.get("audio")
            codes = reduce_payload(payload)
            if codes is not None:
                return codes

        request_output = getattr(output, "request_output", None)
        if request_output is not None and request_output is not output:
            codes = inspect_output(request_output)
            if codes is not None:
                return codes
        for completion in getattr(output, "outputs", None) or []:
            codes = inspect_output(completion)
            if codes is not None:
                return codes
        return None

    codes = inspect_output(stage_output)
    if codes is not None and codes.ndim == 2:
        return codes
    return None


class StepMeter:
    """Cheap per-request measurement: TTFT, ITL, generated-frame count."""

    def __init__(self, meta: ModelMeta, capture_audio_codes: bool = False):
        self.meta = meta
        self.capture_audio_codes = capture_audio_codes
        self.result = RequestResult()
        self.steps = 0
        self._t_start = time.perf_counter()
        self._t_last = None
        self._prev_tokens = 0
        self._finish_reason = None
        remaining_delay = max(0, meta.speech_delay - getattr(meta, "text_prefill_num", 0))
        self._audio_overhead = 1 + remaining_delay

        # Per-step code deltas, concatenated once in finalize() (never per step,
        # so measurement is not polluted by O(n^2) accumulation/serialization).
        self._code_chunks: list = []

    def observe(self, stage_output) -> None:
        now = time.perf_counter()
        ro = _extract_request_output(stage_output)
        self.steps += 1

        out0 = ro.outputs[0] if getattr(ro, "outputs", None) else None
        if out0 is None:
            return
        fr = getattr(out0, "finish_reason", None)
        if fr is not None:
            self._finish_reason = fr
        if self.capture_audio_codes:
            audio_codes = _extract_step_audio_codes(stage_output)
            if audio_codes is not None and audio_codes.numel() > 0:
                self._code_chunks.append(audio_codes.detach().cpu())
        cum = getattr(out0, "cumulative_token_ids", None)
        cur = len(cum) if cum is not None else self._prev_tokens + len(getattr(out0, "token_ids", []) or [])
        if cur <= self._prev_tokens:
            return

        if self.result.ttfa_s == 0.0 and cur > self._audio_overhead:
            self.result.ttfa_s = now - self._t_start
        if self._t_last is None:
            self.result.ttft_s = now - self._t_start
        else:
            self.result.inter_token_latencies.append(now - self._t_last)
        self._t_last = now
        self._prev_tokens = cur

    def finalize(self) -> RequestResult:
        e2e_s = time.perf_counter() - self._t_start
        self.result.success = True
        if self.result.ttft_s == 0.0 and self.steps > 0:
            self.result.ttft_s = e2e_s
        self.result.eos_reached = self._finish_reason == "stop"
        self.result.finish_reason = self._finish_reason
        self.result.generated_tokens = self._prev_tokens
        audio_frames = max(0, self._prev_tokens - self._audio_overhead)
        self.result.audio_frames = audio_frames
        self.result.audio_s = audio_frames * self.meta.frame_stacking_factor / CODEC_FRAME_RATE
        if self.capture_audio_codes and self._code_chunks:
            import torch

            self.result.audio_codes = torch.cat(self._code_chunks, dim=0)
        return self.result

    def mark_error(self, exc) -> RequestResult:
        self.result.error = str(exc)
        return self.result


async def run_one_request(
    omni,
    inputs,
    sampling_params,
    request_id: str,
    meta: ModelMeta,
    capture_audio_codes: bool = False,
    output_observer=None,
    max_steps: Optional[int] = None,
) -> RequestResult:
    meter = StepMeter(meta, capture_audio_codes=capture_audio_codes)
    gen = None
    try:
        gen = omni.generate(inputs, sampling_params_list=[sampling_params], request_id=request_id)
        async for stage_output in gen:
            meter.observe(stage_output)
            if output_observer is not None:
                output_observer(stage_output)
            if max_steps is not None and meter.steps >= max_steps:
                break
        meter.finalize()
    except Exception as exc:
        meter.mark_error(exc)
        logger.error("Request %s failed: %s", request_id, exc)
    finally:
        if gen is not None:
            try:
                await gen.aclose()
            except Exception:
                # Cleanup failure must not replace the request result already recorded above.
                pass
    return meter.result


def build_streaming_request(text: str, meta: ModelMeta, stream_params, max_new_tokens: int, tokens_per_chunk: int = 1):
    from easymagpie_vllm_omni.serving_stream import EasyMagpieInputStream

    text_prefill_num = getattr(meta, "text_prefill_num", 0)
    prefill_info = {
        "context_text": CONTEXT_TEXT,
        "temperature": LT_TEMPERATURE,
        "top_k": LT_TOPK,
        "text_prefill_num": text_prefill_num,
    }
    prefill_info.update(_speaker_info(meta))
    text_ids = list(meta.tokenizer.encode(text, add_special_tokens=False))
    n = max(1, int(tokens_per_chunk))
    if len(text_ids) < text_prefill_num:
        raise ValueError(f"streaming benchmark text must contain at least {text_prefill_num} tokens")
    first_chunk_size = max(n, text_prefill_num)
    chunks = [text_ids[:first_chunk_size]]
    chunks.extend(text_ids[i : i + n] for i in range(first_chunk_size, len(text_ids), n))
    stream = EasyMagpieInputStream(
        prefill_prompt={"prompt_token_ids": [0] * meta.prompt_len, "additional_information": prefill_info},
        sampling_params=stream_params,
        text_eos_id=meta.text_eos_id,
        max_new_tokens=max_new_tokens,
        text_prefill_num=text_prefill_num,
        queue_depth=len(chunks) + 1,
        coalesce_queued_tokens=False,
    )

    async def inputs():
        for chunk in chunks:
            await stream.put_tokens(chunk)
        await stream.finish()
        async for engine_input in stream.inputs():
            yield engine_input

    return inputs(), stream.observe_output


# ---------------------------------------------------------------------------
#  Worker / concurrency
# ---------------------------------------------------------------------------


async def worker(
    worker_id: int,
    omni,
    texts: list,
    text_token_counts: list,
    meta: ModelMeta,
    sampling_params,
    stream_params,
    streaming: bool,
    capture_audio_codes: bool,
    max_new_tokens: int,
    tokens_per_chunk: int,
    results: list,
    counter: dict,
    lock: asyncio.Lock,
):
    while True:
        async with lock:
            if counter["remaining"] <= 0:
                break
            counter["remaining"] -= 1
            idx = counter["issued"]
            counter["issued"] += 1

        text = texts[idx % len(texts)]
        request_id = f"bench-easymp-w{worker_id}-{uuid.uuid4().hex[:8]}"

        if streaming:
            inputs, observe_output = build_streaming_request(
                text, meta, stream_params, max_new_tokens, tokens_per_chunk
            )
            result = await run_one_request(
                omni,
                inputs,
                stream_params,
                request_id,
                meta,
                capture_audio_codes=capture_audio_codes,
                output_observer=observe_output,
                max_steps=4 * max_new_tokens + 16,
            )
        else:
            result = await run_one_request(
                omni,
                build_prompt(text, meta),
                sampling_params,
                request_id,
                meta,
                capture_audio_codes=capture_audio_codes,
            )
        result.request_index = idx
        result.text = text
        result.text_tokens = text_token_counts[idx % len(text_token_counts)]

        async with lock:
            results.append(result)
            done = len(results)
        if done % 10 == 0 or done == counter["total"]:
            logger.info("  progress: %d / %d", done, counter["total"])


# ---------------------------------------------------------------------------
#  Metrics
# ---------------------------------------------------------------------------


def compute_and_print_metrics(
    results: list, duration: float, concurrency: int, print_request_stats: bool = False
) -> dict:
    ok = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    ttfts = [r.ttft_s * 1000 for r in ok]
    ttfas = [r.ttfa_s * 1000 for r in ok if r.ttfa_s > 0.0]
    itls = [t * 1000 for r in ok for t in r.inter_token_latencies]
    total_audio_s = sum(r.audio_s for r in ok)
    eos_hits = sum(1 for r in ok if r.eos_reached)
    produced_frames = [r.audio_frames for r in ok]

    summary = {
        "concurrency": concurrency,
        "completed": len(ok),
        "failed": len(failed),
        "eos_hits": eos_hits,
        "duration_s": duration,
        "req_per_s": len(ok) / duration if duration > 0 else 0.0,
        "ttft_mean_ms": float(np.mean(ttfts)) if ttfts else 0.0,
        "ttft_p95_ms": float(np.percentile(ttfts, 95)) if ttfts else 0.0,
        "ttfa_mean_ms": float(np.mean(ttfas)) if ttfas else 0.0,
        "ttfa_p95_ms": float(np.percentile(ttfas, 95)) if ttfas else 0.0,
        "itl_mean_ms": float(np.mean(itls)) if itls else 0.0,
        "itl_p95_ms": float(np.percentile(itls, 95)) if itls else 0.0,
        "rtf": total_audio_s / duration if duration > 0 else 0.0,
        "frames_per_utterance": float(np.mean(produced_frames)) if produced_frames else 0.0,
    }

    W = 48
    print(f"\n{'=' * W}")
    print(f"{f'Benchmark (concurrency={concurrency})':^{W}}")
    print(f"{'=' * W}")
    if not ok:
        print("ERROR: no requests completed successfully.")
        if failed:
            print(f"  e.g. {failed[0].error[:200]}")
        return summary
    print(f"{'Requests (ok / failed):':<28}{summary['completed']} / {summary['failed']}")
    print(f"{'Reached audio EOS:':<28}{eos_hits} / {summary['completed']}")
    print(f"{'Duration (s):':<28}{duration:.2f}")
    print(f"{'Throughput (req/s):':<28}{summary['req_per_s']:.2f}")
    print(f"{'TTFT mean / p95 (ms):':<28}{summary['ttft_mean_ms']:.2f} / {summary['ttft_p95_ms']:.2f}")
    print(f"{'ITL  mean / p95 (ms):':<28}{summary['itl_mean_ms']:.2f} / {summary['itl_p95_ms']:.2f}")
    print(f"{'TTFA mean / p95 (ms):':<28}{summary['ttfa_mean_ms']:.2f} / {summary['ttfa_p95_ms']:.2f}")
    print(f"{'Produced frames / utterance:':<28}{summary['frames_per_utterance']:.1f}")
    print(f"{'RTF (audio_s / wall):':<28}{summary['rtf']:.2f}x")
    print(f"{'=' * W}\n")
    if print_request_stats:
        print("Per-request generation (one generated token = one stacked acoustic model frame):")
        for r in sorted(ok, key=lambda item: item.request_index):
            text_preview = r.text.replace("\n", " ")
            if len(text_preview) > 72:
                text_preview = text_preview[:69] + "..."
            print(
                f"  [{r.request_index:04d}] text_tokens={r.text_tokens:3d} "
                f"generated={r.generated_tokens:4d} audio_frames={r.audio_frames:4d} "
                f"est_audio={r.audio_s:6.2f}s finish={r.finish_reason or 'unknown':>7}  {text_preview}"
            )
        print()
    return summary


def save_audio_codes(results: list, output_dir: str, meta: ModelMeta, concurrency: int) -> None:
    """Save raw and codec-ready acoustic codes without including disk I/O in benchmark time."""
    import torch

    level_dir = Path(output_dir) / f"concurrency_{concurrency}"
    level_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for result in results:
        raw_codes = result.audio_codes
        if not isinstance(raw_codes, torch.Tensor) or raw_codes.ndim != 2:
            logger.warning("No cumulative audio codes captured for request %d", result.request_index)
            continue

        raw_codes = raw_codes.detach().to(device="cpu", dtype=torch.long).contiguous()
        remaining_delay = max(0, meta.speech_delay - getattr(meta, "text_prefill_num", 0))
        start = min(raw_codes.shape[0], meta.prompt_len + remaining_delay)
        end = raw_codes.shape[0] - (1 if result.eos_reached and raw_codes.shape[0] > start else 0)
        acoustic_codes = raw_codes[start:end].contiguous()
        path = level_dir / f"request_{result.request_index:04d}.pt"
        torch.save(
            {
                "text": result.text,
                "text_tokens": result.text_tokens,
                "finish_reason": result.finish_reason,
                "prompt_len": meta.prompt_len,
                "speech_delay": meta.speech_delay,
                "frame_stacking_factor": meta.frame_stacking_factor,
                "codec_frame_rate": CODEC_FRAME_RATE,
                "raw_codes": raw_codes,
                "audio_codes": acoustic_codes,
            },
            path,
        )
        saved += 1
    logger.info("Saved audio codes for %d requests to %s", saved, level_dir)


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------


async def _run_workers(
    omni, texts, text_token_counts, meta, sampling_params, stream_params, args, n_requests, concurrency
):
    results: list = []
    counter = {"remaining": n_requests, "issued": 0, "total": n_requests}
    lock = asyncio.Lock()
    tasks = [
        asyncio.create_task(
            worker(
                i,
                omni,
                texts,
                text_token_counts,
                meta,
                sampling_params,
                stream_params,
                args.streaming,
                bool(args.audio_codes_dir),
                args.max_new_tokens,
                args.tokens_per_chunk,
                results,
                counter,
                lock,
            )
        )
        for i in range(concurrency)
    ]
    await asyncio.gather(*tasks)
    return results


async def main(args):
    import vllm_plugin_easymagpie_omni

    vllm_plugin_easymagpie_omni.register()

    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind
    from vllm_omni import AsyncOmni

    if args.text_file:
        path = Path(args.text_file)
        if not path.exists():
            print(f"ERROR: text file not found: {path}")
            return
        texts = []
        for line in path.read_text().splitlines():
            line = line.strip()
            if line:
                texts.append(line.split("\t", 1)[1].strip() if "\t" in line else line)
        texts = [t for t in texts if t]
    else:
        texts = DEFAULT_PROMPTS
    if not texts:
        print("ERROR: no texts available.")
        return
    logger.info("Loaded %d texts", len(texts))

    meta = _load_model_meta(
        args.model, lim_prefill=args.lim_prefill, speaker_id=args.speaker_id, use_spkr_emb=args.use_spkr_emb
    )
    logger.info(
        "Speaker mode: %s",
        f"known speaker_id={meta.speaker_id!r}" if meta.speaker_id else "raw speaker_embedding tensor per request",
    )
    logger.info(
        "prompt_len=%d  audio_eos_id=%d  speech_delay=%d  text_prefill=%d  frame_stacking=%d",
        meta.prompt_len,
        meta.audio_eos_id,
        meta.speech_delay,
        meta.text_prefill_num,
        meta.frame_stacking_factor,
    )
    if meta.prompt_len + args.max_new_tokens > args.max_model_len:
        logger.warning("prompt_len + max_new_tokens exceeds max_model_len (%d)", args.max_model_len)
    if args.streaming:
        logger.info("Mode: streaming-text  tokens_per_chunk=%d", max(1, args.tokens_per_chunk))
    else:
        logger.info("Mode: whole-text")
    text_token_counts = [len(meta.tokenizer.encode(text, add_special_tokens=False)) for text in texts]

    deploy_cfg = _build_deploy_config(
        deploy_config=args.deploy_config,
        max_num_seqs=max(args.concurrency),
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_new_tokens=args.max_new_tokens,
        profile=args.profile,
        torch_profiler_dir=args.torch_profiler_dir,
        load_format=args.load_format,
    )
    tmp_config_path = _write_temp_deploy_config(deploy_cfg)

    is_dummy = args.load_format == "dummy"
    stop_token_ids = [] if is_dummy else [meta.stop_token_id]
    if is_dummy:
        logger.info("Dummy weights: dropping stop token; every request runs exactly %d steps", args.max_new_tokens)

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_new_tokens,
        detokenize=False,
        ignore_eos=True,
        stop_token_ids=stop_token_ids,
        output_kind=RequestOutputKind.DELTA,
    )
    stream_params = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        detokenize=False,
        ignore_eos=True,
        stop_token_ids=stop_token_ids,
        output_kind=RequestOutputKind.DELTA,
    )

    try:
        logger.info("Creating AsyncOmni engine for %s (pipeline=easymagpie_lm) ...", args.model)
        omni = AsyncOmni(
            model=args.model,
            deploy_config=tmp_config_path,
            log_stats=False,
            stage_init_timeout=STAGE_INIT_TIMEOUT,
        )
        logger.info("Engine ready.")

        summaries = []
        for concurrency in args.concurrency:
            logger.info("=== concurrency=%d  requests=%d ===", concurrency, args.num_requests)

            warmup_count = 0 if args.no_warmup else args.num_warmups * concurrency
            if warmup_count > 0:
                logger.info("Warming up with %d requests...", warmup_count)
                await _run_workers(
                    omni,
                    texts,
                    text_token_counts,
                    meta,
                    sampling_params,
                    stream_params,
                    args,
                    warmup_count,
                    concurrency,
                )

            if args.profile:
                await omni.start_profile(stages=[0])
            start = time.perf_counter()
            try:
                results = await _run_workers(
                    omni,
                    texts,
                    text_token_counts,
                    meta,
                    sampling_params,
                    stream_params,
                    args,
                    args.num_requests,
                    concurrency,
                )
            finally:
                if args.profile:
                    await omni.stop_profile(stages=[0])
            duration = time.perf_counter() - start

            summaries.append(compute_and_print_metrics(results, duration, concurrency, args.print_request_stats))

            if args.audio_codes_dir:
                save_audio_codes(results, args.audio_codes_dir, meta, concurrency)

        print(f"\n{'=' * 56}")
        print(f"{'Summary':^56}")
        print(f"{'=' * 56}")
        for s in summaries:
            print(
                f"concurrency={s['concurrency']}:  "
                f"req/s {s['req_per_s']:.2f},  "
                f"ttft {s['ttft_mean_ms']:.1f}ms,  "
                f"itl {s['itl_mean_ms']:.1f}ms,  "
                f"rtf {s['rtf']:.2f}x"
            )
        print(f"{'=' * 56}\n")

        omni.shutdown()
    finally:
        os.unlink(tmp_config_path)
    logger.info("Done.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark EasyMagpie LM (acoustic tokens only) via AsyncOmni",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model", type=str, default="./converted_model_multiturn", help="Converted EasyMagpie model dir"
    )
    parser.add_argument(
        "--deploy-config",
        type=str,
        default=str(DEFAULT_DEPLOY_CONFIG),
        help="Base EasyMagpie LM deploy YAML; benchmark runtime values override capacity settings",
    )
    parser.add_argument("--text-file", type=str, default=None, help="One utterance per line (optionally tab-sep)")
    parser.add_argument(
        "--print-request-stats",
        action="store_true",
        help="Print text-token count, generated frames, estimated audio duration, and finish reason per request",
    )
    parser.add_argument(
        "--audio-codes-dir",
        type=str,
        default=None,
        help="Diagnostic: save raw and codec-ready acoustic-code tensors for every request as .pt files",
    )
    parser.add_argument("--streaming", action="store_true", help="Benchmark the token-streamed input path")
    parser.add_argument(
        "--tokens-per-chunk",
        type=int,
        default=1,
        help="Streaming only: number of subword ids to feed per StreamingInput chunk "
        "(the engine free-runs that many frames off one message). Default: %(default)s.",
    )
    parser.add_argument("-c", "--concurrency", type=int, nargs="+", default=[1], help="Concurrency levels to test")
    parser.add_argument("-n", "--num-requests", type=int, default=50, help="Requests per concurrency level")
    parser.add_argument("--num-warmups", type=int, default=3, help="Warmup rounds (total = concurrency * this)")
    parser.add_argument("--no-warmup", action="store_true", help="Skip warmup")
    parser.add_argument("--max-new-tokens", type=int, default=2048, help="Max decode frames per request")
    parser.add_argument(
        "--lim-prefill",
        type=int,
        default=None,
        help="Cap the speaker-embedding prefill to the first N frames (default: no limit). "
        "Use e.g. --lim-prefill 1 to mimic a single-token custom-voice prefill.",
    )
    parser.add_argument(
        "--speaker-id",
        type=str,
        default=SPEAKER,
        help="Known speaker id (string) passed in the prompt; the model holds its embedding as "
        "precomputed state (default: %(default)s).",
    )
    parser.add_argument(
        "--use-spkr-emb",
        action="store_true",
        help="Ship the raw speaker_embedding tensor per request instead of the known speaker_id "
        "(exercises the custom-voice path).",
    )
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument(
        "--load-format",
        type=str,
        default=None,
        choices=["auto", "dummy", "safetensors", "pt"],
        help="Weight loading strategy ('dummy' = random weights, skip checkpoint)",
    )
    parser.add_argument("--profile", action="store_true", help="Enable torch profiler (with stack + shapes)")
    parser.add_argument("--torch-profiler-dir", type=str, default="./profiler_traces", help="Profiler trace dir")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
