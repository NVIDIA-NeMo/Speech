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

Two input modes, selectable with ``--streaming``:

* whole-text (default) — the full target text is handed to the engine up front.
* streaming-text — subword ids are pushed one at a time as the model decodes
  (prefill chunk, then one ``StreamingInput`` chunk per subword with
  ``max_tokens=1``, then a free-running acoustic tail).

Both run on the same engine config. Reports throughput, TTFT, ITL (mean + p95),
EOS hit rate and overall RTF.

Usage:
    python benchmark_model.py --model ./easymp_vllm_model --num-requests 50
    python benchmark_model.py --model ./easymp_vllm_model -n 50 --streaming
    python benchmark_model.py --model ./easymp_vllm_model -n 50 -c 1 4 8
"""

import os

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import argparse
import asyncio
import json
import logging
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Hardcoded run settings ─────────────────────────────────────────────────
SPEAKER = "eng"
CONTEXT_TEXT = "[EN]"
LT_TEMPERATURE = 0.7  # audio (local-transformer) sampling temperature
LT_TOPK = 80  # audio sampling top-k
CODEC_FRAME_RATE = 25.0  # Hz, used to convert decoded frames -> audio seconds (RTF)
GPU_MEMORY_UTILIZATION = 0.8
DISTRIBUTED_EXECUTOR_BACKEND = "uni"
ENFORCE_EAGER = False
DTYPE = "float16"
STAGE_INIT_TIMEOUT = 300
# vLLM CUDA-graph capture strategy; None == vLLM default (FULL_AND_PIECEWISE).
#CUDAGRAPH_MODE: Optional[str] = "PIECEWISE"
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
#  Stage config
# ---------------------------------------------------------------------------


def _build_stage_config(
    max_num_seqs: int,
    max_model_len: int,
    max_num_batched_tokens: int,
    max_new_tokens: int,
    profile: bool,
    torch_profiler_dir: str,
    load_format: Optional[str],
) -> dict:
    """Single-stage YAML dict for the EasyMagpie talker (see the demo notebook)."""
    engine_args: dict[str, Any] = {
        "model_stage": "easymagpie",
        "max_num_seqs": max_num_seqs,
        "model_arch": "EasyMagpieTTSForConditionalGeneration",
        "worker_type": "ar",
        # EasyMagpie-aware scheduler serves both paths: it forwards per-chunk
        # text_token and the raised acoustic-tail max_tokens for streaming, and is
        # a drop-in equivalent of the stock scheduler for whole-text.
        "scheduler_cls": "easymagpie_vllm_omni.scheduler.EasyMagpieARAsyncScheduler",
        "enforce_eager": ENFORCE_EAGER,
        "trust_remote_code": True,
        "async_scheduling": True,
        "enable_prefix_caching": False,
        "engine_output_type": "audio",
        "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
        "distributed_executor_backend": DISTRIBUTED_EXECUTOR_BACKEND,
        "max_num_batched_tokens": max_num_batched_tokens,
        "max_model_len": max_model_len,
        "dtype": DTYPE,
        "mamba_ssm_cache_dtype": "float32",
        "attention_backend": "TRITON_ATTN",
        "skip_tokenizer_init": True,
    }
    if load_format is not None:
        engine_args["load_format"] = load_format
    if CUDAGRAPH_MODE is not None and not ENFORCE_EAGER:
        engine_args["compilation_config"] = {"cudagraph_mode": CUDAGRAPH_MODE}
    if profile:
        engine_args["profiler_config"] = {
            "profiler": "torch",
            "torch_profiler_dir": os.path.abspath(torch_profiler_dir),
            "torch_profiler_with_stack": True,
            "torch_profiler_record_shapes": True,
        }

    return {
        # async_chunk enables the streaming-text feed; no-op for whole-text.
        "async_chunk": True,
        "stage_args": [
            {
                "stage_id": 0,
                "stage_type": "llm",
                "is_comprehension": True,
                "final_output": True,
                "final_output_type": "audio",
                "runtime": {"devices": "0"},
                "engine_args": engine_args,
                "default_sampling_params": {
                    "temperature": 0.0,
                    "max_tokens": max_new_tokens,
                    "detokenize": False,
                    "ignore_eos": True,
                },
            }
        ],
    }


def _write_temp_stage_config(cfg: dict) -> str:
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
    stop_token_id: int  # backbone token emitted at the audio-EOS frame
    text_eos_id: int  # appended to streamed subword ids


def _load_model_meta(
    model_dir: str,
    lim_prefill: Optional[int] = None,
    speaker_id: str = SPEAKER,
    use_spkr_emb: bool = False,
) -> ModelMeta:
    import torch
    from transformers import AutoTokenizer

    from easymagpie_vllm_omni.config import EasyMagpieOmniArch
    from easymagpie_vllm_omni.easymagpie import EasyMagpieTTSForConditionalGeneration

    model_path = Path(model_dir)
    config = json.loads((model_path / "config.json").read_text())
    arch = EasyMagpieOmniArch.from_hf_config(type("Cfg", (), config))

    # Default: pass the known ``speaker_id`` (the model holds the embedding as
    # precomputed state). Ship the raw tensor instead when explicitly requested
    # (--use-spkr-emb) or when ``lim_prefill`` truncates it (so it no longer
    # matches the registered embedding).
    use_id = not (use_spkr_emb or lim_prefill is not None)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    if use_id:
        # Known-speaker flow: the caller never loads the embedding tensor — it
        # passes the speaker_id and the model holds the embedding as state.
        speaker_embedding = None
        prompt_len = EasyMagpieTTSForConditionalGeneration.get_prompt_len(
            speaker_id,
            model_dir,
            tokenize=lambda t: tokenizer.encode(t),
        )
    else:
        emb_path = model_path / "speaker_embeddings" / f"{speaker_id}.pt"
        if not emb_path.exists():
            raise FileNotFoundError(f"Speaker embedding not found: {emb_path}")
        loaded = torch.load(emb_path, map_location="cpu")
        speaker_embedding = (loaded["speaker_encoding"] if isinstance(loaded, dict) else loaded).to(torch.float32)
        # Optionally cap the speaker-embedding prefill to the first ``lim_prefill``
        # frames (mimics a single-token custom-voice prefill, cf. Qwen3-TTS).
        if lim_prefill is not None:
            orig_frames = int(speaker_embedding.shape[0])
            speaker_embedding = speaker_embedding[: max(1, int(lim_prefill))].contiguous()
            logger.info("Limiting speaker-embedding prefill: %d -> %d frames", orig_frames, speaker_embedding.shape[0])
        prompt_len = EasyMagpieTTSForConditionalGeneration.estimate_prompt_len(
            speaker_embedding,
            tokenize=lambda t: tokenizer.encode(t),
            context_text=CONTEXT_TEXT,
            has_task_embedding=arch.num_task_embeddings > 0,
        )

    return ModelMeta(
        tokenizer=tokenizer,
        speaker_embedding=speaker_embedding,
        speaker_id=speaker_id if use_id else None,
        prompt_len=int(prompt_len),
        audio_eos_id=int(arch.audio_eos_id),
        speech_delay=int(getattr(arch, "streaming_speech_delay", 0) or 0),
        frame_stacking_factor=int(arch.frame_stacking_factor),
        stop_token_id=EasyMagpieTTSForConditionalGeneration.audio_eos_stop_token_id(type("Cfg", (), config)),
        text_eos_id=int(config.get("text_vocab_size", config.get("vocab_size", 0))) - 2,
    )


def build_prompt(text: str, meta: ModelMeta) -> dict:
    # Known-speaker path: pass ``speaker_id`` (the model holds the speaker's
    # context-audio embedding as precomputed state) instead of shipping the
    # ``(T_audio, embedding_dim)`` tensor per request. Falls back to a raw
    # ``speaker_embedding`` tensor only when ``--use-spkr-emb`` is set.
    info: dict = {
        "context_text": CONTEXT_TEXT,
        "text": text,
        "temperature": LT_TEMPERATURE,
        "top_k": LT_TOPK,
    }
    info.update(_speaker_info(meta))
    return {"prompt_token_ids": [0] * meta.prompt_len, "additional_information": info}


def _speaker_info(meta: ModelMeta) -> dict:
    """Speaker identifier passed in ``additional_information`` (id vs. raw tensor)."""
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
    eos_reached: bool = False
    ttft_s: float = 0.0
    inter_token_latencies: list = field(default_factory=list)
    error: str = ""


# ---------------------------------------------------------------------------
#  Inference
# ---------------------------------------------------------------------------


def _extract_request_output(stage_output):
    return getattr(stage_output, "request_output", stage_output)


class StepMeter:
    """Cheap per-request measurement: TTFT, ITL, generated-frame count.

    Deliberately does **no** per-step tensor work. End-of-generation (audio-EOS /
    backbone stop token) is detected engine-side via ``stop_token_ids``; here we
    only read timestamps and the running token count so the measurement loop never
    blocks the engine and the throughput number reflects the model, not the harness.
    """

    def __init__(self, meta: ModelMeta):
        self.meta = meta
        self.result = RequestResult()
        self.steps = 0
        self._t_start = time.perf_counter()
        self._t_last = None
        self._prev_tokens = 0
        self._finish_reason = None

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
        cum = getattr(out0, "cumulative_token_ids", None)
        cur = len(cum) if cum is not None else self._prev_tokens + len(getattr(out0, "token_ids", []) or [])
        if cur <= self._prev_tokens:
            return

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
        # "stop" => generation ended on the backbone stop token (audio-EOS);
        # "length" => hit max_tokens without reaching EOS.
        self.result.eos_reached = self._finish_reason == "stop"
        audio_frames = max(0, self._prev_tokens - self.meta.speech_delay)
        self.result.audio_s = audio_frames * self.meta.frame_stacking_factor / CODEC_FRAME_RATE
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
    pace=None,
    max_steps: Optional[int] = None,
) -> RequestResult:
    """Drain one request's per-step outputs and measure it via :class:`StepMeter`.

    ``inputs`` is the prompt dict (whole-text) or an async generator of
    ``StreamingInput`` chunks; ``pace`` (streaming only) is awaited after each
    frame to release the next chunk. Termination is engine-driven: vLLM stops the
    request at the backbone stop token (audio-EOS) or ``max_tokens``, which ends
    the output stream — we just iterate to completion. ``max_steps`` is a streaming
    safety valve only.
    """
    meter = StepMeter(meta)
    gen = None
    try:
        gen = omni.generate(inputs, sampling_params_list=[sampling_params], request_id=request_id)
        async for stage_output in gen:
            meter.observe(stage_output)
            if max_steps is not None and meter.steps >= max_steps:
                break
            if pace is not None:
                await pace()
        meter.finalize()
    except Exception as exc:
        meter.mark_error(exc)
        logger.error("Request %s failed: %s", request_id, exc)
    finally:
        if gen is not None:
            try:
                await gen.aclose()
            except Exception:
                pass
    return meter.result


def _clone_sampling_params(sampling_params, max_tokens: int):
    import copy

    sp = copy.deepcopy(sampling_params)
    sp.max_tokens = max(1, int(max_tokens))
    return sp


def build_streaming_request(text: str, meta: ModelMeta, stream_params, max_new_tokens: int):
    """Async ``StreamingInput`` feed + pacing coroutine (§5 of the demo).

    Prefill (speaker + context, no text), one chunk per subword with
    ``max_tokens=1``, then a ``-1`` mask sentinel with a larger tail budget so the
    model free-runs to audio-EOS. Input is paced by output via the queue.
    """
    try:
        from vllm.engine.protocol import StreamingInput
    except ImportError:
        from vllm.v1.engine.async_llm import StreamingInput

    prefill_info = {
        "context_text": CONTEXT_TEXT,
        "temperature": LT_TEMPERATURE,
        "top_k": LT_TOPK,
    }
    prefill_info.update(_speaker_info(meta))
    text_ids = list(meta.tokenizer.encode(text, add_special_tokens=False)) + [meta.text_eos_id]
    tail_params = _clone_sampling_params(stream_params, max_new_tokens - len(text_ids))
    go_queue: asyncio.Queue = asyncio.Queue()

    async def inputs():
        yield StreamingInput(
            prompt={"prompt_token_ids": [0] * meta.prompt_len, "additional_information": prefill_info},
            sampling_params=stream_params,
        )
        for tok in text_ids:
            await go_queue.get()
            yield StreamingInput(
                prompt={"prompt_token_ids": [0], "additional_information": {"text_token": int(tok)}},
                sampling_params=stream_params,
            )
        await go_queue.get()
        yield StreamingInput(
            prompt={"prompt_token_ids": [0], "additional_information": {"text_token": -1}},
            sampling_params=tail_params,
        )

    async def pace():
        await go_queue.put(True)

    return inputs(), pace


# ---------------------------------------------------------------------------
#  Worker / concurrency
# ---------------------------------------------------------------------------


async def worker(
    worker_id: int,
    omni,
    texts: list,
    meta: ModelMeta,
    sampling_params,
    stream_params,
    streaming: bool,
    max_new_tokens: int,
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
            inputs, pace = build_streaming_request(text, meta, stream_params, max_new_tokens)
            result = await run_one_request(
                omni, inputs, stream_params, request_id, meta, pace=pace, max_steps=4 * max_new_tokens + 16
            )
        else:
            result = await run_one_request(omni, build_prompt(text, meta), sampling_params, request_id, meta)

        async with lock:
            results.append(result)
            done = len(results)
        if done % 10 == 0 or done == counter["total"]:
            logger.info("  progress: %d / %d", done, counter["total"])


# ---------------------------------------------------------------------------
#  Metrics
# ---------------------------------------------------------------------------


def compute_and_print_metrics(results: list, duration: float, concurrency: int) -> dict:
    ok = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    ttfts = [r.ttft_s * 1000 for r in ok]
    itls = [t * 1000 for r in ok for t in r.inter_token_latencies]
    total_audio_s = sum(r.audio_s for r in ok)
    eos_hits = sum(1 for r in ok if r.eos_reached)

    summary = {
        "concurrency": concurrency,
        "completed": len(ok),
        "failed": len(failed),
        "eos_hits": eos_hits,
        "duration_s": duration,
        "req_per_s": len(ok) / duration if duration > 0 else 0.0,
        "ttft_mean_ms": float(np.mean(ttfts)) if ttfts else 0.0,
        "ttft_p95_ms": float(np.percentile(ttfts, 95)) if ttfts else 0.0,
        "itl_mean_ms": float(np.mean(itls)) if itls else 0.0,
        "itl_p95_ms": float(np.percentile(itls, 95)) if itls else 0.0,
        "rtf": total_audio_s / duration if duration > 0 else 0.0,
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
    print(f"{'RTF (audio_s / wall):':<28}{summary['rtf']:.2f}x")
    print(f"{'=' * W}\n")
    return summary


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------


async def _run_workers(omni, texts, meta, sampling_params, stream_params, args, n_requests, concurrency):
    results: list = []
    counter = {"remaining": n_requests, "issued": 0, "total": n_requests}
    lock = asyncio.Lock()
    tasks = [
        asyncio.create_task(
            worker(
                i, omni, texts, meta, sampling_params, stream_params,
                args.streaming, args.max_new_tokens, results, counter, lock,
            )
        )
        for i in range(concurrency)
    ]
    await asyncio.gather(*tasks)
    return results


async def main(args):
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
        "prompt_len=%d  audio_eos_id=%d  speech_delay=%d  frame_stacking=%d",
        meta.prompt_len, meta.audio_eos_id, meta.speech_delay, meta.frame_stacking_factor,
    )
    if meta.prompt_len + args.max_new_tokens > args.max_model_len:
        logger.warning("prompt_len + max_new_tokens exceeds max_model_len (%d)", args.max_model_len)
    logger.info("Mode: %s", "streaming-text" if args.streaming else "whole-text")

    stage_cfg = _build_stage_config(
        max_num_seqs=max(args.concurrency),
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_new_tokens=args.max_new_tokens,
        profile=args.profile,
        torch_profiler_dir=args.torch_profiler_dir,
        load_format=args.load_format,
    )
    tmp_config_path = _write_temp_stage_config(stage_cfg)

    # With dummy (random) weights the backbone emits the audio-EOS stop token at
    # random steps, so requests finish early at random lengths. To force every
    # request to run the full, fixed number of decode steps we DROP the stop
    # token (instead of pinning min_tokens): the model repurposes a 2-token dummy
    # backbone vocab, and vLLM's min_tokens processor would -inf-mask the whole
    # ``all_stop_token_ids`` set — which includes the tokenizer's real eos id
    # (~151k) — indexing far outside the 2-wide logits tensor and tripping a CUDA
    # device-side assert. With no stop token and ignore_eos, only max_tokens ends
    # the request, giving exactly ``max_new_tokens`` steps.
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
    # Streaming: max_tokens=1 -> one chunk per decoded frame.
    stream_params = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        detokenize=False,
        ignore_eos=True,
        stop_token_ids=stop_token_ids,
        output_kind=RequestOutputKind.DELTA,
    )

    try:
        logger.info("Creating AsyncOmni engine for %s ...", args.model)
        omni = AsyncOmni(
            model=args.model,
            stage_configs_path=tmp_config_path,
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
                await _run_workers(omni, texts, meta, sampling_params, stream_params, args, warmup_count, concurrency)

            if args.profile:
                await omni.start_profile(stages=[0])
            start = time.perf_counter()
            try:
                results = await _run_workers(
                    omni, texts, meta, sampling_params, stream_params, args, args.num_requests, concurrency
                )
            finally:
                if args.profile:
                    await omni.stop_profile(stages=[0])
            duration = time.perf_counter() - start

            summaries.append(compute_and_print_metrics(results, duration, concurrency))

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
        description="Benchmark the EasyMagpieTTS talker (AR stage only) via AsyncOmni",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model", type=str, default="./easymp_vllm_model", help="Converted EasyMagpie model dir")
    parser.add_argument("--text-file", type=str, default=None, help="One utterance per line (optionally tab-sep)")
    parser.add_argument("--streaming", action="store_true", help="Benchmark the token-streamed input path")
    parser.add_argument("-c", "--concurrency", type=int, nargs="+", default=[1], help="Concurrency levels to test")
    parser.add_argument("-n", "--num-requests", type=int, default=50, help="Requests per concurrency level")
    parser.add_argument("--num-warmups", type=int, default=3, help="Warmup rounds (total = concurrency * this)")
    parser.add_argument("--no-warmup", action="store_true", help="Skip warmup")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Max decode frames per request")
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
    parser.add_argument("--max-model-len", type=int, default=1024)
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
