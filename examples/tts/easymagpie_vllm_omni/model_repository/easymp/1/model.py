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
"""Triton Python backend for EasyMagpieTTS driven by vllm-omni's AsyncOmni engine.

Wraps ``EasyMagpieTTSForConditionalGeneration`` (the vLLM-Omni talker, same model
used by the inference demo / benchmark): it streams stacked codec frames, which we
chunk-decode (overlap-save) through the ``codec`` TensorRT model.

Two request flavours share one engine (``async_chunk=True`` +
``EasyMagpieARAsyncScheduler``, a drop-in for both paths):

* **whole-text** (default) — a request carries the full ``text``; we build a single
  prompt and run ``AsyncOmni.generate(prompt, ...)``.
* **streaming-text** — the client pushes subword ``text_token`` ids one (or a few)
  at a time across several requests sharing a ``stream_id`` (``stream_start`` on the
  first, ``stream_end`` on the last). We feed those tokens as ``StreamingInput``
  chunks (prefill, then one chunk per subword with ``max_tokens=1``, then a free-
  running acoustic tail) into a single ``AsyncOmni.generate(<async-gen>, ...)`` call.
  All audio for the stream is sent back on the ``stream_start`` request's response
  sender; the follow-up requests just feed tokens and close with no output.

Both flavours converge on the same accumulator/codec pipeline:
  1. Each engine step yields the *cumulative* ``audio_codes`` ``(T_total, C*S)``
     (prefill rows + one row per decode step); we slice the decoded rows, drop the
     leading ``speech_delay`` warm-up frames, and stop at the audio EOS frame.
  2. New frames are streamed out in fixed ``codec_chunk_size``-frame windows (with a
     trimmed ``codec_left_context``) through the ``codec`` BLS, which unstacks +
     index-converts + decodes them to 22.05 kHz audio chunks.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import os
import queue
import tempfile
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import torch
import triton_python_backend_utils as pb_utils
import yaml

logging.basicConfig(
    format="%(asctime)s [%(levelname)s]: %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("easymp_triton")

# Sentinel pushed onto a streaming session's token queue to signal "no more text"
# (``stream_end``): the input generator then appends text-EOS + the mask sentinel.
_STREAM_END = object()


class _StreamingSession:
    """Per-``stream_id`` state for a streaming-text request.

    ``response_sender`` is the ``stream_start`` request's sender; *all* audio for the
    stream is sent there. ``token_q`` is fed (thread-safely, via the event loop) by
    the follow-up chunk requests and drained by the input async-generator.

    ``pace_q`` is the output->input back-pressure channel: vLLM drains the input
    async-generator eagerly (a background task, decoupled from decode steps), and the
    scheduler *replaces* ``additional_information`` per chunk, so feeding faster than
    the model decodes overwrites not-yet-consumed ``text_token``s. We therefore
    release exactly one chunk per observed decode-step output: ``_drive_codec`` puts a
    token here after each step and ``_stream_inputs`` waits for one before each yield.
    """

    __slots__ = ("stream_id", "request_id", "speaker", "context_text", "response_sender", "token_q", "pace_q")

    def __init__(self, stream_id, request_id, speaker, context_text, response_sender, token_q, pace_q):
        self.stream_id = stream_id
        self.request_id = request_id
        self.speaker = speaker
        self.context_text = context_text
        self.response_sender = response_sender
        self.token_q = token_q
        self.pace_q = pace_q


def _require_param(parameters: dict, key: str) -> str:
    val = parameters.get(key)
    if isinstance(val, dict):
        val = val.get("string_value")
    if val is None:
        raise KeyError(f"Missing required model parameter: {key!r}")
    return str(val)


class TritonPythonModel:
    def initialize(self, args):
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

        self.model_config = json.loads(args["model_config"])
        params = self.model_config.get("parameters", {})

        self.vllm_model_path = _require_param(params, "vllm_model_path")
        self.default_speaker = _require_param(params, "default_speaker")
        self.default_context_text = _require_param(params, "default_context_text")

        self.max_model_len = int(_require_param(params, "max_model_len"))
        self.max_num_seqs = int(_require_param(params, "max_num_seqs"))
        self.max_num_batched_tokens = int(_require_param(params, "max_num_batched_tokens"))
        self.max_new_tokens = int(_require_param(params, "max_new_tokens"))
        self.gpu_memory_utilization = float(_require_param(params, "gpu_memory_utilization"))

        self.codec_chunk_size = int(_require_param(params, "codec_chunk_size"))
        self.codec_left_context = int(_require_param(params, "codec_left_context"))
        self.first_chunk_frames = int(_require_param(params, "first_chunk_frames"))

        self.lt_temperature = float(_require_param(params, "lt_temperature"))
        self.lt_top_k = int(_require_param(params, "lt_top_k"))

        self._load_arch_and_tokenizer()
        self._init_sampling_helpers()
        self._speaker_cache: dict = {}
        # Inferred from the first codec decode (audio_len / codec_chunk_size).
        self._spf: int | None = None

        # Active streaming-text sessions keyed by stream_id. Touched from the Triton
        # execute() thread (add/lookup) and the asyncio loop thread (removal on
        # completion), so guard it with a lock.
        self._sessions: dict[str, _StreamingSession] = {}
        self._sessions_lock = threading.Lock()

        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._loop_thread.start()

        # One thread per in-flight request serializes its codec decode +
        # response_sender.send calls, off the asyncio loop and overlapping with
        # vLLM generation; Triton dynamic batching then groups the codec calls.
        self._codec_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, self.max_num_seqs),
            thread_name_prefix="easymp_codec",
        )

        self._start_omni_engine()
        logger.info("EasyMagpie initialized (default_speaker=%s)", self.default_speaker)

    def _load_arch_and_tokenizer(self):
        from transformers import AutoTokenizer

        from easymagpie_vllm_omni.config import EasyMagpieOmniArch
        from easymagpie_vllm_omni.easymagpie import EasyMagpieTTSForConditionalGeneration

        config = json.loads((Path(self.vllm_model_path) / "config.json").read_text())
        cfg_obj = type("Cfg", (), config)
        arch = EasyMagpieOmniArch.from_hf_config(cfg_obj)

        self.audio_eos_id = int(arch.audio_eos_id)
        self.speech_delay = int(getattr(arch, "streaming_speech_delay", 0) or 0)
        self.num_stacked_codebooks = int(arch.num_stacked_codebooks)
        self.has_task_embedding = arch.num_task_embeddings > 0
        self.stop_token_id = EasyMagpieTTSForConditionalGeneration.audio_eos_stop_token_id(cfg_obj)
        # Appended after the client's streamed subword ids to close the text channel
        # before the free-running acoustic tail (matches the demo / benchmark).
        self.text_eos_id = int(config.get("text_vocab_size", config.get("vocab_size", 0))) - 2

        self.tokenizer = AutoTokenizer.from_pretrained(self.vllm_model_path, trust_remote_code=True)
        self._estimate_prompt_len = EasyMagpieTTSForConditionalGeneration.estimate_prompt_len

    def _init_sampling_helpers(self):
        """Cache the vLLM sampling/streaming types used by both request flavours."""
        from vllm import SamplingParams
        from vllm.sampling_params import RequestOutputKind

        try:
            from vllm.engine.protocol import StreamingInput
        except ImportError:
            from vllm.v1.engine.async_llm import StreamingInput

        self._SamplingParams = SamplingParams
        self._RequestOutputKind = RequestOutputKind
        self._StreamingInput = StreamingInput

    def _make_sampling_params(self, max_tokens: int):
        """SamplingParams shared by both paths (audio sampling happens in the LT).

        ``output_kind=DELTA`` is what makes ``audio_codes`` arrive as a growing list
        of per-step frames during decode; the backbone token sampler is a no-op
        (temperature 0) and stops at the audio-EOS ``stop_token_id``.
        """
        return self._SamplingParams(
            temperature=0.0,
            max_tokens=max(1, int(max_tokens)),
            detokenize=False,
            ignore_eos=True,
            stop_token_ids=[self.stop_token_id],
            output_kind=self._RequestOutputKind.DELTA,
        )

    def _build_stage_config_file(self) -> str:
        stage_cfg = {
            # async_chunk enables the streaming-text feed (one subword per chunk);
            # it is a no-op for the whole-text path, so one engine serves both.
            "async_chunk": True,
            "stage_args": [
                {
                    "stage_id": 0,
                    "stage_type": "llm",
                    "is_comprehension": True,
                    "final_output": True,
                    "final_output_type": "audio",
                    "runtime": {"devices": "0"},
                    "engine_args": {
                        "model_stage": "easymagpie",
                        "max_num_seqs": self.max_num_seqs,
                        "model_arch": "EasyMagpieTTSForConditionalGeneration",
                        "worker_type": "ar",
                        # EasyMagpie-aware scheduler: forwards each chunk's text_token
                        # and the raised acoustic-tail max_tokens for streaming-text,
                        # and is a drop-in equivalent of the stock scheduler for
                        # whole-text.
                        "scheduler_cls": "easymagpie_vllm_omni.scheduler.EasyMagpieARAsyncScheduler",
                        "enforce_eager": False,
                        "trust_remote_code": True,
                        "async_scheduling": True,
                        "enable_prefix_caching": False,
                        "engine_output_type": "audio",
                        "gpu_memory_utilization": self.gpu_memory_utilization,
                        "distributed_executor_backend": "uni",
                        "max_num_batched_tokens": self.max_num_batched_tokens,
                        "max_model_len": self.max_model_len,
                        # bf16 overflows the Nemotron-H fused-MoE Triton kernel's
                        # fp32 shared memory; fp16 backbone + fp32 mamba cache.
                        "dtype": "float16",
                        "mamba_ssm_cache_dtype": "float32",
                        "attention_backend": "TRITON_ATTN",
                        # We feed prompt_token_ids directly; the model loads the
                        # bundled tokenizer to tokenize context_text + text.
                        "skip_tokenizer_init": True,
                    },
                    "default_sampling_params": {
                        # Backbone token sampler is a no-op (audio is sampled in the
                        # local transformer via additional_information temperature/top_k).
                        "temperature": 0.0,
                        "max_tokens": self.max_new_tokens,
                        "detokenize": False,
                        # Audio EOS lives in the codes; the model emits stop_token_id
                        # on the backbone stream at the EOS frame.
                        "ignore_eos": True,
                        "stop_token_ids": [self.stop_token_id],
                    },
                }
            ],
        }
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", prefix="easymp_triton_", delete=False)
        yaml.dump(stage_cfg, tmp, sort_keys=False)
        tmp.close()
        return tmp.name

    def _start_omni_engine(self):
        from vllm_omni import AsyncOmni

        self._stage_cfg_path = self._build_stage_config_file()
        self.omni = AsyncOmni(
            model=self.vllm_model_path,
            stage_configs_path=self._stage_cfg_path,
            log_stats=False,
            stage_init_timeout=300,
        )

    def _get_speaker_embedding(self, speaker: str) -> torch.Tensor:
        if speaker not in self._speaker_cache:
            emb_path = Path(self.vllm_model_path) / "speaker_embeddings" / f"{speaker}.pt"
            if not emb_path.exists():
                raise FileNotFoundError(f"Speaker embedding not found: {emb_path}")
            loaded = torch.load(emb_path, map_location="cpu")
            emb = loaded["speaker_encoding"] if isinstance(loaded, dict) else loaded
            self._speaker_cache[speaker] = emb.to(torch.float32)
        return self._speaker_cache[speaker]

    def _prompt_len(self, speaker_embedding: torch.Tensor, context_text: str) -> int:
        return int(
            self._estimate_prompt_len(
                speaker_embedding,
                tokenize=lambda t: self.tokenizer.encode(t),
                context_text=context_text,
                has_task_embedding=self.has_task_embedding,
            )
        )

    def _build_prompt(self, text: str, context_text: str, speaker: str) -> dict:
        speaker_embedding = self._get_speaker_embedding(speaker)
        prompt_len = self._prompt_len(speaker_embedding, context_text)
        return {
            "prompt_token_ids": [0] * prompt_len,
            "additional_information": {
                "speaker_embedding": speaker_embedding,
                "context_text": context_text,
                "text": text,
                "temperature": self.lt_temperature,
                "top_k": self.lt_top_k,
            },
        }

    def _decode_codec(self, codes: torch.Tensor, left_context_frames: int) -> np.ndarray:
        """Decode one ``(<=codec_chunk_size, C*S)`` window, trim left context + pad."""
        codes_np = codes.detach().cpu().to(torch.int64).numpy()
        pad = self.codec_chunk_size - codes_np.shape[0]
        if pad > 0:
            codes_np = np.pad(codes_np, ((0, pad), (0, 0)))

        response = pb_utils.InferenceRequest(
            model_name="codec",
            requested_output_names=["audio_values"],
            inputs=[pb_utils.Tensor("audio_codes", codes_np[np.newaxis])],
        ).exec()
        if response.has_error():
            raise RuntimeError(f"Codec decode failed: {response.error().message()}")

        audio_tensor = pb_utils.get_output_tensor_by_name(response, "audio_values")
        audio = (
            audio_tensor.as_numpy()
            if audio_tensor.is_cpu()
            else torch.from_dlpack(audio_tensor.to_dlpack()).cpu().numpy()
        )
        if audio.ndim > 1:
            audio = audio[0]

        if self._spf is None:
            self._spf = audio.shape[-1] // self.codec_chunk_size
        left = left_context_frames * self._spf
        right = pad * self._spf
        return audio[left:-right] if right > 0 else audio[left:]

    def _send_audio(self, response_sender, audio: np.ndarray, final: bool):
        response_sender.send(
            pb_utils.InferenceResponse(output_tensors=[pb_utils.Tensor("audio", audio.astype(np.float32))]),
            flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL if final else 0,
        )

    def _send_error(self, response_sender, err: Exception):
        try:
            response_sender.send(
                pb_utils.InferenceResponse(output_tensors=[], error=pb_utils.TritonError(str(err))),
                flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL,
            )
        except Exception:
            pass

    def _codec_worker(self, codec_q: queue.Queue, response_sender, state: dict) -> None:
        """Pop ``(chunk, ctx, is_final)`` tuples; ``None`` == send empty final + exit."""
        finalized = False
        try:
            while True:
                item = codec_q.get()
                if item is None:
                    self._send_audio(response_sender, np.array([], dtype=np.float32), final=True)
                    finalized = True
                    return
                chunk, ctx, is_final = item
                audio = self._decode_codec(chunk, ctx)
                self._send_audio(response_sender, audio, final=is_final)
                if state["t_first_audio"] is None:
                    state["t_first_audio"] = time.perf_counter()
                if is_final:
                    finalized = True
                    return
        except Exception as e:
            state["error"] = e
            if not finalized:
                self._send_error(response_sender, e)

    @staticmethod
    def _cumulative_codes(payload):
        """Reduce one step's ``audio_codes`` payload to the cumulative ``(T, C*S)``.

        DELTA decode surfaces a growing list ``[cum_so_far, new_frame, ...]``, but
        every finished segment consolidates to a single cumulative tensor — and with
        ``max_tokens=1`` (streaming-text) *each* fed token finishes its segment, so
        most steps arrive as a tensor, not a list. Both forms reduce to the full
        cumulative here (cat the list / take the tensor); callers keep the largest.
        """
        if isinstance(payload, list):
            parts = [t for t in payload if isinstance(t, torch.Tensor) and t.numel() > 0]
            return torch.cat(parts, dim=0) if parts else None
        if isinstance(payload, torch.Tensor) and payload.numel() > 0:
            return payload
        return None

    async def _drive_codec(
        self, gen, response_sender, request_id: str, speaker: str, text: str, prompt_len: int, pace_q=None
    ):
        """Drain one omni request's per-step ``audio_codes`` and stream audio out.

        ``gen`` is the ``AsyncOmni.generate(...)`` async iterator (whole-text or
        streaming-text); from here on both flavours are identical. All audio is sent
        on ``response_sender`` (for streaming that is the ``stream_start`` sender).

        Each step yields the *cumulative* codes ``(prompt_len prefill rows + decoded
        frames, C*S)`` (as a list to cat or an already-consolidated tensor); the first
        real audio frame is at row ``prompt_len + speech_delay`` and the last decoded
        row is the audio-EOS frame.

        For streaming-text, ``pace_q`` carries one token per observed decode-step
        output so the input feeder releases the next chunk only after the previous one
        has been decoded (see :class:`_StreamingSession`); ``None`` for whole-text.
        """
        t_start = time.perf_counter()

        codec_q: queue.Queue = queue.Queue()
        state: dict = {"t_first_audio": None, "error": None}
        codec_future = self._codec_pool.submit(self._codec_worker, codec_q, response_sender, state)

        # Codes are a cumulative ``(T, C*S)`` tensor: rows [0, prompt_len) are the
        # prefill prefix, the next ``speech_delay`` rows are warm-up, so the first
        # real audio frame is at ``head`` and the last decoded row is audio-EOS.
        L = self.codec_left_context
        head = prompt_len + self.speech_delay  # cumulative row of the first real frame
        sent = 0  # real frames already queued to the codec
        threshold = self.first_chunk_frames
        cum = None  # largest cumulative codes tensor seen
        produced_final = False

        def emit_ready(cum_codes, real_count: int, final: bool) -> None:
            """Queue overlap-save windows of newly-ready real frames (by cum row)."""
            nonlocal sent, threshold, produced_final
            while sent < real_count:
                remaining = real_count - sent
                if not final and remaining < threshold:
                    break
                take = min(threshold, remaining)
                ctx = min(sent, L)
                chunk = cum_codes[head + sent - ctx : head + sent + take]
                sent += take
                threshold = self.codec_chunk_size - L
                is_final = final and sent >= real_count
                codec_q.put((chunk, ctx, is_final))
                produced_final = produced_final or is_final

        step = 0
        try:
            async for out in gen:
                # Release the next input chunk for each decode-step output (one chunk
                # produces one frame); harmless extra puts during the acoustic tail.
                if pace_q is not None:
                    pace_q.put_nowait(True)
                if state["error"] is not None:
                    break
                step += 1
                payload = (getattr(out, "multimodal_output", None) or {}).get("audio_codes")
                cum_now = self._cumulative_codes(payload)
                if cum_now is None:
                    continue
                if cum is None or cum_now.shape[0] > cum.shape[0]:
                    cum = cum_now
                # Hold back the most recent decode row: the audio-EOS frame is always
                # the last one and must not be vocoded.
                real_avail = cum.shape[0] - head - 1
                if pace_q is not None:
                    logger.info(
                        "rid=%s STEP %d: got acoustic codes (cum_rows=%d, real_avail=%d, sent=%d) -> released pace",
                        request_id,
                        step,
                        cum.shape[0],
                        max(0, real_avail),
                        sent,
                    )
                if real_avail > sent:
                    before = sent
                    emit_ready(cum, real_avail, final=False)
                    if pace_q is not None and sent > before:
                        logger.info("rid=%s STEP %d: queued %d new frame(s) to codec", request_id, step, sent - before)

            if state["error"] is None and cum is not None:
                # Authoritative tail: scan for the audio-EOS row (only it carries
                # audio_eos_id > codebook_size) and vocode every real frame before it.
                eos_row = None
                for i in range(cum.shape[0] - 1, head - 1, -1):
                    if bool((cum[i] == self.audio_eos_id).any()):
                        eos_row = i
                        break
                last_excl = eos_row if eos_row is not None else cum.shape[0]
                real_count = max(0, last_excl - head)
                emit_ready(cum, real_count, final=True)
                if not produced_final:
                    codec_q.put(None)
            elif state["error"] is None:
                codec_q.put(None)

            await asyncio.wrap_future(codec_future)
            if state["error"] is not None:
                raise state["error"]

            t_end = time.perf_counter()
            ttfa_ms = ((state["t_first_audio"] or t_end) - t_start) * 1000
            logger.info(
                "rid=%s ttfa=%.1fms total=%.1fms frames=%d speaker=%s text=%r",
                request_id,
                ttfa_ms,
                (t_end - t_start) * 1000,
                sent,
                speaker,
                text[:120],
            )
        except Exception as e:
            logger.error("rid=%s failed: %s", request_id, e, exc_info=True)
            try:
                await self.omni.abort(request_id)
            except Exception:
                pass
            if not codec_future.done():
                codec_q.put(None)
                try:
                    await asyncio.wrap_future(codec_future)
                except Exception:
                    pass
            self._send_error(response_sender, e)
        finally:
            try:
                await gen.aclose()
            except Exception:
                pass

    async def _synthesize_whole_text(self, text: str, context_text: str, speaker: str, response_sender):
        request_id = f"easymp-{uuid.uuid4().hex[:8]}"
        prompt = self._build_prompt(text, context_text, speaker)
        prompt_len = len(prompt["prompt_token_ids"])
        gen = self.omni.generate(
            prompt,
            sampling_params_list=[self._make_sampling_params(self.max_new_tokens)],
            request_id=request_id,
        )
        await self._drive_codec(gen, response_sender, request_id, speaker, text, prompt_len)

    def _text_chunk(self, text_token: int, sampling_params):
        return self._StreamingInput(
            prompt={"prompt_token_ids": [0], "additional_information": {"text_token": int(text_token)}},
            sampling_params=sampling_params,
        )

    async def _stream_inputs(self, session: _StreamingSession):
        """Yield ``StreamingInput`` chunks for a streaming-text session.

        Prefill (speaker + context, no text), then one ``max_tokens=1`` chunk per
        subword id, then — once the client signals ``stream_end`` — the text-EOS id
        and a ``-1`` mask sentinel whose raised ``max_tokens`` lets the model free-run
        the acoustic tail to audio-EOS.

        Every post-prefill chunk is gated on ``session.pace_q`` (one decode-step
        output == one chunk released) so the eagerly-drained feed can't overwrite a
        not-yet-consumed ``text_token``; content tokens are additionally gated on the
        client actually having sent them.
        """
        StreamingInput = self._StreamingInput
        rid = session.request_id
        speaker_embedding = self._get_speaker_embedding(session.speaker)
        prompt_len = self._prompt_len(speaker_embedding, session.context_text)
        sp1 = self._make_sampling_params(1)

        prefill_info = {
            "speaker_embedding": speaker_embedding,
            "context_text": session.context_text,
            "temperature": self.lt_temperature,
            "top_k": self.lt_top_k,
        }
        # Prefill is released immediately; its decode-step output unblocks the first
        # text token (mirrors the demo's go_queue handshake).
        logger.info("rid=%s STREAM: releasing prefill (prompt_len=%d, speaker=%s)", rid, prompt_len, session.speaker)
        yield StreamingInput(
            prompt={"prompt_token_ids": [0] * prompt_len, "additional_information": prefill_info},
            sampling_params=sp1,
        )

        n_text = 0
        pending: list = []
        ended = False
        while True:
            # Refill from the client; block only while we have nothing buffered.
            while not pending and not ended:
                logger.info("rid=%s STREAM: waiting for client text tokens...", rid)
                item = await session.token_q.get()
                if item is _STREAM_END:
                    ended = True
                    logger.info("rid=%s STREAM: client signalled stream_end", rid)
                else:
                    pending.extend(int(t) for t in item)
                    logger.info("rid=%s STREAM: client sent tokens=%s (buffered=%d)", rid, list(item), len(pending))
            if not pending:
                break
            await session.pace_q.get()
            tok = pending.pop(0)
            logger.info(
                "rid=%s STREAM: feeding text_token=%d (n_text=%d, buffered_left=%d)", rid, tok, n_text, len(pending)
            )
            yield self._text_chunk(tok, sp1)
            n_text += 1

        await session.pace_q.get()
        logger.info("rid=%s STREAM: feeding text_eos=%d (n_text=%d)", rid, self.text_eos_id, n_text)
        yield self._text_chunk(self.text_eos_id, sp1)
        n_text += 1

        await session.pace_q.get()
        tail_budget = self.max_new_tokens - n_text
        logger.info("rid=%s STREAM: feeding mask sentinel text_token=-1 (acoustic tail budget=%d)", rid, tail_budget)
        tail_params = self._make_sampling_params(tail_budget)
        yield self._text_chunk(-1, tail_params)
        logger.info("rid=%s STREAM: input generator exhausted (fed %d text tokens incl. eos)", rid, n_text)

    async def _synthesize_streaming(self, session: _StreamingSession):
        prompt_len = self._prompt_len(self._get_speaker_embedding(session.speaker), session.context_text)
        inputs_gen = self._stream_inputs(session)
        gen = self.omni.generate(
            inputs_gen, sampling_params_list=[self._make_sampling_params(1)], request_id=session.request_id
        )
        try:
            await self._drive_codec(
                gen,
                session.response_sender,
                session.request_id,
                session.speaker,
                "<streaming>",
                prompt_len,
                pace_q=session.pace_q,
            )
        finally:
            try:
                await inputs_gen.aclose()
            except Exception:
                pass
            with self._sessions_lock:
                self._sessions.pop(session.stream_id, None)

    @staticmethod
    def _log_future_exception(future: concurrent.futures.Future) -> None:
        try:
            exc = future.exception()
        except concurrent.futures.CancelledError:
            return
        if exc is not None:
            logger.error("synthesis task crashed: %s", exc, exc_info=exc)

    @staticmethod
    def _read_str(request, name: str, default: str) -> str:
        tensor = pb_utils.get_input_tensor_by_name(request, name)
        if tensor is None:
            return default
        return tensor.as_numpy().flatten()[0].decode("utf-8")

    @staticmethod
    def _read_bool(request, name: str, default: bool) -> bool:
        tensor = pb_utils.get_input_tensor_by_name(request, name)
        if tensor is None:
            return default
        return bool(tensor.as_numpy().flatten()[0])

    @staticmethod
    def _read_int_list(request, name: str) -> list:
        tensor = pb_utils.get_input_tensor_by_name(request, name)
        if tensor is None:
            return []
        return [int(x) for x in tensor.as_numpy().flatten().tolist()]

    def _feed_session(self, session: _StreamingSession, item) -> None:
        """Hand ``item`` (a list of token ids or ``_STREAM_END``) to the session's
        queue on the event-loop thread (asyncio.Queue is not thread-safe)."""
        self._loop.call_soon_threadsafe(session.token_q.put_nowait, item)

    def _launch(self, coro) -> None:
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        future.add_done_callback(self._log_future_exception)

    def execute(self, requests):
        for request in requests:
            response_sender = request.get_response_sender()
            try:
                stream_id = self._read_str(request, "stream_id", "")
                if stream_id:
                    self._handle_streaming(request, stream_id, response_sender)
                else:
                    self._handle_whole_text(request, response_sender)
            except Exception as e:
                logger.error("Request parse failed: %s", e, exc_info=True)
                self._send_error(response_sender, e)
        return None

    def _handle_whole_text(self, request, response_sender) -> None:
        text = self._read_str(request, "text", "")
        context_text = self._read_str(request, "context_text", self.default_context_text)
        speaker = self._read_str(request, "speaker", self.default_speaker)
        self._launch(self._synthesize_whole_text(text, context_text, speaker, response_sender))

    def _handle_streaming(self, request, stream_id: str, response_sender) -> None:
        """Route one chunk of a streaming-text request.

        ``stream_start`` opens a session (launching one generate() call whose audio
        streams back on *this* response sender) and ``stream_end`` closes the text
        feed. Follow-up chunks only push tokens, then immediately complete their own
        (output-less) response so the client side of the stream stays tidy.
        """
        start = self._read_bool(request, "stream_start", False)
        end = self._read_bool(request, "stream_end", False)
        tokens = self._read_int_list(request, "text_token")
        logger.info(
            "CLIENT chunk: stream_id=%s start=%s end=%s tokens=%s", stream_id, start, end, tokens
        )

        if start:
            context_text = self._read_str(request, "context_text", self.default_context_text)
            speaker = self._read_str(request, "speaker", self.default_speaker)
            session = _StreamingSession(
                stream_id=stream_id,
                request_id=f"easymp-stream-{uuid.uuid4().hex[:8]}",
                speaker=speaker,
                context_text=context_text,
                response_sender=response_sender,
                token_q=asyncio.Queue(),
                pace_q=asyncio.Queue(),
            )
            with self._sessions_lock:
                self._sessions[stream_id] = session
            logger.info("rid=%s STREAM: opened session for stream_id=%s", session.request_id, stream_id)
            # All audio for the stream is sent on this (stream_start) sender by the
            # coroutine; do NOT complete it here.
            self._launch(self._synthesize_streaming(session))
            if tokens:
                self._feed_session(session, tokens)
            if end:
                self._feed_session(session, _STREAM_END)
            return

        with self._sessions_lock:
            session = self._sessions.get(stream_id)
        if session is not None:
            if tokens:
                self._feed_session(session, tokens)
            if end:
                self._feed_session(session, _STREAM_END)
        else:
            logger.warning("Streaming chunk for unknown/closed stream_id=%s", stream_id)
        # Follow-up chunks produce no audio of their own; close this response.
        response_sender.send(flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)

    def finalize(self):
        if hasattr(self, "omni"):
            try:
                self.omni.shutdown()
            except Exception:
                pass
        if hasattr(self, "_loop") and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if hasattr(self, "_loop_thread"):
            self._loop_thread.join(timeout=10)
        if hasattr(self, "_codec_pool"):
            self._codec_pool.shutdown(wait=False)
        if getattr(self, "_stage_cfg_path", None):
            try:
                os.unlink(self._stage_cfg_path)
            except OSError:
                pass
