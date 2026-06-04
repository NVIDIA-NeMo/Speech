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

Pipeline:
  1. Build ``additional_information`` from ``{speaker_embedding, context_text, text,
     temperature, top_k}`` and a placeholder ``prompt_token_ids`` of length
     ``estimate_prompt_len(...)``.
  2. Submit one request to ``AsyncOmni.generate()``. Each step yields the
     *cumulative* ``audio_codes`` tensor ``(T_total, C*S)`` (prefill rows + one row
     per decode step) and cumulative backbone ``token_ids``; we slice the decoded
     rows, drop the leading ``speech_delay`` warm-up frames, and stop at the audio
     EOS frame.
  3. New frames are streamed out in fixed ``codec_chunk_size``-frame windows (with a
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
        self._speaker_cache: dict = {}
        # Inferred from the first codec decode (audio_len / codec_chunk_size).
        self._spf: int | None = None

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

        self.tokenizer = AutoTokenizer.from_pretrained(self.vllm_model_path, trust_remote_code=True)
        self._estimate_prompt_len = EasyMagpieTTSForConditionalGeneration.estimate_prompt_len

    def _build_stage_config_file(self) -> str:
        stage_cfg = {
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
                        "scheduler_cls": "vllm_omni.core.sched.omni_ar_scheduler.OmniARAsyncScheduler",
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

    def _build_prompt(self, text: str, context_text: str, speaker: str) -> dict:
        speaker_embedding = self._get_speaker_embedding(speaker)
        prompt_len = self._estimate_prompt_len(
            speaker_embedding,
            tokenize=lambda t: self.tokenizer.encode(t),
            context_text=context_text,
            has_task_embedding=self.has_task_embedding,
        )
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

    async def _synthesize(self, text: str, context_text: str, speaker: str, response_sender):
        t_start = time.perf_counter()
        request_id = f"easymp-{uuid.uuid4().hex[:8]}"
        prompt = self._build_prompt(text, context_text, speaker)
        prompt_len = len(prompt["prompt_token_ids"])

        codec_q: queue.Queue = queue.Queue()
        state: dict = {"t_first_audio": None, "error": None}
        codec_future = self._codec_pool.submit(self._codec_worker, codec_q, response_sender, state)

        L = self.codec_left_context
        sent = 0  # real frames (post speech-delay, pre EOS) already queued
        threshold = self.first_chunk_frames
        real: torch.Tensor | None = None
        real_count = 0
        eos_found = False

        try:
            async for out in self.omni.generate(prompt, request_id=request_id):
                if state["error"] is not None:
                    break
                mm = getattr(out, "multimodal_output", None) or {}
                audio_codes = mm.get("audio_codes")
                if not isinstance(audio_codes, torch.Tensor):
                    continue
                # audio_codes accumulates one row per flat-batch token: prompt_len
                # prefill rows + one per decode step. Count decoded frames from the
                # tensor (token_ids on a streaming step is a delta, not cumulative).
                num_decoded = audio_codes.shape[0] - prompt_len
                if num_decoded <= self.speech_delay:
                    continue

                # Decoded rows are everything after prefill; drop the leading
                # speech-delay warm-up frames.
                real = audio_codes[prompt_len + self.speech_delay :]
                eos_rows = (real == self.audio_eos_id).any(dim=1).nonzero()
                if eos_rows.numel() > 0:
                    real_count = int(eos_rows[0].item())  # exclude the EOS frame
                    eos_found = True
                else:
                    real_count = real.shape[0]

                while real_count - sent >= threshold:
                    ctx = min(sent, L)
                    chunk = real[sent - ctx : sent + threshold]
                    codec_q.put((chunk, ctx, False))
                    sent += threshold
                    threshold = self.codec_chunk_size - L
                if eos_found:
                    break

            if state["error"] is None:
                if real is not None and real_count > sent:
                    ctx = min(sent, L)
                    codec_q.put((real[sent - ctx : real_count], ctx, True))
                else:
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

    @staticmethod
    def _read_str(request, name: str, default: str) -> str:
        tensor = pb_utils.get_input_tensor_by_name(request, name)
        if tensor is None:
            return default
        return tensor.as_numpy().flatten()[0].decode("utf-8")

    def execute(self, requests):
        for request in requests:
            response_sender = request.get_response_sender()
            try:
                text = self._read_str(request, "text", "")
                context_text = self._read_str(request, "context_text", self.default_context_text)
                speaker = self._read_str(request, "speaker", self.default_speaker)
                asyncio.run_coroutine_threadsafe(
                    self._synthesize(text, context_text, speaker, response_sender),
                    self._loop,
                )
            except Exception as e:
                logger.error("Request parse failed: %s", e, exc_info=True)
                self._send_error(response_sender, e)
        return None

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
