# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
"""``/v1/audio/speech`` support for EasyMagpieTTS on vLLM-Omni 0.24+."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Callable

from easymagpie_vllm_omni.config import EasyMagpieOmniArch
from easymagpie_vllm_omni.tokenizer import EasyMagpieTextTokenizer

logger = logging.getLogger(__name__)

MODEL_TYPE = "easymagpie"
_TALKER_STAGE = "easymagpie"
_TALKER_ARCH = "EasyMagpieTTSForConditionalGeneration"
_SERVING_MODULE = "vllm_omni.entrypoints.openai.serving_speech"

_DEFAULT_SPEAKER = "eng"
_DEFAULT_CONTEXT_TEXT = "[EN]"
_DEFAULT_TEMPERATURE = 0.7
_DEFAULT_TOP_K = 80

# Legacy checkpoints use the last-but-one text-vocab row for EOS. Converted
# multiturn checkpoints pin the actual ID explicitly in ``config.json``.
_TEXT_EOS_OFFSET_FROM_VOCAB = 2

if TYPE_CHECKING:
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest


@dataclass(frozen=True)
class EasyMagpieStreamingSpec:
    """Model metadata needed by the incremental WebSocket input stream."""

    prefill_prompt: dict[str, Any]
    tokenizer: Any
    text_eos_id: int
    sample_rate: int
    text_prefill_num: int


def _build_adapter_cls() -> type:
    """Define the adapter class lazily (needs vllm_omni imported first)."""
    from vllm_omni.entrypoints.openai.tts_adapters.base import ARTTSAdapter, PreparedRequest

    class EasyMagpieTTSAdapter(ARTTSAdapter):
        """Build speaker-conditioned prompts for ``/v1/audio/speech`` requests."""

        name = MODEL_TYPE
        stage_keys = frozenset({_TALKER_STAGE})

        def __init__(self, ctx: Any) -> None:
            super().__init__(ctx)
            self._tokenizer: Any = None
            self._model_path_cache: str | None = None
            self._prompt_len_cache: dict[str, int] = {}
            self._model_config_cache: dict[str, Any] | None = None
            self._arch_cache: EasyMagpieOmniArch | None = None

        def _model_path(self) -> str:
            if self._model_path_cache is not None:
                return self._model_path_cache
            engine_client = getattr(self.ctx, "engine_client", None)
            model_config = getattr(engine_client, "model_config", None)
            path = getattr(model_config, "model", None)
            if not path:
                for stage in getattr(engine_client, "stage_configs", None) or []:
                    stage_path = getattr(getattr(stage, "engine_args", None), "model", None)
                    if stage_path:
                        path = stage_path
                        break
            if not path:
                raise RuntimeError("EasyMagpie serving adapter could not resolve the model path.")
            self._model_path_cache = path
            return path

        def _tokenize(self) -> Callable[[str], Any]:
            if self._tokenizer is None:
                self._tokenizer = EasyMagpieTextTokenizer.from_pretrained(self._model_path())
            return self._tokenizer.encode_context

        def _model_tokenizer(self):
            self._tokenize()
            return self._tokenizer

        def _prompt_len(self, speaker_id: str) -> int:
            cached = self._prompt_len_cache.get(speaker_id)
            if cached is not None:
                return cached
            from easymagpie_vllm_omni.easymagpie import EasyMagpieTTSForConditionalGeneration

            plen = int(
                EasyMagpieTTSForConditionalGeneration.get_prompt_len(
                    speaker_id, self._model_path(), tokenize=self._tokenize()
                )
            )
            self._prompt_len_cache[speaker_id] = plen
            return plen

        def _model_config(self) -> dict[str, Any]:
            if self._model_config_cache is None:
                path = Path(self._model_path()) / "config.json"
                self._model_config_cache = json.loads(path.read_text())
            return self._model_config_cache

        def _arch(self) -> EasyMagpieOmniArch:
            if self._arch_cache is None:
                self._arch_cache = EasyMagpieOmniArch.from_hf_config(SimpleNamespace(**self._model_config()))
            return self._arch_cache

        def _text_stream_metadata(self) -> tuple[int, int]:
            config = self._model_config()
            text_vocab_size = int(config.get("text_vocab_size", config.get("vocab_size", 0)))
            if text_vocab_size <= _TEXT_EOS_OFFSET_FROM_VOCAB:
                raise ValueError("EasyMagpie config must define text_vocab_size")
            configured_text_eos_id = config.get("text_eos_id")
            text_eos_id = (
                text_vocab_size - _TEXT_EOS_OFFSET_FROM_VOCAB
                if configured_text_eos_id is None
                else int(configured_text_eos_id)
            )
            return text_eos_id, self._arch().text_prefill_num

        def validate(self, request: OpenAICreateSpeechRequest) -> str | None:
            if not request.input or not request.input.strip():
                return "Input text cannot be empty"
            extra = request.extra_params
            if extra is not None and not isinstance(extra, dict):
                return "extra_params must be a JSON object/dict"
            ref_audio = request.ref_audio
            if ref_audio is None:
                return None
            if request.voice and request.voice.strip():
                return "'voice' and 'ref_audio' are mutually exclusive; provide a known voice or raw reference audio"
            if not isinstance(ref_audio, str):
                return "EasyMagpie supports a single reference in 'ref_audio', not a list"
            format_error = self.ctx.server._validate_ref_audio_format(ref_audio)
            if format_error:
                return format_error
            try:
                self._arch().ensure_reference_audio_available()
            except RuntimeError as error:
                return str(error)
            return None

        async def build(
            self,
            request: OpenAICreateSpeechRequest,
            sampling_params_list: list,
            has_inline_ref_audio: bool,
        ) -> "PreparedRequest":
            del sampling_params_list, has_inline_ref_audio
            extra = request.extra_params or {}
            text_eos_id, text_prefill_num = self._text_stream_metadata()
            text_tokens = self._model_tokenizer().encode(request.input, add_special_tokens=False)
            text_tokens.append(text_eos_id)

            prompt = {
                "additional_information": {
                    "context_text": extra.get("context_text", _DEFAULT_CONTEXT_TEXT),
                    "text_tokens": text_tokens,
                    "prefill_text_tokens": text_tokens[:text_prefill_num],
                    "text_prefill_num": text_prefill_num,
                    "temperature": float(extra.get("temperature", _DEFAULT_TEMPERATURE)),
                    "top_k": int(extra.get("top_k", _DEFAULT_TOP_K)),
                },
            }
            ref_audio = request.ref_audio
            if ref_audio is None:
                speaker_id = (request.voice or _DEFAULT_SPEAKER).strip()
                prompt["prompt_token_ids"] = [0] * (self._prompt_len(speaker_id) + text_prefill_num)
                prompt["additional_information"]["speaker_id"] = speaker_id
                return PreparedRequest(prompt=prompt, tts_params={}, model_type=MODEL_TYPE)

            arch = self._arch()
            arch.ensure_reference_audio_available()
            waveform, sample_rate = await self.ctx.server._resolve_ref_audio(ref_audio)
            if sample_rate != arch.codec_input_sample_rate:
                raise ValueError(
                    f"EasyMagpie reference audio must be {arch.codec_input_sample_rate} Hz; received {sample_rate} Hz"
                )
            context_text = prompt["additional_information"]["context_text"]
            context_len = len(self._tokenize()(context_text))
            task_len = int(arch.num_task_embeddings > 0)
            prompt["prompt_token_ids"] = (
                [0] * task_len + [arch.audio_input_token_id] + [0] * context_len + [0] * text_prefill_num
            )
            prompt["multi_modal_data"] = {"audio": [(waveform, sample_rate)]}
            return PreparedRequest(prompt=prompt, tts_params={}, model_type=MODEL_TYPE)

        def build_streaming_spec(self, request: OpenAICreateSpeechRequest) -> EasyMagpieStreamingSpec:
            """Build the speaker prefill and tokenizer metadata without complete text."""
            speaker_id = (request.voice or _DEFAULT_SPEAKER).strip()
            model_path = Path(self._model_path())
            text_eos_id, text_prefill_num = self._text_stream_metadata()

            sample_rate = 22050
            codec_config_path = model_path / "codec_native" / "config.json"
            if codec_config_path.exists():
                codec_config = json.loads(codec_config_path.read_text())
                sample_rate = int(codec_config.get("output_sample_rate", sample_rate))

            return EasyMagpieStreamingSpec(
                prefill_prompt={
                    "prompt_token_ids": [0] * (self._prompt_len(speaker_id) + text_prefill_num),
                    "additional_information": {
                        "context_text": _DEFAULT_CONTEXT_TEXT,
                        "temperature": _DEFAULT_TEMPERATURE,
                        "top_k": _DEFAULT_TOP_K,
                        "speaker_id": speaker_id,
                        "text_prefill_num": text_prefill_num,
                    },
                },
                tokenizer=self._model_tokenizer(),
                text_eos_id=text_eos_id,
                sample_rate=sample_rate,
                text_prefill_num=text_prefill_num,
            )

    return EasyMagpieTTSAdapter


def _register_adapter() -> None:
    from vllm_omni.entrypoints.openai import tts_adapters

    if MODEL_TYPE in tts_adapters.TTS_ADAPTER_REGISTRY:
        return
    tts_adapters.register_tts_adapter(_build_adapter_cls())


def _patch_detection() -> None:
    from vllm_omni.entrypoints.openai import serving_speech as ss

    ss._TTS_MODEL_STAGES.add(_TALKER_STAGE)

    detect = ss.OmniOpenAIServingSpeech._detect_tts_model_type
    if getattr(detect, "_easymagpie_patched", False):
        return
    _orig_detect = detect

    def _detect_tts_model_type(self):
        stage = getattr(self, "_tts_stage", None)
        if stage is not None:
            engine_args = getattr(stage, "engine_args", None)
            model_stage = getattr(engine_args, "model_stage", None)
            model_arch = getattr(engine_args, "model_arch", None)
            if model_stage == _TALKER_STAGE or model_arch == _TALKER_ARCH:
                return MODEL_TYPE
        return _orig_detect(self)

    _detect_tts_model_type._easymagpie_patched = True
    ss.OmniOpenAIServingSpeech._detect_tts_model_type = _detect_tts_model_type


def _patch_streaming_handler() -> None:
    """Select the EasyMagpie-aware handler when API state is initialized."""
    from easymagpie_vllm_omni.serving_stream import EasyMagpieStreamingSpeechHandler
    from vllm_omni.entrypoints.openai import api_server

    api_server.OmniStreamingSpeechHandler = EasyMagpieStreamingSpeechHandler


def apply_serving_patches(force: bool = False) -> None:
    """Install speech support in the API-server process."""
    import sys

    if not force and _SERVING_MODULE not in sys.modules:
        return
    try:
        _patch_detection()
        _register_adapter()
        _patch_streaming_handler()
        logger.info(
            "EasyMagpie: /v1/audio/speech and /v1/audio/speech/stream serving registered (model_type=%r).",
            MODEL_TYPE,
        )
    except Exception:  # never let a serving-layer change break model/pipeline loading
        logger.exception("EasyMagpie: failed to install /v1/audio/speech serving support.")
