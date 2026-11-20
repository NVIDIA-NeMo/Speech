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
"""vLLM multimodal input processing for EasyMagpie user speech."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import numpy as np
import torch
from easymagpie_vllm_omni.config import EasyMagpieOmniArch
from transformers import BatchFeature
from vllm.config.multimodal import BaseDummyOptions
from vllm.inputs import MultiModalDataDict
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import MultiModalDataItems, MultiModalDataParser
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    ProcessorInputs,
    PromptReplacement,
)

_AUDIO_OUTPUT_USER = 0
_AUDIO_OUTPUT_REFERENCE = 1
_AUDIO_OUTPUTS_KWARG = "_easymagpie_audio_outputs"


def _infer_audio_output_kinds(prompt: str | list[int], audio_count: int, marker_id: int) -> list[int]:
    """Classify audio items from their matching markers in the layout-only prompt.

    Each ``marker_id`` occurrence corresponds, in order, to one audio item.
    Non-audio rows use a different dummy token, so a non-final first marker is
    the reference and every remaining marker is user history.
    """
    if audio_count == 0:
        return []
    if not isinstance(prompt, list):
        raise ValueError("EasyMagpie raw audio requires prompt_token_ids so its layout can be inferred")
    marker_positions = [index for index, token_id in enumerate(prompt) if token_id == marker_id]
    if len(marker_positions) != audio_count:
        raise ValueError(
            f"EasyMagpie prompt contains {len(marker_positions)} audio markers for {audio_count} audio items"
        )
    output_kinds = [_AUDIO_OUTPUT_USER] * audio_count
    if marker_positions[0] < len(prompt) - 1:
        output_kinds[0] = _AUDIO_OUTPUT_REFERENCE
    return output_kinds


class EasyMagpieAudioParser(MultiModalDataParser):
    """Require explicit mono audio at the codec encoder input rate."""

    def __init__(self, sample_rate: int, expected_hidden_size: int | None) -> None:
        super().__init__(expected_hidden_size=expected_hidden_size)
        self.sample_rate = sample_rate

    def _get_audio_with_sr(self, audio):
        waveform, sample_rate = super()._get_audio_with_sr(audio)
        if sample_rate is None:
            raise ValueError(
                "EasyMagpie audio must be passed as a (mono_waveform, sample_rate) tuple so its format can be checked"
            )
        if int(sample_rate) != self.sample_rate:
            raise ValueError(f"EasyMagpie codec input must be {self.sample_rate} Hz; received {int(sample_rate)} Hz")
        waveform = np.asarray(waveform)
        if waveform.ndim != 1:
            raise ValueError(f"EasyMagpie codec input must be mono [samples]; received shape {waveform.shape}")
        return waveform, None


class EasyMagpieProcessingInfo(BaseProcessingInfo):
    """Describe raw-audio limits and codec-frame expansion to vLLM."""

    @property
    def arch(self) -> EasyMagpieOmniArch:
        return EasyMagpieOmniArch.from_hf_config(self.get_hf_config())

    def get_data_parser(self) -> MultiModalDataParser:
        return EasyMagpieAudioParser(
            sample_rate=self.arch.codec_input_sample_rate,
            expected_hidden_size=self._get_expected_hidden_size(),
        )

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        if not self.arch.codec_encoder_bundled:
            return {}
        count = 1 + int(self.arch.use_multiturn_dataset and self.arch.condition_on_user_speech)
        return {"audio": count}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        del seq_len, mm_counts
        return {"audio": self.get_max_audio_tokens()}

    def get_max_audio_samples(self) -> int:
        return math.ceil(self.arch.max_audio_seconds * self.arch.codec_input_sample_rate)

    def get_max_audio_tokens(self) -> int:
        max_samples = self.get_max_audio_samples()
        return max(
            self.arch.reference_audio_num_rows(max_samples),
            self.arch.user_audio_num_rows(max_samples),
        )


class EasyMagpieDummyInputsBuilder(BaseDummyInputsBuilder[EasyMagpieProcessingInfo]):
    """Build maximum-size raw audio for vLLM memory profiling."""

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        del mm_counts
        return ""

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict:
        del seq_len
        audio_overrides = mm_options.get("audio")
        audios = self._get_dummy_audios(
            length=self.info.get_max_audio_samples(),
            num_audios=mm_counts.get("audio", 0),
            overrides=audio_overrides,
        )
        return {"audio": [(audio, self.info.arch.codec_input_sample_rate) for audio in audios]}

    def get_dummy_processor_inputs(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> ProcessorInputs:
        dummy_mm_data = self.get_dummy_mm_data(seq_len, mm_counts, mm_options)
        audio_count = mm_counts.get("audio", 0)
        marker_id = self.info.arch.audio_input_token_id
        dummy_prompt = [] if audio_count == 0 else [marker_id, 0, *([marker_id] * (audio_count - 1))]
        return ProcessorInputs(
            prompt=dummy_prompt,
            mm_data_items=self.info.parse_mm_data(dummy_mm_data, validate=False),
            tokenization_kwargs={"truncation": False},
        )


class EasyMagpieMultiModalProcessor(BaseMultiModalProcessor[EasyMagpieProcessingInfo]):
    """Pass normalized waveforms through and expand their prompt placeholder."""

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        tokenizer = self.info.get_tokenizer()
        input_ids = tokenizer.encode(prompt, add_special_tokens=bool(tok_kwargs.get("add_special_tokens", False)))

        audios = mm_data.get("audios", [])
        audio_values: list[torch.Tensor] = []
        for audio in audios if isinstance(audios, (list, tuple)) else [audios]:
            value = torch.as_tensor(np.asarray(audio), dtype=torch.float32)
            if value.ndim != 1 or value.numel() == 0:
                raise ValueError(f"EasyMagpie raw audio must be a non-empty mono waveform, got {tuple(value.shape)}")
            if value.numel() > self.info.get_max_audio_samples():
                raise ValueError(
                    f"EasyMagpie raw audio has {value.numel()} samples; the configured maximum is "
                    f"{self.info.get_max_audio_samples()} "
                    f"({self.info.arch.max_audio_seconds:g} seconds at "
                    f"{self.info.arch.codec_input_sample_rate} Hz)"
                )
            audio_values.append(value.contiguous())

        raw_output_kinds = mm_kwargs.get(_AUDIO_OUTPUTS_KWARG, [_AUDIO_OUTPUT_REFERENCE] * len(audio_values))
        output_kinds = [int(kind) for kind in raw_output_kinds]
        if len(output_kinds) != len(audio_values):
            raise ValueError(f"Got {len(output_kinds)} audio outputs for {len(audio_values)} audio items")
        if any(kind not in {_AUDIO_OUTPUT_USER, _AUDIO_OUTPUT_REFERENCE} for kind in output_kinds):
            raise ValueError("EasyMagpie received an invalid internal audio output kind")
        if _AUDIO_OUTPUT_REFERENCE in output_kinds:
            self.info.arch.ensure_reference_audio_available()
        if _AUDIO_OUTPUT_USER in output_kinds:
            self.info.arch.require_user_audio_prefill()

        return BatchFeature(
            {
                "input_ids": [input_ids],
                "audio_values": audio_values,
                "audio_lens": torch.tensor([audio.numel() for audio in audio_values], dtype=torch.long),
                "audio_output_kinds": torch.tensor(output_kinds, dtype=torch.long),
            }
        )

    def _cached_apply_hf_processor(self, inputs, timing_ctx):
        processor_kwargs = dict(inputs.hf_processor_mm_kwargs)
        if _AUDIO_OUTPUTS_KWARG in processor_kwargs:
            raise ValueError(f"{_AUDIO_OUTPUTS_KWARG} is reserved for EasyMagpie's processor")
        audio_count = inputs.mm_data_items["audio"].get_count() if "audio" in inputs.mm_data_items else 0
        output_kinds = _infer_audio_output_kinds(
            inputs.prompt,
            audio_count,
            self.info.arch.audio_input_token_id,
        )
        processor_kwargs[_AUDIO_OUTPUTS_KWARG] = output_kinds
        updated_inputs = replace(inputs, hf_processor_mm_kwargs=processor_kwargs)
        return self._apply_hf_processor(updated_inputs, timing_ctx)

    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> bool:
        del prompt_text, mm_items, hf_processor_mm_kwargs, tokenization_kwargs
        return False

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        del hf_inputs, hf_processor_mm_kwargs
        return {
            "audio_values": MultiModalFieldConfig.batched("audio"),
            "audio_lens": MultiModalFieldConfig.batched("audio"),
            "audio_output_kinds": MultiModalFieldConfig.batched("audio"),
        }

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptReplacement]:
        del out_mm_kwargs
        audio_items = mm_items["audio"]
        raw_output_kinds = hf_processor_mm_kwargs.get(
            _AUDIO_OUTPUTS_KWARG,
            [_AUDIO_OUTPUT_REFERENCE] * audio_items.get_count(),
        )
        output_kinds = [int(kind) for kind in raw_output_kinds]
        if len(output_kinds) != audio_items.get_count():
            raise ValueError(f"Got {len(output_kinds)} audio outputs for {audio_items.get_count()} audio items")
        if _AUDIO_OUTPUT_REFERENCE in output_kinds:
            self.info.arch.ensure_reference_audio_available()
        if _AUDIO_OUTPUT_USER in output_kinds:
            self.info.arch.require_user_audio_prefill()

        def get_replacement(item_idx: int) -> list[int]:
            audio_len = audio_items.get_audio_length(item_idx)
            if output_kinds[item_idx] == _AUDIO_OUTPUT_REFERENCE:
                num_rows = self.info.arch.reference_audio_num_rows(audio_len)
            else:
                min_samples = self.info.arch.streaming_speech_delay * self.info.arch.codec_samples_per_row
                num_rows = self.info.arch.user_audio_num_rows(max(audio_len, min_samples))
            return [self.info.arch.audio_input_token_id] * num_rows

        return [
            PromptReplacement(
                modality="audio",
                target=[self.info.arch.audio_input_token_id],
                replacement=get_replacement,
            )
        ]
