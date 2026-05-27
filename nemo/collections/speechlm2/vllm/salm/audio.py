# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Audio-side plumbing for the NeMo Speech LM (SALM) vLLM plugin.

All audio handling lives here: helpers (perception loader, tokenizer special-token
patcher, vocab-size padder), audio constants and TensorSchema, and the trio of
classes that bind to vLLM's multimodal registry to drive prompt expansion and
dummy-input generation. Backbone-agnostic; shared by both transformer and
hybrid backends.

Public surface used by the rest of the package:

* ``_AUDIO_PLACEHOLDER`` -- the audio locator string vLLM emits during prompt
  rendering and the processor expands inline.
* ``_load_nemo_perception``, ``_ensure_special_tokens``, ``_pad_to_vocab_size``
  -- small helpers reused at model init and weight load time.
* ``NeMoSpeechLMAudioInputs`` -- vLLM ``TensorSchema`` describing the parsed
  audio tensors that flow into ``embed_multimodal``.
* ``NeMoSpeechLMProcessingInfo`` / ``NeMoSpeechLMMultiModalProcessor`` /
  ``NeMoSpeechLMDummyInputsBuilder`` -- the trio that vLLM's multimodal
  registry binds to the registered model class.
"""

import re
from collections.abc import Mapping
from typing import Annotated, Literal

import torch
from torch import nn
from transformers import BatchFeature, PreTrainedTokenizerBase
from vllm.config.multimodal import BaseDummyOptions

try:
    from vllm.inputs import MultiModalDataDict
except ImportError:
    from vllm.multimodal.inputs import MultiModalDataDict

from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import AudioProcessorItems, MultiModalDataItems, MultiModalDataParser
from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.multimodal.processing.dummy_inputs import BaseDummyInputsBuilder
from vllm.utils.tensor_schema import TensorSchema, TensorShape

from nemo.collections.speechlm2.vllm.salm.config import _AUDIO_PLACEHOLDER

_SAMPLING_RATE = 16000
_AUDIO_CHANNELS = 1
_DUMMY_AUDIO_DURATION_S = 40.0


# ── Helpers ─────────────────────────────────────────────────────────


def _ensure_special_tokens(tokenizer: PreTrainedTokenizerBase) -> None:
    special = [_AUDIO_PLACEHOLDER]
    existing = set(tokenizer.get_vocab().keys())
    to_add = [t for t in special if t not in existing]
    if to_add:
        tokenizer.add_special_tokens({"additional_special_tokens": to_add})


def _load_nemo_perception(perception_cfg: dict) -> nn.Module:
    try:
        from omegaconf import DictConfig

        from nemo.collections.speechlm2.modules import AudioPerceptionModule
    except ImportError as e:
        raise ImportError(
            "NeMo is required for the audio encoder. " "Install with: pip install nemo_toolkit[asr]"
        ) from e

    cfg = DictConfig(perception_cfg)
    perception = AudioPerceptionModule(cfg)
    perception.eval()
    return perception


def _pad_to_vocab_size(tensor: torch.Tensor, target_vocab: int) -> torch.Tensor:
    if tensor.shape[0] < target_vocab:
        pad = torch.zeros(
            target_vocab - tensor.shape[0],
            *tensor.shape[1:],
            dtype=tensor.dtype,
        )
        tensor = torch.cat([tensor, pad], dim=0)
    return tensor


# ── Multimodal contract types ───────────────────────────────────────


class NeMoSpeechLMAudioInputs(TensorSchema):
    type: Literal["audio_features"] = "audio_features"
    audio_signal: Annotated[torch.Tensor | list[torch.Tensor], TensorShape("b", "t")]
    audio_signal_length: Annotated[torch.Tensor, TensorShape("b")]


class NeMoSpeechLMProcessingInfo(BaseProcessingInfo):

    def get_data_parser(self) -> MultiModalDataParser:
        return MultiModalDataParser(
            target_sr=_SAMPLING_RATE,
            target_channels=_AUDIO_CHANNELS,
            expected_hidden_size=self._get_expected_hidden_size(),
        )

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": 1}

    @staticmethod
    def _estimate_audio_tokens(audio_length_samples: int) -> int:
        """Predict the encoder's output frame count for an audio of N samples.

        Mirrors the FastConformer preprocessing chain used by
        ``AudioPerceptionModule``: STFT (n_fft=512, hop_length=160) followed
        by 3x Conv(kernel=3, stride=2) subsampling. Implemented as pure
        Python integer math instead of calling NeMo's ``calc_length`` so
        the scheduler hotpath avoids ~90x tensor-op overhead (measured
        0.18 us vs 16 us per call). If the encoder's downsampling stack
        ever changes upstream, the unit test at
        ``tests/collections/speechlm2/test_vllm_audio_token_estimator.py``
        compares this function against ``calc_length`` on a canonical set
        of lengths and will fail, forcing a rewrite here.
        """
        n_fft = 512
        hop_length = 160
        stft_pad = n_fft // 2
        fbank_len = (audio_length_samples + 2 * stft_pad - n_fft) // hop_length
        kernel, stride, repeat = 3, 2, 3
        add_pad = 1 + 1 - kernel
        length = float(fbank_len)
        for _ in range(repeat):
            length = (length + add_pad) / stride + 1.0
        return max(1, int(length))


class NeMoSpeechLMMultiModalProcessor(
    BaseMultiModalProcessor[NeMoSpeechLMProcessingInfo],
):

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            audio_signal=MultiModalFieldConfig.batched("audio"),
            audio_signal_length=MultiModalFieldConfig.batched("audio"),
        )

    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> bool:
        return False

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> list[PromptUpdate]:
        audios = mm_items.get_items("audio", AudioProcessorItems)

        def get_replacement(item_idx: int):
            audio = audios.get(item_idx)
            n_tokens = self.info._estimate_audio_tokens(audio.shape[-1])
            repl_full = _AUDIO_PLACEHOLDER * n_tokens
            return PromptUpdateDetails.select_text(repl_full, _AUDIO_PLACEHOLDER)

        return [
            PromptReplacement(
                modality="audio",
                target=_AUDIO_PLACEHOLDER,
                replacement=get_replacement,
            )
        ]

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        tokenizer = self.info.get_tokenizer()
        _ensure_special_tokens(tokenizer)
        mm_data = dict(mm_data)
        audios = mm_data.pop("audios", [])

        if audios:
            audio_list: list[torch.Tensor] = []
            audio_lengths: list[int] = []
            parts = re.split(f"({re.escape(_AUDIO_PLACEHOLDER)})", prompt)
            # One placeholder is overwritten with one audio's encoder output
            # at forward time (positional pairing); counts must match or the
            # merge step in get_input_embeddings crashes / silently drops.
            ph_positions = [i for i, p in enumerate(parts) if p == _AUDIO_PLACEHOLDER]
            if len(ph_positions) != len(audios):
                raise ValueError(
                    f"Prompt has {len(ph_positions)} "
                    f"{_AUDIO_PLACEHOLDER!r} placeholders but "
                    f"{len(audios)} audios were provided; counts must match."
                )
            for i, audio in zip(ph_positions, audios):
                audio_tensor = (
                    audio if isinstance(audio, torch.Tensor) else torch.as_tensor(audio, dtype=torch.float32)
                )
                if audio_tensor.dim() > 1:
                    audio_tensor = audio_tensor.squeeze()
                n_tokens = self.info._estimate_audio_tokens(audio_tensor.shape[-1])
                parts[i] = _AUDIO_PLACEHOLDER * n_tokens
                audio_list.append(audio_tensor)
                audio_lengths.append(audio_tensor.shape[-1])

            prompt = "".join(parts)

        prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
        result = BatchFeature(dict(input_ids=[prompt_ids]), tensor_type="pt")

        if audios:
            result["audio_signal"] = audio_list
            result["audio_signal_length"] = torch.tensor(audio_lengths)
        return result


class NeMoSpeechLMDummyInputsBuilder(
    BaseDummyInputsBuilder[NeMoSpeechLMProcessingInfo],
):

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict:
        num_audios = mm_counts.get("audio", 0)
        dummy_audio_len = int(_DUMMY_AUDIO_DURATION_S * _SAMPLING_RATE)
        return {
            "audio": self._get_dummy_audios(
                length=dummy_audio_len,
                num_audios=num_audios,
            )
        }

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_audios = mm_counts.get("audio", 0)
        return "Transcribe the following: " + _AUDIO_PLACEHOLDER * num_audios
