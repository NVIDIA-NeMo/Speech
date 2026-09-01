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

"""Model-scoped vLLM adapter for SpeechLM's training-time prompt contract.

Audio remains registered through vLLM's multimodal parser while the rendered
user content matches ``SALMDataset``: text is followed by one ASCII space and
the final ``<|audio|>`` locator.
"""

from __future__ import annotations

from collections.abc import Iterable
from functools import wraps
from typing import Any

_AUDIO_LOCATOR = "<|audio|>"
_AUDIO_PART_TYPES = frozenset({"audio_url", "input_audio", "audio_embeds"})
_TEXT_PART_TYPES = frozenset({"text", "input_text", "output_text"})


def _is_nemo_speechlm_tracker(mm_tracker: Any) -> bool:
    """Return whether a vLLM chat tracker belongs to a SpeechLM model."""
    model_config = getattr(mm_tracker, "model_config", None)
    hf_config = getattr(model_config, "hf_config", None)
    return getattr(hf_config, "model_type", None) in {
        "nemo_speechlm",
        "nemo_speechlm_mtp",
    }


def _part_type(part: Any) -> str:
    if isinstance(part, str):
        return "text"
    if not isinstance(part, dict):
        raise TypeError(
            "NeMo SpeechLM exact prompt rendering requires string or mapping "
            f"content parts; got {type(part).__name__}."
        )
    part_type = part.get("type")
    if part_type is None:
        direct_audio_types = [kind for kind in _AUDIO_PART_TYPES if kind in part]
        if len(direct_audio_types) == 1:
            return direct_audio_types[0]
    if not isinstance(part_type, str):
        raise TypeError(
            "NeMo SpeechLM exact prompt rendering requires every mapping "
            "content part to have an unambiguous string 'type'."
        )
    return part_type


def _text_value(part: Any, part_type: str) -> str:
    if isinstance(part, str):
        return part
    value = part.get("text")
    if not isinstance(value, str):
        raise TypeError(
            "NeMo SpeechLM exact prompt rendering requires textual content " f"for part type {part_type!r}."
        )
    return value


def _canonicalize_speechlm_parts(parts: Iterable[Any]) -> list[Any]:
    """Retain one audio payload but render ``text <|audio|>``."""
    parts = list(parts)
    audio_parts: list[Any] = []
    text_values: list[str] = []
    for part in parts:
        part_type = _part_type(part)
        if part_type in _AUDIO_PART_TYPES:
            audio_parts.append(part)
        elif part_type in _TEXT_PART_TYPES:
            text_values.append(_text_value(part, part_type))
        else:
            raise ValueError(
                "NeMo SpeechLM exact prompt rendering only supports text and "
                f"audio content parts; got {part_type!r}."
            )

    if not audio_parts:
        return parts
    if len(audio_parts) != 1:
        raise ValueError(
            "NeMo SpeechLM exact prompt rendering supports exactly one audio "
            f"payload per user message; got {len(audio_parts)}."
        )

    text = " ".join(text_values)
    locator_count = text.count(_AUDIO_LOCATOR)
    if locator_count > 1:
        raise ValueError("NeMo SpeechLM prompt contains more than one <|audio|> locator for " "one audio payload.")
    if locator_count == 1:
        expected = (
            _AUDIO_LOCATOR
            if text == _AUDIO_LOCATOR
            else f"{text.removesuffix(_AUDIO_LOCATOR).rstrip()} {_AUDIO_LOCATOR}"
        )
        if text != expected:
            raise ValueError(
                "A pre-existing NeMo SpeechLM <|audio|> locator must be final "
                "and be preceded by exactly one ASCII space."
            )
        canonical_text = text
    elif text:
        canonical_text = f"{text} {_AUDIO_LOCATOR}"
    else:
        canonical_text = _AUDIO_LOCATOR

    # The media item is still parsed and registered. Its explicit locator in
    # the final text consumes the registered placeholder, preventing vLLM from
    # prepending a missing locator.
    return [audio_parts[0], {"type": "text", "text": canonical_text}]


def _patch_speechlm_training_prompt_contract(chat_utils: Any) -> None:
    """Install an idempotent, SpeechLM-only exact prompt adapter."""
    original = getattr(chat_utils, "_parse_chat_message_content_parts", None)
    if not callable(original):
        raise TypeError(
            "Unsupported vLLM chat API: _parse_chat_message_content_parts is missing. "
            "Update the NeMo SpeechLM compatibility adapter before serving."
        )
    if getattr(original, "_nemo_speechlm_exact_training_prompt", False):
        return

    @wraps(original)
    def _parse_chat_message_content_parts(
        role,
        parts,
        mm_tracker,
        *,
        wrap_dicts,
        interleave_strings,
        mm_processor_kwargs=None,
        multimodal_content_part_separator="\n",
    ):
        if _is_nemo_speechlm_tracker(mm_tracker):
            materialized_parts = list(parts)
            parts = materialized_parts
            has_audio = any(_part_type(part) in _AUDIO_PART_TYPES for part in materialized_parts)
            if has_audio:
                if wrap_dicts or interleave_strings:
                    raise RuntimeError(
                        "NeMo SpeechLM exact prompt rendering requires vLLM's "
                        "non-interleaved string chat-content format."
                    )
                if multimodal_content_part_separator != "\n":
                    raise RuntimeError(
                        "NeMo SpeechLM exact prompt rendering does not accept a "
                        "custom multimodal content-part separator."
                    )
                parts = _canonicalize_speechlm_parts(materialized_parts)
        return original(
            role,
            parts,
            mm_tracker,
            wrap_dicts=wrap_dicts,
            interleave_strings=interleave_strings,
            mm_processor_kwargs=mm_processor_kwargs,
            multimodal_content_part_separator=multimodal_content_part_separator,
        )

    _parse_chat_message_content_parts._nemo_speechlm_exact_training_prompt = True
    chat_utils._parse_chat_message_content_parts = _parse_chat_message_content_parts


def install_prompt_contract() -> None:
    """Install the SpeechLM-only vLLM chat rendering adapter."""
    from vllm.entrypoints import chat_utils

    _patch_speechlm_training_prompt_contract(chat_utils)
