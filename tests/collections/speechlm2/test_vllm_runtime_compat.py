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

"""Focused tests for SpeechLM's exact vLLM training-time prompt contract."""

from types import SimpleNamespace

import pytest

from nemo.collections.speechlm2.vllm.salm.runtime_compat import _patch_speechlm_training_prompt_contract

_AUDIO = {"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64,AA=="}}


def _tracker(model_type):
    return SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type=model_type)))


def _part_type(part):
    if isinstance(part, str):
        return "text"
    return part.get("type") or ("audio_url" if "audio_url" in part else None)


def _fake_vllm_chat_utils():
    """Model vLLM 0.23's non-interleaved placeholder assembly."""
    calls = []

    def parse(
        role,
        parts,
        mm_tracker,
        *,
        wrap_dicts,
        interleave_strings,
        mm_processor_kwargs=None,
        multimodal_content_part_separator="\n",
    ):
        parts = list(parts)
        audio_parts = [part for part in parts if _part_type(part) == "audio_url"]
        texts = [part if isinstance(part, str) else part["text"] for part in parts if _part_type(part) == "text"]
        text_prompt = "\n".join(texts)
        missing = len(audio_parts) - text_prompt.count("<|audio|>")
        if missing < 0:
            raise ValueError("more placeholders than audio payloads")
        pieces = ["<|audio|>"] * missing
        if text_prompt:
            pieces.append(text_prompt)
        rendered = multimodal_content_part_separator.join(pieces)
        calls.append(
            {
                "parts": parts,
                "registered_audio_parts": audio_parts,
                "rendered": rendered,
                "separator": multimodal_content_part_separator,
            }
        )
        return [{"role": role, "content": rendered}]

    return SimpleNamespace(_parse_chat_message_content_parts=parse), calls


def _parse(chat_utils, parts, model_type="nemo_speechlm", **kwargs):
    return chat_utils._parse_chat_message_content_parts(
        "user",
        parts,
        _tracker(model_type),
        wrap_dicts=kwargs.pop("wrap_dicts", False),
        interleave_strings=kwargs.pop("interleave_strings", False),
        **kwargs,
    )


def test_audio_only_keeps_payload_and_renders_one_locator():
    chat_utils, calls = _fake_vllm_chat_utils()
    _patch_speechlm_training_prompt_contract(chat_utils)

    result = _parse(chat_utils, [_AUDIO])

    assert result == [{"role": "user", "content": "<|audio|>"}]
    assert calls[0]["registered_audio_parts"] == [_AUDIO]
    assert calls[0]["parts"] == [_AUDIO, {"type": "text", "text": "<|audio|>"}]


def test_text_then_one_ascii_space_then_audio_locator_exactly():
    chat_utils, calls = _fake_vllm_chat_utils()
    _patch_speechlm_training_prompt_contract(chat_utils)

    result = _parse(
        chat_utils,
        [_AUDIO, {"type": "text", "text": "Use these names:"}, " Alice, Bob"],
    )

    assert result == [{"role": "user", "content": "Use these names:  Alice, Bob <|audio|>"}]
    assert calls[0]["registered_audio_parts"] == [_AUDIO]
    assert calls[0]["rendered"].endswith(" <|audio|>")
    assert "\n" not in calls[0]["rendered"]


def test_text_only_generator_is_preserved():
    chat_utils, calls = _fake_vllm_chat_utils()
    _patch_speechlm_training_prompt_contract(chat_utils)
    parts = (part for part in [{"type": "text", "text": "Hello"}])

    result = _parse(chat_utils, parts)

    assert result == [{"role": "user", "content": "Hello"}]
    assert calls[0]["parts"] == [{"type": "text", "text": "Hello"}]
    assert calls[0]["registered_audio_parts"] == []


def test_non_speechlm_is_bit_for_bit_unmodified():
    chat_utils, calls = _fake_vllm_chat_utils()
    _patch_speechlm_training_prompt_contract(chat_utils)
    parts = [_AUDIO, {"type": "text", "text": "Transcribe"}]

    result = _parse(chat_utils, parts, model_type="other_multimodal")

    assert result == [{"role": "user", "content": "<|audio|>\nTranscribe"}]
    assert calls[0]["parts"] == parts
    assert calls[0]["separator"] == "\n"


def test_valid_preexisting_locator_is_preserved_without_duplication():
    chat_utils, calls = _fake_vllm_chat_utils()
    _patch_speechlm_training_prompt_contract(chat_utils)

    result = _parse(chat_utils, [_AUDIO, {"type": "text", "text": "Transcribe <|audio|>"}])

    assert result == [{"role": "user", "content": "Transcribe <|audio|>"}]
    assert calls[0]["rendered"].count("<|audio|>") == 1
    assert calls[0]["registered_audio_parts"] == [_AUDIO]


@pytest.mark.parametrize(
    "text",
    [
        "<|audio|> Transcribe",
        "Transcribe  <|audio|>",
        "Transcribe <|audio|> extra",
        "Transcribe <|audio|> <|audio|>",
    ],
)
def test_malformed_preexisting_locator_fails_closed(text):
    chat_utils, _ = _fake_vllm_chat_utils()
    _patch_speechlm_training_prompt_contract(chat_utils)

    with pytest.raises(ValueError, match="locator"):
        _parse(chat_utils, [_AUDIO, {"type": "text", "text": text}])


def test_multiple_audio_payloads_fail_closed_before_registration():
    chat_utils, calls = _fake_vllm_chat_utils()
    _patch_speechlm_training_prompt_contract(chat_utils)

    with pytest.raises(ValueError, match="exactly one audio"):
        _parse(chat_utils, [_AUDIO, _AUDIO, {"type": "text", "text": "Compare"}])
    assert calls == []


def test_noninterleaved_string_contract_is_enforced():
    chat_utils, _ = _fake_vllm_chat_utils()
    _patch_speechlm_training_prompt_contract(chat_utils)

    with pytest.raises(RuntimeError, match="non-interleaved string"):
        _parse(chat_utils, [_AUDIO], wrap_dicts=True)
    with pytest.raises(RuntimeError, match="non-interleaved string"):
        _parse(chat_utils, [_AUDIO], interleave_strings=True)


def test_adapter_is_idempotent_and_fails_closed_on_vllm_api_change():
    chat_utils, _ = _fake_vllm_chat_utils()
    _patch_speechlm_training_prompt_contract(chat_utils)
    patched = chat_utils._parse_chat_message_content_parts

    _patch_speechlm_training_prompt_contract(chat_utils)

    assert chat_utils._parse_chat_message_content_parts is patched
    with pytest.raises(TypeError, match="_parse_chat_message_content_parts"):
        _patch_speechlm_training_prompt_contract(SimpleNamespace())
