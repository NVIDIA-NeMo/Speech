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
"""Pure-Python tests for :class:`EasyMagpieOmniArch`.

These have no heavy dependencies (no torch / vllm) and validate the derived
quantities and the ``from_hf_config`` merge logic that the rest of the model
relies on for correct vocab sizes and special-token ids.
"""
from __future__ import annotations

import types

import pytest

from easymagpie_vllm_omni.config import (
    EASYMAGPIE_SMALLMAMBA,
    NUM_SPECIAL_AUDIO_TOKENS,
    SPECIAL_AUDIO_EOS,
    SPECIAL_AUDIO_MASK,
    EasyMagpieOmniArch,
)


def test_derived_codebook_counts():
    arch = EasyMagpieOmniArch(num_audio_codebooks=8, frame_stacking_factor=2, codebook_size=1024)
    assert arch.num_stacked_codebooks == 16
    assert arch.num_all_tokens_per_codebook == 1024 + NUM_SPECIAL_AUDIO_TOKENS


def test_special_token_ids_default_to_codebook_offsets():
    arch = EasyMagpieOmniArch(codebook_size=1024)
    assert arch.audio_eos_id == 1024 + SPECIAL_AUDIO_EOS
    assert arch.mask_token_id == 1024 + SPECIAL_AUDIO_MASK
    # EOS must remain inside the per-codebook vocab so it stays sampleable.
    assert arch.audio_eos_id < arch.num_all_tokens_per_codebook


def test_forced_special_token_ids_override_defaults():
    arch = EasyMagpieOmniArch(
        codebook_size=1024,
        forced_audio_bos_id=1024,
        forced_audio_eos_id=1025,
        forced_mask_token_id=1028,
    )
    assert arch.audio_bos_id == 1024
    assert arch.audio_eos_id == 1025
    assert arch.mask_token_id == 1028


def test_phoneme_ids_fall_back_to_tokenizer_convention():
    arch = EasyMagpieOmniArch(phoneme_vocab_size=2051)
    assert arch.resolved_phoneme_bos_id == 2048
    assert arch.resolved_phoneme_eos_id == 2049
    assert arch.resolved_phoneme_unk_id == 2050


def test_text_eos_id_supports_appended_multiturn_token():
    assert EasyMagpieOmniArch().resolved_text_eos_id(text_vocab_size=103) == 101
    arch = EasyMagpieOmniArch(text_eos_id=101, use_multiturn_dataset=True)
    assert arch.resolved_text_eos_id(text_vocab_size=104) == 101


def test_from_hf_config_overrides_and_ignores_unknown():
    hf_config = types.SimpleNamespace(
        num_audio_codebooks=4,
        codebook_size=2048,
        frame_stacking_factor=1,
        text_eos_id=32001,
        use_multiturn_dataset=True,
        local_transformer_n_layers=5,
        some_unrelated_field="ignored",
    )
    arch = EasyMagpieOmniArch.from_hf_config(hf_config)
    assert arch.num_audio_codebooks == 4
    assert arch.codebook_size == 2048
    assert arch.frame_stacking_factor == 1
    assert arch.text_eos_id == 32001
    assert arch.use_multiturn_dataset is True
    assert arch.local_transformer_n_layers == 5
    # Untouched fields keep the default profile.
    assert arch.audio_embedding_dim == EASYMAGPIE_SMALLMAMBA.audio_embedding_dim


def test_from_hf_config_hidden_size_fallback():
    hf_config = types.SimpleNamespace(hidden_size=999)
    arch = EasyMagpieOmniArch.from_hf_config(hf_config)
    assert arch.hidden_dim == 999
    # embedding_dim defaults to the same backbone width when not given explicitly.
    assert arch.embedding_dim == 999


def test_text_prefill_includes_first_phoneme_bos_position():
    arch = EasyMagpieOmniArch(
        streaming_phonemes_delay=3,
        streaming_speech_delay=5,
    )

    assert arch.text_prefill_num == 4


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"hidden_dim": 1024, "embedding_dim": 1536}, "hidden_dim.*embedding_dim"),
        ({"local_transformer_n_heads": 7}, "local_transformer_hidden_dim.*divisible"),
        ({"phoneme_stacking_factor": 0, "phoneme_vocab_size": 2051}, "both enabled or both disabled"),
        (
            {"streaming_phonemes_delay": 3, "streaming_speech_delay": 3},
            "streaming_speech_delay.*greater",
        ),
        ({"max_audio_seconds": 0}, "max_audio_seconds.*positive"),
        ({"forced_audio_eos_id": 7}, "forced_audio_eos_id"),
    ],
)
def test_validate_rejects_unsupported_architectures(kwargs, message):
    with pytest.raises(ValueError, match=message):
        EasyMagpieOmniArch(**kwargs).validate()


def test_from_hf_config_validates_text_eos_id():
    with pytest.raises(ValueError, match="text_eos_id"):
        EasyMagpieOmniArch.from_hf_config(types.SimpleNamespace(text_vocab_size=10, text_eos_id=10))


def test_audio_row_counts_come_from_converted_codec_metadata():
    arch = EasyMagpieOmniArch(codec_samples_per_frame=640, frame_stacking_factor=2)

    assert arch.codec_input_sample_rate == 16000
    assert arch.reference_audio_num_rows(32_000) == 27
    assert arch.user_audio_num_rows(6_400) == 6


def test_reference_audio_capability_is_configuration_only():
    arch = EasyMagpieOmniArch(codec_encoder_bundled=True)

    assert arch.supports_reference_audio is True
    assert arch.is_multiturn_checkpoint is False
    assert arch.supports_user_audio_prefill is False
    arch.ensure_reference_audio_available()


def test_reference_audio_capability_keeps_externally_released_artifact_path_available():
    arch = EasyMagpieOmniArch(codec_encoder_bundled=False)

    assert arch.supports_reference_audio is False
    with pytest.raises(RuntimeError, match="speaker_id"):
        arch.ensure_reference_audio_available()


def test_user_audio_prefill_capability_uses_authoritative_converted_flags():
    with pytest.raises(RuntimeError, match="use_multiturn_dataset is false"):
        EasyMagpieOmniArch().require_user_audio_prefill()

    multiturn_without_audio = EasyMagpieOmniArch(use_multiturn_dataset=True)
    with pytest.raises(RuntimeError, match="condition_on_user_speech is false"):
        multiturn_without_audio.require_user_audio_prefill()

    unbundled_multiturn = EasyMagpieOmniArch(
        use_multiturn_dataset=True,
        condition_on_user_speech=True,
        use_user_speaking_token=True,
    )
    assert unbundled_multiturn.is_multiturn_checkpoint is True
    assert unbundled_multiturn.supports_user_audio_prefill is False
    with pytest.raises(RuntimeError, match="omits the raw-audio encoder tower"):
        unbundled_multiturn.require_user_audio_prefill()

    capable = EasyMagpieOmniArch(
        codec_encoder_bundled=True,
        use_multiturn_dataset=True,
        condition_on_user_speech=True,
        use_user_speaking_token=True,
    )
    assert capable.is_multiturn_checkpoint is True
    assert capable.supports_user_audio_prefill is True
    capable.require_user_audio_prefill()
