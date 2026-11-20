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
from types import SimpleNamespace

import pytest
import torch
from easymagpie_vllm_omni.codec.encoder import EasyMagpieCodecEncoderOutput
from easymagpie_vllm_omni.easymagpie import EasyMagpieTTSForConditionalGeneration
from torch import nn


def test_text_prefill_embeddings_add_phoneme_bos_at_position_three():
    model = EasyMagpieTTSForConditionalGeneration.__new__(EasyMagpieTTSForConditionalGeneration)
    nn.Module.__init__(model)
    model.arch = SimpleNamespace(text_prefill_num=4, phoneme_stacking_factor=1)
    model.embedding_dim = 3
    model.has_phoneme = True
    model.phonemes_delay = 3
    model.phoneme_bos_id = 7
    model.text_embedding = nn.Embedding(32, 3)
    model.phoneme_embeddings = nn.ModuleList([nn.Embedding(16, 3)])

    with torch.no_grad():
        model.text_embedding.weight.zero_()
        model.phoneme_embeddings[0].weight.zero_()
        for index, token_id in enumerate((10, 11, 12, 13), start=1):
            model.text_embedding.weight[token_id] = torch.tensor([index, 0, 0])
        model.phoneme_embeddings[0].weight[7] = torch.tensor([0, 0, 10])

    rows = model._build_text_prefill_embeds(
        torch.device("cpu"),
        torch.float32,
        {"text_prefill_num": 4, "prefill_text_tokens": [10, 11, 12, 13]},
    )

    torch.testing.assert_close(
        rows,
        torch.tensor([[1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 10]], dtype=torch.float32),
    )


class _FakeReferenceCodecEncoder(nn.Module):
    def forward(self, audio, audio_lens, *, audio_frame_embedder=None):
        del audio_frame_embedder
        self.audio_data_ptr = audio.data_ptr()
        self.audio = audio.detach().clone()
        self.audio_lens = audio_lens.detach().clone()
        count = int(audio.shape[0])
        return EasyMagpieCodecEncoderOutput(
            acoustic_codes=torch.zeros(audio.shape[0], 4, 2, dtype=torch.long),
            acoustic_lens=torch.full((audio.shape[0],), 2, dtype=torch.long),
            reference_speaker_embeddings=torch.full((count, 4, 2), 9.0),
            reference_speaker_embedding_lens=torch.full((count,), 4, dtype=torch.long),
        )


def test_embed_multimodal_returns_reference_embeddings_when_codec_encoder_is_present():
    model = EasyMagpieTTSForConditionalGeneration.__new__(EasyMagpieTTSForConditionalGeneration)
    nn.Module.__init__(model)
    model.arch = SimpleNamespace(
        ensure_reference_audio_available=lambda: None,
        codec_samples_per_row=16,
        audio_user_speaking_id=17,
        require_user_audio_prefill=lambda: None,
    )
    model.model_path = "unused"
    model.speech_delay = 0
    model.num_codebooks = 2
    model.codec_encoder = _FakeReferenceCodecEncoder()
    model.code_predictor = SimpleNamespace(embed_audio_frame=lambda codes: codes.float())
    model._combined_embeddings = torch.zeros(1, 2)

    waveform = torch.tensor([1.0, 2.0, 3.0, 4.0])
    outputs = model.embed_multimodal(
        audio_values=waveform,
        audio_lens=torch.tensor([4]),
    )

    assert [output.shape for output in outputs] == [(4, 2)]
    torch.testing.assert_close(outputs[0], torch.full((4, 2), 9.0))
    torch.testing.assert_close(model.codec_encoder.audio[0], torch.tensor([1.0, 2.0, 3.0, 4.0]))
    assert model.codec_encoder.audio_data_ptr == waveform.data_ptr()
    torch.testing.assert_close(model.codec_encoder.audio_lens, torch.tensor([4]))


def test_embed_multimodal_rejects_reference_audio_when_codec_encoder_is_absent():
    model = EasyMagpieTTSForConditionalGeneration.__new__(EasyMagpieTTSForConditionalGeneration)
    nn.Module.__init__(model)
    model.arch = SimpleNamespace(
        ensure_reference_audio_available=lambda: (_ for _ in ()).throw(RuntimeError("no reference-audio encoder"))
    )
    model.codec_encoder = None

    with pytest.raises(RuntimeError, match="no reference-audio encoder"):
        model.embed_multimodal(
            audio_values=[torch.ones(4)],
            audio_lens=torch.tensor([4]),
        )


def _raw_reference_prefill_model():
    model = EasyMagpieTTSForConditionalGeneration.__new__(EasyMagpieTTSForConditionalGeneration)
    nn.Module.__init__(model)
    model._maybe_set_lt_sampling_params = lambda info: None
    model._build_reference_audio_prefill_chunk = lambda **kwargs: (
        torch.full((kwargs["span_len"], 2), 7.0),
        2,
        False,
        2,
    )
    model._build_prefill_embeds = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("known-speaker path must not run")
    )
    model.speech_delay = 2
    return model


def test_prefill_infers_raw_reference_when_speaker_id_is_absent():
    model = _raw_reference_prefill_model()

    _, rows, _ = model._preprocess_prefill(
        input_ids=torch.tensor([1, 1]),
        input_embeds=torch.ones(2, 2),
        span_len=2,
        device=torch.device("cpu"),
        info_dict={
            "_omni_num_computed_tokens": 0,
            "_omni_prompt_len": 2,
        },
    )

    torch.testing.assert_close(rows, torch.full((2, 2), 7.0))


def test_reference_prefill_rejects_non_default_task_mode():
    model = _reference_prefill_model()

    with pytest.raises(ValueError, match="Only task_mode_id=0"):
        model._build_reference_audio_prefill_chunk(
            input_embeds=torch.ones(2, 3),
            info_dict={"task_mode_id": 1},
            text_tokens=[],
            offset=0,
            span_len=2,
            prompt_len=2,
        )


def _reference_prefill_model():
    model = EasyMagpieTTSForConditionalGeneration.__new__(EasyMagpieTTSForConditionalGeneration)
    nn.Module.__init__(model)
    model.arch = SimpleNamespace(
        audio_input_token_id=1,
        require_user_audio_prefill=lambda: None,
        text_prefill_num=4,
    )
    model.model_path = "unused"
    model.embedding_dim = 3
    model.has_phoneme = False
    model.task_embedding = None
    model.num_task_embeddings = 0
    model.text_embedding = nn.Embedding(32, 3)
    model._encode_context_text = lambda text, device: torch.tensor([20], device=device)
    with torch.no_grad():
        model.text_embedding.weight.zero_()
        model.text_embedding.weight[20] = torch.tensor([0, 8, 0])
        for index, token_id in enumerate((10, 11, 12, 13), start=1):
            model.text_embedding.weight[token_id] = torch.tensor([index, 0, 0])
    return model


def test_reference_and_user_prefill_overlays_task_before_reference_boundary_is_known():
    model = _reference_prefill_model()
    model.arch.audio_input_token_id = 1
    model.task_embedding = nn.Embedding(1, 3)
    model.num_task_embeddings = 1
    with torch.no_grad():
        model.task_embedding.weight[0] = torch.tensor([9, 0, 0])

    rows, conditioning_len, has_user_audio, reference_rows = model._build_reference_audio_prefill_chunk(
        input_embeds=torch.ones(2, 3),
        info_dict={"context_text": "[EN]", "text_prefill_num": 4},
        text_tokens=[],
        input_ids=torch.tensor([0, 1]),
        offset=0,
        span_len=2,
        prompt_len=8,
    )

    assert conditioning_len == 0
    assert has_user_audio is False
    assert reference_rows == 0
    torch.testing.assert_close(rows[0], torch.tensor([9, 0, 0], dtype=torch.float32))
    torch.testing.assert_close(rows[1], torch.ones(3))


def test_reference_audio_prefill_rejects_non_default_task_mode():
    model = _reference_prefill_model()

    with pytest.raises(ValueError, match="Only task_mode_id=0"):
        model._build_reference_audio_prefill_chunk(
            input_embeds=torch.ones(2, 3),
            info_dict={"task_mode_id": 1},
            text_tokens=[],
            offset=0,
            span_len=2,
            prompt_len=2,
        )


def test_reference_prefill_rejects_initial_user_only_layout_without_speaker_id():
    model = _reference_prefill_model()

    with pytest.raises(ValueError, match="must place reference audio before context rows"):
        model._build_reference_audio_prefill_chunk(
            input_embeds=torch.ones(2, 3),
            info_dict={"context_text": "[EN]", "text_prefill_num": 0},
            text_tokens=[],
            input_ids=torch.tensor([1, 1]),
            offset=0,
            span_len=2,
            prompt_len=2,
        )


def test_reference_audio_prefill_preserves_speaker_rows_and_is_chunk_invariant():
    model = _reference_prefill_model()
    info = {
        "context_text": "[EN]",
        "text_prefill_num": 4,
        "prefill_text_tokens": [10, 11, 12, 13],
        "reference_audio_num_rows": 2,
    }
    input_embeds = torch.ones(7, 3)
    input_ids = torch.tensor([1, 1, 0, 0, 0, 0, 0])

    whole, conditioning_len, has_user_audio, reference_rows = model._build_reference_audio_prefill_chunk(
        input_embeds=input_embeds,
        info_dict=info,
        text_tokens=[],
        input_ids=input_ids,
        offset=0,
        span_len=7,
        prompt_len=7,
    )
    chunks = torch.cat(
        [
            model._build_reference_audio_prefill_chunk(
                input_embeds=input_embeds[:4],
                info_dict=info,
                text_tokens=[],
                input_ids=input_ids[:4],
                offset=0,
                span_len=4,
                prompt_len=7,
            )[0],
            model._build_reference_audio_prefill_chunk(
                input_embeds=input_embeds[4:],
                info_dict=info,
                text_tokens=[],
                input_ids=input_ids[4:],
                offset=4,
                span_len=3,
                prompt_len=7,
            )[0],
        ]
    )

    assert conditioning_len == 3
    assert has_user_audio is False
    assert reference_rows == 2
    torch.testing.assert_close(chunks, whole)
    torch.testing.assert_close(whole[:2], torch.ones(2, 3))
    torch.testing.assert_close(whole[2], torch.tensor([0, 8, 0], dtype=torch.float32))
    torch.testing.assert_close(
        whole[3:],
        torch.tensor([[1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]], dtype=torch.float32),
    )


def test_reference_only_prefill_state_remains_reference_layout_across_scheduler_chunks():
    model = _reference_prefill_model()
    model.speech_delay = 2
    model._combined_embeddings = torch.zeros(1, 3)
    model._maybe_set_lt_sampling_params = lambda info: None
    base_info = {
        "_omni_prompt_len": 7,
        "context_text": "[EN]",
        "text_prefill_num": 4,
        "prefill_text_tokens": [10, 11, 12, 13],
    }

    _, first_rows, update = model._preprocess_prefill(
        input_ids=torch.tensor([1, 1, 0, 0]),
        input_embeds=torch.ones(4, 3),
        span_len=4,
        device=torch.device("cpu"),
        info_dict={**base_info, "_omni_num_computed_tokens": 0},
    )
    _, second_rows, _ = model._preprocess_prefill(
        input_ids=torch.tensor([0, 0, 0]),
        input_embeds=torch.ones(3, 3),
        span_len=3,
        device=torch.device("cpu"),
        info_dict={
            **base_info,
            **update,
            "_omni_num_computed_tokens": 4,
        },
    )

    torch.testing.assert_close(
        torch.cat((first_rows, second_rows)),
        torch.tensor(
            [
                [1, 1, 1],
                [1, 1, 1],
                [0, 8, 0],
                [1, 0, 0],
                [2, 0, 0],
                [3, 0, 0],
                [4, 0, 0],
            ],
            dtype=torch.float32,
        ),
    )


def test_user_audio_prefill_composition_is_chunk_invariant():
    model = EasyMagpieTTSForConditionalGeneration.__new__(EasyMagpieTTSForConditionalGeneration)
    nn.Module.__init__(model)
    model.speech_delay = 2
    model.embedding_dim = 3
    model.has_phoneme = False
    model.text_embedding = nn.Embedding(32, 3)
    with torch.no_grad():
        model.text_embedding.weight.zero_()
        model.text_embedding.weight[10] = torch.tensor([1, 0, 0])
        model.text_embedding.weight[11] = torch.tensor([2, 0, 0])

    conditioning = torch.tensor([[9, 0, 0], [8, 0, 0]], dtype=torch.float32)
    input_embeds = torch.ones(6, 3)
    kwargs = {
        "conditioning": conditioning,
        "text_tokens": [10, 11],
        "prompt_len": 6,
    }

    whole = model._build_user_audio_prefill_chunk(
        input_embeds=input_embeds,
        offset=0,
        span_len=6,
        **kwargs,
    )
    chunks = torch.cat(
        [
            model._build_user_audio_prefill_chunk(
                input_embeds=input_embeds[:3],
                offset=0,
                span_len=3,
                **kwargs,
            ),
            model._build_user_audio_prefill_chunk(
                input_embeds=input_embeds[3:],
                offset=3,
                span_len=3,
                **kwargs,
            ),
        ]
    )

    torch.testing.assert_close(chunks, whole)
    torch.testing.assert_close(whole[:2], conditioning)
    torch.testing.assert_close(whole[2:4], torch.ones(2, 3))
    torch.testing.assert_close(
        whole[4:],
        torch.tensor([[2, 1, 1], [3, 1, 1]], dtype=torch.float32),
    )


class _FakeCombinedCodecEncoder(nn.Module):
    def forward(self, audio, audio_lens, *, audio_frame_embedder=None):
        del audio_frame_embedder
        self.audio = audio.detach().clone()
        self.audio_lens = audio_lens.detach().clone()
        code_lens = torch.div(audio_lens + 1279, 1280, rounding_mode="floor")
        codes = torch.zeros(audio.shape[0], 2, int(code_lens.max()), dtype=torch.long)
        codes[:, 0] = 1
        codes[:, 1] = 2
        count = int(audio.shape[0])
        return EasyMagpieCodecEncoderOutput(
            acoustic_codes=codes,
            acoustic_lens=code_lens,
            reference_speaker_embeddings=torch.full((count, 4, 2), 9.0),
            reference_speaker_embedding_lens=torch.full((count,), 4, dtype=torch.long),
        )


def test_embed_multimodal_batches_reference_and_variable_user_audio():
    model = EasyMagpieTTSForConditionalGeneration.__new__(EasyMagpieTTSForConditionalGeneration)
    nn.Module.__init__(model)
    model.arch = SimpleNamespace(
        audio_user_speaking_id=37,
        codec_samples_per_row=1280,
        ensure_reference_audio_available=lambda: None,
        require_user_audio_prefill=lambda: None,
    )
    model.model_path = "unused"
    model.speech_delay = 5
    model.num_codebooks = 2
    model.codec_encoder = _FakeCombinedCodecEncoder()
    model.code_predictor = SimpleNamespace(embed_audio_frame=lambda codes: codes.float())
    model._combined_embeddings = torch.zeros(1, 2)

    reference = torch.tensor([1.0, 2.0, 3.0, 4.0])
    user = torch.arange(100, dtype=torch.float32)
    outputs = model.embed_multimodal(
        audio_values=[reference, user],
        audio_lens=torch.tensor([4, 100]),
        audio_output_kinds=torch.tensor([1, 0]),
    )

    assert [output.shape for output in outputs] == [(4, 2), (6, 2)]
    assert model.codec_encoder.audio.shape == (2, 6400)
    assert model.codec_encoder.audio_lens.tolist() == [4, 6400]
    torch.testing.assert_close(model.codec_encoder.audio[0, :4], reference)
    assert torch.count_nonzero(model.codec_encoder.audio[0, 4:]) == 0
    torch.testing.assert_close(model.codec_encoder.audio[1, -100:], user)
    torch.testing.assert_close(outputs[0], torch.full((4, 2), 9.0))
    torch.testing.assert_close(outputs[1][0], torch.tensor([37.0, 37.0]))
    torch.testing.assert_close(outputs[1][1], torch.tensor([38.0, 39.0]))


def test_reference_and_user_audio_prefill_is_chunk_invariant():
    model = _reference_prefill_model()
    model.speech_delay = 2
    info = {
        "context_text": "[EN]",
        "text_prefill_num": 1,
        "reference_audio_num_rows": 2,
    }
    input_embeds = torch.ones(7, 3)
    kwargs = {
        "info_dict": info,
        "text_tokens": [10, 11],
        "prompt_len": 7,
    }

    whole, conditioning_len, has_user_audio, reference_rows = model._build_reference_audio_prefill_chunk(
        input_embeds=input_embeds,
        offset=0,
        span_len=7,
        **kwargs,
    )
    chunks = torch.cat(
        [
            model._build_reference_audio_prefill_chunk(
                input_embeds=input_embeds[:4],
                offset=0,
                span_len=4,
                **kwargs,
            )[0],
            model._build_reference_audio_prefill_chunk(
                input_embeds=input_embeds[4:],
                offset=4,
                span_len=3,
                **kwargs,
            )[0],
        ]
    )

    assert conditioning_len == 3
    assert has_user_audio is True
    assert reference_rows == 2
    torch.testing.assert_close(chunks, whole)
    torch.testing.assert_close(whole[:2], torch.ones(2, 3))
    torch.testing.assert_close(whole[2], torch.tensor([0, 8, 0], dtype=torch.float32))
    torch.testing.assert_close(whole[3:5], torch.ones(2, 3))
    torch.testing.assert_close(
        whole[5:],
        torch.tensor([[2, 1, 1], [3, 1, 1]], dtype=torch.float32),
    )


def test_user_audio_prefill_detection_handles_resumed_audio_only():
    model = EasyMagpieTTSForConditionalGeneration.__new__(EasyMagpieTTSForConditionalGeneration)
    nn.Module.__init__(model)
    model.arch = SimpleNamespace(condition_on_user_speech=True, audio_input_token_id=1)

    common = {"conditioning_len": 2, "prompt_len": 8, "legacy_prompt_len": 6}
    assert model._is_user_audio_prefill_chunk(input_ids=torch.tensor([0, 0]), offset=0, **common)
    assert model._is_user_audio_prefill_chunk(input_ids=torch.tensor([1, 1]), offset=6, **common)
    assert not model._is_user_audio_prefill_chunk(input_ids=torch.tensor([0]), offset=8, **common)


def test_user_audio_prefill_detection_keeps_legacy_text_prefill_unchanged():
    model = EasyMagpieTTSForConditionalGeneration.__new__(EasyMagpieTTSForConditionalGeneration)
    nn.Module.__init__(model)
    model.arch = SimpleNamespace(condition_on_user_speech=True, audio_input_token_id=1)

    assert not model._is_user_audio_prefill_chunk(
        input_ids=torch.tensor([0, 0]),
        offset=0,
        conditioning_len=2,
        prompt_len=6,
        legacy_prompt_len=6,
    )


def test_first_agent_decode_uses_user_speaking_end_token():
    model = EasyMagpieTTSForConditionalGeneration.__new__(EasyMagpieTTSForConditionalGeneration)
    nn.Module.__init__(model)
    model.arch = SimpleNamespace(
        phoneme_stacking_factor=0,
        audio_bos_id=32,
        audio_user_speaking_end_id=38,
        use_user_speaking_end_token=True,
    )
    model.embedding_dim = 3
    model.num_codebooks = 2
    model.speech_delay = 5
    model.has_phoneme = False
    model._combined_embeddings = torch.zeros(1, 3)
    model._dec_text_tokens = torch.zeros(1, dtype=torch.long)
