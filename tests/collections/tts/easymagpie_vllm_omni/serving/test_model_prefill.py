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
from vllm.config import CUDAGraphMode


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


@pytest.mark.parametrize("computed, expected", [(70, "prefill"), (72, "decode")])
def test_text_resume_uses_decode_after_initial_context(computed, expected):
    model = SimpleNamespace(
        _batch_slot_offset=lambda _ids, start: start,
        _dec_text_mask=torch.ones(1),
        _dec_audio_valid=torch.ones(1),
        has_phoneme=False,
        _preprocess_prefill=lambda *args: "prefill",
        _preprocess_decode=lambda *args: "decode",
    )
    result = EasyMagpieTTSForConditionalGeneration.preprocess(
        model,
        torch.zeros(1, dtype=torch.long),
        None,
        _omni_is_prefill=True,
        _omni_num_computed_tokens=computed,
        text_stream_context_len=71,
        text_token=[42],
    )
    assert result == expected


def test_async_omni_output_slices_each_request_in_a_padded_batch():
    from vllm_omni.worker.output.payload_build import build_omni_mm_payload

    hidden = torch.zeros(40, 8)
    model = SimpleNamespace(_out_codes=torch.arange(80).view(40, 2), _single_stage_audio=False)
    output = EasyMagpieTTSForConditionalGeneration.make_omni_output(
        model, hidden, request_token_spans=[(index, index + 1) for index in range(34)]
    )

    assert output.text_hidden_states is hidden
    assert set(output.multimodal_outputs) == {"codes"}
    per_request = build_omni_mm_payload(
        combined_multimodal_outputs=None,
        mm_cpu={"codes.audio": output.multimodal_outputs["codes"]["audio"]},
        rid="request-33",
        idx=33,
        start=33,
        end=34,
        audio_sparse_output=False,
        sparse_mm_index={},
        hidden_seq_len=0,
        scheduled_seq_len=34,
    )
    torch.testing.assert_close(per_request["codes.audio"], model._out_codes[33:34])


def test_runner_owns_the_feedback_snapshots():
    from vllm_omni.worker.gpu_model_runner import OmniGPUModelRunner

    model = SimpleNamespace(
        has_phoneme=True,
        _dec_phoneme_state=torch.tensor([[5, 6, 0], [4, 7, 1]]),
        gpu_resident_buffer_keys=EasyMagpieTTSForConditionalGeneration.gpu_resident_buffer_keys,
    )
    runner = object.__new__(OmniGPUModelRunner)
    runner.model = model
    runner.requests = {"id": SimpleNamespace()}
    runner.model_intermediate_buffer = {}
    codes = torch.tensor([[1, 2], [3, 4]])
    output = EasyMagpieTTSForConditionalGeneration.postprocess(
        model, torch.zeros(2, 3), multimodal_outputs={"codes": {"audio": codes}}
    )

    runner._update_intermediate_buffer("id", output)
    codes.zero_()
    model._dec_phoneme_state.zero_()
    retained = runner.model_intermediate_buffer["id"]["gpu_state"]
    torch.testing.assert_close(retained["last_audio_codes"], torch.tensor([[3, 4]]))
    torch.testing.assert_close(retained["last_phoneme_token"], torch.tensor([[4, 7, 1]]))


@pytest.fixture
def phoneme_feedback_model():
    model = EasyMagpieTTSForConditionalGeneration.__new__(EasyMagpieTTSForConditionalGeneration)
    nn.Module.__init__(model)
    model.arch = SimpleNamespace(phoneme_stacking_factor=2, phoneme_vocab_size=8, audio_bos_id=1)
    model.has_phoneme = True
    model.phonemes_delay = 1
    model.phoneme_bos_id = model.phoneme_eos_id = 7
    model.phoneme_confidence_unk_threshold = 0.0
    model.speech_delay = 9
    model.audio_eos_id = 9
    model.embedding_dim = 2
    model.code_predictor = SimpleNamespace(
        requires_all_tokens=True,
        embed_audio_frame=lambda codes: torch.zeros_like(codes, dtype=torch.float32),
        generate_codes=lambda hidden, **kwargs: torch.zeros_like(hidden, dtype=torch.long),
        advance=lambda *args, **kwargs: None,
    )
    model.text_embedding = nn.Embedding(1, 2)
    model._embed_phoneme = lambda tokens: tokens.float()
    model.phoneme_final_proj = lambda hidden: nn.functional.one_hot(hidden.long(), 8).flatten(1).float()
    model._dec_text_tokens = torch.zeros(3, dtype=torch.long)
    model._dec_text_mask = torch.zeros(3, dtype=torch.long)
    model._dec_audio_codes = torch.zeros(3, 2, dtype=torch.long)
    model._dec_audio_valid = torch.zeros(3, dtype=torch.long)
    model._dec_phoneme_state = torch.zeros(3, 3, dtype=torch.long)
    model._dec_phoneme_valid = torch.zeros(3, dtype=torch.long)
    model._combined_embeddings = torch.zeros(3, 2)
    model._out_codes = torch.zeros(3, 2, dtype=torch.long)
    model._token_stop = torch.zeros(3, dtype=torch.bool)
    model._sample_stop = torch.zeros(3, dtype=torch.bool)
    return model


@torch.inference_mode()
def test_resumed_phoneme_eos_is_fed_once_then_masks_the_channel(phoneme_feedback_model):
    from vllm_omni.worker.gpu_model_runner import OmniGPUModelRunner

    model = phoneme_feedback_model
    runner = object.__new__(OmniGPUModelRunner)
    runner.model = model
    runner.requests = {"id": SimpleNamespace()}
    runner.model_intermediate_buffer = {"id": {}}
    state = runner.model_intermediate_buffer["id"]
    ids = torch.zeros(1, dtype=torch.long)
    for prediction, embedded, ended in [
        ([4, 7], [0, 0], 0),
        ([4, 7], [7, 7], 0),
        ([2, 3], [4, 7], 1),
        ([1, 2], [0, 0], 1),
    ]:
        _, combined, update = model.preprocess(
            ids,
            None,
            _omni_is_prefill=True,
            _omni_num_computed_tokens=5,
            text_stream_context_len=5,
            text_token=[],
            **state,
        )
        model._assemble_decode_embeddings(combined, slice(0, 1))
        torch.testing.assert_close(combined[0], torch.tensor(embedded, dtype=torch.float32))
        hidden = torch.tensor([prediction])
        model._predict_phonemes(hidden, slice(0, 1))
        runner._update_intermediate_buffer("id", {**update, **model.postprocess(hidden)})
        torch.testing.assert_close(state["gpu_state"]["last_phoneme_token"], torch.tensor([[*prediction, ended]]))


@pytest.mark.parametrize(
    "prefill_rows,has_decode,user_audio", [(2, False, True), (2, True, True), (2, False, False), (1, False, True)]
)
@torch.inference_mode()
def test_prefill_refreshes_the_last_phoneme(phoneme_feedback_model, prefill_rows, has_decode, user_audio, monkeypatch):
    model = phoneme_feedback_model
    model._maybe_set_code_sampling_params = lambda _: None
    model._build_user_audio_prefill_chunk = lambda **kwargs: kwargs["input_embeds"]
    model._build_reference_audio_prefill_chunk = lambda **kwargs: (kwargs["input_embeds"], 1, False, 1)
    ids = torch.zeros(prefill_rows + has_decode, dtype=torch.long)
    model._dec_phoneme_state[has_decode : ids.numel()] = torch.tensor([3, 4, 1])
    model._dec_phoneme_valid[has_decode : ids.numel()].fill_(1)
    combined = torch.zeros(ids.numel(), 2)
    _, _, update = model.preprocess(
        ids[-prefill_rows:],
        combined[-prefill_rows:],
        _omni_is_prefill=True,
        _omni_num_computed_tokens=1,
        _omni_prompt_len=prefill_rows + 1,
        speaker_reference_prefilled=user_audio,
        reference_conditioning_len=1,
        gpu_state={"last_phoneme_token": torch.tensor([[1, 2, 1]])},
    )
    assert bool(update.get("user_audio_prefilled")) == user_audio
    model.vllm_config = SimpleNamespace(compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.NONE))
    query_start = torch.tensor([0, 1, ids.numel()] if has_decode else [0, ids.numel()])
    metadata = SimpleNamespace(max_query_len=prefill_rows, query_start_loc=query_start)
    monkeypatch.setattr(
        "easymagpie_vllm_omni.easymagpie.get_forward_context", lambda: SimpleNamespace(attn_metadata=metadata)
    )
    model.backbone = lambda **kwargs: torch.tensor([[1, 2], [3, 4], [5, 6]][-ids.numel() :])
    hidden = model(ids, torch.arange(ids.numel()), inputs_embeds=combined)
    torch.testing.assert_close(model._combined_embeddings[: ids.numel()], combined)
    feedback = model.postprocess(hidden[-prefill_rows:])["gpu_state"]["last_phoneme_token"]
    torch.testing.assert_close(feedback, torch.tensor([[5, 6, int(not user_audio)]]))
