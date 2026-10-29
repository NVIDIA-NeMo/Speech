# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Guidance and shared-feedback contracts with synthetic temporal predictor blocks."""
from types import SimpleNamespace

import pytest
import torch
from easymagpie_vllm_omni.acoustic_predictor import EasyMagpieAcousticCodesPredictor
from easymagpie_vllm_omni.cfg_pairing import CFG_PAIR_METADATA_KEY, CFGPairingLayout
from easymagpie_vllm_omni.easymagpie import EasyMagpieTTSForConditionalGeneration
from torch import nn


@pytest.mark.parametrize("scale,expected", [(0.0, 2), (1.0, 1), (2.5, 0)])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_guidance_before_sampling_and_shared_feedback_with_reordered_rows(scale, expected, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    # Unconditional comes first, and both histories include a non-audio row.
    logits = torch.zeros(4, 1, 4, device=device)
    logits[3, 0] = torch.tensor([3.0, 4.0, 0.0, -10.0], device=device)
    logits[1, 0] = torch.tensor([0.0, 3.0, 5.0, -10.0], device=device)
    predictor, blocks, embedder = _predictor([logits, logits], device)
    layout = CFGPairingLayout(("public",), ((2, 3),), ((0, 1),))
    codes = predictor.generate_codes(
        torch.zeros(4, 2, device=device),
        positions=torch.tensor([0, 1, 0, 1], device=device),
        predict=torch.tensor([False, True, False, True], device=device),
        pairing=layout,
        cfg_scale=torch.full((4,), scale, device=device),
        temperature=0.0,
        top_k=4,
    )
    assert codes.tolist() == [[8, 8], [expected, expected], [8, 8], [expected, expected]]
    assert len(embedder.inputs) == 1
    assert embedder.inputs[0].flatten().tolist() == [8, expected, 8, expected]
    torch.testing.assert_close(blocks[1].inputs, torch.tensor([[8.0, 8.0], [expected, expected]] * 2, device=device))


def test_guidance_precedes_sanitization():
    logits = torch.tensor([[[90.0, 80.0, 0.0, -10.0]], [[130.0, 90.0, 0.0, -10.0]]])
    predictor, _, _ = _predictor([logits])
    result = predictor.generate_codes(
        torch.zeros(2, 2),
        positions=torch.zeros(2, dtype=torch.long),
        pairing=CFGPairingLayout(("r",), ((0,),), ((1,),)),
        cfg_scale=torch.full((2,), 2.0),
        temperature=0.0,
        top_k=4,
    )
    # Raw guidance is [50, 70]; sanitizing branches first incorrectly picks class 0.
    assert result.tolist() == [[1], [1]]


def test_multiple_pairs_and_conditional_row_keep_guidance_sampling_and_feedback_separate():
    # Rows: private A, plain, public B, public A, private B.
    logits = torch.tensor(
        [
            [[0.0, 3.0, 5.0, -10.0]],
            [[0.0, 1.0, 6.0, -10.0]],
            [[0.0, 7.0, 1.0, -10.0]],
            [[3.0, 4.0, 0.0, -10.0]],
            [[0.0, 1.0, 0.0, -10.0]],
        ]
    )
    predictor, _, embedder = _predictor([logits, logits])
    calls = []

    def sample(values, *, temperature, top_k):
        calls.append((values.clone(), temperature.clone(), top_k.clone()))
        return values.argmax(dim=-1)

    predictor._sample_codes = sample
    result = predictor.generate_codes(
        torch.zeros(5, 2),
        positions=torch.ones(5, dtype=torch.long),
        pairing=CFGPairingLayout(('A', 'B'), ((3,), (2,)), ((0,), (4,))),
        cfg_scale=torch.tensor([2.5, 1.0, 3.0, 2.5, 3.0]),
        temperature=torch.tensor([0.7, 0.1, 0.7, 0.7, 0.7]),
        top_k=torch.tensor([80, 10, 30, 80, 30]),
    )
    assert result.tolist() == [[0, 0], [2, 2], [1, 1], [0, 0], [1, 1]]
    for values, temperature, top_k in calls:
        assert values.shape == (3, 1, 4)
        torch.testing.assert_close(temperature.flatten(), torch.tensor([0.1, 0.7, 0.7]))
        assert top_k.flatten().tolist() == [10, 30, 80]
    assert embedder.inputs[0].flatten().tolist() == [0, 2, 1, 0, 1]


def test_bfloat16_guidance_rounds_each_reference_coefficient_independently():
    logits = torch.tensor([[[1.0, 3.0, 0.0, 0.0]], [[16.0, 2.0, 0.0, 0.0]]], dtype=torch.bfloat16)
    predictor, _, _ = _predictor([logits])
    captured = []

    def sample(values, **kwargs):
        captured.append(values.clone())
        return values.argmax(dim=-1)

    predictor._sample_codes = sample
    predictor.generate_codes(
        torch.zeros(2, 2, dtype=torch.bfloat16),
        positions=torch.zeros(2, dtype=torch.long),
        pairing=CFGPairingLayout(("r",), ((0,),), ((1,),)),
        cfg_scale=torch.full((2,), 1.1),
        temperature=0.0,
    )
    expected = 1.1 * logits[:1] + (1.0 - 1.1) * logits[1:]
    torch.testing.assert_close(captured[0], expected, rtol=0, atol=0)


def test_per_request_sampling_and_cutoff_ties(monkeypatch):
    logits = torch.tensor([[[5.0, 5.0, 0.0, -10.0]], [[5.0, 5.0, 0.0, -10.0]]])
    predictor, _, _ = _predictor([logits])

    def noise(tensor, *args):
        tensor.fill_(0.1)
        tensor[..., 1] = 0.99
        return tensor

    monkeypatch.setattr(torch.Tensor, "uniform_", noise)
    codes = predictor.generate_codes(
        torch.zeros(2, 2),
        positions=torch.zeros(2, dtype=torch.long),
        temperature=torch.tensor([0.0, 1.0]),
        top_k=torch.tensor([1, 1]),
    )
    # NeMo retains every logit tied with the kth cutoff; greedy takes the first.
    assert codes.tolist() == [[0], [1]]


def test_compact_eos_is_shared_and_non_audio_rows_use_mask():
    logits = torch.tensor([[[0.0, 0.0, 0.0, 10.0]], [[0.0, 0.0, 0.0, 20.0]]])
    predictor, _, _ = _predictor([logits])
    result = predictor.generate_codes(
        torch.zeros(2, 2),
        positions=torch.zeros(2, dtype=torch.long),
        pairing=CFGPairingLayout(("r",), ((0,),), ((1,),)),
        cfg_scale=torch.ones(2),
        temperature=0.0,
    )
    assert result.tolist() == [[7], [7]]


def test_pure_prefill_advances_both_predictor_histories_with_mask():
    predictor, blocks, embedder = _predictor([torch.zeros(4, 1, 4)] * 2)
    predictor.advance(torch.zeros(4, 2), positions=torch.tensor([0, 1, 0, 1]))
    assert embedder.inputs[0].tolist() == [[8]] * 4
    torch.testing.assert_close(blocks[1].inputs, torch.full((4, 2), 8.0))


def test_unconditional_decode_is_audio_only_with_request_owned_sampling():
    model = _input_model()
    model.arch = SimpleNamespace(cfg_capability={"version": 1}, phoneme_stacking_factor=1)
    model.has_phoneme = True
    model.phonemes_delay = 0
    model.speech_delay = 1
    model.num_codebooks = 1
    model.phoneme_eos_id = 3
    model._dec_text_tokens = torch.zeros(2, dtype=torch.long)
    model._dec_text_mask = torch.zeros(2, dtype=torch.long)
    model._dec_phoneme_tokens = torch.zeros(2, 1, dtype=torch.long)
    model._dec_phoneme_valid = torch.zeros(2, dtype=torch.long)
    model._dec_audio_codes = torch.zeros(2, 1, dtype=torch.long)
    model._dec_audio_valid = torch.zeros(2, dtype=torch.long)
    model._acoustic_temperature = torch.zeros(2)
    model._acoustic_top_k = torch.zeros(2, dtype=torch.long)
    model._cfg_scale = torch.ones(2)
    model.text_embedding = nn.Embedding(4, 2)
    model._embed_phoneme = lambda ids: torch.full((ids.shape[0], 2), 100.0)
    model.code_predictor.embed_audio_frame = lambda codes: codes.float().expand(-1, 2)
    model.code_predictor.temperature = 0.7
    model.code_predictor.top_k = 80
    info = {
        "use_cfg": True,
        "cfg_scale": 2.5,
        "temperature": 0.0,
        "top_k": 1,
        "_omni_is_prefill": False,
        "text_tokens": [1, 2, 3],
        "decode_offset": 2,
        "last_phoneme_token": torch.tensor([1]),
        "last_audio_codes": torch.tensor([5]),
    }
    ids = torch.zeros(2, dtype=torch.long)
    for row, branch in enumerate(("unconditional", "conditional")):
        model.preprocess(ids[row : row + 1], None, start=row, **info, **{CFG_PAIR_METADATA_KEY: {"branch": branch}})
    assert model._dec_text_mask.tolist() == [0, 1]
    assert model._dec_phoneme_valid.tolist() == [0, 1]
    combined = torch.zeros(2, 2)
    model._assemble_decode_embeddings(combined, slice(0, 2))
    torch.testing.assert_close(combined[0], torch.tensor([5.0, 5.0]))
    assert model._acoustic_temperature.tolist() == [0.0, 0.0]
    assert model._acoustic_top_k.tolist() == [1, 1]
    assert model.code_predictor.temperature == 0.7
    assert model.code_predictor.top_k == 80


@pytest.mark.parametrize('branch', ['conditional', 'unconditional'])
def test_paired_stream_append_uses_decode_even_when_engine_marks_grown_prompt_as_prefill(branch):
    model = _input_model()
    model.arch = SimpleNamespace(cfg_capability={'version': 1})
    model.has_phoneme = False
    model._dec_text_mask = torch.ones(1, dtype=torch.long)
    calls = []
    model._preprocess_decode = lambda *args: calls.append('decode') or 'decoded'
    model._preprocess_prefill = lambda *args: calls.append('prefill') or 'speaker prefill'
    result = model.preprocess(
        torch.zeros(1, dtype=torch.long),
        None,
        start=0,
        use_cfg=True,
        _cfg_pair={'branch': branch},
        _cfg_stream_prompt_len=7,
        _omni_is_prefill=True,
        _omni_num_computed_tokens=7,
    )
    assert result == 'decoded'
    assert calls == ['decode']
    assert model._dec_text_mask.item() == (0 if branch == 'unconditional' else 1)


def test_phoneme_predictions_only_use_conditional_rows_including_prefill_tail():
    model = _input_model()
    model.cfg_guidance_layout = CFGPairingLayout(("r",), ((2, 3),), ((0, 1),))
    model.arch = SimpleNamespace(phoneme_stacking_factor=1, phoneme_vocab_size=3)
    model.phoneme_final_proj = nn.Linear(2, 3)
    model.phoneme_confidence_unk_threshold = 0
    model._dec_phoneme_tokens = torch.full((4, 1), -1, dtype=torch.long)
    model._dec_phoneme_valid = torch.zeros(4, dtype=torch.long)
    model._predict_phonemes(torch.ones(4, 2), torch.tensor([1, 3]))
    assert model._dec_phoneme_tokens[:3].tolist() == [[-1], [-1], [-1]]
    assert model._dec_phoneme_valid.tolist() == [0, 0, 0, 1]


def test_next_backbone_step_copies_public_audio_feedback_before_either_branch_runs():
    model = _input_model()
    model.cfg_guidance_layout = CFGPairingLayout(("r",), ((1,),), ((0,),))
    model.has_phoneme = False
    model.audio_eos_id = 7
    model._combined_embeddings = torch.zeros(2, 2)
    model._dec_audio_codes = torch.tensor([[99], [5]])
    model._dec_audio_valid = torch.tensor([0, 1])
    model._dec_text_tokens = torch.zeros(2, dtype=torch.long)
    model._dec_text_mask = torch.zeros(2, dtype=torch.long)
    model._out_codes = torch.zeros(2, 1, dtype=torch.long)
    model._token_stop = torch.zeros(2, dtype=torch.bool)
    model._sample_stop = torch.zeros(2, dtype=torch.bool)
    model.text_embedding = nn.Embedding(4, 2)
    model._get_query_dispatch = lambda: (None, 0, None)
    model.code_predictor.embed_audio_frame = lambda codes: codes.float().expand(-1, 2)
    model.code_predictor.generate_codes = lambda *args, **kwargs: torch.tensor([[2], [2]])

    class Backbone(nn.Module):
        def forward(self, **kwargs):
            self.inputs = kwargs["inputs_embeds"].clone()
            return kwargs["inputs_embeds"]

    model.backbone = Backbone()
    with torch.inference_mode():
        model.forward(torch.zeros(2, dtype=torch.long), positions=torch.tensor([3, 3]))
    assert model._dec_audio_codes.tolist() == [[5], [5]]
    torch.testing.assert_close(model.backbone.inputs, torch.full((2, 2), 5.0))


def test_unconditional_prefill_replaces_complete_context_and_zeros_warmup():
    model = _input_model()
    model._build_prefill_embeds = lambda *args, **kwargs: torch.arange(8.0).reshape(4, 2)
    model._build_text_prefill_embeds = lambda *args: torch.ones(2, 2)
    model._is_user_audio_prefill_chunk = lambda **kwargs: False
    _, rows, update = model._preprocess_prefill(
        torch.zeros(6, dtype=torch.long),
        None,
        6,
        torch.device("cpu"),
        {
            "speaker_id": "eng",
            "use_cfg": True,
            CFG_PAIR_METADATA_KEY: {"branch": "unconditional"},
            "_omni_num_computed_tokens": 0,
            "_omni_prompt_len": 6,
            "text_prefill_num": 2,
        },
    )
    torch.testing.assert_close(rows, torch.tensor([[9.0, 10.0]] * 4 + [[0.0, 0.0]] * 2))
    assert update["decode_offset"] == 2


def test_cfg_weight_is_required_when_advertised_and_validated_before_loading():
    model = _input_model()
    model.arch = SimpleNamespace(cfg_capability={"version": 1, "context_cas_disabled": False})
    with pytest.raises(ValueError, match="missing cfg_context_embedding"):
        model.load_weights([])
    with pytest.raises(ValueError, match="finite floating"):
        model.load_weights([("cfg_context_embedding", torch.ones(1, 2, dtype=torch.long))])
    model.backbone = SimpleNamespace(load_weights=lambda weights: set())
    assert model.load_weights([("cfg_context_embedding", torch.tensor([[1.0, 2.0]]))]) == {"cfg_context_embedding"}
    torch.testing.assert_close(model.cfg_context_embedding, torch.tensor([[1.0, 2.0]]))


def _input_model():
    model = EasyMagpieTTSForConditionalGeneration.__new__(EasyMagpieTTSForConditionalGeneration)
    nn.Module.__init__(model)
    model.embedding_dim = 2
    model.cfg_context_embedding = torch.tensor([[9.0, 10.0]])
    model.code_predictor = SimpleNamespace(requires_all_tokens=True)
    model._combined_embeddings = torch.zeros(6, 2)
    return model


def _predictor(logits, device="cpu"):
    class Block(nn.Module):
        num_codes = 1

        def __init__(self, values, index):
            super().__init__()
            self.values = values
            self.codebook_indices = (index,)

        def forward(self, hidden, positions):
            self.inputs = hidden.clone()
            return hidden

        def compute_logits(self, hidden):
            return self.values

    class Embedder(nn.Module):
        def __init__(self):
            super().__init__()
            self.inputs = []

        def embed_codebooks(self, codes, indices):
            self.inputs.append(codes.clone())
            return codes.float().expand(-1, 2)

    predictor = EasyMagpieAcousticCodesPredictor.__new__(EasyMagpieAcousticCodesPredictor)
    nn.Module.__init__(predictor)
    predictor.arch = SimpleNamespace(codebook_size=3, audio_eos_id=7, mask_token_id=8)
    predictor.num_codebooks = len(logits)
    predictor.num_output_tokens = 4
    predictor.temperature, predictor.top_k = 0.7, 80
    predictor.blocks = nn.ModuleList([Block(values, index) for index, values in enumerate(logits)])
    predictor.audio_embedder = Embedder()
    predictor._gumbel_buf = torch.zeros(logits[0].shape[0], 1, 4, device=device)
    return predictor, predictor.blocks, predictor.audio_embedder
