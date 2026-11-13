# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Training rollout conditioning and action metadata for the bundled engine."""
from types import SimpleNamespace

import pytest
import torch
from easymagpie_vllm_omni.acoustic_predictor import sample_grpo_logits
from easymagpie_vllm_omni.easymagpie import EasyMagpieTTSForConditionalGeneration
from torch import nn

pytestmark = pytest.mark.unit


def _model():
    model = EasyMagpieTTSForConditionalGeneration.__new__(EasyMagpieTTSForConditionalGeneration)
    nn.Module.__init__(model)
    model.grpo_rollout = True
    model.arch = SimpleNamespace(text_prefill_num=4, phoneme_stacking_factor=1, phoneme_vocab_size=12)
    model.embedding_dim = model.hidden_dim = 3
    model.num_codebooks = 2
    model.has_phoneme = True
    model.phonemes_delay = 3
    model.speech_delay = 5
    model.phoneme_bos_id = 7
    model.phoneme_eos_id = 8
    model.phoneme_confidence_unk_threshold = 0.0
    model._single_stage_audio = True
    model.text_embedding = nn.Embedding(32, 3)
    model.phoneme_embeddings = nn.ModuleList([nn.Embedding(12, 3)])
    model.phoneme_final_proj = nn.Linear(3, 12)
    with torch.no_grad():
        model.text_embedding.weight.zero_()
        model.phoneme_embeddings[0].weight.zero_()
        model.phoneme_embeddings[0].weight[7] = torch.tensor([0.0, 0.0, 9.0])
        model.phoneme_final_proj.weight.zero_()
        model.phoneme_final_proj.bias.zero_()
        model.phoneme_final_proj.bias[8] = 10.0
    model.code_predictor = SimpleNamespace(
        temperature=0.7,
        top_k=1,
        last_sampled_logprobs=torch.zeros(16, 2),
        embed_audio_frame=lambda codes: torch.zeros(codes.size(0), 3),
    )
    model._combined_embeddings = torch.zeros(16, 3)
    model._dec_text_tokens = torch.zeros(16, dtype=torch.long)
    model._dec_text_mask = torch.zeros(16, dtype=torch.long)
    model._dec_audio_codes = torch.zeros(16, 2, dtype=torch.long)
    model._dec_audio_valid = torch.zeros(16, dtype=torch.long)
    model._dec_phoneme_tokens = torch.zeros(16, 1, dtype=torch.long)
    model._dec_phoneme_valid = torch.zeros(16, dtype=torch.long)
    model._out_codes = torch.full((16, 2), -1, dtype=torch.long)
    model._grpo_text_embeddings = torch.zeros(16, 3)
    model._grpo_text_override = torch.zeros(16, dtype=torch.bool)
    model._grpo_audio_valid = torch.zeros(16, dtype=torch.bool)
    model._grpo_phoneme_codes = torch.zeros(16, 1, dtype=torch.long)
    model._grpo_phoneme_valid = torch.zeros(16, dtype=torch.bool)
    model._grpo_prefill_phoneme_valid = torch.zeros(16, dtype=torch.bool)
    model._grpo_phoneme_logps = torch.zeros(16, 1)
    model._grpo_phoneme_sample = torch.zeros(16, dtype=torch.bool)
    return model


def _prompt(context, text, ids, offset=0):
    return dict(
        grpo_rollout=True,
        grpo_context_embeddings=context,
        grpo_text_embeddings=text,
        grpo_text_token_ids=ids,
        phoneme_sampling_method="sample",
        _omni_is_prefill=True,
        _omni_num_computed_tokens=offset,
        _omni_prompt_len=context.size(0) + 4,
    )


def test_grpo_prefill_uses_current_conditioning_and_zero_pads_short_text():
    model = _model()
    context = torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    text = torch.tensor([[3.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    _, rows, update = model.preprocess(torch.zeros(6, dtype=torch.long), None, **_prompt(context, text, [10, 11]))

    torch.testing.assert_close(
        rows,
        torch.tensor([[1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0], [0, 0, 0], [0, 0, 9]], dtype=torch.float32),
    )
    assert update['text_tokens'] == [10, 11]
    assert update['decode_offset'] == 4
    assert model._grpo_prefill_phoneme_valid[:6].tolist() == [False] * 5 + [True]


def test_chunked_context_prefill_does_not_report_a_phoneme_action():
    model = _model()
    context, text = torch.ones(5, 3), torch.ones(1, 3)
    model.preprocess(torch.zeros(3, dtype=torch.long), None, **_prompt(context, text, [10]))
    model._predict_phonemes(torch.zeros(3, 3), torch.tensor([2]), prefill=True)
    assert not model._grpo_phoneme_valid[:3].any()


def test_decode_uses_current_text_embeddings_instead_of_the_baked_table():
    model = _model()
    context = torch.ones(2, 3)
    text = torch.tensor([[index, 0, 0] for index in range(10, 16)], dtype=torch.float32)
    _, _, update = model.preprocess(
        torch.zeros(6, dtype=torch.long), None, **_prompt(context, text, list(range(10, 16)))
    )
    model._preprocess_decode(torch.zeros(1, dtype=torch.long), 0, torch.device('cpu'), update)
    combined = torch.zeros(1, 3)
    model._assemble_decode_embeddings(combined, slice(0, 1))
    torch.testing.assert_close(combined[0], text[4])


def test_packed_actions_keep_first_prefill_phoneme_and_mixed_audio_eos():
    model = _model()
    model.preprocess(torch.zeros(6, dtype=torch.long), None, **_prompt(torch.ones(2, 3), torch.ones(1, 3), [10]))
    model._predict_phonemes(torch.zeros(6, 3), torch.tensor([5]), prefill=True)
    model._out_codes[5] = torch.tensor([33, 2])
    model._grpo_audio_valid[5] = True
    model.code_predictor.last_sampled_logprobs[5] = torch.tensor([-0.2, -0.3])

    output = model.make_omni_output(torch.zeros(6, 3)).multimodal_outputs['model_outputs']

    assert output.dtype == torch.float32 and output.shape == (6, 8)
    torch.testing.assert_close(output[5], torch.tensor([33, 2, 8, 1, 1, -0.2, -0.3, 0], dtype=torch.float32))
    assert output[:5, 3:5].sum() == 0


def test_ordinary_code_output_remains_unchanged_when_grpo_is_disabled():
    model = _model()
    model.grpo_rollout = False
    expected = torch.tensor([[1, 2], [33, 4]])
    model._out_codes[:2] = expected
    output = model.make_omni_output(torch.zeros(2, 3)).multimodal_outputs['model_outputs']
    assert output.dtype == torch.long
    torch.testing.assert_close(output, expected)


def test_grpo_sampling_keeps_topk_boundary_ties_and_matches_selected_logps():
    torch.manual_seed(11)
    logits = torch.tensor([[3.0, 2.0, 2.0, -1.0]], dtype=torch.bfloat16).expand(2048, -1)
    actions, logps = sample_grpo_logits(logits, temperature=0.7, top_k=2)
    expected = (logits.float() / 0.7).masked_fill(logits < 2.0, float('-inf')).log_softmax(-1)
    assert actions.max() == 2 and (actions == 1).any() and (actions == 2).any()
    assert logps.dtype == torch.float32 and logps.isfinite().all()
    torch.testing.assert_close(logps, expected.gather(-1, actions[:, None]).squeeze(-1))


def test_grpo_argmax_sanitizes_logits_and_has_zero_behavior_log_probability():
    actions, logps = sample_grpo_logits(
        torch.tensor([[float('nan'), float('inf'), float('-inf'), 0.0]]),
        temperature=0.7,
        top_k=1,
        sampling_method='argmax',
    )
    assert actions.tolist() == [1] and logps.tolist() == [0.0]
