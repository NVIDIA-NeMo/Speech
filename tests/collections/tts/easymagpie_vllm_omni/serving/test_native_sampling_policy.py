# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Behavioral checks for NeMo's sanitized, tie-preserving audio sampling policy."""
from types import SimpleNamespace

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('vllm')

from easymagpie_vllm_omni.local_transformer import EasyMagpieCodeLoop, _generate_cfg_codes  # noqa: E402
from torch import nn  # noqa: E402


def test_cfg_clips_after_guidance_before_masking_and_temperature():
    predictor = _predictor()
    conditional = torch.tensor([[60.0, 44.0, 38.0, 0.0, 400.0]])
    unconditional = torch.zeros_like(conditional)
    # After NeMo's clamp, tokens 0 and 1 both have logit 100. Token 1 wins
    # with this noise; forbidden token 4 remains impossible despite huge noise.
    noise = torch.tensor([[[0.0, 10.0, 0.0, 0.0, 10000.0]]])
    codes, _ = _generate_cfg_codes(
        predictor, conditional, unconditional, noise, torch.tensor([0.7]), torch.tensor([[2.5]]), trace=False
    )
    assert codes.item() == 1


def test_unguided_sampling_uses_the_same_sanitization():
    from conftest import build_vllm_config

    predictor = _predictor()
    loop = EasyMagpieCodeLoop(vllm_config=build_vllm_config())
    loop.bind_predictor(predictor)
    loop.num_codebooks = predictor.num_codebooks
    loop.lt_hidden = predictor.lt_hidden
    logits = torch.tensor([[150.0, 110.0, 95.0, 0.0, 1000.0]])
    codes = loop.forward(logits, torch.tensor([[[0.0, 10.0, 0.0, 0.0, 10000.0]]]), torch.tensor([0.7]))
    assert codes.item() == 1


def test_nonfinite_guided_logits_are_cleaned_without_unmasking_special_tokens():
    predictor = _predictor()
    conditional = torch.tensor([[float('nan'), 3.0, 0.0, -1.0, float('inf')]])
    codes, _ = _generate_cfg_codes(
        predictor,
        conditional,
        torch.zeros_like(conditional),
        torch.zeros(1, 1, 5),
        torch.tensor([0.7]),
        torch.tensor([[2.5]]),
        trace=False,
    )
    assert codes.item() == 1


def test_topk_keeps_every_token_tied_at_the_cutoff():
    predictor = _predictor(top_k=2)
    conditional = torch.tensor([[3.0, 2.0, 2.0, 1.0, 1000.0]])
    # Both tokens tied at the second-largest logit belong to native top-k's
    # support. Noise is associated with vocabulary IDs, not arbitrary top-k ranks.
    codes, _ = _generate_cfg_codes(
        predictor,
        conditional,
        torch.zeros_like(conditional),
        torch.tensor([[[0.0, 0.0, 10.0, 0.0, 10000.0]]]),
        torch.tensor([0.7]),
        torch.tensor([[1.0]]),
        trace=False,
    )
    assert codes.item() == 2


def test_temperature_zero_uses_native_argmax_even_when_clipping_creates_ties():
    predictor = _predictor()
    conditional = torch.tensor([[60.0, 44.0, 38.0, 0.0, 400.0]])
    codes, _ = _generate_cfg_codes(
        predictor,
        conditional,
        torch.zeros_like(conditional),
        torch.tensor([[[0.0, 1000.0, 0.0, 0.0, 10000.0]]]),
        torch.tensor([0.0]),
        torch.tensor([[2.5]]),
        trace=False,
    )
    assert codes.item() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA graph replay requires a GPU')
def test_sanitized_cfg_graph_reads_updated_temperature_and_scale():
    with torch.device('cuda'):
        predictor = _predictor()
        conditional = torch.tensor([[60.0, 44.0, 38.0, 0.0, 400.0]])
        unconditional = torch.zeros_like(conditional)
        noise = torch.tensor([[[0.0, 10.0, 0.0, 0.0, 10000.0]]])
        temperature = torch.tensor([0.7])
        scale = torch.tensor([[2.5]])
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            _generate_cfg_codes(predictor, conditional, unconditional, noise, temperature, scale, trace=False)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result, _ = _generate_cfg_codes(
                predictor, conditional, unconditional, noise, temperature, scale, trace=False
            )
        graph.replay()
        assert result.item() == 1
        temperature.zero_()
        graph.replay()
        assert result.item() == 0
        temperature.fill_(0.7)
        scale.fill_(1)
        graph.replay()
        assert result.item() == 0


def _predictor(top_k=5):
    return SimpleNamespace(
        num_codebooks=1,
        lt_hidden=5,
        _sample_top_k=top_k,
        local_transformer_in_projection=nn.Identity(),
        local_transformer=nn.Identity(),
        local_transformer_audio_out_projection=nn.Identity(),
        local_transformer_out_projections=[nn.Identity()],
        forbidden_mask=torch.tensor([False, False, False, False, True]),
    )
