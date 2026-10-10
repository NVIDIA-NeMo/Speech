# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import copy
from unittest.mock import Mock

import pytest
import torch

from nemo.collections.tts.modules.magpietts_modules import CharAwareSubwordEncoder


def _encoder():
    return CharAwareSubwordEncoder(
        d_embed=32,
        llm_tokenizer_vocab={"a": 0, "b": 1, "aa": 2, "ab": 3, "bbb": 4, "abababab": 5},
        subword_padding_idx=7,
        special_vocab={"<eos>": 6},
    )


@pytest.mark.unit
@pytest.mark.parametrize("mask_kind", ["padding", "expanded", "none"])
def test_cas_chunks_preserve_outputs_and_gradients(mask_kind):
    torch.manual_seed(7)
    legacy = _encoder()
    chunked = copy.deepcopy(legacy)
    ids = torch.tensor([[0, 5, 2, 6, 7], [3, 4, 1, 5, 6]])
    mask = ids.ne(7) if mask_kind != "none" else None
    if mask_kind == "expanded":
        mask = mask.unsqueeze(-1)
    calls = []
    handle = chunked.encoder.register_forward_pre_hook(
        lambda module, args, kwargs: calls.append(kwargs["x"].shape[0]), with_kwargs=True
    )
    expected = legacy(ids, subword_mask=mask)
    actual = chunked(ids, subword_mask=mask, max_subwords_per_chunk=3)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert max(calls) <= 3
    expected.square().sum().backward()
    actual.square().sum().backward()
    handle.remove()
    for (name, param), (actual_name, actual_param) in zip(legacy.named_parameters(), chunked.named_parameters()):
        assert name == actual_name
        torch.testing.assert_close(actual_param.grad, param.grad, rtol=1e-4, atol=1e-6)
    assert chunked.embed_tokens.weight.grad.abs().sum() > 0
    assert legacy.state_dict().keys() == chunked.state_dict().keys()


@pytest.mark.unit
def test_cas_chunks_checkpoint_only_during_training(monkeypatch):
    import nemo.collections.tts.modules.magpietts_modules as modules

    encoder = _encoder()
    ids = torch.tensor([[0, 5, 3, 6]])
    real_checkpoint = modules.checkpoint
    checkpoint = Mock(wraps=real_checkpoint)
    monkeypatch.setattr(modules, "checkpoint", checkpoint)
    encoder(ids, max_subwords_per_chunk=2).sum().backward()
    assert checkpoint.call_count == 2
    assert all(call.kwargs["use_reentrant"] is False for call in checkpoint.call_args_list)
    checkpoint.reset_mock()
    encoder.eval()
    with torch.no_grad():
        actual = encoder(ids, max_subwords_per_chunk=2)
        expected = encoder(ids)
    torch.testing.assert_close(actual, expected)
    checkpoint.assert_not_called()


@pytest.mark.unit
def test_cas_chunks_handle_empty_mask():
    encoder = _encoder()
    ids = torch.full((2, 5), 7)
    actual = encoder(ids, subword_mask=torch.zeros_like(ids), max_subwords_per_chunk=2)
    assert actual.shape == (2, 5, 32)
    assert actual.count_nonzero() == 0


@pytest.mark.unit
@pytest.mark.parametrize("chunk_size", [-1, 1.5, True])
def test_cas_chunks_reject_invalid_sizes(chunk_size):
    with pytest.raises(ValueError, match="max_subwords_per_chunk"):
        _encoder()(torch.tensor([[0]]), max_subwords_per_chunk=chunk_size)
