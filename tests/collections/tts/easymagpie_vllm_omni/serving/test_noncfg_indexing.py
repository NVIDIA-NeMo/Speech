# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unpaired predictor sampling preserves feedback without identity row indexing."""
from __future__ import annotations

import pytest
import torch
from test_cfg_synthesis import _predictor
from torch.utils._python_dispatch import TorchDispatchMode


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("batch", [1, 4])
@pytest.mark.parametrize("sampling", ["fixed", "default", "custom", "greedy"])
def test_unpaired_sampling_avoids_identity_indexing(batch: int, sampling: str, device: str) -> None:
    """Check physical-row sampling, masks, feedback, RNG and redundant operations."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(41)
    logits = [torch.randn(batch, 1, 4, device=device) for _ in range(3)]
    logits[0][..., 0] = float("nan")
    logits[1][..., 1] = float("inf")
    logits[2][..., 3] = 100  # Compact EOS must be expanded before feedback.
    predictor, blocks, embedder = _predictor(logits, device)
    predictor._frame_gumbel_buf = torch.zeros(batch, 3, 4, device=device)
    predict = torch.arange(batch, device=device) % 2 == 0
    settings = {}
    if sampling == "custom":
        settings = dict(
            temperature=torch.linspace(0.1, 1.2, batch, device=device),
            top_k=torch.arange(batch, device=device) % 4 + 1,
        )
    elif sampling == "greedy":
        settings = dict(temperature=0.0, top_k=4)
    fixed = sampling == "fixed"
    expected_columns = []
    torch.manual_seed(19)
    if fixed:
        noise = torch.empty(batch, 3, 4, device=device).uniform_(1e-20, 1.0 - 1e-20)
        noise.log_().neg_().log_().neg_()
    for index, values in enumerate(logits):
        values = torch.nan_to_num(values, nan=0.0, posinf=100.0, neginf=-100.0).clamp(-100, 100)
        if fixed:
            selected, indices = values.topk(4, dim=-1)
            picked = (selected / 0.7 + noise[:, index : index + 1]).argmax(-1, keepdim=True)
            sampled = indices.gather(-1, picked).squeeze(-1)
        else:
            temperature = settings.get("temperature", predictor.temperature)
            top_k = settings.get("top_k", predictor.top_k)
            if isinstance(temperature, torch.Tensor):
                temperature = temperature.reshape(-1, 1, 1)
            if isinstance(top_k, torch.Tensor):
                top_k = top_k.reshape(-1, 1, 1)
            sampled = predictor._sample_codes(values, temperature=temperature, top_k=top_k)
        sampled = torch.where(sampled == 3, 7, sampled)
        expected_columns.append(torch.where(predict[:, None], sampled, 8))
    expected_rng = (torch.cuda.get_rng_state() if device == "cuda" else torch.get_rng_state()).clone()
    torch.manual_seed(19)
    positions = torch.arange(batch, device=device)
    with _RowOperations() as operations:
        actual = predictor.generate_codes(
            torch.zeros(batch, 2, device=device),
            positions=positions,
            predict=predict,
            fixed_sampling=fixed,
            **settings,
        )
    assert torch.equal(actual, torch.cat(expected_columns, dim=1))
    assert torch.equal((torch.cuda.get_rng_state() if device == "cuda" else torch.get_rng_state()), expected_rng)
    hidden = torch.zeros(batch, 2, device=device)
    for index, block in enumerate(blocks):
        if index:
            assert torch.equal(embedder.inputs[index - 1], expected_columns[index - 1])
            hidden = hidden + expected_columns[index - 1].float().expand(-1, 2)
        assert torch.equal(block.inputs, hidden)
    assert not operations.redundant, operations.redundant


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("fixed", [False, True])
def test_unpaired_graph_replay_reads_runtime_prediction_mask(fixed: bool) -> None:
    """Replay with changed masks and verify exact eager output and RNG state."""
    logits = [torch.zeros(4, 1, 4, device="cuda") for _ in range(3)]
    predictor, _, _ = _predictor(logits, "cuda")
    predictor._frame_gumbel_buf = torch.zeros(4, 3, 4, device="cuda")
    hidden = torch.zeros(4, 2, device="cuda")
    positions = torch.arange(4, device="cuda")
    predict = torch.ones(4, dtype=torch.bool, device="cuda")

    def run():
        return predictor.generate_codes(hidden, positions=positions, predict=predict, fixed_sampling=fixed)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = run()
    for mask in ([True, False, True, False], [False] * 4, [True] * 4):
        predict.copy_(torch.tensor(mask, device="cuda"))
        state = torch.cuda.get_rng_state()
        expected = run().clone()
        expected_rng = torch.cuda.get_rng_state()
        torch.cuda.set_rng_state(state)
        graph.replay()
        assert torch.equal(actual, expected)
        assert torch.equal(torch.cuda.get_rng_state(), expected_rng)


class _RowOperations(TorchDispatchMode):
    """Record advanced row indexing and mapping allocation in the predictor."""

    def __init__(self) -> None:
        super().__init__()
        self.redundant = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func in (torch.ops.aten.index.Tensor, torch.ops.aten.index_put_.default, torch.ops.aten.arange.default):
            self.redundant.append(str(func))
        return func(*args, **(kwargs or {}))
