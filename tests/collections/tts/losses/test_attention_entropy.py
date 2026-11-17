# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
import unittest

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from nemo.collections.tts.losses.attention_entropy import (
    _entropy_chunk_sum,
    attention_entropy,
    capture_attention_projections,
    mean_attention_entropy,
)


class TestAttentionEntropy(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_known_attention_distributions(self):
        for probabilities in ([0.25] * 4, [0.7, 0.1, 0.1, 0.1], [0.97, 0.01, 0.01, 0.01], [1.0, 0.0, 0.0, 0.0]):
            with self.subTest(probabilities=probabilities):
                # A query of one produces logits log(p); test the final causal row.
                query = torch.ones(1, 1, 1, 1, 1, requires_grad=True)
                key = torch.tensor(probabilities).clamp_min(1e-30).log().reshape(1, 1, 4, 1)
                result = _entropy_chunk_sum(query, key, torch.ones(1, 4, dtype=torch.bool), 3)
                expected = -sum(p * math.log(p) for p in probabilities if p > 0) / math.log(4)
                self.assertAlmostEqual(result.item(), expected, places=6)
                result.backward()
                self.assertTrue(torch.isfinite(query.grad).all())

    def test_uniform_attention_is_one(self):
        query = torch.zeros(2, 7, 8, requires_grad=True)
        key = torch.zeros(2, 7, 4, requires_grad=True)
        mask = torch.arange(7)[None] < torch.tensor([7, 3])[:, None]
        result = attention_entropy(query, key, mask, 4, 2, chunk_size=2)
        self.assertAlmostEqual(result.item(), 1.0, places=6)
        result.backward()
        self.assertTrue(torch.isfinite(query.grad).all())

    def test_focused_attention_is_near_zero(self):
        query = torch.ones(1, 5, 1)
        key = torch.arange(5).float().reshape(1, 5, 1) * 30
        result = attention_entropy(query, key, torch.ones(1, 5, dtype=torch.bool), 1, 1)
        self.assertLess(result.item(), 1e-8)

    def test_padding_and_forced_rows_have_no_gradient(self):
        for mask in (torch.zeros(1, 5, dtype=torch.bool), torch.tensor([[True, False, False, False, False]])):
            query = torch.randn(1, 5, 4, requires_grad=True)
            key = torch.randn(1, 5, 2, requires_grad=True)
            result = attention_entropy(query, key, mask, 2, 1, chunk_size=2)
            self.assertEqual(result.item(), 0.0)
            result.backward()
            self.assertEqual(query.grad.abs().sum().item(), 0.0)
            self.assertEqual(key.grad.abs().sum().item(), 0.0)

    def test_padding_does_not_change_entropy(self):
        query = torch.randn(1, 7, 4, requires_grad=True)
        key = torch.randn(1, 7, 2, requires_grad=True)
        result = attention_entropy(query, key, torch.tensor([[True] * 4 + [False] * 3]), 2, 1)
        reference = attention_entropy(query[:, :4], key[:, :4], torch.ones(1, 4, dtype=torch.bool), 2, 1)
        torch.testing.assert_close(result, reference)
        result.backward()
        self.assertEqual(query.grad[:, 4:].abs().sum().item(), 0.0)
        self.assertEqual(key.grad[:, 4:].abs().sum().item(), 0.0)

    def test_chunked_gqa_values_and_gradients_match_dense_reference(self):
        query = torch.randn(2, 7, 8, requires_grad=True)
        key = torch.randn(2, 7, 4, requires_grad=True)
        mask = torch.arange(7)[None] < torch.tensor([7, 4])[:, None]
        expected = _dense_reference(query, key, mask, 4, 2)
        expected_grads = torch.autograd.grad(expected, (query, key), retain_graph=True)
        for size in (1, 3, 32):
            actual = attention_entropy(query, key, mask, 4, 2, chunk_size=size)
            actual_grads = torch.autograd.grad(actual, (query, key), retain_graph=True)
            torch.testing.assert_close(actual, expected)
            for actual_grad, expected_grad in zip(actual_grads, expected_grads):
                torch.testing.assert_close(actual_grad, expected_grad)

    def test_minimizing_entropy_sharpens_attention(self):
        query = torch.randn(1, 5, 4, requires_grad=True)
        key = torch.randn(1, 5, 4, requires_grad=True)
        mask = torch.ones(1, 5, dtype=torch.bool)
        before = attention_entropy(query, key, mask, 2, 2)
        dq, dk = torch.autograd.grad(before, (query, key))
        after = attention_entropy(query - 0.1 * dq, key - 0.1 * dk, mask, 2, 2)
        self.assertLess(after.item(), before.item())

    def test_bfloat16_autocast_is_finite_and_uses_fp32(self):
        query = torch.randn(1, 5, 4, dtype=torch.bfloat16, requires_grad=True)
        key = torch.randn(1, 5, 2, dtype=torch.bfloat16, requires_grad=True)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            result = attention_entropy(query, key, torch.ones(1, 5, dtype=torch.bool), 2, 1)
        self.assertEqual(result.dtype, torch.float32)
        result.backward()
        self.assertTrue(torch.isfinite(query.grad).all() and torch.isfinite(key.grad).all())

    def test_capture_preserves_forward_and_limits_gradient_scope(self):
        module = _TinyAttention()
        inputs = torch.randn(1, 5, 4)
        expected = module(inputs)
        with capture_attention_projections([module]) as projections:
            actual = module(inputs)
        torch.testing.assert_close(actual, expected)
        entropy = mean_attention_entropy(projections, torch.ones(1, 5, dtype=torch.bool))
        entropy.backward()
        self.assertGreater(module.q_proj.weight.grad.abs().sum().item(), 0.0)
        self.assertGreater(module.k_proj.weight.grad.abs().sum().item(), 0.0)
        self.assertIsNone(module.v_proj.weight.grad)
        self.assertIsNone(module.o_proj.weight.grad)
        self.assertFalse(module.q_proj._forward_hooks or module.k_proj._forward_hooks)

    def test_capture_cleanup_on_exception_and_disabled_capture(self):
        module = _TinyAttention()
        with self.assertRaisesRegex(RuntimeError, 'deliberate'):
            with capture_attention_projections([module]):
                raise RuntimeError('deliberate')
        self.assertFalse(module.q_proj._forward_hooks or module.k_proj._forward_hooks)
        with capture_attention_projections([module], enabled=False) as projections:
            module(torch.randn(1, 5, 4))
        self.assertEqual(projections, [])
        self.assertEqual(mean_attention_entropy(projections, torch.ones(1, 5, dtype=torch.bool)).item(), 0.0)

    def test_decoder_checkpoint_recomputation_matches_without_checkpoint(self):
        module = _TinyAttention()
        inputs = torch.randn(1, 5, 4, requires_grad=True)
        results = []
        for use_checkpoint in (False, True):
            with capture_attention_projections([module]) as projections:
                output = checkpoint(module, inputs, use_reentrant=False) if use_checkpoint else module(inputs)
            entropy = mean_attention_entropy(projections, torch.ones(1, 5, dtype=torch.bool))
            loss = output.square().mean() + 0.1 * entropy
            gradients = torch.autograd.grad(loss, (inputs, *module.parameters()))
            results.append((loss, gradients))
        torch.testing.assert_close(results[0][0], results[1][0])
        for first, second in zip(results[0][1], results[1][1]):
            torch.testing.assert_close(first, second)


class _TinyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_heads = 2
        self.num_key_value_heads = 1
        self.q_proj = nn.Linear(4, 4, bias=False)
        self.k_proj = nn.Linear(4, 2, bias=False)
        self.v_proj = nn.Linear(4, 2, bias=False)
        self.o_proj = nn.Linear(4, 4, bias=False)

    def forward(self, inputs):
        batch, time, _ = inputs.shape
        query = self.q_proj(inputs).reshape(batch, time, 2, 2).transpose(1, 2)
        key = self.k_proj(inputs).reshape(batch, time, 1, 2).transpose(1, 2)
        value = self.v_proj(inputs).reshape(batch, time, 1, 2).transpose(1, 2)
        output = torch.nn.functional.scaled_dot_product_attention(query, key, value, is_causal=True, enable_gqa=True)
        return self.o_proj(output.transpose(1, 2).reshape(batch, time, 4))


def _dense_reference(query, key, valid_mask, heads, kv_heads):
    batch, time, width = query.shape
    dim = width // heads
    query = query.reshape(batch, time, heads, dim).transpose(1, 2)
    key = key.reshape(batch, time, kv_heads, dim).transpose(1, 2).repeat_interleave(heads // kv_heads, dim=1)
    scores = query @ key.transpose(-1, -2) / math.sqrt(dim)
    rows = []
    for b in range(batch):
        for head in range(heads):
            for position in range(time):
                keys = valid_mask[b, : position + 1]
                if valid_mask[b, position] and keys.sum() > 1:
                    log_p = scores[b, head, position, : position + 1][keys].log_softmax(-1)
                    rows.append(-(log_p.exp() * log_p).sum() / math.log(keys.sum().item()))
    return torch.stack(rows).mean()
