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

import builtins
import copy
import gc

import pytest
import torch
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

from nemo.collections.asr.modules import transformer_encoder as transformer_encoder_module
from nemo.collections.asr.modules.moe_transformer_encoder import MoEFeedForward, MoETransformerEncoder
from nemo.collections.asr.modules.transformer_encoder import (
    MultiHeadAttention,
    TransformerEncoder,
    TransformerEncoderConfig,
)
from nemo.collections.asr.parts.packed_sequence import (
    PackedEncoderActivations,
    pack_encoder_output,
    packed_encoder_position_ids,
    split_encoder_output,
    split_packed_data,
    unpack_encoder_output,
)


def _supports_cuda_varlen_flash_attention():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8:
        return False
    return transformer_encoder_module._get_flash_attention_varlen() is not None


requires_cuda_varlen_flash_attention = pytest.mark.skipif(
    not _supports_cuda_varlen_flash_attention(), reason="requires an SM80+ CUDA varlen FlashAttention provider"
)


def test_packed_encoder_output_round_trip_and_positions():
    padded = torch.arange(3 * 5 * 4, dtype=torch.float32).reshape(3, 5, 4)[..., ::2]
    assert not padded.is_contiguous()
    lengths = torch.tensor([5, 2, 0])

    packed = pack_encoder_output(padded, lengths)

    assert packed.data.shape == (7, 2)
    assert packed.lengths.dtype == torch.int64
    assert packed.cu_seqlens.dtype == torch.int32
    assert packed.cu_seqlens.tolist() == [0, 5, 7, 7]
    assert packed.max_seqlen == 5
    assert packed_encoder_position_ids(packed).tolist() == [0, 1, 2, 3, 4, 0, 1]
    assert [part.shape[0] for part in split_encoder_output(packed)] == [5, 2, 0]
    torch.testing.assert_close(
        unpack_encoder_output(packed), padded * (torch.arange(5)[None, :, None] < lengths[:, None, None])
    )


def test_packed_encoder_output_all_empty_is_differentiable():
    padded = torch.randn(2, 0, 4, requires_grad=True)
    packed = pack_encoder_output(padded, torch.zeros(2, dtype=torch.long))

    assert packed.data.shape == (0, 4)
    assert packed.cu_seqlens.tolist() == [0, 0, 0]
    assert packed.max_seqlen == 0
    packed.data.sum().backward()
    assert padded.grad is not None


def test_split_packed_data_preserves_empty_rows_and_validates_integer_metadata():
    data = torch.arange(6)
    lengths = torch.tensor([2, 0, 4])
    rows = split_packed_data(data, lengths, torch.tensor([0, 2, 2, 6]))

    assert [row.tolist() for row in rows] == [[0, 1], [], [2, 3, 4, 5]]
    with pytest.raises(TypeError, match="integer dtype"):
        split_packed_data(data, lengths.float(), torch.tensor([0, 2, 2, 6]))
    with pytest.raises(TypeError, match="integer dtype"):
        split_packed_data(data, lengths, torch.tensor([0.0, 2.0, 2.0, 6.0]))


@pytest.mark.parametrize(
    ("lengths", "match"),
    [
        (torch.tensor([1]), "shape"),
        (torch.tensor([-1, 1]), "between"),
        (torch.tensor([3, 1]), "between"),
        (torch.tensor([1.0, 1.0]), "integer dtype"),
        (torch.tensor([True, False]), "integer dtype"),
    ],
)
def test_pack_encoder_output_rejects_invalid_lengths(lengths, match):
    padded = torch.randn(2, 2, 4)
    with pytest.raises((TypeError, ValueError), match=match):
        pack_encoder_output(padded, lengths)


def test_pack_encoder_output_uses_prevalidated_metadata_constructor(monkeypatch):
    from nemo.collections.asr.parts import packed_sequence as packed_sequence_module

    def fail_if_revalidated(*args, **kwargs):
        raise AssertionError("pack_encoder_output must not revalidate CUDA metadata")

    monkeypatch.setattr(packed_sequence_module, "_validate_packed_encoder_activations", fail_if_revalidated)
    packed = pack_encoder_output(torch.randn(2, 3, 4), torch.tensor([3, 1]))

    assert packed.cu_seqlens.tolist() == [0, 3, 4]
    assert packed.max_seqlen == 3


def test_packed_encoder_output_validates_manual_metadata():
    with pytest.raises(ValueError, match="differences equal to lengths"):
        PackedEncoderActivations(
            data=torch.randn(3, 4),
            lengths=torch.tensor([1, 2]),
            cu_seqlens=torch.tensor([0, 2, 3], dtype=torch.int32),
            max_seqlen=2,
        )


def test_packed_encoder_output_rejects_noncontiguous_cu_seqlens():
    cu_seqlens = torch.tensor([0, 99, 1, 99, 3, 99], dtype=torch.int32)[::2]
    assert not cu_seqlens.is_contiguous()
    with pytest.raises(ValueError, match="cu_seqlens must be contiguous"):
        PackedEncoderActivations(
            data=torch.randn(3, 4),
            lengths=torch.tensor([1, 2]),
            cu_seqlens=cu_seqlens,
            max_seqlen=2,
        )


def test_packed_encoder_output_pytree_transforms_do_not_revalidate_temporary_leaves():
    packed = pack_encoder_output(torch.randn(2, 3, 4), torch.tensor([3, 1]))

    leaves, spec = torch.utils._pytree.tree_flatten(packed)
    restored = torch.utils._pytree.tree_unflatten(leaves, spec)
    placeholder = torch.utils._pytree.tree_map(lambda _: None, packed)

    assert restored.data is packed.data
    assert restored.lengths is packed.lengths
    assert restored.cu_seqlens is packed.cu_seqlens
    assert restored.max_seqlen == packed.max_seqlen
    assert restored.padding_value == packed.padding_value
    assert restored.padded_length == packed.padded_length
    assert placeholder.data is None
    assert placeholder.lengths is None
    assert placeholder.cu_seqlens is None
    assert placeholder.max_seqlen is None
    assert placeholder.padding_value is None
    assert placeholder.padded_length is None


def _make_encoder(*, position: str, attention: str = "full", qk_norm: bool = False, rotary_fraction: float = 1.0):
    return TransformerEncoder(
        feat_in=8,
        d_model=32,
        n_heads=2,
        n_layers=2,
        subsampling_factor=2,
        drop_rate=0.0,
        dropout_pre_encoder=0.0,
        dropout_emb=0.0,
        self_attention_model=position,
        attn_mode=attention,
        qk_norm=qk_norm,
        qkv_bias=True,
        rotary_fraction=rotary_fraction,
        sync_max_audio_length=False,
    ).eval()


@pytest.mark.parametrize("position", ["rope", "abs_pos", "no_pos", "rel_pos"])
@pytest.mark.parametrize("attention", ["full", "causal"])
@pytest.mark.parametrize("qk_norm", [False, True])
def test_sequence_packed_matches_padded_valid_outputs_cpu(position, attention, qk_norm):
    kwargs = {"rotary_fraction": 0.5} if position == "rope" else {}
    torch.manual_seed(0)
    encoder = _make_encoder(position=position, attention=attention, qk_norm=qk_norm, **kwargs)
    audio = torch.randn(3, 8, 12)
    lengths = torch.tensor([12, 7, 4])

    with torch.no_grad():
        padded, output_lengths = encoder(audio, lengths)
        packed = encoder.forward_sequence_packed(audio, lengths)

    restored = unpack_encoder_output(packed, total_length=padded.shape[-1])
    valid = torch.arange(padded.shape[-1])[None, :] < output_lengths[:, None]
    torch.testing.assert_close(restored[valid], padded.transpose(1, 2)[valid], rtol=1e-5, atol=1e-6)
    assert packed.total_tokens == int(output_lengths.sum())
    assert encoder.layers[0].attn._last_sequence_packed_backend == "flex_attention_reference"
    assert encoder.layers[0].attn._last_sequence_packed_provider is None


@pytest.mark.parametrize("position", ["rope", "abs_pos", "no_pos", "rel_pos"])
def test_sequence_packed_accepts_token_flat_features_without_dense_pre_encode(position):
    torch.manual_seed(0)
    encoder = _make_encoder(position=position)
    audio = torch.randn(3, 8, 12)
    lengths = torch.tensor([12, 7, 4])
    audio.masked_fill_(torch.arange(audio.shape[-1])[None, None, :] >= lengths[:, None, None], 0.0)
    packed_features = pack_encoder_output(audio.transpose(1, 2), lengths)

    with torch.no_grad():
        from_dense = encoder.forward_sequence_packed(audio, lengths)
        from_packed = encoder.forward_sequence_packed(packed_features, lengths)

    assert from_packed.total_tokens == int(from_packed.lengths.sum())
    assert torch.equal(from_packed.lengths, from_dense.lengths)
    torch.testing.assert_close(from_packed.data, from_dense.data, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("use_moe", [False, True])
def test_sequence_packed_token_flat_feature_outputs_and_gradients_match_dense_input(use_moe):
    torch.manual_seed(0)
    if use_moe:
        dense_encoder = MoETransformerEncoder(
            feat_in=8,
            d_model=32,
            n_heads=2,
            n_layers=2,
            subsampling_factor=2,
            drop_rate=0.0,
            dropout_pre_encoder=0.0,
            dropout_emb=0.0,
            self_attention_model="rope",
            moe_num_experts=4,
            moe_top_k=2,
            sync_max_audio_length=False,
        ).train()
    else:
        dense_encoder = _make_encoder(position="rope").train()
    packed_encoder = copy.deepcopy(dense_encoder)
    lengths = torch.tensor([12, 7, 4])
    dense_features = torch.randn(3, 8, 12)
    dense_features.masked_fill_(torch.arange(12)[None, None, :] >= lengths[:, None, None], 0.0)
    dense_features.requires_grad_()
    packed_source = pack_encoder_output(dense_features.detach().transpose(1, 2), lengths)
    packed_features = PackedEncoderActivations(
        packed_source.data.clone().requires_grad_(),
        packed_source.lengths,
        packed_source.cu_seqlens,
        packed_source.max_seqlen,
    )

    dense_output = dense_encoder.forward_sequence_packed(dense_features, lengths)
    packed_output = packed_encoder.forward_sequence_packed(packed_features, lengths)
    dense_output.data.square().mean().backward()
    packed_output.data.square().mean().backward()

    torch.testing.assert_close(packed_output.data, dense_output.data, rtol=1e-5, atol=1e-6)
    dense_valid_grads = pack_encoder_output(dense_features.grad.transpose(1, 2), lengths).data
    torch.testing.assert_close(packed_features.data.grad, dense_valid_grads, rtol=1e-5, atol=1e-6)
    for (dense_name, dense_parameter), (packed_name, packed_parameter) in zip(
        dense_encoder.named_parameters(), packed_encoder.named_parameters()
    ):
        assert dense_name == packed_name
        if dense_parameter.grad is not None:
            torch.testing.assert_close(packed_parameter.grad, dense_parameter.grad, rtol=1e-5, atol=1e-6)


def test_sequence_packed_token_flat_feature_stacking_preserves_activation_checkpointing():
    encoder = _make_encoder(position="rope").train()
    encoder.pre_encode = checkpoint_wrapper(encoder.pre_encode)
    lengths = torch.tensor([12, 7])
    padded = torch.randn(2, 12, 8)
    padded.masked_fill_(torch.arange(12)[None, :, None] >= lengths[:, None, None], 0.0)
    packed = pack_encoder_output(padded, lengths)
    packed = PackedEncoderActivations(
        packed.data.detach().clone().requires_grad_(), packed.lengths, packed.cu_seqlens, packed.max_seqlen
    )

    output = encoder.forward_sequence_packed(packed, lengths)
    output.data.square().mean().backward()

    assert packed.data.grad is not None
    assert encoder.pre_encode._checkpoint_wrapped_module.proj.weight.grad is not None


@pytest.mark.parametrize(("position", "qk_norm"), [("rope", True), ("rel_pos", False)])
@pytest.mark.parametrize("fused_qkv", [False, True])
def test_sequence_packed_all_empty_keeps_attention_parameters_in_backward(position, qk_norm, fused_qkv):
    encoder = _make_encoder(position=position, qk_norm=qk_norm).train()
    inputs = torch.empty(2, 0, encoder.d_model, requires_grad=True)

    packed = encoder.forward_sequence_packed(
        inputs,
        torch.zeros(2, dtype=torch.long),
        bypass_pre_encode=True,
        fused_qkv=fused_qkv,
    )
    packed.data.sum().backward()

    assert inputs.grad is not None
    for layer in encoder.layers:
        for name, parameter in layer.attn.named_parameters():
            assert parameter.grad is not None, f"missing gradient for attention parameter {name}"
            assert torch.count_nonzero(parameter.grad) == 0


def test_sequence_packed_boundaries_isolate_other_utterances_and_causal_future():
    torch.manual_seed(0)
    encoder = _make_encoder(position="rope", attention="causal")
    encoded = torch.randn(2, 6, encoder.d_model)
    lengths = torch.tensor([6, 4])
    changed = encoded.clone()
    changed[0, 4:] = torch.randn_like(changed[0, 4:]) * 100
    changed[1] = torch.randn_like(changed[1]) * 100

    with torch.no_grad():
        original = encoder.forward_sequence_packed(encoded, lengths, bypass_pre_encode=True)
        mutated = encoder.forward_sequence_packed(changed, lengths, bypass_pre_encode=True)

    torch.testing.assert_close(original.data[:4], mutated.data[:4], rtol=1e-5, atol=1e-6)


def test_sequence_packed_fused_qkv_matches_default_and_preserves_checkpoint_keys():
    torch.manual_seed(0)
    encoder = _make_encoder(position="rope", qk_norm=True)
    inputs = torch.randn(2, 6, encoder.d_model)
    lengths = torch.tensor([6, 3])
    state_keys = set(encoder.state_dict())

    with torch.no_grad():
        independent = encoder.forward_sequence_packed(inputs, lengths, bypass_pre_encode=True)
        fused = encoder.forward_sequence_packed(inputs, lengths, bypass_pre_encode=True, fused_qkv=True)

    torch.testing.assert_close(fused.data, independent.data, rtol=1e-5, atol=1e-6)
    assert set(encoder.state_dict()) == state_keys


def test_sequence_packed_layers_receive_only_valid_tokens(monkeypatch):
    encoder = _make_encoder(position="rope")
    encoded = torch.randn(3, 7, encoder.d_model)
    lengths = torch.tensor([7, 3, 1])
    observed = []
    original = encoder.layers[0].ffn.forward

    def record(x):
        observed.append(tuple(x.shape))
        return original(x)

    monkeypatch.setattr(encoder.layers[0].ffn, "forward", record)
    with torch.no_grad():
        encoder.forward_sequence_packed(encoded, lengths, bypass_pre_encode=True)

    assert observed == [(11, encoder.d_model)]


def test_sequence_packed_varlen_dispatch_contract(monkeypatch):
    cfg = TransformerEncoderConfig(d_model=32, n_heads=2, self_attention_model="no_pos", qkv_bias=False)
    attention = MultiHeadAttention(cfg)
    recorded = {}

    def fake_flash(q, k, v, cu_q, cu_k, max_q, max_k, **kwargs):
        recorded.update(q=q, k=k, v=v, cu_q=cu_q, cu_k=cu_k, max_q=max_q, max_k=max_k, kwargs=kwargs)
        return v

    monkeypatch.setattr(transformer_encoder_module, "_can_use_flash_attention_varlen", lambda q: True)
    monkeypatch.setattr(transformer_encoder_module, "_get_flash_attention_varlen", lambda: fake_flash)
    lengths = torch.tensor([3, 2])
    cu_seqlens = torch.tensor([0, 3, 5], dtype=torch.int32)

    output = attention.forward_sequence_packed(
        torch.randn(5, 32),
        lengths=lengths,
        cu_seqlens=cu_seqlens,
        max_seqlen=3,
        causal=True,
        sequence_offsets=(0, 3, 5),
    )

    assert output.shape == (5, 32)
    for name in ("q", "k", "v"):
        assert recorded[name].shape == (5, 2, 16)
        assert recorded[name].is_contiguous()
    assert recorded["cu_q"] is recorded["cu_k"] is cu_seqlens
    assert recorded["cu_q"].dtype == torch.int32 and recorded["cu_q"].is_contiguous()
    assert recorded["max_q"] == recorded["max_k"] == 3
    assert recorded["kwargs"] == {"dropout_p": 0.0, "softmax_scale": None, "causal": True}
    assert attention._last_sequence_packed_provider == "external"


def test_flash_attention_varlen_aten_provider_is_reported(monkeypatch):
    if getattr(torch.ops.aten, "_flash_attention_forward", None) is None:
        pytest.skip("PyTorch build does not expose the ATen FlashAttention operator")
    original_import = builtins.__import__

    def import_without_external_flash(name, *args, **kwargs):
        if name == "flash_attn":
            raise ImportError("simulate flash-attn not installed")
        return original_import(name, *args, **kwargs)

    transformer_encoder_module._get_flash_attention_varlen.cache_clear()
    monkeypatch.setattr(builtins, "__import__", import_without_external_flash)
    try:
        provider = transformer_encoder_module._get_flash_attention_varlen()
        assert provider is not None
        assert provider._sequence_packed_provider == "aten"
    finally:
        transformer_encoder_module._get_flash_attention_varlen.cache_clear()


def test_sequence_packed_adds_no_state_dict_keys_and_loads_strictly():
    encoder = _make_encoder(position="rope")
    before = set(encoder.state_dict())
    with torch.no_grad():
        encoder.forward_sequence_packed(torch.randn(2, 5, encoder.d_model), torch.tensor([5, 2]), True)
    after = set(encoder.state_dict())
    clone = _make_encoder(position="rope")

    result = clone.load_state_dict(encoder.state_dict(), strict=True)

    assert before == after
    assert result.missing_keys == []
    assert result.unexpected_keys == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlexAttention backward requires CUDA")
def test_sequence_packed_supports_activation_checkpoint_wrapped_layers():
    encoder = _make_encoder(position="rope").to(device="cuda", dtype=torch.bfloat16).train()
    state_keys = set(encoder.state_dict())
    for idx, layer in enumerate(encoder.layers):
        encoder.layers[idx] = checkpoint_wrapper(layer)
    inputs = torch.randn(2, 8, 10, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    packed = encoder.forward_sequence_packed(inputs, torch.tensor([10, 4], device="cuda"))
    packed.data.square().mean().backward()

    assert inputs.grad is not None
    assert encoder.layers[0]._checkpoint_wrapped_module.attn.w_qkv.weight.grad is not None
    assert set(encoder.state_dict()) == state_keys


def _make_moe_encoder():
    return MoETransformerEncoder(
        feat_in=8,
        d_model=32,
        n_heads=2,
        n_layers=2,
        subsampling_factor=2,
        drop_rate=0.0,
        dropout_pre_encoder=0.0,
        self_attention_model="rope",
        moe_num_experts=4,
        moe_top_k=2,
        sync_max_audio_length=False,
    )


def test_moe_sequence_packed_matches_padded_and_excludes_padding_from_routing():
    torch.manual_seed(0)
    encoder = _make_moe_encoder().eval()
    encoded = torch.randn(3, 7, encoder.d_model)
    lengths = torch.tensor([7, 3, 1])

    with torch.no_grad():
        padded, _ = encoder(encoded, lengths, bypass_pre_encode=True)
        packed = encoder.forward_sequence_packed(encoded, lengths, bypass_pre_encode=True)

    restored = unpack_encoder_output(packed, total_length=7)
    valid = torch.arange(7)[None, :] < lengths[:, None]
    torch.testing.assert_close(restored[valid], padded.transpose(1, 2)[valid], rtol=1e-5, atol=1e-6)
    for layer_idx in encoder.moe_layer_indices:
        assert encoder.layers[layer_idx].ffn._num_tokens == int(lengths.sum())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlexAttention backward requires CUDA")
@pytest.mark.parametrize(("top_k", "router_type"), [(1, "switch"), (2, "omni")])
def test_moe_sequence_packed_matches_valid_token_gradients(top_k, router_type):
    torch.manual_seed(0)
    padded_encoder = (
        MoETransformerEncoder(
            feat_in=8,
            d_model=32,
            n_heads=2,
            n_layers=1,
            subsampling_factor=2,
            drop_rate=0.0,
            dropout_pre_encoder=0.0,
            self_attention_model="rope",
            moe_num_experts=4,
            moe_top_k=top_k,
            moe_router_type=router_type,
            sync_max_audio_length=False,
        )
        .to(device="cuda", dtype=torch.bfloat16)
        .eval()
    )
    packed_encoder = copy.deepcopy(padded_encoder)
    padded_input = torch.randn(3, 7, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    packed_input = padded_input.detach().clone().requires_grad_()
    lengths = torch.tensor([7, 3, 1], device="cuda")

    padded, _ = padded_encoder(padded_input, lengths, bypass_pre_encode=True)
    packed = packed_encoder.forward_sequence_packed(packed_input, lengths, bypass_pre_encode=True)
    valid = torch.arange(7, device="cuda")[None, :] < lengths[:, None]
    padded.transpose(1, 2)[valid].square().mean().backward()
    packed.data.square().mean().backward()

    torch.testing.assert_close(packed_input.grad[valid], padded_input.grad[valid], rtol=3e-2, atol=3e-2)
    for suffix in ("router.w_gate.weight", "experts.0.net.0.weight"):
        packed_grad = dict(packed_encoder.named_parameters())[f"layers.0.ffn.{suffix}"].grad
        padded_grad = dict(padded_encoder.named_parameters())[f"layers.0.ffn.{suffix}"].grad
        torch.testing.assert_close(packed_grad, padded_grad, rtol=3e-2, atol=3e-2)


def test_moe_feed_forward_all_empty_records_zero_stats_and_keeps_parameters_in_graph():
    cfg = TransformerEncoderConfig(d_model=32, n_heads=2, ff_expansion=1.0, drop_rate=0.0)
    moe = MoEFeedForward(cfg, num_experts=3, top_k=2)
    x = torch.empty(0, 32, requires_grad=True)

    output = moe(x)
    (output.sum() + moe._aux_loss).backward()

    assert output.shape == x.shape
    assert moe._num_tokens == 0
    assert moe._expert_counts.tolist() == [0, 0, 0]
    assert moe._gate_prob_sum.tolist() == [0.0, 0.0, 0.0]
    assert moe.router.w_gate.weight.grad is not None
    assert all(parameter.grad is not None for expert in moe.experts for parameter in expert.parameters())


@requires_cuda_varlen_flash_attention
def test_sequence_packed_thd_cuda_matches_padded_outputs_and_gradients():
    torch.manual_seed(0)
    padded_encoder = _make_encoder(position="rope", qk_norm=True).to(device="cuda", dtype=torch.bfloat16).train()
    packed_encoder = copy.deepcopy(padded_encoder)
    padded_input = torch.randn(3, 12, padded_encoder.d_model, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    packed_input = padded_input.detach().clone().requires_grad_()
    lengths = torch.tensor([12, 7, 3], device="cuda")

    padded, output_lengths = padded_encoder(padded_input, lengths, bypass_pre_encode=True)
    packed = packed_encoder.forward_sequence_packed(packed_input, lengths, bypass_pre_encode=True)
    restored = unpack_encoder_output(packed, total_length=padded.shape[-1])
    valid = torch.arange(padded.shape[-1], device="cuda")[None, :] < output_lengths[:, None]
    padded_valid = padded.transpose(1, 2)[valid]
    packed_valid = restored[valid]
    padded_valid.float().square().mean().backward()
    packed_valid.float().square().mean().backward()

    torch.testing.assert_close(packed_valid, padded_valid, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(packed_input.grad, padded_input.grad, rtol=3e-2, atol=3e-2)
    for name in (
        "layers.0.attn.w_qkv.weight",
        "layers.0.attn.out_proj.weight",
        "layers.0.attn.q_norm.weight",
        "layers.0.ffn.net.0.weight",
    ):
        torch.testing.assert_close(
            dict(packed_encoder.named_parameters())[name].grad,
            dict(padded_encoder.named_parameters())[name].grad,
            rtol=3e-2,
            atol=3e-2,
        )
    assert packed_encoder.layers[0].attn._last_sequence_packed_backend == "flash_attention_varlen"
    assert packed_encoder.layers[0].attn._last_sequence_packed_provider in {"aten", "external"}
    assert packed.data.shape == (int(lengths.sum()), packed_encoder.d_model)


@requires_cuda_varlen_flash_attention
@pytest.mark.parametrize(
    ("dtype", "position", "attention", "qk_norm", "rotary_fraction"),
    [
        (torch.float16, "rope", "full", False, 0.5),
        (torch.bfloat16, "rope", "causal", False, 1.0),
        (torch.bfloat16, "abs_pos", "causal", True, 1.0),
        (torch.float16, "no_pos", "full", True, 1.0),
    ],
)
def test_sequence_packed_cuda_fast_path_matrix(dtype, position, attention, qk_norm, rotary_fraction):
    torch.manual_seed(0)
    encoder = _make_encoder(
        position=position,
        attention=attention,
        qk_norm=qk_norm,
        rotary_fraction=rotary_fraction,
    ).to(device="cuda", dtype=dtype)
    inputs = torch.randn(3, 12, 32, device="cuda", dtype=dtype)
    lengths = torch.tensor([12, 0, 5], device="cuda")

    with torch.no_grad():
        padded, output_lengths = encoder(inputs, lengths, bypass_pre_encode=True)
        packed = encoder.forward_sequence_packed(inputs, lengths, bypass_pre_encode=True)

    restored = unpack_encoder_output(packed, total_length=padded.shape[-1])
    valid = torch.arange(padded.shape[-1], device="cuda")[None, :] < output_lengths[:, None]
    torch.testing.assert_close(restored[valid], padded.transpose(1, 2)[valid], rtol=3e-2, atol=3e-2)
    assert encoder.layers[0].attn._last_sequence_packed_backend == "flash_attention_varlen"
    assert encoder.layers[0].attn._last_sequence_packed_provider in {"aten", "external"}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA FlexAttention")
def test_sequence_packed_cuda_fp32_relative_position_fallback_gradients():
    torch.manual_seed(0)
    padded_encoder = _make_encoder(position="rel_pos").cuda().train()
    packed_encoder = copy.deepcopy(padded_encoder)
    padded_input = torch.randn(2, 8, 32, device="cuda", requires_grad=True)
    packed_input = padded_input.detach().clone().requires_grad_()
    lengths = torch.tensor([8, 3], device="cuda")

    padded, _ = padded_encoder(padded_input, lengths, bypass_pre_encode=True)
    packed = packed_encoder.forward_sequence_packed(packed_input, lengths, bypass_pre_encode=True)
    valid = torch.arange(8, device="cuda")[None, :] < lengths[:, None]
    padded.transpose(1, 2)[valid].square().mean().backward()
    packed.data.square().mean().backward()

    torch.testing.assert_close(packed_input.grad[valid], padded_input.grad[valid], rtol=2e-4, atol=2e-5)
    for suffix in ("linear_pos.weight", "pos_bias_u", "pos_bias_v"):
        packed_grad = dict(packed_encoder.named_parameters())[f"layers.0.attn.{suffix}"].grad
        padded_grad = dict(padded_encoder.named_parameters())[f"layers.0.attn.{suffix}"].grad
        torch.testing.assert_close(packed_grad, padded_grad, rtol=2e-4, atol=2e-5)
    assert packed_encoder.layers[0].attn._last_sequence_packed_backend == "flex_attention_reference"


@requires_cuda_varlen_flash_attention
def test_sequence_packed_reduces_forward_backward_peak_memory_for_uneven_batch():
    torch.manual_seed(0)
    encoder = TransformerEncoder(
        feat_in=64,
        d_model=128,
        n_heads=4,
        n_layers=2,
        subsampling_factor=1,
        drop_rate=0.0,
        dropout_pre_encoder=0.0,
        dropout_emb=0.0,
        self_attention_model="rope",
        sync_max_audio_length=False,
    ).to(device="cuda", dtype=torch.bfloat16)
    encoder.train()
    source = torch.randn(4, 64, 256, device="cuda", dtype=torch.bfloat16)
    lengths = torch.tensor([256, 64, 32, 16], device="cuda")

    def run(sequence_packed: bool, measure: bool = False) -> int:
        gc.collect()
        torch.cuda.empty_cache()
        encoder.zero_grad(set_to_none=True)
        inputs = source.detach().clone().requires_grad_()
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        if measure:
            torch.cuda.reset_peak_memory_stats()
        if sequence_packed:
            output = encoder.forward_sequence_packed(inputs, lengths).data
        else:
            output = encoder(inputs, lengths)[0]
        output.float().sum().backward()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() - baseline if measure else 0
        del output, inputs
        return peak

    # Warm both kernel families so compilation and allocator setup are excluded.
    run(sequence_packed=False)
    run(sequence_packed=True)
    padded_peak = run(sequence_packed=False, measure=True)
    packed_peak = run(sequence_packed=True, measure=True)

    assert encoder.layers[0].attn._last_sequence_packed_backend == "flash_attention_varlen"
    assert encoder.layers[0].attn._last_sequence_packed_provider in {"aten", "external"}
    assert packed_peak < padded_peak * 0.7, (
        f"Expected native THD to materially reduce peak activation memory; "
        f"padded={padded_peak:,} bytes, packed={packed_peak:,} bytes."
    )
