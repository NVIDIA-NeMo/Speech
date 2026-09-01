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

import copy

import pytest
import torch

import nemo.collections.asr.modules.ggemm_transformer_encoder as ggemm_module
import nemo.collections.asr.modules.transformer_encoder as transformer_module
from nemo.collections.asr.parts.packed_sequence import PackedEncoderActivations, pack_encoder_output
from tests.collections.asr.test_parallel_expert_encoder_ggemm import (
    _MEL_FEATURES,
    _N_SPK,
    build_toy_pe_encoder,
    toy_sound_expert_cfg,
    toy_speaker_expert_cfg,
    toy_speech_expert_cfg,
)


@pytest.mark.unit
@pytest.mark.parametrize('fused_qkv', [False, True])
def test_grouped_sequence_packed_matches_serial_reference_and_records_groups(fused_qkv):
    torch.manual_seed(0)
    encoder = build_toy_pe_encoder().eval()
    signal, signal_lengths = encoder._prepare_input(
        torch.randn(3, _MEL_FEATURES, 40),
        torch.tensor([40, 23, 9]),
    )

    with torch.no_grad():
        serial = encoder.pee.forward_all_sequence_packed(signal, signal_lengths, fused_qkv=fused_qkv)
        grouped = encoder.pee.forward_grouped_sequence_packed(signal, signal_lengths, fused_qkv=fused_qkv)

    for name in encoder.pee.expert_names:
        torch.testing.assert_close(grouped[name].data, serial[name].data, rtol=1e-5, atol=1e-6)
        assert torch.equal(grouped[name].lengths, serial[name].lengths)
        assert torch.equal(grouped[name].cu_seqlens, serial[name].cu_seqlens)
        assert grouped[name].max_seqlen == serial[name].max_seqlen

    trace = encoder.pee._last_sequence_packed_execution
    assert trace['mode'] == 'grouped_thd'
    assert trace['attention_groups'] == len(encoder.pee.experts['speech'].layers)
    assert trace['attention_grouped_experts'] == len(encoder.pee.expert_names) * trace['layers']
    assert trace['qkv_grouped_experts'] == 2 * trace['layers']
    assert trace['qkv_grouped_projection_calls'] == (1 if fused_qkv else 3) * trace['layers']
    assert trace['out_grouped_experts'] == 2 * trace['layers']


@pytest.mark.unit
def test_pee_grouped_and_serial_thd_fusion_match():
    encoder = build_toy_pe_encoder().eval()
    mels = torch.randn(2, _MEL_FEATURES, 40)
    lengths = torch.tensor([40, 17])
    targets = torch.zeros(2, 5, _N_SPK)
    with torch.no_grad():
        serial = encoder._forward_sequence_packed(mels, lengths, spk_targets=targets, grouped=False)
        grouped = encoder.forward_sequence_packed(mels, lengths, spk_targets=targets)
    assert torch.equal(grouped.lengths, serial.lengths)
    assert torch.equal(grouped.cu_seqlens, serial.cu_seqlens)
    torch.testing.assert_close(grouped.data, serial.data, rtol=1e-5, atol=1e-6)


@pytest.mark.unit
def test_pee_grouped_accepts_token_flat_mels_and_matches_dense_mel_input():
    torch.manual_seed(0)
    encoder = build_toy_pe_encoder().eval()
    mels = torch.randn(2, _MEL_FEATURES, 40)
    lengths = torch.tensor([40, 17])
    mels.masked_fill_(torch.arange(mels.shape[-1])[None, None, :] >= lengths[:, None, None], 0.0)
    packed_mels = pack_encoder_output(mels.transpose(1, 2), lengths)
    targets = torch.zeros(2, 5, _N_SPK)

    with torch.no_grad():
        dense_input = encoder.forward_sequence_packed(mels, lengths, spk_targets=targets)
        packed_input = encoder.forward_sequence_packed(packed_mels, lengths, spk_targets=targets)

    assert torch.equal(packed_input.lengths, dense_input.lengths)
    assert packed_input.total_tokens == int(packed_input.lengths.sum())
    torch.testing.assert_close(packed_input.data, dense_input.data, rtol=1e-5, atol=1e-6)


@pytest.mark.unit
def test_pee_grouped_preserves_nonzero_partial_stack_padding():
    torch.manual_seed(0)
    encoder = build_toy_pe_encoder().eval()
    encoder.asr_normalize_type = None
    lengths = torch.tensor([39, 17])
    mels = torch.randn(2, _MEL_FEATURES, 40)
    mels.masked_fill_(torch.arange(40)[None, None, :] >= lengths[:, None, None], -3.0)
    source = pack_encoder_output(mels.transpose(1, 2), lengths)
    packed_mels = PackedEncoderActivations(
        source.data,
        source.lengths,
        source.cu_seqlens,
        source.max_seqlen,
        padding_value=-3.0,
        padded_length=40,
    )
    targets = torch.zeros(2, 5, _N_SPK)

    with torch.no_grad():
        expected = encoder.forward_sequence_packed(mels, lengths, spk_targets=targets)
        actual = encoder.forward_sequence_packed(packed_mels, lengths, spk_targets=targets)

    torch.testing.assert_close(actual.data, expected.data, rtol=1e-5, atol=1e-6)


@pytest.mark.unit
def test_pee_grouped_token_flat_mel_gradients_match_dense_input_with_checkpointing():
    torch.manual_seed(0)
    dense_encoder = build_toy_pe_encoder().train()
    dense_encoder.set_activation_checkpointing(True)
    packed_encoder = copy.deepcopy(dense_encoder)
    lengths = torch.tensor([40, 17])
    dense_mels = torch.randn(2, _MEL_FEATURES, 40)
    dense_mels.masked_fill_(torch.arange(40)[None, None, :] >= lengths[:, None, None], 0.0)
    dense_mels.requires_grad_()
    packed_source = pack_encoder_output(dense_mels.detach().transpose(1, 2), lengths)
    packed_mels = PackedEncoderActivations(
        packed_source.data.clone().requires_grad_(),
        packed_source.lengths,
        packed_source.cu_seqlens,
        packed_source.max_seqlen,
    )
    targets = torch.zeros(2, 5, _N_SPK)

    dense_output = dense_encoder.forward_sequence_packed(dense_mels, lengths, spk_targets=targets)
    packed_output = packed_encoder.forward_sequence_packed(packed_mels, lengths, spk_targets=targets)
    dense_output.data.square().mean().backward()
    packed_output.data.square().mean().backward()

    torch.testing.assert_close(packed_output.data, dense_output.data, rtol=1e-5, atol=1e-6)
    dense_valid_grads = pack_encoder_output(dense_mels.grad.transpose(1, 2), lengths).data
    torch.testing.assert_close(packed_mels.data.grad, dense_valid_grads, rtol=1e-5, atol=1e-6)
    for name in dense_encoder.pee.expert_names:
        dense_grad = dense_encoder.pee.experts[name].pre_encode.proj.weight.grad
        packed_grad = packed_encoder.pee.experts[name].pre_encode.proj.weight.grad
        torch.testing.assert_close(packed_grad, dense_grad, rtol=1e-5, atol=1e-6)


@pytest.mark.unit
def test_pee_production_packed_path_never_calls_serial_expert_forwards(monkeypatch):
    encoder = build_toy_pe_encoder().eval()

    def reject_serial(*args, **kwargs):
        raise AssertionError("production PEE packed execution must stay layer-grouped")

    monkeypatch.setattr(encoder.pee, 'forward_all_sequence_packed', reject_serial)
    for expert in encoder.pee.experts.values():
        monkeypatch.setattr(expert, 'forward_sequence_packed', reject_serial)

    packed = encoder.forward_sequence_packed(
        torch.randn(2, _MEL_FEATURES, 40),
        torch.tensor([40, 17]),
        spk_targets=torch.zeros(2, 5, _N_SPK),
    )
    assert packed.total_tokens == int(packed.lengths.sum())
    assert encoder.pee._last_sequence_packed_execution['mode'] == 'grouped_thd'


@pytest.mark.unit
def test_grouped_sequence_packed_uses_one_attention_group_and_grouped_ffns(monkeypatch):
    encoder = build_toy_pe_encoder().eval()
    signal, signal_lengths = encoder._prepare_input(
        torch.randn(2, _MEL_FEATURES, 40),
        torch.tensor([40, 23]),
    )
    attention_calls = 0
    grouped_linear_calls = 0
    original_attention = transformer_module.flex_attention
    original_grouped_linear = ggemm_module._grouped_linear

    def record_attention(*args, **kwargs):
        nonlocal attention_calls
        attention_calls += 1
        return original_attention(*args, **kwargs)

    def record_grouped_linear(*args, **kwargs):
        nonlocal grouped_linear_calls
        grouped_linear_calls += 1
        return original_grouped_linear(*args, **kwargs)

    monkeypatch.setattr(transformer_module, 'flex_attention', record_attention)
    monkeypatch.setattr(ggemm_module, '_grouped_linear', record_grouped_linear)
    with torch.no_grad():
        encoder.pee.forward_grouped_sequence_packed(signal, signal_lengths, moe_mode='dense', fused_qkv=True)

    # The compact CPU reference invokes FlexAttention once per non-empty utterance,
    # not once per utterance and expert. Four grouped linear launches cover
    # wide QKV, output projection, and the two wide speech-MoE/sound FFN GEMMs;
    # the narrow singleton stays native.
    assert attention_calls == 2
    assert grouped_linear_calls == 4


@pytest.mark.unit
def test_grouped_sequence_packed_matches_serial_gradients_and_moe_routing():
    torch.manual_seed(0)
    serial_encoder = build_toy_pe_encoder().train()
    grouped_encoder = build_toy_pe_encoder().train()
    grouped_encoder.load_state_dict(serial_encoder.state_dict())
    signal, signal_lengths = serial_encoder._prepare_input(
        torch.randn(2, _MEL_FEATURES, 32),
        torch.tensor([32, 17]),
    )
    serial_input = signal.detach().requires_grad_(True)
    grouped_input = signal.detach().clone().requires_grad_(True)

    serial = serial_encoder.pee.forward_all_sequence_packed(serial_input, signal_lengths)
    serial_loss = _expert_loss(serial, serial_encoder.pee.experts['speech'])
    serial_loss.backward()
    grouped = grouped_encoder.pee.forward_grouped_sequence_packed(
        grouped_input,
        signal_lengths,
        moe_mode='dense',
        fused_qkv=True,
    )
    grouped_loss = _expert_loss(grouped, grouped_encoder.pee.experts['speech'])
    grouped_loss.backward()

    for name in serial:
        torch.testing.assert_close(grouped[name].data, serial[name].data, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(grouped_input.grad, serial_input.grad, rtol=2e-4, atol=2e-5)
    for (name, serial_parameter), (_, grouped_parameter) in zip(
        serial_encoder.pee.named_parameters(),
        grouped_encoder.pee.named_parameters(),
    ):
        if not serial_parameter.requires_grad:
            assert serial_parameter.grad is None and grouped_parameter.grad is None
            continue
        assert serial_parameter.grad is not None, name
        assert grouped_parameter.grad is not None, name
        torch.testing.assert_close(grouped_parameter.grad, serial_parameter.grad, rtol=3e-4, atol=3e-5)

    serial_moe = serial_encoder.pee.experts['speech'].layers[0].ffn
    grouped_moe = grouped_encoder.pee.experts['speech'].layers[0].ffn
    assert torch.equal(grouped_moe._expert_counts, serial_moe._expert_counts)
    torch.testing.assert_close(grouped_moe._gate_prob_sum, serial_moe._gate_prob_sum)
    assert grouped_moe._num_tokens == serial_moe._num_tokens == sum(grouped['speech'].lengths.tolist())
    torch.testing.assert_close(grouped_moe._aux_loss, serial_moe._aux_loss)


@pytest.mark.unit
@pytest.mark.parametrize('group_speech_moe', [False, True])
def test_serial_checkpointed_training_matches_serial_reference_gradients(group_speech_moe):
    torch.manual_seed(0)
    reference_encoder = build_toy_pe_encoder(freeze_sound=True).train()
    checkpointed_encoder = build_toy_pe_encoder(freeze_sound=True).train()
    checkpointed_encoder.load_state_dict(reference_encoder.state_dict())
    checkpointed_encoder.set_activation_checkpointing(True)
    checkpointed_encoder.sequence_packed_execution_mode = 'serial_checkpointed'
    checkpointed_encoder.sequence_packed_serial_speech_grouped_moe = group_speech_moe
    mels = torch.randn(2, _MEL_FEATURES, 32)
    signal_lengths = torch.tensor([32, 17])
    packed_mels = pack_encoder_output(mels.transpose(1, 2), signal_lengths)
    signal, signal_lengths = reference_encoder._prepare_input(packed_mels, signal_lengths)
    checkpointed_signal, _ = checkpointed_encoder._prepare_input(packed_mels, signal_lengths)

    reference = reference_encoder.pee.forward_all_sequence_packed(signal, signal_lengths, fused_qkv=True)
    reference_loss = _expert_loss(reference, reference_encoder.pee.experts['speech'])
    reference_loss.backward()
    checkpointed = checkpointed_encoder._forward_all_sequence_packed_serial_training(
        checkpointed_signal, signal_lengths
    )
    checkpointed_loss = _expert_loss(checkpointed, checkpointed_encoder.pee.experts['speech'])
    checkpointed_loss.backward()
    if group_speech_moe:
        backend = checkpointed_encoder.pee.experts['speech'].layers[0].ffn._last_grouped_backend
        assert backend in ('grouped_mm', 'capacity_baddbmm')

    for name in reference:
        torch.testing.assert_close(checkpointed[name].data, reference[name].data, rtol=1e-5, atol=1e-6)
        assert torch.equal(checkpointed[name].lengths, reference[name].lengths)
        assert torch.equal(checkpointed[name].cu_seqlens, reference[name].cu_seqlens)
    for (name, reference_parameter), (_, checkpointed_parameter) in zip(
        reference_encoder.pee.named_parameters(),
        checkpointed_encoder.pee.named_parameters(),
    ):
        if not reference_parameter.requires_grad:
            assert reference_parameter.grad is None and checkpointed_parameter.grad is None
            continue
        assert reference_parameter.grad is not None, name
        assert checkpointed_parameter.grad is not None, name
        torch.testing.assert_close(checkpointed_parameter.grad, reference_parameter.grad, rtol=3e-4, atol=3e-5)


@pytest.mark.unit
def test_grouped_sequence_packed_respects_frozen_expert_dropout_mode():
    def with_dropout(factory):
        config = factory()
        config.drop_rate = 0.5
        return config

    serial_encoder = build_toy_pe_encoder(
        speech_expert_cfg=with_dropout(toy_speech_expert_cfg),
        speaker_expert_cfg=with_dropout(toy_speaker_expert_cfg),
        sound_expert_cfg=with_dropout(toy_sound_expert_cfg),
    ).train()
    grouped_encoder = build_toy_pe_encoder(
        speech_expert_cfg=with_dropout(toy_speech_expert_cfg),
        speaker_expert_cfg=with_dropout(toy_speaker_expert_cfg),
        sound_expert_cfg=with_dropout(toy_sound_expert_cfg),
    ).train()
    grouped_encoder.load_state_dict(serial_encoder.state_dict())
    serial_encoder.pee.experts['sound'].eval()
    grouped_encoder.pee.experts['sound'].eval()
    signal, signal_lengths = serial_encoder._prepare_input(
        torch.randn(2, _MEL_FEATURES, 32),
        torch.tensor([32, 17]),
    )

    torch.manual_seed(0)
    serial = serial_encoder.pee.forward_all_sequence_packed(signal, signal_lengths, fused_qkv=True)
    torch.manual_seed(0)
    grouped = grouped_encoder.pee.forward_grouped_sequence_packed(
        signal,
        signal_lengths,
        moe_mode='dense',
        fused_qkv=True,
    )

    assert grouped_encoder.pee.training
    assert not grouped_encoder.pee.experts['sound'].training
    torch.testing.assert_close(grouped['sound'].data, serial['sound'].data, rtol=1e-5, atol=1e-6)


@pytest.mark.unit
def test_grouped_sequence_packed_preserves_independent_ffn_dropout_sites():
    serial_encoder = build_toy_pe_encoder(freeze_speaker=False).train()
    grouped_encoder = build_toy_pe_encoder(freeze_speaker=False).train()
    grouped_encoder.load_state_dict(serial_encoder.state_dict())
    serial_encoder.pee.experts['sound'].layers[0].ffn.net[4].p = 0.5
    grouped_encoder.pee.experts['sound'].layers[0].ffn.net[4].p = 0.5
    signal, signal_lengths = serial_encoder._prepare_input(
        torch.randn(2, _MEL_FEATURES, 32),
        torch.tensor([32, 17]),
    )

    torch.manual_seed(0)
    serial = serial_encoder.pee.forward_all_sequence_packed(signal, signal_lengths, fused_qkv=True)
    torch.manual_seed(0)
    grouped = grouped_encoder.pee.forward_grouped_sequence_packed(
        signal,
        signal_lengths,
        moe_mode='dense',
        fused_qkv=True,
    )

    torch.testing.assert_close(grouped['sound'].data, serial['sound'].data, rtol=1e-5, atol=1e-6)


@pytest.mark.unit
def test_grouped_sequence_packed_buckets_mixed_attention_features():
    speech_config = toy_speech_expert_cfg()
    sound_config = toy_sound_expert_cfg()
    speaker_config = toy_speaker_expert_cfg()
    for config in (speech_config, sound_config, speaker_config):
        config.qkv_bias = True
        config.qk_norm = True
    speaker_config.self_attention_model = 'rel_pos'
    speaker_config.attn_mode = 'causal'
    encoder = build_toy_pe_encoder(
        speech_expert_cfg=speech_config,
        speaker_expert_cfg=speaker_config,
        sound_expert_cfg=sound_config,
    ).eval()
    signal, signal_lengths = encoder._prepare_input(
        torch.randn(2, _MEL_FEATURES, 32),
        torch.tensor([32, 17]),
    )

    with torch.no_grad():
        serial = encoder.pee.forward_all_sequence_packed(signal, signal_lengths, fused_qkv=True)
        grouped = encoder.pee.forward_grouped_sequence_packed(
            signal,
            signal_lengths,
            fused_qkv=True,
        )

    for name in encoder.pee.expert_names:
        torch.testing.assert_close(grouped[name].data, serial[name].data, rtol=2e-5, atol=2e-6)
    trace = encoder.pee._last_sequence_packed_execution
    assert trace['attention_groups'] == 2 * trace['layers']
    assert trace['attention_grouped_experts'] == 3 * trace['layers']


@pytest.mark.unit
@pytest.mark.parametrize('training', [False, True])
def test_grouped_sequence_packed_validates_shared_metadata_once(monkeypatch, training):
    encoder = build_toy_pe_encoder().train(training)
    signal, signal_lengths = encoder._prepare_input(
        torch.randn(2, _MEL_FEATURES, 32),
        torch.tensor([32, 17]),
    )
    calls = 0
    original_pack = ggemm_module.pack_encoder_output

    def record_pack(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_pack(*args, **kwargs)

    monkeypatch.setattr(ggemm_module, 'pack_encoder_output', record_pack)
    with torch.no_grad():
        outputs = encoder.pee.forward_grouped_sequence_packed(signal, signal_lengths)

    assert calls == 1
    lengths = [output.lengths for output in outputs.values()]
    cu_seqlens = [output.cu_seqlens for output in outputs.values()]
    assert all(item is lengths[0] for item in lengths[1:])
    assert all(item is cu_seqlens[0] for item in cu_seqlens[1:])


@pytest.mark.unit
def test_loop_backend_is_an_actual_dense_ffn_reference(monkeypatch):
    encoder = build_toy_pe_encoder().eval()
    signal, signal_lengths = encoder._prepare_input(
        torch.randn(2, _MEL_FEATURES, 32),
        torch.tensor([32, 17]),
    )
    grouped_linear_calls = 0
    original_grouped_linear = ggemm_module._grouped_linear

    def record_grouped_linear(*args, **kwargs):
        nonlocal grouped_linear_calls
        grouped_linear_calls += 1
        return original_grouped_linear(*args, **kwargs)

    monkeypatch.setattr(ggemm_module, '_grouped_linear', record_grouped_linear)
    with torch.no_grad():
        output = encoder.pee.forward_grouped_sequence_packed(
            signal,
            signal_lengths,
            backend='loop',
            moe_mode='dense',
            fused_qkv=True,
        )

    assert all(item.total_tokens for item in output.values())
    # Only grouped QKV and output projections remain; both FFN projections honor loop.
    assert grouped_linear_calls == 2
    trace = encoder.pee._last_sequence_packed_execution
    assert trace['dense_ffn_backend'] == 'loop'
    assert trace['moe_grouped_backends'] == ['dense_loop']


@pytest.mark.unit
def test_grouped_sequence_packed_trace_resets_after_all_empty_input():
    encoder = build_toy_pe_encoder().eval()
    signal, signal_lengths = encoder._prepare_input(
        torch.randn(2, _MEL_FEATURES, 32),
        torch.tensor([32, 17]),
    )
    with torch.no_grad():
        encoder.pee.forward_grouped_sequence_packed(signal, signal_lengths, backend='loop', moe_mode='topk')
        encoder.pee.forward_grouped_sequence_packed(
            signal,
            torch.zeros_like(signal_lengths),
            backend='grouped_mm',
            moe_mode='topk',
        )

    trace = encoder.pee._last_sequence_packed_execution
    assert trace['moe_grouped_backends'] == ['native_empty']


@pytest.mark.unit
def test_grouped_biasless_linear_uses_bmm_without_zero_bias_allocation(monkeypatch):
    linears = torch.nn.ModuleList([torch.nn.Linear(4, 6, bias=False) for _ in range(3)])
    inputs = torch.randn(3, 5, 4, requires_grad=True)
    expected = torch.stack([linear(inputs[index]) for index, linear in enumerate(linears)])

    def reject_baddbmm(*args, **kwargs):
        raise AssertionError('an all-biasless group must use bmm')

    monkeypatch.setattr(torch, 'baddbmm', reject_baddbmm)
    actual = ggemm_module._grouped_linear(inputs, linears)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()

    assert inputs.grad is not None
    assert all(linear.weight.grad is not None for linear in linears)


@pytest.mark.unit
def test_sequence_packed_serial_capability_keeps_original_signature():
    class PackedOnly(torch.nn.Module):
        supports_sequence_packed_output = True

        def forward_sequence_packed(self, audio_signal, length, bypass_pre_encode=False):
            cu_seqlens = torch.tensor([0, audio_signal.shape[0]], dtype=torch.int32)
            return PackedEncoderActivations(audio_signal, length, cu_seqlens, int(length.max()))

    expert = PackedOnly()
    container = ggemm_module.GGEMMTransformerEncoder({'custom': expert})
    data = torch.randn(3, 4)
    output = container.forward_all_sequence_packed(data, torch.tensor([3]))
    assert output['custom'].data is data
    with pytest.raises(TypeError, match='does not advertise fused'):
        container.forward_all_sequence_packed(data, torch.tensor([3]), fused_qkv=True)


@pytest.mark.unit
def test_pee_defaults_to_sparse_ragged_grouped_packed_execution():
    encoder = build_toy_pe_encoder()
    assert encoder.sequence_packed_moe_mode == 'auto'
    assert encoder.sequence_packed_ggemm_backend == 'grouped_mm'
    assert encoder.moe_mode == 'dense'
    assert encoder.ggemm_backend == 'baddbmm'
    encoder.eval()
    assert encoder._sequence_packed_moe_execution_mode() == 'dense'
    encoder.train()
    assert encoder._sequence_packed_moe_execution_mode() == 'topk'


@pytest.mark.unit
@pytest.mark.run_only_on('GPU')
@pytest.mark.skipif(
    not torch.cuda.is_available()
    or not hasattr(torch.nn.functional, 'grouped_mm')
    or torch.cuda.get_device_capability()[0] < 8,
    reason='ragged grouped-mm requires PyTorch grouped_mm and SM80+',
)
@pytest.mark.parametrize('offset_values, empty_expert', [([0, 2, 7], 0), ([2, 2, 7], 1), ([2, 7, 7], 2)])
def test_ragged_grouped_mm_backward_handles_empty_expert_and_sum_loss(offset_values, empty_expert):
    x = torch.randn(7, 16, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    weights = [torch.randn(16, 8, device='cuda', requires_grad=True) for _ in range(3)]
    offsets = torch.tensor(offset_values, device='cuda', dtype=torch.int32)
    output = ggemm_module._ragged_grouped_mm(x, offsets, weights)
    output.sum().backward()

    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(weight.grad is not None and torch.isfinite(weight.grad).all() for weight in weights)
    assert torch.count_nonzero(weights[empty_expert].grad) == 0


@pytest.mark.unit
@pytest.mark.run_only_on('GPU')
@pytest.mark.skipif(not torch.cuda.is_available(), reason='grouped-mm fallback requires CUDA')
@pytest.mark.parametrize('training', [False, True])
def test_unaligned_sparse_moe_fallback_preserves_eval_and_train_gradients(training):
    speech_config = toy_speech_expert_cfg()
    speech_config.ff_expansion = 10 / speech_config.d_model
    encoder = (
        build_toy_pe_encoder(speech_expert_cfg=speech_config, freeze_speaker=False)
        .cuda()
        .to(torch.bfloat16)
        .train(training)
    )
    inputs = torch.randn(2, _MEL_FEATURES, 32, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    lengths = torch.tensor([32, 17], device='cuda')
    signal, signal_lengths = encoder._prepare_input(inputs, lengths)

    outputs = encoder.pee.forward_grouped_sequence_packed(
        signal,
        signal_lengths,
        backend='grouped_mm',
        moe_mode='topk',
    )
    loss = sum(output.data.float().square().mean() for output in outputs.values())
    auxiliary = encoder.pee.experts['speech'].get_moe_auxiliary_loss()
    (loss + auxiliary).backward()

    moe = encoder.pee.experts['speech'].layers[-1].ffn
    assert moe.experts[0].net[0].out_features == 10
    assert moe._last_grouped_backend == 'capacity_baddbmm'
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
    for name, expert in encoder.pee.experts.items():
        gradient = expert.pre_encode.proj.weight.grad
        assert gradient is not None, name
        assert torch.isfinite(gradient).all(), name


@pytest.mark.unit
@pytest.mark.run_only_on('GPU')
@pytest.mark.skipif(not torch.cuda.is_available(), reason='grouped THD numerical test requires CUDA')
def test_grouped_thd_cuda_matches_legacy_bhsd_outputs_and_full_gradients():
    torch.manual_seed(0)
    legacy = build_toy_pe_encoder().cuda().to(torch.bfloat16).train()
    grouped = build_toy_pe_encoder().cuda().to(torch.bfloat16).train()
    grouped.load_state_dict(legacy.state_dict())
    legacy_input = torch.randn(3, _MEL_FEATURES, 40, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    grouped_input = legacy_input.detach().clone().requires_grad_(True)
    lengths = torch.tensor([40, 23, 9], device='cuda')
    targets = torch.zeros(3, 5, _N_SPK, device='cuda', dtype=torch.bfloat16)

    legacy_output, output_lengths = legacy(legacy_input, lengths, spk_targets=targets)
    valid = torch.arange(legacy_output.shape[-1], device='cuda')[None, :] < output_lengths[:, None]
    legacy_valid = legacy_output.transpose(1, 2)[valid]
    legacy_valid.float().square().mean().backward()

    grouped_output = grouped.forward_sequence_packed(grouped_input, lengths, spk_targets=targets)
    grouped_output.data.float().square().mean().backward()

    assert torch.equal(grouped_output.lengths, output_lengths)
    torch.testing.assert_close(grouped_output.data, legacy_valid, rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(grouped_input.grad, legacy_input.grad, rtol=5e-2, atol=5e-2)
    for (name, legacy_parameter), (_, grouped_parameter) in zip(legacy.named_parameters(), grouped.named_parameters()):
        if not legacy_parameter.requires_grad:
            assert legacy_parameter.grad is None and grouped_parameter.grad is None
            continue
        assert legacy_parameter.grad is not None, name
        assert grouped_parameter.grad is not None, name
        torch.testing.assert_close(grouped_parameter.grad, legacy_parameter.grad, rtol=6e-2, atol=6e-2)

    trace = grouped.pee._last_sequence_packed_execution
    assert trace['attention_groups'] == trace['layers']
    expected_moe_backend = (
        'grouped_mm'
        if hasattr(torch.nn.functional, 'grouped_mm') and torch.cuda.get_device_capability()[0] >= 8
        else 'capacity_baddbmm'
    )
    assert grouped.pee.experts['speech'].layers[-1].ffn._last_grouped_backend == expected_moe_backend


@pytest.mark.unit
def test_grouped_sequence_packed_all_empty_keeps_parameter_gradients():
    encoder = build_toy_pe_encoder().train()
    signal = torch.randn(2, _MEL_FEATURES, 16, requires_grad=True)
    outputs = encoder.pee.forward_grouped_sequence_packed(signal, torch.zeros(2, dtype=torch.long), fused_qkv=True)
    loss = sum(output.data.sum() for output in outputs.values())
    loss = loss + encoder.pee.experts['speech'].get_moe_auxiliary_loss()
    loss.backward()

    assert all(output.total_tokens == 0 for output in outputs.values())
    for name, parameter in encoder.pee.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name


def _expert_loss(outputs, speech_expert):
    loss = sum(output.data.float().square().mean() for output in outputs.values())
    auxiliary = speech_expert.get_moe_auxiliary_loss()
    return loss if auxiliary is None else loss + auxiliary
