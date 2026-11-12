# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
from unittest.mock import patch

import pytest
import torch
from torch import nn

from nemo.collections.tts.models.nemotron_tts.v2.base import NemotronTTSBase
from nemo.collections.tts.modules.aligner import init_training_alignment
from nemo.collections.tts.modules.magpietts_modules import CodecHelper
from tests.collections.tts.modules.nemotron_tts.v2.test_training_model import _inputs, _training_model

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_model_conversion_preserves_fp32_modules_and_shared_quantizer(dtype):
    model = _precision_model()
    converter = nn.Module()
    converter.vector_quantizer_original = model._codec_model.vector_quantizer
    converter.vector_quantizer_new = _Quantizer()
    model._codec_converter = converter.requires_grad_(False)
    protected = (model._codec_model, model._codec_converter, model.text_aligner)
    originals = [{name: value.clone() for name, value in module.state_dict().items()} for module in protected]
    parameter_ids = {name: id(parameter) for name, parameter in model.named_parameters()}
    trainable = {name: parameter.requires_grad for name, parameter in model.named_parameters()}
    keys = set(model.state_dict())

    model.to(dtype=dtype)

    assert model.decoder.get_input_embeddings().weight.dtype == dtype
    assert model.final_proj.weight.dtype == dtype
    assert parameter_ids == {name: id(parameter) for name, parameter in model.named_parameters()}
    assert trainable == {name: parameter.requires_grad for name, parameter in model.named_parameters()}
    assert keys == set(model.state_dict())
    assert model._codec_converter.vector_quantizer_original is model._codec_model.vector_quantizer
    for module, original in zip(protected, originals):
        for name, value in module.state_dict().items():
            assert value.dtype == original[name].dtype
            torch.testing.assert_close(value, original[name], rtol=0, atol=0)

    if torch.cuda.is_available():
        model.cuda()
        assert all(value.device.type == 'cuda' for value in model.parameters())
        assert all(value.device.type == 'cuda' for value in model.buffers())
        assert model._codec_converter.vector_quantizer_original is model._codec_model.vector_quantizer
        for module, original in zip(protected, originals):
            for name, value in module.state_dict().items():
                torch.testing.assert_close(value.cpu(), original[name], rtol=0, atol=0)
        model.cpu()
        assert all(value.device.type == 'cpu' for value in model.parameters())


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_aligner_initialization_keeps_fp32_under_true_precision_default_dtype(dtype):
    model = _training_model(enabled=False)
    model._cfg.use_attention_alignment = True
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        init_training_alignment(model, model.cfg)
        assert torch.get_default_dtype() == dtype
        assert all(value.dtype == torch.float32 for value in model.text_aligner.parameters())
        assert all(
            not value.is_floating_point() or value.dtype == torch.float32 for value in model.text_aligner.buffers()
        )
    finally:
        torch.set_default_dtype(previous)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='True precision training requires CUDA')
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('alignment', [False, True])
def test_true_precision_process_batch_has_finite_losses_and_gradients(dtype, alignment):
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip('CUDA device does not support bfloat16')
    model = _precision_model(enabled=alignment).to(device='cuda', dtype=dtype)
    inputs = {name: value.cuda() for name, value in _inputs(model).items()}

    output = model.process_batch(**inputs, mode='val')

    assert output.logits.dtype == dtype
    assert output.loss.dtype == torch.float32 and torch.isfinite(output.loss)
    output.loss.backward()
    assert model.decoder.get_input_embeddings().weight.grad.dtype == dtype
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    assert model.final_proj.weight.grad.abs().sum() > 0
    if alignment:
        assert output.aligner_loss.dtype == torch.float32 and torch.isfinite(output.aligner_loss)
        assert output.attention_alignment_loss.dtype == torch.float32
        teacher_gradients = [
            parameter.grad for parameter in model.text_aligner.parameters() if parameter.grad is not None
        ]
        assert teacher_gradients and all(gradient.dtype == torch.float32 for gradient in teacher_gradients)
        assert any(gradient.abs().sum() > 0 for gradient in teacher_gradients)
    else:
        assert output.aligner_loss is None and output.attention_alignment_loss is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Low precision CUDA cross entropy regression')
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('head', ['audio', 'phoneme'])
def test_long_sequence_token_loss_accumulates_in_fp32(dtype, head):
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip('CUDA device does not support bfloat16')
    model = _training_model(enabled=False)
    model.phoneme_stacking_factor = 1
    model.phoneme_vocab_size = model.num_all_tokens_per_codebook
    steps = 70000
    logits = torch.zeros(1, steps, model.num_all_tokens_per_codebook, device='cuda', dtype=dtype, requires_grad=True)
    targets = torch.zeros(1, 1, steps, device='cuda', dtype=torch.long)
    lengths = torch.tensor([steps], device='cuda')
    compute = model.compute_loss if head == 'audio' else model.compute_phoneme_loss

    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        loss, _ = compute(logits, targets, lengths)
    finally:
        torch.set_default_dtype(previous)

    assert loss.dtype == torch.float32 and torch.isfinite(loss)
    torch.testing.assert_close(
        loss, loss.new_tensor(math.log(model.num_all_tokens_per_codebook)), rtol=1e-6, atol=1e-6
    )
    loss.backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0


class _Quantizer(nn.Module):
    codebook_dim = 4

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1.0001, 0.1234567]))
        self.register_buffer('statistics', torch.tensor([0.9876543]))
        self.register_buffer('indices', torch.tensor([1, 2]))

    def decode(self, indices, input_len):
        values = indices.transpose(0, 1).float().repeat_interleave(2, dim=1) / 16
        mask = torch.arange(values.size(2), device=values.device)[None, :] < input_len[:, None]
        return values * mask[:, None]


def _precision_model(enabled=True):
    model = _training_model(enabled=enabled)
    codec = nn.Module()
    codec.vector_quantizer = _Quantizer()
    codec.codebook_size = 16
    model._codec_model = codec.requires_grad_(False)
    return model


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_codec_helper_preserves_fp32_default_dtype_and_waveforms(dtype):
    class Codec:
        def eval(self):
            return self

        def encode(self, audio, audio_len, sample_rate):
            assert audio.dtype == torch.float32 and torch.get_default_dtype() == torch.float32
            return audio.long().unsqueeze(1), audio_len

        def decode(self, tokens, tokens_len):
            assert torch.get_default_dtype() == torch.float32
            return tokens[:, 0].float(), tokens_len

    helper = CodecHelper(Codec(), force_fp32=True)
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        codes, lengths = helper.audio_to_codes(torch.ones(1, 32), torch.tensor([32]))
        audio, _, _ = helper.codes_to_audio(codes, lengths)
        assert audio.dtype == torch.float32 and torch.get_default_dtype() == dtype
    finally:
        torch.set_default_dtype(previous)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_context_audio_resampling_runs_in_fp32(dtype):
    import numpy as np

    def resample(waveform, orig_freq, new_freq):
        assert waveform.dtype == torch.float32 and torch.get_default_dtype() == torch.float32
        return waveform

    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        with (
            patch(
                'nemo.collections.tts.models.nemotron_tts.v2.base.sf.read',
                return_value=(np.zeros(32, np.float32), 16000),
            ),
            patch('nemo.collections.tts.models.nemotron_tts.v2.base.resample', side_effect=resample),
        ):
            audio = NemotronTTSBase._load_audio_for_inference('context.wav', 24000)
        assert audio.dtype == torch.float32 and torch.get_default_dtype() == dtype
    finally:
        torch.set_default_dtype(previous)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('channel', ['audio', 'context'])
def test_codec_conversion_uses_fp32_integer_division_under_true_precision(dtype, channel):
    model = _training_model(enabled=False)
    calls = []

    def convert(audio_tokens, audio_lens):
        decoded = audio_tokens / torch.ones_like(audio_tokens)
        assert decoded.dtype == torch.float32
        calls.append(decoded)
        return decoded.long()

    converter = nn.Module()
    converter.convert_original_to_new = convert
    model._codec_converter = converter
    inputs = _inputs(model)
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        if channel == 'audio':
            model.prepare_audio_channel_embeddings(
                inputs['audio_codes'], inputs['audio_codes_lens'], delay=torch.tensor([5])
            )
        else:
            model.use_speaker_encoder = False
            model.task_embedding = None
            NemotronTTSBase.prepare_context_tensors(
                model,
                inputs['context_text_tokens'],
                inputs['context_text_tokens_lens'],
                inputs['context_audio_codes'],
                inputs['context_audio_codes_lens'],
            )
        assert len(calls) == 1 and torch.get_default_dtype() == dtype
    finally:
        torch.set_default_dtype(previous)
