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

import pytest
import torch

from nemo.collections.asr.modules.conformer_encoder import ConformerEncoder
from nemo.collections.asr.modules.transformer_encoder import TransformerEncoder
from nemo.collections.asr.parts.packed_sequence import pack_encoder_output
from nemo.collections.speechlm2.modules.perception import IndependentDualEncoder
from nemo.utils import logging


def make_encoder(d_model: int, *, subsampling_factor: int = 2) -> TransformerEncoder:
    return TransformerEncoder(
        feat_in=4,
        d_model=d_model,
        n_heads=2,
        n_layers=2,
        subsampling="feature_stacking",
        subsampling_factor=subsampling_factor,
        ff_expansion=1.0,
        self_attention_model="rope",
        pos_emb_max_len=64,
        drop_rate=0.0,
        dropout_pre_encoder=0.0,
        sync_max_audio_length=False,
    )


def test_independent_dual_encoder_chunks_only_frozen_auxiliary_branch():
    torch.manual_seed(7)
    asr = make_encoder(32)
    auxiliary = make_encoder(32)
    dual = IndependentDualEncoder(
        asr,
        auxiliary,
        frame_shift_seconds=0.01,
        asr_chunk_size_seconds=None,
        auxiliary_chunk_size_seconds=0.04,
        freeze_auxiliary=True,
    ).train()

    features = torch.randn(2, 4, 17)
    lengths = torch.tensor([17, 10], dtype=torch.int64)
    packed_features = pack_encoder_output(features.transpose(1, 2), lengths)
    with torch.no_grad():
        asr_reference = asr.forward_sequence_packed(packed_features, packed_features.lengths)

    auxiliary_calls = []
    auxiliary_forward = auxiliary.forward_sequence_packed

    def record_auxiliary_call(audio_signal, length, bypass_pre_encode=False, **kwargs):
        auxiliary_calls.append((length.detach().clone(), bypass_pre_encode))
        return auxiliary_forward(audio_signal, length, bypass_pre_encode=bypass_pre_encode, **kwargs)

    auxiliary.forward_sequence_packed = record_auxiliary_call
    output = dual.forward_sequence_packed(packed_features, packed_features.lengths)

    assert output.lengths.tolist() == [9, 5]
    assert output.data.shape == (14, 64)
    torch.testing.assert_close(output.data[:, :32], asr_reference.data)
    assert len(auxiliary_calls) == 1
    assert auxiliary_calls[0][0].tolist() == [2, 2, 2, 2, 1, 2, 2, 1]
    assert auxiliary_calls[0][1] is True
    assert not auxiliary.training
    assert all(not parameter.requires_grad for parameter in auxiliary.parameters())

    output.data.square().mean().backward()
    assert any(parameter.grad is not None for parameter in asr.parameters())
    assert all(parameter.grad is None for parameter in auxiliary.parameters())


def test_independent_dual_encoder_rejects_mismatched_frame_rates():
    asr = make_encoder(32, subsampling_factor=2)
    auxiliary = make_encoder(32, subsampling_factor=4)
    try:
        IndependentDualEncoder(asr, auxiliary, frame_shift_seconds=0.01)
    except ValueError as error:
        assert "subsampling_factor" in str(error)
    else:
        raise AssertionError("Expected mismatched subsampling factors to be rejected.")


def make_conformer_encoder(dropout_att: float) -> ConformerEncoder:
    return ConformerEncoder(
        feat_in=4,
        n_layers=1,
        d_model=32,
        n_heads=2,
        subsampling="striding",
        subsampling_factor=2,
        subsampling_conv_channels=4,
        self_attention_model="rel_pos",
        dropout=0.0,
        dropout_pre_encoder=0.0,
        dropout_emb=0.0,
        dropout_att=dropout_att,
    )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_independent_dual_encoder_needs_the_padded_fallback_for_a_branch_without_native_packed_support():
    """A ConformerEncoder branch training with attention dropout over rel_pos attention has no native packed form
    on CUDA (FlexAttention has no dropout): the dual encoder still constructs in training mode and refuses
    the step with that reason; with ``allow_padded_fallback=True`` it runs that branch padded and re-packs it,
    and the step completes forward and backward."""
    torch.manual_seed(7)
    asr = make_conformer_encoder(dropout_att=0.1)
    auxiliary = make_encoder(32)
    dual = IndependentDualEncoder(asr, auxiliary, frame_shift_seconds=0.01, freeze_auxiliary=True).train().cuda()
    assert not asr.supports_sequence_packed_output
    native_calls = []
    native_forward = asr.forward_sequence_packed

    def record_native_call(*args, **kwargs):
        native_calls.append(1)
        return native_forward(*args, **kwargs)

    asr.forward_sequence_packed = record_native_call
    features = torch.randn(2, 4, 17, device="cuda")
    lengths = torch.tensor([17, 10], dtype=torch.int64, device="cuda")
    with pytest.raises(NotImplementedError, match="FlexAttention has no dropout"):
        dual.forward_sequence_packed(pack_encoder_output(features.transpose(1, 2), lengths), lengths)
    dual.allow_padded_fallback = True
    output = dual.forward_sequence_packed(pack_encoder_output(features.transpose(1, 2), lengths), lengths)
    assert native_calls == [] and output.data.shape == (14, 64) and torch.isfinite(output.data).all()
    output.data.square().mean().backward()
    assert any(parameter.grad is not None for parameter in asr.parameters())
    dual.eval()
    assert asr.supports_sequence_packed_output


_UNSUPPORTED_REASON = "This toy configuration has no sequence-packed form."


def test_independent_dual_encoder_warns_once_about_the_padded_fallback_of_a_branch(monkeypatch, caplog, request):
    """The ASR branch reports no sequence-packed form: it runs padded, which the log says once, not per forward."""
    # caplog reads the root logger, to which the NeMo logger propagates only in some environments:
    # attach its handler to the NeMo logger alone, so that a record is captured exactly once
    monkeypatch.setattr(logging._logger, "propagate", False)
    logging._logger.addHandler(caplog.handler)
    request.addfinalizer(lambda: logging._logger.removeHandler(caplog.handler))
    torch.manual_seed(7)
    asr = make_conformer_encoder(dropout_att=0.0)
    dual = IndependentDualEncoder(asr, make_encoder(32), frame_shift_seconds=0.01, allow_padded_fallback=True)
    dual.eval()
    features = torch.randn(2, 17, 4)
    lengths = torch.tensor([17, 10], dtype=torch.int64)
    packed_features = pack_encoder_output(features, lengths)
    with torch.no_grad():
        native = dual.forward_sequence_packed(packed_features, lengths)
    assert "no sequence-packed form" not in caplog.text

    monkeypatch.setattr(logging, "once_logged", set())
    monkeypatch.setattr(asr, "sequence_packed_unsupported_reason", lambda: _UNSUPPORTED_REASON)
    assert not asr.supports_sequence_packed_output
    with torch.no_grad():
        first = dual.forward_sequence_packed(packed_features, lengths)
        second = dual.forward_sequence_packed(packed_features, lengths)

    warning = f"ConformerEncoder has no sequence-packed form in this configuration: {_UNSUPPORTED_REASON}"
    assert caplog.text.count(warning) == 1
    assert caplog.text.count("allow_padded_fallback=True") == 1
    assert torch.equal(first.lengths, native.lengths) and torch.equal(first.data, second.data)
    torch.testing.assert_close(first.data, native.data, rtol=1e-4, atol=1e-5)


def test_independent_dual_encoder_raises_with_the_reason_for_a_padded_fallback_by_default(monkeypatch):
    """Without the option, the branch without a sequence-packed form is an error; native branches still run."""
    asr = make_conformer_encoder(dropout_att=0.0)
    dual = IndependentDualEncoder(asr, make_encoder(32), frame_shift_seconds=0.01).eval()
    assert dual.allow_padded_fallback is False
    lengths = torch.tensor([17, 10], dtype=torch.int64)
    packed_features = pack_encoder_output(torch.randn(2, 17, 4), lengths)
    with torch.no_grad():
        assert torch.isfinite(dual.forward_sequence_packed(packed_features, lengths).data).all()

    monkeypatch.setattr(asr, "sequence_packed_unsupported_reason", lambda: _UNSUPPORTED_REASON)
    with pytest.raises(NotImplementedError, match=_UNSUPPORTED_REASON) as error, torch.no_grad():
        dual.forward_sequence_packed(packed_features, lengths)
    assert "ConformerEncoder" in str(error.value) and "allow_padded_fallback=True" in str(error.value)
    with pytest.raises(TypeError, match="allow_padded_fallback must be a bool"):
        IndependentDualEncoder(asr, make_encoder(32), frame_shift_seconds=0.01, allow_padded_fallback="false")
