# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import nemo.collections.asr.modules.rnnt as rnnt_module
from nemo.collections.asr.losses.rnnt import RNNTLoss
from nemo.collections.asr.models.rnnt_models import EncDecRNNTModel
from nemo.collections.asr.modules.hybrid_autoregressive_transducer import HATJoint
from nemo.collections.asr.modules.rnnt import RNNTJoint, SampledRNNTJoint
from nemo.collections.asr.modules.rnnt_abstract import AbstractRNNTJoint
from nemo.collections.asr.parts.turbo_transducer.rnnt_consistency import ConsistencyRNNTLoss
from nemo.collections.asr.parts.turbo_transducer.types import TransducerLossInputs
from nemo.core.utils.optional_libs import TRITON_AVAILABLE


def _jointnet_config(dropout: float = 0.0, activation: str = "relu"):
    return {
        "encoder_hidden": 5,
        "pred_hidden": 6,
        "joint_hidden": 16,
        "activation": activation,
        "dropout": dropout,
    }


def _turbo_kernel_config(kernel_type: str):
    return {
        "type": kernel_type,
        "kwargs": {
            "backend": "triton",
            "mm_full_precision": True,
            "input_grad_full_precision": True,
            "use_high_precision_mm": True,
            "weight_grad_full_precision": True,
        },
    }


class _PlainJoint(AbstractRNNTJoint):
    def project_encoder(self, encoder_output):
        return encoder_output

    def project_prednet(self, prednet_output):
        return prednet_output

    def joint_after_projection(self, encoder_output, predictor_output):
        return encoder_output + predictor_output


@pytest.mark.unit
def test_abstract_joint_does_not_use_turbo_kernel_by_default():
    assert not _PlainJoint().uses_turbo_kernel


@pytest.mark.unit
@pytest.mark.parametrize("kernel_type", ["fused_vocab", "fused"])
def test_joint_passes_live_vocab_projection_parameters(monkeypatch, kernel_type):
    calls = []

    def fake_fused_vocab(joint_hidden, weight, bias, **kwargs):
        calls.append({"weight": weight, "bias": bias})
        output_shape = joint_hidden.shape[:-1]
        output = torch.zeros(output_shape, device=joint_hidden.device)
        return output, output

    def fake_fused(
        encoder_output_projected,
        predictor_output_projected,
        weight,
        bias,
        dropout_p,
        **kwargs,
    ):
        calls.append({"weight": weight, "bias": bias, "dropout_p": dropout_p})
        output_shape = (
            encoder_output_projected.shape[0],
            encoder_output_projected.shape[1],
            predictor_output_projected.shape[1],
        )
        output = torch.zeros(output_shape, device=encoder_output_projected.device)
        return output, output

    monkeypatch.setattr(rnnt_module, "rnnt_joint_fused_vocab", fake_fused_vocab)
    monkeypatch.setattr(rnnt_module, "rnnt_fused_joint", fake_fused)

    joint = RNNTJoint(
        jointnet=_jointnet_config(dropout=0.25),
        num_classes=8,
        log_softmax=False,
        turbo_kernel=_turbo_kernel_config(kernel_type),
    )
    encoder_outputs = torch.randn(2, 5, 4)
    decoder_outputs = torch.randn(2, 6, 4)
    targets = torch.randint(0, 8, [2, 3])
    source_lengths = torch.tensor([4, 3])
    target_lengths = torch.tensor([3, 2])

    joint.train()
    joint.forward_for_loss(
        encoder_outputs=encoder_outputs,
        decoder_outputs=decoder_outputs,
        encoder_lengths=source_lengths,
        transcripts=targets,
        transcript_lengths=target_lengths,
    )

    vocab_projection = joint.joint_net[-1]
    assert calls[-1]["weight"] is vocab_projection.weight
    assert calls[-1]["bias"] is vocab_projection.bias
    if kernel_type == "fused":
        assert calls[-1]["dropout_p"] == 0.25
        joint.eval()
        joint.forward_for_loss(
            encoder_outputs=encoder_outputs,
            decoder_outputs=decoder_outputs,
            encoder_lengths=source_lengths,
            transcripts=targets,
            transcript_lengths=target_lengths,
        )
        assert calls[-1]["dropout_p"] == 0.0


@pytest.mark.unit
def test_model_routes_joint_output_to_loss_logprobs_path():
    target_logprobs = torch.randn(2, 4, 4)
    blank_logprobs = torch.randn(2, 4, 4)
    loss_value = torch.tensor(1.5)
    joint = SimpleNamespace(
        uses_turbo_kernel=True,
        supports_loss_logprobs=True,
        forward_for_loss=Mock(return_value=TransducerLossInputs(target_logprobs, blank_logprobs)),
    )
    loss = SimpleNamespace(supports_logprobs_input=True, forward_from_logprobs=Mock(return_value=loss_value))
    model = SimpleNamespace(joint=joint, loss=loss, consistency_loss=None)
    model.uses_loss_logprobs_path = EncDecRNNTModel.uses_loss_logprobs_path.fget(model)
    encoded = torch.randn(2, 5, 4)
    decoder = torch.randn(2, 6, 4)
    transcript = torch.randint(0, 8, [2, 3])
    source_lengths = torch.tensor([4, 3])
    target_lengths = torch.tensor([3, 2])

    actual_loss, consistency_loss = EncDecRNNTModel._compute_losses(
        model,
        encoded=encoded,
        decoder=decoder,
        transcript=transcript,
        encoded_len=source_lengths,
        target_length=target_lengths,
    )

    assert actual_loss is loss_value
    assert consistency_loss is None
    joint.forward_for_loss.assert_called_once_with(
        encoder_outputs=encoded,
        decoder_outputs=decoder,
        encoder_lengths=source_lengths,
        transcripts=transcript,
        transcript_lengths=target_lengths,
        compute_consistency=False,
        consistency_symmetrical=True,
    )
    loss.forward_from_logprobs.assert_called_once_with(
        target_logprobs=target_logprobs,
        blank_logprobs=blank_logprobs,
        input_lengths=source_lengths,
        target_lengths=target_lengths,
    )


@pytest.mark.unit
def test_model_routes_consistency_loss_to_joint():
    target_logprobs = torch.randn(4, 4, 4)
    blank_logprobs = torch.randn(4, 4, 4)
    consistency_kl = torch.randn(2, 4, 4)
    loss_value = torch.tensor(1.5)
    consistency_loss_value = torch.tensor(0.25)
    joint = SimpleNamespace(
        uses_turbo_kernel=False,
        supports_loss_logprobs=True,
        forward_for_loss=Mock(return_value=TransducerLossInputs(target_logprobs, blank_logprobs, consistency_kl)),
    )
    loss = SimpleNamespace(supports_logprobs_input=True, forward_from_logprobs=Mock(return_value=loss_value))
    consistency_loss = SimpleNamespace(symmetrical=False, forward_from_kl=Mock(return_value=consistency_loss_value))
    model = SimpleNamespace(joint=joint, loss=loss, consistency_loss=consistency_loss)
    model.uses_loss_logprobs_path = EncDecRNNTModel.uses_loss_logprobs_path.fget(model)
    encoded = torch.randn(4, 5, 4)
    decoder = torch.randn(4, 6, 4)
    transcript = torch.randint(0, 8, [4, 3])
    source_lengths = torch.tensor([4, 3, 4, 3])
    target_lengths = torch.tensor([3, 2, 3, 2])

    actual_loss, actual_consistency_loss = EncDecRNNTModel._compute_losses(
        model,
        encoded=encoded,
        decoder=decoder,
        transcript=transcript,
        encoded_len=source_lengths,
        target_length=target_lengths,
        compute_consistency=True,
    )

    assert actual_loss is loss_value
    assert actual_consistency_loss is consistency_loss_value
    assert joint.forward_for_loss.call_args.kwargs["compute_consistency"] is True
    assert joint.forward_for_loss.call_args.kwargs["consistency_symmetrical"] is False
    consistency_call = consistency_loss.forward_from_kl.call_args.kwargs
    assert consistency_call["consistency_kl"] is consistency_kl
    # only the teacher (first) half of the lengths is used for the consistency loss
    assert torch.equal(consistency_call["src_lengths"], source_lengths[:2])
    assert torch.equal(consistency_call["tgt_lengths"], target_lengths[:2])


@pytest.mark.unit
def test_joint_rejects_incompatible_turbo_configuration():
    with pytest.raises(ValueError, match="fuse_loss_wer"):
        RNNTJoint(
            jointnet=_jointnet_config(),
            num_classes=8,
            turbo_kernel=_turbo_kernel_config("fused"),
            fuse_loss_wer=True,
            fused_batch_size=1,
        )

    with pytest.raises(ValueError, match="only relu"):
        RNNTJoint(
            jointnet=_jointnet_config(activation="tanh"),
            num_classes=8,
            turbo_kernel=_turbo_kernel_config("fused"),
        )


@pytest.mark.unit
def test_joints_declare_support_for_loss_logprobs():
    assert not _PlainJoint().supports_loss_logprobs
    assert RNNTJoint(jointnet=_jointnet_config(), num_classes=8).supports_loss_logprobs
    # both subclasses require the full logits path
    assert not SampledRNNTJoint(jointnet=_jointnet_config(), num_classes=8, n_samples=4).supports_loss_logprobs
    assert not HATJoint(jointnet=_jointnet_config(), num_classes=8).supports_loss_logprobs


@pytest.mark.unit
@pytest.mark.parametrize("kernel_type", ["fused_vocab", "fused"])
def test_turbo_joint_rejects_consistency_loss(kernel_type):
    joint = RNNTJoint(
        jointnet=_jointnet_config(),
        num_classes=8,
        log_softmax=False,
        turbo_kernel=_turbo_kernel_config(kernel_type),
    )
    with pytest.raises(NotImplementedError, match="Consistency loss"):
        joint.forward_for_loss(
            encoder_outputs=torch.randn(2, 5, 4),
            decoder_outputs=torch.randn(2, 6, 4),
            encoder_lengths=torch.tensor([4, 4]),
            transcripts=torch.randint(0, 8, [2, 3]),
            transcript_lengths=torch.tensor([3, 3]),
            compute_consistency=True,
        )


@pytest.mark.unit
@pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton is not installed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("kernel_type", ["fused_vocab", "fused"])
def test_joint_and_loss_integration_matches_standard(kernel_type):
    torch.manual_seed(42)
    device = torch.device("cuda")
    standard_joint = RNNTJoint(
        jointnet=_jointnet_config(),
        num_classes=8,
        log_softmax=False,
    ).to(device)
    optimized_joint = RNNTJoint(
        jointnet=_jointnet_config(),
        num_classes=8,
        log_softmax=False,
        turbo_kernel=_turbo_kernel_config(kernel_type),
    ).to(device)
    optimized_joint.load_state_dict(standard_joint.state_dict(), strict=True)

    standard_encoder = torch.randn(2, 5, 4, device=device, requires_grad=True)
    standard_decoder = torch.randn(2, 6, 4, device=device, requires_grad=True)
    optimized_encoder = standard_encoder.detach().clone().requires_grad_(True)
    optimized_decoder = standard_decoder.detach().clone().requires_grad_(True)
    targets = torch.randint(0, 8, [2, 3], device=device)
    source_lengths = torch.tensor([4, 3], device=device)
    target_lengths = torch.tensor([3, 2], device=device)

    with torch.no_grad():
        standard_logits = standard_joint(
            encoder_outputs=standard_encoder,
            decoder_outputs=standard_decoder,
        )
        optimized_standard_logits = optimized_joint(
            encoder_outputs=optimized_encoder,
            decoder_outputs=optimized_decoder,
        )
    assert torch.equal(optimized_standard_logits, standard_logits)

    standard_loss = RNNTLoss(
        num_classes=8,
        reduction="mean_volume",
        loss_name="turbo_rnnt",
        loss_kwargs={"backend": "triton"},
    )
    optimized_loss = RNNTLoss(
        num_classes=8,
        reduction="mean_volume",
        loss_name="turbo_rnnt",
        loss_kwargs={"backend": "triton"},
    )

    standard_logits = standard_joint(
        encoder_outputs=standard_encoder,
        decoder_outputs=standard_decoder,
    )
    standard_loss_value = standard_loss(
        log_probs=standard_logits,
        targets=targets,
        input_lengths=source_lengths,
        target_lengths=target_lengths,
    )
    loss_inputs = optimized_joint.forward_for_loss(
        encoder_outputs=optimized_encoder,
        decoder_outputs=optimized_decoder,
        encoder_lengths=source_lengths,
        transcripts=targets,
        transcript_lengths=target_lengths,
    )
    assert loss_inputs.consistency_kl is None
    optimized_loss_value = optimized_loss.forward_from_logprobs(
        target_logprobs=loss_inputs.target_logprobs,
        blank_logprobs=loss_inputs.blank_logprobs,
        input_lengths=source_lengths,
        target_lengths=target_lengths,
    )

    assert torch.allclose(optimized_loss_value, standard_loss_value, atol=1e-4, rtol=1e-4)
    standard_loss_value.backward()
    optimized_loss_value.backward()

    assert torch.allclose(optimized_encoder.grad, standard_encoder.grad, atol=2e-4, rtol=2e-4)
    assert torch.allclose(optimized_decoder.grad, standard_decoder.grad, atol=2e-4, rtol=2e-4)
    optimized_parameters = dict(optimized_joint.named_parameters())
    for parameter_name, standard_parameter in standard_joint.named_parameters():
        optimized_parameter = optimized_parameters[parameter_name]
        assert standard_parameter.grad is not None, parameter_name
        assert optimized_parameter.grad is not None, parameter_name
        assert torch.allclose(
            optimized_parameter.grad,
            standard_parameter.grad,
            atol=2e-4,
            rtol=2e-4,
        ), parameter_name

    assert optimized_joint.joint_net[-1].weight.grad is not None
    assert optimized_joint.joint_net[-1].bias.grad is not None


@pytest.mark.unit
@pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton is not installed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("symmetrical", [True, False])
@pytest.mark.parametrize("reduction", ["mean", "mean_volume"])
def test_joint_with_consistency_matches_separate_losses(symmetrical, reduction):
    """The fused (single gradient buffer) path must match the reference: full logits + two separate losses."""
    torch.manual_seed(42)
    device = torch.device("cuda")
    num_classes = 8
    consistency_weight = 0.3
    half_batch_size = 2

    reference_joint = RNNTJoint(jointnet=_jointnet_config(), num_classes=num_classes, log_softmax=False).to(device)
    fused_joint = RNNTJoint(jointnet=_jointnet_config(), num_classes=num_classes, log_softmax=False).to(device)
    fused_joint.load_state_dict(reference_joint.state_dict(), strict=True)

    reference_encoder = torch.randn(2 * half_batch_size, 5, 4, device=device, requires_grad=True)
    reference_decoder = torch.randn(2 * half_batch_size, 6, 4, device=device, requires_grad=True)
    fused_encoder = reference_encoder.detach().clone().requires_grad_(True)
    fused_decoder = reference_decoder.detach().clone().requires_grad_(True)
    targets = torch.randint(0, num_classes, [half_batch_size, 3], device=device).repeat(2, 1)
    source_lengths = torch.tensor([4, 3], device=device).repeat(2)
    target_lengths = torch.tensor([3, 2], device=device).repeat(2)

    rnnt_loss = RNNTLoss(
        num_classes=num_classes, reduction="mean_volume", loss_name="turbo_rnnt", loss_kwargs={"backend": "triton"}
    )
    consistency_loss = ConsistencyRNNTLoss(symmetrical=symmetrical, reduction=reduction, backend="triton")

    # reference: full logits, two independent losses
    reference_logits = reference_joint(encoder_outputs=reference_encoder, decoder_outputs=reference_decoder)
    reference_rnnt_loss_value = rnnt_loss(
        log_probs=reference_logits, targets=targets, input_lengths=source_lengths, target_lengths=target_lengths
    )
    reference_consistency_value = consistency_loss(
        teacher_logits=reference_logits[:half_batch_size],
        student_logits=reference_logits[half_batch_size:],
        src_lengths=source_lengths[:half_batch_size],
        tgt_lengths=target_lengths[:half_batch_size],
    )
    (reference_rnnt_loss_value + consistency_weight * reference_consistency_value).backward()

    # fused: per-position quantities, one gradient buffer
    loss_inputs = fused_joint.forward_for_loss(
        encoder_outputs=fused_encoder,
        decoder_outputs=fused_decoder,
        encoder_lengths=source_lengths,
        transcripts=targets,
        transcript_lengths=target_lengths,
        compute_consistency=True,
        consistency_symmetrical=symmetrical,
    )
    # [B, T, U+1]; encoder/decoder outputs above are [B, D, T] and [B, D, U+1]
    assert loss_inputs.consistency_kl.shape == (half_batch_size, 4, 4)
    fused_rnnt_loss_value = rnnt_loss.forward_from_logprobs(
        target_logprobs=loss_inputs.target_logprobs,
        blank_logprobs=loss_inputs.blank_logprobs,
        input_lengths=source_lengths,
        target_lengths=target_lengths,
    )
    fused_consistency_value = consistency_loss.forward_from_kl(
        consistency_kl=loss_inputs.consistency_kl,
        src_lengths=source_lengths[:half_batch_size],
        tgt_lengths=target_lengths[:half_batch_size],
    )
    (fused_rnnt_loss_value + consistency_weight * fused_consistency_value).backward()

    assert torch.allclose(fused_rnnt_loss_value, reference_rnnt_loss_value, atol=1e-5, rtol=1e-5)
    assert torch.allclose(fused_consistency_value, reference_consistency_value, atol=1e-5, rtol=1e-5)
    assert torch.allclose(fused_encoder.grad, reference_encoder.grad, atol=1e-5, rtol=1e-5)
    assert torch.allclose(fused_decoder.grad, reference_decoder.grad, atol=1e-5, rtol=1e-5)
    fused_parameters = dict(fused_joint.named_parameters())
    for parameter_name, reference_parameter in reference_joint.named_parameters():
        fused_parameter = fused_parameters[parameter_name]
        assert reference_parameter.grad is not None, parameter_name
        assert torch.allclose(fused_parameter.grad, reference_parameter.grad, atol=1e-5, rtol=1e-5), parameter_name
