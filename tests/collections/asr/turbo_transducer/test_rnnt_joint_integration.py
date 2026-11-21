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

"""Integration of the Joint turbo kernel (`RNNTJoint(turbo_kernel=...)`) with the Joint module and the RNN-T loss."""

from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf
from omegaconf.errors import ConfigKeyError, ValidationError

import nemo.collections.asr.modules.rnnt as rnnt_module
from nemo.collections.asr.losses.rnnt import RNNTLoss
from nemo.collections.asr.modules.hybrid_autoregressive_transducer import HATJoint
from nemo.collections.asr.modules.rnnt import RNNTJoint, SampledRNNTJoint
from nemo.collections.asr.modules.rnnt_abstract import AbstractRNNTJoint
from nemo.collections.asr.parts.turbo_transducer.rnnt_consistency import ConsistencyRNNTLoss
from nemo.collections.asr.parts.turbo_transducer.types import (
    TurboJointBackwardPruningConfig,
    TurboJointKernelConfig,
    TurboJointKernelType,
)
from nemo.core.utils.optional_libs import TRITON_AVAILABLE

EXAMPLE_CONFIG_PATH = (
    Path(__file__).parents[4] / "examples" / "asr" / "conf" / "transformer" / "transformer_stacking_rnnt_bpe.yaml"
)
NUM_CLASSES = 8

requires_triton_cuda = pytest.mark.skipif(
    not (TRITON_AVAILABLE and torch.cuda.is_available()), reason="Triton and CUDA are required"
)


def _jointnet_config(dropout: float = 0.0, activation: str = "relu"):
    return {
        "encoder_hidden": 5,
        "pred_hidden": 6,
        "joint_hidden": 16,
        "activation": activation,
        "dropout": dropout,
    }


def _turbo_kernel_config(full_precision: bool = True, **kwargs):
    """Flat turbo kernel config; full precision allows comparing with the reference computed from the logits"""
    return {
        "type": "fused_vocab",
        "backend": "triton",
        "mm_full_precision": full_precision,
        "input_grad_full_precision": full_precision,
        "use_high_precision_mm": full_precision,
        "weight_grad_full_precision": full_precision,
        **kwargs,
    }


def _joint_inputs(batch_size: int = 2, device: str | torch.device = "cpu", requires_grad: bool = False):
    """Encoder output [B, D, T], prediction network output [B, D, U+1], lengths and targets"""
    encoder_outputs = torch.randn(batch_size, 5, 4, device=device, requires_grad=requires_grad)
    decoder_outputs = torch.randn(batch_size, 6, 4, device=device, requires_grad=requires_grad)
    targets = torch.randint(0, NUM_CLASSES, [batch_size, 3], device=device)
    source_lengths = torch.tensor([4, 3] * (batch_size // 2), device=device)
    target_lengths = torch.tensor([3, 2] * (batch_size // 2), device=device)
    return encoder_outputs, decoder_outputs, source_lengths, targets, target_lengths


class _PlainJoint(AbstractRNNTJoint):
    def project_encoder(self, encoder_output):
        return encoder_output

    def project_prednet(self, prednet_output):
        return prednet_output

    def joint_after_projection(self, encoder_output, predictor_output):
        return encoder_output + predictor_output


class TestTurboJointKernelConfig:
    @pytest.mark.unit
    def test_defaults(self):
        for config in [None, {}, {"type": None}]:
            kernel_config = TurboJointKernelConfig.from_config(config)
            assert kernel_config.kernel_type is None
            assert not kernel_config.uses_backward_pruning
            assert kernel_config.get_backward_pruning_range_size(current_step=10) is None

    @pytest.mark.unit
    @pytest.mark.parametrize("as_dict_config", [False, True])
    def test_flat_parameters_and_nested_pruning(self, as_dict_config: bool):
        config = _turbo_kernel_config(full_precision=False, backward_pruning={"range_size": 8, "start_step": 100})
        config["mm_full_precision"] = True
        kernel_config = TurboJointKernelConfig.from_config(OmegaConf.create(config) if as_dict_config else config)
        assert kernel_config.kernel_type is TurboJointKernelType.FUSED_VOCAB
        assert kernel_config.backend == "triton"
        assert kernel_config.mm_full_precision
        assert not kernel_config.use_high_precision_mm
        assert not kernel_config.input_grad_full_precision
        assert not kernel_config.weight_grad_full_precision
        assert kernel_config.backward_pruning == TurboJointBackwardPruningConfig(range_size=8, start_step=100)
        assert kernel_config.uses_backward_pruning

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "config,error",
        [
            # nested kwargs were supported by the draft implementation; parameters are flat now
            ({"type": "fused_vocab", "kwargs": {"backend": "triton"}}, ConfigKeyError),
            ({"type": "fused_vocab", "backward_pruning_range_size": 8}, ConfigKeyError),
            ({"type": "fused_vocab", "backward_pruning": {"range": 8}}, ConfigKeyError),
            ({"type": "fused_vocab", "backward_pruning": 8}, ValidationError),
            ({"type": "fused_vocab", "backward_pruning": {"range_size": "wide"}}, ValidationError),
            ({"type": "fused_vocab", "mm_full_precision": "maybe"}, ValidationError),
            ({"type": "fused"}, ValueError),  # fully fused Joint is not implemented yet
            ({"type": "fused_vocab", "backend": "torch"}, ValueError),
            ({"type": "fused_vocab", "backend": "unknown"}, ValueError),
            (["fused_vocab"], TypeError),
        ],
    )
    def test_rejects_invalid_config(self, config, error):
        with pytest.raises(error):
            TurboJointKernelConfig.from_config(config)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "pruning_config,current_step,expected_range_size",
        [
            (None, None, None),
            ({}, None, None),
            ({"range_size": None}, None, None),
            ({"range_size": None, "start_step": 5}, 10, None),
            ({"range_size": 0}, None, None),
            ({"range_size": -1}, None, None),
            ({"range_size": 2}, None, 2),
            ({"range_size": 2, "start_step": None}, None, 2),
            ({"range_size": 2, "start_step": 0}, None, 2),
            ({"range_size": 2, "start_step": -1}, None, 2),
            ({"range_size": 2, "start_step": 5}, 4, None),
            ({"range_size": 2, "start_step": 5}, 5, 2),
            ({"range_size": 2, "start_step": 5}, 6, 2),
        ],
    )
    def test_backward_pruning_schedule(self, pruning_config, current_step, expected_range_size):
        kernel_config = TurboJointKernelConfig.from_config({"type": "fused_vocab", "backward_pruning": pruning_config})
        assert kernel_config.get_backward_pruning_range_size(current_step) == expected_range_size
        assert kernel_config.uses_backward_pruning == ((pruning_config or {}).get("range_size") == 2)

    @pytest.mark.unit
    def test_scheduled_backward_pruning_requires_step(self):
        kernel_config = TurboJointKernelConfig.from_config(
            {"type": "fused_vocab", "backward_pruning": {"range_size": 2, "start_step": 5}}
        )
        with pytest.raises(ValueError, match="current step is required"):
            kernel_config.get_backward_pruning_range_size(current_step=None)

    @pytest.mark.unit
    def test_missing_backward_pruning_section_is_equivalent_to_null(self):
        without_section = TurboJointKernelConfig.from_config(OmegaConf.create({"type": "fused_vocab"}))
        with_null = TurboJointKernelConfig.from_config(
            OmegaConf.create({"type": "fused_vocab", "backward_pruning": None})
        )
        assert without_section == with_null
        assert without_section.backward_pruning is None


class TestJointTurboKernel:
    @pytest.mark.unit
    def test_turbo_kernel_is_disabled_by_default(self):
        assert not _PlainJoint().uses_turbo_kernel
        joint = RNNTJoint(jointnet=_jointnet_config(), num_classes=NUM_CLASSES)
        assert not joint.uses_turbo_kernel
        assert joint.turbo_kernel_type is None
        # `type: null` disables the kernel, other parameters are ignored
        joint = RNNTJoint(
            jointnet=_jointnet_config(),
            num_classes=NUM_CLASSES,
            turbo_kernel={"type": None, "backend": "cutile", "backward_pruning": {"range_size": 8}},
        )
        assert not joint.uses_turbo_kernel

    @pytest.mark.unit
    def test_joints_declare_support_for_loss_logprobs(self):
        assert not _PlainJoint().supports_loss_logprobs
        assert RNNTJoint(jointnet=_jointnet_config(), num_classes=NUM_CLASSES).supports_loss_logprobs
        # both subclasses require the full logits path
        for joint in [
            SampledRNNTJoint(jointnet=_jointnet_config(), num_classes=NUM_CLASSES, n_samples=4),
            HATJoint(jointnet=_jointnet_config(), num_classes=NUM_CLASSES),
        ]:
            assert not joint.supports_loss_logprobs
            assert not joint.supports_consistency_loss
            assert not joint.uses_turbo_kernel
            encoder_outputs, decoder_outputs, source_lengths, targets, target_lengths = _joint_inputs()
            with pytest.raises(NotImplementedError, match="forward_for_loss"):
                joint.forward_for_loss(encoder_outputs, decoder_outputs, source_lengths, targets, target_lengths)

    @pytest.mark.unit
    def test_example_config_is_accepted(self):
        """Turbo kernel section of the example config must be consistent with the Joint"""
        joint_config = OmegaConf.load(EXAMPLE_CONFIG_PATH).model.joint
        assert set(joint_config.turbo_kernel.keys()) == set(TurboJointKernelConfig.__dataclass_fields__.keys())
        assert set(joint_config.turbo_kernel.backward_pruning.keys()) == set(
            TurboJointBackwardPruningConfig.__dataclass_fields__.keys()
        )
        joint = RNNTJoint(
            jointnet=_jointnet_config(),
            num_classes=NUM_CLASSES,
            fuse_loss_wer=joint_config.fuse_loss_wer,
            fused_batch_size=joint_config.fused_batch_size,
            turbo_kernel=joint_config.turbo_kernel,
        )
        assert joint.turbo_kernel_type is TurboJointKernelType.FUSED_VOCAB
        # pruning is disabled by default (`range_size: null`)
        joint.train()
        assert joint._get_backward_pruning_range_size(current_step=10**6) is None

    @pytest.mark.unit
    def test_joint_rejects_incompatible_configuration(self):
        with pytest.raises(ValueError, match="fuse_loss_wer"):
            RNNTJoint(
                jointnet=_jointnet_config(),
                num_classes=NUM_CLASSES,
                turbo_kernel=_turbo_kernel_config(),
                fuse_loss_wer=True,
                fused_batch_size=1,
            )
        with pytest.raises(ValueError, match="extra Joint outputs"):
            RNNTJoint(
                jointnet=_jointnet_config(),
                num_classes=NUM_CLASSES,
                num_extra_outputs=5,
                turbo_kernel=_turbo_kernel_config(),
            )
        with pytest.raises(ValueError, match="preserve_memory"):
            RNNTJoint(
                jointnet=_jointnet_config(),
                num_classes=NUM_CLASSES,
                preserve_memory=True,
                turbo_kernel=_turbo_kernel_config(),
            )
        with pytest.raises(NotImplementedError, match="[Bb]ackward pruning"):
            RNNTJoint(
                jointnet=_jointnet_config(),
                num_classes=NUM_CLASSES,
                turbo_kernel=_turbo_kernel_config(backend="cutile", backward_pruning={"range_size": 8}),
            )
        with pytest.raises(ConfigKeyError):
            RNNTJoint(
                jointnet=_jointnet_config(),
                num_classes=NUM_CLASSES,
                turbo_kernel={"type": "fused_vocab", "kwargs": {"backend": "triton"}},
            )

        joint = RNNTJoint(jointnet=_jointnet_config(), num_classes=NUM_CLASSES, turbo_kernel=_turbo_kernel_config())
        with pytest.raises(ValueError, match="fuse_loss_wer"):
            joint.set_fuse_loss_wer(True)

    @pytest.mark.unit
    def test_forward_for_loss_requires_unit_temperature(self):
        joint = RNNTJoint(jointnet=_jointnet_config(), num_classes=NUM_CLASSES, turbo_kernel=_turbo_kernel_config())
        joint.temperature = 2.0
        with pytest.raises(ValueError, match="temperature"):
            joint.forward_for_loss(*_joint_inputs())

    @pytest.mark.unit
    def test_joint_passes_flat_parameters_to_kernel(self, monkeypatch):
        calls = []

        def fake_fused_vocab(joint_hidden, **kwargs):
            calls.append(kwargs)
            output = torch.zeros(joint_hidden.shape[:-1], device=joint_hidden.device)
            return output, output

        monkeypatch.setattr(rnnt_module, "rnnt_joint_fused_vocab", fake_fused_vocab)
        joint = RNNTJoint(
            jointnet=_jointnet_config(dropout=0.25),
            num_classes=NUM_CLASSES,
            turbo_kernel={
                "type": "fused_vocab",
                "backend": "cutile",
                "mm_full_precision": True,
                "weight_grad_full_precision": True,
            },
        )
        encoder_outputs, decoder_outputs, source_lengths, targets, target_lengths = _joint_inputs()
        joint.train()
        loss_inputs = joint.forward_for_loss(
            encoder_outputs=encoder_outputs,
            decoder_outputs=decoder_outputs,
            encoder_lengths=source_lengths,
            transcripts=targets,
            transcript_lengths=target_lengths,
        )
        assert loss_inputs.target_logprobs.shape == (2, 4, 4)
        assert loss_inputs.consistency_kl is None

        (kernel_kwargs,) = calls
        vocab_projection = joint.joint_net[-1]
        # live parameters (not copies): the kernel computes gradients for them
        assert kernel_kwargs.pop("weight") is vocab_projection.weight
        assert kernel_kwargs.pop("bias") is vocab_projection.bias
        assert kernel_kwargs.pop("targets") is targets
        assert kernel_kwargs.pop("src_lengths") is source_lengths
        assert kernel_kwargs.pop("tgt_lengths") is target_lengths
        assert kernel_kwargs == {
            "blank_id": NUM_CLASSES,
            "backend": "cutile",
            "use_high_precision_mm": False,
            "mm_full_precision": True,
            "input_grad_full_precision": False,
            "weight_grad_full_precision": True,
            "backward_pruning_range_size": None,
        }

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "pruning_config,current_step,training,grad_enabled,expected_range_size",
        [
            (None, None, True, True, None),
            ({}, 6, True, True, None),
            ({"start_step": 5}, 6, True, True, None),
            ({"range_size": None, "start_step": 5}, 6, True, True, None),
            ({"range_size": 0}, None, True, True, None),
            ({"range_size": -1}, None, True, True, None),
            ({"range_size": 2}, None, True, True, 2),
            ({"range_size": 2, "start_step": None}, None, True, True, 2),
            ({"range_size": 2, "start_step": 0}, 0, True, True, 2),
            ({"range_size": 2, "start_step": 5}, 4, True, True, None),
            ({"range_size": 2, "start_step": 5}, 5, True, True, 2),
            ({"range_size": 2, "start_step": 5}, 6, True, True, 2),
            # pruning approximates the gradient: never applied in evaluation mode / without gradients
            ({"range_size": 2}, None, False, True, None),
            ({"range_size": 2}, None, True, False, None),
            ({"range_size": 2, "start_step": 5}, None, False, True, None),
            ({"range_size": 2, "start_step": 5}, None, True, False, None),
        ],
    )
    def test_backward_pruning_schedule(
        self, monkeypatch, pruning_config, current_step, training, grad_enabled, expected_range_size
    ):
        range_sizes = []

        def fake_fused_vocab(backward_pruning_range_size, **kwargs):
            range_sizes.append(backward_pruning_range_size)
            return torch.zeros(2, 4, 4), torch.zeros(2, 4, 4)

        monkeypatch.setattr(rnnt_module, "rnnt_joint_fused_vocab", fake_fused_vocab)
        config = OmegaConf.create(_turbo_kernel_config(backward_pruning=pruning_config))
        joint = RNNTJoint(jointnet=_jointnet_config(), num_classes=NUM_CLASSES, turbo_kernel=config)
        # the schedule depends only on the step supplied by the model: a restored module behaves identically
        restored = RNNTJoint(jointnet=_jointnet_config(), num_classes=NUM_CLASSES, turbo_kernel=config)
        restored.load_state_dict(joint.state_dict())
        for module in [joint, restored]:
            module.train(training)
            with torch.set_grad_enabled(grad_enabled):
                module.forward_for_loss(*_joint_inputs(), current_step=current_step)
        assert range_sizes == [expected_range_size, expected_range_size]

    @pytest.mark.unit
    def test_scheduled_backward_pruning_requires_step_in_training(self):
        joint = RNNTJoint(
            jointnet=_jointnet_config(),
            num_classes=NUM_CLASSES,
            turbo_kernel=_turbo_kernel_config(backward_pruning={"range_size": 2, "start_step": 5}),
        )
        joint.train()
        with pytest.raises(ValueError, match="current step is required"):
            joint.forward_for_loss(*_joint_inputs())

    @pytest.mark.unit
    def test_turbo_joint_rejects_unsupported_consistency_loss(self):
        joint = RNNTJoint(
            jointnet=_jointnet_config(),
            num_classes=NUM_CLASSES,
            log_softmax=False,
            turbo_kernel=_turbo_kernel_config(backend="cutile"),
        )
        assert not joint.supports_consistency_loss
        with pytest.raises(NotImplementedError, match="Consistency loss"):
            joint.forward_for_loss(*_joint_inputs(batch_size=4), compute_consistency=True)

    @pytest.mark.unit
    @pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton is not installed")
    def test_turbo_joint_supports_consistency_loss_with_triton(self):
        for backend in ["triton", None]:
            joint = RNNTJoint(
                jointnet=_jointnet_config(),
                num_classes=NUM_CLASSES,
                turbo_kernel=_turbo_kernel_config(backend=backend),
            )
            assert joint.supports_consistency_loss

    @pytest.mark.unit
    @pytest.mark.gpu
    @requires_triton_cuda
    @pytest.mark.parametrize("masking_prob", [-1.0, 0.5])
    def test_turbo_kernel_does_not_change_standard_forward(self, masking_prob: float):
        """Decoding uses the standard forward: the turbo kernel must not affect it"""
        device = torch.device("cuda")
        standard_joint = RNNTJoint(
            jointnet=_jointnet_config(dropout=0.2), num_classes=NUM_CLASSES, masking_prob=masking_prob
        )
        turbo_joint = RNNTJoint(
            jointnet=_jointnet_config(dropout=0.2),
            num_classes=NUM_CLASSES,
            masking_prob=masking_prob,
            turbo_kernel=_turbo_kernel_config(),
        )
        turbo_joint.load_state_dict(standard_joint.state_dict(), strict=True)
        encoder_outputs, decoder_outputs, _, _, _ = _joint_inputs(device=device)
        for training in [False, True]:
            outputs = []
            for joint in [standard_joint, turbo_joint]:
                joint.to(device).train(training)
                torch.manual_seed(7)
                outputs.append(joint(encoder_outputs=encoder_outputs, decoder_outputs=decoder_outputs))
            assert torch.equal(outputs[0], outputs[1])

    @pytest.mark.unit
    @pytest.mark.gpu
    @requires_triton_cuda
    @pytest.mark.parametrize("masking_prob", [-1.0, 0.5])
    def test_joint_and_loss_integration_matches_standard(self, masking_prob: float):
        torch.manual_seed(42)
        device = torch.device("cuda")
        standard_joint = RNNTJoint(
            jointnet=_jointnet_config(), num_classes=NUM_CLASSES, log_softmax=False, masking_prob=masking_prob
        ).to(device)
        turbo_joint = RNNTJoint(
            jointnet=_jointnet_config(),
            num_classes=NUM_CLASSES,
            log_softmax=False,
            masking_prob=masking_prob,
            turbo_kernel=_turbo_kernel_config(),
        ).to(device)
        turbo_joint.load_state_dict(standard_joint.state_dict(), strict=True)

        standard_encoder, standard_decoder, source_lengths, targets, target_lengths = _joint_inputs(
            device=device, requires_grad=True
        )
        turbo_encoder = standard_encoder.detach().clone().requires_grad_(True)
        turbo_decoder = standard_decoder.detach().clone().requires_grad_(True)
        loss = RNNTLoss(
            num_classes=NUM_CLASSES, reduction="mean_volume", loss_name="turbo_rnnt", loss_kwargs={"backend": "triton"}
        )

        torch.manual_seed(7)  # prediction network masking
        standard_logits = standard_joint(encoder_outputs=standard_encoder, decoder_outputs=standard_decoder)
        standard_loss_value = loss(
            log_probs=standard_logits, targets=targets, input_lengths=source_lengths, target_lengths=target_lengths
        )
        torch.manual_seed(7)
        loss_inputs = turbo_joint.forward_for_loss(
            encoder_outputs=turbo_encoder,
            decoder_outputs=turbo_decoder,
            encoder_lengths=source_lengths,
            transcripts=targets,
            transcript_lengths=target_lengths,
        )
        assert loss_inputs.consistency_kl is None
        turbo_loss_value = loss.forward_from_logprobs(
            target_logprobs=loss_inputs.target_logprobs,
            blank_logprobs=loss_inputs.blank_logprobs,
            input_lengths=source_lengths,
            target_lengths=target_lengths,
        )

        assert torch.allclose(turbo_loss_value, standard_loss_value, atol=1e-4, rtol=1e-4)
        standard_loss_value.backward()
        turbo_loss_value.backward()

        assert torch.allclose(turbo_encoder.grad, standard_encoder.grad, atol=2e-4, rtol=2e-4)
        assert torch.allclose(turbo_decoder.grad, standard_decoder.grad, atol=2e-4, rtol=2e-4)
        turbo_parameters = dict(turbo_joint.named_parameters())
        for parameter_name, standard_parameter in standard_joint.named_parameters():
            turbo_parameter = turbo_parameters[parameter_name]
            assert standard_parameter.grad is not None, parameter_name
            assert turbo_parameter.grad is not None, parameter_name
            assert torch.allclose(turbo_parameter.grad, standard_parameter.grad, atol=2e-4, rtol=2e-4), parameter_name

    @pytest.mark.unit
    @pytest.mark.gpu
    @requires_triton_cuda
    def test_backward_pruning_keeps_loss_and_changes_gradient(self):
        """Pruning with a narrow range keeps the loss value, but approximates the gradient"""
        torch.manual_seed(42)
        device = torch.device("cuda")
        loss = RNNTLoss(num_classes=NUM_CLASSES, reduction="mean_volume", loss_name="turbo_rnnt")
        dense_joint = RNNTJoint(
            jointnet=_jointnet_config(), num_classes=NUM_CLASSES, turbo_kernel=_turbo_kernel_config()
        ).to(device)
        pruned_joint = RNNTJoint(
            jointnet=_jointnet_config(),
            num_classes=NUM_CLASSES,
            turbo_kernel=_turbo_kernel_config(backward_pruning={"range_size": 1, "start_step": 5}),
        ).to(device)
        pruned_joint.load_state_dict(dense_joint.state_dict(), strict=True)
        encoder_outputs, decoder_outputs, source_lengths, targets, target_lengths = _joint_inputs(device=device)

        def loss_and_weight_grad(joint: RNNTJoint, current_step: int):
            joint.zero_grad()
            loss_inputs = joint.forward_for_loss(
                encoder_outputs, decoder_outputs, source_lengths, targets, target_lengths, current_step=current_step
            )
            loss_value = loss.forward_from_logprobs(
                target_logprobs=loss_inputs.target_logprobs,
                blank_logprobs=loss_inputs.blank_logprobs,
                input_lengths=source_lengths,
                target_lengths=target_lengths,
            )
            loss_value.backward()
            return loss_value.detach(), joint.joint_net[-1].weight.grad.clone()

        dense_loss, dense_grad = loss_and_weight_grad(dense_joint, current_step=10)
        # before `start_step`: full gradient
        loss_before_start, grad_before_start = loss_and_weight_grad(pruned_joint, current_step=4)
        assert torch.equal(loss_before_start, dense_loss)
        assert torch.equal(grad_before_start, dense_grad)
        # after `start_step`: the same loss, pruned gradient
        loss_after_start, grad_after_start = loss_and_weight_grad(pruned_joint, current_step=5)
        assert torch.equal(loss_after_start, dense_loss)
        assert torch.isfinite(grad_after_start).all()
        assert not torch.allclose(grad_after_start, dense_grad, atol=1e-6, rtol=1e-4)

    @pytest.mark.unit
    @pytest.mark.gpu
    @requires_triton_cuda
    @pytest.mark.parametrize("use_turbo_kernel", [False, True])
    @pytest.mark.parametrize("symmetrical", [True, False])
    @pytest.mark.parametrize("reduction", ["mean", "mean_volume"])
    def test_joint_with_consistency_matches_separate_losses(
        self, use_turbo_kernel: bool, symmetrical: bool, reduction: str
    ):
        """`forward_for_loss` (single gradient buffer) must match the reference: full logits + two separate losses"""
        torch.manual_seed(42)
        device = torch.device("cuda")
        consistency_weight = 0.3
        half_batch_size = 2

        reference_joint = RNNTJoint(jointnet=_jointnet_config(), num_classes=NUM_CLASSES, log_softmax=False).to(device)
        fused_joint = RNNTJoint(
            jointnet=_jointnet_config(),
            num_classes=NUM_CLASSES,
            log_softmax=False,
            turbo_kernel=_turbo_kernel_config() if use_turbo_kernel else None,
        ).to(device)
        fused_joint.load_state_dict(reference_joint.state_dict(), strict=True)

        reference_encoder, reference_decoder, source_lengths, targets, target_lengths = _joint_inputs(
            batch_size=2 * half_batch_size, device=device, requires_grad=True
        )
        fused_encoder = reference_encoder.detach().clone().requires_grad_(True)
        fused_decoder = reference_decoder.detach().clone().requires_grad_(True)
        # double batch: targets of the student (second half) are the same as for the teacher
        targets = targets[:half_batch_size].repeat(2, 1)

        rnnt_loss = RNNTLoss(
            num_classes=NUM_CLASSES, reduction="mean_volume", loss_name="turbo_rnnt", loss_kwargs={"backend": "triton"}
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

        tolerance = dict(atol=2e-4, rtol=2e-4) if use_turbo_kernel else dict(atol=1e-5, rtol=1e-5)
        assert torch.allclose(fused_rnnt_loss_value, reference_rnnt_loss_value, **tolerance)
        assert torch.allclose(fused_consistency_value, reference_consistency_value, **tolerance)
        assert torch.allclose(fused_encoder.grad, reference_encoder.grad, **tolerance)
        assert torch.allclose(fused_decoder.grad, reference_decoder.grad, **tolerance)
        fused_parameters = dict(fused_joint.named_parameters())
        for parameter_name, reference_parameter in reference_joint.named_parameters():
            fused_parameter = fused_parameters[parameter_name]
            assert reference_parameter.grad is not None, parameter_name
            assert torch.allclose(fused_parameter.grad, reference_parameter.grad, **tolerance), parameter_name


class TestRNNTLossFromLogprobs:
    @pytest.mark.unit
    def test_supports_logprobs_input(self):
        assert RNNTLoss(num_classes=NUM_CLASSES, loss_name="turbo_rnnt").supports_logprobs_input
        # gradient clamping is applied to the gradient for the logits
        assert not RNNTLoss(
            num_classes=NUM_CLASSES, loss_name="turbo_rnnt", loss_kwargs={"clamp": 1.0}
        ).supports_logprobs_input

    @pytest.mark.unit
    @pytest.mark.parametrize("loss_name,loss_kwargs", [("warprnnt_numba", None), ("turbo_rnnt", {"clamp": 1.0})])
    def test_unsupported_loss_rejects_logprobs(self, loss_name, loss_kwargs):
        try:
            loss = RNNTLoss(num_classes=NUM_CLASSES, loss_name=loss_name, loss_kwargs=loss_kwargs)
        except ImportError:
            pytest.skip(f"{loss_name} loss is not available")
        assert not loss.supports_logprobs_input
        with pytest.raises(NotImplementedError, match="log-probabilities"):
            loss.forward_from_logprobs(
                target_logprobs=torch.zeros(2, 4, 4),
                blank_logprobs=torch.zeros(2, 4, 4),
                input_lengths=torch.tensor([4, 3]),
                target_lengths=torch.tensor([3, 2]),
            )

    @pytest.mark.unit
    @pytest.mark.gpu
    @requires_triton_cuda
    @pytest.mark.parametrize("reduction", [None, "mean", "sum", "mean_batch", "mean_volume"])
    @pytest.mark.parametrize("fastemit_lambda", [0.0, 0.01])
    def test_forward_from_logprobs_matches_forward(self, reduction, fastemit_lambda: float):
        torch.manual_seed(42)
        device = torch.device("cuda")
        loss = RNNTLoss(
            num_classes=NUM_CLASSES,
            reduction=reduction,
            loss_name="turbo_rnnt",
            loss_kwargs={"fastemit_lambda": fastemit_lambda},
        )
        logits = torch.randn(2, 4, 4, NUM_CLASSES + 1, device=device)
        targets = torch.randint(0, NUM_CLASSES, [2, 3], device=device)
        # int32 lengths: both entry points cast lengths
        source_lengths = torch.tensor([4, 3], device=device, dtype=torch.int32)
        target_lengths = torch.tensor([3, 2], device=device, dtype=torch.int32)
        logprobs = torch.log_softmax(logits, dim=-1)
        target_logprobs = torch.zeros(2, 4, 4, device=device)
        target_logprobs[:, :, :3] = logprobs[:, :, :3].gather(
            dim=-1, index=targets[:, None, :, None].expand(-1, 4, -1, -1)
        )[..., 0]
        blank_logprobs = logprobs[..., NUM_CLASSES]

        expected = loss(log_probs=logits, targets=targets, input_lengths=source_lengths, target_lengths=target_lengths)
        actual = loss.forward_from_logprobs(
            target_logprobs=target_logprobs,
            blank_logprobs=blank_logprobs,
            input_lengths=source_lengths,
            target_lengths=target_lengths,
        )
        assert actual.shape == expected.shape
        assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)
        # reduction of the wrapped loss is restored
        assert loss._loss.reduction is None
