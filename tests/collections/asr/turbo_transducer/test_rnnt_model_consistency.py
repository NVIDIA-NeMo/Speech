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

"""Model-level integration tests for the consistency loss (`EncDecRNNTModel`)."""

import pytest
import torch
from omegaconf import DictConfig, ListConfig

from nemo.collections.asr.models.rnnt_models import EncDecRNNTModel
from nemo.core.utils.optional_libs import TRITON_AVAILABLE

LABELS = [' ', 'a', 'b', 'c', 'd']
ENCODER_HIDDEN = 16
PREDICTOR_HIDDEN = 8


def _model_config(
    loss_name: str = "turbo_rnnt",
    consistency_weight: float = 0.2,
    consistency_symmetrical: bool = True,
    turbo_kernel_type: str = None,
    fuse_loss_wer: bool = False,
) -> DictConfig:
    preprocessor = {'_target_': 'nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor', 'features': 64}
    encoder = {
        '_target_': 'nemo.collections.asr.modules.ConvASREncoder',
        'feat_in': 64,
        'activation': 'relu',
        'conv_mask': True,
        'jasper': [
            {
                'filters': ENCODER_HIDDEN,
                'repeat': 1,
                'kernel': [1],
                'stride': [1],
                'dilation': [1],
                'dropout': 0.0,
                'residual': False,
                'separable': True,
            }
        ],
    }
    decoder = {
        '_target_': 'nemo.collections.asr.modules.RNNTDecoder',
        'prednet': {'pred_hidden': PREDICTOR_HIDDEN, 'pred_rnn_layers': 1},
    }
    joint = {
        '_target_': 'nemo.collections.asr.modules.RNNTJoint',
        'jointnet': {'joint_hidden': 8, 'activation': 'relu'},
        'log_softmax': False,
        'turbo_kernel': {'type': turbo_kernel_type, 'kwargs': {'backend': 'triton'}},
        'fuse_loss_wer': fuse_loss_wer,
        'fused_batch_size': 1 if fuse_loss_wer else None,
    }
    loss = {
        'loss_name': loss_name,
        'consistency': {
            'weight': consistency_weight,
            'symmetrical': consistency_symmetrical,
            'reduction': 'mean_volume',
            'backend': 'triton',
        },
    }
    return DictConfig(
        {
            'labels': ListConfig(LABELS),
            'preprocessor': DictConfig(preprocessor),
            'model_defaults': DictConfig({'enc_hidden': ENCODER_HIDDEN, 'pred_hidden': PREDICTOR_HIDDEN}),
            'encoder': DictConfig(encoder),
            'decoder': DictConfig(decoder),
            'joint': DictConfig(joint),
            'decoding': DictConfig({'strategy': 'greedy_batch', 'greedy': {'max_symbols': 5}}),
            'loss': DictConfig(loss),
            'rnnt_reduction': 'mean_volume',
        }
    )


@pytest.mark.unit
def test_consistency_loss_enables_double_batch():
    model = EncDecRNNTModel(cfg=_model_config())
    assert model.use_double_batch
    assert model.consistency_loss is not None
    assert model.consistency_loss_weight == 0.2
    assert model.uses_loss_logprobs_path


@pytest.mark.unit
def test_consistency_attributes_are_defined_without_consistency_config():
    config = _model_config()
    config.loss.consistency = None
    model = EncDecRNNTModel(cfg=config)
    assert model.consistency_loss is None
    assert not model.use_double_batch
    assert model.consistency_loss_weight == 0.0
    assert model.consistency_loss_after_step == -1


@pytest.mark.unit
def test_consistency_loss_rejects_fuse_loss_wer():
    with pytest.raises(ValueError, match="fuse_loss_wer"):
        EncDecRNNTModel(cfg=_model_config(fuse_loss_wer=True))


@pytest.mark.unit
@pytest.mark.parametrize("turbo_kernel_type", ["fused_vocab", "fused"])
def test_consistency_loss_rejects_turbo_kernel(turbo_kernel_type: str):
    with pytest.raises(NotImplementedError, match="Consistency loss"):
        EncDecRNNTModel(cfg=_model_config(turbo_kernel_type=turbo_kernel_type))


@pytest.mark.unit
def test_legacy_loss_does_not_use_logprobs_path():
    """`warprnnt_numba` requires full logits: the model must keep the legacy path (with a warning)."""
    model = EncDecRNNTModel(cfg=_model_config(loss_name="warprnnt_numba"))
    assert not model.uses_loss_logprobs_path
    assert model.consistency_loss is not None


@pytest.mark.unit
@pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton is not installed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("symmetrical", [True, False])
def test_model_compute_losses_with_consistency_matches_reference(symmetrical: bool):
    """The model-level fused path must match the reference (full logits + two separate losses)."""
    torch.manual_seed(17)
    device = torch.device("cuda")
    model = EncDecRNNTModel(cfg=_model_config(consistency_symmetrical=symmetrical)).to(device)
    model.train()

    half_batch_size, num_frames, num_text_units = 2, 6, 3
    encoded = torch.randn(2 * half_batch_size, ENCODER_HIDDEN, num_frames, device=device, requires_grad=True)
    predictor_output = torch.randn(
        2 * half_batch_size, PREDICTOR_HIDDEN, num_text_units + 1, device=device, requires_grad=True
    )
    transcript = torch.randint(0, len(LABELS), [half_batch_size, num_text_units], device=device).repeat(2, 1)
    encoded_len = torch.tensor([num_frames, 4], device=device).repeat(2)
    target_length = torch.tensor([num_text_units, 2], device=device).repeat(2)

    loss_value, consistency_loss_value = model._compute_losses(
        encoded=encoded,
        decoder=predictor_output,
        transcript=transcript,
        encoded_len=encoded_len,
        target_length=target_length,
        compute_consistency=True,
    )
    (loss_value + model.consistency_loss_weight * consistency_loss_value).backward()

    reference_encoded = encoded.detach().clone().requires_grad_(True)
    reference_predictor_output = predictor_output.detach().clone().requires_grad_(True)
    reference_logits = model.joint(encoder_outputs=reference_encoded, decoder_outputs=reference_predictor_output)
    reference_loss_value = model.loss(
        log_probs=reference_logits,
        targets=transcript,
        input_lengths=encoded_len,
        target_lengths=target_length,
    )
    reference_consistency_value = model.consistency_loss(
        teacher_logits=reference_logits[:half_batch_size],
        student_logits=reference_logits[half_batch_size:],
        src_lengths=encoded_len[:half_batch_size],
        tgt_lengths=target_length[:half_batch_size],
    )

    assert torch.allclose(loss_value, reference_loss_value, atol=1e-5, rtol=1e-5)
    assert torch.allclose(consistency_loss_value, reference_consistency_value, atol=1e-5, rtol=1e-5)

    reference_gradients = torch.autograd.grad(
        reference_loss_value + model.consistency_loss_weight * reference_consistency_value,
        [reference_encoded, reference_predictor_output],
    )
    assert torch.allclose(encoded.grad, reference_gradients[0], atol=1e-5, rtol=1e-5)
    assert torch.allclose(predictor_output.grad, reference_gradients[1], atol=1e-5, rtol=1e-5)


@pytest.mark.unit
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_model_compute_losses_legacy_logits_path_with_consistency():
    """The legacy path (loss requiring full logits) must still compute both losses."""
    torch.manual_seed(19)
    device = torch.device("cuda")
    model = EncDecRNNTModel(cfg=_model_config(loss_name="warprnnt_numba")).to(device)
    model.train()
    assert not model.uses_loss_logprobs_path

    half_batch_size, num_frames, num_text_units = 2, 6, 3
    encoded = torch.randn(2 * half_batch_size, ENCODER_HIDDEN, num_frames, device=device, requires_grad=True)
    predictor_output = torch.randn(
        2 * half_batch_size, PREDICTOR_HIDDEN, num_text_units + 1, device=device, requires_grad=True
    )
    transcript = torch.randint(0, len(LABELS), [half_batch_size, num_text_units], device=device).repeat(2, 1)
    encoded_len = torch.tensor([num_frames, 4], device=device).repeat(2)
    target_length = torch.tensor([num_text_units, 2], device=device).repeat(2)

    loss_value, consistency_loss_value = model._compute_losses(
        encoded=encoded,
        decoder=predictor_output,
        transcript=transcript,
        encoded_len=encoded_len,
        target_length=target_length,
        compute_consistency=True,
    )
    assert torch.isfinite(loss_value)
    assert torch.isfinite(consistency_loss_value)
    (loss_value + model.consistency_loss_weight * consistency_loss_value).backward()
    assert encoded.grad is not None and torch.isfinite(encoded.grad).all()
    assert predictor_output.grad is not None and torch.isfinite(predictor_output.grad).all()


@pytest.mark.unit
@pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton is not installed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_model_compute_losses_without_consistency():
    torch.manual_seed(18)
    device = torch.device("cuda")
    model = EncDecRNNTModel(cfg=_model_config()).to(device)
    model.train()

    encoded = torch.randn(2, ENCODER_HIDDEN, 5, device=device)
    predictor_output = torch.randn(2, PREDICTOR_HIDDEN, 4, device=device)
    transcript = torch.randint(0, len(LABELS), [2, 3], device=device)
    encoded_len = torch.tensor([5, 4], device=device)
    target_length = torch.tensor([3, 2], device=device)

    loss_value, consistency_loss_value = model._compute_losses(
        encoded=encoded,
        decoder=predictor_output,
        transcript=transcript,
        encoded_len=encoded_len,
        target_length=target_length,
    )
    assert consistency_loss_value is None
    assert torch.isfinite(loss_value)
