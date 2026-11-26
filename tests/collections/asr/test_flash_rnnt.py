# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
from hydra.errors import InstantiationException

from nemo.collections.asr.losses.rnnt import NUMBA_RNNT_AVAILABLE, RNNTLoss
from nemo.collections.asr.models import EncDecRNNTModel
from nemo.collections.asr.modules.hybrid_autoregressive_transducer import HATJoint
from nemo.collections.asr.parts.turbo_transducer.rnnt_logprobs import rnnt_logprobs
from nemo.core.utils.optional_libs import TRITON_AVAILABLE
from tests.collections.asr.turbo_transducer.test_rnnt_model_turbo_kernel import _model_config

pytestmark = pytest.mark.unit
requires_cuda_triton = pytest.mark.skipif(
    not (TRITON_AVAILABLE and torch.cuda.is_available()), reason="CUDA and Triton are required"
)

ENCODER_HIDDEN = 6
PRED_HIDDEN = 7
JOINT_HIDDEN = 8
NUM_LABELS = 7


def _make_model(
    activation="relu",
    dropout=0.0,
    loss_name="turbo_rnnt",
    reduction="mean_batch",
    loss_kwargs=None,
    max_joint_rows=200_000,
    **joint_options,
):
    kernel = {"type": "flash", "max_joint_rows": max_joint_rows} if loss_name == "turbo_rnnt" else None
    cfg = _model_config(EncDecRNNTModel, "", training_kernel=kernel, loss_name=loss_name, loss_kwargs=loss_kwargs)
    cfg.labels = cfg.labels[:NUM_LABELS]
    cfg.model_defaults.enc_hidden = cfg.encoder.jasper[0].filters = ENCODER_HIDDEN
    cfg.model_defaults.pred_hidden = cfg.decoder.prednet.pred_hidden = PRED_HIDDEN
    cfg.joint.jointnet.update(dict(joint_hidden=JOINT_HIDDEN, activation=activation, dropout=dropout))
    cfg.joint.update(joint_options)
    cfg.rnnt_reduction = reduction
    return EncDecRNNTModel(cfg=cfg)


def _batch(dtype=torch.float32, device="cuda"):
    return dict(
        encoded=torch.randn(4, ENCODER_HIDDEN, 7, device=device, dtype=dtype, requires_grad=True),
        decoder=torch.randn(4, PRED_HIDDEN, 5, device=device, dtype=dtype, requires_grad=True),
        encoded_len=torch.tensor([7, 6, 4, 3], device=device),
        transcript=torch.randint(NUM_LABELS, (4, 4), device=device),
        target_length=torch.tensor([4, 3, 2, 0], device=device),
    )


@pytest.mark.parametrize(
    "options,match",
    [
        ({"max_joint_rows": 0}, "max_joint_rows must be positive"),
        ({"pruning": {"type": "two_pass", "range_size": 2}}, "pruning"),
    ],
)
def test_flash_rnnt_rejects_invalid_configuration(options, match):
    with pytest.raises(InstantiationException, match=match):
        _make_model(training_kernel={"type": "flash", **options})


def test_flash_rnnt_requires_triton(monkeypatch):
    model = _make_model()
    monkeypatch.setattr("nemo.core.utils.optional_libs.TRITON_AVAILABLE", False)
    with pytest.raises(RuntimeError, match="Triton is required"):
        model._compute_losses(**_batch(device="cpu"))


@requires_cuda_triton
@pytest.mark.gpu
def test_flash_rnnt_reads_labels_from_a_strided_transcript():
    model = _make_model(max_joint_rows=13).cuda()
    batch = _batch()
    encoder = batch["encoded"].transpose(1, 2)
    predictor = batch["decoder"].transpose(1, 2)
    labels = batch["transcript"]
    lengths = batch["encoded_len"], batch["target_length"]
    expected = rnnt_logprobs(model.joint.joint(encoder, predictor), labels, NUM_LABELS, *lengths)
    storage = labels.new_empty(labels.size(0), 2 * labels.size(1))
    strided = storage[:, ::2]
    strided.copy_(labels)
    scores = model.joint.forward_for_loss(batch["encoded"], batch["decoder"], lengths[0], strided, lengths[1])
    torch.testing.assert_close(scores.target_logprobs, expected[0])
    torch.testing.assert_close(scores.blank_logprobs, expected[1])


@requires_cuda_triton
@pytest.mark.gpu
def test_rnnt_loss_forward_from_logprobs():
    # One frame and one token admit a single alignment: emit the token, then blank.
    target = torch.tensor([[[-2.0, 0.0]]], device="cuda", requires_grad=True)
    blank = torch.tensor([[[-0.25, -0.75]]], device="cuda", requires_grad=True)
    lengths = torch.ones(1, dtype=torch.long, device="cuda")
    loss = RNNTLoss(num_classes=NUM_LABELS, reduction="mean_batch", loss_name="turbo_rnnt")
    cost = loss.forward_from_logprobs(target, blank, lengths, lengths)
    torch.testing.assert_close(cost, target.new_tensor(2.75))
    target_grad, blank_grad = torch.autograd.grad(cost, (target, blank))
    torch.testing.assert_close(target_grad, target.new_tensor([[[-1.0, 0.0]]]))
    torch.testing.assert_close(blank_grad, blank.new_tensor([[[0.0, -1.0]]]))


@requires_cuda_triton
@pytest.mark.gpu
@pytest.mark.parametrize("key", ["encoded_len", "target_length"])
def test_flash_rnnt_rejects_lengths_past_the_states_they_index(key):
    model = _make_model().cuda()
    batch = _batch()
    batch[key][0] += 1
    with pytest.raises(ValueError, match="lengths reach beyond the states they index"):
        model._compute_losses(**batch)


@pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton is required")
@pytest.mark.parametrize(
    "layer,error", [(torch.nn.Dropout(1.0), "joint dropout"), (torch.nn.Identity(), "optional Dropout")]
)
def test_flash_rnnt_rejects_invalid_joint_layer(layer, error):
    model = _make_model(dropout=0.1)
    model.joint.joint_net[1] = layer
    with pytest.raises(ValueError, match=error):
        model._compute_losses(**_batch(device="cpu"))


@pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton is required")
def test_flash_rnnt_rejects_hat_joint_without_reading_blank_out_of_bounds():
    joint = HATJoint(
        jointnet=dict(
            encoder_hidden=ENCODER_HIDDEN, pred_hidden=PRED_HIDDEN, joint_hidden=JOINT_HIDDEN, activation="relu"
        ),
        num_classes=NUM_LABELS,
        log_softmax=False,
    )
    batch = _batch(device="cpu")
    with pytest.raises(NotImplementedError, match="does not support"):
        joint.forward_for_loss(
            batch["encoded"], batch["decoder"], batch["encoded_len"], batch["transcript"], batch["target_length"]
        )


@requires_cuda_triton
@pytest.mark.gpu
@pytest.mark.skipif(not NUMBA_RNNT_AVAILABLE, reason="Numba RNN-T is required")
@pytest.mark.parametrize("activation", ["relu", "sigmoid", "tanh"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_flash_rnnt_matches_numba_loss_and_gradients(dtype, activation):
    torch.manual_seed(19)
    batch = _batch(dtype)
    flash = _make_model(activation=activation, max_joint_rows=13, loss_kwargs={"fastemit_lambda": 0.01}).to(
        device="cuda", dtype=dtype
    )
    dense = _make_model(
        activation=activation,
        loss_name="warprnnt_numba",
        loss_kwargs={"fastemit_lambda": 0.01},
    ).cuda()
    dense.joint.load_state_dict(flash.joint.state_dict())

    inputs = batch["encoded"], batch["decoder"]
    flash_loss = flash._compute_losses(**batch)[0]
    # Numba's reference runs in FP32 with the same (possibly BF16-rounded) weights and inputs.
    dense_loss = dense._compute_losses(**(batch | {"encoded": inputs[0].float(), "decoder": inputs[1].float()}))[0]
    flash_grads = torch.autograd.grad(flash_loss, (*inputs, *flash.joint.parameters()))
    dense_grads = torch.autograd.grad(dense_loss, (*inputs, *dense.joint.parameters()))

    atol, rtol = (2e-5, 2e-4) if dtype == torch.float32 else (2e-2, 2e-2)
    torch.testing.assert_close(flash_loss, dense_loss, atol=atol, rtol=rtol)
    for actual, expected in zip(flash_grads, dense_grads):
        torch.testing.assert_close(actual.float(), expected.float(), atol=atol, rtol=rtol)


@requires_cuda_triton
@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("max_joint_rows", [10_000, 13])
def test_flash_rnnt_tile_budget_equivalence(dtype, max_joint_rows):
    torch.manual_seed(23)
    batch = _batch(dtype)
    full = _make_model().to(device="cuda", dtype=dtype)
    tiled = _make_model(max_joint_rows=max_joint_rows).to(device="cuda", dtype=dtype)
    tiled.joint.load_state_dict(full.joint.state_dict())
    inputs = batch["encoded"], batch["decoder"]
    single_loss = full._compute_losses(**batch)[0]
    single_grads = torch.autograd.grad(single_loss, (*inputs, *full.joint.parameters()))
    tiled_loss = tiled._compute_losses(**batch)[0]
    tiled_grads = torch.autograd.grad(tiled_loss, (*inputs, *tiled.joint.parameters()))

    atol, rtol = (2e-5, 2e-4) if dtype == torch.float32 else (2e-2, 2e-2)
    torch.testing.assert_close(tiled_loss, single_loss, atol=atol, rtol=rtol)
    for actual, expected in zip(tiled_grads, single_grads):
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


@requires_cuda_triton
@pytest.mark.gpu
@pytest.mark.skipif(not NUMBA_RNNT_AVAILABLE, reason="Numba RNN-T is required")
@pytest.mark.parametrize("weighted,scale", [(False, 1.0), (False, 1024.0), (True, 1.0), (True, 1024.0), (True, 1e-6)])
def test_flash_rnnt_clamp_matches_numba_across_tiles(weighted, scale):
    torch.manual_seed(23)
    batch = _batch()
    batch["target_length"][-1] = 1
    options = dict(clamp=0.02, fastemit_lambda=0.01)
    reduction = None if weighted else "mean_batch"
    flash = _make_model(max_joint_rows=4, reduction=reduction, loss_kwargs=options.copy()).cuda()
    dense = _make_model(loss_name="warprnnt_numba", reduction=reduction, loss_kwargs=options).cuda()
    dense.joint.load_state_dict(flash.joint.state_dict())
    inputs = batch["encoded"], batch["decoder"]
    flash_loss = flash._compute_losses(**batch)[0]
    dense_loss = dense._compute_losses(**batch)[0]
    weights = flash_loss.new_tensor([0.5, 0.0, -2.0, 1.0]) if weighted else torch.ones_like(flash_loss)
    flash_grads = torch.autograd.grad(flash_loss, (*inputs, *flash.joint.parameters()), weights * scale)
    dense_grads = torch.autograd.grad(dense_loss, (*inputs, *dense.joint.parameters()), weights * scale)

    torch.testing.assert_close(flash_loss, dense_loss, atol=1e-5, rtol=1e-5)
    for actual, expected in zip(flash_grads, dense_grads):
        torch.testing.assert_close(actual / scale, expected / scale, atol=2e-5, rtol=2e-4)


@requires_cuda_triton
@pytest.mark.gpu
def test_flash_rnnt_joint_dropout_backward_matches_forward():
    torch.manual_seed(137)
    model = _make_model(activation="tanh", dropout=0.25, max_joint_rows=13).cuda()
    batch = _batch()
    encoder = batch["encoded"]
    direction = torch.randn_like(encoder)

    def loss_at(offset):
        torch.cuda.manual_seed(17)  # Use the same forward dropout draw at all three offsets.
        return model._compute_losses(**(batch | {"encoded": encoder + offset * direction}))[0]

    (gradient,) = torch.autograd.grad(loss_at(0), encoder)
    step = 0.01
    slope = (loss_at(step) - loss_at(-step)) / (2 * step)
    torch.testing.assert_close((gradient * direction).sum(), slope, atol=3e-4, rtol=0.01)


@requires_cuda_triton
@pytest.mark.gpu
@pytest.mark.skipif(not NUMBA_RNNT_AVAILABLE, reason="Numba RNN-T is required")
def test_flash_rnnt_prediction_masking_matches_numba(monkeypatch):
    torch.manual_seed(31)
    flash = _make_model(activation="tanh", masking_prob=0.5, max_joint_rows=13).cuda()
    dense = _make_model(activation="tanh", masking_prob=0.5, loss_name="warprnnt_numba").cuda()
    dense.joint.load_state_dict(flash.joint.state_dict())
    batch = _batch()
    encoder, predictor = batch["encoded"], batch["decoder"]
    draws = torch.rand(predictor.shape[0], predictor.shape[2])

    # Flash draws [B, U+1] on CUDA; dense draws [B, 1, U+1, 1] on CPU.
    with monkeypatch.context() as rng:
        rng.setattr(torch, "rand", lambda shape, *, device=None: draws.reshape(shape).to(device=device))
        flash_loss = flash._compute_losses(**batch)[0]
        dense_loss = dense._compute_losses(**batch)[0]

    flash_grads = torch.autograd.grad(flash_loss, (encoder, predictor, *flash.joint.parameters()))
    dense_grads = torch.autograd.grad(dense_loss, (encoder, predictor, *dense.joint.parameters()))
    torch.testing.assert_close(flash_loss, dense_loss, atol=2e-5, rtol=2e-4)
    for actual, expected in zip(flash_grads, dense_grads, strict=True):
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)

    valid = torch.arange(predictor.size(2), device=predictor.device) <= batch["target_length"][:, None]
    kept = (draws.to(predictor.device) > 0.5)[valid]
    assert kept.any() and not kept.all()
    torch.testing.assert_close(flash_grads[1].any(dim=1)[valid], kept)
    flash.eval()
    (eval_grad,) = torch.autograd.grad(flash._compute_losses(**batch)[0], predictor)
    assert eval_grad.any(dim=1)[valid].all()


@requires_cuda_triton
@pytest.mark.gpu
def test_flash_rnnt_vocabulary_projection_respects_tile_budget():
    model = _make_model(max_joint_rows=13).cuda()
    batch = _batch()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True) as profile:
        model._compute_losses(**batch)[0].backward()
    projections = [
        event.input_shapes[0]
        for event in profile.events()
        if event.name == "aten::linear" and event.input_shapes[1] == [NUM_LABELS + 1, JOINT_HIDDEN]
    ]
    assert projections
    assert all(len(shape) == 2 and shape[0] <= 13 for shape in projections)
