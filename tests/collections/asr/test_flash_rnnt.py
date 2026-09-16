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

from nemo.collections.asr.losses.flash_rnnt import FlashRNNTLoss
from nemo.collections.asr.losses.rnnt import NUMBA_RNNT_AVAILABLE, RNNTLoss
from nemo.collections.asr.modules.hybrid_autoregressive_transducer import HATJoint
from nemo.collections.asr.modules.rnnt import RNNTJoint
from nemo.collections.asr.parts.triton.rnnt_joint import _PackedJoint, lattice_layout, packed_scatter
from nemo.core.utils.optional_libs import TRITON_AVAILABLE

CUDA_TRITON_AVAILABLE = TRITON_AVAILABLE and torch.cuda.is_available()

# The joint geometry the tests share. The three hidden sizes differ so a transposed or swapped axis
# surfaces as a shape error instead of running silently, and the vocabulary follows the standard
# RNN-T layout the flash path requires: every label first, blank in the final column.
ENCODER_HIDDEN = 6
PRED_HIDDEN = 7
JOINT_HIDDEN = 8
NUM_LABELS = 7
BLANK = NUM_LABELS
VOCAB = NUM_LABELS + 1


def _joint_hidden_state(encoder, predictor, activation, dropout_p=0.0):
    """Dense ``[B, T, U + 1, H]`` joint state from the packed kernels.

    Giving every sample the full source and transcript extent makes the packed row order the same
    as a dense reshape, so the kernels can be compared elementwise against an eager broadcast add.
    """
    batch, source_steps, hidden_size = encoder.shape
    target_states = predictor.shape[1]
    source_lengths = torch.full((batch,), source_steps, device=encoder.device, dtype=torch.int32)
    target_lengths = torch.full((batch,), target_states - 1, device=encoder.device, dtype=torch.int32)
    offsets, states, total_rows = lattice_layout(source_lengths, target_lengths, source_steps, target_states)
    seed = source_lengths
    if dropout_p > 0.0:
        seed = torch.randint(0, 2**31 - 1, (1,), device=encoder.device, dtype=torch.int32)
    predictor_mask = torch.ones(batch, target_states, device=encoder.device, dtype=torch.bool)
    hidden = _PackedJoint.apply(
        encoder, predictor, offsets, states, source_lengths, seed, predictor_mask, 0, total_rows, activation, dropout_p
    )
    return hidden.view(batch, source_steps, target_states, hidden_size)


def _scatter_row_numbers(source_lengths, target_lengths, frames, target_states):
    """Scatter each row's own number into the planes, so the layout can be read back off them."""
    offsets, states, total_rows = lattice_layout(source_lengths, target_lengths, frames, target_states)
    numbers = torch.arange(total_rows, device="cuda", dtype=torch.float32) + 1.0
    planes = [torch.zeros(len(source_lengths) * frames * target_states, device="cuda") for _ in range(2)]
    shape = (len(source_lengths), frames, target_states)
    planes = packed_scatter(numbers, numbers, planes[0], planes[1], offsets, states, 0, shape)
    return planes[0], total_rows


@pytest.mark.unit
@pytest.mark.skipif(not CUDA_TRITON_AVAILABLE, reason="CUDA and Triton are required")
def test_packed_rows_run_frame_major_within_a_sample():
    """Rows advance through transcript states first, then frames, with samples laid end to end.

    The scatter derives each address from the row rather than from a stored index, so this is what
    pins that addressing down.
    """
    source_lengths = torch.tensor([3, 1, 2], device="cuda")
    target_lengths = torch.tensor([2, 0, 1], device="cuda")
    plane, total_rows = _scatter_row_numbers(source_lengths, target_lengths, 3, 3)

    # sample 0 fills its 3x3 plane, sample 1 contributes one row, sample 2 a 2x2 corner
    occupied = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 18, 19, 21, 22], device="cuda")
    assert total_rows == occupied.numel()
    torch.testing.assert_close(plane[occupied], torch.arange(total_rows, device="cuda", dtype=torch.float32) + 1.0)
    empty = torch.ones_like(plane, dtype=torch.bool)
    empty[occupied] = False
    assert torch.equal(plane[empty], torch.zeros_like(plane[empty]))


@pytest.mark.unit
@pytest.mark.skipif(not CUDA_TRITON_AVAILABLE, reason="CUDA and Triton are required")
def test_packed_rows_of_a_full_lattice_are_a_dense_reshape():
    """With every sample at full extent nothing is padded, so packed order is dense order."""
    batch, frames, target_states = 4, 5, 3
    source_lengths = torch.full((batch,), frames, device="cuda")
    target_lengths = torch.full((batch,), target_states - 1, device="cuda")
    plane, total_rows = _scatter_row_numbers(source_lengths, target_lengths, frames, target_states)

    torch.testing.assert_close(plane, torch.arange(total_rows, device="cuda", dtype=torch.float32) + 1.0)


@pytest.mark.unit
@pytest.mark.skipif(not CUDA_TRITON_AVAILABLE, reason="CUDA and Triton are required")
def test_flash_rnnt_reads_labels_from_a_strided_transcript():
    """The extraction addresses a label by row stride alone, so the columns must be adjacent.

    A caller is free to pass a view whose columns are not, and the wrong labels would be scored
    without any shape error, so the loss has to make the layout it assumes.
    """
    torch.manual_seed(29)
    encoder, predictor, source_lengths, target_lengths, labels = _end_to_end_batch()
    batch, target_tokens = labels.shape
    # every other column of a wider tensor: same labels, stride(1) == 2
    strided = torch.zeros(batch, 2 * target_tokens, device="cuda", dtype=labels.dtype)
    strided[:, ::2] = labels
    strided = strided[:, ::2]
    assert not strided.is_contiguous() and torch.equal(strided, labels)

    values = []
    for transcripts in (labels, strided):
        joint = _make_joint(4)
        torch.manual_seed(29)
        for parameter in joint.parameters():
            torch.nn.init.normal_(parameter, std=0.2)
        joint.set_loss(RNNTLoss(num_classes=NUM_LABELS, reduction="mean_batch", loss_name="flash_rnnt"))
        joint.set_wer(object())
        values.append(
            joint(
                encoder_outputs=encoder,
                decoder_outputs=predictor,
                encoder_lengths=source_lengths,
                transcripts=transcripts,
                transcript_lengths=target_lengths,
            )[0]
        )
    torch.testing.assert_close(values[1], values[0], atol=0.0, rtol=0.0)


@pytest.mark.unit
@pytest.mark.skipif(not CUDA_TRITON_AVAILABLE, reason="CUDA and Triton are required")
@pytest.mark.parametrize("axis", ["source", "target"])
def test_flash_rnnt_rejects_lengths_past_the_states_they_index(axis):
    """A length beyond its tensor would be read, not masked, so it has to be refused up front."""
    joint = _make_joint(4)
    joint.set_loss(RNNTLoss(num_classes=NUM_LABELS, reduction="mean_batch", loss_name="flash_rnnt"))
    joint.set_wer(object())
    encoder, predictor, source_lengths, target_lengths, labels = _end_to_end_batch()
    if axis == "source":
        source_lengths = source_lengths + encoder.shape[2]
    else:
        target_lengths = target_lengths + predictor.shape[2]

    with pytest.raises(ValueError, match="lengths reach beyond the states they index"):
        joint(
            encoder_outputs=encoder,
            decoder_outputs=predictor,
            encoder_lengths=source_lengths,
            transcripts=labels,
            transcript_lengths=target_lengths,
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"max_joint_rows": 0}, "max_joint_rows must be positive"),
        ({"fastemit_lambda": -0.01}, "fastemit_lambda must be nonnegative"),
    ],
)
def test_flash_rnnt_rejects_invalid_configuration(kwargs, match):
    with pytest.raises(ValueError, match=match):
        FlashRNNTLoss(blank=BLANK, **kwargs)


@pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton is required")
@pytest.mark.unit
def test_flash_rnnt_rejects_dense_path():
    batch, source_steps, target_tokens = 1, 1, 0
    loss = RNNTLoss(num_classes=NUM_LABELS, loss_name="flash_rnnt")

    with pytest.raises(RuntimeError, match="fuse_loss_wer=true"):
        loss(
            log_probs=torch.empty(batch, source_steps, target_tokens + 1, VOCAB),
            targets=torch.empty(batch, target_tokens, dtype=torch.long),
            input_lengths=torch.full((batch,), source_steps, dtype=torch.long),
            target_lengths=torch.full((batch,), target_tokens, dtype=torch.long),
        )


@pytest.mark.unit
@pytest.mark.skipif(not CUDA_TRITON_AVAILABLE, reason="CUDA and Triton are required")
def test_flash_rnnt_rejects_hat_joint_without_reading_blank_out_of_bounds():
    """A HAT joint scores blank in a separate head, so its joint_net is one column short.

    The flash path reads the blank column out of the joint's own weights, which on this joint would
    run past the end of every row, so configuring the two together has to be refused rather than
    silently read whatever follows.
    """
    batch, source_steps, target_tokens = 2, 4, 2
    joint = HATJoint(
        jointnet={
            "encoder_hidden": ENCODER_HIDDEN,
            "pred_hidden": PRED_HIDDEN,
            "joint_hidden": JOINT_HIDDEN,
            "activation": "relu",
        },
        num_classes=NUM_LABELS,
        log_softmax=False,
        fuse_loss_wer=True,
        fused_batch_size=1,
    ).cuda()
    assert joint.joint_net[-1].out_features == NUM_LABELS
    assert joint.num_classes_with_blank == VOCAB
    joint.set_loss(RNNTLoss(num_classes=NUM_LABELS, reduction="mean_batch", loss_name="flash_rnnt"))
    joint.set_wer(object())

    with pytest.raises(ValueError, match="include every label and the blank"):
        joint(
            encoder_outputs=torch.randn(batch, ENCODER_HIDDEN, source_steps, device="cuda"),
            decoder_outputs=torch.randn(batch, PRED_HIDDEN, target_tokens + 1, device="cuda"),
            encoder_lengths=torch.tensor([source_steps, 3], device="cuda"),
            transcripts=torch.randint(0, NUM_LABELS, (batch, target_tokens), device="cuda"),
            transcript_lengths=torch.tensor([target_tokens, 1], device="cuda"),
        )


@pytest.mark.unit
def test_flash_rnnt_early_return_preserves_requested_empty_hypotheses():
    class FlashLoss:
        requires_factorized_joint = True

    batch, source_steps = 2, 3
    joint = RNNTJoint(
        jointnet={
            "encoder_hidden": ENCODER_HIDDEN,
            "pred_hidden": PRED_HIDDEN,
            "joint_hidden": JOINT_HIDDEN,
            "activation": "relu",
        },
        num_classes=NUM_LABELS,
        log_softmax=False,
        fuse_loss_wer=True,
        fused_batch_size=1,
    )
    joint.set_loss(FlashLoss())
    joint.set_wer(object())
    result = joint(
        encoder_outputs=torch.zeros(batch, ENCODER_HIDDEN, source_steps),
        decoder_outputs=None,
        encoder_lengths=torch.tensor([source_steps, source_steps - 1]),
        # Every transcript is empty, which is the early return under test.
        transcript_lengths=torch.zeros(batch, dtype=torch.long),
        keep_hypotheses=True,
    )

    assert result == (None, None, None, None)
    assert joint.get_hypotheses() == []


@pytest.mark.unit
@pytest.mark.skipif(not CUDA_TRITON_AVAILABLE, reason="CUDA and Triton are required")
@pytest.mark.parametrize("activation", ["relu", "sigmoid", "tanh"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_flash_rnnt_join_matches_the_eager_broadcast_add(activation, dtype):
    """The fused kernel must compute the same function as an eager broadcast-add plus activation.

    It keeps the pre-activation in FP32 where eager rounds it to the input dtype, so in BF16 the
    two agree only to about one unit in the last place -- and for ReLU a rounded pre-activation can
    cross zero, flipping the derivative mask outright. Both effects are inherent to the precision
    rather than to the kernel, so compare at matched dtype with a precision-appropriate tolerance.
    """
    torch.manual_seed(7)
    # An odd hidden size leaves a partial block, which is where a mishandled tail would show.
    batch, source_steps, target_states, hidden_size = 3, 7, 5, 129
    encoder = torch.randn(batch, source_steps, hidden_size, device="cuda", dtype=dtype)
    predictor = torch.randn(batch, target_states, hidden_size, device="cuda", dtype=dtype)
    upstream = torch.randn(batch, source_steps, target_states, hidden_size, device="cuda", dtype=dtype)

    def forward_and_gradients(join):
        leaf_encoder = encoder.detach().clone().requires_grad_(True)
        leaf_predictor = predictor.detach().clone().requires_grad_(True)
        hidden = join(leaf_encoder, leaf_predictor)
        return hidden, torch.autograd.grad(hidden, (leaf_encoder, leaf_predictor), upstream)

    def eager(leaf_encoder, leaf_predictor):
        activate = {"relu": torch.relu, "sigmoid": torch.sigmoid, "tanh": torch.tanh}[activation]
        return activate(leaf_encoder.unsqueeze(2) + leaf_predictor.unsqueeze(1))

    fused_hidden, fused_gradients = forward_and_gradients(lambda e, p: _joint_hidden_state(e, p, activation))
    eager_hidden, eager_gradients = forward_and_gradients(eager)

    atol, rtol = (1e-6, 1e-5) if dtype == torch.float32 else (3e-2, 3e-2)
    torch.testing.assert_close(fused_hidden, eager_hidden, atol=atol, rtol=rtol)
    for actual, expected in zip(fused_gradients, eager_gradients):
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


@pytest.mark.unit
def test_flash_rnnt_requires_triton(monkeypatch):
    monkeypatch.setattr("nemo.collections.asr.losses.flash_rnnt.TRITON_AVAILABLE", False)
    batch, source_steps, target_tokens, hidden_size = 1, 2, 1, 3
    with pytest.raises(RuntimeError, match="Triton is required"):
        FlashRNNTLoss(blank=BLANK)(
            joint=object(),
            encoder=torch.zeros(batch, source_steps, hidden_size),
            predictor=torch.zeros(batch, target_tokens + 1, hidden_size),
            targets=torch.zeros(batch, target_tokens, dtype=torch.long),
            source_lengths=torch.full((batch,), source_steps, dtype=torch.long),
            target_lengths=torch.full((batch,), target_tokens, dtype=torch.long),
        )


def _end_to_end_batch(dtype=torch.float32):
    """One ragged batch for the end-to-end joint tests, including an empty transcript."""
    batch, source_steps, target_tokens = 4, 7, 4
    encoder = torch.randn(batch, ENCODER_HIDDEN, source_steps, device="cuda", dtype=dtype, requires_grad=True)
    predictor = torch.randn(batch, PRED_HIDDEN, target_tokens + 1, device="cuda", dtype=dtype, requires_grad=True)
    source_lengths = torch.tensor([source_steps, 6, 4, 3], device="cuda")
    target_lengths = torch.tensor([target_tokens, 3, 2, 0], device="cuda")
    labels = torch.randint(0, NUM_LABELS, (batch, target_tokens), device="cuda")
    return encoder, predictor, source_lengths, target_lengths, labels


def _make_joint(fused_batch_size, activation="relu", log_softmax=False, dropout=0.0, masking_prob=-1.0):
    return RNNTJoint(
        jointnet={
            "encoder_hidden": ENCODER_HIDDEN,
            "pred_hidden": PRED_HIDDEN,
            "joint_hidden": JOINT_HIDDEN,
            "activation": activation,
            "dropout": dropout,
        },
        num_classes=NUM_LABELS,
        log_softmax=log_softmax,
        fuse_loss_wer=True,
        fused_batch_size=fused_batch_size,
        masking_prob=masking_prob,
    ).cuda()


@pytest.mark.unit
@pytest.mark.skipif(
    not CUDA_TRITON_AVAILABLE or not NUMBA_RNNT_AVAILABLE,
    reason="CUDA, Triton, and Numba RNN-T are required",
)
@pytest.mark.parametrize(
    ("objective_reduction", "amp_scale"), [("mean_batch", 1.0), ("mean_batch", 1024.0), ("weighted", 1.0)]
)
def test_flash_rnnt_clamp_matches_numba_across_tiles(objective_reduction, amp_scale):
    """Clamping applies to the unit-scale gradient, which autograd has already scaled.

    Dividing that scale back out has to survive tiling, the loss reduction and the AMP scale at
    once, so compare against warprnnt_numba, which clamps the same unit-scale gradient, rather than
    against another Triton path sharing the same plumbing.
    """
    from nemo.collections.asr.parts.numba.rnnt_loss import RNNTLossNumba

    torch.manual_seed(23)
    joint = _make_joint(4)
    for parameter in joint.parameters():
        torch.nn.init.normal_(parameter, std=0.3)
    batch, source_steps, target_tokens = 6, 9, 5
    max_joint_rows = 4
    flash_loss = RNNTLoss(
        num_classes=NUM_LABELS,
        reduction=None,
        loss_name="flash_rnnt",
        loss_kwargs={"fastemit_lambda": 0.01, "clamp": 0.02, "max_joint_rows": max_joint_rows},
    )
    joint.set_loss(flash_loss)
    joint.set_wer(object())

    encoder = torch.randn(batch, ENCODER_HIDDEN, source_steps, device="cuda")
    predictor = torch.randn(batch, PRED_HIDDEN, target_tokens + 1, device="cuda")
    source_lengths = torch.tensor([source_steps, 8, 6, 5, 3, 2], device="cuda")
    target_lengths = torch.tensor([target_tokens, 4, 3, 2, 1, 1], device="cuda")
    labels = torch.randint(0, NUM_LABELS, (batch, target_tokens), device="cuda")

    def objective(per_sample):
        if objective_reduction == "mean_batch":
            return per_sample.mean() * amp_scale
        weights = torch.tensor([0.5, 0.0, -2.0, 1.0, 0.25, -0.75], device="cuda")
        return (per_sample * weights).sum() * amp_scale

    flash_encoder = encoder.clone().requires_grad_(True)
    flash_predictor = predictor.clone().requires_grad_(True)
    flash_costs = joint(
        encoder_outputs=flash_encoder,
        decoder_outputs=flash_predictor,
        encoder_lengths=source_lengths,
        transcripts=labels,
        transcript_lengths=target_lengths,
    )[0]
    flash_gradients = torch.autograd.grad(
        objective(flash_costs), (flash_encoder, flash_predictor, *joint.parameters())
    )

    dense_encoder = encoder.clone().requires_grad_(True)
    dense_predictor = predictor.clone().requires_grad_(True)
    logits = joint.joint(dense_encoder.transpose(1, 2), dense_predictor.transpose(1, 2))
    dense_costs = RNNTLossNumba(blank=BLANK, reduction="none", fastemit_lambda=0.01, clamp=0.02)(
        logits, labels, source_lengths, target_lengths
    )
    dense_gradients = torch.autograd.grad(
        objective(dense_costs), (dense_encoder, dense_predictor, *joint.parameters())
    )

    torch.testing.assert_close(flash_costs, dense_costs, atol=1e-5, rtol=1e-5)
    for actual, expected in zip(flash_gradients, dense_gradients):
        torch.testing.assert_close(actual / amp_scale, expected / amp_scale, atol=2e-5, rtol=2e-4)


@pytest.mark.unit
@pytest.mark.skipif(not CUDA_TRITON_AVAILABLE, reason="CUDA and Triton are required")
@pytest.mark.parametrize("hidden_size", [6, 8, 1536])
def test_joint_dropout_mask_agrees_across_kernels(hidden_size):
    """The forward and both backward kernels must reach the same verdict on every element.

    Nothing stores the mask and the three kernels walk the joint grid along different axes, so each
    redraws it from the element's position. Under relu with a positive pre-activation the activation
    derivative is exactly 1, so a backward reduces to the mask weighted by its grad_output along the
    axis it collapses. A grad_output that is one at a single index and zero elsewhere therefore
    leaves exactly that slice of the backward's mask, which compares against the forward's directly
    rather than through a sum that a compensating disagreement could still satisfy.

    ``hidden_size`` covers a size that is not a multiple of four, one block, and several blocks.
    """
    torch.manual_seed(53)
    dropout_p = 0.25
    batch, source_steps, target_states = 2, 5, 4
    # Ones against zeros makes every pre-activation exactly one, so relu's derivative is exactly one
    # and each gradient below is the mask itself, scaled.
    encoder = torch.ones(batch, source_steps, hidden_size, device="cuda", requires_grad=True)
    predictor = torch.zeros(batch, target_states, hidden_size, device="cuda", requires_grad=True)

    hidden = _joint_hidden_state(encoder, predictor, "relu", dropout_p)
    forward_mask = hidden != 0.0

    for target_index in range(target_states):
        encoder.grad = None
        probe = torch.zeros_like(hidden)
        probe[:, :, target_index] = 1.0
        hidden.backward(probe, retain_graph=True)
        torch.testing.assert_close(encoder.grad != 0.0, forward_mask[:, :, target_index])

    for source_index in range(source_steps):
        predictor.grad = None
        probe = torch.zeros_like(hidden)
        probe[:, source_index] = 1.0
        hidden.backward(probe, retain_graph=True)
        torch.testing.assert_close(predictor.grad != 0.0, forward_mask[:, source_index])

    survivors = hidden[hidden != 0.0]
    torch.testing.assert_close(survivors, torch.full_like(survivors, 1.0 / (1.0 - dropout_p)))
    assert 0.6 < survivors.numel() / hidden.numel() < 0.9


@pytest.mark.unit
@pytest.mark.skipif(not CUDA_TRITON_AVAILABLE, reason="CUDA and Triton are required")
def test_flash_rnnt_rejects_invalid_joint_dropout():
    joint = _make_joint(4, dropout=1.0)
    joint.set_loss(RNNTLoss(num_classes=NUM_LABELS, reduction="mean_batch", loss_name="flash_rnnt"))
    joint.set_wer(object())
    encoder, predictor, source_lengths, target_lengths, labels = _end_to_end_batch()
    with pytest.raises(ValueError, match=r"joint dropout must be in \[0, 1\)"):
        joint(
            encoder_outputs=encoder,
            decoder_outputs=predictor,
            encoder_lengths=source_lengths,
            transcripts=labels,
            transcript_lengths=target_lengths,
        )


@pytest.mark.unit
@pytest.mark.skipif(not CUDA_TRITON_AVAILABLE, reason="CUDA and Triton are required")
@pytest.mark.parametrize("max_joint_rows", [10_000, 13])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_flash_rnnt_tile_budget_equivalence(dtype, max_joint_rows):
    """The tile budget bounds the workspace without changing the answer.

    The small budget is coprime with every sample's lattice, so tiles start and end inside a sample
    and each backward has to bound itself to the sample it landed in rather than to a whole one.
    """
    torch.manual_seed(23)
    reference_joint = _make_joint(4).to(dtype)
    packed_joint = _make_joint(4).to(dtype)
    packed_joint.load_state_dict(reference_joint.state_dict())
    reference_joint.set_loss(RNNTLoss(num_classes=NUM_LABELS, reduction="mean_batch", loss_name="flash_rnnt"))
    packed_joint.set_loss(
        RNNTLoss(
            num_classes=NUM_LABELS,
            reduction="mean_batch",
            loss_name="flash_rnnt",
            loss_kwargs={"max_joint_rows": max_joint_rows},
        )
    )
    reference_joint.set_wer(object())
    packed_joint.set_wer(object())

    encoder, predictor, source_lengths, target_lengths, labels = _end_to_end_batch(dtype)
    packed_encoder = encoder.detach().clone().requires_grad_(True)
    packed_predictor = predictor.detach().clone().requires_grad_(True)

    reference_value = reference_joint(
        encoder_outputs=encoder,
        decoder_outputs=predictor,
        encoder_lengths=source_lengths,
        transcripts=labels,
        transcript_lengths=target_lengths,
    )[0]
    packed_value = packed_joint(
        encoder_outputs=packed_encoder,
        decoder_outputs=packed_predictor,
        encoder_lengths=source_lengths,
        transcripts=labels,
        transcript_lengths=target_lengths,
    )[0]
    reference_gradients = torch.autograd.grad(reference_value, (encoder, predictor, *reference_joint.parameters()))
    packed_gradients = torch.autograd.grad(
        packed_value, (packed_encoder, packed_predictor, *packed_joint.parameters())
    )

    atol, rtol = (2e-5, 2e-4) if dtype == torch.float32 else (2e-2, 2e-2)
    torch.testing.assert_close(packed_value, reference_value, atol=atol, rtol=rtol)
    for actual, expected in zip(packed_gradients, reference_gradients):
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


@pytest.mark.unit
@pytest.mark.skipif(not CUDA_TRITON_AVAILABLE, reason="CUDA and Triton are required")
def test_flash_rnnt_joint_dropout_recomputes_the_same_mask(monkeypatch):
    """A tile draws its own seed, so its recompute has to land on the mask the forward used."""
    from nemo.collections.asr.parts.triton import rnnt_joint

    torch.manual_seed(137)
    joint = _make_joint(4, activation="tanh", dropout=0.25)
    joint.set_loss(
        RNNTLoss(
            num_classes=NUM_LABELS,
            reduction="mean_batch",
            loss_name="flash_rnnt",
            loss_kwargs={"max_joint_rows": 13},
        )
    )
    joint.set_wer(object())

    produced = []
    original = rnnt_joint._PackedJoint.apply

    def recording(*args):
        hidden = original(*args)
        produced.append(hidden.detach().clone())
        return hidden

    monkeypatch.setattr(rnnt_joint._PackedJoint, "apply", staticmethod(recording))
    encoder, predictor, source_lengths, target_lengths, labels = _end_to_end_batch()
    value = joint(
        encoder_outputs=encoder,
        decoder_outputs=predictor,
        encoder_lengths=source_lengths,
        transcripts=labels,
        transcript_lengths=target_lengths,
    )[0]
    forward_tiles = len(produced)
    assert forward_tiles > 1, "the budget must split the lattice so recomputation is exercised"
    value.backward()

    recomputed = produced[forward_tiles:]
    assert len(recomputed) == forward_tiles, "backward must recompute every checkpointed tile once"
    # Backward walks the tiles in its own order, so pair each recomputed tile with the forward tile
    # it reproduces rather than assuming the sequence lines up.
    for tile in recomputed:
        assert any(
            tile.shape == original_tile.shape and torch.equal(tile, original_tile)
            for original_tile in produced[:forward_tiles]
        ), "a recomputed tile does not match any tile the forward produced"


@pytest.mark.unit
@pytest.mark.skipif(
    not CUDA_TRITON_AVAILABLE or not NUMBA_RNNT_AVAILABLE,
    reason="CUDA, Triton, and Numba RNN-T are required",
)
@pytest.mark.parametrize("activation", ["relu", "sigmoid", "tanh"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_flash_rnnt_matches_numba_loss_and_gradients(dtype, activation):
    """The whole path -- joint, extraction, dynamic program -- against an independent backend.

    warprnnt_numba materializes the joint tensor and shares no kernel with this loss, so a defect
    anywhere in the chain shows up here. It reads float32, which also makes the reference the more
    accurate of the two in the reduced-precision arm.
    """
    from nemo.collections.asr.parts.numba.rnnt_loss import RNNTLossNumba

    torch.manual_seed(19)
    dense_joint = _make_joint(4, activation=activation).to(dtype)
    flash_joint = _make_joint(4, activation=activation).to(dtype)
    flash_joint.load_state_dict(dense_joint.state_dict())
    flash_loss = RNNTLoss(
        num_classes=NUM_LABELS,
        reduction="mean_batch",
        loss_name="flash_rnnt",
        loss_kwargs={"fastemit_lambda": 0.01},
    )
    flash_joint.set_loss(flash_loss)
    flash_joint.set_wer(object())
    encoder, predictor, source_lengths, target_lengths, labels = _end_to_end_batch(dtype)
    flash_encoder = encoder.detach().clone().requires_grad_(True)
    flash_predictor = predictor.detach().clone().requires_grad_(True)

    logits = dense_joint.joint(encoder.transpose(1, 2), predictor.transpose(1, 2))
    dense_value = RNNTLossNumba(blank=BLANK, reduction="none", fastemit_lambda=0.01)(
        logits.float(), labels, source_lengths, target_lengths
    ).mean()
    flash_value = flash_joint(
        encoder_outputs=flash_encoder,
        decoder_outputs=flash_predictor,
        encoder_lengths=source_lengths,
        transcripts=labels,
        transcript_lengths=target_lengths,
    )[0]
    dense_gradients = torch.autograd.grad(dense_value, (encoder, predictor, *dense_joint.parameters()))
    flash_gradients = torch.autograd.grad(flash_value, (flash_encoder, flash_predictor, *flash_joint.parameters()))

    atol, rtol = (2e-5, 2e-4) if dtype == torch.float32 else (2e-2, 2e-2)
    torch.testing.assert_close(flash_value, dense_value, atol=atol, rtol=rtol)
    for actual, expected in zip(flash_gradients, dense_gradients):
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


@pytest.mark.unit
@pytest.mark.skipif(
    not CUDA_TRITON_AVAILABLE or not NUMBA_RNNT_AVAILABLE,
    reason="CUDA, Triton, and Numba RNN-T are required",
)
def test_flash_rnnt_prediction_masking_matches_numba():
    """HAINAN masking zeroes a transcript state's projected predictor row; the kernels skip it instead.

    The eager joint multiplies the projection by the mask, so the reference applies the same mask by
    hand and scores the dense joint. A masked state's predictor gradient is zero there, so a backward
    that still accumulates into it fails the comparison. The small tile budget makes the mask cross
    tiles and the checkpoint recompute.
    """
    from nemo.collections.asr.parts.numba.rnnt_loss import RNNTLossNumba

    masking_prob = 0.5
    torch.manual_seed(31)
    dense_joint = _make_joint(4)
    flash_joint = _make_joint(4, masking_prob=masking_prob)
    flash_joint.load_state_dict(dense_joint.state_dict())
    flash_joint.set_loss(
        RNNTLoss(
            num_classes=NUM_LABELS,
            reduction="mean_batch",
            loss_name="flash_rnnt",
            loss_kwargs={"max_joint_rows": 13},
        )
    )
    flash_joint.set_wer(object())
    encoder, predictor, source_lengths, target_lengths, labels = _end_to_end_batch()
    flash_encoder = encoder.detach().clone().requires_grad_(True)
    flash_predictor = predictor.detach().clone().requires_grad_(True)

    # The loss draws its mask as the first CUDA random call after the projections, so the same seed
    # reproduces it here.
    torch.manual_seed(7)
    flash_value = flash_joint(
        encoder_outputs=flash_encoder,
        decoder_outputs=flash_predictor,
        encoder_lengths=source_lengths,
        transcripts=labels,
        transcript_lengths=target_lengths,
    )[0]
    torch.manual_seed(7)
    predictor_mask = torch.rand(encoder.shape[0], predictor.shape[2], device="cuda") > masking_prob
    assert predictor_mask.any() and not predictor_mask.all(), "the draw must mask some states and keep others"

    projected_encoder = dense_joint.project_encoder(encoder.transpose(1, 2))
    projected_predictor = dense_joint.project_prednet(predictor.transpose(1, 2)) * predictor_mask[..., None]
    logits = dense_joint.joint_after_projection(projected_encoder, projected_predictor)
    dense_value = RNNTLossNumba(blank=BLANK, reduction="none")(logits, labels, source_lengths, target_lengths).mean()
    dense_gradients = torch.autograd.grad(dense_value, (encoder, predictor, *dense_joint.parameters()))
    flash_gradients = torch.autograd.grad(flash_value, (flash_encoder, flash_predictor, *flash_joint.parameters()))

    torch.testing.assert_close(flash_value, dense_value, atol=2e-5, rtol=2e-4)
    for actual, expected in zip(flash_gradients, dense_gradients):
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)


@pytest.mark.unit
@pytest.mark.skipif(not CUDA_TRITON_AVAILABLE, reason="CUDA and Triton are required")
def test_flash_rnnt_does_not_materialize_dense_joint_logits(monkeypatch):
    joint = _make_joint(1, activation="tanh")
    loss = RNNTLoss(num_classes=NUM_LABELS, reduction="mean_batch", loss_name="flash_rnnt")

    def reject_dense_joint(*args, **kwargs):
        raise AssertionError("Flash RNN-T must not call RNNTJoint.joint")

    monkeypatch.setattr(joint, "joint", reject_dense_joint)
    joint.set_loss(loss)
    joint.set_wer(object())
    batch, source_steps, target_tokens = 1, 3, 2
    value = joint(
        encoder_outputs=torch.randn(batch, ENCODER_HIDDEN, source_steps, device="cuda"),
        decoder_outputs=torch.randn(batch, PRED_HIDDEN, target_tokens + 1, device="cuda"),
        encoder_lengths=torch.full((batch,), source_steps, device="cuda"),
        transcripts=torch.randint(0, NUM_LABELS, (batch, target_tokens), device="cuda"),
        transcript_lengths=torch.full((batch,), target_tokens, device="cuda"),
    )[0]
    assert value.isfinite()
