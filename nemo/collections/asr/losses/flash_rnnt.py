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

"""Exact RNN-T loss computed without materializing the joint tensor.

Scores the ``B * T * (U + 1)`` lattice one tile at a time from the encoder and prediction-network
projections, reduces each tile to the target and blank log-probabilities the dynamic program reads,
and recomputes it in backward. Requires the fused joint step: ``RNNTJoint`` invokes it through
``forward_from_joint``.
"""

from functools import partial

import torch
from torch.utils.checkpoint import checkpoint

from nemo.collections.asr.parts.triton.rnnt_joint import (
    ACTIVATIONS,
    lattice_layout,
    packed_scatter,
    packed_tile_scores,
)
from nemo.collections.asr.parts.triton.rnnt_loss import rnnt_loss_triton
from nemo.core.utils.optional_libs import TRITON_AVAILABLE

_DEFAULT_MAX_JOINT_ROWS = 200_000


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def _validate_joint(joint, blank: int) -> None:
    """Reject joint configurations this path does not reproduce."""
    if joint.is_adapter_available():
        raise ValueError("Flash RNN-T does not support adapters")
    if joint.num_extra_outputs != 0 or blank != joint.num_classes_with_blank - 1:
        raise ValueError("Flash RNN-T requires standard RNN-T with a final blank output")
    # HAT joints score blank in a separate head, leaving joint_net one column short.
    if joint.joint_net[-1].out_features != joint.num_classes_with_blank:
        raise ValueError("Flash RNN-T requires the joint output to include every label and the blank")
    if joint.log_softmax is True or joint.temperature != 1.0:
        raise ValueError("Flash RNN-T requires unnormalized joint logits with temperature 1")
    if joint.activation not in ACTIVATIONS:
        raise ValueError(f"Unsupported RNN-T joint activation: {joint.activation}")
    if not 0.0 <= joint.dropout < 1.0:
        raise ValueError(f"joint dropout must be in [0, 1), got {joint.dropout}")


class FlashRNNTLoss(torch.nn.Module):
    """Exact RNN-T loss for a fused joint step.

    Args:
        blank: blank label index; must be the joint output's final column.
        fastemit_lambda: FastEmit regularization weight.
        clamp: bound applied to the unit-scale gradient at the joint output, before the loss
            reduction and any AMP scale. Disabled when not positive.
        max_joint_rows: rows per lattice tile. Sets peak memory consumption.
    """

    def __init__(
        self,
        blank: int,
        fastemit_lambda: float = 0.0,
        clamp: float = -1.0,
        max_joint_rows: int = _DEFAULT_MAX_JOINT_ROWS,
    ):
        super().__init__()
        if fastemit_lambda < 0.0:
            raise ValueError("fastemit_lambda must be nonnegative")
        if max_joint_rows < 1:
            raise ValueError("max_joint_rows must be positive")
        self.blank = blank
        self.fastemit_lambda = float(fastemit_lambda)
        self.clamp = float(clamp) if clamp > 0.0 else 0.0
        self.max_joint_rows = max_joint_rows

    def forward(
        self,
        joint,
        encoder: torch.Tensor,
        predictor: torch.Tensor,
        targets: torch.Tensor,
        source_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Return per-sample losses from encoder [B, T, D] and prediction network [B, U + 1, D] states."""
        return _compute_flash_rnnt(
            joint=joint,
            encoder=encoder,
            predictor=predictor,
            targets=targets,
            source_lengths=source_lengths,
            target_lengths=target_lengths,
            blank=self.blank,
            fastemit_lambda=self.fastemit_lambda,
            clamp=self.clamp,
            max_joint_rows=self.max_joint_rows,
        )


def _packed_scores(
    projected_encoder: torch.Tensor,
    projected_predictor: torch.Tensor,
    targets: torch.Tensor,
    source_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    output_weight: torch.Tensor,
    output_bias: torch.Tensor,
    activation: str,
    dropout_p: float,
    blank: int,
    clamp: float,
    max_joint_rows: int,
    loss_grad_scale: torch.Tensor | None,
    predictor_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the target and blank score planes, ``[B, T, U + 1]`` each."""
    batch, source_steps = projected_encoder.shape[0], projected_encoder.shape[1]
    target_states = projected_predictor.shape[1]
    offsets, states, total_rows = lattice_layout(source_lengths, target_lengths, source_steps, target_states)
    device = projected_encoder.device
    lengths = source_lengths.to(torch.int32)
    # The extraction addresses a label by row stride alone, so the columns have to be adjacent.
    targets = targets.contiguous()
    tile_rows = _ceil_div(total_rows, _ceil_div(total_rows, max_joint_rows))

    # Bound rather than passed as an argument: the loss fills this buffer during its own backward,
    # and checkpoint rejects a tensor argument whose version changed since the forward.
    score_tile = partial(packed_tile_scores, loss_grad_scale=loss_grad_scale)

    # The dynamic program reads a rectangular lattice, so each tile scatters straight into one.
    shape = (batch, source_steps, target_states)
    target_scores = torch.zeros(batch * source_steps * target_states, device=device, dtype=torch.float32)
    blank_scores = torch.zeros_like(target_scores)

    for start in range(0, total_rows, tile_rows):
        rows = min(tile_rows, total_rows - start)
        # Drawn outside the tile so the recomputation replays the forward's mask. With dropout off
        # the branch that reads it is not compiled, so any valid address works.
        seed = lengths
        if dropout_p > 0.0:
            seed = torch.randint(0, 2**31 - 1, (1,), device=device, dtype=torch.int32)
        target_score_tile, blank_score_tile = checkpoint(
            score_tile,
            projected_encoder,
            projected_predictor,
            output_weight,
            output_bias,
            targets,
            offsets,
            states,
            lengths,
            seed,
            predictor_mask,
            start,
            rows,
            activation,
            dropout_p,
            blank,
            clamp,
            use_reentrant=False,
            preserve_rng_state=False,
        )
        # The scatter derives each address from its row, so no index over the lattice is held.
        target_scores, blank_scores = packed_scatter(
            target_score_tile, blank_score_tile, target_scores, blank_scores, offsets, states, start, shape
        )

    return target_scores.view(shape), blank_scores.view(shape)


def _compute_flash_rnnt(
    joint,
    encoder: torch.Tensor,
    predictor: torch.Tensor,
    targets: torch.Tensor,
    source_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    blank: int,
    fastemit_lambda: float,
    clamp: float,
    max_joint_rows: int,
) -> torch.Tensor:
    """Per-sample losses from the projections, before the joint network is applied."""
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is required for flash RNN-T training")
    if not encoder.is_cuda:
        raise RuntimeError("Flash RNN-T training requires CUDA tensors")
    _validate_joint(joint, blank)
    batch = encoder.shape[0]
    if predictor.shape[0] != batch or targets.shape[0] != batch:
        raise ValueError("encoder, predictor, and targets must have the same batch size")
    if source_lengths.shape != (batch,) or target_lengths.shape != (batch,):
        raise ValueError("source_lengths and target_lengths must contain one value per batch item")

    projected_encoder = joint.project_encoder(encoder)
    projected_predictor = joint.project_prednet(predictor)

    masking_prob = joint.masking_prob if joint.training else -1.0
    predictor_mask = torch.rand(batch, projected_predictor.shape[1], device=encoder.device) > masking_prob

    # Clamping bounds the gradient at unit scale, before the loss reduction and any AMP scale
    # multiply it. Those factors are already folded in by the time the extraction backward runs, so
    # the dynamic program publishes each sample's loss gradient in this buffer and the extraction
    # divides it out, clamps, and reapplies it.
    loss_grad_scale = torch.zeros(batch, device=encoder.device, dtype=torch.float32) if clamp > 0.0 else None

    output_layer = joint.joint_net[-1]
    dropout_p = joint.dropout if joint.training else 0.0

    target_scores, blank_scores = _packed_scores(
        projected_encoder,
        projected_predictor,
        targets,
        source_lengths,
        target_lengths,
        output_layer.weight,
        output_layer.bias,
        joint.activation,
        dropout_p,
        blank,
        clamp,
        max_joint_rows,
        loss_grad_scale,
        predictor_mask,
    )
    return rnnt_loss_triton(
        target_scores[..., :-1],
        blank_scores,
        source_lengths,
        target_lengths,
        fastemit_lambda,
        loss_grad_scale=loss_grad_scale,
    )
