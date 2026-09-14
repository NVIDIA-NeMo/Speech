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

"""RNN-T joint kernels over a packed lattice.

Each utterance contributes ``T * (U + 1)`` rows to one flat list, and tiles are cut from it at a
caller-chosen row budget, so a tile carries no padding whatever lengths it spans. Rows resolve their
own coordinates through ``_locate``, storing no per-row index array. Score planes stay rectangular
``[B, T, U + 1]`` for the dynamic program; ``packed_positions`` maps rows onto them.
"""

from __future__ import annotations

import torch

from nemo.core.utils.optional_libs import TRITON_AVAILABLE

ACTIVATIONS = ("relu", "sigmoid", "tanh")

if TRITON_AVAILABLE:
    import triton
    import triton.language as tl

    @triton.jit
    def _activate_fwd(value, activation: tl.constexpr):
        # No fallback branch: an unlisted activation matches nothing and fails to compile.
        if activation == "relu":
            return tl.maximum(value, 0.0)
        if activation == "sigmoid":
            return tl.sigmoid(value)
        if activation == "tanh":
            return (2.0 * tl.sigmoid(2.0 * value)) - 1.0

    @triton.jit
    def _activate_bwd(pre_activation, activation: tl.constexpr):
        """d act / d pre-activation, recomputed from the pre-activation."""
        if activation == "relu":
            return tl.where(pre_activation > 0.0, 1.0, 0.0)
        output = _activate_fwd(pre_activation, activation)
        if activation == "sigmoid":
            return output * (1.0 - output)
        if activation == "tanh":
            return 1.0 - (output * output)

    @triton.jit
    def _rng_row_stride(hidden_size):
        """Row spacing, in draws, that keeps each block's first element on a Philox boundary.

        Four elements share a draw, so rows round up to a multiple of four. Memory is addressed with
        ``hidden_size``, not this.
        """
        return ((hidden_size + 3) // 4) * 4

    @triton.jit
    def _apply_dropout(value, seed, base_index, dropout_p: tl.constexpr, block_hidden: tl.constexpr):
        """Apply the joint's dropout to a hidden block, rescaling survivors by ``1 / (1 - p)``.

        An element's draw is fixed by its position in the grid ``_rng_row_stride`` spaces, so no mask
        is stored and every kernel reaches the same one.
        """
        if dropout_p > 0.0:
            keys = base_index // 4 + tl.arange(0, block_hidden // 4)
            first, second, third, fourth = tl.rand4x(seed, keys)
            # Interleave back into element order, so the draw does not depend on block_hidden
            uniform = tl.reshape(tl.join(tl.join(first, third), tl.join(second, fourth)), [block_hidden])
            return tl.where(uniform > dropout_p, value / (1.0 - dropout_p), 0.0)
        return value

    @triton.jit
    def _locate(row_starts_ptr, states_ptr, row, batch, batch_pow2: tl.constexpr):
        """Resolve a packed row to its sample, frame, state, and that sample's state count."""
        lanes = tl.arange(0, batch_pow2)
        starts = tl.load(row_starts_ptr + lanes, mask=lanes <= batch, other=2**30)
        batch_idx = tl.sum(tl.where(starts <= row, 1, 0)) - 1
        start = tl.sum(tl.where(lanes == batch_idx, starts, 0))
        states = tl.sum(tl.where(lanes == batch_idx, tl.load(states_ptr + lanes, mask=lanes < batch, other=1), 0))
        within = row - start
        return batch_idx, within // states, within % states, states

    @triton.autotune(configs=[triton.Config({}, num_warps=warps) for warps in (1, 2, 4)], key=["hidden_size"])
    @triton.jit
    def _packed_joint_fwd(
        encoder_ptr,
        predictor_ptr,
        row_starts_ptr,
        states_ptr,
        hidden_ptr,
        seed_ptr,
        predictor_mask_ptr,
        tile_start,
        batch,
        source_stride,
        target_stride,
        hidden_size,
        activation: tl.constexpr,
        dropout_p: tl.constexpr,
        batch_pow2: tl.constexpr,
        block_hidden: tl.constexpr,
    ):
        """Build one packed row of the joint state: add the two strips, activate, drop out.

        The hidden buffer is indexed from the tile's start, the strips by the resolved coordinates.
        """
        local = tl.program_id(0).to(tl.int64)
        row = tile_start + local
        batch_idx, source_idx, target_idx, _ = _locate(row_starts_ptr, states_ptr, row, batch, batch_pow2)

        offsets = tl.arange(0, block_hidden)
        mask = offsets < hidden_size
        encoder = tl.load(
            encoder_ptr + (batch_idx * source_stride + source_idx) * hidden_size + offsets, mask=mask, other=0.0
        ).to(tl.float32)
        predictor = tl.load(
            predictor_ptr + (batch_idx * target_stride + target_idx) * hidden_size + offsets,
            mask=mask & tl.load(predictor_mask_ptr + batch_idx * target_stride + target_idx),
            other=0.0,
        ).to(tl.float32)
        hidden = _activate_fwd(encoder + predictor, activation)
        rng_base = row * _rng_row_stride(hidden_size)
        hidden = _apply_dropout(hidden, tl.load(seed_ptr), rng_base, dropout_p, block_hidden)
        tl.store(hidden_ptr + local * hidden_size + offsets, hidden, mask=mask)

    @triton.jit
    def _apply_dropout_rows(
        value, seed, base_index, dropout_p: tl.constexpr, block_rows: tl.constexpr, block_hidden: tl.constexpr
    ):
        """Apply the joint's dropout to a block of rows.

        Each row draws from its own grid position, so the mask does not depend on the block shape.
        """
        if dropout_p > 0.0:
            keys = base_index[:, None] // 4 + tl.arange(0, block_hidden // 4)[None, :]
            first, second, third, fourth = tl.rand4x(seed, keys)
            uniform = tl.reshape(tl.join(tl.join(first, third), tl.join(second, fourth)), [block_rows, block_hidden])
            return tl.where(uniform > dropout_p, value / (1.0 - dropout_p), 0.0)
        return value

    @triton.autotune(
        configs=[
            triton.Config(
                {"block_frames": frames, "block_states": states, "block_hidden": hidden, "span": span},
                num_warps=warps,
            )
            for frames, states, hidden, span, warps in (
                (32, 1, 64, 32, 4),
                (32, 1, 64, 128, 4),
                (16, 1, 64, 64, 4),
                (16, 4, 32, 256, 4),
            )
        ],
        # Keyed on the model's hidden size alone: the transcript extent changes every batch, and
        # tuning on it would retune for the whole run to pick between configurations that measure
        # the same on real shapes.
        key=["hidden_size"],
        # Timing a config runs the kernel repeatedly, and both gradients may close with atomics, so
        # without this every trial would accumulate on top of the last.
        reset_to_zero=["grad_encoder_ptr", "grad_predictor_ptr"],
    )
    @triton.jit
    def _packed_joint_bwd(
        encoder_ptr,
        predictor_ptr,
        row_starts_ptr,
        states_ptr,
        lengths_ptr,
        grad_hidden_ptr,
        grad_encoder_ptr,
        grad_predictor_ptr,
        seed_ptr,
        predictor_mask_ptr,
        tile_start,
        tile_rows,
        batch,
        source_stride,
        target_stride,
        hidden_size,
        activation: tl.constexpr,
        dropout_p: tl.constexpr,
        batch_pow2: tl.constexpr,
        block_frames: tl.constexpr,
        block_states: tl.constexpr,
        block_hidden: tl.constexpr,
        span: tl.constexpr,
    ):
        """Reduce the hidden-state gradient onto both axes in one pass over it.

        A program owns a block of frames, a chunk of the hidden size, and ``span`` of the transcript.
        A span covering the transcript leaves one program per frame block and closes the encoder sum,
        so it stores; a shorter span puts more programs on the lattice, which is what keeps
        parallelism up on long transcripts, and closes that sum atomically instead. The decoder sum
        is partial either way.
        """
        # matches the grid, which splits the transcript the same way
        state_blocks = tl.cdiv(target_stride, span)
        chunk = tl.program_id(0)
        frame_block = tl.program_id(1)
        batch_idx = tl.program_id(2) // state_blocks
        state_block = tl.program_id(2) % state_blocks

        lanes = tl.arange(0, batch_pow2)
        start = tl.sum(tl.where(lanes == batch_idx, tl.load(row_starts_ptr + lanes, mask=lanes <= batch, other=0), 0))
        states = tl.sum(tl.where(lanes == batch_idx, tl.load(states_ptr + lanes, mask=lanes < batch, other=1), 0))
        source_length = tl.sum(
            tl.where(lanes == batch_idx, tl.load(lengths_ptr + lanes, mask=lanes < batch, other=0), 0)
        )

        frame = frame_block * block_frames + tl.arange(0, block_frames)
        hidden = chunk * block_hidden + tl.arange(0, block_hidden)
        frame_mask = frame < source_length
        hidden_mask = hidden < hidden_size
        live = frame_mask[:, None] & hidden_mask[None, :]

        # The rows this program owns lie between these bounds, so one outside the tile does nothing.
        low = start + frame_block * block_frames * states + state_block * span
        high = start + min((frame_block + 1) * block_frames, source_length) * states
        if high <= tile_start or low >= tile_start + tile_rows:
            return

        encoder = tl.load(
            encoder_ptr + (batch_idx * source_stride + frame)[:, None] * hidden_size + hidden[None, :],
            mask=live,
            other=0.0,
        ).to(tl.float32)
        total = tl.zeros((block_frames, block_hidden), tl.float32)
        seed = tl.load(seed_ptr)
        rng_stride = _rng_row_stride(hidden_size)

        for step in tl.range(0, span, block_states):
            state = state_block * span + step + tl.arange(0, block_states)
            state_mask = state < states
            row = start + frame[:, None] * states + state[None, :]
            local = row - tile_start
            covered = frame_mask[:, None] & state_mask[None, :] & (local >= 0) & (local < tile_rows)

            state_mask &= tl.load(predictor_mask_ptr + batch_idx * target_stride + state, mask=state_mask, other=False)
            predictor = tl.load(
                predictor_ptr + (batch_idx * target_stride + state)[:, None] * hidden_size + hidden[None, :],
                mask=state_mask[:, None] & hidden_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            inside = covered[:, :, None] & hidden_mask[None, None, :]
            grad = tl.load(
                grad_hidden_ptr + local[:, :, None] * hidden_size + hidden[None, None, :], mask=inside, other=0.0
            ).to(tl.float32)
            # the mask is drawn per row, so the tile is flattened over its rows to draw it
            grad = tl.reshape(
                _apply_dropout_rows(
                    tl.reshape(grad, [block_frames * block_states, block_hidden]),
                    seed,
                    tl.reshape(row, [block_frames * block_states]) * rng_stride + chunk * block_hidden,
                    dropout_p,
                    block_frames * block_states,
                    block_hidden,
                ),
                [block_frames, block_states, block_hidden],
            )
            grad *= _activate_bwd(encoder[:, None, :] + predictor[None, :, :], activation)
            grad = tl.where(inside, grad, 0.0)
            total += tl.sum(grad, axis=1)
            if tl.sum(covered.to(tl.int32)) > 0:
                tl.atomic_add(
                    grad_predictor_ptr + (batch_idx * target_stride + state)[:, None] * hidden_size + hidden[None, :],
                    tl.sum(grad, axis=0),
                    mask=state_mask[:, None] & hidden_mask[None, :],
                )

        address = grad_encoder_ptr + (batch_idx * source_stride + frame)[:, None] * hidden_size + hidden[None, :]
        if state_blocks == 1:
            tl.store(address, total, mask=live)
        else:
            tl.atomic_add(address, total, mask=live)

    @triton.autotune(
        configs=[
            triton.Config({"block_rows": rows}, num_warps=warps)
            for rows, warps in ((64, 1), (128, 1), (128, 4), (256, 4), (512, 4), (512, 8))
        ],
        # The work per row is the same at every shape, so one choice serves the whole run.
        key=[],
    )
    @triton.jit
    def _packed_scatter(
        target_ptr,
        blank_ptr,
        target_plane_ptr,
        blank_plane_ptr,
        row_starts_ptr,
        states_ptr,
        tile_start,
        rows,
        batch,
        source_stride,
        target_states,
        gather: tl.constexpr,
        batch_pow2: tl.constexpr,
        block_rows: tl.constexpr,
    ):
        """Move a block of rows' scores between packed order and the rectangular planes.

        Addresses are derived from the rows rather than read from an index, so neither direction
        stores one. A block amortises that derivation over its rows; ``gather`` runs it backwards.
        """
        local = tl.program_id(0) * block_rows + tl.arange(0, block_rows)
        live = local < rows
        row = (tile_start + local).to(tl.int64)

        lanes = tl.arange(0, batch_pow2)
        starts = tl.load(row_starts_ptr + lanes, mask=lanes <= batch, other=2**30)
        counts = tl.load(states_ptr + lanes, mask=lanes < batch, other=1)
        batch_idx = tl.sum(tl.where(starts[None, :] <= row[:, None], 1, 0), axis=1) - 1
        picked = lanes[None, :] == batch_idx[:, None]
        start = tl.sum(tl.where(picked, starts[None, :], 0), axis=1)
        states = tl.sum(tl.where(picked, counts[None, :], 0), axis=1)

        within = row - start
        plane = (batch_idx * source_stride + within // states) * target_states + within % states
        if gather:
            tl.store(target_ptr + local, tl.load(target_plane_ptr + plane, mask=live, other=0.0), mask=live)
            tl.store(blank_ptr + local, tl.load(blank_plane_ptr + plane, mask=live, other=0.0), mask=live)
        else:
            tl.store(target_plane_ptr + plane, tl.load(target_ptr + local, mask=live, other=0.0), mask=live)
            tl.store(blank_plane_ptr + plane, tl.load(blank_ptr + local, mask=live, other=0.0), mask=live)

    @triton.autotune(configs=[triton.Config({}, num_warps=warps) for warps in (1, 2, 4)], key=["vocab"])
    @triton.jit
    def _packed_logprobs_fwd(
        logits_ptr,
        targets_ptr,
        row_starts_ptr,
        states_ptr,
        target_out_ptr,
        blank_out_ptr,
        logsumexp_ptr,
        tile_start,
        batch,
        vocab,
        blank_id,
        target_stride,
        batch_pow2: tl.constexpr,
        block_vocab: tl.constexpr,
    ):
        """Reduce one packed row to the target and blank log-probabilities, keeping its log-sum-exp.

        The saved log-sum-exp lets the backward reach the softmax with one exponential.
        """
        local = tl.program_id(0).to(tl.int64)
        row = tile_start + local
        batch_idx, source_idx, target_idx, states = _locate(row_starts_ptr, states_ptr, row, batch, batch_pow2)

        offsets = tl.arange(0, block_vocab)
        mask = offsets < vocab
        logits = tl.load(logits_ptr + local * vocab + offsets, mask=mask, other=-float("inf")).to(tl.float32)
        peak = tl.max(logits, axis=0)
        logsumexp = peak + tl.log(tl.sum(tl.exp(tl.where(mask, logits - peak, -float("inf"))), axis=0))

        blank_logit = tl.load(logits_ptr + local * vocab + blank_id).to(tl.float32)
        tl.store(blank_out_ptr + local, blank_logit - logsumexp)
        tl.store(logsumexp_ptr + local, logsumexp)
        # the last state of a transcript has no token to score; the plane keeps its zero there
        token_logit = 0.0
        if target_idx < states - 1:
            token = tl.load(targets_ptr + batch_idx * target_stride + target_idx)
            token_logit = tl.load(logits_ptr + local * vocab + token).to(tl.float32) - logsumexp
        tl.store(target_out_ptr + local, token_logit)

    @triton.autotune(
        configs=[triton.Config({}, num_warps=warps) for warps in (1, 2, 4)],
        key=["vocab"],
        restore_value=["logits_ptr"],
    )
    @triton.jit
    def _packed_logprobs_bwd(
        logits_ptr,
        grad_logits_ptr,
        targets_ptr,
        row_starts_ptr,
        states_ptr,
        logsumexp_ptr,
        grad_target_ptr,
        grad_blank_ptr,
        loss_grad_scale_ptr,
        tile_start,
        batch,
        vocab,
        blank_id,
        target_stride,
        clamp: float,
        clamp_grad: tl.constexpr,
        batch_pow2: tl.constexpr,
        block_vocab: tl.constexpr,
    ):
        """Scatter the two score gradients over one packed row's vocabulary.

        A log-probability's gradient is a one-hot at the label it scored minus the softmax, weighted
        by the upstream gradient.
        """
        local = tl.program_id(0).to(tl.int64)
        row = tile_start + local
        batch_idx, source_idx, target_idx, states = _locate(row_starts_ptr, states_ptr, row, batch, batch_pow2)

        offsets = tl.arange(0, block_vocab)
        mask = offsets < vocab
        logits = tl.load(logits_ptr + local * vocab + offsets, mask=mask, other=-float("inf")).to(tl.float32)
        softmax = tl.exp(logits - tl.load(logsumexp_ptr + local))

        grad_blank = tl.load(grad_blank_ptr + local).to(tl.float32)
        scored = target_idx < states - 1
        grad_token = tl.load(grad_target_ptr + local, mask=scored, other=0.0).to(tl.float32)
        token = tl.load(targets_ptr + batch_idx * target_stride + target_idx, mask=scored, other=-1)

        if clamp_grad:
            # Clamping bounds the gradient at unit scale, before the loss reduction or any AMP
            # scale multiplies it, so divide that scale out here, clamp, and reapply below.
            loss_grad_scale = tl.load(loss_grad_scale_ptr + batch_idx).to(tl.float32)
            inverse_scale = tl.where(loss_grad_scale != 0.0, 1.0 / loss_grad_scale, 0.0)
            grad_blank *= inverse_scale
            grad_token *= inverse_scale

        grad = -softmax * (grad_blank + grad_token)
        grad += tl.where(offsets == blank_id, grad_blank, 0.0)
        grad += tl.where(offsets == token, grad_token, 0.0)
        if clamp_grad:
            grad = tl.maximum(tl.minimum(grad, clamp), -clamp)
            grad *= loss_grad_scale
        tl.store(grad_logits_ptr + local * vocab + offsets, grad, mask=mask)


def lattice_layout(source_lengths, target_lengths, source_steps, target_states):
    """Return per-sample start rows, transcript-state counts, and the total row count.

    Raises if a length reaches past the extent it indexes, since the kernels read those positions
    rather than mask them. The check shares the transfer that brings the row count to the host.
    """
    states = (target_lengths + 1).to(torch.int32)
    sizes = source_lengths.to(torch.int32) * states
    offsets = torch.zeros(len(sizes) + 1, device=sizes.device, dtype=torch.int32)
    offsets[1:] = torch.cumsum(sizes, 0)
    total_rows, longest_source, most_states = torch.stack(
        (offsets[-1].long(), source_lengths.max().long(), states.max().long())
    ).tolist()
    if longest_source > source_steps or most_states > target_states:
        raise ValueError(
            f"lengths reach beyond the states they index: {longest_source} source steps and "
            f"{most_states} transcript states, against tensors holding {source_steps} and {target_states}"
        )
    return offsets, states, total_rows


class _PackedJoint(torch.autograd.Function):
    """Joint state for one tile of packed rows, ``[rows, H]``.

    Saves the two projections rather than the state, and rebuilds the pre-activation in backward.
    """

    @staticmethod
    def forward(
        ctx, encoder, predictor, offsets, states, lengths, seed, predictor_mask, start, rows, activation, dropout_p
    ):
        hidden_size = encoder.shape[2]
        hidden = torch.empty(rows, hidden_size, device=encoder.device, dtype=encoder.dtype)
        block = triton.next_power_of_2(hidden_size)
        batch_pow2 = triton.next_power_of_2(len(states) + 1)
        _packed_joint_fwd[(rows,)](
            encoder,
            predictor,
            offsets,
            states,
            hidden,
            seed,
            predictor_mask,
            start,
            len(states),
            encoder.shape[1],
            predictor.shape[1],
            hidden_size,
            activation=activation,
            dropout_p=dropout_p,
            batch_pow2=batch_pow2,
            block_hidden=block,
        )
        ctx.save_for_backward(encoder, predictor, offsets, states, lengths, seed, predictor_mask)
        ctx.meta = (activation, dropout_p, batch_pow2, block)
        ctx.tile_start = start
        return hidden

    @staticmethod
    def backward(ctx, grad_hidden):
        encoder, predictor, offsets, states, lengths, seed, predictor_mask = ctx.saved_tensors
        activation, dropout_p, batch_pow2, _ = ctx.meta
        grad_hidden = grad_hidden.contiguous()
        batch, hidden_size = len(states), encoder.shape[2]
        grad_encoder = torch.zeros_like(encoder, dtype=torch.float32)
        grad_predictor = torch.zeros_like(predictor, dtype=torch.float32)
        max_states, max_frames = predictor.shape[1], encoder.shape[1]
        grid = lambda meta: (  # noqa: E731
            triton.cdiv(hidden_size, meta["block_hidden"]),
            triton.cdiv(max_frames, meta["block_frames"]),
            batch * triton.cdiv(max_states, meta["span"]),
        )
        _packed_joint_bwd[grid](
            encoder,
            predictor,
            offsets,
            states,
            lengths,
            grad_hidden,
            grad_encoder,
            grad_predictor,
            seed,
            predictor_mask,
            ctx.tile_start,
            grad_hidden.shape[0],
            batch,
            max_frames,
            max_states,
            hidden_size,
            activation=activation,
            dropout_p=dropout_p,
            batch_pow2=batch_pow2,
        )
        return (
            grad_encoder.to(encoder.dtype),
            grad_predictor.to(predictor.dtype),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class _PackedLogProbs(torch.autograd.Function):
    """The target and blank log-probabilities for one tile, ``[rows]`` each."""

    @staticmethod
    def forward(ctx, logits, targets, offsets, states, start, blank_id, clamp, loss_grad_scale):
        rows, vocab = logits.shape
        batch_pow2 = triton.next_power_of_2(len(states) + 1)
        block = triton.next_power_of_2(vocab)
        target_out = torch.empty(rows, device=logits.device, dtype=torch.float32)
        blank_out = torch.empty(rows, device=logits.device, dtype=torch.float32)
        logsumexp = torch.empty(rows, device=logits.device, dtype=torch.float32)
        _packed_logprobs_fwd[(rows,)](
            logits,
            targets,
            offsets,
            states,
            target_out,
            blank_out,
            logsumexp,
            start,
            len(states),
            vocab,
            blank_id,
            targets.stride(0),
            batch_pow2=batch_pow2,
            block_vocab=block,
        )
        ctx.save_for_backward(logits, targets, offsets, states, logsumexp)
        ctx.meta = (start, blank_id, batch_pow2, block, targets.stride(0))
        ctx.clamp = float(clamp) if clamp > 0.0 else 0.0
        # Held outside the saved tensors: the loss backward fills this buffer after the forward, and
        # the version counter would reject a saved tensor that changed in between.
        ctx.loss_grad_scale = loss_grad_scale
        return target_out, blank_out

    @staticmethod
    def backward(ctx, grad_target, grad_blank):
        logits, targets, offsets, states, logsumexp = ctx.saved_tensors
        start, blank_id, batch_pow2, block, target_stride = ctx.meta
        rows, vocab = logits.shape
        # The tile owns its logits -- they are produced by the projection inside it and read by
        # nothing else, and the linear's own backward wants the joint state, not this output -- so
        # the gradient is written over them. Each row is loaded before it is stored and no program
        # reads another row, so the overwrite races with nothing.
        grad_logits = logits
        # Any valid pointer will do when clamping is off: clamp_grad is a constexpr, so the
        # kernel that reads it is never compiled.
        loss_grad_scale = ctx.loss_grad_scale if ctx.loss_grad_scale is not None else logsumexp
        _packed_logprobs_bwd[(rows,)](
            logits,
            grad_logits,
            targets,
            offsets,
            states,
            logsumexp,
            grad_target.contiguous(),
            grad_blank.contiguous(),
            loss_grad_scale,
            start,
            len(states),
            vocab,
            blank_id,
            target_stride,
            clamp=ctx.clamp,
            clamp_grad=ctx.clamp > 0.0,
            batch_pow2=batch_pow2,
            block_vocab=block,
        )
        return grad_logits, None, None, None, None, None, None, None


def packed_tile_scores(
    encoder,
    predictor,
    weight,
    bias,
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
    blank_id,
    clamp,
    loss_grad_scale,
):
    """Score rows ``[start, start + rows)``: joint state, vocabulary projection, extraction.

    Both intermediates are local to the call, so it can be wrapped in ``torch.utils.checkpoint``.
    """
    hidden = _PackedJoint.apply(
        encoder, predictor, offsets, states, lengths, seed, predictor_mask, start, rows, activation, dropout_p
    )
    logits = torch.nn.functional.linear(hidden, weight, bias)
    return _PackedLogProbs.apply(logits, targets, offsets, states, start, blank_id, clamp, loss_grad_scale)


class _PackedScatter(torch.autograd.Function):
    """Place one tile's two score vectors into the rectangular planes the dynamic program reads."""

    @staticmethod
    def forward(ctx, target_tile, blank_tile, target_plane, blank_plane, offsets, states, start, shape):
        rows = target_tile.shape[0]
        _packed_scatter[lambda meta: (triton.cdiv(rows, meta["block_rows"]),)](
            target_tile,
            blank_tile,
            target_plane,
            blank_plane,
            offsets,
            states,
            start,
            rows,
            len(states),
            shape[1],
            shape[2],
            gather=False,
            batch_pow2=triton.next_power_of_2(len(states) + 1),
        )
        ctx.save_for_backward(offsets, states)
        ctx.meta = (start, rows, shape)
        return target_plane, blank_plane

    @staticmethod
    def backward(ctx, grad_target_plane, grad_blank_plane):
        offsets, states = ctx.saved_tensors
        start, rows, shape = ctx.meta
        grad_target = torch.empty(rows, device=offsets.device, dtype=torch.float32)
        grad_blank = torch.empty_like(grad_target)
        _packed_scatter[lambda meta: (triton.cdiv(rows, meta["block_rows"]),)](
            grad_target,
            grad_blank,
            grad_target_plane.contiguous(),
            grad_blank_plane.contiguous(),
            offsets,
            states,
            start,
            rows,
            len(states),
            shape[1],
            shape[2],
            gather=True,
            batch_pow2=triton.next_power_of_2(len(states) + 1),
        )
        return grad_target, grad_blank, grad_target_plane, grad_blank_plane, None, None, None, None


packed_scatter = _PackedScatter.apply
