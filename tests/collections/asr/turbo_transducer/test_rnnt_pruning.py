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

"""
Tests for the pruning ranges of the transducer lattice (`rnnt_pruning`): the connectivity constraints and
the estimation of the ranges from the log-probabilities of the full lattice.

The lattices have U+1 in (4, 8] and the range size `PRUNING_RANGE_SIZE`, as in the tests of the pruned Joint:
the kernel selecting the ranges is compiled for each range size and block size (the next power of 2 of U+1).
"""

import pytest
import torch

from nemo.collections.asr.parts.turbo_transducer.rnnt_pruning import (
    make_pruning_ranges_connected,
    rnnt_pruning_range_starts,
)
from nemo.core.utils.optional_libs import TRITON_AVAILABLE
from tests.collections.asr.decoding.utils import avoid_sync_operations
from tests.collections.asr.turbo_transducer.utils import PRUNING_RANGE_SIZE, make_joint_inputs, reference_outputs

if TRITON_AVAILABLE:
    from nemo.collections.asr.parts.turbo_transducer.rnnt_loss_triton import rnnt_loss_from_logprobs_triton

pytestmark = pytest.mark.unit

requires_triton_cuda = pytest.mark.skipif(
    not (TRITON_AVAILABLE and torch.cuda.is_available()), reason="Triton and CUDA are required"
)


class TestConnectedRanges:
    @pytest.mark.parametrize("range_size", [2, 3, 5])
    def test_constraints_for_random_starts(self, range_size):
        """Arbitrary starts are adjusted to satisfy the constraints; connected ranges are not changed"""
        max_src_length = 12
        # single frame, empty target, target shorter than the range, more targets than the ranges can emit
        src_lengths = torch.tensor([12, 1, 7, 12, 3, 5, 12, 2])
        tgt_lengths = torch.tensor([9, 0, 3, 30, 20, 1, 11, 4])
        generator = torch.Generator().manual_seed(range_size)
        starts = torch.randint(-2, 40, (len(src_lengths), max_src_length), generator=generator)
        connected_starts = make_pruning_ranges_connected(starts, src_lengths, tgt_lengths, range_size)
        assert connected_starts.shape == starts.shape
        assert connected_starts.dtype == torch.int32
        num_feasible = 0
        for utterance_starts, src_length, tgt_length in zip(
            connected_starts.tolist(), src_lengths.tolist(), tgt_lengths.tolist()
        ):
            utterance_starts = utterance_starts[:src_length]
            last_start = max(tgt_length + 1 - range_size, 0)
            assert utterance_starts[0] == 0
            steps = [end - start for start, end in zip(utterance_starts[:-1], utterance_starts[1:])]
            assert all(0 <= step <= range_size - 1 for step in steps)
            # the last range ends at the last target position, if it can be reached
            max_reachable_start = (range_size - 1) * (src_length - 1)
            assert utterance_starts[-1] == min(last_start, max_reachable_start)
            num_feasible += last_start <= max_reachable_start
        assert 0 < num_feasible < len(src_lengths)
        assert torch.equal(
            make_pruning_ranges_connected(connected_starts, src_lengths, tgt_lengths, range_size), connected_starts
        )

    @pytest.mark.parametrize(
        "starts,tgt_length,expected_starts",
        [
            # connected ranges are not changed
            ([0, 0, 2, 2, 2], 4, [0, 0, 2, 2, 2]),
            ([0, 2, 4, 6, 8], 10, [0, 2, 4, 6, 8]),
            # the first and the last frame are fixed; decreasing start (the 4th frame) is a lower bound for
            # the previous frames, then the starts are increased to reach the last range by steps of 2
            ([3, 5, 5, 1, 0], 10, [0, 2, 4, 6, 8]),
            # too large step after the 2nd frame: the previous starts are increased
            ([0, 0, 0, 5, 5], 7, [0, 1, 3, 5, 5]),
            # target shorter than the range: a single range
            ([1, 2, 0, 1, 2], 1, [0, 0, 0, 0, 0]),
        ],
    )
    def test_adjustment(self, starts, tgt_length, expected_starts):
        connected_starts = make_pruning_ranges_connected(
            torch.tensor([starts]), torch.tensor([len(starts)]), torch.tensor([tgt_length]), range_size=3
        )
        assert connected_starts.tolist() == [expected_starts]

    @pytest.mark.parametrize("range_size", [1, 0, True, 2.5])
    def test_rejects_range_without_alignments(self, range_size):
        with pytest.raises(ValueError, match="range size"):
            make_pruning_ranges_connected(
                torch.zeros(1, 3, dtype=torch.int32), torch.tensor([3]), torch.tensor([2]), range_size
            )


@pytest.mark.gpu
@requires_triton_cuda
class TestRangeEstimation:
    @pytest.mark.parametrize("paired", [False, True])
    def test_ranges_for_random_lattice(self, paired):
        """Ranges are inside the lattice and connected: the loss of the pruned lattice is finite"""
        inputs = make_joint_inputs(double_batch=paired)
        batch_size, max_src_length, max_tgt_length_plus_1 = inputs.lattice_shape
        target_logprobs, blank_logprobs = reference_outputs(inputs, [inputs.joint_hidden, inputs.weight, inputs.bias])
        range_starts, full_lattice_loss = rnnt_pruning_range_starts(
            target_logprobs, blank_logprobs, inputs.src_lengths, inputs.tgt_lengths, PRUNING_RANGE_SIZE, paired=paired
        )
        num_utterances = batch_size // 2 if paired else batch_size
        assert range_starts.shape == (num_utterances, max_src_length)
        assert range_starts.dtype == torch.int32
        assert range_starts.min() >= 0
        assert (range_starts + PRUNING_RANGE_SIZE).max() <= max_tgt_length_plus_1
        assert torch.equal(
            make_pruning_ranges_connected(
                range_starts,
                inputs.src_lengths[:num_utterances],
                inputs.tgt_lengths[:num_utterances],
                PRUNING_RANGE_SIZE,
            ),
            range_starts,
        )

        expected_full_lattice_loss = rnnt_loss_from_logprobs_triton(
            target_logprobs, blank_logprobs, inputs.src_lengths, inputs.tgt_lengths
        )
        torch.testing.assert_close(full_lattice_loss, expected_full_lattice_loss)
        pruned_lattice_loss = _pruned_lattice_loss(inputs, target_logprobs, blank_logprobs, range_starts)
        assert torch.isfinite(pruned_lattice_loss).all()
        # the pruned lattice contains a part of the alignments (all of them for the short targets)
        assert (pruned_lattice_loss >= full_lattice_loss - 1e-4).all()
        is_pruned = inputs.tgt_lengths + 1 > PRUNING_RANGE_SIZE
        assert (pruned_lattice_loss[is_pruned] > full_lattice_loss[is_pruned]).all()
        torch.testing.assert_close(pruned_lattice_loss[~is_pruned], full_lattice_loss[~is_pruned])

    def test_ranges_contain_dominant_alignment(self):
        """Peaked distributions: the ranges follow the alignment with the probability close to 1"""
        src_length, tgt_length, emit_logprob = 12, 7, -0.01
        # frames of the emissions; up to `PRUNING_RANGE_SIZE - 1` targets per frame can be emitted inside a range
        emission_frames = [0, 0, 3, 4, 4, 9, 11]
        target_logprobs = torch.full((1, src_length, tgt_length + 1), -10.0, device="cuda")
        blank_logprobs = torch.full_like(target_logprobs, emit_logprob)
        states = []  # (frame, target position) of the dominant alignment
        for src_index in range(src_length):
            first_position = sum(frame < src_index for frame in emission_frames)
            last_position = sum(frame <= src_index for frame in emission_frames)
            for tgt_index in range(first_position, last_position):
                target_logprobs[0, src_index, tgt_index] = emit_logprob
                blank_logprobs[0, src_index, tgt_index] = -10.0
            states.extend((src_index, tgt_index) for tgt_index in range(first_position, last_position + 1))
        src_lengths = torch.tensor([src_length], device="cuda")
        tgt_lengths = torch.tensor([tgt_length], device="cuda")

        range_starts, full_lattice_loss = rnnt_pruning_range_starts(
            target_logprobs, blank_logprobs, src_lengths, tgt_lengths, PRUNING_RANGE_SIZE
        )
        starts = range_starts[0].tolist()
        assert all(
            starts[src_index] <= tgt_index < starts[src_index] + PRUNING_RANGE_SIZE for src_index, tgt_index in states
        )
        # the loss is determined by the dominant alignment: `src_length + tgt_length` transitions
        expected_loss = -emit_logprob * (src_length + tgt_length)
        assert full_lattice_loss.item() == pytest.approx(expected_loss, abs=1e-2)
        range_mask = _get_range_mask(range_starts, tgt_length + 1)
        pruned_lattice_loss = rnnt_loss_from_logprobs_triton(
            target_logprobs.masked_fill(~range_mask, float("-inf")),
            blank_logprobs.masked_fill(~range_mask, float("-inf")),
            src_lengths,
            tgt_lengths,
        )
        assert pruned_lattice_loss.item() == pytest.approx(full_lattice_loss.item(), abs=1e-3)

    def test_no_gradient_and_cuda_sync_operations(self):
        inputs = make_joint_inputs()
        leaves = inputs.leaves()
        target_logprobs, blank_logprobs = reference_outputs(inputs, leaves)

        def estimate_ranges():
            return rnnt_pruning_range_starts(
                target_logprobs, blank_logprobs, inputs.src_lengths, inputs.tgt_lengths, PRUNING_RANGE_SIZE
            )

        estimate_ranges()  # warmup: compile the kernels
        torch.cuda.synchronize()
        with avoid_sync_operations(torch.device("cuda")):
            range_starts, full_lattice_loss = estimate_ranges()
        assert not range_starts.requires_grad and not full_lattice_loss.requires_grad


def _get_range_mask(range_starts: torch.Tensor, max_tgt_length_plus_1: int) -> torch.Tensor:
    """Mask of the lattice positions inside the pruning ranges, [B, T, U+1]"""
    tgt_positions = torch.arange(max_tgt_length_plus_1, device=range_starts.device)
    starts = range_starts.long().unsqueeze(-1)
    return (tgt_positions >= starts) & (tgt_positions < starts + PRUNING_RANGE_SIZE)


def _pruned_lattice_loss(inputs, target_logprobs, blank_logprobs, range_starts) -> torch.Tensor:
    """RNN-T loss with zero probability outside the pruning ranges (shared by the halves of a double batch)"""
    batch_size, _, max_tgt_length_plus_1 = inputs.lattice_shape
    range_mask = _get_range_mask(range_starts, max_tgt_length_plus_1)
    range_mask = range_mask.repeat(batch_size // range_mask.shape[0], 1, 1)
    return rnnt_loss_from_logprobs_triton(
        target_logprobs.masked_fill(~range_mask, float("-inf")),
        blank_logprobs.masked_fill(~range_mask, float("-inf")),
        inputs.src_lengths,
        inputs.tgt_lengths,
    )
