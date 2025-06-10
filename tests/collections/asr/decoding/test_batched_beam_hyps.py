# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from contextlib import contextmanager
from typing import List, Literal, Union

import pytest
import torch

from nemo.collections.asr.parts.utils.batched_beam_decoding_utils import (
    INIT_POINTER_VALUE,
    NON_EXISTENT_LABEL_VALUE,
    BatchedBeamHyps,
)
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis, NBestHypotheses


NestedFloatList = Union[float, List["NestedFloatList"]]  # recursive type alias


def assert_nested_lists_approx(
    actual: NestedFloatList, expected: NestedFloatList, rel_tol: float = 1e-4, abs_tol: float = 1e-4
) -> None:
    """
    Recursively asserts that two nested lists of floats are approximately equal
    within a given relative and absolute tolerance.
    """
    if isinstance(actual, list) and isinstance(expected, list):
        assert len(actual) == len(expected), f"Length mismatch: {len(actual)} != {len(expected)}"
        for act, exp in zip(actual, expected):
            assert_nested_lists_approx(act, exp, rel_tol, abs_tol)
    else:
        assert actual == pytest.approx(
            expected, rel=rel_tol, abs=abs_tol
        ), f"Values differ: actual={actual}, expected={expected}, rel_tol={rel_tol}, abs_tol={abs_tol}"


def assert_hyps_sequence_equal(
    actual: Union[List[int], torch.Tensor], expected: list[int], rel_tol: float = 1e-4, abs_tol: float = 1e-4
):
    """
    Asserts that two sequences of hypotheses are approximately equal.
    """
    if isinstance(actual, torch.Tensor):
        actual = actual.cpu().tolist()
    assert_nested_lists_approx(actual, expected, rel_tol, abs_tol)


def assert_hyps_timestamps_equal(
    actual: Union[List[int], torch.Tensor], expected: list[int], rel_tol: float = 1e-4, abs_tol: float = 1e-4
):
    """
    Asserts that two sequences of timestamp values are approximately equal.
    """
    if isinstance(actual, torch.Tensor):
        actual = actual.cpu().tolist()
    assert_nested_lists_approx(actual, expected, rel_tol, abs_tol)


@contextmanager
def avoid_sync_operations(device: torch.device):
    try:
        if device.type == "cuda":
            torch.cuda.set_sync_debug_mode(2)  # fail if a blocking operation
        yield
    finally:
        if device.type == "cuda":
            torch.cuda.set_sync_debug_mode(0)  # default, blocking operations are allowed


DEVICES: List[torch.device] = [torch.device("cpu")]

if torch.cuda.is_available():
    DEVICES.append(torch.device("cuda"))

if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICES.append(torch.device("mps"))


class TestBatchedBeamHyps:
    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    def test_rnnt_instantiate(self, device: torch.device):
        _ = BatchedBeamHyps(batch_size=2, beam_size=3, init_length=4, device=device, blank_index=1024)

    @pytest.mark.unit
    @pytest.mark.parametrize("batch_size", [-1, 0])
    def test_rnnt_instantiate_incorrect_batch_size(self, batch_size: Literal[-1] | Literal[0]):
        with pytest.raises(ValueError):
            _ = BatchedBeamHyps(batch_size=batch_size, beam_size=4, init_length=3, blank_index=1024)

    @pytest.mark.unit
    @pytest.mark.parametrize("beam_size", [-1, 0])
    def test_rnnt_instantiate_incorrect_beam_size(self, beam_size: Literal[-1] | Literal[0]):
        with pytest.raises(ValueError):
            _ = BatchedBeamHyps(batch_size=2, beam_size=beam_size, init_length=3, blank_index=1024)

    @pytest.mark.unit
    @pytest.mark.parametrize("init_length", [-1, 0])
    def test_rnnt_instantiate_incorrect_init_length(self, init_length: Literal[-1] | Literal[0]):
        with pytest.raises(ValueError):
            _ = BatchedBeamHyps(batch_size=1, beam_size=4, init_length=init_length, blank_index=1024)

    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    def test_rnnt_add_results(self, device: torch.device):
        # batch of size 2, add label for first utterance
        hyps = BatchedBeamHyps(batch_size=2, beam_size=3, init_length=1, device=device, blank_index=1024)
        assert hyps._max_length == 1
        hyps.add_results_(
            next_indices=torch.tensor([[0, 1, 2], [0, 1, 2]], device=device),
            next_labels=torch.tensor([[0, 1024, 1], [2, 1024, 1024]], device=device),
            next_hyps_prob=torch.tensor([[0.5, 0.6, 0.8], [0.1, 0.2, 0.3]], device=device),
        )
        assert hyps._max_length == 2
        assert hyps.current_lengths_nb.tolist() == [[1, 0, 1], [1, 0, 0]]
        assert hyps.current_lengths_wb.tolist() == [[1, 1, 1], [1, 1, 1]]
        assert_nested_lists_approx(actual=hyps.scores.tolist(), expected=[[0.5, 0.6, 0.8], [0.1, 0.2, 0.3]])
        assert hyps.transcript_wb.tolist() == [
            [[0, NON_EXISTENT_LABEL_VALUE], [1024, NON_EXISTENT_LABEL_VALUE], [1, NON_EXISTENT_LABEL_VALUE]],
            [[2, NON_EXISTENT_LABEL_VALUE], [1024, NON_EXISTENT_LABEL_VALUE], [1024, NON_EXISTENT_LABEL_VALUE]],
        ]
        assert hyps.transcript_wb_prev_ptr.tolist() == [
            [[0, INIT_POINTER_VALUE], [1, INIT_POINTER_VALUE], [2, INIT_POINTER_VALUE]],
            [[0, INIT_POINTER_VALUE], [1, INIT_POINTER_VALUE], [2, INIT_POINTER_VALUE]],
        ]
        assert hyps.timestamps.tolist() == [
            [[0, 0], [1, 0], [0, 0]],
            [[0, 0], [1, 0], [1, 0]],
        ]
        assert hyps.next_timestamp.tolist() == [
            [0, 1, 0],
            [0, 1, 1],
        ]

    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    def test_rnnt_add_multiple_results(self, device: torch.device):
        hyps = BatchedBeamHyps(batch_size=2, beam_size=3, init_length=1, device=device, blank_index=1024)
        assert hyps._max_length == 1

        hyps.add_results_(
            next_indices=torch.tensor([[0, 1, 2], [0, 1, 2]], device=device),
            next_labels=torch.tensor([[0, 1024, 1], [2, 1024, 1024]], device=device),
            next_hyps_prob=torch.tensor([[0.5, 0.6, 0.8], [0.1, 0.2, 0.3]], device=device),
        )

        hyps.add_results_(
            next_indices=torch.tensor([[0, 1, 1], [2, 1, 0]], device=device),
            next_labels=torch.tensor([[3, 4, 1024], [5, 1024, 6]], device=device),
            next_hyps_prob=torch.tensor([[0.3, 0.2, 0.1], [0.4, 0.5, 0.6]], device=device),
        )

        assert hyps._max_length == 4
        assert hyps.current_lengths_nb.tolist() == [[2, 1, 0], [1, 0, 2]]
        assert hyps.current_lengths_wb.tolist() == [[2, 2, 2], [2, 2, 2]]
        assert_nested_lists_approx(actual=hyps.scores.tolist(), expected=[[0.3, 0.2, 0.1], [0.4, 0.5, 0.6]])
        assert hyps.transcript_wb.tolist() == [
            [
                [0, 3, NON_EXISTENT_LABEL_VALUE, NON_EXISTENT_LABEL_VALUE],
                [1024, 4, NON_EXISTENT_LABEL_VALUE, NON_EXISTENT_LABEL_VALUE],
                [1, 1024, NON_EXISTENT_LABEL_VALUE, NON_EXISTENT_LABEL_VALUE],
            ],
            [
                [2, 5, NON_EXISTENT_LABEL_VALUE, NON_EXISTENT_LABEL_VALUE],
                [1024, 1024, NON_EXISTENT_LABEL_VALUE, NON_EXISTENT_LABEL_VALUE],
                [1024, 6, NON_EXISTENT_LABEL_VALUE, NON_EXISTENT_LABEL_VALUE],
            ],
        ]
        assert hyps.transcript_wb_prev_ptr.tolist() == [
            [
                [0, 0, INIT_POINTER_VALUE, INIT_POINTER_VALUE],
                [1, 1, INIT_POINTER_VALUE, INIT_POINTER_VALUE],
                [2, 1, INIT_POINTER_VALUE, INIT_POINTER_VALUE],
            ],
            [
                [0, 2, INIT_POINTER_VALUE, INIT_POINTER_VALUE],
                [1, 1, INIT_POINTER_VALUE, INIT_POINTER_VALUE],
                [2, 0, INIT_POINTER_VALUE, INIT_POINTER_VALUE],
            ],
        ]
        assert hyps.timestamps.tolist() == [
            [
                [0, 0, 0, 0],
                [1, 1, 0, 0],
                [0, 2, 0, 0],
            ],
            [
                [0, 1, 0, 0],
                [1, 2, 0, 0],
                [1, 0, 0, 0],
            ],
        ]
        assert hyps.next_timestamp.tolist() == [
            [0, 1, 2],
            [1, 2, 0],
        ]

    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    def test_rnnt_add_with_invalid_results(self, device: torch.device):
        hyps = BatchedBeamHyps(batch_size=2, beam_size=3, init_length=1, device=device, blank_index=1024)
        assert hyps._max_length == 1

        hyps.add_results_(
            next_indices=torch.tensor([[0, 1, 2], [0, 1, 2]], device=device),
            next_labels=torch.tensor([[0, 1024, 1], [2, 1024, 1024]], device=device),
            next_hyps_prob=torch.tensor([[0.5, 0.6, 0.8], [0.1, 0.2, 0.3]], device=device),
        )

        hyps.add_results_(
            next_indices=torch.tensor([[0, 1, 1], [2, 1, 0]], device=device),
            next_labels=torch.tensor([[3, 4, 1024], [5, 1024, 6]], device=device),
            next_hyps_prob=torch.tensor([[0.3, 0.2, 0.1], [0.4, 0.5, 0.6]], device=device),
        )

        hyps.add_results_(
            next_indices=torch.tensor([[1, 0, 2], [2, 0, 1]], device=device),
            next_labels=torch.tensor([[-1, 7, 8], [10, -1, 9]], device=device),
            next_hyps_prob=torch.tensor([[0.35, 0.4, 0.1], [0.4, 0.55, 0.6]], device=device),
        )

        assert hyps._max_length == 4
        assert hyps.current_lengths_nb.tolist() == [[1, 3, 1], [3, 1, 1]]
        assert hyps.current_lengths_wb.tolist() == [[3, 3, 3], [3, 3, 3]]
        assert_nested_lists_approx(actual=hyps.scores.tolist(), expected=[[0.35, 0.4, 0.1], [0.4, 0.55, 0.6]])
        assert hyps.transcript_wb.tolist() == [
            [
                [0, 3, -1, NON_EXISTENT_LABEL_VALUE],
                [1024, 4, 7, NON_EXISTENT_LABEL_VALUE],
                [1, 1024, 8, NON_EXISTENT_LABEL_VALUE],
            ],
            [
                [2, 5, 10, NON_EXISTENT_LABEL_VALUE],
                [1024, 1024, -1, NON_EXISTENT_LABEL_VALUE],
                [1024, 6, 9, NON_EXISTENT_LABEL_VALUE],
            ],
        ]
        assert hyps.transcript_wb_prev_ptr.tolist() == [
            [[0, 0, 1, INIT_POINTER_VALUE], [1, 1, 0, INIT_POINTER_VALUE], [2, 1, 2, INIT_POINTER_VALUE]],
            [[0, 2, 2, INIT_POINTER_VALUE], [1, 1, 0, INIT_POINTER_VALUE], [2, 0, 1, INIT_POINTER_VALUE]],
        ]
        assert hyps.timestamps.tolist() == [
            [
                [0, 0, 1, 0],
                [1, 1, 0, 0],
                [0, 2, 2, 0],
            ],
            [
                [0, 1, 0, 0],
                [1, 2, 1, 0],
                [1, 0, 2, 0],
            ],
        ]
        assert hyps.next_timestamp.tolist() == [
            [1, 0, 2],
            [0, 1, 2],
        ]

    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    def test_tdt_instantiate(self, device: torch.device):
        _ = BatchedBeamHyps(
            batch_size=2, beam_size=3, init_length=4, device=device, blank_index=1024, model_type='tdt'
        )

    @pytest.mark.unit
    @pytest.mark.parametrize("batch_size", [-1, 0])
    def test_tdt_instantiate_incorrect_batch_size(self, batch_size: Literal[-1] | Literal[0]):
        with pytest.raises(ValueError):
            _ = BatchedBeamHyps(batch_size=batch_size, beam_size=4, init_length=3, blank_index=1024, model_type='tdt')

    @pytest.mark.unit
    @pytest.mark.parametrize("beam_size", [-1, 0])
    def test_tdt_instantiate_incorrect_beam_size(self, beam_size: Literal[-1] | Literal[0]):
        with pytest.raises(ValueError):
            _ = BatchedBeamHyps(batch_size=2, beam_size=beam_size, init_length=3, blank_index=1024, model_type='tdt')

    @pytest.mark.unit
    @pytest.mark.parametrize("init_length", [-1, 0])
    def test_tdt_instantiate_incorrect_init_length(self, init_length: Literal[-1] | Literal[0]):
        with pytest.raises(ValueError):
            _ = BatchedBeamHyps(batch_size=1, beam_size=4, init_length=init_length, blank_index=1024, model_type='tdt')

    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    def test_tdt_add_results(self, device: torch.device):
        # batch of size 2, add label for first utterance
        hyps = BatchedBeamHyps(
            batch_size=2, beam_size=3, init_length=1, device=device, blank_index=1024, model_type='tdt'
        )
        assert hyps._max_length == 1
        hyps.add_results_(
            next_indices=torch.tensor([[0, 1, 2], [0, 1, 2]], device=device),
            next_labels=torch.tensor([[0, 1024, 1], [2, 1024, 1024]], device=device),
            next_hyps_prob=torch.tensor([[0.5, 0.6, 0.8], [0.1, 0.2, 0.3]], device=device),
            next_label_durations=torch.tensor([[0, 3, 1], [2, 3, 4]], device=device),
        )
        assert hyps._max_length == 2
        assert hyps.current_lengths_nb.tolist() == [[1, 0, 1], [1, 0, 0]]
        assert hyps.current_lengths_wb.tolist() == [[1, 1, 1], [1, 1, 1]]
        assert_nested_lists_approx(actual=hyps.scores.tolist(), expected=[[0.5, 0.6, 0.8], [0.1, 0.2, 0.3]])
        assert hyps.transcript_wb.tolist() == [
            [[0, NON_EXISTENT_LABEL_VALUE], [1024, NON_EXISTENT_LABEL_VALUE], [1, NON_EXISTENT_LABEL_VALUE]],
            [[2, NON_EXISTENT_LABEL_VALUE], [1024, NON_EXISTENT_LABEL_VALUE], [1024, NON_EXISTENT_LABEL_VALUE]],
        ]
        assert hyps.transcript_wb_prev_ptr.tolist() == [
            [[0, INIT_POINTER_VALUE], [1, INIT_POINTER_VALUE], [2, INIT_POINTER_VALUE]],
            [[0, INIT_POINTER_VALUE], [1, INIT_POINTER_VALUE], [2, INIT_POINTER_VALUE]],
        ]

        assert hyps.timestamps.tolist() == [[[0, 0], [3, 0], [1, 0]], [[2, 0], [3, 0], [4, 0]]]

    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    def test_tdt_add_multiple_results(self, device: torch.device):
        hyps = BatchedBeamHyps(
            batch_size=2, beam_size=3, init_length=1, device=device, blank_index=1024, model_type='tdt'
        )
        assert hyps._max_length == 1

        hyps.add_results_(
            next_indices=torch.tensor([[0, 1, 2], [0, 1, 2]], device=device),
            next_labels=torch.tensor([[0, 1024, 1], [2, 1024, 1024]], device=device),
            next_hyps_prob=torch.tensor([[0.5, 0.6, 0.8], [0.1, 0.2, 0.3]], device=device),
            next_label_durations=torch.tensor([[0, 3, 1], [2, 3, 4]], device=device),
        )

        hyps.add_results_(
            next_indices=torch.tensor([[0, 1, 1], [2, 1, 0]], device=device),
            next_labels=torch.tensor([[3, 4, 1024], [5, 1024, 6]], device=device),
            next_hyps_prob=torch.tensor([[0.3, 0.2, 0.1], [0.4, 0.5, 0.6]], device=device),
            next_label_durations=torch.tensor([[2, 4, 1], [0, 1, 1]], device=device),
        )

        assert hyps._max_length == 4
        assert hyps.current_lengths_nb.tolist() == [[2, 1, 0], [1, 0, 2]]
        assert hyps.current_lengths_wb.tolist() == [[2, 2, 2], [2, 2, 2]]
        assert_nested_lists_approx(actual=hyps.scores.tolist(), expected=[[0.3, 0.2, 0.1], [0.4, 0.5, 0.6]])
        assert hyps.transcript_wb.tolist() == [
            [
                [0, 3, NON_EXISTENT_LABEL_VALUE, NON_EXISTENT_LABEL_VALUE],
                [1024, 4, NON_EXISTENT_LABEL_VALUE, NON_EXISTENT_LABEL_VALUE],
                [1, 1024, NON_EXISTENT_LABEL_VALUE, NON_EXISTENT_LABEL_VALUE],
            ],
            [
                [2, 5, NON_EXISTENT_LABEL_VALUE, NON_EXISTENT_LABEL_VALUE],
                [1024, 1024, NON_EXISTENT_LABEL_VALUE, NON_EXISTENT_LABEL_VALUE],
                [1024, 6, NON_EXISTENT_LABEL_VALUE, NON_EXISTENT_LABEL_VALUE],
            ],
        ]
        assert hyps.transcript_wb_prev_ptr.tolist() == [
            [
                [0, 0, INIT_POINTER_VALUE, INIT_POINTER_VALUE],
                [1, 1, INIT_POINTER_VALUE, INIT_POINTER_VALUE],
                [2, 1, INIT_POINTER_VALUE, INIT_POINTER_VALUE],
            ],
            [
                [0, 2, INIT_POINTER_VALUE, INIT_POINTER_VALUE],
                [1, 1, INIT_POINTER_VALUE, INIT_POINTER_VALUE],
                [2, 0, INIT_POINTER_VALUE, INIT_POINTER_VALUE],
            ],
        ]

        assert hyps.timestamps.tolist() == [
            [[0, 2, 0, 0], [3, 7, 0, 0], [1, 4, 0, 0]],
            [[2, 4, 0, 0], [3, 4, 0, 0], [4, 3, 0, 0]],
        ]

    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    def test_tdt_add_with_invalid_results(self, device: torch.device):
        hyps = BatchedBeamHyps(
            batch_size=2, beam_size=3, init_length=1, device=device, blank_index=1024, model_type='tdt'
        )
        assert hyps._max_length == 1

        hyps.add_results_(
            next_indices=torch.tensor([[0, 1, 2], [0, 1, 2]], device=device),
            next_labels=torch.tensor([[0, 1024, 1], [2, 1024, 1024]], device=device),
            next_hyps_prob=torch.tensor([[0.5, 0.6, 0.8], [0.1, 0.2, 0.3]], device=device),
            next_label_durations=torch.tensor([[0, 3, 1], [2, 3, 4]], device=device),
        )

        hyps.add_results_(
            next_indices=torch.tensor([[0, 1, 1], [2, 1, 0]], device=device),
            next_labels=torch.tensor([[3, 4, 1024], [5, 1024, 6]], device=device),
            next_hyps_prob=torch.tensor([[0.3, 0.2, 0.1], [0.4, 0.5, 0.6]], device=device),
            next_label_durations=torch.tensor([[2, 4, 1], [0, 1, 1]], device=device),
        )

        hyps.add_results_(
            next_indices=torch.tensor([[1, 0, 2], [2, 0, 1]], device=device),
            next_labels=torch.tensor([[-1, 7, 8], [10, -1, 9]], device=device),
            next_hyps_prob=torch.tensor([[0.35, 0.4, 0.1], [0.4, 0.55, 0.6]], device=device),
            next_label_durations=torch.tensor([[2, 1, 3], [2, 1, 2]], device=device),
        )

        assert hyps._max_length == 4
        assert hyps.current_lengths_nb.tolist() == [[1, 3, 1], [3, 1, 1]]
        assert hyps.current_lengths_wb.tolist() == [[3, 3, 3], [3, 3, 3]]
        assert_nested_lists_approx(actual=hyps.scores.tolist(), expected=[[0.35, 0.4, 0.1], [0.4, 0.55, 0.6]])
        assert hyps.transcript_wb.tolist() == [
            [
                [0, 3, -1, NON_EXISTENT_LABEL_VALUE],
                [1024, 4, 7, NON_EXISTENT_LABEL_VALUE],
                [1, 1024, 8, NON_EXISTENT_LABEL_VALUE],
            ],
            [
                [2, 5, 10, NON_EXISTENT_LABEL_VALUE],
                [1024, 1024, -1, NON_EXISTENT_LABEL_VALUE],
                [1024, 6, 9, NON_EXISTENT_LABEL_VALUE],
            ],
        ]
        assert hyps.transcript_wb_prev_ptr.tolist() == [
            [[0, 0, 1, INIT_POINTER_VALUE], [1, 1, 0, INIT_POINTER_VALUE], [2, 1, 2, INIT_POINTER_VALUE]],
            [[0, 2, 2, INIT_POINTER_VALUE], [1, 1, 0, INIT_POINTER_VALUE], [2, 0, 1, INIT_POINTER_VALUE]],
        ]

        assert hyps.timestamps.tolist() == [
            [[0, 2, 7, 0], [3, 7, 3, 0], [1, 4, 7, 0]],
            [[2, 4, 5, 0], [3, 4, 4, 0], [4, 3, 6, 0]],
        ]

    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    def test_ctc_instantiate(self, device: torch.device):
        _ = BatchedBeamHyps(
            batch_size=2, beam_size=3, init_length=4, device=device, blank_index=1024, model_type='ctc'
        )

    @pytest.mark.unit
    @pytest.mark.parametrize("batch_size", [-1, 0])
    def test_ctc_instantiate_incorrect_batch_size(self, batch_size: Literal[-1] | Literal[0]):
        with pytest.raises(ValueError):
            _ = BatchedBeamHyps(batch_size=batch_size, beam_size=4, init_length=3, blank_index=1024, model_type='ctc')

    @pytest.mark.unit
    @pytest.mark.parametrize("beam_size", [-1, 0])
    def test_ctc_instantiate_incorrect_beam_size(self, beam_size: Literal[-1] | Literal[0]):
        with pytest.raises(ValueError):
            _ = BatchedBeamHyps(batch_size=2, beam_size=beam_size, init_length=3, blank_index=1024, model_type='ctc')

    @pytest.mark.unit
    @pytest.mark.parametrize("init_length", [-1, 0])
    def test_ctc_instantiate_incorrect_init_length(self, init_length: Literal[-1] | Literal[0]):
        with pytest.raises(ValueError):
            _ = BatchedBeamHyps(batch_size=1, beam_size=4, init_length=init_length, blank_index=1024)

    @pytest.mark.unit
    @pytest.mark.parametrize("y", [torch.tensor([1, 1024, 1024, 2, 2, 1024, 2, 3, 3, 1024, 3, 2, 2, 2])])
    def test_ctc_create_fold_consecutive_mask(self, y: torch.Tensor):
        batched_hyps = BatchedBeamHyps(batch_size=1, beam_size=4, init_length=30, blank_index=1024, model_type='ctc')
        mask = batched_hyps._create_fold_consecutive_mask(transcript=y)

        assert y[mask].tolist() == [1, 2, 2, 3, 3, 2]

    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    def test_ctc_add_results(self, device: torch.device):
        # batch of size 2, add label for first utterance
        hyps = BatchedBeamHyps(
            batch_size=2, beam_size=3, init_length=1, device=device, blank_index=1024, model_type='ctc'
        )
        assert hyps._max_length == 1
        hyps.add_results_(
            next_indices=torch.tensor([[0, 1, 2], [0, 1, 2]], device=device),
            next_labels=torch.tensor([[0, 1024, 1], [2, 1024, 1024]], device=device),
            next_hyps_prob=torch.tensor([[0.5, 0.6, 0.8], [0.1, 0.2, 0.3]], device=device),
        )
        assert hyps._max_length == 2
        assert hyps.current_lengths_nb.tolist() == [[1, 0, 1], [1, 0, 0]]
        assert hyps.current_lengths_wb.tolist() == [[1, 1, 1], [1, 1, 1]]
        assert_nested_lists_approx(actual=hyps.scores.tolist(), expected=[[0.5, 0.6, 0.8], [0.1, 0.2, 0.3]])
        assert hyps.transcript_wb.tolist() == [
            [[0, NON_EXISTENT_LABEL_VALUE], [1024, NON_EXISTENT_LABEL_VALUE], [1, NON_EXISTENT_LABEL_VALUE]],
            [[2, NON_EXISTENT_LABEL_VALUE], [1024, NON_EXISTENT_LABEL_VALUE], [1024, NON_EXISTENT_LABEL_VALUE]],
        ]
        assert hyps.transcript_wb_prev_ptr.tolist() == [
            [[0, INIT_POINTER_VALUE], [1, INIT_POINTER_VALUE], [2, INIT_POINTER_VALUE]],
            [[0, INIT_POINTER_VALUE], [1, INIT_POINTER_VALUE], [2, INIT_POINTER_VALUE]],
        ]
        assert hyps.timestamps.tolist() == [
            [[0, 1], [0, 1], [0, 1]],
            [[0, 1], [0, 1], [0, 1]],
        ]
        assert hyps.last_label.tolist() == [
            [0, 1024, 1],
            [2, 1024, 1024],
        ]

    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    def test_rnnt_add_multiple_results(self, device: torch.device):
        hyps = BatchedBeamHyps(
            batch_size=2, beam_size=3, init_length=1, device=device, blank_index=1024, model_type='ctc'
        )
        assert hyps._max_length == 1

        hyps.add_results_(
            next_indices=torch.tensor([[0, 1, 2], [0, 1, 2]], device=device),
            next_labels=torch.tensor([[0, 1024, 1], [2, 1024, 1024]], device=device),
            next_hyps_prob=torch.tensor([[0.5, 0.6, 0.8], [0.1, 0.2, 0.3]], device=device),
        )

        hyps.add_results_(
            next_indices=torch.tensor([[0, 1, 1], [2, 1, 0]], device=device),
            next_labels=torch.tensor([[3, 4, 1024], [5, 1024, 6]], device=device),
            next_hyps_prob=torch.tensor([[0.3, 0.2, 0.1], [0.4, 0.5, 0.6]], device=device),
        )

        assert hyps._max_length == 4
        assert hyps.current_lengths_nb.tolist() == [[2, 1, 0], [1, 0, 2]]
        assert hyps.current_lengths_wb.tolist() == [[2, 2, 2], [2, 2, 2]]
        assert_nested_lists_approx(actual=hyps.scores.tolist(), expected=[[0.3, 0.2, 0.1], [0.4, 0.5, 0.6]])
        assert hyps.transcript_wb.tolist() == [
            [
                [0, 3, NON_EXISTENT_LABEL_VALUE, NON_EXISTENT_LABEL_VALUE],
                [1024, 4, NON_EXISTENT_LABEL_VALUE, NON_EXISTENT_LABEL_VALUE],
                [1, 1024, NON_EXISTENT_LABEL_VALUE, NON_EXISTENT_LABEL_VALUE],
            ],
            [
                [2, 5, NON_EXISTENT_LABEL_VALUE, NON_EXISTENT_LABEL_VALUE],
                [1024, 1024, NON_EXISTENT_LABEL_VALUE, NON_EXISTENT_LABEL_VALUE],
                [1024, 6, NON_EXISTENT_LABEL_VALUE, NON_EXISTENT_LABEL_VALUE],
            ],
        ]
        assert hyps.transcript_wb_prev_ptr.tolist() == [
            [
                [0, 0, INIT_POINTER_VALUE, INIT_POINTER_VALUE],
                [1, 1, INIT_POINTER_VALUE, INIT_POINTER_VALUE],
                [2, 1, INIT_POINTER_VALUE, INIT_POINTER_VALUE],
            ],
            [
                [0, 2, INIT_POINTER_VALUE, INIT_POINTER_VALUE],
                [1, 1, INIT_POINTER_VALUE, INIT_POINTER_VALUE],
                [2, 0, INIT_POINTER_VALUE, INIT_POINTER_VALUE],
            ],
        ]
        assert hyps.timestamps.tolist() == [
            [
                [0, 1, 2, 3],
                [0, 1, 2, 3],
                [0, 1, 2, 3],
            ],
            [
                [0, 1, 2, 3],
                [0, 1, 2, 3],
                [0, 1, 2, 3],
            ],
        ]
        assert hyps.last_label.tolist() == [[3, 4, 1024], [5, 1024, 6]]

    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    def test_rnnt_add_with_invalid_results(self, device: torch.device):
        hyps = BatchedBeamHyps(
            batch_size=2, beam_size=3, init_length=1, device=device, blank_index=1024, model_type='ctc'
        )
        assert hyps._max_length == 1

        hyps.add_results_(
            next_indices=torch.tensor([[0, 1, 2], [0, 1, 2]], device=device),
            next_labels=torch.tensor([[0, 1024, 1], [2, 1024, 1024]], device=device),
            next_hyps_prob=torch.tensor([[0.5, 0.6, 0.8], [0.1, 0.2, 0.3]], device=device),
        )

        hyps.add_results_(
            next_indices=torch.tensor([[0, 1, 1], [2, 1, 0]], device=device),
            next_labels=torch.tensor([[3, 4, 1024], [5, 1024, 6]], device=device),
            next_hyps_prob=torch.tensor([[0.3, 0.2, 0.1], [0.4, 0.5, 0.6]], device=device),
        )

        hyps.add_results_(
            next_indices=torch.tensor([[1, 0, 2], [2, 0, 1]], device=device),
            next_labels=torch.tensor([[-1, 7, 8], [10, -1, 9]], device=device),
            next_hyps_prob=torch.tensor([[0.35, 0.4, 0.1], [0.4, 0.55, 0.6]], device=device),
        )

        assert hyps._max_length == 4
        assert hyps.current_lengths_nb.tolist() == [[1, 3, 1], [3, 1, 1]]
        assert hyps.current_lengths_wb.tolist() == [[3, 3, 3], [3, 3, 3]]
        assert_nested_lists_approx(actual=hyps.scores.tolist(), expected=[[0.35, 0.4, 0.1], [0.4, 0.55, 0.6]])
        assert hyps.transcript_wb.tolist() == [
            [
                [0, 3, -1, NON_EXISTENT_LABEL_VALUE],
                [1024, 4, 7, NON_EXISTENT_LABEL_VALUE],
                [1, 1024, 8, NON_EXISTENT_LABEL_VALUE],
            ],
            [
                [2, 5, 10, NON_EXISTENT_LABEL_VALUE],
                [1024, 1024, -1, NON_EXISTENT_LABEL_VALUE],
                [1024, 6, 9, NON_EXISTENT_LABEL_VALUE],
            ],
        ]
        assert hyps.transcript_wb_prev_ptr.tolist() == [
            [[0, 0, 1, INIT_POINTER_VALUE], [1, 1, 0, INIT_POINTER_VALUE], [2, 1, 2, INIT_POINTER_VALUE]],
            [[0, 2, 2, INIT_POINTER_VALUE], [1, 1, 0, INIT_POINTER_VALUE], [2, 0, 1, INIT_POINTER_VALUE]],
        ]
        assert hyps.timestamps.tolist() == [
            [
                [0, 1, 2, 3],
                [0, 1, 2, 3],
                [0, 1, 2, 3],
            ],
            [
                [0, 1, 2, 3],
                [0, 1, 2, 3],
                [0, 1, 2, 3],
            ],
        ]
        assert hyps.last_label.tolist() == [
            [4, 7, 8],
            [10, 5, 9],
        ]


class TestConvertToHypotheses:
    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    def test_rnnt_flatten_sort(self, device: torch.device):
        hyps = BatchedBeamHyps(batch_size=2, beam_size=3, init_length=1, device=device, blank_index=1024)

        hyps.add_results_(
            next_indices=torch.tensor([[0, 1, 2], [0, 1, 2]], device=device),
            next_labels=torch.tensor([[0, 1024, 1], [2, 1024, 1024]], device=device),
            next_hyps_prob=torch.tensor([[0.5, 0.6, 0.8], [0.1, 0.2, 0.3]], device=device),
        )

        hyps.add_results_(
            next_indices=torch.tensor([[0, 1, 1], [2, 1, 0]], device=device),
            next_labels=torch.tensor([[3, 4, 1024], [5, 1024, 6]], device=device),
            next_hyps_prob=torch.tensor([[0.3, 0.2, 0.1], [0.4, 0.5, 0.6]], device=device),
        )

        hyps.add_results_(
            next_indices=torch.tensor([[1, 0, 2], [2, 0, 1]], device=device),
            next_labels=torch.tensor([[-1, 7, 8], [10, -1, 9]], device=device),
            next_hyps_prob=torch.tensor([[0.35, 0.4, 0.1], [0.4, 0.55, 0.6]], device=device),
        )
        hyps.flatten_sort_(score_norm=False)

        assert hyps.current_lengths_nb.tolist() == [[3, 1, 1], [1, 1, 3]]
        assert hyps.current_lengths_wb.tolist() == [[3, 3, 3], [3, 3, 3]]
        assert_nested_lists_approx(actual=hyps.scores.tolist(), expected=[[0.4, 0.35, 0.1], [0.6, 0.55, 0.4]])
        assert hyps.transcript_wb.tolist() == [
            [
                [0, 3, 7, NON_EXISTENT_LABEL_VALUE],
                [1024, 4, -1, NON_EXISTENT_LABEL_VALUE],
                [1024, 1024, 8, NON_EXISTENT_LABEL_VALUE],
            ],
            [
                [1024, 1024, 9, NON_EXISTENT_LABEL_VALUE],
                [1024, 5, -1, NON_EXISTENT_LABEL_VALUE],
                [2, 6, 10, NON_EXISTENT_LABEL_VALUE],
            ],
        ]
        assert hyps.transcript_wb_prev_ptr.tolist() == [
            [[0, 0, 0, INIT_POINTER_VALUE], [1, 1, 1, INIT_POINTER_VALUE], [2, 2, 2, INIT_POINTER_VALUE]],
            [[0, 0, 0, INIT_POINTER_VALUE], [1, 1, 1, INIT_POINTER_VALUE], [2, 2, 2, INIT_POINTER_VALUE]],
        ]
        assert hyps.timestamps.tolist() == [
            [
                [0, 0, 0, 0],
                [1, 1, 1, 0],
                [1, 2, 2, 0],
            ],
            [
                [1, 2, 2, 0],
                [1, 1, 1, 0],
                [0, 0, 0, 0],
            ],
        ]
        assert hyps.next_timestamp.tolist() == [
            [0, 1, 2],
            [2, 1, 0],
        ]

    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    def test_rnnt_flatten_sort_norm(self, device: torch.device):
        hyps = BatchedBeamHyps(batch_size=2, beam_size=3, init_length=1, device=device, blank_index=1024)

        hyps.add_results_(
            next_indices=torch.tensor([[0, 1, 2], [0, 1, 2]], device=device),
            next_labels=torch.tensor([[0, 1024, 1], [2, 1024, 1024]], device=device),
            next_hyps_prob=torch.tensor([[0.5, 0.6, 0.8], [0.1, 0.2, 0.3]], device=device),
        )

        hyps.add_results_(
            next_indices=torch.tensor([[0, 1, 1], [2, 1, 0]], device=device),
            next_labels=torch.tensor([[3, 4, 1024], [5, 1024, 6]], device=device),
            next_hyps_prob=torch.tensor([[0.3, 0.2, 0.1], [0.4, 0.5, 0.6]], device=device),
        )

        hyps.add_results_(
            next_indices=torch.tensor([[1, 0, 2], [2, 0, 1]], device=device),
            next_labels=torch.tensor([[-1, 7, 8], [10, -1, 9]], device=device),
            next_hyps_prob=torch.tensor([[0.35, 0.4, 0.1], [0.4, 0.55, 0.6]], device=device),
        )

        hyps.flatten_sort_(score_norm=True)

        assert hyps.current_lengths_nb.tolist() == [[1, 3, 1], [1, 1, 3]]
        assert hyps.current_lengths_wb.tolist() == [[3, 3, 3], [3, 3, 3]]
        assert_nested_lists_approx(actual=hyps.scores.tolist(), expected=[[0.35, 0.4, 0.1], [0.6, 0.55, 0.4]])
        assert hyps.transcript_wb.tolist() == [
            [
                [1024, 4, -1, NON_EXISTENT_LABEL_VALUE],
                [0, 3, 7, NON_EXISTENT_LABEL_VALUE],
                [1024, 1024, 8, NON_EXISTENT_LABEL_VALUE],
            ],
            [
                [1024, 1024, 9, NON_EXISTENT_LABEL_VALUE],
                [1024, 5, -1, NON_EXISTENT_LABEL_VALUE],
                [2, 6, 10, NON_EXISTENT_LABEL_VALUE],
            ],
        ]
        assert hyps.transcript_wb_prev_ptr.tolist() == [
            [[0, 0, 0, INIT_POINTER_VALUE], [1, 1, 1, INIT_POINTER_VALUE], [2, 2, 2, INIT_POINTER_VALUE]],
            [[0, 0, 0, INIT_POINTER_VALUE], [1, 1, 1, INIT_POINTER_VALUE], [2, 2, 2, INIT_POINTER_VALUE]],
        ]
        assert hyps.timestamps.tolist() == [
            [
                [1, 1, 1, 0],
                [0, 0, 0, 0],
                [1, 2, 2, 0],
            ],
            [
                [1, 2, 2, 0],
                [1, 1, 1, 0],
                [0, 0, 0, 0],
            ],
        ]
        assert hyps.next_timestamp.tolist() == [
            [1, 0, 2],
            [2, 1, 0],
        ]

    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    def test_rnnt_to_hyps_list(self, device: torch.device):
        hyps = BatchedBeamHyps(batch_size=2, beam_size=3, init_length=1, device=device, blank_index=1024)

        hyps.add_results_(
            next_indices=torch.tensor([[0, 1, 2], [0, 1, 2]], device=device),
            next_labels=torch.tensor([[0, 1024, 1], [2, 1024, 1024]], device=device),
            next_hyps_prob=torch.tensor([[0.5, 0.6, 0.8], [0.1, 0.2, 0.3]], device=device),
        )

        hyps.add_results_(
            next_indices=torch.tensor([[0, 1, 1], [2, 1, 0]], device=device),
            next_labels=torch.tensor([[3, 4, 1024], [5, 1024, 6]], device=device),
            next_hyps_prob=torch.tensor([[0.3, 0.2, 0.1], [0.4, 0.5, 0.6]], device=device),
        )

        hyps.add_results_(
            next_indices=torch.tensor([[1, 0, 2], [2, 0, 1]], device=device),
            next_labels=torch.tensor([[-1, 7, 8], [10, -1, 9]], device=device),
            next_hyps_prob=torch.tensor([[0.3, 0.4, 0.1], [0.4, 0.5, 0.6]], device=device),
        )

        hypotheses = hyps.to_hyps_list(score_norm=False)

        assert type(hypotheses) == list
        assert type(hypotheses[0]) == Hypothesis
        assert type(hypotheses[1]) == Hypothesis

        assert len(hypotheses) == 2

        assert_hyps_sequence_equal(hypotheses[0].y_sequence, [0, 3, 7])
        assert_hyps_sequence_equal(hypotheses[1].y_sequence, [9])

        assert_hyps_timestamps_equal(hypotheses[0].timestamp, [0, 0, 0])
        assert_hyps_timestamps_equal(hypotheses[1].timestamp, [2])

        assert hypotheses[0].score == pytest.approx(0.4)
        assert hypotheses[1].score == pytest.approx(0.6)

    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    def test_rnnt_to_nbest_hyps_list(self, device: torch.device):
        hyps = BatchedBeamHyps(batch_size=2, beam_size=3, init_length=1, device=device, blank_index=1024)

        hyps.add_results_(
            next_indices=torch.tensor([[0, 1, 2], [0, 1, 2]], device=device),
            next_labels=torch.tensor([[0, 1024, 1], [2, 1024, 1024]], device=device),
            next_hyps_prob=torch.tensor([[0.5, 0.6, 0.8], [0.1, 0.2, 0.3]], device=device),
        )

        hyps.add_results_(
            next_indices=torch.tensor([[0, 1, 1], [2, 1, 0]], device=device),
            next_labels=torch.tensor([[3, 4, 1024], [5, 1024, 6]], device=device),
            next_hyps_prob=torch.tensor([[0.3, 0.2, 0.1], [0.4, 0.5, 0.6]], device=device),
        )

        hyps.add_results_(
            next_indices=torch.tensor([[1, 0, 2], [2, 0, 1]], device=device),
            next_labels=torch.tensor([[-1, 7, 8], [10, -1, 9]], device=device),
            next_hyps_prob=torch.tensor([[0.35, 0.4, 0.1], [0.4, 0.55, 0.6]], device=device),
        )

        hypotheses = hyps.to_nbest_hyps_list(score_norm=False)

        assert type(hypotheses) == list
        assert type(hypotheses[0]) == NBestHypotheses
        assert type(hypotheses[1]) == NBestHypotheses

        assert len(hypotheses) == 2
        assert len(hypotheses[0].n_best_hypotheses) == 3
        assert len(hypotheses[1].n_best_hypotheses) == 3

        assert_hyps_sequence_equal(hypotheses[0].n_best_hypotheses[0].y_sequence, [0, 3, 7])
        assert_hyps_sequence_equal(hypotheses[0].n_best_hypotheses[1].y_sequence, [4])
        assert_hyps_sequence_equal(hypotheses[0].n_best_hypotheses[2].y_sequence, [8])
        assert_hyps_sequence_equal(hypotheses[1].n_best_hypotheses[0].y_sequence, [9])
        assert_hyps_sequence_equal(hypotheses[1].n_best_hypotheses[1].y_sequence, [5])
        assert_hyps_sequence_equal(hypotheses[1].n_best_hypotheses[2].y_sequence, [2, 6, 10])

        assert_hyps_timestamps_equal(hypotheses[0].n_best_hypotheses[0].timestamp, [0, 0, 0])
        assert_hyps_timestamps_equal(hypotheses[0].n_best_hypotheses[1].timestamp, [1])
        assert_hyps_timestamps_equal(hypotheses[0].n_best_hypotheses[2].timestamp, [2])
        assert_hyps_timestamps_equal(hypotheses[1].n_best_hypotheses[0].timestamp, [2])
        assert_hyps_timestamps_equal(hypotheses[1].n_best_hypotheses[1].timestamp, [1])
        assert_hyps_timestamps_equal(hypotheses[1].n_best_hypotheses[2].timestamp, [0, 0, 0])

        assert hypotheses[0].n_best_hypotheses[0].score == pytest.approx(0.4)
        assert hypotheses[0].n_best_hypotheses[1].score == pytest.approx(0.35)
        assert hypotheses[0].n_best_hypotheses[2].score == pytest.approx(0.1)
        assert hypotheses[1].n_best_hypotheses[0].score == pytest.approx(0.6)
        assert hypotheses[1].n_best_hypotheses[1].score == pytest.approx(0.55)
        assert hypotheses[1].n_best_hypotheses[2].score == pytest.approx(0.4)

    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    def test_tdt_flatten_sort(self, device: torch.device):
        hyps = BatchedBeamHyps(
            batch_size=2, beam_size=3, init_length=1, device=device, blank_index=1024, model_type='tdt'
        )

        hyps.add_results_(
            next_indices=torch.tensor([[0, 1, 2], [0, 1, 2]], device=device),
            next_labels=torch.tensor([[0, 1024, 1], [2, 1024, 1024]], device=device),
            next_hyps_prob=torch.tensor([[0.5, 0.6, 0.8], [0.1, 0.2, 0.3]], device=device),
            next_label_durations=torch.tensor([[0, 3, 1], [2, 3, 4]], device=device),
        )

        hyps.add_results_(
            next_indices=torch.tensor([[0, 1, 1], [2, 1, 0]], device=device),
            next_labels=torch.tensor([[3, 4, 1024], [5, 1024, 6]], device=device),
            next_hyps_prob=torch.tensor([[0.3, 0.2, 0.1], [0.4, 0.5, 0.6]], device=device),
            next_label_durations=torch.tensor([[2, 4, 1], [0, 1, 1]], device=device),
        )

        hyps.add_results_(
            next_indices=torch.tensor([[1, 0, 2], [2, 0, 1]], device=device),
            next_labels=torch.tensor([[-1, 7, 8], [10, -1, 9]], device=device),
            next_hyps_prob=torch.tensor([[0.35, 0.4, 0.1], [0.4, 0.55, 0.6]], device=device),
            next_label_durations=torch.tensor([[2, 1, 3], [2, 1, 2]], device=device),
        )

        hyps.flatten_sort_(score_norm=False)

        assert hyps.current_lengths_nb.tolist() == [[3, 1, 1], [1, 1, 3]]
        assert hyps.current_lengths_wb.tolist() == [[3, 3, 3], [3, 3, 3]]
        assert_nested_lists_approx(actual=hyps.scores.tolist(), expected=[[0.4, 0.35, 0.1], [0.6, 0.55, 0.4]])
        assert hyps.transcript_wb.tolist() == [
            [
                [0, 3, 7, NON_EXISTENT_LABEL_VALUE],
                [1024, 4, -1, NON_EXISTENT_LABEL_VALUE],
                [1024, 1024, 8, NON_EXISTENT_LABEL_VALUE],
            ],
            [
                [1024, 1024, 9, NON_EXISTENT_LABEL_VALUE],
                [1024, 5, -1, NON_EXISTENT_LABEL_VALUE],
                [2, 6, 10, NON_EXISTENT_LABEL_VALUE],
            ],
        ]
        assert hyps.transcript_wb_prev_ptr.tolist() == [
            [[0, 0, 0, INIT_POINTER_VALUE], [1, 1, 1, INIT_POINTER_VALUE], [2, 2, 2, INIT_POINTER_VALUE]],
            [[0, 0, 0, INIT_POINTER_VALUE], [1, 1, 1, INIT_POINTER_VALUE], [2, 2, 2, INIT_POINTER_VALUE]],
        ]

        assert hyps.timestamps.tolist() == [
            [[0, 2, 3, 0], [3, 7, 7, 0], [3, 4, 7, 0]],
            [[3, 4, 6, 0], [4, 4, 4, 0], [2, 3, 5, 0]],
        ]

    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    def test_tdt_flatten_sort_norm(self, device: torch.device):
        hyps = BatchedBeamHyps(
            batch_size=2, beam_size=3, init_length=1, device=device, blank_index=1024, model_type='tdt'
        )

        hyps.add_results_(
            next_indices=torch.tensor([[0, 1, 2], [0, 1, 2]], device=device),
            next_labels=torch.tensor([[0, 1024, 1], [2, 1024, 1024]], device=device),
            next_hyps_prob=torch.tensor([[0.5, 0.6, 0.8], [0.1, 0.2, 0.3]], device=device),
            next_label_durations=torch.tensor([[0, 3, 1], [2, 3, 4]], device=device),
        )

        hyps.add_results_(
            next_indices=torch.tensor([[0, 1, 1], [2, 1, 0]], device=device),
            next_labels=torch.tensor([[3, 4, 1024], [5, 1024, 6]], device=device),
            next_hyps_prob=torch.tensor([[0.3, 0.2, 0.1], [0.4, 0.5, 0.6]], device=device),
            next_label_durations=torch.tensor([[2, 4, 1], [0, 0, 1]], device=device),
        )

        hyps.add_results_(
            next_indices=torch.tensor([[1, 0, 2], [2, 0, 1]], device=device),
            next_labels=torch.tensor([[-1, 7, 8], [10, -1, 9]], device=device),
            next_hyps_prob=torch.tensor([[0.3, 0.4, 0.1], [0.4, 0.5, 0.6]], device=device),
            next_label_durations=torch.tensor([[2, 1, 3], [2, 1, 2]], device=device),
        )

        hyps.flatten_sort_(score_norm=True)

        assert hyps.current_lengths_nb.tolist() == [[1, 3, 1], [1, 1, 3]]
        assert hyps.current_lengths_wb.tolist() == [[3, 3, 3], [3, 3, 3]]
        assert_nested_lists_approx(actual=hyps.scores.tolist(), expected=[[0.3, 0.4, 0.1], [0.6, 0.5, 0.4]])
        assert hyps.transcript_wb.tolist() == [
            [
                [1024, 4, -1, NON_EXISTENT_LABEL_VALUE],
                [0, 3, 7, NON_EXISTENT_LABEL_VALUE],
                [1024, 1024, 8, NON_EXISTENT_LABEL_VALUE],
            ],
            [
                [1024, 1024, 9, NON_EXISTENT_LABEL_VALUE],
                [1024, 5, -1, NON_EXISTENT_LABEL_VALUE],
                [2, 6, 10, NON_EXISTENT_LABEL_VALUE],
            ],
        ]
        assert hyps.transcript_wb_prev_ptr.tolist() == [
            [[0, 0, 0, INIT_POINTER_VALUE], [1, 1, 1, INIT_POINTER_VALUE], [2, 2, 2, INIT_POINTER_VALUE]],
            [[0, 0, 0, INIT_POINTER_VALUE], [1, 1, 1, INIT_POINTER_VALUE], [2, 2, 2, INIT_POINTER_VALUE]],
        ]

        assert hyps.timestamps.tolist() == [
            [[3, 7, 7, 0], [0, 2, 3, 0], [3, 4, 7, 0]],
            [[3, 3, 5, 0], [4, 4, 4, 0], [2, 3, 5, 0]],
        ]

    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    def test_tdt_to_hyps_list(self, device: torch.device):
        hyps = BatchedBeamHyps(
            batch_size=2, beam_size=3, init_length=1, device=device, blank_index=1024, model_type='tdt'
        )

        hyps.add_results_(
            next_indices=torch.tensor([[0, 1, 2], [0, 1, 2]], device=device),
            next_labels=torch.tensor([[0, 1024, 1], [2, 1024, 1024]], device=device),
            next_hyps_prob=torch.tensor([[0.5, 0.6, 0.8], [0.1, 0.2, 0.3]], device=device),
            next_label_durations=torch.tensor([[0, 3, 1], [2, 3, 4]], device=device),
        )

        hyps.add_results_(
            next_indices=torch.tensor([[0, 1, 1], [2, 1, 0]], device=device),
            next_labels=torch.tensor([[3, 4, 1024], [5, 1024, 6]], device=device),
            next_hyps_prob=torch.tensor([[0.3, 0.2, 0.1], [0.4, 0.5, 0.6]], device=device),
            next_label_durations=torch.tensor([[2, 4, 1], [0, 1, 1]], device=device),
        )

        hyps.add_results_(
            next_indices=torch.tensor([[1, 0, 2], [2, 0, 1]], device=device),
            next_labels=torch.tensor([[-1, 7, 8], [10, -1, 9]], device=device),
            next_hyps_prob=torch.tensor([[0.35, 0.4, 0.1], [0.4, 0.55, 0.6]], device=device),
            next_label_durations=torch.tensor([[2, 1, 3], [2, 1, 2]], device=device),
        )

        hypotheses = hyps.to_hyps_list(score_norm=False)

        assert type(hypotheses) == list
        assert type(hypotheses[0]) == Hypothesis
        assert type(hypotheses[1]) == Hypothesis

        assert len(hypotheses) == 2

        assert_hyps_sequence_equal(hypotheses[0].y_sequence, [0, 3, 7])
        assert_hyps_sequence_equal(hypotheses[1].y_sequence, [9])

        assert_hyps_timestamps_equal(hypotheses[0].timestamp, [0, 2, 3])
        assert_hyps_timestamps_equal(hypotheses[1].timestamp, [6])

        assert hypotheses[0].score == pytest.approx(0.4)
        assert hypotheses[1].score == pytest.approx(0.6)

    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    def test_tdt_to_nbest_hyps_list(self, device: torch.device):
        hyps = BatchedBeamHyps(
            batch_size=2, beam_size=3, init_length=1, device=device, blank_index=1024, model_type='tdt'
        )

        hyps.add_results_(
            next_indices=torch.tensor([[0, 1, 2], [0, 1, 2]], device=device),
            next_labels=torch.tensor([[0, 1024, 1], [2, 1024, 1024]], device=device),
            next_hyps_prob=torch.tensor([[0.5, 0.6, 0.8], [0.1, 0.2, 0.3]], device=device),
            next_label_durations=torch.tensor([[0, 3, 1], [2, 3, 4]], device=device),
        )

        hyps.add_results_(
            next_indices=torch.tensor([[0, 1, 1], [2, 1, 0]], device=device),
            next_labels=torch.tensor([[3, 4, 1024], [5, 1024, 6]], device=device),
            next_hyps_prob=torch.tensor([[0.3, 0.2, 0.1], [0.4, 0.5, 0.6]], device=device),
            next_label_durations=torch.tensor([[2, 4, 1], [0, 1, 1]], device=device),
        )

        hyps.add_results_(
            next_indices=torch.tensor([[1, 0, 2], [2, 0, 1]], device=device),
            next_labels=torch.tensor([[-1, 7, 8], [10, -1, 9]], device=device),
            next_hyps_prob=torch.tensor([[0.35, 0.4, 0.1], [0.4, 0.55, 0.6]], device=device),
            next_label_durations=torch.tensor([[2, 1, 3], [2, 1, 2]], device=device),
        )

        hypotheses = hyps.to_nbest_hyps_list(score_norm=False)

        assert type(hypotheses) == list
        assert type(hypotheses[0]) == NBestHypotheses
        assert type(hypotheses[1]) == NBestHypotheses

        assert len(hypotheses) == 2
        assert len(hypotheses[0].n_best_hypotheses) == 3
        assert len(hypotheses[1].n_best_hypotheses) == 3

        assert_hyps_sequence_equal(hypotheses[0].n_best_hypotheses[0].y_sequence, [0, 3, 7])
        assert_hyps_sequence_equal(hypotheses[0].n_best_hypotheses[1].y_sequence, [4])
        assert_hyps_sequence_equal(hypotheses[0].n_best_hypotheses[2].y_sequence, [8])
        assert_hyps_sequence_equal(hypotheses[1].n_best_hypotheses[0].y_sequence, [9])
        assert_hyps_sequence_equal(hypotheses[1].n_best_hypotheses[1].y_sequence, [5])
        assert_hyps_sequence_equal(hypotheses[1].n_best_hypotheses[2].y_sequence, [2, 6, 10])

        assert_hyps_timestamps_equal(hypotheses[0].n_best_hypotheses[0].timestamp, [0, 2, 3])
        assert_hyps_timestamps_equal(hypotheses[0].n_best_hypotheses[1].timestamp, [7])
        assert_hyps_timestamps_equal(hypotheses[0].n_best_hypotheses[2].timestamp, [7])
        assert_hyps_timestamps_equal(hypotheses[1].n_best_hypotheses[0].timestamp, [6])
        assert_hyps_timestamps_equal(hypotheses[1].n_best_hypotheses[1].timestamp, [4])
        assert_hyps_timestamps_equal(hypotheses[1].n_best_hypotheses[2].timestamp, [2, 3, 5])

        assert hypotheses[0].n_best_hypotheses[0].score == pytest.approx(0.4)
        assert hypotheses[0].n_best_hypotheses[1].score == pytest.approx(0.35)
        assert hypotheses[0].n_best_hypotheses[2].score == pytest.approx(0.1)
        assert hypotheses[1].n_best_hypotheses[0].score == pytest.approx(0.6)
        assert hypotheses[1].n_best_hypotheses[1].score == pytest.approx(0.55)
        assert hypotheses[1].n_best_hypotheses[2].score == pytest.approx(0.4)

    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    def test_ctc_flatten_sort(self, device: torch.device):
        hyps = BatchedBeamHyps(
            batch_size=2, beam_size=3, init_length=1, device=device, blank_index=1024, model_type='ctc'
        )

        hyps.add_results_(
            next_indices=torch.tensor([[0, 1, 2], [0, 1, 2]], device=device),
            next_labels=torch.tensor([[3, 1024, 1], [2, 1024, 1024]], device=device),
            next_hyps_prob=torch.tensor([[0.5, 0.6, 0.8], [0.1, 0.2, 0.3]], device=device),
        )

        hyps.add_results_(
            next_indices=torch.tensor([[0, 1, 1], [2, 1, 0]], device=device),
            next_labels=torch.tensor([[3, 4, 1024], [5, 1024, 6]], device=device),
            next_hyps_prob=torch.tensor([[0.3, 0.2, 0.1], [0.4, 0.5, 0.6]], device=device),
        )

        hyps.add_results_(
            next_indices=torch.tensor([[1, 0, 2], [2, 0, 1]], device=device),
            next_labels=torch.tensor([[-1, 7, 8], [2, -1, 9]], device=device),
            next_hyps_prob=torch.tensor([[0.35, 0.4, 0.1], [0.4, 0.55, 0.6]], device=device),
        )
        hyps.flatten_sort_(score_norm=False)

        assert hyps.current_lengths_nb.tolist() == [[2, 1, 1], [1, 1, 3]]
        assert hyps.current_lengths_wb.tolist() == [[3, 3, 3], [3, 3, 3]]
        assert_nested_lists_approx(actual=hyps.scores.tolist(), expected=[[0.4, 0.35, 0.1], [0.6, 0.55, 0.4]])
        assert hyps.transcript_wb.tolist() == [
            [
                [3, 3, 7, NON_EXISTENT_LABEL_VALUE],
                [1024, 4, -1, NON_EXISTENT_LABEL_VALUE],
                [1024, 1024, 8, NON_EXISTENT_LABEL_VALUE],
            ],
            [
                [1024, 1024, 9, NON_EXISTENT_LABEL_VALUE],
                [1024, 5, -1, NON_EXISTENT_LABEL_VALUE],
                [2, 6, 2, NON_EXISTENT_LABEL_VALUE],
            ],
        ]
        assert hyps.transcript_wb_prev_ptr.tolist() == [
            [[0, 0, 0, INIT_POINTER_VALUE], [1, 1, 1, INIT_POINTER_VALUE], [2, 2, 2, INIT_POINTER_VALUE]],
            [[0, 0, 0, INIT_POINTER_VALUE], [1, 1, 1, INIT_POINTER_VALUE], [2, 2, 2, INIT_POINTER_VALUE]],
        ]
        assert hyps.timestamps.tolist() == [
            [
                [0, 1, 2, 3],
                [0, 1, 2, 3],
                [0, 1, 2, 3],
            ],
            [
                [0, 1, 2, 3],
                [0, 1, 2, 3],
                [0, 1, 2, 3],
            ],
        ]
        assert hyps.last_label.tolist() == [
            [7, 4, 8],
            [9, 5, 2],
        ]

    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    def test_ctc_to_hyps_list(self, device: torch.device):
        hyps = BatchedBeamHyps(
            batch_size=2, beam_size=3, init_length=1, device=device, blank_index=1024, model_type='ctc'
        )

        hyps.add_results_(
            next_indices=torch.tensor([[0, 1, 2], [0, 1, 2]], device=device),
            next_labels=torch.tensor([[3, 1024, 1], [2, 1024, 1024]], device=device),
            next_hyps_prob=torch.tensor([[0.5, 0.6, 0.8], [0.1, 0.2, 0.3]], device=device),
        )

        hyps.add_results_(
            next_indices=torch.tensor([[0, 1, 1], [2, 1, 0]], device=device),
            next_labels=torch.tensor([[3, 4, 1024], [5, 1024, 6]], device=device),
            next_hyps_prob=torch.tensor([[0.3, 0.2, 0.1], [0.4, 0.5, 0.6]], device=device),
        )

        hyps.add_results_(
            next_indices=torch.tensor([[1, 0, 2], [2, 0, 1]], device=device),
            next_labels=torch.tensor([[-1, 7, 8], [2, -1, 9]], device=device),
            next_hyps_prob=torch.tensor([[0.35, 0.4, 0.1], [0.4, 0.55, 0.6]], device=device),
        )

        hypotheses = hyps.to_hyps_list(score_norm=False)

        assert type(hypotheses) == list
        assert type(hypotheses[0]) == Hypothesis
        assert type(hypotheses[1]) == Hypothesis

        assert len(hypotheses) == 2

        assert_hyps_sequence_equal(hypotheses[0].y_sequence, [3, 7])
        assert_hyps_sequence_equal(hypotheses[1].y_sequence, [9])

        assert_hyps_timestamps_equal(hypotheses[0].timestamp, [0, 2])
        assert_hyps_timestamps_equal(hypotheses[1].timestamp, [2])

        assert hypotheses[0].score == pytest.approx(0.4)
        assert hypotheses[1].score == pytest.approx(0.6)

    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    def test_ctc_to_nbest_hyps_list(self, device: torch.device):
        hyps = BatchedBeamHyps(
            batch_size=2, beam_size=3, init_length=1, device=device, blank_index=1024, model_type='ctc'
        )

        hyps.add_results_(
            next_indices=torch.tensor([[0, 1, 2], [0, 1, 2]], device=device),
            next_labels=torch.tensor([[3, 1024, 1], [2, 1024, 1024]], device=device),
            next_hyps_prob=torch.tensor([[0.5, 0.6, 0.8], [0.1, 0.2, 0.3]], device=device),
        )

        hyps.add_results_(
            next_indices=torch.tensor([[0, 1, 1], [2, 1, 0]], device=device),
            next_labels=torch.tensor([[3, 4, 1024], [5, 1024, 6]], device=device),
            next_hyps_prob=torch.tensor([[0.3, 0.2, 0.1], [0.4, 0.5, 0.6]], device=device),
        )

        hyps.add_results_(
            next_indices=torch.tensor([[1, 0, 2], [2, 0, 1]], device=device),
            next_labels=torch.tensor([[-1, 7, 8], [2, -1, 9]], device=device),
            next_hyps_prob=torch.tensor([[0.35, 0.4, 0.1], [0.4, 0.55, 0.6]], device=device),
        )

        hypotheses = hyps.to_nbest_hyps_list(score_norm=False)

        assert type(hypotheses) == list
        assert type(hypotheses[0]) == NBestHypotheses
        assert type(hypotheses[1]) == NBestHypotheses

        assert len(hypotheses) == 2
        assert len(hypotheses[0].n_best_hypotheses) == 3
        assert len(hypotheses[1].n_best_hypotheses) == 3

        assert_hyps_sequence_equal(hypotheses[0].n_best_hypotheses[0].y_sequence, [3, 7])
        assert_hyps_sequence_equal(hypotheses[0].n_best_hypotheses[1].y_sequence, [4])
        assert_hyps_sequence_equal(hypotheses[0].n_best_hypotheses[2].y_sequence, [8])
        assert_hyps_sequence_equal(hypotheses[1].n_best_hypotheses[0].y_sequence, [9])
        assert_hyps_sequence_equal(hypotheses[1].n_best_hypotheses[1].y_sequence, [5])
        assert_hyps_sequence_equal(hypotheses[1].n_best_hypotheses[2].y_sequence, [2, 6, 2])

        assert_hyps_timestamps_equal(hypotheses[0].n_best_hypotheses[0].timestamp, [0, 2])
        assert_hyps_timestamps_equal(hypotheses[0].n_best_hypotheses[1].timestamp, [1])
        assert_hyps_timestamps_equal(hypotheses[0].n_best_hypotheses[2].timestamp, [2])
        assert_hyps_timestamps_equal(hypotheses[1].n_best_hypotheses[0].timestamp, [2])
        assert_hyps_timestamps_equal(hypotheses[1].n_best_hypotheses[1].timestamp, [1])
        assert_hyps_timestamps_equal(hypotheses[1].n_best_hypotheses[2].timestamp, [0, 1, 2])

        assert hypotheses[0].n_best_hypotheses[0].score == pytest.approx(0.4)
        assert hypotheses[0].n_best_hypotheses[1].score == pytest.approx(0.35)
        assert hypotheses[0].n_best_hypotheses[2].score == pytest.approx(0.1)
        assert hypotheses[1].n_best_hypotheses[0].score == pytest.approx(0.6)
        assert hypotheses[1].n_best_hypotheses[1].score == pytest.approx(0.55)
        assert hypotheses[1].n_best_hypotheses[2].score == pytest.approx(0.4)
