# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import itertools
import pytest
import torch

from nemo.collections.asr.parts.utils.asr_multispeaker_utils import (
    find_best_permutation,
    find_first_nonzero,
    get_ats_targets,
    get_hidden_length_from_sample_length,
    get_pil_targets,
    reconstruct_labels,
)


def reconstruct_labels_forloop(labels: torch.Tensor, batch_perm_inds: torch.Tensor) -> torch.Tensor:
    """
    This is a for-loop implementation of reconstruct_labels built for testing purposes.
    """
    # Expanding batch_perm_inds to align with labels dimensions
    batch_size, num_frames, num_speakers = labels.shape
    batch_perm_inds_exp = batch_perm_inds.unsqueeze(1).expand(-1, num_frames, -1)

    # Reconstructing the labels using advanced indexing
    reconstructed_labels = torch.gather(labels, 2, batch_perm_inds_exp)
    return reconstructed_labels


class TestSortingUtils:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "mat, max_cap_val, thres, expected",
        [
            # Test 1: Basic case with clear first nonzero values
            (torch.tensor([[0.1, 0.6, 0.0], [0.0, 0.0, 0.9]]), -1, 0.5, torch.tensor([1, 2])),
            # Test 2: All elements are below threshold
            (torch.tensor([[0.1, 0.2], [0.3, 0.4]]), -1, 0.5, torch.tensor([-1, -1])),
            # Test 3: No nonzero elements, should return max_cap_val (-1)
            (torch.tensor([[0.0, 0.0], [0.0, 0.0]]), -1, 0.5, torch.tensor([-1, -1])),
            # Test 4: Large matrix with mixed values, some rows with all values below threshold
            (torch.tensor([[0.1, 0.7, 0.3], [0.0, 0.0, 0.9], [0.5, 0.6, 0.7]]), -1, 0.5, torch.tensor([1, 2, 0])),
            # Test 5: Single row matrix
            (torch.tensor([[0.0, 0.0, 0.6]]), -1, 0.5, torch.tensor([2])),
            # Test 6: Single column matrix
            (torch.tensor([[0.1], [0.6], [0.0]]), -1, 0.5, torch.tensor([-1, 0, -1])),
            # Test 7: One element matrix
            (torch.tensor([[0.501]]), -1, 0.5, torch.tensor([0], dtype=torch.long)),
            # Test 8: All values are zero, should return max_cap_val
            (torch.tensor([[0.0, 0.0], [0.0, 0.0]]), -1, 0.5, torch.tensor([-1, -1])),
            # Test 9: All values are above threshold
            (torch.tensor([[0.6, 0.7], [0.8, 0.9]]), -1, 0.5, torch.tensor([0, 0])),
            # Test 10: Custom max_cap_val different from default
            (torch.tensor([[0.0, 0.0], [0.0, 0.0]]), 99, 0.5, torch.tensor([99, 99])),
            # Test 11: Matrix with 101 columns, first nonzero value is towards the end
            (torch.cat([torch.zeros(1, 100), torch.ones(1, 1)], dim=1), -1, 0.5, torch.tensor([100])),
            # Test 12: Matrix with 1000 columns, all below threshold except one near the middle
            (
                torch.cat([torch.zeros(1, 499), torch.tensor([[0.6]]), torch.zeros(1, 500)], dim=1),
                -1,
                0.5,
                torch.tensor([499]),
            ),
        ],
    )
    def test_find_first_nonzero(self, mat, max_cap_val, thres, expected):
        result = find_first_nonzero(mat, max_cap_val, thres)
        assert torch.equal(result, expected), f"Expected {expected} but got {result}"

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "match_score, speaker_permutations, expected",
        [
            # Test 1: Simple case with batch size 1, clear best match
            (
                torch.tensor([[0.1, 0.9, 0.2]]),  # match_score (batch_size=1, num_permutations=3)
                torch.tensor([[0, 1], [1, 0], [0, 1]]),  # speaker_permutations (num_permutations=3, num_speakers=2)
                torch.tensor([[1, 0]]),  # expected best permutation for the batch
            ),
            # Test 2: Batch size 2, different best matches for each batch
            (
                torch.tensor([[0.5, 0.3, 0.7], [0.2, 0.6, 0.4]]),  # match_score (batch_size=2, num_permutations=3)
                torch.tensor([[0, 1], [1, 0], [0, 1]]),  # speaker_permutations
                torch.tensor([[0, 1], [1, 0]]),  # expected best permutations
            ),
            # Test 3: Larger number of speakers and permutations
            (
                torch.tensor(
                    [[0.1, 0.4, 0.9, 0.5], [0.6, 0.3, 0.7, 0.2]]
                ),  # match_score (batch_size=2, num_permutations=4)
                torch.tensor(
                    [[0, 1, 2], [1, 0, 2], [2, 1, 0], [1, 2, 0]]
                ),  # speaker_permutations (num_permutations=4, num_speakers=3)
                torch.tensor([[2, 1, 0], [2, 1, 0]]),  # expected best permutations
            ),
            # Test 4: All match scores are the same, should pick the first permutation (argmax behavior)
            (
                torch.tensor([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]),  # equal match_score across permutations
                torch.tensor([[0, 1], [1, 0], [0, 1]]),  # speaker_permutations
                torch.tensor([[0, 1], [0, 1]]),  # first permutation is chosen as tie-breaker
            ),
            # Test 5: Single speaker case (num_speakers = 1)
            (
                torch.tensor([[0.8, 0.2]]),  # match_score (batch_size=1, num_permutations=2)
                torch.tensor([[0], [0]]),  # speaker_permutations (num_permutations=2, num_speakers=1)
                torch.tensor([[0]]),  # expected best permutation
            ),
            # Test 6: Batch size 3, varying permutations
            (
                torch.tensor([[0.3, 0.6], [0.4, 0.1], [0.2, 0.7]]),  # match_score (batch_size=3, num_permutations=2)
                torch.tensor([[0, 1], [1, 0]]),  # speaker_permutations
                torch.tensor([[1, 0], [0, 1], [1, 0]]),  # expected best permutations for each batch
            ),
        ],
    )
    def test_find_best_permutation(self, match_score, speaker_permutations, expected):
        result = find_best_permutation(match_score, speaker_permutations)
        assert torch.equal(result, expected), f"Expected {expected} but got {result}"

    @pytest.mark.parametrize(
        "batch_size, num_frames, num_speakers",
        [
            (2, 4, 3),  # Original test case
            (3, 5, 2),  # More frames and speakers
            (1, 6, 4),  # Single batch with more frames and speakers
            (5, 3, 5),  # More batch size with equal frames and speakers
        ],
    )
    def test_reconstruct_labels_with_forloop_ver(self, batch_size, num_frames, num_speakers):
        # Generate random labels and batch_perm_inds tensor for testing
        labels = torch.rand(batch_size, num_frames, num_speakers)
        batch_perm_inds = torch.stack([torch.randperm(num_speakers) for _ in range(batch_size)])

        # Call both functions
        result_matrix = reconstruct_labels(labels, batch_perm_inds)
        result_forloop = reconstruct_labels_forloop(labels, batch_perm_inds)

        # Assert that both methods return the same result
        assert torch.allclose(result_matrix, result_forloop), "The results are not equal!"

    @pytest.mark.parametrize(
        "labels, batch_perm_inds, expected_output",
        [
            # Example 1: Small batch size with a few frames and speakers
            (
                torch.tensor(
                    [
                        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],  # First batch
                        [[0.9, 0.8, 0.7], [0.6, 0.5, 0.4], [0.3, 0.2, 0.1]],  # Second batch
                    ]
                ),
                torch.tensor([[2, 0, 1], [1, 2, 0]]),
                torch.tensor(
                    [
                        [[0.3, 0.1, 0.2], [0.6, 0.4, 0.5], [0.9, 0.7, 0.8]],  # First batch reconstructed
                        [[0.8, 0.7, 0.9], [0.5, 0.4, 0.6], [0.2, 0.1, 0.3]],  # Second batch reconstructed
                    ]
                ),
            ),
            # Example 2: batch_size = 1 with more frames and speakers
            (
                torch.tensor(
                    [[[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.9, 1.0, 1.1, 1.2], [1.3, 1.4, 1.5, 1.6]]]
                ),
                torch.tensor([[3, 0, 1, 2]]),
                torch.tensor(
                    [[[0.4, 0.1, 0.2, 0.3], [0.8, 0.5, 0.6, 0.7], [1.2, 0.9, 1.0, 1.1], [1.6, 1.3, 1.4, 1.5]]]
                ),
            ),
            # Example 3: Larger batch size with fewer frames and speakers
            (
                torch.tensor(
                    [
                        [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],  # First batch
                        [[0.7, 0.8], [0.9, 1.0], [1.1, 1.2]],  # Second batch
                        [[1.3, 1.4], [1.5, 1.6], [1.7, 1.8]],  # Third batch
                        [[1.9, 2.0], [2.1, 2.2], [2.3, 2.4]],  # Fourth batch
                    ]
                ),
                torch.tensor([[1, 0], [0, 1], [1, 0], [0, 1]]),
                torch.tensor(
                    [
                        [[0.2, 0.1], [0.4, 0.3], [0.6, 0.5]],  # First batch reconstructed
                        [[0.7, 0.8], [0.9, 1.0], [1.1, 1.2]],  # Second batch unchanged
                        [[1.4, 1.3], [1.6, 1.5], [1.8, 1.7]],  # Third batch reconstructed
                        [[1.9, 2.0], [2.1, 2.2], [2.3, 2.4]],  # Fourth batch unchanged
                    ]
                ),
            ),
        ],
    )
    def test_reconstruct_labels(self, labels, batch_perm_inds, expected_output):
        # Call the reconstruct_labels function
        result = reconstruct_labels(labels, batch_perm_inds)
        # Assert that the result matches the expected output
        assert torch.allclose(result, expected_output), f"Expected {expected_output}, but got {result}"


class TestTargetGenerators:

    @pytest.mark.parametrize(
        "labels, preds, num_speakers, expected_output",
        [
            # Test 1: Basic case with simple permutations
            (
                torch.tensor(
                    [
                        [[0.9, 0.1, 0.0], [0.1, 0.8, 0.0], [0.0, 0.1, 0.9]],  # Batch 1
                        [[0.0, 0.0, 0.9], [0.0, 0.9, 0.1], [0.9, 0.1, 0.0]],  # Batch 2
                    ]
                ),
                torch.tensor(
                    [
                        [[0.8, 0.2, 0.0], [0.2, 0.7, 0.0], [0.0, 0.1, 0.9]],  # Batch 1
                        [[0.0, 0.0, 0.8], [0.0, 0.8, 0.2], [0.9, 0.1, 0.0]],  # Batch 2
                    ]
                ),
                3,  # Number of speakers
                torch.tensor(
                    [
                        [[0.9, 0.1, 0.0], [0.1, 0.8, 0.0], [0.0, 0.1, 0.9]],  # Expected labels for Batch 1
                        [[0.9, 0.0, 0.0], [0.1, 0.9, 0.0], [0.0, 0.1, 0.9]],  # Expected labels for Batch 2
                    ]
                ),
            ),
            # Test 2: Ambiguous case
            (
                torch.tensor([[[0.9, 0.8, 0.7], [0.2, 0.8, 0.7], [0.2, 0.3, 0.9]]]),  # Labels
                torch.tensor([[[0.6, 0.7, 0.2], [0.9, 0.4, 0.0], [0.1, 0.7, 0.1]]]),  # Preds
                3,  # Number of speakers
                torch.tensor([[[0.8, 0.7, 0.9], [0.8, 0.7, 0.2], [0.3, 0.9, 0.2]]]),  # Expected output
            ),
            # Test 3: Ambiguous case
            (
                torch.tensor([[[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 0, 1], [0, 0, 0, 0]]]),  # Labels
                torch.tensor(
                    [[[0.6, 0.6, 0.1, 0.9], [0.7, 0.7, 0.2, 0.8], [0.4, 0.6, 0.2, 0.7], [0.1, 0.1, 0.1, 0.7]]]
                ),  # Preds
                4,  # Number of speakers
                torch.tensor([[[1, 1, 0, 0], [1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]]]),  # Expected output
            ),
        ],
    )
    def test_get_ats_targets(self, labels, preds, num_speakers, expected_output):
        # Generate all permutations for the given number of speakers
        speaker_inds = list(range(num_speakers))
        speaker_permutations = torch.tensor(list(itertools.permutations(speaker_inds)))

        # Call the function under test
        result = get_ats_targets(labels, preds, speaker_permutations)
        # Assert that the result matches the expected output
        assert torch.allclose(result, expected_output), f"Expected {expected_output}, but got {result}"

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "labels, preds, num_speakers, expected_output",
        [
            # Test 1: Basic case with simple permutations
            (
                torch.tensor(
                    [[[1, 0], [0, 1]], [[1, 0], [0, 1]]]
                ),  # Labels (batch_size=2, num_speakers=2, num_classes=2)
                torch.tensor(
                    [[[1, 0], [0, 1]], [[0, 1], [1, 0]]]
                ),  # Preds (batch_size=2, num_speakers=2, num_classes=2)
                2,  # Number of speakers
                torch.tensor([[[1, 0], [0, 1]], [[0, 1], [1, 0]]]),  # expected max_score_permed_labels
            ),
            # Test 2: Batch size 1 with more complex permutations
            (
                torch.tensor([[[0.8, 0.2], [0.3, 0.7]]]),  # Labels
                torch.tensor([[[0.9, 0.1], [0.2, 0.8]]]),  # Preds
                2,  # Number of speakers
                torch.tensor(
                    [[[0.8, 0.2], [0.3, 0.7]]]
                ),  # expected output (labels remain the same as preds are close)
            ),
            # Test 3: Ambiguous case
            (
                torch.tensor([[[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 0, 1], [0, 0, 0, 0]]]),  # Labels
                torch.tensor(
                    [[[0.61, 0.6, 0.1, 0.9], [0.7, 0.7, 0.2, 0.8], [0.4, 0.6, 0.2, 0.7], [0.1, 0.1, 0.1, 0.7]]]
                ),  # Preds
                4,  # Number of speakers
                torch.tensor([[[1, 0, 0, 1], [1, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0]]]),  # Expected output
            ),
        ],
    )
    def test_get_pil_targets(self, labels, preds, num_speakers, expected_output):
        # Generate all permutations for the given number of speakers
        speaker_inds = list(range(num_speakers))
        speaker_permutations = torch.tensor(list(itertools.permutations(speaker_inds)))

        result = get_pil_targets(labels, preds, speaker_permutations)
        assert torch.equal(result, expected_output), f"Expected {expected_output} but got {result}"


class TestGetHiddenLengthFromSampleLength:
    @pytest.mark.parametrize(
        "num_samples, num_sample_per_mel_frame, num_mel_frame_per_asr_frame, expected_hidden_length",
        [
            (160, 160, 8, 1),
            (1280, 160, 8, 1),
            (0, 160, 8, 0),
            (159, 160, 8, 1),
            (129, 100, 5, 1),
            (300, 150, 3, 1),
        ],
    )
    def test_various_cases(
        self, num_samples, num_sample_per_mel_frame, num_mel_frame_per_asr_frame, expected_hidden_length
    ):
        result = get_hidden_length_from_sample_length(
            num_samples, num_sample_per_mel_frame, num_mel_frame_per_asr_frame
        )
        assert result == expected_hidden_length

    def test_default_parameters(self):
        assert get_hidden_length_from_sample_length(160) == 1
        assert get_hidden_length_from_sample_length(1280) == 1
        assert get_hidden_length_from_sample_length(0) == 0
        assert get_hidden_length_from_sample_length(159) == 1

    def test_edge_cases(self):
        assert get_hidden_length_from_sample_length(159, 160, 8) == 1
        assert get_hidden_length_from_sample_length(160, 160, 8) == 1
        assert get_hidden_length_from_sample_length(161, 160, 8) == 1
        assert get_hidden_length_from_sample_length(1279, 160, 8) == 1

    def test_real_life_examples(self):
        # The samples tried when this function was designed.
        assert get_hidden_length_from_sample_length(160000) == 125
        assert get_hidden_length_from_sample_length(159999) == 125
        assert get_hidden_length_from_sample_length(158720) == 124
        assert get_hidden_length_from_sample_length(158719) == 124

        assert get_hidden_length_from_sample_length(158880) == 125
        assert get_hidden_length_from_sample_length(158879) == 125
        assert get_hidden_length_from_sample_length(1600) == 2
        assert get_hidden_length_from_sample_length(1599) == 2
