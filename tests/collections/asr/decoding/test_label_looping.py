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

from types import SimpleNamespace

import torch

from nemo.collections.asr.parts.submodules.transducer_decoding.rnnt_label_looping import (
    GreedyBatchedRNNTLabelLoopingComputer,
)


class TestLabelLooping:
    """
    Unit tests for label-looping implementation.
    Note that most of the tests are end-to-end, located in other test modules
    """

    def test_wind_selection(self):
        mock_computer = SimpleNamespace(_blank_index=4, window_size=4)
        logits = torch.tensor(
            [
                # element 1 - non-blank (label 2)
                [[0, 0, 0, 0, 10], [0, 0, 5, 0, 0], [0, 0, 0, 10, 0], [0, 0, 0, 0, 10]],
                # element 0 - non-blank (label 3)
                [[0, 0, 0, 7, 0], [0, 0, 0, 0, 10], [0, 10, 0, 0, 0], [0, 0, 0, 0, 10]],
                # all elements - blank
                [[0, 0, 0, 0, 2], [0, 0, 0, 0, 11], [0, 0, 0, 0, 7], [0, 0, 0, 0, 10]],
            ],
            dtype=torch.float32,
        )
        assert logits.shape == (3, 4, 5)  # Batch x Window x Vocab
        selected_window_idx, best_scores, best_labels = (
            GreedyBatchedRNNTLabelLoopingComputer._wind_selection_stateless(mock_computer, logits)
        )
        assert selected_window_idx.shape == (3,)
        assert best_labels.shape == (3,)
        assert best_scores.shape == (3,)
        assert selected_window_idx.tolist() == [1, 0, 3]
        assert best_labels.tolist() == [2, 3, 4]
        # note that for the last element, we expect 10 - logit for last blank
        assert torch.allclose(best_scores, torch.tensor([5, 7, 10], dtype=torch.float32))
