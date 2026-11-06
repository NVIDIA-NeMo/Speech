# SPDX-FileCopyrightText: Copyright (c) 2026, Elias Braun. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gc
import weakref

import pytest
import torch

from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis


@pytest.mark.unit
class TestHypothesisTimestampMerge:
    @pytest.mark.parametrize("initial", [[], [11]])
    def test_tensor_values_are_materialized_in_list(self, initial):
        hypothesis = Hypothesis(score=0.0, y_sequence=[], timestamp=list(initial))
        incoming = Hypothesis(score=1.0, y_sequence=[3, 4], timestamp=torch.tensor([1024, 2048]))

        hypothesis.merge_(incoming)

        assert hypothesis.timestamp == initial + [1024, 2048]
        assert all(type(value) is int for value in hypothesis.timestamp)
        assert hypothesis.y_sequence == [3, 4]
        assert hypothesis.score == 1.0
        assert torch.is_tensor(incoming.timestamp)

    def test_timestamp_history_does_not_retain_batch_storage(self):
        hypothesis = Hypothesis(score=0.0, y_sequence=[])
        references = []
        for step in range(4):
            batch = torch.full((2, 8), step + 1024, dtype=torch.long)
            references.append(weakref.ref(batch))
            incoming = Hypothesis(score=0.0, y_sequence=[], timestamp=batch[0, :2])
            hypothesis.merge_(incoming)
            del incoming, batch

        gc.collect()

        assert all(reference() is None for reference in references)
        assert hypothesis.timestamp == [value for step in range(4) for value in [step + 1024] * 2]

    def test_list_values_survive_reuse_of_incoming_storage(self):
        batch = torch.tensor([[1024, 2048], [3072, 4096]])
        incoming = Hypothesis(score=0.0, y_sequence=[], timestamp=batch[0])
        hypothesis = Hypothesis(score=0.0, y_sequence=[])

        hypothesis.merge_(incoming)
        batch.fill_(-1)

        assert hypothesis.timestamp == [1024, 2048]
        assert torch.equal(incoming.timestamp, torch.tensor([-1, -1]))

    def test_list_to_list_merge(self):
        hypothesis = Hypothesis(score=0.0, y_sequence=[], timestamp=[2])
        hypothesis.merge_(Hypothesis(score=0.0, y_sequence=[], timestamp=[3, 4]))
        assert hypothesis.timestamp == [2, 3, 4]

    def test_tensor_to_tensor_merge(self):
        hypothesis = Hypothesis(score=0.0, y_sequence=[], timestamp=torch.tensor([2]))
        hypothesis.merge_(Hypothesis(score=0.0, y_sequence=[], timestamp=torch.tensor([3, 4])))
        assert torch.equal(hypothesis.timestamp, torch.tensor([2, 3, 4]))

    def test_none_adopts_incoming_timestamps(self):
        timestamps = torch.tensor([2, 3])
        hypothesis = Hypothesis(score=0.0, y_sequence=[], timestamp=None)
        hypothesis.merge_(Hypothesis(score=0.0, y_sequence=[], timestamp=timestamps))
        assert hypothesis.timestamp is timestamps

    def test_empty_tensor_preserves_list(self):
        hypothesis = Hypothesis(score=0.0, y_sequence=[], timestamp=[2])
        hypothesis.merge_(Hypothesis(score=0.0, y_sequence=[], timestamp=torch.empty(0, dtype=torch.long)))
        assert hypothesis.timestamp == [2]
