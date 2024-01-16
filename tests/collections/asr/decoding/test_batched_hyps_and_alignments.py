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

from typing import List

import pytest
import torch

from nemo.collections.asr.parts.utils.rnnt_utils import BatchedAlignments, BatchedHyps, batched_hyps_to_hypotheses

DEVICES: List[torch.device] = [torch.device("cpu")]

if torch.cuda.is_available():
    DEVICES.append(torch.device("cuda"))

if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICES.append(torch.device("mps"))


class TestBatchedHyps:
    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    def test_instantiate(self, device: torch.device):
        hyps = BatchedHyps(batch_size=2, init_length=3, device=device)
        assert torch.is_tensor(hyps.timesteps)
        # device: for mps device we need to use `type`, not directly compare
        assert hyps.timesteps.device.type == device.type
        assert hyps.timesteps.shape == (2, 3)

    @pytest.mark.unit
    @pytest.mark.parametrize("batch_size", [-1, 0])
    def test_instantiate_incorrect_batch_size(self, batch_size):
        with pytest.raises(ValueError):
            _ = BatchedHyps(batch_size=batch_size, init_length=3)

    @pytest.mark.unit
    @pytest.mark.parametrize("init_length", [-1, 0])
    def test_instantiate_incorrect_init_length(self, init_length):
        with pytest.raises(ValueError):
            _ = BatchedHyps(batch_size=1, init_length=init_length)

    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    def test_add_results(self, device: torch.device):
        # batch of size 2, add label for first utterance
        hyps = BatchedHyps(batch_size=2, init_length=1, device=device)
        hyps.add_results_(
            active_indices=torch.tensor([0], device=device),
            labels=torch.tensor([5], device=device),
            time_indices=torch.tensor([1], device=device),
            scores=torch.tensor([0.5], device=device),
        )
        assert hyps.current_lengths.tolist() == [1, 0]
        assert hyps.transcript.tolist()[0][:1] == [5]
        assert hyps.timesteps.tolist()[0][:1] == [1]
        assert hyps.scores.tolist() == pytest.approx([0.5, 0.0])
        assert hyps.last_timestep.tolist() == [1, -1]
        assert hyps.last_timestep_lasts.tolist() == [1, 0]

    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    def test_add_multiple_results(self, device: torch.device):
        # batch of size 2, add label for first utterance, then add labels for both utterances
        hyps = BatchedHyps(batch_size=2, init_length=1, device=device)
        hyps.add_results_(
            active_indices=torch.tensor([0], device=device),
            labels=torch.tensor([5], device=device),
            time_indices=torch.tensor([1], device=device),
            scores=torch.tensor([0.5], device=device),
        )
        hyps.add_results_(
            active_indices=torch.tensor([0, 1], device=device),
            labels=torch.tensor([2, 4], device=device),
            time_indices=torch.tensor([1, 2], device=device),
            scores=torch.tensor([1.0, 1.0], device=device),
        )
        assert hyps.current_lengths.tolist() == [2, 1]
        assert hyps.transcript.tolist()[0][:2] == [5, 2]
        assert hyps.transcript.tolist()[1][:1] == [4]
        assert hyps.timesteps.tolist()[0][:2] == [1, 1]
        assert hyps.timesteps.tolist()[1][:1] == [2]
        assert hyps.scores.tolist() == pytest.approx([1.5, 1.0])
        assert hyps.last_timestep.tolist() == [1, 2]
        assert hyps.last_timestep_lasts.tolist() == [2, 1]

    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    def test_torch_jit_compatibility(self, device: torch.device):
        @torch.jit.script
        def hyps_add_wrapper(
            active_indices: torch.Tensor, labels: torch.Tensor, time_indices: torch.Tensor, scores: torch.Tensor
        ):
            hyps = BatchedHyps(batch_size=2, init_length=3, device=active_indices.device)
            hyps.add_results_(active_indices=active_indices, labels=labels, time_indices=time_indices, scores=scores)
            return hyps

        scores = torch.tensor([0.1, 0.1], device=device)
        hyps = hyps_add_wrapper(
            torch.tensor([0, 1], device=device),
            torch.tensor([2, 4], device=device),
            torch.tensor([0, 0], device=device),
            scores,
        )
        assert torch.allclose(hyps.scores, scores)


class TestBatchedAlignments:
    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    def test_instantiate(self, device: torch.device):
        alignments = BatchedAlignments(batch_size=2, logits_dim=7, init_length=3, device=device)
        assert torch.is_tensor(alignments.logits)
        # device: for mps device we need to use `type`, not directly compare
        assert alignments.logits.device.type == device.type
        assert alignments.logits.shape == (2, 3, 7)

    @pytest.mark.unit
    @pytest.mark.parametrize("batch_size", [-1, 0])
    def test_instantiate_incorrect_batch_size(self, batch_size):
        with pytest.raises(ValueError):
            _ = BatchedAlignments(batch_size=batch_size, logits_dim=7, init_length=3)

    @pytest.mark.unit
    @pytest.mark.parametrize("init_length", [-1, 0])
    def test_instantiate_incorrect_init_length(self, init_length):
        with pytest.raises(ValueError):
            _ = BatchedAlignments(batch_size=1, logits_dim=7, init_length=init_length)

    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    def test_add_results(self, device: torch.device):
        # batch of size 2, add label for first utterance
        batch_size = 2
        logits_dim = 7
        sample_logits = torch.rand((batch_size, 1, logits_dim), device=device)
        alignments = BatchedAlignments(batch_size=batch_size, logits_dim=logits_dim, init_length=1, device=device)
        alignments.add_results_(
            active_indices=torch.arange(batch_size, device=device),
            logits=sample_logits[:, 0],
            labels=torch.argmax(sample_logits[:, 0], dim=-1),
            time_indices=torch.tensor([0, 0], device=device),
        )
        assert alignments.current_lengths.tolist() == [1, 1]
        assert torch.allclose(alignments.logits[:, 0], sample_logits[:, 0])
        assert alignments.timesteps[:, 0].tolist() == [0, 0]

    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    def test_add_multiple_results(self, device: torch.device):
        # batch of size 2, add label for first utterance
        batch_size = 2
        seq_length = 5
        logits_dim = 7
        alignments = BatchedAlignments(batch_size=batch_size, logits_dim=logits_dim, init_length=1, device=device)
        sample_logits = torch.rand((batch_size, seq_length, logits_dim), device=device)
        add_logits_mask = torch.rand((batch_size, seq_length), device=device) < 0.6
        for t in range(seq_length):
            alignments.add_results_(
                active_indices=torch.arange(batch_size, device=device)[add_logits_mask[:, t]],
                logits=sample_logits[add_logits_mask[:, t], t],
                labels=torch.argmax(sample_logits[add_logits_mask[:, t], t], dim=-1),
                time_indices=torch.tensor([0, 0], device=device)[add_logits_mask[:, t]],
            )

        assert (alignments.current_lengths == add_logits_mask.sum(dim=-1)).all()
        for i in range(batch_size):
            assert (
                alignments.logits[i, : alignments.current_lengths[i]] == sample_logits[i, add_logits_mask[i]]
            ).all()


class TestConvertToHypotheses:
    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    def test_convert_to_hypotheses(self, device: torch.device):
        hyps = BatchedHyps(batch_size=2, init_length=1, device=device)
        hyps.add_results_(
            active_indices=torch.tensor([0], device=device),
            labels=torch.tensor([5], device=device),
            time_indices=torch.tensor([1], device=device),
            scores=torch.tensor([0.5], device=device),
        )
        hyps.add_results_(
            active_indices=torch.tensor([0, 1], device=device),
            labels=torch.tensor([2, 4], device=device),
            time_indices=torch.tensor([1, 2], device=device),
            scores=torch.tensor([1.0, 1.0], device=device),
        )
        hypotheses = batched_hyps_to_hypotheses(hyps)
        assert (hypotheses[0].y_sequence == torch.tensor([5, 2], device=device)).all()
        assert (hypotheses[1].y_sequence == torch.tensor([4], device=device)).all()
        assert hypotheses[0].score == pytest.approx(1.5)
        assert hypotheses[1].score == pytest.approx(1.0)
        assert (hypotheses[0].timestep == torch.tensor([1, 1], device=device)).all()
        assert (hypotheses[1].timestep == torch.tensor([2], device=device)).all()

    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    def test_convert_to_hypotheses_with_alignments(self, device: torch.device):
        batch_size = 2
        logits_dim = 7
        blank_index = 6
        hyps = BatchedHyps(batch_size=batch_size, init_length=1, device=device)
        alignments = BatchedAlignments(batch_size=batch_size, init_length=1, logits_dim=logits_dim, device=device)
        sample_logits = torch.rand((batch_size, 4, logits_dim), device=device)
        # sequence 0: [[5, blank], [2, blank]] -> [5, 2]
        # sequence 1: [[blank   ], [4, blank]] -> [4]

        # frame 0
        hyps.add_results_(
            active_indices=torch.tensor([0], device=device),
            labels=torch.tensor([5], device=device),
            time_indices=torch.tensor([0], device=device),
            scores=torch.tensor([0.5], device=device),
        )
        alignments.add_results_(
            active_indices=torch.arange(batch_size, device=device),
            logits=sample_logits[:, 0],
            labels=torch.tensor([5, blank_index], device=device),
            time_indices=torch.tensor([0, 0], device=device),
        )
        alignments.add_results_(
            active_indices=torch.tensor([0], device=device),
            logits=sample_logits[:1, 1],
            labels=torch.tensor([blank_index], device=device),
            time_indices=torch.tensor([0], device=device),
        )

        # frame 1
        hyps.add_results_(
            active_indices=torch.arange(batch_size, device=device),
            labels=torch.tensor([2, 4], device=device),
            time_indices=torch.tensor([1, 1], device=device),
            scores=torch.tensor([1.0, 1.0], device=device),
        )
        alignments.add_results_(
            active_indices=torch.arange(batch_size, device=device),
            logits=sample_logits[:, 2],
            labels=torch.tensor([2, 4], device=device),
            time_indices=torch.tensor([1, 1], device=device),
        )
        alignments.add_results_(
            active_indices=torch.arange(batch_size, device=device),
            logits=sample_logits[:, 3],
            labels=torch.tensor([blank_index, blank_index], device=device),
            time_indices=torch.tensor([1, 1], device=device),
        )

        hypotheses = batched_hyps_to_hypotheses(hyps, alignments)
        assert (hypotheses[0].y_sequence == torch.tensor([5, 2], device=device)).all()
        assert (hypotheses[1].y_sequence == torch.tensor([4], device=device)).all()
        assert hypotheses[0].score == pytest.approx(1.5)
        assert hypotheses[1].score == pytest.approx(1.0)
        assert (hypotheses[0].timestep == torch.tensor([0, 1], device=device)).all()
        assert (hypotheses[1].timestep == torch.tensor([1], device=device)).all()

        etalon = [
            [
                [
                    (torch.tensor(5), sample_logits[0, 0].cpu()),
                    (torch.tensor(blank_index), sample_logits[0, 1].cpu()),
                ],
                [
                    (torch.tensor(2), sample_logits[0, 2].cpu()),
                    (torch.tensor(blank_index), sample_logits[0, 3].cpu()),
                ],
            ],
            [
                [(torch.tensor(blank_index), sample_logits[1, 0].cpu())],
                [(torch.tensor(4), sample_logits[1, 2].cpu()), (torch.tensor(blank_index), sample_logits[1, 3].cpu())],
            ],
        ]
        for batch_i in range(batch_size):
            for t, group_for_timestep in enumerate(etalon[batch_i]):
                for step, (label, current_logits) in enumerate(group_for_timestep):
                    assert torch.allclose(hypotheses[batch_i].alignments[t][step][0], current_logits)
                    assert hypotheses[batch_i].alignments[t][step][1] == label
