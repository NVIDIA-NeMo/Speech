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

from nemo.collections.asr.parts.submodules.transducer_decoding.batched_hyps import BatchedHyps
from nemo.collections.asr.parts.utils.rnnt_utils import batched_hyps_to_hypotheses
from tests.collections.asr.decoding.utils import avoid_sync_operations

DEVICES: List[torch.device] = [torch.device("cpu")]

if torch.cuda.is_available():
    DEVICES.append(torch.device("cuda"))

if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICES.append(torch.device("mps"))

# blank id that does not collide with any non-blank label used in the "no blank steps" tests
NON_COLLIDING_BLANK_ID = 1024


class TestBatchedHyps:
    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    def test_instantiate(self, device: torch.device):
        hyps = BatchedHyps(batch_size=2, init_length=3, blank_id=NON_COLLIDING_BLANK_ID, device=device)
        assert torch.is_tensor(hyps.timestamps)
        # device: for mps device we need to use `type`, not directly compare
        assert hyps.timestamps.device.type == device.type
        assert hyps.timestamps.shape == (2, 3)
        assert hyps.transcript.shape == (2, 3)
        assert hyps.scores.shape == (2,)
        assert hyps.current_lengths.tolist() == [0, 0]
        # optional storage is disabled by default
        assert hyps.token_durations is None
        assert hyps.step_confidence is None
        assert hyps.logits is None

    @pytest.mark.unit
    @pytest.mark.parametrize("batch_size", [-1, 0])
    def test_instantiate_incorrect_batch_size(self, batch_size):
        with pytest.raises(ValueError):
            _ = BatchedHyps(batch_size=batch_size, init_length=3, blank_id=0)

    @pytest.mark.unit
    @pytest.mark.parametrize("init_length", [-1, 0])
    def test_instantiate_incorrect_init_length(self, init_length):
        with pytest.raises(ValueError):
            _ = BatchedHyps(batch_size=1, init_length=init_length, blank_id=0)

    @pytest.mark.unit
    def test_instantiate_with_logits_requires_logits_dim(self):
        # `with_logits=True` without `logits_dim` is invalid
        with pytest.raises(ValueError):
            _ = BatchedHyps(batch_size=1, init_length=3, blank_id=0, with_logits=True, logits_dim=None)

    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    def test_instantiate_optional_storage(self, device: torch.device):
        logits_dim = 7
        hyps = BatchedHyps(
            batch_size=2,
            init_length=3,
            blank_id=0,
            logits_dim=logits_dim,
            device=device,
            with_durations=True,
            with_step_confidence=True,
            with_duration_confidence=True,
            with_logits=True,
        )
        assert hyps.token_durations.shape == (2, 3)
        # duration confidence makes the confidence tensor store a pair (step + duration) per token
        assert hyps.step_confidence.shape == (2, 3, 2)
        assert hyps.logits.shape == (2, 3, logits_dim)

        # without duration confidence the confidence tensor is 2d
        hyps_no_dur_conf = BatchedHyps(
            batch_size=2,
            init_length=3,
            blank_id=0,
            device=device,
            with_step_confidence=True,
            with_duration_confidence=False,
        )
        assert hyps_no_dur_conf.step_confidence.shape == (2, 3)

    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    def test_add_results_masked(self, device: torch.device):
        # batch of size 2, add label for first utterance only (second is inactive)
        hyps = BatchedHyps(batch_size=2, init_length=1, blank_id=NON_COLLIDING_BLANK_ID, device=device)
        hyps.add_results_masked_(
            active_mask=torch.tensor([True, False], device=device),
            labels=torch.tensor([5, 1], device=device),
            time_indices=torch.tensor([1, 0], device=device),
            scores=torch.tensor([0.5, 10.0], device=device),
        )
        assert hyps.current_lengths.tolist() == [1, 0]
        assert hyps.transcript.tolist()[0][:1] == [5]
        assert hyps.timestamps.tolist()[0][:1] == [1]
        assert hyps.scores.tolist() == pytest.approx([0.5, 0.0])  # inactive score should be ignored!
        assert hyps.last_nb_timestamp.tolist() == [1, -1]
        assert hyps.last_nb_timestamp_lasts.tolist() == [1, 0]
        assert hyps.last_nb_labels.tolist() == [5, -1]

    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    def test_add_multiple_results_masked(self, device: torch.device):
        # batch of size 2, add label for first utterance, then add labels for both utterances
        hyps = BatchedHyps(batch_size=2, init_length=1, blank_id=NON_COLLIDING_BLANK_ID, device=device)
        hyps.add_results_masked_(
            active_mask=torch.tensor([True, False], device=device),
            labels=torch.tensor([5, 2], device=device),
            time_indices=torch.tensor([1, 0], device=device),
            scores=torch.tensor([0.5, 10.0], device=device),
        )
        hyps.add_results_masked_(
            active_mask=torch.tensor([True, True], device=device),
            labels=torch.tensor([2, 4], device=device),
            time_indices=torch.tensor([1, 2], device=device),
            scores=torch.tensor([1.0, 1.0], device=device),
        )
        assert hyps.current_lengths.tolist() == [2, 1]
        assert hyps.transcript.tolist()[0][:2] == [5, 2]
        assert hyps.transcript.tolist()[1][:1] == [4]
        assert hyps.timestamps.tolist()[0][:2] == [1, 1]
        assert hyps.timestamps.tolist()[1][:1] == [2]
        assert hyps.scores.tolist() == pytest.approx([1.5, 1.0])
        assert hyps.last_nb_timestamp.tolist() == [1, 2]
        assert hyps.last_nb_timestamp_lasts.tolist() == [2, 1]

    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    def test_add_results_masked_no_checks(self, device: torch.device):
        # `check_lengths=False` must contain no host<->device synchronization (blocking) operations
        hyps = BatchedHyps(batch_size=2, init_length=4, blank_id=NON_COLLIDING_BLANK_ID, device=device)
        active_mask = torch.tensor([True, False], device=device)
        time_indices = torch.tensor([1, 0], device=device)
        scores = torch.tensor([0.5, 10.0], device=device)
        labels = torch.tensor([5, 1], device=device)
        # check there are no blocking operations
        with avoid_sync_operations(device=device):
            hyps.add_results_masked_(
                active_mask=active_mask,
                labels=labels,
                time_indices=time_indices,
                scores=scores,
                check_lengths=False,
            )
        assert hyps.current_lengths.tolist() == [1, 0]
        assert hyps.transcript.tolist()[0][:1] == [5]
        assert hyps.timestamps.tolist()[0][:1] == [1]
        assert hyps.scores.tolist() == pytest.approx([0.5, 0.0])  # inactive score should be ignored!
        assert hyps.last_nb_timestamp.tolist() == [1, -1]
        assert hyps.last_nb_timestamp_lasts.tolist() == [1, 0]

    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    def test_add_results_masked_reallocates(self, device: torch.device):
        # init_length is intentionally small; storage must grow transparently when check_lengths=True
        hyps = BatchedHyps(batch_size=2, init_length=1, blank_id=NON_COLLIDING_BLANK_ID, device=device)
        for step in range(5):
            hyps.add_results_masked_(
                active_mask=torch.tensor([True, True], device=device),
                labels=torch.tensor([step, step + 10], device=device),
                time_indices=torch.tensor([step, step], device=device),
                scores=torch.tensor([1.0, 1.0], device=device),
                check_lengths=True,
            )
        assert hyps._max_length >= 5
        assert hyps.current_lengths.tolist() == [5, 5]
        assert hyps.transcript.tolist()[0][:5] == [0, 1, 2, 3, 4]
        assert hyps.transcript.tolist()[1][:5] == [10, 11, 12, 13, 14]

    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    def test_add_results_masked_with_blank_steps(self, device: torch.device):
        # with_blank_steps=True: blank labels are stored in the transcript, but they do NOT advance
        # the score / last-non-blank tracking. Single utterance for clarity.
        blank_id = 0
        hyps = BatchedHyps(batch_size=1, init_length=2, blank_id=blank_id, device=device, with_blank_steps=True)
        # (label, time, score): two non-blank tokens at t=0, then blank, then non-blank at t=1, then blank
        steps = [
            (3, 0, 1.0),  # non-blank
            (5, 0, 1.5),  # non-blank, same timestamp
            (blank_id, 0, 0.1),  # blank
            (7, 1, 2.0),  # non-blank
            (blank_id, 1, 0.2),  # blank
        ]
        for label, time, score in steps:
            hyps.add_results_masked_(
                active_mask=torch.tensor([True], device=device),
                labels=torch.tensor([label], device=device),
                time_indices=torch.tensor([time], device=device),
                scores=torch.tensor([score], device=device),
            )
        # all steps (including blanks) are stored
        assert hyps.current_lengths.tolist() == [5]
        assert hyps.transcript.tolist()[0][:5] == [3, 5, blank_id, 7, blank_id]
        assert hyps.timestamps.tolist()[0][:5] == [0, 0, 0, 1, 1]
        # only non-blank scores accumulate: 1.0 + 1.5 + 2.0
        assert hyps.scores.tolist() == pytest.approx([4.5])
        # last-non-blank tracking ignores blanks
        assert hyps.last_nb_timestamp.tolist() == [1]
        assert hyps.last_nb_timestamp_lasts.tolist() == [1]
        assert hyps.last_nb_labels.tolist() == [7]

    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    def test_get_data_without_blank_no_blank_steps(self, device: torch.device):
        # with_blank_steps=False: data is returned as-is (it never contained blanks)
        hyps = BatchedHyps(batch_size=2, init_length=2, blank_id=NON_COLLIDING_BLANK_ID, device=device)
        hyps.add_results_masked_(
            active_mask=torch.tensor([True, True], device=device),
            labels=torch.tensor([5, 4], device=device),
            time_indices=torch.tensor([0, 1], device=device),
            scores=torch.tensor([1.0, 1.0], device=device),
        )
        hyps.add_results_masked_(
            active_mask=torch.tensor([True, False], device=device),
            labels=torch.tensor([2, 0], device=device),
            time_indices=torch.tensor([1, 0], device=device),
            scores=torch.tensor([1.0, 0.0], device=device),
        )
        lengths, transcript, timestamps, durations, confidence = hyps.get_data_without_blank()
        # returned objects are the underlying (unmodified) tensors
        assert lengths is hyps.current_lengths
        assert transcript is hyps.transcript
        assert timestamps is hyps.timestamps
        assert durations is None
        assert confidence is None
        assert lengths.tolist() == [2, 1]

    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    def test_get_data_without_blank_with_blank_steps(self, device: torch.device):
        # with_blank_steps=True: blanks are stripped and non-blank order is preserved
        blank_id = 0
        hyps = BatchedHyps(batch_size=2, init_length=2, blank_id=blank_id, device=device, with_blank_steps=True)
        # seq 0: [5, blank, 2, blank]  -> [5, 2]
        # seq 1: [blank, 4]            -> [4]
        steps = [
            # (labels, times, active_mask)
            ([5, blank_id], [0, 0], [True, True]),
            ([blank_id, 4], [0, 1], [True, True]),
            ([2, blank_id], [1, 1], [True, False]),
            ([blank_id, 0], [1, 0], [True, False]),
        ]
        for labels, times, active in steps:
            hyps.add_results_masked_(
                active_mask=torch.tensor(active, device=device),
                labels=torch.tensor(labels, device=device),
                time_indices=torch.tensor(times, device=device),
                scores=torch.tensor([1.0, 1.0], device=device),
            )
        lengths, transcript, timestamps, durations, confidence = hyps.get_data_without_blank()
        assert lengths.tolist() == [2, 1]
        assert transcript[0, :2].tolist() == [5, 2]
        assert transcript[1, :1].tolist() == [4]
        assert timestamps[0, :2].tolist() == [0, 1]
        assert timestamps[1, :1].tolist() == [1]
        assert durations is None
        assert confidence is None

    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    def test_get_last_labels(self, device: torch.device):
        hyps = BatchedHyps(batch_size=2, init_length=2, blank_id=NON_COLLIDING_BLANK_ID, device=device)
        # no labels yet -> pad_id everywhere
        assert hyps.get_last_labels(pad_id=-1).tolist() == [-1, -1]
        hyps.add_results_masked_(
            active_mask=torch.tensor([True, False], device=device),
            labels=torch.tensor([5, 1], device=device),
            time_indices=torch.tensor([0, 0], device=device),
            scores=torch.tensor([1.0, 0.0], device=device),
        )
        assert hyps.get_last_labels(pad_id=-1).tolist() == [5, -1]
        assert hyps.get_last_labels(pad_id=100).tolist() == [5, 100]

    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    def test_clear(self, device: torch.device):
        hyps = BatchedHyps(batch_size=2, init_length=2, blank_id=NON_COLLIDING_BLANK_ID, device=device)
        hyps.add_results_masked_(
            active_mask=torch.tensor([True, True], device=device),
            labels=torch.tensor([5, 4], device=device),
            time_indices=torch.tensor([0, 0], device=device),
            scores=torch.tensor([1.0, 1.0], device=device),
        )
        hyps.clear_()
        assert hyps.current_lengths.tolist() == [0, 0]
        assert hyps.scores.tolist() == pytest.approx([0.0, 0.0])
        assert hyps.transcript.tolist() == [[0, 0], [0, 0]]
        assert hyps.last_nb_timestamp.tolist() == [-1, -1]
        assert hyps.last_nb_timestamp_lasts.tolist() == [0, 0]
        assert hyps.last_nb_labels.tolist() == [-1, -1]

    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    def test_clone(self, device: torch.device):
        logits_dim = 7
        hyps = BatchedHyps(
            batch_size=2,
            init_length=2,
            blank_id=0,
            logits_dim=logits_dim,
            device=device,
            with_durations=True,
            with_step_confidence=True,
            with_logits=True,
            with_blank_steps=True,
        )
        hyps.add_results_masked_(
            active_mask=torch.tensor([True, True], device=device),
            labels=torch.tensor([5, 4], device=device),
            time_indices=torch.tensor([0, 0], device=device),
            scores=torch.tensor([1.0, 1.0], device=device),
            token_durations=torch.tensor([1, 2], device=device),
            confidence=torch.tensor([0.9, 0.8], device=device),
            logits=torch.rand((2, logits_dim), device=device),
        )
        clone = hyps.clone()
        # flags carried over
        assert clone.with_durations and clone.with_step_confidence and clone.with_logits and clone.with_blank_steps
        assert clone.blank_id == hyps.blank_id
        # values copied
        assert clone.current_lengths.tolist() == hyps.current_lengths.tolist()
        assert torch.equal(clone.transcript, hyps.transcript)
        assert torch.allclose(clone.logits, hyps.logits)
        assert torch.allclose(clone.step_confidence, hyps.step_confidence)
        assert torch.equal(clone.token_durations, hyps.token_durations)
        # clone is independent of the original
        hyps.transcript.fill_(0)
        hyps.scores.fill_(0.0)
        assert clone.transcript.tolist()[0][:1] == [5]
        assert clone.scores.tolist() == pytest.approx([1.0, 1.0])

    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    def test_merge(self, device: torch.device):
        # merge two batched hypotheses (basic case: no blank steps, no optional storage)
        def build(labels_per_step, times_per_step, masks_per_step, scores_per_step):
            hyps = BatchedHyps(batch_size=2, init_length=4, blank_id=NON_COLLIDING_BLANK_ID, device=device)
            for labels, times, mask, scores in zip(labels_per_step, times_per_step, masks_per_step, scores_per_step):
                hyps.add_results_masked_(
                    active_mask=torch.tensor(mask, device=device),
                    labels=torch.tensor(labels, device=device),
                    time_indices=torch.tensor(times, device=device),
                    scores=torch.tensor(scores, device=device),
                )
            return hyps

        # A: seq0=[5, 2], seq1=[4]
        hyps_a = build(
            labels_per_step=[[5, 4], [2, 0]],
            times_per_step=[[0, 0], [1, 0]],
            masks_per_step=[[True, True], [True, False]],
            scores_per_step=[[0.5, 0.7], [0.5, 0.0]],
        )
        # B: seq0=[7], seq1=[8, 9]
        hyps_b = build(
            labels_per_step=[[7, 8], [0, 9]],
            times_per_step=[[2, 2], [0, 3]],
            masks_per_step=[[True, True], [False, True]],
            scores_per_step=[[0.3, 0.3], [0.0, 0.4]],
        )

        hyps_a.merge_(hyps_b)
        assert hyps_a.current_lengths.tolist() == [3, 3]
        assert hyps_a.transcript[0, :3].tolist() == [5, 2, 7]
        assert hyps_a.transcript[1, :3].tolist() == [4, 8, 9]
        assert hyps_a.scores.tolist() == pytest.approx([1.3, 1.4])

    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    def test_merge_with_logits(self, device: torch.device):
        # Regression test: merge_ must use dim=1 (sequence axis) for logits scatter/cat,
        # not dim=-1 (logits-dim axis), which would silently corrupt the data.
        logits_dim = 5
        blank_id = NON_COLLIDING_BLANK_ID

        def build_with_logits(labels_per_step, times_per_step, masks_per_step, scores_per_step, logits_per_step):
            hyps = BatchedHyps(
                batch_size=2,
                init_length=4,
                blank_id=blank_id,
                logits_dim=logits_dim,
                device=device,
                float_dtype=torch.float32,
                with_logits=True,
            )
            for labels, times, mask, scores, logits in zip(
                labels_per_step, times_per_step, masks_per_step, scores_per_step, logits_per_step
            ):
                hyps.add_results_masked_(
                    active_mask=torch.tensor(mask, device=device),
                    labels=torch.tensor(labels, device=device),
                    time_indices=torch.tensor(times, device=device),
                    scores=torch.tensor(scores, device=device),
                    logits=logits,
                )
            return hyps

        # Fixed logits so we can assert exact values after merge
        logits_a0 = torch.full((2, logits_dim), 1.0, device=device)  # step for seq0=[5], seq1=[4]
        logits_a1 = torch.full((2, logits_dim), 2.0, device=device)  # step for seq0=[2], seq1 inactive

        logits_b0 = torch.full((2, logits_dim), 3.0, device=device)  # step for seq0=[7], seq1=[8]
        logits_b1 = torch.full((2, logits_dim), 4.0, device=device)  # step for seq0 inactive, seq1=[9]

        # A: seq0=[5, 2], seq1=[4]
        hyps_a = build_with_logits(
            labels_per_step=[[5, 4], [2, 0]],
            times_per_step=[[0, 0], [1, 0]],
            masks_per_step=[[True, True], [True, False]],
            scores_per_step=[[0.5, 0.7], [0.5, 0.0]],
            logits_per_step=[logits_a0, logits_a1],
        )
        # B: seq0=[7], seq1=[8, 9]
        hyps_b = build_with_logits(
            labels_per_step=[[7, 8], [0, 9]],
            times_per_step=[[2, 2], [0, 3]],
            masks_per_step=[[True, True], [False, True]],
            scores_per_step=[[0.3, 0.3], [0.0, 0.4]],
            logits_per_step=[logits_b0, logits_b1],
        )

        hyps_a.merge_(hyps_b)

        # Sequence lengths: seq0=2+1=3, seq1=1+2=3
        assert hyps_a.current_lengths.tolist() == [3, 3]

        # seq0 logits: positions [0,1] from A (value=1 then 2), position [2] from B (value=3)
        assert torch.allclose(hyps_a.logits[0, 0], torch.full((logits_dim,), 1.0, device=device))
        assert torch.allclose(hyps_a.logits[0, 1], torch.full((logits_dim,), 2.0, device=device))
        assert torch.allclose(hyps_a.logits[0, 2], torch.full((logits_dim,), 3.0, device=device))

        # seq1 logits: position [0] from A (value=1), positions [1,2] from B (value=3 then 4)
        assert torch.allclose(hyps_a.logits[1, 0], torch.full((logits_dim,), 1.0, device=device))
        assert torch.allclose(hyps_a.logits[1, 1], torch.full((logits_dim,), 3.0, device=device))
        assert torch.allclose(hyps_a.logits[1, 2], torch.full((logits_dim,), 4.0, device=device))

    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    def test_merge_with_logits_triggers_reallocation(self, device: torch.device):
        # When the combined length exceeds _max_length, merge_ must reallocate logits along dim=1
        # (sequence axis). With the dim=-1 bug, the reallocation would expand the logits-dim
        # axis instead, causing a shape mismatch on the subsequent scatter_.
        logits_dim = 5
        blank_id = NON_COLLIDING_BLANK_ID

        # Use init_length=1 to force reallocation during merge
        hyps_a = BatchedHyps(
            batch_size=1,
            init_length=1,
            blank_id=blank_id,
            logits_dim=logits_dim,
            device=device,
            float_dtype=torch.float32,
            with_logits=True,
        )
        hyps_b = BatchedHyps(
            batch_size=1,
            init_length=1,
            blank_id=blank_id,
            logits_dim=logits_dim,
            device=device,
            float_dtype=torch.float32,
            with_logits=True,
        )

        logits_a = torch.full((1, logits_dim), 1.0, device=device)
        logits_b = torch.full((1, logits_dim), 2.0, device=device)

        hyps_a.add_results_masked_(
            active_mask=torch.tensor([True], device=device),
            labels=torch.tensor([5], device=device),
            time_indices=torch.tensor([0], device=device),
            scores=torch.tensor([1.0], device=device),
            logits=logits_a,
        )
        hyps_b.add_results_masked_(
            active_mask=torch.tensor([True], device=device),
            labels=torch.tensor([7], device=device),
            time_indices=torch.tensor([1], device=device),
            scores=torch.tensor([1.0], device=device),
            logits=logits_b,
        )

        # cur_len=1, other_len=1 -> combined=2 >= init_length=1, so reallocation is triggered
        hyps_a.merge_(hyps_b)

        assert hyps_a.current_lengths.tolist() == [2]
        assert hyps_a.logits.shape == (1, hyps_a._max_length, logits_dim)
        assert torch.allclose(hyps_a.logits[0, 0], torch.full((logits_dim,), 1.0, device=device))
        assert torch.allclose(hyps_a.logits[0, 1], torch.full((logits_dim,), 2.0, device=device))


class TestConvertToHypotheses:
    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    def test_convert_no_blank_steps(self, device: torch.device):
        # with_blank_steps=False: transcript already contains only non-blank labels
        hyps = BatchedHyps(batch_size=2, init_length=1, blank_id=NON_COLLIDING_BLANK_ID, device=device)
        hyps.add_results_masked_(
            active_mask=torch.tensor([True, False], device=device),
            labels=torch.tensor([5, 0], device=device),
            time_indices=torch.tensor([1, 0], device=device),
            scores=torch.tensor([0.5, 0.0], device=device),
        )
        hyps.add_results_masked_(
            active_mask=torch.tensor([True, True], device=device),
            labels=torch.tensor([2, 4], device=device),
            time_indices=torch.tensor([1, 2], device=device),
            scores=torch.tensor([1.0, 1.0], device=device),
        )
        hypotheses = batched_hyps_to_hypotheses(hyps)
        assert (hypotheses[0].y_sequence == torch.tensor([5, 2], device="cpu")).all()
        assert (hypotheses[1].y_sequence == torch.tensor([4], device="cpu")).all()
        assert hypotheses[0].score == pytest.approx(1.5)
        assert hypotheses[1].score == pytest.approx(1.0)
        assert (hypotheses[0].timestamp == torch.tensor([1, 1], device="cpu")).all()
        assert (hypotheses[1].timestamp == torch.tensor([2], device="cpu")).all()
        # no blank steps -> no alignments / frame confidence
        assert hypotheses[0].alignments is None
        assert hypotheses[1].alignments is None
        assert hypotheses[0].frame_confidence is None

    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    def test_convert_batch_size_arg(self, device: torch.device):
        # batch_size arg returns only the first `batch_size` hypotheses (CUDA-graph constant batch)
        hyps = BatchedHyps(batch_size=4, init_length=1, blank_id=NON_COLLIDING_BLANK_ID, device=device)
        hyps.add_results_masked_(
            active_mask=torch.tensor([True, True, True, True], device=device),
            labels=torch.tensor([5, 4, 3, 2], device=device),
            time_indices=torch.tensor([0, 0, 0, 0], device=device),
            scores=torch.tensor([1.0, 1.0, 1.0, 1.0], device=device),
        )
        hypotheses = batched_hyps_to_hypotheses(hyps, batch_size=2)
        assert len(hypotheses) == 2

    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    def test_convert_with_blank_steps_strips_blanks(self, device: torch.device):
        # with_blank_steps=True but no logits/confidence: blanks must be stripped from y_sequence/timestamps,
        # while alignments are NOT produced (no logits recorded)
        blank_id = 0
        hyps = BatchedHyps(batch_size=2, init_length=2, blank_id=blank_id, device=device, with_blank_steps=True)
        # seq 0: [5, blank, 2, blank] -> [5, 2]
        # seq 1: [blank, 4]           -> [4]
        steps = [
            ([5, blank_id], [0, 0], [True, True], [0.5, 0.1]),
            ([blank_id, 4], [0, 1], [True, True], [0.1, 1.0]),
            ([2, blank_id], [1, 1], [True, False], [1.0, 0.0]),
            ([blank_id, 0], [1, 0], [True, False], [0.1, 0.0]),
        ]
        for labels, times, active, scores in steps:
            hyps.add_results_masked_(
                active_mask=torch.tensor(active, device=device),
                labels=torch.tensor(labels, device=device),
                time_indices=torch.tensor(times, device=device),
                scores=torch.tensor(scores, device=device),
            )
        hypotheses = batched_hyps_to_hypotheses(hyps)
        assert (hypotheses[0].y_sequence == torch.tensor([5, 2], device="cpu")).all()
        assert (hypotheses[1].y_sequence == torch.tensor([4], device="cpu")).all()
        assert (hypotheses[0].timestamp == torch.tensor([0, 1], device="cpu")).all()
        assert (hypotheses[1].timestamp == torch.tensor([1], device="cpu")).all()
        # only non-blank scores accumulated
        assert hypotheses[0].score == pytest.approx(1.5)
        assert hypotheses[1].score == pytest.approx(1.0)
        # no logits recorded -> alignments stay None even though blank steps were stored
        assert hypotheses[0].alignments is None
        assert hypotheses[1].alignments is None

    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    def test_convert_logits_no_alignments_without_blank_steps(self, device: torch.device):
        # logits are recorded but with_blank_steps=False -> alignments must NOT be produced
        logits_dim = 7
        hyps = BatchedHyps(
            batch_size=2,
            init_length=2,
            blank_id=6,
            logits_dim=logits_dim,
            device=device,
            with_logits=True,
            with_blank_steps=False,
        )
        hyps.add_results_masked_(
            active_mask=torch.tensor([True, True], device=device),
            labels=torch.tensor([5, 4], device=device),
            time_indices=torch.tensor([0, 0], device=device),
            scores=torch.tensor([1.0, 1.0], device=device),
            logits=torch.rand((2, logits_dim), device=device),
        )
        hypotheses = batched_hyps_to_hypotheses(hyps)
        assert hypotheses[0].alignments is None
        assert hypotheses[1].alignments is None

    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    def test_convert_with_blank_steps_and_logits_alignments(self, device: torch.device):
        # Reproduces alignment recovery: with_blank_steps=True + with_logits=True
        batch_size = 2
        logits_dim = 7
        blank_index = 6
        hyps = BatchedHyps(
            batch_size=batch_size,
            init_length=1,
            blank_id=blank_index,
            logits_dim=logits_dim,
            device=device,
            with_logits=True,
            with_blank_steps=True,
        )
        # sequence 0: [[5, blank], [2, blank]] -> [5, 2]
        # sequence 1: [[blank   ], [4, blank]] -> [4]
        # one logits row per (batch, add-call); rows belonging to inactive entries are ignored
        L = [torch.rand((batch_size, logits_dim), device=device) for _ in range(4)]
        # call0: seq0=5@t0, seq1=blank@t0
        hyps.add_results_masked_(
            active_mask=torch.tensor([True, True], device=device),
            labels=torch.tensor([5, blank_index], device=device),
            time_indices=torch.tensor([0, 0], device=device),
            scores=torch.tensor([0.5, 0.1], device=device),
            logits=L[0],
        )
        # call1: seq0=blank@t0, seq1=4@t1
        hyps.add_results_masked_(
            active_mask=torch.tensor([True, True], device=device),
            labels=torch.tensor([blank_index, 4], device=device),
            time_indices=torch.tensor([0, 1], device=device),
            scores=torch.tensor([0.1, 1.0], device=device),
            logits=L[1],
        )
        # call2: seq0=2@t1, seq1=blank@t1
        hyps.add_results_masked_(
            active_mask=torch.tensor([True, True], device=device),
            labels=torch.tensor([2, blank_index], device=device),
            time_indices=torch.tensor([1, 1], device=device),
            scores=torch.tensor([1.0, 0.1], device=device),
            logits=L[2],
        )
        # call3: seq0=blank@t1, seq1 inactive
        hyps.add_results_masked_(
            active_mask=torch.tensor([True, False], device=device),
            labels=torch.tensor([blank_index, 0], device=device),
            time_indices=torch.tensor([1, 0], device=device),
            scores=torch.tensor([0.1, 0.0], device=device),
            logits=L[3],
        )

        hypotheses = batched_hyps_to_hypotheses(hyps)
        assert (hypotheses[0].y_sequence == torch.tensor([5, 2], device="cpu")).all()
        assert (hypotheses[1].y_sequence == torch.tensor([4], device="cpu")).all()
        assert hypotheses[0].score == pytest.approx(1.5)
        assert hypotheses[1].score == pytest.approx(1.0)
        assert (hypotheses[0].timestamp == torch.tensor([0, 1], device="cpu")).all()
        assert (hypotheses[1].timestamp == torch.tensor([1], device="cpu")).all()

        # alignments are grouped by timestamp; each entry is a (logits, label) tuple
        etalon = [
            [
                [(L[0][0].cpu(), 5), (L[1][0].cpu(), blank_index)],
                [(L[2][0].cpu(), 2), (L[3][0].cpu(), blank_index)],
            ],
            [
                [(L[0][1].cpu(), blank_index)],
                [(L[1][1].cpu(), 4), (L[2][1].cpu(), blank_index)],
            ],
        ]
        for batch_i in range(batch_size):
            assert len(hypotheses[batch_i].alignments) == len(etalon[batch_i])
            for t, group_for_timestamp in enumerate(etalon[batch_i]):
                assert len(hypotheses[batch_i].alignments[t]) == len(group_for_timestamp)
                for step, (current_logits, label) in enumerate(group_for_timestamp):
                    assert torch.allclose(hypotheses[batch_i].alignments[t][step][0], current_logits)
                    assert hypotheses[batch_i].alignments[t][step][1] == label

    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    def test_convert_with_durations(self, device: torch.device):
        # TDT-style: token durations are stored and (with blank steps) stripped together with blanks
        blank_id = 0
        hyps = BatchedHyps(
            batch_size=1,
            init_length=2,
            blank_id=blank_id,
            device=device,
            with_durations=True,
            with_blank_steps=True,
        )
        # transcript [3, blank, 7, blank] with durations [2, 1, 4, 1] -> [3, 7] with durations [2, 4]
        steps = [
            (3, 0, 2, 1.0),
            (blank_id, 2, 1, 0.1),
            (7, 3, 4, 2.0),
            (blank_id, 7, 1, 0.2),
        ]
        for label, time, duration, score in steps:
            hyps.add_results_masked_(
                active_mask=torch.tensor([True], device=device),
                labels=torch.tensor([label], device=device),
                time_indices=torch.tensor([time], device=device),
                scores=torch.tensor([score], device=device),
                token_durations=torch.tensor([duration], device=device),
            )
        hypotheses = batched_hyps_to_hypotheses(hyps)
        assert (hypotheses[0].y_sequence == torch.tensor([3, 7], device="cpu")).all()
        assert (hypotheses[0].timestamp == torch.tensor([0, 3], device="cpu")).all()
        assert (hypotheses[0].token_duration == torch.tensor([2, 4], device="cpu")).all()
        assert hypotheses[0].score == pytest.approx(3.0)

    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    def test_convert_with_step_confidence_no_blank_steps(self, device: torch.device):
        # with_blank_steps=False: per-token confidence is precomputed, no frame_confidence (no blank steps)
        hyps = BatchedHyps(
            batch_size=1,
            init_length=2,
            blank_id=NON_COLLIDING_BLANK_ID,
            device=device,
            with_step_confidence=True,
        )
        hyps.add_results_masked_(
            active_mask=torch.tensor([True], device=device),
            labels=torch.tensor([5], device=device),
            time_indices=torch.tensor([0], device=device),
            scores=torch.tensor([1.0], device=device),
            confidence=torch.tensor([0.9], device=device),
        )
        hyps.add_results_masked_(
            active_mask=torch.tensor([True], device=device),
            labels=torch.tensor([2], device=device),
            time_indices=torch.tensor([1], device=device),
            scores=torch.tensor([1.0], device=device),
            confidence=torch.tensor([0.8], device=device),
        )
        hypotheses = batched_hyps_to_hypotheses(hyps)
        assert hypotheses[0].non_blank_step_confidence_precomputed == pytest.approx([0.9, 0.8])
        assert hypotheses[0].frame_confidence is None

    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    def test_convert_with_step_confidence_and_blank_steps(self, device: torch.device):
        # with_blank_steps=True: frame_confidence is grouped per timestamp (incl. blanks),
        # while non_blank_step_confidence_precomputed holds only non-blank tokens
        blank_id = 0
        hyps = BatchedHyps(
            batch_size=1,
            init_length=2,
            blank_id=blank_id,
            device=device,
            with_step_confidence=True,
            with_blank_steps=True,
        )
        # transcript [3, blank, 7], confidence [0.9, 0.5, 0.8], timestamps [0, 0, 1]
        steps = [
            (3, 0, 0.9, 1.0),
            (blank_id, 0, 0.5, 0.1),
            (7, 1, 0.8, 2.0),
        ]
        for label, time, confidence, score in steps:
            hyps.add_results_masked_(
                active_mask=torch.tensor([True], device=device),
                labels=torch.tensor([label], device=device),
                time_indices=torch.tensor([time], device=device),
                scores=torch.tensor([score], device=device),
                confidence=torch.tensor([confidence], device=device),
            )
        hypotheses = batched_hyps_to_hypotheses(hyps)
        # non-blank tokens only
        assert hypotheses[0].non_blank_step_confidence_precomputed == pytest.approx([0.9, 0.8])
        # grouped by timestamp: t=0 has 2 steps (token + blank), t=1 has 1 step
        frame_confidence = hypotheses[0].frame_confidence
        assert len(frame_confidence) == 2
        assert len(frame_confidence[0]) == 2
        assert len(frame_confidence[1]) == 1
        flat = [float(c) for group in frame_confidence for c in group]
        assert flat == pytest.approx([0.9, 0.5, 0.8])
