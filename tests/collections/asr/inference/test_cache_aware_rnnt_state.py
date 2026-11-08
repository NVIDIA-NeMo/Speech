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

from types import SimpleNamespace

import pytest
import torch

from nemo.collections.asr.inference.streaming.state.cache_aware_rnnt_state import CacheAwareRNNTBeamStreamingState


def _state_with_carry(score: list[float], length: list[float]) -> CacheAwareRNNTBeamStreamingState:
    state = CacheAwareRNNTBeamStreamingState()
    state.hyp_decoding_state = SimpleNamespace(
        score=torch.tensor(score, dtype=torch.float32),
        current_lengths_nb=torch.tensor(length, dtype=torch.float32),
    )
    return state


class TestSelectBestBeamIdx:

    @pytest.mark.unit
    def test_raw_score_ignores_length(self):
        state = _state_with_carry(score=[-1.0, -5.0], length=[100.0, 1.0])
        assert state.select_best_beam_idx_(score_norm=False) == 0

    @pytest.mark.unit
    def test_gnmt_penalty_differs_from_plain_average(self):
        # beam 0: short (length 1), beam 1: long (length 9). Plain average divides the long beam by
        # 10 and the short one by 2 (ratio 5); GNMT (power 1) divides them by 2.333 and 1 (ratio 2.33).
        # Scores with ratio 3 fall between the two, so the formulas disagree:
        #   plain:   -1/2 = -0.5  vs  -3/10   = -0.3   -> long beam (1) wins
        #   GNMT p1: -1/1 = -1.0  vs  -3/2.33 = -1.29  -> short beam (0) wins
        state = _state_with_carry(score=[-1.0, -3.0], length=[1.0, 9.0])
        assert state.select_best_beam_idx_(score_norm=True, length_norm_power=None) == 1
        assert state.select_best_beam_idx_(score_norm=True, length_norm_power=1.0) == 0

    @pytest.mark.unit
    def test_none_matches_offline_plain_average(self):
        # length_norm_power=None must reproduce score / (length + 1), the offline flatten_sort_ ranking.
        state = _state_with_carry(score=[-1.0, -3.0, -2.5], length=[1.0, 9.0, 4.0])
        expected = int((state.hyp_decoding_state.score / (state.hyp_decoding_state.current_lengths_nb + 1)).argmax())
        assert state.select_best_beam_idx_(score_norm=True, length_norm_power=None) == expected == 1

    @pytest.mark.unit
    def test_length_norm_power_scales_the_penalty(self):
        # Long-beam denominators: 2.333 ** 0.75 = 1.89 vs 2.333 ** 1.0 = 2.333. A score ratio of 2
        # lies between them, so power 0.75 picks the short beam and power 1.0 picks the long one.
        state = _state_with_carry(score=[-1.0, -2.0], length=[1.0, 9.0])
        assert state.select_best_beam_idx_(score_norm=True, length_norm_power=0.75) == 0
        assert state.select_best_beam_idx_(score_norm=True, length_norm_power=1.0) == 1

    @pytest.mark.unit
    def test_length_norm_power_zero_disables_length_normalization(self):
        # Plain average picks the long beam (-5/101 > -1/2); power 0 ignores length, so raw score wins.
        state = _state_with_carry(score=[-1.0, -5.0], length=[1.0, 100.0])
        assert state.select_best_beam_idx_(score_norm=True, length_norm_power=None) == 1
        assert state.select_best_beam_idx_(score_norm=True, length_norm_power=0.0) == 0

    @pytest.mark.unit
    def test_length_norm_power_defaults_to_0_75(self):
        state = _state_with_carry(score=[-1.0, -2.0], length=[1.0, 9.0])
        assert state.select_best_beam_idx_(score_norm=True) == 0  # same as power 0.75
        assert state.select_best_beam_idx_(score_norm=True, length_norm_power=1.0) == 1

    @pytest.mark.unit
    def test_raises_without_decoding_carry(self):
        state = CacheAwareRNNTBeamStreamingState()
        with pytest.raises(RuntimeError):
            state.select_best_beam_idx_(score_norm=True)


class TestResetBeamScore:

    @pytest.mark.unit
    def test_reset_zeroes_winning_beam_score_and_length(self):
        # Simulates a beam carry that has drifted far from zero over a long prior session.
        state = _state_with_carry(score=[-12000.0, -float("inf")], length=[30000.0, 30000.0])
        state.reset_beam_score_()
        assert state.hyp_decoding_state.score[0].item() == 0.0
        assert state.hyp_decoding_state.current_lengths_nb[0].item() == 0.0
        # Only the winning beam (index 0) is reset here; select_beam_in_state_item_ is what
        # replicates it across the other beam slots before this is called in production.
        assert state.hyp_decoding_state.score[1].item() == -float("inf")

    @pytest.mark.unit
    def test_reset_prevents_long_session_from_swamping_next_utterance_ranking(self):
        """
        Reproduces the bug this fix addresses: previously only a baseline snapshot was subtracted at
        ranking time, while the actual score/length carry kept drifting for the life of the whole
        stream. Here the carry has drifted the way it would on long audio with LM shallow fusion, and
        resetting it (instead of only baselining the ranking) is what keeps utterance 2's ranking
        correct.
        """
        prior_score, prior_length = -900.0, 2000.0
        state = _state_with_carry(score=[prior_score], length=[prior_length])

        # Without a reset, the huge prior mass swamps the ratio for any two candidates added on top of
        # it -- this is the failure mode the fix removes.
        diluted = _state_with_carry(
            score=[prior_score - 5.0, prior_score - 8.0], length=[prior_length + 10.0, prior_length + 10.0]
        )
        diluted_ranking = diluted.hyp_decoding_state.score / (diluted.hyp_decoding_state.current_lengths_nb + 1)
        assert abs(diluted_ranking[0].item() - diluted_ranking[1].item()) < 0.01

        # With the fix, the carry is reset to zero right after the EOU that produced `prior_score`, so
        # utterance 2's candidates start from zero instead of `prior_score`/`prior_length`.
        state.reset_beam_score_()
        assert state.hyp_decoding_state.score[0].item() == 0.0
        utterance_2 = _state_with_carry(score=[-5.0, -8.0], length=[10.0, 10.0])
        assert utterance_2.select_best_beam_idx_(score_norm=True) == 0

    @pytest.mark.unit
    def test_reset_without_decoding_carry_is_a_noop(self):
        state = CacheAwareRNNTBeamStreamingState()
        state.reset_beam_score_()  # must not raise

    @pytest.mark.unit
    def test_reset_clones_inference_mode_tensors(self):
        # select_beam_in_state_item_ builds score/current_lengths_nb inside torch.inference_mode();
        # an in-place write to those tensors outside that context must not raise.
        with torch.inference_mode():
            score = torch.tensor([-12000.0, -float("inf")])
            length = torch.tensor([30000.0, 30000.0])
        state = CacheAwareRNNTBeamStreamingState()
        state.hyp_decoding_state = SimpleNamespace(score=score, current_lengths_nb=length)
        state.reset_beam_score_()  # must not raise
        assert state.hyp_decoding_state.score[0].item() == 0.0
        assert state.hyp_decoding_state.current_lengths_nb[0].item() == 0.0
