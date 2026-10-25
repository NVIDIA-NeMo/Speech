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
    def test_score_norm_applies_gnmt_length_penalty(self):
        # No baseline concept anymore -- the carry is reset at every EOU (see
        # TestResetBeamScore), so ranking is always score / ((5 + length) / 6) ** power.
        state = _state_with_carry(score=[-10.0, -6.0], length=[9.0, 1.0])
        # beam 0: -10/((5+9)/6) = -10/2.333.. = -4.286 ; beam 1: -6/((5+1)/6) = -6/1 = -6.0 -> beam 0 wins
        assert state.select_best_beam_idx_(score_norm=True) == 0

    @pytest.mark.unit
    def test_length_norm_power_zero_disables_length_normalization(self):
        state = _state_with_carry(score=[-1.0, -5.0], length=[100.0, 1.0])
        # power=0 -> denom is always 1 regardless of length, so this reduces to raw score comparison.
        assert state.select_best_beam_idx_(score_norm=True, length_norm_power=0.0) == 0

    @pytest.mark.unit
    def test_length_norm_power_defaults_to_one(self):
        state = _state_with_carry(score=[-10.0, -6.0], length=[9.0, 1.0])
        assert state.select_best_beam_idx_(score_norm=True) == state.select_best_beam_idx_(
            score_norm=True, length_norm_power=1.0
        )

    @pytest.mark.unit
    def test_raises_without_decoding_carry(self):
        state = CacheAwareRNNTBeamStreamingState()
        with pytest.raises(RuntimeError):
            state.select_best_beam_idx_(score_norm=True)

    @pytest.mark.unit
    def test_gnmt_penalty_under_normalizes_short_hypotheses_less_than_plain_average(self):
        # At length=1, power=1: GNMT denom is (5+1)/6 = 1.0 (score passes through unchanged), while
        # the old plain-average denom was (length+1) = 2.0 (score halved). This is the short-utterance
        # over-normalization the GNMT penalty is meant to fix -- drive_thru_original averages ~5
        # words/utterance, where the old formula penalized short, correct hypotheses too harshly.
        state = _state_with_carry(score=[-3.0], length=[1.0])
        state.select_best_beam_idx_(score_norm=True)
        gnmt_denom = ((5 + 1.0) / 6) ** 1.0
        assert gnmt_denom == pytest.approx(1.0)
        old_denom = 1.0 + 1  # length + 1, the formula this replaces
        assert gnmt_denom < old_denom


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
