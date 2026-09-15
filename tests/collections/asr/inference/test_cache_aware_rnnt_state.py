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
    def test_score_norm_without_prior_baseline_matches_plain_average(self):
        # No fold has happened yet (baselines are 0), so this degenerates to score / (length + 1).
        state = _state_with_carry(score=[-10.0, -6.0], length=[9.0, 1.0])
        # beam 0: -10/10 = -1.0 ; beam 1: -6/2 = -3.0 -> beam 0 wins
        assert state.select_best_beam_idx_(score_norm=True) == 0

    @pytest.mark.unit
    def test_stale_cumulative_baseline_dilutes_ranking_signal(self):
        """
        Reproduces the bug this fix addresses: without subtracting a per-utterance baseline, a long
        prior session (large cumulative score/length) swamps the normalized ranking, and the beam that
        is actually worse *for the current utterance* can win.
        """
        # Utterance 2 candidates, added on top of a long prior session (baseline not yet applied):
        # beam 0 (genuinely better): cumulative score/length = prior + (-5, 10)
        # beam 1 (genuinely worse):  cumulative score/length = prior + (-8, 10)
        prior_score, prior_length = -900.0, 2000.0
        state = _state_with_carry(
            score=[prior_score - 5.0, prior_score - 8.0],
            length=[prior_length + 10.0, prior_length + 10.0],
        )
        # Baseline intentionally left at 0 (as it would be pre-fix): the huge prior mass dominates both
        # ratios almost equally, and this assertion documents that the *raw* normalized signal alone
        # cannot reliably tell them apart once diluted -- both ratios round to the same 3 decimals.
        lengths = state.hyp_decoding_state.current_lengths_nb
        scores = state.hyp_decoding_state.score
        diluted_ranking = scores / (lengths + 1)
        assert abs(diluted_ranking[0].item() - diluted_ranking[1].item()) < 0.01

        # With the per-utterance baseline set to the prior session's mass, the same two candidates are
        # ranked correctly: beam 0 (Δscore=-5) beats beam 1 (Δscore=-8) for equal Δlength.
        state._score_baseline = prior_score
        state._length_baseline = prior_length
        assert state.select_best_beam_idx_(score_norm=True) == 0

    @pytest.mark.unit
    def test_length_norm_power_zero_disables_length_normalization(self):
        state = _state_with_carry(score=[-1.0, -5.0], length=[100.0, 1.0])
        # power=0 -> denom is always 1, so this reduces to raw (baselined) score comparison.
        assert state.select_best_beam_idx_(score_norm=True, length_norm_power=0.0) == 0

    @pytest.mark.unit
    def test_length_norm_power_default_is_plain_average(self):
        state = _state_with_carry(score=[-10.0, -6.0], length=[9.0, 1.0])
        assert state.select_best_beam_idx_(score_norm=True) == state.select_best_beam_idx_(
            score_norm=True, length_norm_power=1.0
        )

    @pytest.mark.unit
    def test_raises_without_decoding_carry(self):
        state = CacheAwareRNNTBeamStreamingState()
        with pytest.raises(RuntimeError):
            state.select_best_beam_idx_(score_norm=True)
