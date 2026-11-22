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
    def test_raises_without_decoding_carry(self):
        state = CacheAwareRNNTBeamStreamingState()
        with pytest.raises(RuntimeError):
            state.select_best_beam_idx_(score_norm=True, length_norm_power=None)


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


# Token ids used as "word starts" in this test's fake vocabulary; every other id is a continuation
# sub-token. Chosen arbitrarily -- only membership in this set matters.
WORD_STARTS = {10, 12, 14}


def is_token_start_of_word(token_id: int) -> bool:
    return token_id in WORD_STARTS


def run_eou_cycle(state: CacheAwareRNNTBeamStreamingState, resume_global_frame: int) -> None:
    """Mirror `CacheAwareRNNTPipeline._apply_beam_update_`'s EOU-with-carryover path."""
    old_start = state._cumulative_tokens_len
    beam_size = len(state.partial_tokens) if state.partial_tokens is not None else None
    cum_tokens, cum_ts, cum_conf = state.refresh_publish_tokens()

    state.prepare_finalize(resume_global_frame, is_token_start_of_word=is_token_start_of_word)
    fold_at = old_start + len(state.tokens)
    state.set_eou_search_start(resume_global_frame)
    carryover = (state._carryover_tokens, state._carryover_timesteps, state._carryover_confidences)

    state.fold_cumulative_(cum_tokens, cum_ts, cum_conf, fold_at=fold_at, carryover=carryover, beam_size=beam_size)

    # Mirrors the pipeline: bpe-decode the finalized `state.tokens` here (skipped, not under test),
    # then clean up and restore the carryover as the next utterance's start.
    state.cleanup_after_eou()
    state.restore_carryover()


class TestCacheAwareRNNTBeamStreamingStateCarryover:
    """
    Regression coverage for the beam carryover fold-point fix: `CacheAwareRNNTBeamStreamingState`
    must not duplicate or drop tokens that survive past an EoU point when the underlying beam
    decoder's own `cumulative_*`/`partial_*` bookkeeping keeps growing across EoU cycles.

    Before the fix, `update_()` always folded `cumulative_tokens` at `len(cum_tokens)` (the full
    stream) regardless of where the carryover split fell, so any tokens manually restored into
    `state.tokens` for the next utterance were silently dropped again the moment the next chunk's
    `refresh_publish_tokens`/`fold_cumulative_` re-derived `state.tokens` from `cumulative_* +
    partial_*` -- the carryover was never reflected in that bookkeeping. This test's first two
    assertions after the second `append_chunk_beam_` call (`cum_tokens`/`state.tokens` below) are the
    ones that would fail under that unmodified behavior: `state.tokens` would come back as just
    `[14, 15]` (the carryover silently dropped) instead of `[12, 13, 14, 15]`.
    """

    @pytest.mark.unit
    def test_two_consecutive_eou_cycles_preserve_all_tokens_without_duplication(self):
        state = CacheAwareRNNTBeamStreamingState()
        state.hyp_decoding_state = object()  # sentinel: only None-checked by `_get_tokens`

        # Chunk 1: produces utterance-1 tokens [10, 11] plus utterance-2's opening tokens [12, 13],
        # all in a single beam (beam_size=1).
        state.append_chunk_beam_(
            chunk_tokens=[[10, 11, 12, 13]],
            chunk_timestamps=[[0, 1, 5, 6]],
            root_ptrs=[0],
            beam_size=1,
            best_hyp_idx=0,
        )
        # EoU fires with the resume point at token 12 (timestep 5): utterance 1 = [10, 11],
        # carryover (utterance 2's start) = [12, 13].
        run_eou_cycle(state, resume_global_frame=5)

        assert state.cumulative_tokens == [10, 11]
        assert state.tokens == [12, 13]
        assert state.timesteps == [5, 6]
        assert state.partial_tokens == [[12, 13]]

        # Chunk 2: the beam decoder keeps growing its own hypothesis from where it left off --
        # lineage-appends new tokens [14, 15] onto the (seeded) carryover in `partial_tokens`.
        state.append_chunk_beam_(
            chunk_tokens=[[14, 15]],
            chunk_timestamps=[[10, 11]],
            root_ptrs=[0],
            beam_size=1,
            best_hyp_idx=0,
        )
        cum_tokens, cum_ts, _ = state.refresh_publish_tokens()

        # No duplication: the carryover [12, 13] appears exactly once in the combined stream.
        assert cum_tokens == [10, 11, 12, 13, 14, 15]
        assert cum_ts == [0, 1, 5, 6, 10, 11]
        # No loss: the presentation copy is carryover + new tokens, not just the new chunk.
        assert state.tokens == [12, 13, 14, 15]

        # Second EoU fires with the resume point at token 14 (timestep 10): utterance 2 = [12, 13],
        # carryover (utterance 3's start) = [14, 15].
        run_eou_cycle(state, resume_global_frame=10)

        assert state.cumulative_tokens == [10, 11, 12, 13]
        assert state.tokens == [14, 15]
        assert state.timesteps == [10, 11]

    @pytest.mark.unit
    def test_full_fold_with_no_carryover_clears_partial_state(self):
        state = CacheAwareRNNTBeamStreamingState()
        state.hyp_decoding_state = object()
        state.append_chunk_beam_(
            chunk_tokens=[[10, 11]], chunk_timestamps=[[0, 1]], root_ptrs=[0], beam_size=1, best_hyp_idx=0
        )
        cum_tokens, cum_ts, cum_conf = state.refresh_publish_tokens()

        # No carryover (fold_at=None): mirrors the final-frame flush, where the whole utterance is
        # finalized and there is nothing left over for a next utterance.
        state.fold_cumulative_(cum_tokens, cum_ts, cum_conf)

        assert state.cumulative_tokens == [10, 11]
        assert state.partial_tokens is None
        assert state.best_hyp_idx is None
