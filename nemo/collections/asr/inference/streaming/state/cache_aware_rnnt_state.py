# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

from __future__ import annotations

from typing import TYPE_CHECKING

from nemo.collections.asr.inference.streaming.state.cache_aware_state import CacheAwareStreamingState
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis

if TYPE_CHECKING:
    from nemo.collections.asr.parts.submodules.rnnt_malsd_batched_computer import MALSDStateItem


class CacheAwareRNNTStreamingState(CacheAwareStreamingState):
    """
    State of the cache aware RNNT streaming pipelines
    """

    def __init__(self):
        """
        Initialize the CacheAwareRNNTStreamingState
        """
        super().__init__()
        self._additional_params_reset()

    def reset(self) -> None:
        """
        Reset the state
        """
        super().reset()
        self._additional_params_reset()

    def _additional_params_reset(self) -> None:
        """
        Reset non-inherited parameters
        """
        super()._additional_params_reset()
        self.previous_hypothesis = None

    def set_previous_hypothesis(self, previous_hypothesis: Hypothesis) -> None:
        """
        Set the previous hypothesis
        Args:
            previous_hypothesis: (Hypothesis) The previous hypothesis to store for the next transcribe step
        """
        self.previous_hypothesis = previous_hypothesis

    def get_previous_hypothesis(self) -> Hypothesis | None:
        """
        Get the previous hypothesis
        Returns:
            (Hypothesis) The previous hypothesis
        """
        return self.previous_hypothesis

    def reset_previous_hypothesis(self) -> None:
        """
        Reset the previous hypothesis to None
        """
        self.previous_hypothesis = None


class CacheAwareRNNTBeamStreamingState(CacheAwareRNNTStreamingState):
    """Beam search streaming state; decoder carry + cumulative/partial tokens.

    ``hyp_decoding_state``: K-beam carry across chunks (collapsed to top1 on EOU in the pipeline).
    ``cumulative_*``: tokens/timestamps sealed at each EOU (prior utterances in a stream).
    ``partial_*[k]``: per-beam in-flight suffix since last EOU (chunk-local exports merged via lineage).
    ``best_hyp_idx``: index into ``partial_*`` for the chunk argmax beam used to publish.
    """

    def _additional_params_reset(self) -> None:
        super()._additional_params_reset()
        self.hyp_decoding_state: MALSDStateItem | None = None
        self.cumulative_tokens: list[int] = []
        self.cumulative_timestamps: list[int] = []
        self.cumulative_confidences: list[float] = []
        self.partial_tokens: list[list[int]] | None = None
        self.partial_timestamps: list[list[int]] | None = None
        self.partial_confidences: list[list[float]] | None = None
        self._cumulative_tokens_len: int = 0
        self.best_hyp_idx: int | None = None

    def reset_beam_decoding_state_(self) -> None:
        """Clear beam search carry and cumulative/partial tokens when a stream ends."""
        self.hyp_decoding_state = None
        self.cumulative_tokens = []
        self.cumulative_timestamps = []
        self.cumulative_confidences = []
        self.partial_tokens = None
        self.partial_timestamps = None
        self.partial_confidences = None
        self._cumulative_tokens_len = 0
        self.best_hyp_idx = None

    def append_chunk_beam_(
        self,
        chunk_tokens: list[list[int]],
        chunk_timestamps: list[list[int]],
        root_ptrs: list[int],
        beam_size: int,
        best_hyp_idx: int,
        chunk_confidences: list[list[float]] | None = None,
    ) -> None:
        """Append chunk-local beam exports into state."""
        prev_t = self.partial_tokens or [[] for _ in range(beam_size)]
        prev_ts = self.partial_timestamps or [[] for _ in range(beam_size)]
        prev_conf = self.partial_confidences or [[] for _ in range(beam_size)]
        next_tokens: list[list[int]] = []
        next_timestamps: list[list[int]] = []
        next_confidences: list[list[float]] = []
        for k in range(beam_size):
            lineage = int(root_ptrs[k])
            next_tokens.append(prev_t[lineage] + list(chunk_tokens[k]))
            next_timestamps.append(prev_ts[lineage] + list(chunk_timestamps[k]))
            if chunk_confidences is not None:
                next_confidences.append(prev_conf[lineage] + list(chunk_confidences[k]))
            else:
                next_confidences.append(prev_conf[lineage] + [0.0] * len(chunk_tokens[k]))
        self.partial_tokens = next_tokens
        self.partial_timestamps = next_timestamps
        self.partial_confidences = next_confidences
        self.best_hyp_idx = best_hyp_idx

    def select_best_beam_idx_(self, *, score_norm: bool = False, length_norm_power: float | None = None) -> int:
        """Pick beam index into ``partial_*``; updates ``best_hyp_idx``.

        Per-chunk publish uses raw ``scores.argmax`` (via ``append_chunk_beam_``). At EOU, use
        ``score_norm=True`` to rank by length-normalized score; with ``length_norm_power=None`` this
        matches offline :meth:`BatchedBeamHyps.flatten_sort_` exactly.

        ``score``/``current_lengths_nb`` on ``hyp_decoding_state`` are zeroed for the winning beam
        right after every EOU collapse (see ``reset_beam_score_``), so both this ranking and the raw
        per-chunk publish between EOUs already read values scoped to the current utterance -- no
        baseline needs subtracting here.

        ``length_norm_power`` selects the normalization:

        * ``None``: ``score / (length + 1)``, the plain average used by offline
          :meth:`BatchedBeamHyps.flatten_sort_`; use it for a direct streaming/offline comparison.
        * a float: ``score / ((5 + length) / 6) ** length_norm_power``, the GNMT-style length penalty
          (https://arxiv.org/abs/1609.08144). The ``5``/``6`` constants keep the penalty close to
          ``1.0`` for short hypotheses (where ``length + 1`` over-normalizes) while still approaching
          a plain length average as ``length`` grows. ``0.0`` disables length normalization entirely
          (raw score) -- this is unsafe with LM fusion, since an unnormalized RNNT+LM score can favor
          near-empty hypotheses.
        """
        if self.hyp_decoding_state is None:
            raise RuntimeError("Cannot select beam without decoding carry.")

        scores = self.hyp_decoding_state.score
        lengths_nb = self.hyp_decoding_state.current_lengths_nb
        if not score_norm:
            ranking = scores
        elif length_norm_power is None:
            ranking = scores / (lengths_nb.to(dtype=scores.dtype) + 1)
        else:
            denom = ((5 + lengths_nb.to(dtype=scores.dtype)) / 6) ** length_norm_power
            ranking = scores / denom

        self.best_hyp_idx = int(ranking.argmax().item())
        return self.best_hyp_idx

    def reset_beam_score_(self) -> None:
        """
        Zero the winning beam's cumulative score/length right after an EOU beam collapse.

        Resetting the actual ``hyp_decoding_state.score``/``current_lengths_nb`` carry here, instead
        of only snapshotting a baseline for the next ``select_best_beam_idx_(score_norm=True)`` call,
        means every reader of that carry -- EOU ranking *and* the raw per-chunk publish that runs
        between EOUs -- sees values scoped to the current utterance, not a session-long accumulator.
        """
        if self.hyp_decoding_state is None:
            return
        # `select_beam_in_state_item_` builds these tensors inside `torch.inference_mode()`, so an
        # in-place write here (outside that context) needs a clone first.
        self.hyp_decoding_state.score = self.hyp_decoding_state.score.clone()
        self.hyp_decoding_state.score[0] = 0.0
        if self.hyp_decoding_state.current_lengths_nb is not None:
            self.hyp_decoding_state.current_lengths_nb = self.hyp_decoding_state.current_lengths_nb.clone()
            self.hyp_decoding_state.current_lengths_nb[0] = 0

    def get_best_hyp_idx(self) -> int:
        """Index into ``partial_*`` for publish (chunk argmax, or score argmax from carry)."""
        if self.best_hyp_idx is not None:
            return int(self.best_hyp_idx)
        if self.hyp_decoding_state is None:
            raise RuntimeError("Cannot resolve top-1 beam index without decoding carry.")
        return int(self.hyp_decoding_state.score.argmax().item())

    def _get_tokens(self) -> tuple[list[int], list[int], list[float]]:
        """``cumulative_*`` plus the current top-1 ``partial_*`` suffix."""
        if self.partial_tokens is None or self.hyp_decoding_state is None:
            return [], [], []
        best_hyp_idx = self.get_best_hyp_idx()
        partial_tokens = list(self.partial_tokens[best_hyp_idx])
        partial_conf = (
            list(self.partial_confidences[best_hyp_idx])
            if self.partial_confidences is not None
            else [0.0] * len(partial_tokens)
        )
        return (
            self.cumulative_tokens + partial_tokens,
            self.cumulative_timestamps + list(self.partial_timestamps[best_hyp_idx]),
            self.cumulative_confidences + partial_conf,
        )

    def get_hypothesis(self, score: float) -> Hypothesis:
        """Build the publishable cumulative hypothesis for the current top-1 beam."""
        cum_tokens, cum_ts, cum_conf = self._get_tokens()
        return Hypothesis(
            score=score,
            y_sequence=cum_tokens,
            timestamp=cum_ts,
            length=len(cum_tokens),
            non_blank_step_confidence_precomputed=cum_conf if cum_conf else None,
        )

    def refresh_publish_tokens(self) -> tuple[list[int], list[int], list[float]]:
        """
        Refresh ``self.tokens/timesteps/confidences`` (the tokens since the last fold) from the
        current beam's cumulative+partial stream, without folding.
        Returns:
            tuple[list[int], list[int], list[float]]: the full ``cumulative_* + partial_*`` stream
                (``cum_tokens``, ``cum_timestamps``, ``cum_confidences``) as of this call.
        """
        cum_tokens, cum_ts, cum_conf = self._get_tokens()
        if cum_tokens:
            start = max(0, min(int(self._cumulative_tokens_len), len(cum_tokens)))
            tokens = list(cum_tokens[start:])
            timesteps = list(cum_ts[start:])
            confidences = list(cum_conf[start:]) if cum_conf else [0.0] * len(tokens)
            self.tokens = tokens
            self.timesteps = timesteps
            self.confidences = confidences
            if tokens:
                self.last_token = tokens[-1]
                self.last_token_idx = timesteps[-1] if timesteps else None
        return cum_tokens, cum_ts, cum_conf

    def fold_cumulative_(
        self,
        cum_tokens: list[int],
        cum_timestamps: list[int],
        cum_confidences: list[float],
        fold_at: int | None = None,
        carryover: tuple[list[int], list[int], list[float]] | None = None,
        beam_size: int | None = None,
    ) -> None:
        """
        Fold the utterance into ``cumulative_*`` at ``fold_at`` (an index into ``cum_tokens``).

        ``partial_*`` (the beam decoder's own per-beam in-flight bookkeeping) is NEVER left holding
        content that overlaps ``cumulative_*`` -- growing it via lineage (`append_chunk_beam_`) would
        otherwise duplicate whatever both sides already contain. So:
          - No carryover (``fold_at`` is None, or empty ``carryover``): a full fold, matching a plain
            EOU with nothing pending (e.g. the final frame of a session). ``cumulative_*`` absorbs the
            whole stream and ``partial_*``/``best_hyp_idx`` are cleared.
          - Carryover survives past the EoU point (``fold_at`` leaves a suffix of ``cum_tokens``
            un-folded): ``cumulative_*`` absorbs only the prefix up to ``fold_at``. ``partial_*`` is
            RESET and re-seeded with the carryover content for every beam slot (rather than left as
            the old, now-overlapping content) -- the underlying decoder's live hypothesis lattice was
            already collapsed to a single (top-1) hypothesis on this EOU (see
            ``select_beam_in_state_item_`` in the pipeline), so every beam slot re-diversifies from
            that same carryover point going forward, and the next `append_chunk_beam_` call correctly
            appends new tokens after the (uniformly-seeded) carryover with no duplication.
        Args:
            cum_tokens: (list[int]) the full ``cumulative_* + partial_*`` token stream (from
                `refresh_publish_tokens`).
            cum_timestamps: (list[int]) matching timestamps.
            cum_confidences: (list[float]) matching confidences.
            fold_at: (int | None) index into `cum_tokens` to fold at; None folds the whole stream.
            carryover: (tuple[list[int], list[int], list[float]] | None) the carryover
                (tokens, timestamps, confidences) surviving past `fold_at`, if any.
            beam_size: (int | None) number of beam slots to seed `partial_*` with; defaults to 1.
        """
        if not cum_tokens:
            return
        if fold_at is None:
            fold_at = len(cum_tokens)
        fold_at = max(0, min(int(fold_at), len(cum_tokens)))

        self._cumulative_tokens_len = fold_at
        self.cumulative_tokens = list(cum_tokens[:fold_at])
        self.cumulative_timestamps = list(cum_timestamps[:fold_at])
        self.cumulative_confidences = list(cum_confidences[:fold_at])

        carryover_tokens = carryover[0] if carryover else []
        if fold_at >= len(cum_tokens) or not carryover_tokens:
            # No carryover survives past the fold point: safe to reset the beam's own bookkeeping.
            self.partial_tokens = None
            self.partial_timestamps = None
            self.partial_confidences = None
            self.best_hyp_idx = None
            return

        carryover_timestamps, carryover_confidences = carryover[1], carryover[2]
        beam_size = beam_size or 1
        self.partial_tokens = [list(carryover_tokens) for _ in range(beam_size)]
        self.partial_timestamps = [list(carryover_timestamps) for _ in range(beam_size)]
        self.partial_confidences = [list(carryover_confidences) for _ in range(beam_size)]
        self.best_hyp_idx = 0

    def update_(self, eou_detected: bool) -> None:
        """
        Refresh publish tokens; on EOU fold the whole utterance into ``cumulative_*`` and clear
        ``partial_*``. Convenience wrapper for the no-carryover case (e.g. final-frame flush); the
        carryover-aware path calls `refresh_publish_tokens`/`fold_cumulative_` directly with an
        explicit `fold_at`.
        """
        cum_tokens, cum_ts, cum_conf = self.refresh_publish_tokens()
        if eou_detected:
            self.fold_cumulative_(cum_tokens, cum_ts, cum_conf)
