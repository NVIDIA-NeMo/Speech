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
from dataclasses import dataclass
from typing import Optional

import torch

from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis, NBestHypotheses
from nemo.utils.enum import PrettyStrEnum

# Constants used for hashing text sequences.
MULTIPLIER = 6364136223846793005
INCREMENT = 1
MODULUS = 2**64

# Constants used for initializing and managing beam search hypotheses.
INACTIVE_SCORE = -float("inf")  # Represents the score of inactive hypotheses.
INIT_POINTER_VALUE = -1  # Initial value for pointers in the hypothesis tree structure.
INIT_HASH_VALUE = 0  # Initial hash value for transcript hashes.
INIT_PREFIX_HASH_VALUE = 0  # Initial hash value for prefix hashes.
NON_EXISTENT_LABEL_VALUE = -1  # Placeholder value for non-existent labels in hypotheses. Needs to be negative.


def hash_text(prev_hash: torch.Tensor, add_labels: torch.Tensor) -> torch.Tensor:
    """
    Computes a new hash value by updating previous hash tensor with added labels tensor.
    Reference: https://stackoverflow.com/a/77213071

    Args:
        prev_hash (torch.Tensor): A tensor representing the previous hash value.
        add_labels (torch.Tensor): A tensor containing added labels.

    Returns:
        torch.Tensor: A tensor representing the updated hash value.
    """
    return prev_hash * MULTIPLIER + INCREMENT + add_labels


class BlankLMScoreMode(PrettyStrEnum):
    """
    Defines the strategies for handling blank token scores in a external Ngram LM
    when combined with an automatic speech recognition (ASR) model.
    """

    # No score for blank.
    NO_SCORE = "no_score"
    # Blank score for LM is set equal to blank score from ASR model; non-blank LM scores are reweighted to sum to 1.
    LM_WEIGHTED_FULL = "lm_weighted_full"


class PruningMode(PrettyStrEnum):
    """Specifies when pruning is applied external Ngram LM shallow fusion.."""

    # Hyps are pruned based on ASR probs, then rescored with LM
    EARLY = "early"
    # Hyps are scored based on combined ASR and LM probs., then pruned
    LATE = "late"


class ASRModelTypeEnum(PrettyStrEnum):
    """Specifies model type."""

    RNNT = "rnnt"
    TDT = "tdt"
    CTC = "ctc"


@dataclass
class BatchedBeamHypsState:
    """Per-beam state carried across streaming chunk boundaries."""

    scores: torch.Tensor
    last_label: torch.Tensor
    transcript_hash: torch.Tensor
    current_lengths_nb: torch.Tensor
    last_timestamp_lasts: Optional[torch.Tensor] = None
    transcript_prefix_hash: Optional[torch.Tensor] = None


def seed_batched_hyps_from_state(
    hyps: "BatchedBeamHyps",
    state: BatchedBeamHypsState,
    batch_size: Optional[int] = None,
) -> None:
    """Copy cross-chunk per-beam fields from a :class:`BatchedBeamHypsState` snapshot
    into ``hyps`` (in-place). Inverse of
    :meth:`BatchedBeamHyps.export_cross_chunk_state`.

    Used by streaming beam-search decoders to seed a ``BatchedBeamHyps`` from the previous
    chunk's snapshot. Chunk-local buffers (prefix tree / timestamps / write cursor)
    and the per-beam time cursor are NOT touched -- the caller is responsible for
    wiping them.

    Args:
        hyps: destination ``BatchedBeamHyps`` (modified in place).
        state: source snapshot.
        batch_size: optional number of leading rows to copy. Defaults to
            ``state.scores.shape[0]``.
    """
    bs = state.scores.shape[0] if batch_size is None else batch_size
    hyps.scores[:bs].copy_(state.scores[:bs])
    hyps.last_label[:bs].copy_(state.last_label[:bs])
    hyps.transcript_hash[:bs].copy_(state.transcript_hash[:bs])
    hyps.current_lengths_nb[:bs].copy_(state.current_lengths_nb[:bs])
    if hyps.store_prefix_hashes and state.transcript_prefix_hash is not None:
        hyps.transcript_prefix_hash[:bs].copy_(state.transcript_prefix_hash[:bs])
    if hyps.model_type != ASRModelTypeEnum.CTC and state.last_timestamp_lasts is not None:
        hyps.last_timestamp_lasts[:bs].copy_(state.last_timestamp_lasts[:bs])


class BatchedBeamHyps:
    """Class to store batch of beam hypotheses (labels, time_indices, scores) for efficient batched beam decoding"""

    def __init__(
        self,
        batch_size: int,
        beam_size: int,
        init_length: int,
        blank_index: int,
        device: torch.device = None,
        float_dtype: torch.dtype = None,
        store_prefix_hashes: Optional[bool] = False,
        model_type: Optional[ASRModelTypeEnum | str] = ASRModelTypeEnum.RNNT,
    ):
        """
        Initializes the batched beam hypotheses utility for Transducer decoding (RNN-T and TDT models).
        Args:
            batch_size (int): Batch size.
            beam_size (int): Beam size.
            init_length (int): The initial maximum length of the hypotheses.
            blank_index (int): The index representing the blank token in the vocabulary.
            device (torch.device): The device on which tensors will be allocated. Defaults to None.
            float_dtype (torch.dtype): The floating-point data type. Defaults to None.
            store_prefix_hashes (bool, optional): Whether to store prefix hashes for hypotheses. Defaults to False.
            model_type: (str or ModelTypeEnum, optional): Model type, either 'rnnt', 'tdt' or 'ctc'. Defaults to 'rnnt'.
        """

        if beam_size <= 0:
            raise ValueError("Beam size must be greater than 0.")
        if batch_size <= 0:
            raise ValueError("Batch size must be greater than 0.")
        if init_length <= 0:
            raise ValueError("Initial hypothesis lengths must be greater than 0.")

        self.device = device
        self.INACTIVE_SCORE_TENSOR = torch.tensor(INACTIVE_SCORE, device=device, dtype=float_dtype)
        self.ZERO_TENSOR = torch.tensor(0, device=device, dtype=torch.long)

        self.model_type = ASRModelTypeEnum(model_type)
        self.store_prefix_hashes = store_prefix_hashes
        self._max_length = init_length
        self.beam_size = beam_size
        self.blank_index = blank_index
        self.batch_size = batch_size
        self.batch_indices = torch.arange(self.batch_size, device=device)
        self.beam_indices = torch.arange(self.beam_size, device=device)

        # Non-blank (non-blank and non-repeating for CTC) and full lengths
        self.current_lengths_nb = torch.zeros([batch_size, self.beam_size], device=device, dtype=torch.long)
        self.current_lengths_wb = torch.zeros([batch_size, self.beam_size], device=device, dtype=torch.long)

        # Initializing tree structure for hypothesis storing
        self.transcript_wb = torch.full(
            (batch_size, self.beam_size, self._max_length),
            fill_value=NON_EXISTENT_LABEL_VALUE,
            device=device,
            dtype=torch.long,
        )  # current labels
        self.transcript_wb_prev_ptr = torch.full(
            (batch_size, self.beam_size, self._max_length),
            fill_value=INIT_POINTER_VALUE,
            device=device,
            dtype=torch.long,
        )  # links to prefices

        # Initializing beam scores: Initially, only a single hypothesis is active within the beam.
        self.scores = torch.full(
            [batch_size, self.beam_size], device=device, dtype=float_dtype, fill_value=INACTIVE_SCORE
        )
        self.scores[:, 0].fill_(0.0)

        self.last_label = torch.full(
            (batch_size, self.beam_size), fill_value=NON_EXISTENT_LABEL_VALUE, device=device, dtype=torch.long
        )

        self.transcript_hash = torch.full(
            [batch_size, self.beam_size], device=device, dtype=torch.long, fill_value=INIT_HASH_VALUE
        )
        if store_prefix_hashes:
            self.transcript_prefix_hash = torch.full(
                [batch_size, self.beam_size], device=device, dtype=torch.long, fill_value=INIT_PREFIX_HASH_VALUE
            )

        if self.model_type == ASRModelTypeEnum.CTC:
            # CTC frames and tokens are aligned, so we can precompute timestamps
            self.timestamps = self._create_timestamps_tensor(self._max_length)  # timestamps
        else:
            # timestamps for transducer models
            self.timestamps = torch.zeros(
                (batch_size, self.beam_size, self._max_length), device=device, dtype=torch.long
            )  # timestamps

            # tracking last frame index and number of labels for the last frama
            self.next_timestamp = torch.zeros((batch_size, self.beam_size), device=device, dtype=torch.long)
            self.last_timestamp_lasts = torch.zeros((batch_size, self.beam_size), device=device, dtype=torch.long)

    def clear_(self):
        """
        Clears and resets the internal state of the object.
        """

        self.current_lengths_nb.fill_(0)
        self.current_lengths_wb.fill_(0)

        self.transcript_wb.fill_(NON_EXISTENT_LABEL_VALUE)
        self.transcript_wb_prev_ptr.fill_(INIT_POINTER_VALUE)

        self.scores.fill_(INACTIVE_SCORE)
        self.scores[:, 0].fill_(0.0)

        self.last_label.fill_(NON_EXISTENT_LABEL_VALUE)

        self.transcript_hash.fill_(INIT_HASH_VALUE)
        if self.store_prefix_hashes:
            self.transcript_prefix_hash.fill_(INIT_PREFIX_HASH_VALUE)

        # model specific parameters
        if self.model_type == ASRModelTypeEnum.CTC:
            self.timestamps.copy_(self._create_timestamps_tensor(self._max_length))
        else:
            self.timestamps.fill_(0)
            self.next_timestamp.fill_(0)
            self.last_timestamp_lasts.fill_(0)

    def export_cross_chunk_state(self, batch_size: Optional[int] = None) -> "BatchedBeamHypsState":
        """
        Snapshot the cross-chunk per-beam state into a :class:`BatchedBeamHypsState`.

        Exports the fields needed to seed beam search on the next chunk (scores,
        last_label, transcript_hash, current_lengths_nb, and optionally
        last_timestamp_lasts / transcript_prefix_hash). The chunk-local prefix tree
        and timestamps are NOT exported -- they reach the caller via the first
        element of the computer's 3-tuple return.

        Args:
            batch_size: optional number of rows to export. Defaults to
                ``self.batch_size``. Used by the CUDA-graphs paths to trim
                capture-time buffers down to the live batch.

        Returns:
            A new :class:`BatchedBeamHypsState` with cloned tensors (safe to keep
            across subsequent in-place mutations of this object).
        """
        out_batch = self.batch_size if batch_size is None else batch_size
        if out_batch <= 0 or out_batch > self.batch_size:
            raise ValueError(f"batch_size must be in (0, {self.batch_size}], got {out_batch}")
        return BatchedBeamHypsState(
            scores=self.scores[:out_batch].clone(),
            last_label=self.last_label[:out_batch].clone(),
            transcript_hash=self.transcript_hash[:out_batch].clone(),
            current_lengths_nb=self.current_lengths_nb[:out_batch].clone(),
            last_timestamp_lasts=(
                self.last_timestamp_lasts[:out_batch].clone() if self.model_type != ASRModelTypeEnum.CTC else None
            ),
            transcript_prefix_hash=(
                self.transcript_prefix_hash[:out_batch].clone() if self.store_prefix_hashes else None
            ),
        )

    def clone(self, batch_size: Optional[int] = None) -> "BatchedBeamHyps":
        """
        Create a deep copy of this BatchedBeamHyps object.

        Args:
            batch_size: optional output batch size. If provided, must satisfy
                ``1 <= batch_size <= self.batch_size``, and the returned object
                holds a deep copy of the first ``batch_size`` rows. Defaults to
                ``self.batch_size`` (i.e. a full copy). Used by streaming/chunked
                decoding to trim graph-captured buffers (sized at the capture-time
                max) down to the live batch.

        Returns:
            New BatchedBeamHyps with copied state.
        """
        out_batch = self.batch_size if batch_size is None else batch_size
        if out_batch <= 0 or out_batch > self.batch_size:
            raise ValueError(f"batch_size must be in [1, {self.batch_size}], got {out_batch}")
        new_hyps = BatchedBeamHyps(
            batch_size=out_batch,
            beam_size=self.beam_size,
            init_length=self._max_length,
            blank_index=self.blank_index,
            device=self.device,
            float_dtype=self.scores.dtype,
            store_prefix_hashes=self.store_prefix_hashes,
            model_type=self.model_type,
        )
        # Destination is freshly allocated at exactly [out_batch, beam_size, _max_length],
        # so we can copy whole rows of `self` directly without per-axis trimming.
        new_hyps.current_lengths_nb.copy_(self.current_lengths_nb[:out_batch])
        new_hyps.current_lengths_wb.copy_(self.current_lengths_wb[:out_batch])
        new_hyps.transcript_wb.copy_(self.transcript_wb[:out_batch])
        new_hyps.transcript_wb_prev_ptr.copy_(self.transcript_wb_prev_ptr[:out_batch])
        new_hyps.scores.copy_(self.scores[:out_batch])
        new_hyps.last_label.copy_(self.last_label[:out_batch])
        new_hyps.transcript_hash.copy_(self.transcript_hash[:out_batch])
        if self.store_prefix_hashes:
            new_hyps.transcript_prefix_hash.copy_(self.transcript_prefix_hash[:out_batch])
        new_hyps.timestamps.copy_(self.timestamps[:out_batch])
        if self.model_type != ASRModelTypeEnum.CTC:
            new_hyps.next_timestamp.copy_(self.next_timestamp[:out_batch])
            new_hyps.last_timestamp_lasts.copy_(self.last_timestamp_lasts[:out_batch])
        return new_hyps

    def get_last_labels(self, pad_id: int = -1) -> torch.Tensor:
        """
        Get last labels for each hypothesis in the beam.

        Args:
            pad_id: Value to use for padding (for hypotheses without labels). Defaults to -1.

        Returns:
            Tensor of shape [batch_size, beam_size] with the last label for each hypothesis.
        """
        # last_label already contains the last label for each beam
        # Replace NON_EXISTENT_LABEL_VALUE with pad_id
        return torch.where(self.last_label != NON_EXISTENT_LABEL_VALUE, self.last_label, pad_id)

    def _allocate_more(self):
        """
        Dynamically allocates more memory for the internal buffers.
        This method doubles the size of the following tensors: `transcript_wb`, `transcript_wb_prev_ptr`.
        """
        self.transcript_wb = torch.cat(
            (self.transcript_wb, torch.full_like(self.transcript_wb, fill_value=NON_EXISTENT_LABEL_VALUE)), dim=-1
        )
        self.transcript_wb_prev_ptr = torch.cat(
            (self.transcript_wb_prev_ptr, torch.full_like(self.transcript_wb_prev_ptr, fill_value=INIT_POINTER_VALUE)),
            dim=-1,
        )
        if self.model_type == ASRModelTypeEnum.CTC:
            self.timestamps = self._create_timestamps_tensor(2 * self._max_length)
        else:
            self.timestamps = torch.cat((self.timestamps, torch.zeros_like(self.timestamps)), dim=-1)

        self._max_length *= 2

    def add_results_(
        self,
        next_indices: torch.Tensor,
        next_labels: torch.Tensor,
        next_hyps_prob: torch.Tensor,
        next_label_durations: Optional[torch.Tensor] = None,
    ):
        """
        Updates batch of beam hypotheses with labels. If the maximum allowed length
        is exceeded, underlying memory is doubled.
        Args:
            next_indices (torch.Tensor): Indices of the hypotheses to be updated.
            next_labels (torch.Tensor): Labels corresponding to the next step in the beam search.
            next_hyps_prob (torch.Tensor): Probabilities of the next hypotheses.
            next_label_durations (torch.Tensor, optional): Durations associated with the next labels. Required when `model_type='tdt'`.
        """

        if self.model_type == ASRModelTypeEnum.TDT and next_label_durations is None:
            raise ValueError("`next_label_durations` is required when model type is TDT.")

        if (self.current_lengths_wb + 1).max() >= self._max_length:
            self._allocate_more()

        self.add_results_no_checks_(
            next_indices=next_indices,
            next_labels=next_labels,
            next_hyps_prob=next_hyps_prob,
            next_label_durations=next_label_durations,
        )

    def add_results_no_checks_(
        self,
        next_indices: torch.Tensor,
        next_labels: torch.Tensor,
        next_hyps_prob: torch.Tensor,
        next_label_durations: Optional[torch.Tensor] = None,
    ):
        """
        Updates batch of beam hypotheses with labels.
        Args:
            next_indices (torch.Tensor): Indices of the hypotheses to be updated.
            next_labels (torch.Tensor): Labels corresponding to the next step in the beam search.
            next_hyps_prob (torch.Tensor): Probabilities of the next hypotheses.
            next_label_durations (torch.Tensor, optional): Durations associated with the next labels. Required when `model_type='tdt'`.
        """
        if self.model_type == ASRModelTypeEnum.TDT and next_label_durations is None:
            raise ValueError("`next_label_durations` is required when model type is TDT.")

        last_labels = torch.gather(self.last_label, dim=-1, index=next_indices)
        self.transcript_wb.scatter_(dim=-1, index=self.current_lengths_wb.unsqueeze(-1), src=next_labels.unsqueeze(-1))
        self.transcript_wb_prev_ptr.scatter_(
            dim=-1, index=self.current_lengths_wb.unsqueeze(-1), src=next_indices.unsqueeze(-1)
        )

        is_extended = next_labels >= 0
        extended_with_blank = next_labels == self.blank_index
        extended_with_label = (is_extended) & (~extended_with_blank)

        if self.model_type == ASRModelTypeEnum.CTC:
            # for CTC last non-blank and non-repeated label
            extended_with_label = (extended_with_label) & (next_labels != last_labels)  # non-repeated non-blank label

        if self.model_type == ASRModelTypeEnum.RNNT:
            timesteps = torch.gather(self.next_timestamp, dim=-1, index=next_indices)
            self.timestamps.scatter_(
                dim=-1,
                index=self.current_lengths_wb.unsqueeze(-1),
                src=(timesteps + extended_with_blank).unsqueeze(-1),
            )
            self.next_timestamp.copy_(timesteps + extended_with_blank)
            torch.where(
                extended_with_blank,
                self.ZERO_TENSOR,
                torch.gather(self.last_timestamp_lasts, dim=-1, index=next_indices) + extended_with_label,
                out=self.last_timestamp_lasts,
            )
        elif self.model_type == ASRModelTypeEnum.TDT:
            timesteps = torch.gather(self.next_timestamp, dim=-1, index=next_indices)
            next_label_durations = torch.where(is_extended, next_label_durations, 0)
            self.timestamps.scatter_(
                dim=-1,
                index=self.current_lengths_wb.unsqueeze(-1),
                src=(timesteps + next_label_durations).unsqueeze(-1),
            )
            torch.where(is_extended, timesteps + next_label_durations, timesteps, out=self.next_timestamp)
            torch.where(
                is_extended & (next_label_durations > 0),
                self.ZERO_TENSOR,
                torch.gather(self.last_timestamp_lasts, dim=-1, index=next_indices) + extended_with_label,
                out=self.last_timestamp_lasts,
            )

        self.current_lengths_nb.copy_(
            torch.gather(self.current_lengths_nb, dim=-1, index=next_indices) + extended_with_label
        )
        torch.add(self.current_lengths_wb, 1, out=self.current_lengths_wb)
        self.scores.copy_(next_hyps_prob)

        prev_transcript_hash = torch.gather(self.transcript_hash, dim=-1, index=next_indices)
        # update hashes and prefix hashes
        torch.where(
            extended_with_label,
            hash_text(prev_transcript_hash, next_labels),
            prev_transcript_hash,
            out=self.transcript_hash,
        )

        if self.model_type == ASRModelTypeEnum.CTC:
            # track last label
            torch.where(is_extended, next_labels, last_labels, out=self.last_label)
        else:
            # track last non-blank label
            torch.where(extended_with_label, next_labels, last_labels, out=self.last_label)

        # store prefix hashes for batched maes
        if self.store_prefix_hashes:
            prev_transcript_prefix_hash = torch.gather(self.transcript_prefix_hash, dim=-1, index=next_indices)
            torch.where(
                extended_with_label, prev_transcript_hash, prev_transcript_prefix_hash, out=self.transcript_prefix_hash
            )

    def recombine_hyps_(self):
        """
        Recombines hypotheses in the beam search by merging equivalent hypotheses and updating their scores.
        This method identifies hypotheses that are equivalent based on their transcript hash, last label,
        and current lengths. It then merges these equivalent hypotheses by computing a new score using
        log-sum-exp over their scores and updates the scores tensor accordingly.
        Returns:
            Note: The method modifies the `self.scores` tensor in place to reflect the recombined hypotheses.
        """

        if self.beam_size <= 1:
            return

        hyps_equal = (
            (self.transcript_hash[:, :, None] == self.transcript_hash[:, None, :])
            & (self.last_label[:, :, None] == self.last_label[:, None, :])
            & (self.current_lengths_nb[:, :, None] == self.current_lengths_nb[:, None, :])
        )

        if self.model_type == ASRModelTypeEnum.TDT:
            hyps_equal &= self.next_timestamp[:, :, None] == self.next_timestamp[:, None, :]

        scores_matrix = torch.where(
            hyps_equal,
            self.scores[:, None, :].expand(self.batch_size, self.beam_size, self.beam_size),
            self.INACTIVE_SCORE_TENSOR,
        )

        scores_argmax = scores_matrix.argmax(-1, keepdim=False)
        scores_to_keep = (
            torch.arange(self.beam_size, device=scores_argmax.device, dtype=torch.long)[None, :] == scores_argmax
        )
        if self.model_type == ASRModelTypeEnum.CTC:
            new_scores = torch.max(scores_matrix, dim=-1, keepdim=False).values
        else:
            new_scores = torch.logsumexp(scores_matrix, dim=-1, keepdim=False)

        torch.where(scores_to_keep, new_scores.to(self.scores.dtype), self.INACTIVE_SCORE_TENSOR, out=self.scores)

    def remove_duplicates(self, labels: torch.Tensor, total_logps: torch.Tensor):
        """
        Removes duplicate hypotheses that may arise after updating beam hypotheses with labels during the beam search process.
        Args:
            labels (torch.Tensor): A tensor containing the labels for the current beam
                search step. Shape: [batch_size, beam_size, ...].
            total_logps (torch.Tensor): A tensor containing the total log probabilities
                for the current beam search step. Shape: [batch_size, beam_size, ...].
        Returns:
            torch.Tensor: Updated total log probabilities with duplicates removed.
                Shape: [batch_size, beam_size, ...].
        """

        if self.beam_size <= 1:
            return total_logps

        # updating hashes for label expansions
        non_blank_mask = labels != self.blank_index
        expansion_hashes = hash_text(self.transcript_hash.unsqueeze(-1), labels)
        expansion_hashes = torch.where(non_blank_mask, expansion_hashes, self.transcript_hash.unsqueeze(-1)).view(
            self.batch_size, -1
        )

        # masking inactive hypotheses
        inactive_hyps_mask = self.scores != INACTIVE_SCORE
        masked_hashes = torch.where(inactive_hyps_mask, self.transcript_hash, -1)

        init_expansions_equal = (expansion_hashes[:, :, None] == masked_hashes[:, None, :]).any(dim=-1)

        init_expansions_equal = torch.logical_and(non_blank_mask.view(self.batch_size, -1), init_expansions_equal)
        expansions_equal = expansion_hashes[:, :, None] == expansion_hashes[:, None, :]
        expansion_scores = total_logps.view(self.batch_size, -1)
        expansion_scores = torch.where(init_expansions_equal, INACTIVE_SCORE, expansion_scores)
        expansion_scores = expansion_scores[:, None, :].expand(expansions_equal.shape)

        expansion_scores = torch.where(expansions_equal, expansion_scores, INACTIVE_SCORE)
        expansion_scores, expansion_scores_argmax = expansion_scores.max(dim=-1)

        scores_range = torch.arange(
            expansion_scores_argmax.shape[-1], device=expansion_scores_argmax.device, dtype=torch.long
        )
        scores_to_keep = scores_range[None, :] == expansion_scores_argmax
        total_logps = torch.where(scores_to_keep, expansion_scores, INACTIVE_SCORE).view(
            self.batch_size, self.beam_size, -1
        )

        return total_logps

    def recombine_prefixes(self, label_logps: torch.Tensor, active_mask: torch.Tensor):
        """
        Recombines prefixes (prefix search) in the beam search process by updating scores for hypotheses
        that share common prefixes.
        Args:
            label_logps (torch.Tensor): A tensor of shape (batch_size, beam_size, vocab_size)
                containing the log probabilities of the labels for each beam.
            active_mask (torch.Tensor): A boolean tensor of shape (batch_size, beam_size)
                indicating which beams are active.
        """

        if self.beam_size <= 1:
            return

        # if hypotheses are empty skip
        if (self.current_lengths_wb == 0).any():
            return

        # mask prefix hashes if hypotheses of the beam do not have prefixes (e.g. no non-blank labels were appended)
        prefix_hashes = torch.where(self.current_lengths_nb == 0, -2, self.transcript_prefix_hash)

        prefix_equal = self.transcript_hash[:, None, :] == prefix_hashes[:, :, None]

        last_labels = torch.where(self.last_label == NON_EXISTENT_LABEL_VALUE, self.blank_index, self.last_label)
        prefix_labels = last_labels.unsqueeze(1).repeat((1, self.beam_size, 1))
        prefix_scores = self.scores.unsqueeze(1).repeat((1, self.beam_size, 1))

        prefix_label_logps = torch.gather(label_logps, dim=-1, index=prefix_labels)
        prefix_label_logps = prefix_scores + prefix_label_logps.transpose(dim0=-1, dim1=-2)
        prefix_label_logps = torch.where(prefix_equal, prefix_label_logps, INACTIVE_SCORE)
        prefix_label_logps = torch.logsumexp(prefix_label_logps, dim=-1)

        to_update_mask = torch.logical_and(active_mask, self.scores != INACTIVE_SCORE)
        self.scores = torch.where(to_update_mask, torch.logaddexp(self.scores, prefix_label_logps), self.scores)

    def to_hyps_list(self, score_norm: bool = True) -> list[Hypothesis]:
        """
        Converts the batched beam search results into a list of signle best hypotheses for each batch.
        Args:
            score_norm (bool):  If True, normalize the scores before sorting. Defaults to True.
        Returns:
            list[Hypothesis]: A list where each element corresponds to a batch and contains
            best hypothesis.
        """
        self.flatten_sort_(score_norm)

        scores = self.scores[self.batch_indices, 0].tolist()

        max_idx = self.current_lengths_wb.max() - 1
        timestamps = self.timestamps[..., 0, : max_idx + 1]
        transcripts = self.transcript_wb[..., 0, : max_idx + 1]
        hypotheses = [
            Hypothesis(
                score=scores[batch_idx],
                y_sequence=transcripts[batch_idx][mask := self._create_transcripts_mask(transcripts[batch_idx])]
                .cpu()
                .detach()
                .numpy(),
                timestamp=timestamps[batch_idx][mask].cpu().detach().numpy(),
                alignments=None,
                dec_state=None,
            )
            for batch_idx in range(self.batch_size)
        ]
        return hypotheses

    def to_nbest_hyps_list(self, score_norm: bool = True) -> list[NBestHypotheses]:
        """
        Converts the batched beam search results into a list of N-best hypotheses for each batch.
        Args:
            score_norm (bool, optional): If True, normalize the scores before sorting. Defaults to True.
        Returns:
            list[NBestHypotheses]: A list where each element corresponds to a batch and contains
            N-best hypotheses.
        """

        self.flatten_sort_(score_norm)

        scores = self.scores.tolist()

        max_idx = self.current_lengths_wb.max() - 1
        transcripts = self.transcript_wb[..., : max_idx + 1]
        timestamps = self.timestamps[..., : max_idx + 1]
        hypotheses = [
            NBestHypotheses(
                [
                    Hypothesis(
                        score=scores[batch_idx][beam_idx],
                        y_sequence=transcripts[batch_idx][beam_idx][
                            mask := self._create_transcripts_mask(transcripts[batch_idx][beam_idx])
                        ]
                        .cpu()
                        .detach()
                        .numpy(),
                        timestamp=timestamps[batch_idx][beam_idx][mask].cpu().detach().numpy(),
                        alignments=None,
                        dec_state=None,
                    )
                    for beam_idx in range(self.beam_size)
                    if scores[batch_idx][beam_idx] > INACTIVE_SCORE
                ]
            )
            for batch_idx in range(self.batch_size)
        ]
        return hypotheses

    def flatten_sort_(self, score_norm: bool = True):
        """
        Sorts and flattens the tree structure of hypotheses in a batched beam search decoding process.
        Args:
            score_norm (bool, optional): If True, normalizes the scores by dividing
                them by the current lengths of the hypotheses plus one. Defaults to True.
        This method performs the following steps:
        1. Normalizes the scores if `score_norm` is True.
        2. Sorts the normalized scores in descending order and retrieves the corresponding indices.
        3. Iteratively reconstructs the tokens and timestamps for each hypothesis in reverse order.
        4. Updates the internal state of the object, including transcripts, timestamps, scores,
           lengths, labels, and other metadata, based on the sorted order.
        """

        # add one for consistency with non-batched decodings, that use SOS.
        normalized_scores = (
            self.scores / (self.current_lengths_nb.to(self.scores.dtype) + 1) if score_norm else self.scores
        )
        _, indices = torch.sort(normalized_scores, dim=-1, descending=True)
        self._flatten_with_permutation_(indices)

    def flatten_(self) -> torch.Tensor:
        """
        Flatten the tree structure of hypotheses without changing beam order.

        Like :meth:`flatten_sort_` but uses the identity permutation, so beam ``i`` keeps
        its identity (its decoded prefix and its cross-chunk per-beam state stay aligned
        with the corresponding beam in any other ``BatchedBeamHyps`` constructed under the
        same decoding run). Required for inter-chunk :meth:`merge_` calls in streaming
        beam decoding where beam indices must correspond across chunks.

        Returns:
            ``root_ptrs`` of shape ``[batch_size, beam_size]``: the beam index at the
            chunk's *start* (i.e. before the first ``add_results_*`` write) from which
            each output beam ultimately descends. For chunked streaming beam search, this
            tells the caller how to permute the previous chunks' accumulated per-beam
            transcripts so they align with this chunk's beam ordering before merging.

            If the prefix tree is empty (``current_lengths_wb.max() == 0``) the identity
            permutation is returned.
        """
        identity = self.beam_indices.unsqueeze(0).expand(self.batch_size, self.beam_size).contiguous()
        return self._flatten_with_permutation_(identity)

    def _flatten_with_permutation_(self, indices: torch.Tensor) -> torch.Tensor:
        """
        In-place flatten of the prefix tree using ``indices`` as the new beam permutation.

        Walks ``transcript_wb_prev_ptr`` from the most recent step back to step 0,
        gathering tokens and timestamps for each output beam from the source beam given
        by ``indices``. Updates all per-beam metadata to match the new ordering.

        Args:
            indices: ``[batch_size, beam_size]`` long tensor giving the source beam index
                for each output beam (e.g. ``arange(beam_size)`` for no permutation).

        Returns:
            ``root_ptrs`` of shape ``[batch_size, beam_size]``: the beam index *before*
            step 0 of the prefix tree from which each output beam descends. If the prefix
            tree is empty (``max_idx < 0``) this equals ``indices``.
        """
        max_idx = self.current_lengths_wb.max() - 1
        ptrs = indices

        for idx in range(max_idx, -1, -1):
            self.transcript_wb[..., idx].copy_(self.transcript_wb[self.batch_indices.unsqueeze(-1), ptrs, idx])
            if self.model_type == ASRModelTypeEnum.TDT or self.model_type == ASRModelTypeEnum.RNNT:
                self.timestamps[..., idx].copy_(self.timestamps[self.batch_indices.unsqueeze(-1), ptrs, idx])
            ptrs = self.transcript_wb_prev_ptr[self.batch_indices.unsqueeze(-1), ptrs, idx]
        self.transcript_wb_prev_ptr[..., : max_idx + 1].copy_(self.beam_indices.unsqueeze(0).unsqueeze(-1))

        self.scores.copy_(torch.gather(self.scores, dim=-1, index=indices))
        self.current_lengths_nb.copy_(torch.gather(self.current_lengths_nb, dim=-1, index=indices))
        self.current_lengths_wb.copy_(torch.gather(self.current_lengths_wb, dim=-1, index=indices))

        self.last_label.copy_(torch.gather(self.last_label, dim=-1, index=indices))

        if self.model_type == ASRModelTypeEnum.TDT or self.model_type == ASRModelTypeEnum.RNNT:
            self.next_timestamp.copy_(torch.gather(self.next_timestamp, dim=-1, index=indices))
            self.last_timestamp_lasts.copy_(torch.gather(self.last_timestamp_lasts, dim=-1, index=indices))

        self.transcript_hash.copy_(torch.gather(self.transcript_hash, dim=-1, index=indices))
        if self.store_prefix_hashes:
            self.transcript_prefix_hash.copy_(torch.gather(self.transcript_prefix_hash, dim=-1, index=indices))

        return ptrs

    def _create_fold_consecutive_mask(self, transcript):
        """
        Creates a mask to filter consecutive duplicates, blanks, and invalid tokens in a transcript.
        Args:
            transcript (torch.Tensor): 1D tensor of token sequence.
        Returns:
            torch.Tensor: Boolean mask indicating valid tokens.
        """
        device = transcript.device
        mask = (
            (transcript >= 0)
            & torch.cat([torch.tensor([True], device=device), transcript[1:] != transcript[:-1]])
            & (transcript != self.blank_index)
        )

        return mask

    def _create_timestamps_tensor(self, max_time):
        """
        Generates a tensor of timestamps.

        In CTC, labels align with input frames, allowing timestamps to be precomputed.

        Args:
            max_time (int): The maximum number of time steps (frames) to include in the tensor.

        Returns:
            torch.Tensor: A tensor of shape (batch_size, beam_size, max_time) containing
            sequential timestamps for each batch and beam.
        """
        return torch.arange(max_time, device=self.device, dtype=torch.long)[None, None, :].repeat(
            self.batch_size, self.beam_size, 1
        )

    def _create_transcripts_mask(self, transcripts: torch.Tensor):
        """
        Processes the transcripts.
        For RNN-T and TDT removes blanks.
        For CTC removes remove consecutive duplicates and blanks.
        Args:
            transcripts (torch.Tensor): 1D tensor of token sequence.
        Returns:
            torch.Tensor: Binary mask indicating valid tokens.
        """
        if self.model_type == ASRModelTypeEnum.CTC:
            return self._create_fold_consecutive_mask(transcripts)
        else:
            return (transcripts >= 0) & (transcripts != self.blank_index)

    def merge_(
        self,
        other: "BatchedBeamHyps",
        is_chunk_continuation: bool = False,
        boundary_prev_ptr: Optional[torch.Tensor] = None,
    ) -> "BatchedBeamHyps":
        """
        Merge two batched beam hypotheses structures by concatenating transcripts.
        Used for streaming/chunked inference where results from multiple chunks need to be combined.

        Prerequisites:
            - Both self and other should have been processed with flatten_sort_() before merging,
              so that each beam contains an independent flattened hypothesis.
            - Beam indices should correspond across chunks (beam i in self matches beam i in other).

        Notes:
            - Timestamps in 'other' should already be cumulative (adjusted for time offset).
            - The transcript_hash values are copied from 'other' and won't reflect the full
              merged transcript. This means recombine_hyps_() should NOT be called on merged
              results without recomputing hashes. This is acceptable for output-only use.

        Args:
            other: BatchedBeamHyps from the next chunk to merge.
            is_chunk_continuation: If True, treat ``other`` as a beam-search continuation
                chunk in which the cross-chunk per-beam fields (``scores``,
                ``current_lengths_nb``) already hold cumulative across-chunks values rather
                than chunk-local deltas. In that case those fields are *replaced* with the
                values from ``other`` instead of summed, to avoid double-counting. The
                default (False) preserves the original "deltas" semantics used by greedy
                streaming-style merges.
            boundary_prev_ptr: Optional ``[batch_size, beam_size]`` long tensor. When
                provided, written into ``transcript_wb_prev_ptr`` at the very first
                position of the merged region (i.e. at ``self.current_lengths_wb`` before
                the update). All other positions of the merged region still receive
                ``beam_indices`` (identity) pointers. This is how chunked streaming beam
                search threads the cross-chunk beam permutation (the "root ptrs" returned
                by :meth:`flatten_` on ``other``) into the accumulator's prefix tree so
                that the final :meth:`flatten_sort_` walk redirects from beam ``i`` in
                ``other``'s region back to its source beam in ``self``'s region.

        Returns:
            Self (modified in-place)
        """
        max_other_len = other.current_lengths_wb.max().item()

        # Early return if other has nothing to merge
        if max_other_len == 0:
            return self

        # Check if we need more storage (using allocated buffer size, not current shape)
        # Compute max needed length: current max + other max
        max_needed = self.current_lengths_wb.max().item() + max_other_len

        # Expand storage if needed - use existing _allocate_more() method
        while max_needed > self._max_length:
            self._allocate_more()

        # Create a range tensor: [0, 1, 2, ..., max_other_len-1]
        other_indices = torch.arange(max_other_len, device=self.device, dtype=torch.long)

        # Create shifted indices: current_lengths + [0, 1, 2, ...]
        # Shape: [batch_size, beam_size, max_other_len]
        shifted_indices = self.current_lengths_wb.unsqueeze(-1) + other_indices.unsqueeze(0).unsqueeze(0)

        # Scatter other's transcripts into self at shifted positions
        self.transcript_wb.scatter_(
            dim=-1,
            index=shifted_indices,
            src=other.transcript_wb[..., :max_other_len],
        )

        # Update pointers: in the merged region every position points to its own beam
        # (identity), except the *first* merged position which optionally encodes the
        # cross-chunk root permutation so the final flatten walk redirects from the new
        # region back to the right beam in the old region.
        identity_src = self.beam_indices.view(1, self.beam_size, 1).expand(self.batch_size, -1, max_other_len)
        if boundary_prev_ptr is not None:
            ptr_src = identity_src.clone()
            ptr_src[..., 0] = boundary_prev_ptr
        else:
            ptr_src = identity_src
        self.transcript_wb_prev_ptr.scatter_(
            dim=-1,
            index=shifted_indices,
            src=ptr_src,
        )

        # Scatter timestamps
        self.timestamps.scatter_(
            dim=-1,
            index=shifted_indices,
            src=other.timestamps[..., :max_other_len],
        )

        # Lengths in the chunk-local write cursor are always additive (``other`` always
        # reports a chunk-local ``current_lengths_wb``).
        self.current_lengths_wb += other.current_lengths_wb

        if is_chunk_continuation:
            # Beam-search streaming: ``other`` carries cumulative cross-chunk state in
            # these fields, so replace rather than accumulate.
            self.current_lengths_nb.copy_(other.current_lengths_nb)
            self.scores.copy_(other.scores)
        else:
            # Original ("deltas") semantics.
            self.current_lengths_nb += other.current_lengths_nb
            self.scores += other.scores

        # Update transcript hash by combining hashes
        # The hash of the merged transcript should account for all non-blank labels
        self.transcript_hash.copy_(other.transcript_hash)

        # Update prefix hashes if used
        if self.store_prefix_hashes:
            self.transcript_prefix_hash.copy_(other.transcript_prefix_hash)

        # Update tracking fields from other (they reflect the end state after other chunk)
        self.last_label.copy_(other.last_label)

        # Only update timestamp tracking fields for transducer models
        if self.model_type != ASRModelTypeEnum.CTC:
            self.next_timestamp.copy_(other.next_timestamp)
            self.last_timestamp_lasts.copy_(other.last_timestamp_lasts)

        return self
