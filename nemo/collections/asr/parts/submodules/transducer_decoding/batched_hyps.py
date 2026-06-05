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

import torch


class BatchedHyps:
    """Class to store batched hypotheses (labels, time_indices, scores) for efficient RNNT decoding"""

    def __init__(
        self,
        batch_size: int,
        init_length: int,
        blank_id: int,
        logits_dim: int | None = None,
        device: torch.device | None = None,
        float_dtype: torch.dtype | None = None,
        with_durations: bool = False,
        with_step_confidence: bool = False,
        with_duration_confidence: bool = False,
        with_logits: bool = False,
        with_blank_steps: bool = False,
    ):
        """

        Args:
            batch_size: batch size for hypotheses
            init_length: initial estimate for the length of hypotheses (if the real length is higher,
                tensors will be reallocated)
            device: device for storing hypotheses
            float_dtype: float type for scores
        """
        if init_length <= 0:
            raise ValueError(f"init_length must be > 0, got {init_length}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")
        self._max_length = init_length
        self.batch_size = batch_size
        self.device = device
        self.float_dtype = float_dtype
        self.with_durations = with_durations
        self.with_step_confidence = with_step_confidence
        self.with_duration_confidence = with_duration_confidence
        self.with_logits = with_logits
        self.with_blank_steps = with_blank_steps
        self.blank_id = blank_id

        # batch of current lengths of hypotheses and corresponding timestamps
        self.current_lengths = torch.zeros(batch_size, device=device, dtype=torch.long)
        # tensor for storing transcripts
        self.transcript = torch.zeros((batch_size, self._max_length), device=device, dtype=torch.long)
        # tensor for storing timestamps corresponding to transcripts
        self.timestamps = torch.zeros((batch_size, self._max_length), device=device, dtype=torch.long)

        # optional storage: 1) durations, 2) confidence, 3) logits
        # optional (1) durations corresponding to tokens
        if with_durations:
            self.token_durations = torch.zeros((batch_size, self._max_length), device=device, dtype=torch.long)
        else:
            self.token_durations = None
        # optional (2) step confidence
        if self.with_step_confidence:
            confidence_shape = (
                [batch_size, self._max_length, 2] if self.with_duration_confidence else [batch_size, self._max_length]
            )
            self.step_confidence = torch.zeros(confidence_shape, device=device, dtype=float_dtype)
        else:
            self.step_confidence = None
        # optional (3) logits
        if self.with_logits:
            if logits_dim is None:
                raise ValueError("`logits_dim` is None, incompatible with `with_logits=True`")
            self.logits = torch.zeros((batch_size, self._max_length, logits_dim), device=device, dtype=float_dtype)
        else:
            self.logits = None
        self.logits_dim = logits_dim

        # accumulated scores for hypotheses
        self.scores = torch.zeros(batch_size, device=device, dtype=float_dtype)

        # tracking last timestamp of each hyp to avoid infinite looping (when max symbols per frame is restricted)
        # last observed non-blank timestamp (with label) for each hypothesis
        self.last_nb_timestamp = torch.full((batch_size,), -1, device=device, dtype=torch.long)
        # number of non-blank labels for the last timestamp
        self.last_nb_timestamp_lasts = torch.zeros(batch_size, device=device, dtype=torch.long)
        self.last_nb_labels = torch.full((batch_size,), fill_value=-1, device=device, dtype=torch.long)
        self._batch_indices = torch.arange(batch_size, device=device)
        self._ones_batch = torch.ones_like(self._batch_indices)

    def clear_(self):
        """
        Clears batched hypotheses state.
        """
        self.current_lengths.fill_(0)
        self.transcript.fill_(0)
        self.timestamps.fill_(0)
        self.scores.fill_(0.0)
        self.last_nb_timestamp.fill_(-1)
        self.last_nb_timestamp_lasts.fill_(0)
        self.last_nb_labels.fill_(-1)

        # optional (1) durations corresponding to tokens
        if self.with_durations:
            self.token_durations.fill_(0)

        # optional (2) step confidence
        if self.with_step_confidence:
            self.step_confidence.fill_(0.0)

        # optional (3) logits
        if self.with_logits:
            self.logits.fill_(0.0)

    def _allocate_more(self):
        """
        Allocate 2x space for tensors, similar to common C++ std::vector implementations
        to maintain O(1) insertion time complexity
        """
        self.transcript = torch.cat((self.transcript, torch.zeros_like(self.transcript)), dim=-1)
        self.timestamps = torch.cat((self.timestamps, torch.zeros_like(self.timestamps)), dim=-1)

        # optional (1) durations corresponding to tokens
        if self.with_durations:
            self.token_durations = torch.cat((self.token_durations, torch.zeros_like(self.token_durations)), dim=-1)

        # optional (2) step confidence
        if self.with_step_confidence:
            self.step_confidence = torch.cat((self.step_confidence, torch.zeros_like(self.step_confidence)), dim=1)

        # optional (3) logits
        if self.with_logits:
            self.logits = torch.cat((self.logits, torch.zeros_like(self.logits)), dim=1)
        self._max_length *= 2

    def _to_confidence_shape(self, x: torch.Tensor) -> torch.Tensor:
        """Convert 2d tensor x to confidence-compatible shape"""
        if not self.with_duration_confidence:
            # shape compatible
            return x
        return x[..., None].expand(-1, -1, 2)

    def add_results_masked_(
        self,
        active_mask: torch.Tensor,
        labels: torch.Tensor,
        time_indices: torch.Tensor,
        scores: torch.Tensor,
        token_durations: torch.Tensor | None = None,
        confidence: torch.Tensor | None = None,
        logits: torch.Tensor | None = None,
        check_lengths: bool = True,
    ):
        """
        Add results (inplace) from a decoding step to the batched hypotheses.
        We assume that all tensors have the same first dimension, and labels are non-blanks.
        Useful if all the memory is pre-allocated, especially with cuda graphs
        (otherwise prefer a more safe `add_results_`)
        Args:
            active_mask: tensor with mask for active hypotheses (of batch_size)
            labels: non-blank labels to add
            time_indices: tensor of time index for each label
            scores: label scores
            token_durations: token durations for TDT
            confidence: optional tensor with step confidence
            logits: optional logits
            check_lengths: check storage length before adding results; blocking operation
        """
        if check_lengths:
            if (self.current_lengths + active_mask).max() >= self._max_length:
                self._allocate_more()
        # store transcript and timestamps
        self.transcript[self._batch_indices, self.current_lengths] = labels
        self.timestamps[self._batch_indices, self.current_lengths] = time_indices
        if self.with_durations and token_durations is not None:
            self.token_durations[self._batch_indices, self.current_lengths] = token_durations
        if self.with_step_confidence and confidence is not None:
            self.step_confidence[self._batch_indices, self.current_lengths] = confidence
        if self.with_logits and logits is not None:
            self.logits[self._batch_indices, self.current_lengths] = logits
        # store last observed timestamp + number of observation for the current timestamp
        # if last_timestamp == time_indices, increase; else set to 1
        if self.with_blank_steps:
            active_nb_mask = active_mask & (labels != self.blank_id)
        else:
            active_nb_mask = active_mask
        torch.where(
            torch.logical_and(active_nb_mask, self.last_nb_timestamp == time_indices),
            self.last_nb_timestamp_lasts + 1,
            self.last_nb_timestamp_lasts,
            out=self.last_nb_timestamp_lasts,
        )
        torch.where(
            torch.logical_and(active_nb_mask, self.last_nb_timestamp != time_indices),
            self._ones_batch,
            self.last_nb_timestamp_lasts,
            out=self.last_nb_timestamp_lasts,
        )
        # same as: self.last_nb_timestamp[active_mask] = time_indices[active_mask], but non-blocking
        torch.where(active_nb_mask, time_indices, self.last_nb_timestamp, out=self.last_nb_timestamp)
        torch.where(active_nb_mask, labels, self.last_nb_labels, out=self.last_nb_labels)
        # accumulate scores
        # same as self.scores[active_mask] += scores[active_mask], but non-blocking
        torch.where(active_nb_mask, self.scores + scores, self.scores, out=self.scores)
        # increase lengths
        self.current_lengths += active_mask

    def get_last_labels(self, pad_id: int = -1):
        """Get last labels. For elements without labels use pad_id"""
        return torch.where(self.last_nb_labels != -1, self.last_nb_labels, pad_id)

    def clone(self) -> "BatchedHyps":
        """Return a copy of self"""
        batched_hyps = BatchedHyps(
            batch_size=self.batch_size,
            init_length=self._max_length,
            blank_id=self.blank_id,
            logits_dim=self.logits_dim,
            device=self.device,
            float_dtype=self.float_dtype,
            with_durations=self.with_durations,
            with_step_confidence=self.with_step_confidence,
            with_duration_confidence=self.with_duration_confidence,
            with_logits=self.with_logits,
            with_blank_steps=self.with_blank_steps,
        )
        batched_hyps.current_lengths.copy_(self.current_lengths)
        batched_hyps.transcript.copy_(self.transcript)
        batched_hyps.timestamps.copy_(self.timestamps)
        # optional (1) durations corresponding to tokens
        if self.with_durations:
            batched_hyps.token_durations.copy_(self.token_durations)
        # optional (2) step confidence
        if self.with_step_confidence:
            batched_hyps.step_confidence.copy_(self.step_confidence)
        # optional (3) logits
        if self.with_logits:
            batched_hyps.logits.copy_(self.logits)
        batched_hyps.scores.copy_(self.scores)
        batched_hyps.last_nb_timestamp.copy_(self.last_nb_timestamp)
        batched_hyps.last_nb_timestamp_lasts.copy_(self.last_nb_timestamp_lasts)
        batched_hyps.last_nb_labels.copy_(self.last_nb_labels)
        return batched_hyps

    def merge_(self, other: "BatchedHyps") -> "BatchedHyps":
        """
        Merge two batched hypotheses structures.
        NB: this will reallocate memory

        Args:
            other: BatchedHyps
        """
        cur_len = self.current_lengths.max().item()
        other_len = other.current_lengths.max().item()
        if cur_len + other_len >= self._max_length:
            add_len = cur_len + other_len - self._max_length + 1
            device = self.transcript.device
            add_shape = [self.batch_size, add_len]
            self.transcript = torch.cat(
                (self.transcript, torch.zeros(add_shape, dtype=torch.long, device=device)), dim=-1
            )
            self.timestamps = torch.cat(
                (self.timestamps, torch.zeros(add_shape, dtype=torch.long, device=device)), dim=-1
            )
            # optional (1) durations corresponding to tokens
            if self.with_durations:
                self.token_durations = torch.cat(
                    (self.token_durations, torch.zeros(add_shape, dtype=torch.long, device=device)), dim=-1
                )
            # optional (2) step confidence
            if self.with_step_confidence:
                self.step_confidence = torch.cat(
                    (
                        self.step_confidence,
                        torch.zeros(
                            add_shape + list(self.step_confidence.shape)[2:], dtype=self.float_dtype, device=device
                        ),
                    ),
                    dim=1,
                )
            # optional (3) logits
            if self.with_logits:
                self.logits = torch.cat(
                    (self.logits, torch.zeros(add_shape + [self.logits_dim], dtype=self.float_dtype, device=device)),
                    dim=1,
                )
            self._max_length += add_len

        indices = torch.arange(other_len, device=self.current_lengths.device)
        shifted_indices = self.current_lengths[:, None] + indices[None, :]
        self.transcript.scatter_(dim=1, index=shifted_indices, src=other.transcript)
        self.timestamps.scatter_(dim=1, index=shifted_indices, src=other.timestamps)
        # optional (1) durations corresponding to tokens
        if self.with_durations:
            self.token_durations.scatter_(dim=1, index=shifted_indices, src=other.token_durations)
        # optional (2) step confidence
        if self.with_step_confidence:
            self.step_confidence.scatter_(
                dim=1, index=self._to_confidence_shape(shifted_indices), src=other.step_confidence
            )
        # optional (3) logits
        if self.with_logits:
            self.logits.scatter_(
                dim=1, index=shifted_indices[..., None].expand([-1, -1, self.logits_dim]), src=other.logits
            )
        self.current_lengths += other.current_lengths
        self.scores += other.scores
        other_has_last_nb = other.last_nb_labels != -1
        torch.where(other_has_last_nb, other.last_nb_timestamp, self.last_nb_timestamp, out=self.last_nb_timestamp)
        torch.where(
            other_has_last_nb,
            other.last_nb_timestamp_lasts,
            self.last_nb_timestamp_lasts,
            out=self.last_nb_timestamp_lasts,
        )
        torch.where(other_has_last_nb, other.last_nb_labels, self.last_nb_labels, out=self.last_nb_labels)
        return self

    def get_data_without_blank(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Return lengths, transcript, timestamps, (optional) step_confidence without blank labels"""
        if not self.with_blank_steps:
            return self.current_lengths, self.transcript, self.timestamps, self.token_durations, self.step_confidence
        mask_nb_int = torch.logical_and(
            torch.arange(self.transcript.shape[1], device=self.device)[None, :] < self.current_lengths[:, None],
            self.transcript != self.blank_id,
        ).to(torch.int)
        lengths_nb = mask_nb_int.sum(dim=-1)
        indices_nb_first = torch.argsort(mask_nb_int, dim=1, descending=True, stable=True)
        transcript_nb = self.transcript.gather(dim=1, index=indices_nb_first)
        timestamps_nb = self.timestamps.gather(dim=1, index=indices_nb_first)
        if self.with_durations:
            token_durations_nb = self.token_durations.gather(dim=1, index=indices_nb_first)
        else:
            token_durations_nb = None
        if self.with_step_confidence:
            step_confidence_nb = self.step_confidence.gather(dim=1, index=self._to_confidence_shape(indices_nb_first))
        else:
            step_confidence_nb = None
        return lengths_nb, transcript_nb, timestamps_nb, token_durations_nb, step_confidence_nb
