# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from typing import Any, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from omegaconf import DictConfig, ListConfig

from nemo.collections.asr.parts.utils import rnnt_utils
from nemo.collections.asr.parts.utils.asr_confidence_utils import ConfidenceMethodMixin


class GreedyBatchedTDTLoopLabelsComputer(ConfidenceMethodMixin):
    """
    Loop Labels algorithm implementation. Callable.
    """

    def __init__(
        self,
        decoder,
        joint,
        blank_index: int,
        durations: Union[list[int], ListConfig[int]],
        max_symbols_per_step: Optional[int] = None,
        preserve_alignments=False,
        preserve_frame_confidence=False,
        confidence_method_cfg: Optional[DictConfig] = None,
    ):
        """
        Init method.
        Args:
            decoder: Prediction network from RNN-T
            joint: Joint module from RNN-T
            blank_index: index of blank symbol
            durations: list of TDT durations, e.g., [0, 1, 2, 4, 8]
            max_symbols_per_step: max symbols to emit on each step (to avoid infinite looping)
            preserve_alignments: if alignments are needed
            preserve_frame_confidence: if frame confidence is needed
            confidence_method_cfg: config for the confidence
        """
        super().__init__()
        self.decoder = decoder
        self.joint = joint
        # keep durations on CPU to avoid side effects in multi-gpu environment
        self.durations = torch.tensor(list(durations), device="cpu").to(torch.long)
        self._blank_index = blank_index
        self.max_symbols = max_symbols_per_step
        self.preserve_alignments = preserve_alignments
        self.preserve_frame_confidence = preserve_frame_confidence
        self._SOS = self._blank_index
        self._init_confidence_method(confidence_method_cfg=confidence_method_cfg)
        assert self._SOS == self._blank_index  # "blank as pad" algorithm only

    def __call__(
        self, x: torch.Tensor, out_len: torch.Tensor,
    ) -> Tuple[rnnt_utils.BatchedHyps, Optional[rnnt_utils.BatchedAlignments], Any]:
        """
        Optimized batched greedy decoding.
        Iterates over labels, on each step finding the next non-blank label
        (evaluating Joint multiple times in inner loop); It uses a minimal possible amount of calls
        to prediction network (with maximum possible batch size),
        which makes it especially useful for scaling the prediction network.
        During decoding all active hypotheses ("texts") have the same lengths.

        Args:
            x: output from the encoder
            out_len: lengths of the utterances in `x`
        """
        batch_size, max_time, _unused = x.shape
        device = x.device

        x = self.joint.project_encoder(x)  # do not recalculate joint projection, project only once

        # init output structures: BatchedHyps (for results), BatchedAlignments + last decoder state
        # init empty batched hypotheses
        batched_hyps = rnnt_utils.BatchedHyps(
            batch_size=batch_size,
            init_length=max_time * self.max_symbols if self.max_symbols is not None else max_time,
            device=x.device,
            float_dtype=x.dtype,
        )
        # sample state, will be replaced further when the decoding for hypothesis is done
        last_decoder_state = self.decoder.initialize_state(x)
        # init alignments if necessary
        use_alignments = self.preserve_alignments or self.preserve_frame_confidence
        # always use alignments variable - for torch.jit adaptation, but keep it as minimal as possible
        alignments = rnnt_utils.BatchedAlignments(
            batch_size=batch_size,
            logits_dim=self.joint.num_classes_with_blank,
            init_length=max_time * 2 if use_alignments else 1,  # blank for each timestep + text tokens
            device=x.device,
            float_dtype=x.dtype,
            store_alignments=self.preserve_alignments,
            store_frame_confidence=self.preserve_frame_confidence,
        )

        # durations
        all_durations = self.durations.to(device, non_blocking=True)
        num_durations = all_durations.shape[0]

        # initial state, needed for torch.jit to compile (cannot handle None)
        state = self.decoder.initialize_state(x)
        # indices of elements in batch (constant)
        batch_indices = torch.arange(batch_size, dtype=torch.long, device=device)
        # last found labels - initially <SOS> (<blank>) symbol
        labels = torch.full_like(batch_indices, fill_value=self._SOS)

        # time indices
        time_indices = torch.zeros_like(batch_indices)
        safe_time_indices = torch.zeros_like(time_indices)  # time indices, guaranteed to be < out_len
        time_indices_current_labels = torch.zeros_like(time_indices)
        last_timesteps = out_len - 1

        # masks for utterances in batch
        active_mask: torch.Tensor = out_len > 0
        advance_mask = torch.empty_like(active_mask)

        # for storing the last state we need to know what elements became "inactive" on this step
        active_mask_prev = torch.empty_like(active_mask)
        became_inactive_mask = torch.empty_like(active_mask)

        # loop while there are active utterances
        first_step = True
        while active_mask.any():
            active_mask_prev.copy_(active_mask, non_blocking=True)
            # stage 1: get decoder (prediction network) output
            if first_step:
                # start of the loop, SOS symbol is passed into prediction network, state is None
                # we need to separate this for torch.jit
                decoder_output, state, *_ = self.decoder.predict(
                    labels.unsqueeze(1), None, add_sos=False, batch_size=batch_size
                )
                first_step = False
            else:
                decoder_output, state, *_ = self.decoder.predict(
                    labels.unsqueeze(1), state, add_sos=False, batch_size=batch_size
                )
            decoder_output = self.joint.project_prednet(decoder_output)  # do not recalculate joint projection

            # stage 2: get joint output, iteratively seeking for non-blank labels
            # blank label in `labels` tensor means "end of hypothesis" (for this index)
            logits = (
                self.joint.joint_after_projection(x[batch_indices, safe_time_indices].unsqueeze(1), decoder_output,)
                .squeeze(1)
                .squeeze(1)
            )
            scores, labels = logits[:, :-num_durations].max(dim=-1)
            jump_durations_indices = logits[:, -num_durations:].argmax(dim=-1)
            durations = all_durations[jump_durations_indices]

            # search for non-blank labels using joint, advancing time indices for blank labels
            # checking max_symbols is not needed, since we already forced advancing time indices for such cases
            blank_mask = labels == self._blank_index
            # for blank labels force duration >= 1
            durations.masked_fill_(torch.logical_and(durations == 0, blank_mask), 1)
            time_indices_current_labels.copy_(time_indices, non_blocking=True)
            if use_alignments:
                alignments.add_results_masked_(
                    active_mask=active_mask,
                    time_indices=time_indices_current_labels,
                    logits=logits if self.preserve_alignments else None,
                    labels=labels if self.preserve_alignments else None,
                    confidence=self._get_confidence_tensor(F.log_softmax(logits[:, :-num_durations], dim=-1))
                    if self.preserve_frame_confidence
                    else None,
                )

            # advance_mask is a mask for current batch for searching non-blank labels;
            # each element is True if non-blank symbol is not yet found AND we can increase the time index
            time_indices += durations
            torch.minimum(time_indices, last_timesteps, out=safe_time_indices)
            torch.less(time_indices, out_len, out=active_mask)
            torch.logical_and(active_mask, blank_mask, out=advance_mask)
            # inner loop: find next non-blank labels (if exist)
            while advance_mask.any():
                # same as: time_indices_current_labels[advance_mask] = time_indices[advance_mask], but non-blocking
                # store current time indices to use further for storing the results
                torch.where(advance_mask, time_indices, time_indices_current_labels, out=time_indices_current_labels)
                logits = (
                    self.joint.joint_after_projection(
                        x[batch_indices, safe_time_indices].unsqueeze(1), decoder_output,
                    )
                    .squeeze(1)
                    .squeeze(1)
                )
                # get labels (greedy) and scores from current logits, replace labels/scores with new
                # labels[advance_mask] are blank, and we are looking for non-blank labels
                more_scores, more_labels = logits[:, :-num_durations].max(dim=-1)
                # same as: labels[advance_mask] = more_labels[advance_mask], but non-blocking
                torch.where(advance_mask, more_labels, labels, out=labels)
                # same as: scores[advance_mask] = more_scores[advance_mask], but non-blocking
                torch.where(advance_mask, more_scores, scores, out=scores)
                jump_durations_indices = logits[:, -num_durations:].argmax(dim=-1)
                durations = all_durations[jump_durations_indices]

                if use_alignments:
                    alignments.add_results_masked_(
                        active_mask=advance_mask,
                        time_indices=time_indices_current_labels,
                        logits=logits if self.preserve_alignments else None,
                        labels=more_labels if self.preserve_alignments else None,
                        confidence=self._get_confidence_tensor(F.log_softmax(logits[:, :-num_durations], dim=-1))
                        if self.preserve_frame_confidence
                        else None,
                    )

                blank_mask = labels == self._blank_index
                # for blank labels force duration >= 1
                durations.masked_fill_(torch.logical_and(durations == 0, blank_mask), 1)
                # same as time_indices[advance_mask] += durations[advance_mask], but non-blocking
                torch.where(advance_mask, time_indices + durations, time_indices, out=time_indices)
                torch.minimum(time_indices, last_timesteps, out=safe_time_indices)
                torch.less(time_indices, out_len, out=active_mask)
                torch.logical_and(active_mask, blank_mask, out=advance_mask)

            # stage 3: filter labels and state, store hypotheses
            # select states for hyps that became inactive (is it necessary?)
            # this seems to be redundant, but used in the `loop_frames` output
            torch.ne(active_mask, active_mask_prev, out=became_inactive_mask)
            self.decoder.batch_replace_states_mask(
                src_states=state, dst_states=last_decoder_state, mask=became_inactive_mask,
            )

            # store hypotheses
            if self.max_symbols is not None:
                # pre-allocated memory, no need for checks
                batched_hyps.add_results_masked_no_checks_(
                    active_mask, labels, time_indices_current_labels, scores,
                )
            else:
                # auto-adjusted storage
                batched_hyps.add_results_masked_(
                    active_mask, labels, time_indices_current_labels, scores,
                )

            # stage 4: to avoid looping, go to next frame after max_symbols emission
            if self.max_symbols is not None:
                # if labels are non-blank (not end-of-utterance), check that last observed timestep with label:
                # if it is equal to the current time index, and number of observations is >= max_symbols, force blank
                force_blank_mask = torch.logical_and(
                    active_mask,
                    torch.logical_and(
                        torch.logical_and(
                            labels != self._blank_index, batched_hyps.last_timestep_lasts >= self.max_symbols,
                        ),
                        batched_hyps.last_timestep == time_indices,
                    ),
                )
                time_indices += force_blank_mask  # emit blank => advance time indices
                # update safe_time_indices, non-blocking
                torch.minimum(time_indices, last_timesteps, out=safe_time_indices)
                # same as: active_mask = time_indices < out_len
                torch.less(time_indices, out_len, out=active_mask)
        if use_alignments:
            return batched_hyps, alignments, last_decoder_state
        return batched_hyps, None, last_decoder_state
