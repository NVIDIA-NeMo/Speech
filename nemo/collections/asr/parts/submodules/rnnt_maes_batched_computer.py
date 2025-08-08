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
from pathlib import Path
from typing import Optional

import torch

from nemo.collections.asr.parts.utils.asr_confidence_utils import ConfidenceMethodMixin
from nemo.collections.asr.parts.utils.batched_beam_decoding_utils import (
    INACTIVE_SCORE,
    NON_EXISTENT_LABEL_VALUE,
    BatchedBeamHyps,
    BlankLMScoreMode,
    PruningMode,
)
from nemo.utils import logging


class ModifiedAESBatchedRNNTComputer(ConfidenceMethodMixin):
    """
    Batched Adaptive Expansion search implementation. Callable.
    Based on https://ieeexplore.ieee.org/document/9250505 with the following modficiations:
        - does not support prediction network caching
        - supports prefix search with only longest prefix
    """

    def __init__(
        self,
        decoder,
        joint,
        blank_index: int,
        beam_size: int,
        maes_num_steps: int,
        maes_expansion_beta: int,
        maes_expansion_gamma: int,
        preserve_alignments=False,
        ngram_lm_model: Optional[str | Path] = None,
        ngram_lm_alpha: float = 0.0,
        blank_lm_score_mode: Optional[str | BlankLMScoreMode] = BlankLMScoreMode.NO_SCORE,
        pruning_mode: Optional[str | PruningMode] = PruningMode.EARLY,
        allow_cuda_graphs: Optional[bool] = True,
    ):
        """
        Init method.
        Args:
            decoder: Prediction network from RNN-T
            joint: Joint module from RNN-T
            blank_index: index of blank symbol
            maes_num_steps:  Number of adaptive steps to take. From the paper, 2 steps is generally sufficient. int > 1.
            maes_expansion_beta: Maximum number of prefix expansions allowed, in addition to the beam size.
                Effectively, the number of hypothesis = beam_size + maes_expansion_beta. Must be an int >= 0,
                and affects the speed of inference since large values will perform large beam search in the next step.
            maes_expansion_gamma: Float pruning threshold used in the prune-by-value step when computing the expansions.
                The default (2.3) is selected from the paper. It performs a comparison
                (max_log_prob - gamma <= log_prob[v]) where v is all vocabulary indices in the Vocab set and max_log_prob
                is the "most" likely token to be predicted. Gamma therefore provides a margin of additional tokens which
                can be potential candidates for expansion apart from the "most likely" candidate.
                Lower values will reduce the number of expansions (by increasing pruning-by-value, thereby improving speed
                but hurting accuracy). Higher values will increase the number of expansions (by reducing pruning-by-value,
                thereby reducing speed but potentially improving accuracy). This is a hyper parameter to be experimentally
                tuned on a validation set.
            preserve_alignments: if alignments are needed
            ngram_lm_model: path to the NGPU-LM n-gram LM model: .arpa or .nemo formats
            ngram_lm_alpha: weight for the n-gram LM scores
            blank_lm_score_mode: mode for scoring blank symbol with LM
            pruning_mode: mode for pruning hypotheses with LM
            allow_cuda_graphs: whether to allow CUDA graphs
        """

        super().__init__()
        self.decoder = decoder
        self.joint = joint
        self._blank_index = blank_index

        self.beam_size = beam_size
        self.maes_num_steps = maes_num_steps
        self.maes_expansion_beta = maes_expansion_beta
        self.maes_expansion_gamma = maes_expansion_gamma
        self.preserve_alignments = preserve_alignments
        self._SOS = self._blank_index
        self.pruning_mode = pruning_mode
        self.blank_lm_score_mode = blank_lm_score_mode

        self.maes_num_expansions = self.beam_size + self.maes_expansion_beta

        if self.preserve_alignments:
            raise NotImplementedError("Preserve alignments is not supported")

        if allow_cuda_graphs:
            logging.info("CUDA Graphs are unsupported for `maes_batch`; preceeding pure pytorch decoding")

        if ngram_lm_model is not None:
            expected_blank_index = self.joint.num_classes_with_blank - self.joint.num_extra_outputs - 1
            if self._blank_index != expected_blank_index:
                raise ValueError(f"Invalid blank index: expected {expected_blank_index}, got {self._blank_index}")
            self.ngram_lm_batch = ngram_lm_model

            self.pruning_mode = PruningMode.EARLY if pruning_mode is None else PruningMode(pruning_mode)
            self.blank_lm_score_mode = (
                BlankLMScoreMode.LM_WEIGHTED_FULL
                if blank_lm_score_mode is None
                else BlankLMScoreMode(blank_lm_score_mode)
            )
        else:
            self.ngram_lm_batch = None
            self.blank_lm_score_mode = None
        self.ngram_lm_alpha = ngram_lm_alpha

    def batched_modified_adaptive_expansion_search_torch(
        self,
        encoder_output: torch.Tensor,
        encoder_output_length: torch.Tensor,
    ) -> BatchedBeamHyps:
        """
        Pure PyTorch implementation

        Args:
            encoder_output: output from the encoder
            encoder_output_length: lengths of the utterances in `encoder_output`
        """
        batch_size, max_time, _unused = encoder_output.shape
        device = encoder_output.device

        encoder_output_projected = self.joint.project_encoder(encoder_output)
        float_dtype = encoder_output_projected.dtype

        # init empty batched hypotheses
        batched_hyps = BatchedBeamHyps(
            batch_size=batch_size,
            beam_size=self.beam_size,
            blank_index=self._blank_index,
            init_length=max_time * (self.maes_num_steps + 1) if self.maes_num_steps is not None else max_time,
            device=device,
            float_dtype=float_dtype,
            store_prefix_hashes=True,
        )

        last_labels_wb = torch.full(
            [batch_size, self.beam_size], fill_value=self._SOS, device=device, dtype=torch.long
        )

        batch_indices = (
            torch.arange(batch_size, device=device)[:, None].expand(batch_size, self.beam_size).clone()
        )  # size: batch_size x beam_size
        beam_indices = (
            torch.arange(self.beam_size, device=device)[None, :].expand(batch_size, self.beam_size).clone()
        )  # size: batch_size x beam_size
        expansion_beam_indices = (
            torch.arange(self.beam_size, device=device)[None, :, None]
            .expand(batch_size, self.beam_size, self.maes_num_expansions)
            .clone()
        )  # size: batch_size x beam_size x beam_size + maes_expansion_beta

        time_indices = torch.zeros_like(batch_indices)
        safe_time_indices = torch.zeros_like(time_indices)
        last_timesteps = (encoder_output_length - 1)[:, None].expand(batch_size, self.beam_size)
        active_mask = time_indices <= last_timesteps

        # setup N-gram LM if available
        if self.ngram_lm_batch is not None:
            self.ngram_lm_batch.to(device)
            batch_lm_states = self.ngram_lm_batch.get_init_states(batch_size=batch_size * self.beam_size, bos=True)
            lm_scores, batch_lm_states_candidates = self.ngram_lm_batch.advance(
                states=batch_lm_states
            )  # vocab_size_no_blank
            lm_scores = lm_scores.to(dtype=float_dtype).view(batch_size, self.beam_size, -1) * self.ngram_lm_alpha

        decoder_output, decoder_state, *_ = self.decoder.predict(
            last_labels_wb.view(-1, 1), None, add_sos=False, batch_size=batch_size * self.beam_size
        )
        # do not recalculate joint projection
        decoder_output = self.joint.project_prednet(decoder_output)

        while active_mask.any():  # frames loop
            to_update = active_mask.clone()  # mask for expansions loop

            # step 1: get joint output
            logits = (
                self.joint.joint_after_projection(
                    encoder_output_projected[batch_indices.flatten(), safe_time_indices.flatten()].unsqueeze(1),
                    decoder_output,
                )
                .squeeze(1)
                .squeeze(1)
            )
            logps = torch.log_softmax(logits, dim=-1).view(batch_size, self.beam_size, -1)

            # step 2: perform prefix search
            updated_logps = self.combine_scores(logps, lm_scores) if self.ngram_lm_batch is not None else logps
            batched_hyps.recombine_prefixes(updated_logps, active_mask)

            expansion_steps = 0
            # step 3: performs `maes_num_steps` non-blank expansions
            while to_update.any() and expansion_steps < self.maes_num_steps:  # expansions loop
                # step 3.1: get `maes_num_expansion` best expansions (in total beam x maes_num_expansion expansions)
                if self.ngram_lm_batch is None:
                    # step 3.1.1: choose topk expansions (beam x beam hypotheses for each sample)
                    label_logps, next_labels = logps.topk(self.maes_num_expansions, dim=-1, largest=True, sorted=True)
                    next_hyps_probs = batched_hyps.scores.unsqueeze(-1) + label_logps

                    # step 3.1.2: prune with threshold parameter gamma
                    next_hyps_probs[
                        next_hyps_probs <= next_hyps_probs.max(dim=-1, keepdim=True).values - self.maes_expansion_gamma
                    ] = INACTIVE_SCORE
                else:
                    next_labels, next_hyps_probs = self.topk_lm(batched_hyps, lm_scores, logps)

                # step 3.2: get `beam` best expansions
                # step 3.2.1: mask inactive hypotheses
                next_labels = torch.where(to_update.unsqueeze(-1), next_labels, NON_EXISTENT_LABEL_VALUE)
                next_hyps_probs = torch.where(to_update.unsqueeze(-1), next_hyps_probs, INACTIVE_SCORE)

                # step 3.2.2: remove duplicate hypotheses
                next_hyps_probs = batched_hyps.remove_duplicates(next_labels, next_hyps_probs)

                # step 3.2.3: add hypotheses from the previous expansion steps of current frame hypotheses to the beam.
                # Expansions from step s are compared against the top beam expansions from steps 1 to s-1.
                next_hyps_probs[..., -1] = torch.where(to_update, next_hyps_probs[..., -1], batched_hyps.scores)

                # step 3.2.4: get top-k expansions
                next_hyps_probs, idx = next_hyps_probs.view(batch_size, -1).topk(
                    self.beam_size, dim=-1, largest=True, sorted=True
                )
                next_labels = next_labels.view(batch_size, -1)[batch_indices, idx]
                hyp_indices = expansion_beam_indices.view(batch_size, -1)[batch_indices, idx]

                # step 3.3: update batched beam hypotheses structure
                batched_hyps.add_results_(hyp_indices, next_labels, next_hyps_probs)

                # step 3.4: update
                last_labels_wb = torch.where(next_labels >= 0, next_labels, self._blank_index)
                preserve_state = last_labels_wb == self._blank_index

                # size: decoder_output [(B x Beam), 1, Dim]
                # size: state tuple, each is of [Layers, (BxBeam), Dim]
                # step 3.5: update decoder + lm state
                # step 3.5.1: storing current decoder output and states of extended hypotheses
                prev_decoder_output = torch.gather(
                    decoder_output.view(batch_size, self.beam_size, 1, -1),
                    dim=1,
                    index=hyp_indices[:, :, None, None].expand(
                        batch_size, self.beam_size, 1, decoder_output.shape[-1]
                    ),
                ).view(batch_size * self.beam_size, 1, -1)
                prev_decoder_state = self.decoder.batch_aggregate_states_beam(
                    decoder_state, batch_size, self.beam_size, hyp_indices
                )

                # step 3.5.2: get next decoder output and states for extended hypotheses
                decoder_output, decoder_state, *_ = self.decoder.predict(
                    last_labels_wb.view(-1, 1),
                    prev_decoder_state,
                    add_sos=False,
                    batch_size=batch_size * self.beam_size,
                )
                decoder_output = self.joint.project_prednet(decoder_output)  # do not recalculate joint projection

                # step 3.5.3: update decoder state and output only for non-blank and active hypotheses
                decoder_output = torch.where(
                    preserve_state.view(-1)[:, None, None], prev_decoder_output, decoder_output
                )
                self.decoder.batch_replace_states_mask(
                    src_states=prev_decoder_state, dst_states=decoder_state, mask=preserve_state.view(-1)
                )

                if self.ngram_lm_batch is not None:
                    # batch_lm_states: size: [(batch_size x beam_size)]
                    # batch_lm_states_candidates: [(batch_size x beam_size) x V (without blank)]
                    batch_lm_states_candidates = torch.gather(
                        batch_lm_states_candidates.view(batch_size, self.beam_size, -1),
                        dim=1,
                        index=hyp_indices[:, :, None].expand(
                            batch_size, self.beam_size, batch_lm_states_candidates.shape[-1]
                        ),
                    )
                    batch_lm_states_prev = torch.gather(
                        batch_lm_states.view(batch_size, self.beam_size), dim=1, index=hyp_indices
                    )
                    last_labels_wb_blank_replaced = torch.where(preserve_state, 0, last_labels_wb)

                    batch_lm_states = torch.gather(
                        batch_lm_states_candidates, dim=-1, index=last_labels_wb_blank_replaced.unsqueeze(-1)
                    ).squeeze(-1)
                    batch_lm_states = torch.where(preserve_state, batch_lm_states_prev, batch_lm_states).view(-1)

                    lm_scores, batch_lm_states_candidates = self.ngram_lm_batch.advance(
                        states=batch_lm_states
                    )  # vocab_size_no_blank
                    lm_scores = (
                        lm_scores.to(dtype=float_dtype).view(batch_size, self.beam_size, -1) * self.ngram_lm_alpha
                    )

                # step 3.6: get log-probs for next expansion step
                logits = self.joint.joint_after_projection(
                    encoder_output_projected[batch_indices.flatten(), safe_time_indices.flatten()].unsqueeze(1),
                    decoder_output,
                )
                logps = torch.log_softmax(logits, dim=-1).squeeze(1).squeeze(1).view(batch_size, self.beam_size, -1)
                to_update = torch.logical_and(to_update, last_labels_wb != self._blank_index)

                expansion_steps += 1
            if to_update.any():
                # step 4: force blank to active hypotheses
                next_hyps_probs = torch.where(to_update, batched_hyps.scores + logps[..., -1], batched_hyps.scores)
                next_labels = torch.where(to_update, self._blank_index, -1)
                batched_hyps.add_results_(beam_indices, next_labels, next_hyps_probs)

            # step 5: update time indices + active mask
            time_indices += 1
            active_mask = time_indices <= last_timesteps
            safe_time_indices = torch.where(active_mask, time_indices, last_timesteps)

        return batched_hyps

    def combine_scores(self, log_probs, lm_scores):
        """
        Combines acoustic model log probabilities with language model scores based on the specified blank LM score mode.

        Args:
            log_probs (torch.Tensor): Log probabilities from the acoustic model.
                Shape: (..., vocab_size), where the last dimension corresponds to the vocabulary size.
            lm_scores (torch.Tensor): Scores from the language model.
                Shape: (..., vocab_size - 1), excluding the blank token.

        Returns:
            torch.Tensor: Combined scores with the same shape as `log_probs`.

        Raises:
            NotImplementedError: If the `blank_lm_score_mode` is not supported.
        """
        res = log_probs.clone()
        if self.blank_lm_score_mode is BlankLMScoreMode.NO_SCORE:
            # choosing topk from acoustic and Ngram models
            res[..., :-1] += lm_scores
        else:
            blank_logprob = log_probs[..., -1]
            non_blank_logprob = torch.log1p(-torch.clamp(torch.exp(blank_logprob), max=1.0 - 1e-6))
            res[..., :-1] += non_blank_logprob.unsqueeze(-1) * self.ngram_lm_alpha + lm_scores
            res[..., -1] *= 1 + self.ngram_lm_alpha

        return res

    def topk_lm(self, batched_hyps, lm_scores, log_probs):
        """
        Performs top-k selection and pruning for language model (LM) and automatic speech recognition (ASR) outputs
        based on the specified pruning and blank scoring modes.
        Args:
            batched_hyps (object): Hypotheses from the ASR model, containing scores and other relevant information.
            lm_scores (Tensor): Precomputed language model scores for the current batch.
            log_probs (Tensor): Log probabilities from the ASR model.
        Returns:
            Tuple[Tensor, Tensor]:
                - labels (Tensor): The top-k labels selected after pruning and scoring.
                - total_logps (Tensor): The corresponding total log probabilities for the selected labels.
        Raises:
            NotImplementedError: If the combination of `blank_lm_score_mode` and `pruning_mode` is not implemented.
        """

        match self.pruning_mode, self.blank_lm_score_mode:
            case PruningMode.LATE, BlankLMScoreMode.NO_SCORE | BlankLMScoreMode.LM_WEIGHTED_FULL:
                # step 1: combining LM and ASR outputs + choosing top `beam` most probable
                log_probs = self.combine_scores(log_probs, lm_scores)
                label_logps, labels = log_probs.topk(
                    self.beam_size + self.maes_expansion_beta, dim=-1, largest=True, sorted=True
                )

                # step 2: pruning with threshold gamma
                total_logps = batched_hyps.scores.unsqueeze(-1) + label_logps
                total_logps[
                    total_logps <= total_logps.max(dim=-1, keepdim=True).values - self.maes_expansion_gamma
                ] = INACTIVE_SCORE

            case PruningMode.EARLY, BlankLMScoreMode.NO_SCORE:
                # step 1: choosing topk from ASR output
                label_logps, labels = log_probs.topk(
                    self.beam_size + self.maes_expansion_beta, dim=-1, largest=True, sorted=True
                )

                # step 2: pruning with threshold gamma
                total_logps = batched_hyps.scores.unsqueeze(-1) + label_logps
                total_logps[
                    total_logps <= total_logps.max(dim=-1, keepdim=True).values - self.maes_expansion_gamma
                ] = INACTIVE_SCORE

                # step 3: adding scores from ngram LM
                masked_labels = torch.where(labels == self._blank_index, 0, labels)
                total_logps = torch.where(
                    labels == self._blank_index,
                    total_logps,
                    total_logps + torch.gather(lm_scores, dim=-1, index=masked_labels),
                )

            case PruningMode.EARLY, BlankLMScoreMode.LM_WEIGHTED_FULL:
                # step 1: choosing topk from ASR output
                label_logps, labels = log_probs.topk(
                    self.beam_size + self.maes_expansion_beta, dim=-1, largest=True, sorted=True
                )

                # step 2: pruning with threshold gamma
                total_logps = batched_hyps.scores.unsqueeze(-1) + label_logps
                label_logps[
                    total_logps <= total_logps.max(dim=-1, keepdim=True).values - self.maes_expansion_gamma
                ] = INACTIVE_SCORE

                # step 3: adding scores from ngram LM
                blank_logprob = log_probs[..., -1]
                non_blank_logprob = torch.log1p(-torch.clamp(torch.exp(blank_logprob), max=1.0 - 1e-6))

                masked_labels = torch.where(labels == self._blank_index, 0, labels)
                total_logps = torch.where(
                    labels == self._blank_index,
                    total_logps + label_logps * (1 + self.ngram_lm_alpha),
                    total_logps
                    + label_logps
                    + non_blank_logprob.unsqueeze(-1) * self.ngram_lm_alpha
                    + torch.gather(lm_scores, dim=-1, index=masked_labels),
                )

            case _:
                raise NotImplementedError(
                    f"Unsupported pruning mode {self.pruning_mode} or blank LM score mode {self.blank_lm_score_mode}"
                )

        return labels, total_logps

    def __call__(
        self,
        x: torch.Tensor,
        out_len: torch.Tensor,
    ) -> BatchedBeamHyps:
        return self.batched_modified_adaptive_expansion_search_torch(encoder_output=x, encoder_output_length=out_len)
