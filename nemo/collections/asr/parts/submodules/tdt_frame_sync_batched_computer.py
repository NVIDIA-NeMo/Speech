# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from nemo.collections.asr.parts.submodules.ngram_lm import NGramGPULanguageModel
from nemo.collections.asr.parts.utils.batched_beam_decoding_utils import (
    INACTIVE_SCORE,
    INIT_HASH_VALUE,
    BlankLMScoreMode,
    PruningMode,
    hash_text,
)
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis, NBestHypotheses
from nemo.collections.common.parts.optional_cuda_graphs import WithOptionalCudaGraphs
from nemo.utils import logging


class FrameSyncBatchedTDTComputer(WithOptionalCudaGraphs):
    """
    Frame-synchronous batched beam search for TDT models (strategy ``frame_sync_batch``).

    ``ModifiedALSDBatchedTDTComputer`` extends every hypothesis by one step per iteration. A TDT step advances by
    the predicted duration, so its beam holds hypotheses that have consumed different amounts of audio, and they are
    ranked against each other. Here every hypothesis waits in the bucket of the frame it has reached:

    - only hypotheses in the same bucket are recombined (identical transcripts, scores log-added) and pruned to
      ``beam_size``;
    - blank is expanded with each of its top non-zero durations, labels as the top (label, duration) pairs;
    - a label with duration 0 stays on its frame, at most ``max_symbols_per_step`` times in a row, after which the
      hypothesis moves on by one frame.

    sherpa-onnx uses the same search for TDT models (https://github.com/k2-fsa/sherpa-onnx/pull/4084).

    Every utterance has its own frame pointer, so an utterance that revisits a frame does not hold the batch back.
    The loop body does not synchronize with the host: with CUDA graphs, one iteration is captured and replayed,
    and the host checks for the end of the batch every ``check_every`` iterations.
    """

    INITIAL_MAX_TIME = 375  # initial max time, used to allocate the decoding state
    INITIAL_NODES_PER_FRAME = 2.0  # label nodes allocated per frame and hypothesis; doubled when a batch needs more
    BLANK_EPS = 1e-2  # bound on P(blank) in the `lm_weighted_full` blank scoring, as in malsd_batch

    def __init__(
        self,
        decoder,
        joint,
        durations: List[int],
        blank_index: int,
        beam_size: int,
        max_symbols_per_step: Optional[int] = 10,
        fusion_models: Optional[List[NGramGPULanguageModel]] = None,
        fusion_models_alpha: Optional[List[float]] = None,
        blank_lm_score_mode: Optional[str | BlankLMScoreMode] = None,
        pruning_mode: Optional[str | PruningMode] = None,
        beam_beta: float = 0.0,
        allow_cuda_graphs: bool = False,
        check_every: int = 8,
    ):
        """
        Init method.
        Args:
            decoder: Prediction network from TDT, with states of shape [layers, batch, hidden] (e.g., RNNTDecoder)
            joint: Joint module from TDT
            durations: TDT durations
            blank_index: index of blank symbol, the last label
            beam_size: beam size, the number of hypotheses kept per frame
            max_symbols_per_step: max zero-duration labels in a row on one frame
            fusion_models: list of fusion models (ngram_lm_model and boosting_tree_model)
            fusion_models_alpha: list of weights for the fusion models scores
            blank_lm_score_mode: mode for scoring blank symbol with fusion models
            pruning_mode: mode for pruning hypotheses with fusion models
            beam_beta: score added for every non-blank label, so the hypothesis score is
                acoustic_score + fusion_scores + beam_beta * seq_length; positive values counter deletions
            allow_cuda_graphs: whether to allow CUDA graphs
            check_every: number of iterations between checks for the end of the batch
        """
        super().__init__()
        if beam_size < 1:
            raise ValueError(f"Expected beam_size >= 1, got {beam_size}")
        if max_symbols_per_step is None:
            logging.warning(
                "Max symbols per step is None, which is not allowed with frame_sync_batch. Setting to `10`"
            )
            max_symbols_per_step = 10
        if max_symbols_per_step < 1:
            raise ValueError(f"Expected max_symbols_per_step >= 1, got {max_symbols_per_step}")
        if not any(duration > 0 for duration in durations):
            raise ValueError(f"Durations {list(durations)} have no non-zero value for blank")
        expected_blank_index = joint.num_classes_with_blank - joint.num_extra_outputs - 1
        if blank_index != expected_blank_index:
            raise ValueError(f"Invalid blank index: expected {expected_blank_index}, got {blank_index}")

        self.decoder = decoder
        self.joint = joint
        self.durations = list(durations)
        self._blank_index = blank_index
        self.beam_size = beam_size
        self.label_topk = min(beam_size, blank_index)  # top labels expanded per hypothesis
        self.max_symbols = max_symbols_per_step
        self.beam_beta = beam_beta
        self.check_every = max(1, check_every)

        self.fusion_models: list[NGramGPULanguageModel] = fusion_models if fusion_models is not None else []
        self.fusion_models_alpha: list[float] = fusion_models_alpha if fusion_models_alpha is not None else []
        if len(self.fusion_models) != len(self.fusion_models_alpha):
            raise ValueError("Expected the same number of fusion models and fusion models alphas")
        if self.fusion_models:
            self.pruning_mode = PruningMode.EARLY if pruning_mode is None else PruningMode(pruning_mode)
            self.blank_lm_score_mode = (
                BlankLMScoreMode.LM_WEIGHTED_FULL
                if blank_lm_score_mode is None
                else BlankLMScoreMode(blank_lm_score_mode)
            )
        else:
            self.pruning_mode = None
            self.blank_lm_score_mode = None

        if allow_cuda_graphs:
            for fusion_model in self.fusion_models:
                if not fusion_model.compatible_with_cuda_graphs():
                    logging.warning(
                        "Fusion model used that is incompatible with CUDA graphs."
                        " Switching off CUDA graphs, decoding may be slow."
                    )
                    allow_cuda_graphs = False
                    break
        self.allow_cuda_graphs = allow_cuda_graphs
        self.cuda_graphs_enabled = False
        self.nodes_per_frame = self.INITIAL_NODES_PER_FRAME
        self.state: Optional[_FrameSyncState] = None
        self.maybe_enable_cuda_graphs()

    def maybe_enable_cuda_graphs(self) -> bool:
        """Enable CUDA graphs if allowed"""
        if self.cuda_graphs_enabled or not self.allow_cuda_graphs:
            return False
        self.cuda_graphs_enabled = True
        self.reset_cuda_graphs_state()
        return True

    def disable_cuda_graphs(self) -> bool:
        """Disable CUDA graphs, can be used to disable graphs temporary, e.g., in training process"""
        if not self.cuda_graphs_enabled:
            return False
        self.cuda_graphs_enabled = False
        self.reset_cuda_graphs_state()
        return True

    def reset_cuda_graphs_state(self):
        """Release the decoding state and the captured graph"""
        self.state = None

    def __call__(
        self,
        x: torch.Tensor,
        out_len: torch.Tensor,
        prev_batched_state=None,
        multi_biasing_ids: Optional[torch.Tensor] = None,
    ) -> tuple["FrameSyncBatchedHyps", None]:
        """
        Args:
            x: encoder output, tensor of shape [B, T, D]
            out_len: encoder output lengths, tensor of shape [B]
            prev_batched_state: not supported, frame-synchronous search decodes whole utterances
            multi_biasing_ids: not supported, per-stream biasing is not implemented
        Returns:
            final hypotheses and the decoding state (always None)
        """
        if prev_batched_state is not None:
            raise NotImplementedError("Decoding by chunks is not supported by frame_sync_batch")
        if multi_biasing_ids is not None:
            raise NotImplementedError("Per-stream biasing is not supported by frame_sync_batch")
        # Autocast is disabled, as in malsd_batch with CUDA graphs: the captured graph must not use its weights cache.
        with torch.inference_mode(), torch.amp.autocast(device_type=x.device.type, enabled=False):
            return self.frame_sync_search(encoder_output=x, encoder_output_length=out_len), None

    def frame_sync_search(
        self, encoder_output: torch.Tensor, encoder_output_length: torch.Tensor
    ) -> "FrameSyncBatchedHyps":
        """
        Frame-synchronous beam search over a batch of whole utterances.

        Args:
            encoder_output: encoder output, tensor of shape [B, T, D]
            encoder_output_length: encoder output lengths, tensor of shape [B]
        Returns:
            final hypotheses, `beam_size` per utterance
        """
        batch_size = encoder_output.shape[0]
        encoder_output = encoder_output.to(_dtype_of(self.joint, encoder_output.dtype))
        encoder_output = self.joint.project_encoder(encoder_output)
        encoder_output_length = encoder_output_length.to(device=encoder_output.device, dtype=torch.long)
        if batch_size == 0:
            return FrameSyncBatchedHyps.empty(self.beam_size)

        while True:
            if self.state is None or self.state.need_reinit(encoder_output):
                self._init_state(encoder_output)
            self._start_batch(encoder_output, encoder_output_length)
            self._run_loop()
            if not bool(self.state.nodes_overflow):
                break
            # later batches are as likely to need the room, so the budget stays raised
            self.nodes_per_frame *= 2
            self.state = None
            logging.info(
                f"frame_sync_batch: batch outgrew label nodes, retrying with {self.nodes_per_frame} per frame"
            )

        state = self.state
        finished = state.finished_bucket
        return FrameSyncBatchedHyps(
            scores=state.hyps["score"][finished, :batch_size].cpu(),
            lengths=state.hyps["length"][finished, :batch_size].cpu(),
            last_nodes=state.hyps["node"][finished, :batch_size].cpu(),
            nodes=state.nodes[:, : int(state.num_nodes)].cpu().numpy(),
        )

    def _init_state(self, encoder_output: torch.Tensor):
        """Allocate the state for the batch (and capture the CUDA graph)"""
        self.state = None  # release the old state first
        batch_size, max_time = encoder_output.shape[:2]
        max_time = max(max_time, self.INITIAL_MAX_TIME)
        self.state = _FrameSyncState(
            computer=self,
            batch_size=batch_size,
            max_time=max_time,
            encoder_dim=encoder_output.shape[-1],
            encoder_dtype=encoder_output.dtype,
            device=encoder_output.device,
            max_nodes=int(self.nodes_per_frame * batch_size * self.beam_size * max_time),
        )
        if self.cuda_graphs_enabled and encoder_output.device.type == "cuda":
            self._capture_graph()

    def _capture_graph(self):
        """Capture one iteration of the search as a CUDA graph"""
        state = self.state
        is_ddp = torch.distributed.is_available() and torch.distributed.is_initialized()
        # 11 warmup steps required in DDP mode
        # see https://pytorch.org/docs/stable/notes/cuda.html#usage-with-distributeddataparallel
        num_warmup_steps = 11 if is_ddp else 3
        state.encoder_output_length.fill_(state.max_time)
        stream = torch.cuda.Stream(state.device)
        stream.wait_stream(torch.cuda.current_stream(device=state.device))
        with torch.cuda.stream(stream):
            self._reset_hypotheses()
            for _ in range(num_warmup_steps):
                self._step()
        torch.cuda.current_stream(device=state.device).wait_stream(stream)

        self._reset_hypotheses()
        state.graph = torch.cuda.CUDAGraph()
        stream_for_graph = torch.cuda.Stream(state.device)
        stream_for_graph.wait_stream(torch.cuda.default_stream(state.device))
        with (
            torch.cuda.stream(stream_for_graph),
            torch.cuda.graph(state.graph, stream=stream_for_graph, capture_error_mode="thread_local"),
        ):
            self._step()

    def _start_batch(self, encoder_output: torch.Tensor, encoder_output_length: torch.Tensor):
        """Copy the batch into the state; rows past the batch have length 0 and are finished from the start"""
        state = self.state
        batch_size, max_time = encoder_output.shape[:2]
        state.encoder_output.zero_()
        state.encoder_output[:batch_size, :max_time].copy_(encoder_output)
        state.encoder_output_length.zero_()
        state.encoder_output_length[:batch_size].copy_(encoder_output_length)
        self._reset_hypotheses()

    def _reset_hypotheses(self):
        """Empty all buckets but the first, which gets the start hypothesis of every utterance"""
        state = self.state
        hyps = state.hyps
        hyps["score"].fill_(INACTIVE_SCORE)
        hyps["length"].zero_()
        hyps["hash"].fill_(INIT_HASH_VALUE)
        hyps["node"].fill_(-1)
        hyps["last_label"].fill_(self._blank_index)
        hyps["num_zero"].zero_()
        for i, init_states in enumerate(state.init_fusion_states):
            hyps[f"fusion_{i}"].copy_(init_states)
        for buffer, init_state in zip(state.decoder_states, state.init_decoder_state):
            buffer.copy_(init_state)
        not_empty = state.encoder_output_length > 0
        start_bucket = torch.where(not_empty, 0, state.finished_bucket)
        hyps["score"][start_bucket, state.batch_indices, 0] = 0.0
        state.time.zero_()
        state.bucket.zero_()
        state.done.copy_(~not_empty)
        state.num_nodes.zero_()
        state.nodes_overflow.zero_()

    def _run_loop(self):
        """Iterate until every utterance has finished, synchronizing with the host every `check_every` steps"""
        state = self.state
        step = state.graph.replay if state.graph is not None else self._step
        # a frame is expanded at most `max_symbols + 1` times
        max_steps = (self.max_symbols + 1) * state.max_time + self.check_every
        for _ in range(0, max_steps, self.check_every):
            for _ in range(self.check_every):
                step()
            if bool(torch.logical_or(state.done.all(), state.nodes_overflow)):
                return
        raise RuntimeError(f"frame_sync_batch did not finish in {max_steps} steps")

    def _step(self):
        """One iteration: every live utterance expands the hypotheses in the bucket of its current frame"""
        state = self.state
        hyps, decoder_states = state.hyps, state.decoder_states
        batch_size, beam = state.batch_size, self.beam_size
        num_buckets, ring_size, finished = state.num_buckets, state.ring_size, state.finished_bucket
        batch_indices, time, bucket = state.batch_indices, state.time, state.bucket
        lengths = state.encoder_output_length
        live = ~state.done

        parents = {key: value[bucket, batch_indices] for key, value in hyps.items()}  # [B, beam]
        parents["score"] = torch.where(live[:, None], parents["score"], INACTIVE_SCORE)
        parent_decoder_states = [buffer[:, bucket, batch_indices] for buffer in decoder_states]
        hyps["score"].index_put_((bucket, batch_indices), state.inactive_scores)
        active = parents["score"] > INACTIVE_SCORE
        forced = active & (parents["num_zero"] >= self.max_symbols)
        expand = active & ~forced

        decoder_output, new_decoder_state = self.decoder.predict(
            parents["last_label"].view(-1, 1),
            tuple(s.reshape(s.shape[0], batch_size * beam, s.shape[-1]) for s in parent_decoder_states),
            add_sos=False,
            batch_size=batch_size * beam,
        )[:2]
        decoder_output = self.joint.project_prednet(decoder_output)  # [B * beam, 1, H]
        frame = torch.minimum(time, (lengths - 1).clamp(min=0))
        encoder_frame = state.encoder_output[batch_indices, frame][:, None, None, :]
        encoder_frame = encoder_frame.expand(batch_size, beam, 1, -1).reshape(batch_size * beam, 1, -1)
        logits = self.joint.joint_after_projection(encoder_frame, decoder_output).reshape(batch_size, beam, -1).float()
        num_durations = len(self.durations)
        log_probs = F.log_softmax(logits[..., :-num_durations], dim=-1)
        duration_log_probs = F.log_softmax(logits[..., -num_durations:], dim=-1)
        label_scores, labels, blank_step_scores, fusion_states_candidates = self._topk_labels(
            log_probs, [parents[f"fusion_{i}"] for i in range(len(self.fusion_models))]
        )

        # blank: the parent's labels and states, moved on by each of the top non-zero durations
        blank_duration_scores, blank_duration_indices = duration_log_probs[..., state.nonzero_durations].topk(
            state.blank_topk, dim=-1
        )
        blank_durations = state.durations[state.nonzero_durations[blank_duration_indices]]
        blank_scores = parents["score"][..., None] + blank_step_scores[..., None] + blank_duration_scores
        blank_scores = torch.where(expand[..., None], blank_scores, INACTIVE_SCORE)

        # labels: the top (label, duration) pairs among the top labels and the top durations
        duration_scores, duration_indices = duration_log_probs.topk(state.duration_topk, dim=-1)
        pair_scores = (label_scores[..., :, None] + duration_scores[..., None, :]).flatten(-2)
        pair_scores, pairs = pair_scores.topk(state.num_label_children, dim=-1)
        child_labels = labels.gather(-1, torch.div(pairs, state.duration_topk, rounding_mode="floor"))
        child_durations = state.durations[duration_indices.gather(-1, pairs % state.duration_topk)]
        child_scores = torch.where(expand[..., None], parents["score"][..., None] + pair_scores, INACTIVE_SCORE)

        def target_bucket(duration: torch.Tensor) -> torch.Tensor:
            shape = (batch_size,) + (1,) * (duration.dim() - 1)
            next_bucket = (bucket.view(shape) + duration) % ring_size
            target = torch.where(time.view(shape) + duration >= lengths.view(shape), finished, next_bucket)
            return target.reshape(batch_size, -1)

        def repeat(value: torch.Tensor, times: int) -> torch.Tensor:
            return value[..., None].expand(batch_size, beam, times).reshape(batch_size, -1)

        # candidates: [hypotheses in buckets, blank children, label children, forced moves]
        beam_indices = state.beam_indices
        candidates = {
            "score": [
                blank_scores.reshape(batch_size, -1),
                child_scores.reshape(batch_size, -1),
                torch.where(forced, parents["score"], INACTIVE_SCORE),
            ],
            "target": [target_bucket(blank_durations), target_bucket(child_durations), target_bucket(state.ones)],
            "length": [
                repeat(parents["length"], state.blank_topk),
                repeat(parents["length"] + 1, state.num_label_children),
                parents["length"],
            ],
            "hash": [
                repeat(parents["hash"], state.blank_topk),
                hash_text(parents["hash"][..., None], child_labels).reshape(batch_size, -1),
                parents["hash"],
            ],
            # label children get their own node below, if they are kept
            "node": [
                repeat(parents["node"], state.blank_topk),
                repeat(parents["node"], state.num_label_children),
                parents["node"],
            ],
            "last_label": [
                repeat(parents["last_label"], state.blank_topk),
                child_labels.reshape(batch_size, -1),
                parents["last_label"],
            ],
            "num_zero": [
                torch.zeros_like(blank_durations).reshape(batch_size, -1),
                torch.where(child_durations > 0, 0, parents["num_zero"][..., None] + 1).reshape(batch_size, -1),
                torch.zeros_like(parents["num_zero"]),
            ],
            # index into [states in buckets, parents' states, parents' states after their last label]
            "source": [
                repeat(num_buckets * beam + beam_indices, state.blank_topk),
                repeat((num_buckets + 1) * beam + beam_indices, state.num_label_children),
                num_buckets * beam + beam_indices,
            ],
        }
        for i, states_candidates in enumerate(fusion_states_candidates):
            fusion_states = parents[f"fusion_{i}"]
            candidates[f"fusion_{i}"] = [
                repeat(fusion_states, state.blank_topk),
                states_candidates.gather(-1, child_labels).reshape(batch_size, -1),
                fusion_states,
            ]
        stored = {key: value.permute(1, 0, 2).reshape(batch_size, -1) for key, value in hyps.items()}
        stored["target"] = state.stored_target
        stored["source"] = state.stored_source
        candidates = {key: torch.cat([stored[key]] + value, dim=1) for key, value in candidates.items()}

        top_scores, top_indices = _merge_and_prune(candidates, num_buckets=num_buckets, beam_size=beam)
        top_indices = top_indices.view(batch_size, -1)
        top_scores = top_scores.view(batch_size, -1)

        # nodes for the kept label children: parent node, label, frame, duration
        num_children = beam * state.num_label_children
        label_offsets = top_indices - state.first_label_candidate
        is_new = (label_offsets >= 0) & (label_offsets < num_children)
        is_new = is_new & (top_scores > INACTIVE_SCORE) & live[:, None]
        new_nodes = state.num_nodes + is_new.view(-1).cumsum(0).view(batch_size, -1) - 1
        fits = new_nodes < state.max_nodes
        state.nodes_overflow.logical_or_((is_new & ~fits).any())
        node_indices = torch.where(is_new & fits, new_nodes, state.max_nodes)  # the last column is scratch
        top_nodes = candidates["node"].gather(1, top_indices)
        new_entries = torch.stack(
            [
                top_nodes,
                candidates["last_label"].gather(1, top_indices),
                time[:, None].expand_as(top_indices),
                child_durations.view(batch_size, -1).gather(1, label_offsets.clamp(0, num_children - 1)),
            ]
        )
        state.nodes[:, node_indices.view(-1)] = new_entries.view(4, -1).to(state.nodes.dtype)
        state.num_nodes.add_(is_new.sum())

        keep = live[None, :, None]
        for key, buffer in hyps.items():
            if key == "score":
                new_values = top_scores
            elif key == "node":
                new_values = torch.where(is_new, new_nodes, top_nodes)
            else:
                new_values = candidates[key].gather(1, top_indices)
            new_values = new_values.view(batch_size, num_buckets, beam).permute(1, 0, 2)
            buffer.copy_(torch.where(keep, new_values, buffer))

        sources = candidates["source"].gather(1, top_indices)
        for buffer, parent_state, new_state in zip(decoder_states, parent_decoder_states, new_decoder_state):
            num_layers, hidden = buffer.shape[0], buffer.shape[-1]
            bank = torch.cat(
                [
                    buffer.permute(0, 2, 1, 3, 4).reshape(num_layers, batch_size, num_buckets * beam, hidden),
                    parent_state.reshape(num_layers, batch_size, beam, hidden),
                    new_state.reshape(num_layers, batch_size, beam, hidden),
                ],
                dim=2,
            )
            picked = bank.gather(2, sources[None, :, :, None].expand(num_layers, -1, -1, hidden))
            picked = picked.view(num_layers, batch_size, num_buckets, beam, hidden).permute(0, 2, 1, 3, 4)
            buffer.copy_(torch.where(live.view(1, 1, batch_size, 1, 1), picked, buffer))

        # stay on the frame while its bucket holds zero-duration arrivals, else move to the next occupied frame
        occupied = (hyps["score"][:ring_size] > INACTIVE_SCORE).any(-1)  # [ring_size, B]
        ahead = occupied.gather(0, (bucket[None, :] + state.ring_offsets[:, None]) % ring_size)
        move = ahead.int().argmax(0)
        done = state.done | ~ahead.any(0)
        state.time.copy_(torch.where(done, time, time + move))
        state.bucket.copy_(torch.where(done, bucket, (bucket + move) % ring_size))
        state.done.copy_(done)

    def _topk_labels(
        self, log_probs: torch.Tensor, fusion_states_list: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        """
        Top labels of every hypothesis with the fusion models, see `_topk_fusion_tokens` of malsd_batch.

        Args:
            log_probs: label and blank log-probs, tensor of shape [B, beam, V + 1]
            fusion_states_list: states of the fusion models, tensors of shape [B, beam]
        Returns:
            label scores and labels [B, beam, label_topk], blank scores [B, beam],
            successor states of the fusion models [B, beam, V]
        """
        batch_size, beam = log_probs.shape[:2]
        label_log_probs = log_probs[..., : self._blank_index]
        blank_log_probs = log_probs[..., self._blank_index]
        if not self.fusion_models:
            label_scores, labels = label_log_probs.topk(self.label_topk, dim=-1)
            return label_scores + self.beam_beta, labels, blank_log_probs, []

        fusion_scores = self.beam_beta
        fusion_states_candidates = []
        for fusion_model, alpha, states in zip(self.fusion_models, self.fusion_models_alpha, fusion_states_list):
            scores, states_candidates = fusion_model.advance(states=states.view(-1))
            fusion_scores = fusion_scores + alpha * scores.to(log_probs.dtype).view(batch_size, beam, -1)
            fusion_states_candidates.append(states_candidates.view(batch_size, beam, -1))
        if self.blank_lm_score_mode is BlankLMScoreMode.LM_WEIGHTED_FULL:
            alpha_sum = sum(self.fusion_models_alpha)
            non_blank_log_probs = torch.log1p(-torch.clamp(blank_log_probs.exp(), max=1.0 - self.BLANK_EPS))
            fusion_scores = fusion_scores + alpha_sum * non_blank_log_probs[..., None]
            blank_log_probs = blank_log_probs * (1 + alpha_sum)
        elif self.blank_lm_score_mode is not BlankLMScoreMode.NO_SCORE:
            raise NotImplementedError(f"Unsupported blank LM score mode {self.blank_lm_score_mode}")

        match self.pruning_mode:
            case PruningMode.LATE:
                label_scores, labels = (label_log_probs + fusion_scores).topk(self.label_topk, dim=-1)
            case PruningMode.EARLY:
                label_scores, labels = label_log_probs.topk(self.label_topk, dim=-1)
                label_scores = label_scores + fusion_scores.gather(-1, labels)
            case _:
                raise NotImplementedError(f"Unsupported pruning mode {self.pruning_mode}")
        return label_scores, labels, blank_log_probs, fusion_states_candidates


class FrameSyncBatchedHyps:
    """Final hypotheses of frame-synchronous beam search, exported as `BatchedBeamHyps` exports them"""

    def __init__(self, scores: torch.Tensor, lengths: torch.Tensor, last_nodes: torch.Tensor, nodes: np.ndarray):
        """
        Args:
            scores: scores of the hypotheses [B, beam], INACTIVE_SCORE for empty slots
            lengths: numbers of labels in the hypotheses [B, beam]
            last_nodes: node of the last label of each hypothesis [B, beam], -1 for an empty transcript
            nodes: label nodes [4, num_nodes]: parent node, label, frame, duration
        """
        self.scores = scores
        self.lengths = lengths
        self.last_nodes = last_nodes
        self.parents, self.labels, self.frames, self.durations = nodes

    @classmethod
    def empty(cls, beam_size: int) -> "FrameSyncBatchedHyps":
        """Hypotheses of an empty batch"""
        return cls(
            scores=torch.zeros(0, beam_size),
            lengths=torch.zeros(0, beam_size, dtype=torch.long),
            last_nodes=torch.zeros(0, beam_size, dtype=torch.long),
            nodes=np.zeros((4, 0), dtype=np.int64),
        )

    def to_hyps_list(self, score_norm: bool = True) -> list[Hypothesis]:
        """
        Best hypothesis of every utterance.
        Args:
            score_norm: if True, hypotheses are ranked by score / (number of labels + 1)
        """
        order = self._order(score_norm)
        return [self._hypothesis(b, int(order[b, 0])) for b in range(self.scores.shape[0])]

    def to_nbest_hyps_list(self, score_norm: bool = True) -> list[NBestHypotheses]:
        """
        N-best hypotheses of every utterance, best first.
        Args:
            score_norm: if True, hypotheses are ranked by score / (number of labels + 1)
        """
        order = self._order(score_norm)
        return [
            NBestHypotheses([self._hypothesis(b, int(k)) for k in order[b] if self.scores[b, k] > INACTIVE_SCORE])
            for b in range(self.scores.shape[0])
        ]

    def _order(self, score_norm: bool) -> torch.Tensor:
        scores = self.scores / (self.lengths + 1) if score_norm else self.scores
        return torch.argsort(scores, dim=-1, descending=True, stable=True)

    def _hypothesis(self, batch_index: int, beam_index: int) -> Hypothesis:
        nodes = []
        node = int(self.last_nodes[batch_index, beam_index])
        while node >= 0:
            nodes.append(node)
            node = int(self.parents[node])
        nodes = np.array(nodes[::-1], dtype=np.int64)
        return Hypothesis(
            score=float(self.scores[batch_index, beam_index]),
            y_sequence=self.labels[nodes].astype(np.int64),
            timestamp=self.frames[nodes].astype(np.int64),
            token_duration=self.durations[nodes].astype(np.int64),
            alignments=None,
            dec_state=None,
        )


class _FrameSyncState:
    """Buffers of frame-synchronous search for up to `batch_size` utterances of up to `max_time` frames"""

    def __init__(
        self,
        computer: FrameSyncBatchedTDTComputer,
        batch_size: int,
        max_time: int,
        encoder_dim: int,
        encoder_dtype: torch.dtype,
        device: torch.device,
        max_nodes: int,
    ):
        beam = computer.beam_size
        self.batch_size = batch_size
        self.max_time = max_time
        self.encoder_dim = encoder_dim
        self.encoder_dtype = encoder_dtype
        self.device = device
        self.max_nodes = max_nodes

        self.durations = torch.tensor(computer.durations, dtype=torch.long, device=device)
        self.nonzero_durations = torch.nonzero(self.durations > 0).squeeze(-1)
        self.blank_topk = min(beam, len(self.nonzero_durations))
        self.duration_topk = min(beam, len(computer.durations))
        # label children per hypothesis, fewer than the beam only for tiny vocabularies
        self.num_label_children = min(beam, computer.label_topk * self.duration_topk)
        # buckets 0..ring_size - 1 hold frames t..t + max duration in rotation, the last one finished hypotheses
        self.ring_size = int(self.durations.max()) + 1
        self.num_buckets = self.ring_size + 1
        self.finished_bucket = self.ring_size
        self.first_label_candidate = self.num_buckets * beam + beam * self.blank_topk  # in the step's candidates

        self.batch_indices = torch.arange(batch_size, device=device)
        self.beam_indices = torch.arange(beam, device=device)[None].expand(batch_size, beam)
        self.ring_offsets = torch.arange(self.ring_size, device=device)
        self.ones = torch.ones([batch_size, beam], dtype=torch.long, device=device)
        # a tensor, since a Python scalar would be copied from the host on every write
        self.inactive_scores = torch.full([batch_size, beam], INACTIVE_SCORE, device=device)
        self.stored_target = (
            torch.arange(self.num_buckets, device=device).repeat_interleave(beam)[None].expand(batch_size, -1)
        )
        self.stored_source = torch.arange(self.num_buckets * beam, device=device)[None].expand(batch_size, -1)

        self.encoder_output = torch.zeros([batch_size, max_time, encoder_dim], dtype=encoder_dtype, device=device)
        self.encoder_output_length = torch.zeros([batch_size], dtype=torch.long, device=device)
        shape = [self.num_buckets, batch_size, beam]
        self.hyps = {
            "score": torch.full(shape, INACTIVE_SCORE, device=device),
            "length": torch.zeros(shape, dtype=torch.long, device=device),
            "hash": torch.full(shape, INIT_HASH_VALUE, dtype=torch.long, device=device),
            # node of the last label, -1 for an empty transcript
            "node": torch.full(shape, -1, dtype=torch.long, device=device),
            # the decoder state of a hypothesis is the state before its last label
            "last_label": torch.full(shape, computer._blank_index, dtype=torch.long, device=device),
            # number of zero-duration labels in a row on the current frame
            "num_zero": torch.zeros(shape, dtype=torch.long, device=device),
        }
        self.init_fusion_states = []
        for i, fusion_model in enumerate(computer.fusion_models):
            fusion_model.to(device)
            init_states = fusion_model.get_init_states(batch_size=batch_size * beam, bos=True).to(device)
            self.init_fusion_states.append(init_states.view(1, batch_size, beam).expand(shape))
            self.hyps[f"fusion_{i}"] = torch.empty(shape, dtype=init_states.dtype, device=device)

        num_states = batch_size * beam
        init_decoder_state = computer.decoder.initialize_state(
            torch.empty([num_states], dtype=_dtype_of(computer.decoder, encoder_dtype), device=device)
        )
        if any(s.dim() != 3 or s.shape[1] != num_states for s in init_decoder_state):
            raise NotImplementedError(
                "frame_sync_batch supports prediction networks with states of shape [layers, batch, hidden]"
            )
        # [layers, buckets, batch, beam, hidden]
        self.init_decoder_state = [
            s.view(s.shape[0], 1, batch_size, beam, s.shape[-1]).expand(-1, self.num_buckets, -1, -1, -1)
            for s in init_decoder_state
        ]
        self.decoder_states = [s.clone(memory_format=torch.contiguous_format) for s in self.init_decoder_state]

        self.time = torch.zeros([batch_size], dtype=torch.long, device=device)  # current frame
        self.bucket = torch.zeros([batch_size], dtype=torch.long, device=device)  # bucket of the current frame
        self.done = torch.zeros([batch_size], dtype=torch.bool, device=device)
        # label nodes: parent node, label, frame, duration; the last column is scratch for the rejected writes
        self.nodes = torch.zeros([4, max_nodes + 1], dtype=torch.int32, device=device)
        self.num_nodes = torch.zeros([], dtype=torch.long, device=device)
        self.nodes_overflow = torch.zeros([], dtype=torch.bool, device=device)
        self.graph: Optional[torch.cuda.CUDAGraph] = None

    def need_reinit(self, encoder_output: torch.Tensor) -> bool:
        """Check if the state needs to be reallocated for the batch"""
        return (
            self.batch_size < encoder_output.shape[0]
            or self.max_time < encoder_output.shape[1]
            or self.encoder_dim != encoder_output.shape[-1]
            or self.encoder_dtype != encoder_output.dtype
            or self.device != encoder_output.device
        )


def _merge_and_prune(
    candidates: dict[str, torch.Tensor], num_buckets: int, beam_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Recombine identical transcripts bound for the same bucket (scores log-added), then keep the best `beam_size`
    candidates of every bucket.

    Args:
        candidates: "score", "target" (bucket), "hash" and "length" of the candidates, tensors of shape [B, N]
        num_buckets: number of buckets
        beam_size: hypotheses kept per bucket
    Returns:
        scores and indices of the kept candidates, tensors of shape [B, num_buckets, beam_size]
    """
    scores, target = candidates["score"], candidates["target"]
    same = (
        (target[:, :, None] == target[:, None, :])
        & (candidates["hash"][:, :, None] == candidates["hash"][:, None, :])
        & (candidates["length"][:, :, None] == candidates["length"][:, None, :])
    )
    group_scores = torch.where(same, scores[:, None, :], INACTIVE_SCORE)
    # the best candidate of a group carries the group's score, argmax picks the first of equals
    is_best = group_scores.argmax(-1) == torch.arange(scores.shape[1], device=scores.device)
    merged_scores = torch.where(
        is_best & (scores > INACTIVE_SCORE), torch.logsumexp(group_scores, dim=-1), INACTIVE_SCORE
    )
    buckets = torch.arange(num_buckets, device=scores.device)
    bucket_scores = torch.where(
        target[:, None, :] == buckets[None, :, None], merged_scores[:, None, :], INACTIVE_SCORE
    )
    return bucket_scores.topk(beam_size, dim=-1)


def _dtype_of(module: torch.nn.Module, default: torch.dtype) -> torch.dtype:
    """dtype of the module's floating point parameters"""
    for parameter in module.parameters():
        if parameter.is_floating_point():
            return parameter.dtype
    return default
