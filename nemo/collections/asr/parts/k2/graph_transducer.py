# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import abc
from contextlib import nullcontext
from typing import ContextManager

import torch
import torch.nn.functional as F

from nemo.core.classes.loss import Loss
from nemo.core.utils.k2_guard import k2
from nemo.core.utils.optional_libs import TRITON_AVAILABLE
from nemo.utils import logging

if TRITON_AVAILABLE:
    from nemo.collections.asr.parts.rnnt_triton.rnnt_logprobs_triton import rnnt_logprobs_triton


def force_float32_context() -> ContextManager:
    """Get context manager to force float32 precision in autocast mode."""
    if torch.is_autocast_enabled():
        return torch.amp.autocast('cuda', dtype=torch.float32)
    return nullcontext()


class GraphTransducerLossBase(Loss):
    """
    Base class for graph transducer losses.
    Implementation of the approach described in "Powerful and Extensible WFST Framework for RNN-Transducer Losses"
    https://ieeexplore.ieee.org/document/10096679

    Compose-Transducer: compose the unit (target text) and temporal schemas (graphs) into lattice.
        Subclass should implement `get_unit_schema` and `get_temporal_schema` methods.
    Grid-Transducer: construct the RNN-T lattice (grid) directly in code.
        Subclass should implement `get_grid` method.
    """

    def __init__(
        self, use_grid_implementation: bool, connect_composed=False, double_scores=False, cast_to_float32=False
    ):
        """

        Args:
            use_grid_implementation: Whether to use the grid implementation (Grid-Transducer).
            connect_composed: Connect graph after composing unit and temporal schemas (only for Compose-Transducer).
                `connect` operation is slow, it is useful for visualization, but not necessary for loss computation.
            double_scores: Use calculation of loss in double precision (float64) in the lattice.
                Does not significantly affect memory usage since the lattice is ~V/2 times smaller
                than the joint tensor.
            cast_to_float32: Force cast joint tensor to float32 before log-softmax calculation.
        """
        super().__init__()
        self.use_grid_implementation = use_grid_implementation
        self.connect_composed = connect_composed
        self.double_scores = double_scores
        self.cast_to_float32 = cast_to_float32

    @abc.abstractmethod
    def get_unit_schema(self, units_tensor: torch.Tensor, vocab_size: int) -> "k2.Fsa":
        """
        Get unit schema (target text) graph for Compose-Transducer.

        Args:
            units_tensor: tensor with target text
            vocab_size: number of labels (including blank). Needed to construct additional eps-arcs (in some cases).

        Returns:
            unit schema graph (k2.Fsa).
            Labels: <unit>:<unit>:<unit_position> (k2.Fsa: labels, aux_labels, unit_positions)
        """
        pass

    @abc.abstractmethod
    def get_temporal_schema(self, num_frames: int, vocab_size: int, device: torch.device) -> "k2.Fsa":
        """
        Get temporal schema graph for Compose-Transducer.

        Args:
            num_frames: length of the sequence (in frames)
            vocab_size: number of labels (including blank)
            device: device for tensor to construct

        Returns:
            temporal schema graph (k2.Fsa).
            Labels: <unit>:<frame_index>. <unit> is a unit from vocab + special units (e.g., additional eps).
        """
        pass

    @abc.abstractmethod
    def get_grid(self, units_tensor: torch.Tensor, num_frames: int, vocab_size: int) -> "k2.Fsa":
        """
        Construct the transducer lattice (grid) directly for Grid-Transducer.

        Args:
            units_tensor: tensor with target text
            num_frames: length of the sequence (in frames)
            vocab_size: number of labels (including blank)

        Returns:
            transducer lattice (k2.Fsa).
            Labels: <unit>:<frame_index>:<unit_position> (k2.Fsa: labels, aux_labels, unit_positions)
        """
        pass

    def get_composed_lattice(self, units_tensor: torch.Tensor, num_frames: int, vocab_size: int) -> "k2.Fsa":
        """
        Get composed lattice (unit and temporal schemas) for Compose-Transducer. Useful for visualization.
        Should be equivalent to the lattice from `get_grid` method.

        Args:
            units_tensor: tensor with target text
            num_frames: length of the sequence (in frames)
            vocab_size: vocab size (including blank)

        Returns:
            composed lattice (k2.Fsa) from unit and temporal schemas
        """
        fsa_text = self.get_unit_schema(units_tensor, vocab_size)
        fsa_temporal = self.get_temporal_schema(num_frames, vocab_size, units_tensor.device)
        composed = k2.compose(fsa_text, fsa_temporal, treat_epsilons_specially=False)
        if self.connect_composed:
            composed = k2.connect(composed)
        return composed

    def get_graphs_batched(
        self, source_lengths: torch.Tensor, targets: torch.Tensor, target_lengths: torch.Tensor, vocab_size: int
    ) -> "k2.Fsa":
        """
        Get batched lattice (grid or composed) for the batch of sequences.

        Args:
            source_lengths: tensor with lengths of logits
            targets: tensor with target units
            target_lengths: tensor with lengths of targets
            vocab_size: vocab size (including blank)

        Returns:
            batched lattice - FsaVec (k2.Fsa)
        """
        batch_size = source_lengths.shape[0]
        with torch.no_grad():
            if self.use_grid_implementation:
                source_lengths_list = source_lengths.tolist()
                target_lengths_list = target_lengths.tolist()
                return k2.create_fsa_vec(
                    [
                        self.get_grid(
                            units_tensor=targets[i, : target_lengths_list[i]],
                            num_frames=source_lengths_list[i],
                            vocab_size=vocab_size,
                        )
                        for i in range(batch_size)
                    ]
                )

            # composed version
            text_fsas = [
                self.get_unit_schema(
                    units_tensor=targets[i, : target_lengths[i].item()],
                    vocab_size=vocab_size,
                )
                for i in range(batch_size)
            ]
            temporal_fsas = [
                self.get_temporal_schema(
                    num_frames=source_lengths[i].item(), vocab_size=vocab_size, device=targets.device
                )
                for i in range(batch_size)
            ]
            target_fsas_vec = k2.compose(
                k2.create_fsa_vec(text_fsas), k2.create_fsa_vec(temporal_fsas), treat_epsilons_specially=False
            )
            if self.connect_composed:
                target_fsas_vec = k2.connect(target_fsas_vec)
        return target_fsas_vec

    @staticmethod
    def get_arc_sizes(target_fsas_vec: "k2.Fsa") -> torch.Tensor:
        batch_size = target_fsas_vec.shape[0]
        device = target_fsas_vec.device
        arc_sizes = torch.tensor(
            [target_fsas_vec.arcs.index(0, i)[0].values().shape[0] for i in range(batch_size)],
            device=device,
        )
        return arc_sizes

    @staticmethod
    def get_batch_indices(target_fsas_vec: "k2.Fsa") -> torch.Tensor:
        """
        Get batch indices (for logits) for each arc in the lattices.

        Args:
            target_fsas_vec: batch of target FSAs with lattices

        Returns:
            1d tensor with indices
        """
        batch_size = target_fsas_vec.shape[0]
        device = target_fsas_vec.device
        arc_sizes = GraphTransducerLossBase.get_arc_sizes(target_fsas_vec=target_fsas_vec)
        scores_to_batch_i = torch.repeat_interleave(
            torch.arange(batch_size, device=device, dtype=torch.int64),
            arc_sizes,
        )
        return scores_to_batch_i

    def get_logits_indices(self, target_fsas_vec: k2.Fsa, logits_shape: torch.Size) -> torch.Tensor:
        """
        Get indices of flatten logits for each arc in the lattices.

        Args:
            target_fsas_vec: batch of target FSAs with lattices
            logits_shape: shape of the logits tensor

        Returns:
            1d tensor with indices
        """
        # logits_shape: B x Time x Text+1 x Labels
        scores_to_batch_i = self.get_batch_indices(target_fsas_vec=target_fsas_vec)
        indices = (
            scores_to_batch_i * logits_shape[1] * logits_shape[2] * logits_shape[3]  # Batch
            + target_fsas_vec.aux_labels.to(torch.int64) * logits_shape[2] * logits_shape[3]  # Time indices
            + target_fsas_vec.unit_positions.to(torch.int64) * logits_shape[3]  # Units (text) indices
            + target_fsas_vec.labels.to(torch.int64)  # Labels
        )
        return indices


class GraphRnntLoss(GraphTransducerLossBase):
    """
    RNN-T loss implementation based on WFST according
    to "Powerful and Extensible WFST Framework for RNN-Transducer Losses"
    https://ieeexplore.ieee.org/document/10096679
    """

    def __init__(
        self,
        blank: int,
        use_grid_implementation=True,
        connect_composed=False,
        double_scores=False,
        cast_to_float32=False,
        return_graph=False,
        use_triton=True,
    ):
        """
        Init method

        Args:
            blank: blank label index
            use_grid_implementation: Whether to use the grid implementation (Grid-Transducer).
            connect_composed: Connect graph after composing unit and temporal schemas (only for Compose-Transducer).
                `connect` operation is slow, it is useful for visualization, but not necessary for loss computation.
            double_scores: Use calculation of loss in double precision (float64) in the lattice.
                Does not significantly affect memory usage since the lattice is ~V/2 times smaller
                than the joint tensor.
            cast_to_float32: Force cast joint tensor to float32 before log-softmax calculation.
            return_graph: Return graph (along with loss) from `forward` function
            use_triton: use optimized log probs calculations with Triton (faster and more memory efficient)
        """
        super().__init__(
            use_grid_implementation=use_grid_implementation,
            connect_composed=connect_composed,
            double_scores=double_scores,
            cast_to_float32=cast_to_float32,
        )
        self.blank = blank
        self.return_graph = return_graph
        self.use_triton = use_triton and TRITON_AVAILABLE
        if not self.use_triton:
            logging.warning("Triton is disabled, memory usage can be larger")

    def get_unit_schema(self, units_tensor: torch.Tensor, vocab_size: int) -> "k2.Fsa":
        """
        Get unit schema (target text) graph for RNN-T loss (Compose-Transducer).
        Forward arcs represent text labels.

        Example graph: text [1, 2], blank=0.

        graph::

                0:0:0                  0:0:1                  0:0:2
              +-------+              +-------+              +-------+
              v       |              v       |              v       |
            +-----------+  1:1:0   +-----------+  2:2:1   +-----------+  -1:-1:-1  #===#
            |     0     | -------> |     1     | -------> |     2     | ---------> H 3 H
            +-----------+          +-----------+          +-----------+            #===#

        Args:
            units_tensor: 1d tensor with text units
            vocab_size: number of total labels (vocab size including blank)

        Returns:
            unit schema graph (k2.Fsa).
            Labels: <unit>:<unit>:<unit_position> (k2.Fsa: labels, aux_labels, unit_positions)
        """

        blank_id = self.blank
        device = units_tensor.device
        text_len = units_tensor.shape[0]

        # arcs
        # text_len + 1 states, in every state - self-loops (blank) and forward (text label / last forward -1)
        arcs = torch.zeros(((text_len + 1) * 2, 4), dtype=torch.int32, device=device)
        text_indices = torch.arange(0, text_len + 1, dtype=torch.int32, device=device)
        # blank labels
        arcs[::2, 0] = text_indices  # from state
        arcs[::2, 1] = text_indices  # to state
        arcs[::2, 2] = blank_id

        # text labels
        arcs[1::2, 0] = text_indices  # from state
        arcs[1::2, 1] = text_indices + 1  # to state
        arcs[1:-1:2, 2] = units_tensor  # labels: text

        arcs[-1, 2] = -1  # last transition to final state, ilabel=-1 (special for k2)
        olabels = arcs[:, 2].detach().clone()  # same as ilabels

        fsa_text = k2.Fsa(arcs, olabels)
        fsa_text.unit_positions = text_indices.expand(2, -1).transpose(0, 1).flatten()
        fsa_text.unit_positions[-1] = -1  # last transition to final state
        return fsa_text

    def get_temporal_schema(self, num_frames: int, vocab_size: int, device: torch.device) -> "k2.Fsa":
        """
        Get temporal schema graph for RNN-T loss (Compose-Transducer).
        Forward arc - blank, self-loops - all labels excluding blank

        Example graph: blank=0, num_frames=3, vocab_size=3.
        Labels: <unit>:<frame_index>. <unit> is a unit from vocab.

        graph::

                1:0                1:1                1:2
              +-----+            +-----+            +-----+
              v     |            v     |            v     |
            +---------+  0:0   +---------+  0:1   +---------+  0:2   +---+  -1:-1   #===#
            |    0    | -----> |    1    | -----> |    2    | -----> | 3 | -------> H 4 H
            +---------+        +---------+        +---------+        +---+          #===#
              ^ 2:0 |            ^ 2:1 |            ^ 2:2 |
              +-----+            +-----+            +-----+

        Args:
            num_frames: length of the sequence (in frames)
            vocab_size: number of labels (including blank)
            device: device for tensor to construct

        Returns:
            temporal schema graph (k2.Fsa).
            Labels: <unit>:<frame_index>. <unit> is a unit from vocab.
        """
        blank_id = self.blank

        fsa_temporal_arcs = torch.zeros((num_frames * vocab_size + 1, 4), dtype=torch.int32, device=device)
        sequence_states = torch.arange(0, num_frames, dtype=torch.int32, device=device)
        # for every state - vocab_size arcs, [0, 1, ..., vocab_size-1, 0, 1, ..., vocab_size-1, ...]
        start_states = sequence_states.expand(vocab_size, num_frames).transpose(0, 1).flatten()
        # first: make all arcs - self-loops
        fsa_temporal_arcs[:-1, 0] = start_states  # from
        fsa_temporal_arcs[:-1, 1] = start_states  # to
        fsa_temporal_arcs[:-1, 2] = (
            torch.arange(0, vocab_size, dtype=torch.int32, device=device).expand(num_frames, vocab_size).flatten()
        )

        # blank-arcs: forward
        fsa_temporal_arcs[blank_id:-1:vocab_size, 1] = sequence_states + 1  # blanks

        # transition to last final state
        fsa_temporal_arcs[-1, :3] = torch.tensor((num_frames, num_frames + 1, -1), dtype=torch.int32, device=device)

        # output symbols: position in the sequence, same as start states for arcs
        olabels = fsa_temporal_arcs[:, 0].detach().clone()
        olabels[-1] = -1  # last arc to final state

        fsa_temporal = k2.Fsa(fsa_temporal_arcs, olabels)
        fsa_temporal = k2.arc_sort(fsa_temporal)  # need for compose
        return fsa_temporal

    @staticmethod
    def relabel_states(states: torch.Tensor, n: int, m: int) -> torch.Tensor:
        """
        Relabel states to be in topological order: by diagonals

        Args:
            states: tensor with states
            n: number of rows
            m: number of columns

        Returns:
            tensor with relabeled states (same shape as `states`)
        """
        i = states % n
        j = torch.div(states, n, rounding_mode='floor')  # states // n, torch.div to avoid pytorch warnings
        min_mn = min(m, n)
        max_mn = max(m, n)
        diag = i + j
        anti_diag = m + n - 1 - diag
        max_idx = n * m - 1
        cur_diag_idx = i if m > n else m - j - 1
        new_states = (
            diag.lt(min_mn) * ((diag * (diag + 1) >> 1) + i)
            + torch.logical_and(diag.ge(min_mn), diag.lt(max_mn))
            * ((min_mn * (min_mn + 1) >> 1) + (diag - min_mn) * min_mn + cur_diag_idx)
            + diag.ge(max_mn) * (max_idx - (anti_diag * (anti_diag + 1) >> 1) + m - j)
        )
        torch.where(states >= n * m, states, new_states, out=new_states)
        return new_states

    def get_grid(self, units_tensor: torch.Tensor, num_frames: int, vocab_size: int) -> "k2.Fsa":
        """
        Construct the RNN-T lattice directly (Grid-Transducer).

        Args:
            units_tensor: 1d tensor with text units
            num_frames: length of the sequence (number of frames)
            vocab_size: number of total labels (vocab size including blank)

        Returns:
            transducer lattice (k2.Fsa).
            Labels: <unit>:<frame_index>:<unit_position> (k2.Fsa: labels, aux_labels, unit_positions)
        """
        blank_id = self.blank
        text_length = units_tensor.shape[0]
        device = units_tensor.device
        num_grid_states = num_frames * (text_length + 1)
        num_forward_arcs = (num_frames - 1) * (text_length + 1)
        num_text_arcs = text_length * num_frames
        arcs = torch.zeros((num_forward_arcs + num_text_arcs + 2, 4), dtype=torch.int32, device=device)
        # blank transitions
        # i, i+<text_len + 1>, 0 <blank>, i / <text_len+1>, i % <text_len + 1>
        from_states = torch.arange(num_forward_arcs, device=device)
        to_states = from_states + (text_length + 1)
        arcs[:num_forward_arcs, 0] = from_states
        arcs[:num_forward_arcs, 1] = to_states
        arcs[:num_forward_arcs, 2] = blank_id

        # text arcs
        from_states = (
            torch.arange(num_grid_states, dtype=torch.int32, device=device)
            .reshape(num_frames, text_length + 1)[:, :-1]
            .flatten()
        )
        to_states = from_states + 1
        ilabels = units_tensor.expand(num_frames, -1).flatten()
        arcs[num_forward_arcs:-2, 0] = from_states
        arcs[num_forward_arcs:-2, 1] = to_states
        arcs[num_forward_arcs:-2, 2] = ilabels

        # last 2 states
        arcs[-2, :3] = torch.tensor((num_grid_states - 1, num_grid_states, blank_id), dtype=torch.int32, device=device)
        arcs[-1, :3] = torch.tensor((num_grid_states, num_grid_states + 1, -1), dtype=torch.int32, device=device)

        # sequence indices, time indices
        olabels = torch.div(arcs[:, 0], (text_length + 1), rounding_mode="floor")  # arcs[:, 0] // (text_length + 1)
        unit_positions = arcs[:, 0] % (text_length + 1)
        # last state: final
        olabels[-1] = -1
        unit_positions[-1] = -1

        # relabel
        # instead of using top sort (extremely expensive) k2.top_sort(rnnt_graph)
        arcs[:-2, 0] = self.relabel_states(arcs[:-2, 0], text_length + 1, num_frames)
        arcs[:-3, 1] = self.relabel_states(arcs[:-3, 1], text_length + 1, num_frames)

        # sort by start state - required in k2
        # TODO: maybe it is more optimal to avoid sort, construct arcs in ascending order
        _, indices = torch.sort(arcs[:, 0], dim=0)
        sorted_arcs = arcs[indices]
        olabels = olabels[indices]
        unit_positions = unit_positions[indices]

        rnnt_graph = k2.Fsa(sorted_arcs, olabels)
        rnnt_graph.unit_positions = unit_positions
        return rnnt_graph

    def get_weighted_graphs(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        source_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
        use_graph_weight=False,
    ) -> "k2.Fsa":
        """
        Get batch of graphs (FsaVec) for RNN-T loss calculation.

        Args:
            logits: activations (joint tensor). NB: raw logits, not after log-softmax
            targets: target labels
            source_lengths: lengths of source sequences
            target_lengths: length of target sequences
            use_graph_weight: uses weight from graphs (if `get_graphs_batched` returns graphs with weights)

        Returns:
            FsaVec containing RNN-T graphs for all utterances.
        """
        vocab_size = logits.shape[-1]
        target_fsas_vec = self.get_graphs_batched(source_lengths, targets, target_lengths, vocab_size)

        with torch.no_grad():
            # last transitions in the graph are labeled with -1 label
            last_transition_mask = target_fsas_vec.labels == -1
            batch_indices = self.get_batch_indices(target_fsas_vec=target_fsas_vec)
            time_indices = target_fsas_vec.aux_labels.clone().to(torch.int64)
            unit_indices = target_fsas_vec.unit_positions.clone().to(torch.int64)
            text_units = target_fsas_vec.labels.clone().to(torch.int64)
            # fill in the indices outside the logits with 0, replace later
            text_units.masked_fill_(last_transition_mask, 0)

        cast_context = force_float32_context() if self.cast_to_float32 else nullcontext()
        with cast_context:
            # NB: do not assign scores -> modify, k2 will not update all scores correctly (modify -> assign)
            if self.use_triton and logits.device.type == "cuda":
                unit_scores, blank_scores = rnnt_logprobs_triton(
                    logits=logits,
                    targets=targets,
                    blank_id=self.blank,
                    src_lengths=source_lengths,
                    tgt_lengths=target_lengths,
                )
                text_units_blank_mask = text_units == self.blank
                scores = torch.where(
                    text_units_blank_mask,
                    blank_scores[batch_indices, time_indices, unit_indices],
                    unit_scores[batch_indices, time_indices, unit_indices],
                ).to(torch.float32)
                scores[last_transition_mask] = 0.0  # fix weights for the arcs to the last state
            else:
                log_probs = F.log_softmax(logits, dim=-1)
                scores = log_probs[batch_indices, time_indices, unit_indices, text_units].to(torch.float32)
                scores[last_transition_mask] = 0.0

            if use_graph_weight:
                target_fsas_vec.scores = target_fsas_vec.scores + scores
            else:
                target_fsas_vec.scores = scores
        return target_fsas_vec

    @torch.inference_mode(mode=False)
    def align_graphs(
        self,
        target_fsas_vec: "k2.Fsa",
    ) -> torch.Tensor:
        # extract alignment
        shortest_paths = k2.shortest_path(target_fsas_vec, use_double_scores=self.double_scores)
        batch_size = shortest_paths.shape[0]
        # TODO: speedup, avoid iteration
        alignments = [
            shortest_paths[i].aux_labels[(shortest_paths[i].labels != self.blank) & (shortest_paths[i].labels != -1)]
            for i in range(batch_size)
        ]
        # assert [alignment.shape[0] for alignment in alignments] == target_lengths.tolist()
        padded_alignments = torch.nn.utils.rnn.pad_sequence(alignments, batch_first=True, padding_value=0)
        return padded_alignments

    @torch.inference_mode(mode=False)
    def align_from_logits(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        source_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Align from Joint logits"""
        target_fsas_vec = self.get_weighted_graphs(
            logits=logits,
            targets=targets,
            source_lengths=source_lengths,
            target_lengths=target_lengths,
            use_graph_weight=False,
        )
        return self.align_graphs(target_fsas_vec=target_fsas_vec)

    def get_arc_post_batch(self, target_fsas_vec: "k2.Fsa", pad_value: float = 0.0):
        arc_post = target_fsas_vec.get_arc_post(use_double_scores=self.double_scores, log_semiring=True)
        assert arc_post.dim() == 1
        device = arc_post.device
        arc_sizes = self.get_arc_sizes(target_fsas_vec=target_fsas_vec)
        max_arcs = int(arc_sizes.max().item())
        total_arcs = int(arc_sizes.sum().item())
        batch_size = target_fsas_vec.shape[0]
        arc_post_batch = arc_post.new_full([batch_size, max_arcs], fill_value=pad_value)
        row = torch.repeat_interleave(torch.arange(batch_size, device=device), arc_sizes)  # [total_arcs]
        starts = arc_sizes.cumsum(dim=0) - arc_sizes
        col = torch.arange(total_arcs, device=device) - torch.repeat_interleave(starts, arc_sizes)
        arc_post_batch[row, col] = arc_post
        return arc_post_batch, arc_sizes

    def consistency_loss(
        self,
        teacher_fsas_vec: "k2.Fsa",
        student_fsas_vec: "k2.Fsa",
        source_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
        symmetrical: bool = False,
        sanity_check: bool = False,
    ) -> torch.Tensor:
        device = teacher_fsas_vec.device
        arc_post_batch_teacher, arc_sizes = self.get_arc_post_batch(target_fsas_vec=teacher_fsas_vec)
        arc_post_batch_student, _ = self.get_arc_post_batch(target_fsas_vec=student_fsas_vec)
        arc_sizes_minus_2 = arc_sizes - 2  # ignore last 2 transitions - probability always 1
        batch_size = arc_sizes.shape[0]
        mask = (
            torch.arange(arc_post_batch_teacher.shape[1], dtype=torch.long, device=device)[None, :]
            < arc_sizes_minus_2[:, None]
        )
        kl_loss_full = F.kl_div(
            input=arc_post_batch_student,
            target=arc_post_batch_teacher.detach(),
            log_target=True,
            reduction="none",
        )
        if symmetrical:
            kl_loss_full = 0.5 * (
                kl_loss_full
                + F.kl_div(
                    input=arc_post_batch_teacher,
                    target=arc_post_batch_student.detach(),
                    log_target=True,
                    reduction="none",
                )
            )
        kl_loss_full *= mask
        norm_factor = (source_lengths + target_lengths - 1).clamp(min=1)
        kl_loss = kl_loss_full.sum(dim=-1) / norm_factor
        kl_loss_value = kl_loss.mean()
        if sanity_check:
            with torch.no_grad():
                assert kl_loss_value.item() >= 0.0
                float_dtype = arc_post_batch_teacher.dtype
                assert torch.allclose(
                    torch.logsumexp(torch.where(mask, arc_post_batch_teacher, float("-inf")), dim=-1).exp(),
                    norm_factor.to(float_dtype),
                )
                assert torch.allclose(
                    torch.logsumexp(torch.where(mask, arc_post_batch_student, float("-inf")), dim=-1).exp(),
                    norm_factor.to(float_dtype),
                )
        return kl_loss_value

    def forward(
        self,
        acts: torch.Tensor,
        labels: torch.Tensor,
        act_lens: torch.Tensor,
        label_lens: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, "k2.Fsa"]:
        """
        Compute forward method for RNN-T.

        Args:
            acts: activations (joint tensor). NB: raw logits, not after log-softmax
            labels: target labels
            act_lens: lengths of activations
            label_lens: length of labels sequences

        Returns:
            batch of RNN-T scores (loss)
        """
        # argument names are consistent with NeMo, see RNNTLoss.forward:
        # self._loss(acts=log_probs, labels=targets, act_lens=input_lengths, label_lens=target_lengths)
        target_fsas_vec = self.get_weighted_graphs(
            logits=acts, targets=labels, source_lengths=act_lens, target_lengths=label_lens, use_graph_weight=False
        )

        scores = -1 * target_fsas_vec.get_tot_scores(use_double_scores=self.double_scores, log_semiring=True)
        if self.return_graph:
            return scores, target_fsas_vec
        return scores
