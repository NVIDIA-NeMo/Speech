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


def force_float32_context() -> ContextManager:
    """Get context manager to force float32 precision in autocast mode."""
    if torch.is_autocast_enabled():
        return torch.cuda.amp.autocast(dtype=torch.float32)
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
        self, logits_lengths: torch.Tensor, targets: torch.Tensor, target_lengths: torch.Tensor, vocab_size: int
    ) -> "k2.Fsa":
        """
        Get batched lattice (grid or composed) for the batch of sequences.

        Args:
            logits_lengths: tensor with lengths of logits
            targets: tensor with target units
            target_lengths: tensor with lengths of targets
            vocab_size: vocab size (including blank)

        Returns:
            batched lattice - FsaVec (k2.Fsa)
        """
        batch_size = logits_lengths.shape[0]
        with torch.no_grad():
            if self.use_grid_implementation:
                return k2.create_fsa_vec(
                    [
                        self.get_grid(
                            units_tensor=targets[i, : target_lengths[i].item()],
                            num_frames=logits_lengths[i].item(),
                            vocab_size=vocab_size,
                        )
                        for i in range(batch_size)
                    ]
                )

            # composed version
            text_fsas = [
                self.get_unit_schema(units_tensor=targets[i, : target_lengths[i].item()], vocab_size=vocab_size,)
                for i in range(batch_size)
            ]
            temporal_fsas = [
                self.get_temporal_schema(
                    num_frames=logits_lengths[i].item(), vocab_size=vocab_size, device=targets.device
                )
                for i in range(batch_size)
            ]
            target_fsas_vec = k2.compose(
                k2.create_fsa_vec(text_fsas), k2.create_fsa_vec(temporal_fsas), treat_epsilons_specially=False
            )
            if self.connect_composed:
                k2.connect(target_fsas_vec)
        return target_fsas_vec

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
        batch_size = logits_shape[0]
        device = target_fsas_vec.device
        scores_to_batch_i = torch.repeat_interleave(
            torch.arange(batch_size, device=device, dtype=torch.int64),
            torch.tensor(
                [target_fsas_vec.arcs.index(0, i)[0].values().shape[0] for i in range(batch_size)], device=device,
            ),
        )
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
    ):
        """
        Init method

        Args:
            blank: blank label index
            use_grid_implementation: Whether to use the grid implementation (Grid-Transducer).
            connect_composed: Connect graph after composing unit and temporal schemas (only for Compose-Transducer).
                `connect` operation is slow, it is useful for visualization, but not necessary for loss computation.
            double_scores: Use calculation of loss in double precision (float64) in the lattice.
                Does not significantly affect memory usage since the lattice is ~V/2 times smaller than the joint tensor.
            cast_to_float32: Force cast joint tensor to float32 before log-softmax calculation.
        """
        super().__init__(
            use_grid_implementation=use_grid_implementation,
            connect_composed=connect_composed,
            double_scores=double_scores,
            cast_to_float32=cast_to_float32,
        )
        self.blank = blank

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
        states = (
            diag.lt(min_mn) * ((diag * (diag + 1) >> 1) + i)
            + torch.logical_and(diag.ge(min_mn), diag.lt(max_mn))
            * ((min_mn * (min_mn + 1) >> 1) + (diag - min_mn) * min_mn + cur_diag_idx)
            + diag.ge(max_mn) * (max_idx - (anti_diag * (anti_diag + 1) >> 1) + m - j)
        )
        return states

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

    def forward(
        self, acts: torch.Tensor, labels: torch.Tensor, act_lens: torch.Tensor, label_lens: torch.Tensor,
    ) -> torch.Tensor:
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
        logits, targets, logits_lengths, target_lengths = acts, labels, act_lens, label_lens

        # logits: B x Time x Text+1 x C
        vocab_size = logits.shape[-1]
        target_fsas_vec = self.get_graphs_batched(logits_lengths, targets, target_lengths, vocab_size)

        cast_context = force_float32_context() if self.cast_to_float32 else nullcontext()
        with cast_context:
            log_probs = F.log_softmax(logits, dim=-1)
            with torch.no_grad():
                indices = self.get_logits_indices(target_fsas_vec, logits.shape)
                # transition to the last state
                # use 0 index (for valid index_select) and manually assign score after index_select for this case
                indices[target_fsas_vec.labels == -1] = 0

            # NB: do not assign scores -> modify, k2 will not update all scores correctly (modify -> assign)
            scores = log_probs.flatten().index_select(-1, indices)
            # fix weights for the arcs to the last state
            scores[target_fsas_vec.labels == -1] = 0

            target_fsas_vec.scores = scores
            scores = -1 * target_fsas_vec.get_tot_scores(use_double_scores=self.double_scores, log_semiring=True)
            return scores
