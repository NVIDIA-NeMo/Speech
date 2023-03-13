# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

from abc import ABC
from typing import List, Optional, Tuple

import torch

from nemo.collections.asr.parts.k2.grad_utils import GradExpNormalize
from nemo.collections.asr.parts.k2.utils import (
    create_supervision,
    get_arc_weights,
    get_uniform_rnnt_prune_ranges,
    make_non_pad_mask,
    make_non_pad_mask_3d,
    prep_padded_densefsavec,
)
from nemo.core.utils.k2_guard import k2  # import k2 from guard module


class CtcK2Mixin(ABC):
    """k2 Mixin class that simplifies the construction of various k2-based CTC-like losses.
    
    It does the following:
        -   Prepares and adapts the input tensors (method _prepare_log_probs_and_targets).
        -   Creates Emissions graphs (method _prepare_emissions_graphs).
        -   Extracts the labels and probabilities of the best lattice path (method _extract_labels_and_probabilities).
    """

    def _prepare_log_probs_and_targets(
        self,
        log_probs: torch.Tensor,
        input_lengths: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        target_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Creates k2-style supervisions and shifts targets by one if the <blank> number is not zero.
        """
        assert log_probs.size(-1) == self.num_classes
        supervisions = create_supervision(input_lengths)
        # shift targets to make output epsilon ID zero
        return (
            log_probs,
            supervisions,
            torch.where(targets < self.blank, targets + 1, targets) if targets is not None else None,
            target_lengths,
        )

    def _prepare_emissions_graphs(self, log_probs: torch.Tensor, supervisions: torch.Tensor) -> 'k2.DenseFsaVec':
        """Creates DenseFsaVec, padding it with <epsilon> frames if the topology is `compact`.
        In particular, every second frame of the DenseFsaVec is the <epsilon> frame.

        <epsilon> frame is a frame with <epsilon> log-probability zero and every other log-probability is -inf.
        """
        return (
            prep_padded_densefsavec(log_probs, supervisions)
            if self.pad_fsavec
            else k2.DenseFsaVec(log_probs, supervisions)
        )

    def _maybe_normalize_gradients(self, log_probs: torch.Tensor, input_lengths: torch.Tensor) -> torch.Tensor:
        """PyTorch is doing the log-softmax normalization as part of the CTC computation.
        More: https://github.com/k2-fsa/k2/issues/575
        """
        return GradExpNormalize.apply(log_probs, input_lengths, "mean" if self.reduction != "sum" else "none")

    def _extract_labels_and_probabilities(
        self, shortest_path_fsas: 'k2.Fsa', return_ilabels: bool = False, output_aligned: bool = True
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Extracts the labels and probabilities of the best lattice path,
        dropping <epsilon> arcs and restoring the targets shift, if needed.
        """
        shortest_paths = []
        probs = []
        # direct iterating does not work as expected
        for i in range(shortest_path_fsas.shape[0]):
            shortest_path_fsa = shortest_path_fsas[i]
            # suppose that artificial input epsilon numbers >= self.num_classes
            non_eps_mask = (shortest_path_fsa.labels != -1) & (shortest_path_fsa.labels < self.num_classes)
            if return_ilabels:
                labels = shortest_path_fsa.labels[non_eps_mask]
            else:
                labels = shortest_path_fsa.aux_labels[non_eps_mask]
                if self.blank != 0:
                    # suppose output epsilon number == 0
                    # since the input epsilons were removed, we treat all remaining epsilons as blanks
                    labels[labels == 0] = self.blank
                    labels[(labels > 0) & (labels < self.blank)] -= 1
            labels = labels.to(dtype=torch.long)
            if not return_ilabels and not output_aligned:
                labels = labels[labels != self.blank]
            shortest_paths.append(labels)
            probs.append(get_arc_weights(shortest_path_fsa)[non_eps_mask].exp().to(device=shortest_path_fsas.device))
        return shortest_paths, probs


class RnntK2Mixin(CtcK2Mixin):
    """k2 Mixin class that simplifies the construction of various k2-based RNNT-like losses. Inherits CtcK2Mixin.
    
    It does the following:
        -   Prepares and adapts the input tensors.
        -   Creates Emissions graphs.
        -   Extracts the labels and probabilities of the best lattice path (method _extract_labels_and_probabilities).
    """

    def _prepare_log_probs_and_targets(
        self,
        log_probs: torch.Tensor,
        input_lengths: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Before calling super()._prepare_log_probs_and_targets, this method reshapes the log_probs tensor 
        from (B, T, U+1, D) to (B, T', D) where T' = T*(U+1), shifts paddings along T and U towards the end of T', 
        and recomputes input_lengths.

        It also calculates indices on which <epsilon> steps should be applied to the log_probs tensor to emulate 
        <blank> arcs shift of the Emissions graph for the pruned RNNT variant.
        """
        assert len(log_probs.size()) == 4  # B T U D
        B, T, U, D = log_probs.size()
        TU = T * U

        # save step indices if, as we assume, decoder output pruning has been applied
        if self.predictor_window_size > 0 and self.predictor_window_size < target_lengths.max():
            window_size_with_blank = self.predictor_window_size + 1
            ranges_begin = get_uniform_rnnt_prune_ranges(
                input_lengths, target_lengths, window_size_with_blank, self.predictor_step_size, T, True
            )
            step_sizes = ranges_begin[:, 1:] - ranges_begin[:, :-1]
            raw_step_indices = torch.where(step_sizes > 0)
            if self.predictor_step_size > 1:
                raw_step_indices = torch.repeat_interleave(
                    torch.stack(raw_step_indices).T, step_sizes[raw_step_indices], dim=0
                ).T
                raw_step_indices = (raw_step_indices[0], raw_step_indices[1])
            unique, count = torch.unique(raw_step_indices[0], return_counts=True)
            shift_mask = raw_step_indices[0].unsqueeze(0).repeat(len(unique), 1) == unique.unsqueeze(-1)
            step_indices = (
                raw_step_indices[0],
                (
                    torch.arange(ranges_begin.size(1)).unsqueeze(0).repeat(ranges_begin.size(0), 1)
                    * window_size_with_blank
                )[(raw_step_indices[0], raw_step_indices[1] + 1)]
                + torch.cumsum(shift_mask, 1)[shift_mask]
                - 1,
            )
            max_count = count.max()
            max_count_vec = torch.full((B,), max_count)
            max_count_vec[unique] -= count
            pad_indices_row = torch.repeat_interleave(torch.arange(B), max_count_vec)
            pad_unique = torch.unique(pad_indices_row)
            pad_shift_mask = pad_indices_row.unsqueeze(0).repeat(len(pad_unique), 1) == pad_unique.unsqueeze(-1)
            pad_indices = (
                pad_indices_row,
                T * window_size_with_blank + max_count - torch.cumsum(pad_shift_mask, 1)[pad_shift_mask],
            )
            self.__step_indices = (
                torch.cat((step_indices[0], pad_indices[0])),
                torch.cat((step_indices[1], pad_indices[1])),
            )
            self.__supervisions_add = max_count - max_count_vec
        else:
            self.__step_indices = None
            self.__supervisions_add = None

        # reshape 4D log_probs to 3D with respect to target_lengths
        non_pad_mask_true = make_non_pad_mask_3d(input_lengths, target_lengths + 1, T, U).flatten(1)
        input_lengths = non_pad_mask_true.sum(1)
        non_pad_mask_fake = make_non_pad_mask(input_lengths, TU).flatten()
        non_pad_mask_true = non_pad_mask_true.flatten()
        rearranged_indices = torch.arange(TU * B, device=log_probs.device)
        rearranged_indices_buffer = rearranged_indices.clone()
        rearranged_indices[non_pad_mask_fake] = rearranged_indices_buffer[non_pad_mask_true]
        rearranged_indices[~non_pad_mask_fake] = rearranged_indices_buffer[~non_pad_mask_true]
        log_probs = log_probs.reshape(-1, D)[rearranged_indices].view(B, -1, D)

        return super()._prepare_log_probs_and_targets(log_probs, input_lengths, targets, target_lengths)

    def _prepare_emissions_graphs(self, log_probs: torch.Tensor, supervisions: torch.Tensor) -> 'k2.DenseFsaVec':
        """Overrides super()._prepare_emissions_graphs.
        Creates DenseFsaVec, adding <epsilon> outputs to the end of the D dimension.

        If pruning is used, this method also pads the DenseFsaVec with <epsilon> frames 
        according to the <epsilon> steps, calculated before.

        <epsilon> frame is a frame with <epsilon> log-probability zero and every other log-probability is -inf.
        """
        if self.__step_indices is None or self.__supervisions_add is None:
            log_probs_eps = torch.cat(
                (log_probs, torch.zeros((log_probs.size(0), log_probs.size(1), 1), device=log_probs.device)), dim=2
            )
        else:
            mask = torch.zeros(
                (log_probs.size(0), log_probs.size(1) + int(len(self.__step_indices[0]) / log_probs.size(0))),
                dtype=torch.bool,
            )
            mask[self.__step_indices] = True
            log_probs_eps = torch.zeros((mask.size(0), mask.size(1), log_probs.size(2) + 1), device=log_probs.device)
            log_probs_eps[mask] = torch.tensor(
                [torch.finfo(torch.float32).min] * log_probs.size(2) + [0], device=log_probs.device
            )
            log_probs_eps[~mask] = torch.cat(
                (log_probs, torch.zeros((log_probs.size(0), log_probs.size(1), 1), device=log_probs.device)), dim=2
            ).view(-1, log_probs.size(-1) + 1)
            input_lengths = supervisions[:, -1] + self.__supervisions_add[supervisions[:, 0].to(dtype=torch.long)]
            if not torch.all(input_lengths[:-1] - input_lengths[1:] >= 0):
                # have to reorder supervisions inplace
                order = torch.argsort(input_lengths, descending=True)
                # the second column is assumed to be zero
                supervisions[:, 0] = supervisions[order, 0]
                supervisions[:, -1] = input_lengths[order]
            else:
                supervisions[:, -1] = input_lengths
            self.__step_indices = None
            self.__supervisions_add = None
        return k2.DenseFsaVec(log_probs_eps, supervisions)

    def _maybe_normalize_gradients(self, log_probs: torch.Tensor, input_lengths: torch.Tensor) -> torch.Tensor:
        """Not required for RNNT.
        """
        return log_probs
