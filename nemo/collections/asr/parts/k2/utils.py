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

# Copyright (c) 2020, Xiaomi CORPORATION.  All rights reserved.
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

import os
import struct
from pickle import UnpicklingError
from typing import List, Optional, Tuple, Union

import torch

from nemo.core.utils.k2_guard import k2  # import k2 from guard module
from nemo.utils import logging


def create_supervision(input_lengths: torch.Tensor) -> torch.Tensor:
    """Creates a special supervisions tensor from input lengths.
    These supervisions are required for some k2 methods.
    """
    supervisions = torch.stack(
        (torch.tensor(range(input_lengths.shape[0])), torch.zeros(input_lengths.shape[0]), input_lengths.cpu(),), 1,
    ).to(dtype=torch.int32)
    # the duration column has to be sorted in decreasing order
    return supervisions[torch.argsort(supervisions[:, -1], descending=True)]


def invert_permutation(indices: torch.Tensor) -> torch.Tensor:
    """Produces a tensor of reverse permutation for a given indices.
    
    Based on https://github.com/k2-fsa/snowfall/blob/master/snowfall/common.py
    """
    ans = torch.zeros(indices.shape, device=indices.device, dtype=indices.dtype)
    ans[indices.to(dtype=torch.long)] = torch.arange(0, indices.shape[0], device=indices.device, dtype=indices.dtype)
    return ans


def make_non_pad_mask(input_lengths: torch.Tensor, seq_len: int):
    """Converts input_lengths to a non-padding mask. The mask is 2D.
    """
    batch_size = input_lengths.shape[0]
    seq_range = torch.arange(0, seq_len, device=input_lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, seq_len)
    seq_length_expand = input_lengths.clone().detach().to(seq_range_expand.device).unsqueeze(-1)
    mask = seq_range_expand < seq_length_expand
    return mask


def make_non_pad_mask_3d(
    lengths_x: torch.Tensor, lengths_y: torch.Tensor, max_length_x: int, max_length_y: int
) -> torch.Tensor:
    """Converts two orthogonal input_lengths to a non-padding mask. The mask is 3D.
    """
    assert lengths_x.size() == lengths_y.size()
    return make_non_pad_mask(lengths_x, max_length_x).unsqueeze(2) & make_non_pad_mask(
        lengths_y, max_length_y
    ).unsqueeze(1)


def ragged_to_tensor_2axes_simple(rt: k2.RaggedTensor) -> Optional[torch.Tensor]:
    """Converts k2.RaggedTensor to torch.Tensor if the RaggedTensor is shallow (has two axes).
    """
    rt_list = rt.tolist()
    result_list = []
    for e in rt_list:
        if len(e) == 0:
            result_list.append(0)
        elif len(e) == 1:
            result_list.append(e[0])
        else:
            return None
    return torch.tensor(result_list, dtype=torch.int32)


def load_graph(graph_path: str) -> 'k2.Fsa':
    """Fsa graph loading helper function. Loads graphs stored in different formats.
    """
    if os.path.exists(graph_path):
        errors = []
        try:
            graph_dict = torch.load(graph_path, map_location="cpu")
            graph = k2.Fsa.from_dict(graph_dict)
            return graph
        except UnpicklingError as e:
            errors.append(e)
            with open(graph_path, "rt", encoding="utf-8") as f:
                graph_txt = f.read()
            # order from the most frequent case to the least
            for func, acceptor in [(k2.Fsa.from_openfst, False), (k2.Fsa.from_str, True), (k2.Fsa.from_str, False)]:
                try:
                    graph = func(graph_txt, acceptor=acceptor)
                    return graph
                except (TypeError, ValueError, RuntimeError) as e:
                    errors.append(e)
        raise Exception(errors)
    else:
        logging.warning(f"""No such file: '{graph_path}'""")
        return None


def intersect_with_self_loops(base_graph: 'k2.Fsa', aux_graph: 'k2.Fsa') -> 'k2.Fsa':
    """Intersection helper function.
    """
    assert hasattr(base_graph, "aux_labels")
    assert not hasattr(aux_graph, "aux_labels")
    aux_graph_with_self_loops = k2.arc_sort(k2.add_epsilon_self_loops(aux_graph)).to(base_graph.device)
    result = k2.intersect(k2.arc_sort(base_graph), aux_graph_with_self_loops, treat_epsilons_specially=False)
    setattr(result, "phones", result.labels)
    return result


def compose_with_self_loops(base_graph: 'k2.Fsa', aux_graph: 'k2.Fsa') -> 'k2.Fsa':
    """Composition helper function.
    """
    aux_graph_with_self_loops = k2.arc_sort(k2.add_epsilon_self_loops(aux_graph)).to(base_graph.device)
    return k2.compose(base_graph, aux_graph_with_self_loops, treat_epsilons_specially=False, inner_labels="phones")


def create_sparse_wrapped(
    indices: List[torch.Tensor],
    values: torch.Tensor,
    size: Optional[Union[Tuple[int, int], Tuple[int, int, int]]] = None,
    min_col_index: Optional[int] = None,
) -> torch.Tensor:
    """Wraps up k2.create_sparse to create 2- or 3-dimensional sparse tensors.
    """
    assert size is None or len(indices) == len(size)

    if len(indices) == 2:
        return k2.create_sparse(
            rows=indices[0], cols=indices[1], values=values, size=size, min_col_index=min_col_index,
        )
    elif len(indices) == 3:
        assert indices[0].ndim == indices[1].ndim == indices[2].ndim == 1
        assert indices[0].numel() == indices[1].numel() == indices[2].numel() == values.numel()

        if min_col_index is not None:
            assert isinstance(min_col_index, int)
            kept_indices = indices[-1] >= min_col_index
            indices = [i[kept_indices] for i in indices]
            values = values[kept_indices]
        if size is not None:
            return torch.sparse_coo_tensor(
                torch.stack(indices), values, size=size, device=values.device, requires_grad=values.requires_grad,
            )
        else:
            return torch.sparse_coo_tensor(
                torch.stack(indices), values, device=values.device, requires_grad=values.requires_grad,
            )
    else:
        raise ValueError(f"len(indices) = {len(indices)}")


def prep_padded_densefsavec(log_softmax: torch.Tensor, supervisions: torch.Tensor) -> 'k2.DenseFsaVec':
    """Performs special epsilon-padding required for composition with some of the topologies.
    """
    log_softmax_eps = torch.cat(
        [
            log_softmax,
            torch.full((log_softmax.shape[0], log_softmax.shape[1], 1), -float("inf"), device=log_softmax.device,),
        ],
        axis=-1,
    )
    log_softmax_padded = torch.zeros(
        (log_softmax_eps.shape[0], log_softmax_eps.shape[1] * 2, log_softmax_eps.shape[2],), device=log_softmax.device,
    )
    log_softmax_padded[:, ::2] = log_softmax_eps
    supervisions_padded = supervisions.clone()
    supervisions_padded[:, 2] *= 2
    dense_log_softmax_padded = k2.DenseFsaVec(log_softmax_padded, supervisions_padded)
    return dense_log_softmax_padded


def shift_labels_inpl(lattices: List['k2.Fsa'], shift: int):
    """Shifts lattice labels and aux_labels by a given number.
    This is an in-place operation, if the lattice is on GPU.
    """
    for lattice in lattices:
        mask = lattice.labels > 0
        lattice.labels[mask] += shift
        if hasattr(lattice, "aux_labels"):
            mask = lattice.aux_labels > 0
            lattice.aux_labels[mask] += shift
    return reset_properties_fsa(lattices)


def reset_properties_fsa(graph: 'k2.Fsa'):
    """Resets properties of a graph.
    In-place (does not create a new graph) if the graph is on GPU.
    Use this every time you alter a graph in-place.
    See https://github.com/k2-fsa/k2/issues/978 for more information."""
    graph.__dict__["_properties"] = None
    # CPU graphs need to be sorted e.g. for intersection
    if graph.device == torch.device("cpu"):
        graph = k2.arc_sort(graph)
    return graph


def add_self_loops(graph: 'k2.Fsa', label: int = 0, mode: str = "auto"):
    """Adds self-loops with given label to a graph.
    Supported modes are ``input``, ``output``, and ``auto``,
    Where ``input`` leaves aux_labels zeroes, if present, ``output`` leaves labels zeroes"""
    assert mode in ("input", "output", "auto"), "Supported modes are ``input``, ``output``, and ``auto``: {mode}"
    assert mode != "output" or hasattr(graph, "aux_labels"), "Graph must have aux_labels for mode ``output``"
    new_graph, arc_map = k2.add_epsilon_self_loops(graph, ret_arc_map=True)

    if mode != "output":
        new_graph.labels[arc_map == -1] = label
    if mode != "input" and hasattr(graph, "aux_labels"):
        new_graph.aux_labels[arc_map == -1] = label
    return reset_properties_fsa(new_graph)


def get_arc_weights(graph: 'k2.Fsa') -> torch.Tensor:
    """Returns 1d torch.Tensor with arc weights of a given graph.
    """
    if len(graph.shape) > 2:
        raise NotImplementedError("FsaVec is not supported at the moment.")
    weights_int = graph.arcs.values()[:, -1].tolist()
    weights_float = struct.unpack('%sf' % len(weights_int), struct.pack('%si' % len(weights_int), *weights_int))
    return torch.Tensor(weights_float)


def get_tot_objf_and_finite_mask(tot_scores: torch.Tensor, reduction: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Figures out the total score(log-prob) over all successful supervision segments
    (i.e. those for which the total score wasn't -infinity).
        Args:
            tot_scores: a Torch tensor of shape (num_segments,) containing total scores
                       from forward-backward
            reduction: a reduction type ('mean', 'sum' or 'none')
        Returns:
             Returns a tuple of 2 scalar tensors: (tot_score, finite_mask)
        where finite_mask is a tensor containing successful segment mask.
    
    Based on get_tot_objf_and_num_frames
    from https://github.com/k2-fsa/snowfall/blob/master/snowfall/objectives/common.py
    """
    finite_mask = ~torch.isnan(tot_scores) & torch.ne(tot_scores, -float("inf"))
    if reduction == "mean":
        tot_scores = tot_scores[finite_mask].mean()
    elif reduction == "sum":
        tot_scores = tot_scores[finite_mask].sum()
    return tot_scores, finite_mask


def get_uniform_rnnt_prune_ranges(
    encoded_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    window_size_with_blank: int,
    step: int = 1,
    max_seq_len: Optional[int] = None,
    begin_only: bool = False,
) -> torch.Tensor:
    """Creates the pruning ranges for the Encoder and Predictor of RNNT.
    The ranges are similar to https://k2-fsa.github.io/k2/python_api/api.html#k2.get_rnnt_prune_ranges
    but they are constructed under the assumption of the uniform distribution token activations across time frames
    and without any posterior knowledge.
    """
    assert window_size_with_blank > 1
    assert step >= 1
    assert window_size_with_blank > step
    assert len(encoded_lengths) == len(target_lengths)
    ranges_begin = torch.zeros(
        (
            len(encoded_lengths),
            encoded_lengths.max() if max_seq_len is None else max(max_seq_len, encoded_lengths.max()),
        ),
        dtype=torch.long,
    )
    for i in (target_lengths >= window_size_with_blank).nonzero(as_tuple=True)[0]:
        encoded_len = encoded_lengths[i]
        ranges_begin_raw = torch.arange(int((target_lengths[i] - window_size_with_blank) / step + 2)) * step
        ranges_begin_raw[-1] = target_lengths[i] - window_size_with_blank + 1
        ranges_begin[i, :encoded_len] = torch.nn.functional.interpolate(
            ranges_begin_raw.reshape(1, 1, -1).to(dtype=torch.float), encoded_len, mode="nearest-exact"
        ).to(dtype=torch.long)
        ranges_begin[i, encoded_len:] = ranges_begin[i, encoded_len - 1]
    return (
        ranges_begin
        if begin_only
        else ranges_begin.unsqueeze(-1).repeat(1, 1, window_size_with_blank) + torch.arange(window_size_with_blank)
    )


def apply_rnnt_prune_ranges(
    encoder_outputs: torch.Tensor, decoder_outputs: torch.Tensor, ranges: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prepares pruned encoder and decoder outputs according to the prune ranges.
    Based on k2.do_rnnt_pruning(...)
    """
    B, T, window_size_with_blank = ranges.size()
    D1 = encoder_outputs.size(-1)
    _, U, D2 = decoder_outputs.size()
    assert B == encoder_outputs.size(0)
    assert T == encoder_outputs.size(1)
    assert B == decoder_outputs.size(0)
    encoder_outputs_pruned = encoder_outputs.unsqueeze(2).expand((B, T, window_size_with_blank, D1))
    decoder_outputs_pruned = torch.gather(
        decoder_outputs.unsqueeze(1).expand((B, T, U, D2)),
        dim=2,
        index=ranges.reshape((B, T, window_size_with_blank, 1)).expand((B, T, window_size_with_blank, D2)),
    )
    return encoder_outputs_pruned, decoder_outputs_pruned
