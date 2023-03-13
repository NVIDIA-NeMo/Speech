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

from functools import lru_cache
from typing import List, Optional, Union

import torch

from nemo.core.utils.k2_guard import k2  # import k2 from guard module


def build_topo(name: str, tokens: List[int], blank_num: int, with_self_loops: bool = True) -> 'k2.Fsa':
    """Helper function to build a topology.
    It allows to build topologies with a non-zero blank ID.
    Args:
      name:
        The topology name. Choices: default, compact, shared_blank, minimal
      tokens:
        A list of tokens, e.g., phones, characters, etc.
      blank_num:
        Blank number. Must be in tokens
      with_self_loops:
        Whether to add token-to-epsilon self-loops to a topology
    Returns:
      Returns a topology FST.
    """
    if name == "default":
        ans = build_default_topo(tokens, with_self_loops)
    elif name == "compact":
        ans = build_compact_topo(tokens, with_self_loops)
    elif name == "shared_blank":
        ans = build_shared_blank_topo(tokens, with_self_loops)
    elif name == "minimal":
        ans = build_minimal_topo(tokens)
    else:
        raise ValueError(f"Unknown topo name: {name}")
    if blank_num != 0:
        blank_mask = ans.labels == 0
        ans.labels[(ans.labels != -1) & (ans.labels <= blank_num)] -= 1
        ans.labels[blank_mask] = blank_num
    ans = k2.arc_sort(ans)
    return ans


def build_default_topo(tokens: List[int], with_self_loops: bool = True) -> 'k2.Fsa':
    """Build the default CTC topology.
    Zero is assumed to be the ID of the blank symbol.
    """
    assert -1 not in tokens, "We assume -1 is ID of the final transition"
    assert 0 in tokens, "We assume 0 is the ID of the blank symbol"

    num_states = len(tokens)
    final_state = num_states
    arcs = "" if with_self_loops else f"0 0 0 0 0.0\n"
    for i in range(num_states):
        for j in range(num_states):
            if i == j:
                if with_self_loops:
                    arcs += f"{i} {i} {tokens[i]} 0 0.0\n"
            else:
                arcs += f"{i} {j} {tokens[j]} {tokens[j]} 0.0\n"
        arcs += f"{i} {final_state} -1 -1 0.0\n"
    arcs += f"{final_state}"
    ans = k2.Fsa.from_str(arcs, num_aux_labels=1)
    ans = k2.arc_sort(ans)
    return ans


def build_compact_topo(tokens: List[int], with_self_loops: bool = True) -> 'k2.Fsa':
    """Build the compact CTC topology.
    Zero is assumed to be the ID of the blank symbol.
    See https://arxiv.org/abs/2110.03098
    """
    assert -1 not in tokens, "We assume -1 is ID of the final transition"
    assert 0 in tokens, "We assume 0 is the ID of the blank symbol"

    eps_num = tokens[-1] + 1
    selfloops_shift = int(with_self_loops)
    num_states = len(tokens) + selfloops_shift
    final_state = num_states
    arcs = ""
    for i in range(selfloops_shift, num_states):
        arcs += f"0 {i} {tokens[i - selfloops_shift]} {tokens[i - selfloops_shift]} 0.0\n"
    arcs += f"0 {final_state} -1 -1 0.0\n"
    for i in range(1, num_states):
        arcs += f"{i} 0 {eps_num} 0 0.0\n"
        if with_self_loops:
            arcs += f"{i} {i} {tokens[i - selfloops_shift]} 0 0.0\n"
    arcs += f"{final_state}"
    ans = k2.Fsa.from_str(arcs, num_aux_labels=1)
    ans = k2.arc_sort(ans)
    return ans


def build_shared_blank_topo(tokens: List[int], with_self_loops: bool = True) -> 'k2.Fsa':
    """Build the shared blank CTC topology.
    Zero is assumed to be the ID of the blank symbol.
    See https://github.com/k2-fsa/k2/issues/746#issuecomment-856421616
    """
    assert -1 not in tokens, "We assume -1 is ID of the final transition"
    assert 0 in tokens, "We assume 0 is the ID of the blank symbol"

    tokens = tokens.copy()
    tokens.remove(0)
    num_tokens = len(tokens)
    start = 0
    final = num_tokens + 1
    arcs = []
    arcs.append([start, start, 0, 0, 0])
    arcs.append([start, final, -1, -1, 0])
    arcs.append([final])
    for i, p in enumerate(tokens):
        i += 1
        arcs.append([start, start, p, p, 0])
        arcs.append([start, i, p, p, 0])
        arcs.append([i, start, p, 0, 0])
        if with_self_loops:
            arcs.append([i, i, p, 0, 0])
    arcs = sorted(arcs, key=lambda arc: arc[0])
    arcs = [[str(i) for i in arc] for arc in arcs]
    arcs = [" ".join(arc) for arc in arcs]
    arcs = "\n".join(arcs)
    ans = k2.Fsa.from_str(arcs, num_aux_labels=1)
    ans = k2.arc_sort(ans)
    return ans


def build_minimal_topo(tokens: List[int]) -> 'k2.Fsa':
    """Build the minimal topology.
    Zero is assumed to be the ID of the blank symbol.
    See https://arxiv.org/abs/2110.03098
    """
    assert -1 not in tokens, "We assume -1 is ID of the final transition"
    assert 0 in tokens, "We assume 0 is the ID of the blank symbol"

    num_tokens = len(tokens)
    final_state = 1
    arcs = ""
    for i in range(num_tokens):
        arcs += f"0 0 {tokens[i]} {tokens[i]} 0.0\n"
    arcs += f"0 {final_state} -1 -1 0.0\n"
    arcs += f"{final_state}"
    ans = k2.Fsa.from_str(arcs, num_aux_labels=1)
    ans = k2.arc_sort(ans)
    return ans


class RnntEmissionAdapterBuilder(object):
    """Builder class for RNNT Emission Adapters.

    An Emission Adapter is an FSA used to emulate desired temporal Emissions FSA properties of a trivial Emissions FSA.
    Temporal properties are emulated by <epsilon>-arcs with zero log-weight.
    These additional arcs do not contribute to the lattice scores and can be easily removed from the best path.

    k2 does not have Emissions FSAs. Instead, it has DenseFsaVec, which is not a real FSA.
    Thus, Emission Adapters should be composed with Supervision FSAs.
    IMPOTRANT: <epsilon>-outputs are expected to be present in the DenseFsaVec.

    These RNNT adapters do only the <blank> re-routing (emulate <blank> hopping over U dimension).
    Redundant non-<blank> are not removed by these adapters.

    At initialization, the builder expects a list of tokens, <blank> number and <epsilon> number.
    When called, the builder returns adapters according to the provided text lengths.
    """

    def __init__(self, tokens: List[int], blank_num: int, eps_num: Optional[int] = None):
        assert -1 not in tokens, "We assume -1 is ID of the final transition"
        assert blank_num in tokens, "The blank ID must be in tokens"
        assert eps_num is None or eps_num not in tokens, "The epsion ID must not be in tokens"

        self.tokens = tokens
        self.blank_num = blank_num
        self.eps_num = self.tokens[-1] + 1 if eps_num is None else eps_num

    def __call__(self, adapter_lengths: Union[torch.Tensor, List[int]]) -> 'k2.Fsa':
        # if you don't make adapter_lengths a list beforehand,
        # "i" will be implicitly converted to int, and this will always be considered a cache miss
        return k2.create_fsa_vec([self._build_single_adapter(i) for i in adapter_lengths.tolist()])

    @lru_cache(maxsize=1024)
    def _build_single_adapter(self, adapter_length: int) -> 'k2.Fsa':
        assert adapter_length >= 1, "`adapter_length` cannot be less than one"

        first_eps_state = adapter_length + 1
        final_state = adapter_length * 2 + 1
        arcs = ""
        for i in range(adapter_length):
            for j in range(len(self.tokens)):
                if j != self.blank_num:
                    arcs += f"{i} {i + 1} {self.tokens[j]} 0.0\n"
            arcs += f"{i} {first_eps_state} {self.blank_num} 0.0\n"
        arcs += f"{adapter_length} {first_eps_state} {self.blank_num} 0.0\n"
        for i in range(first_eps_state, final_state):
            arcs += f"{i} {i + 1 if i < final_state - 1 else 0} {self.eps_num} 0.0\n"
            arcs += f"{i} {final_state} -1 0.0\n"
        arcs += f"{final_state}"

        return k2.arc_sort(k2.Fsa.from_str(arcs, acceptor=True))
