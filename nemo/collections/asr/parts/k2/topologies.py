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

from typing import List

from nemo.core.utils.k2_guard import k2  # import k2 from guard module


def build_topo(name: str, tokens: List[int], with_self_loops: bool = True) -> 'k2.Fsa':
    """Helper function to build a topology.
    Args:
      name:
        The topology name. Choices: default, compact, shared_blank, minimal
      tokens:
        A list of tokens, e.g., phones, characters, etc.
      with_self_loops:
        Whether to add token-to-epsilon self-loops to a topology
    Returns:
      Returns a topology FST.
    """
    if name == "default":
        return build_default_topo(tokens, with_self_loops)
    elif name == "compact":
        return build_compact_topo(tokens, with_self_loops)
    elif name == "shared_blank":
        return build_shared_blank_topo(tokens, with_self_loops)
    elif name == "minimal":
        return build_minimal_topo(tokens)
    else:
        raise ValueError(f"Unknown topo name: {name}")


def build_default_topo(tokens: List[int], with_self_loops: bool = True) -> 'k2.Fsa':
    """Build the default CTC topology.
    """
    assert 0 in tokens, "We assume 0 is ID of the blank symbol"

    num_states = len(tokens)
    final_state = num_states
    arcs = "" if with_self_loops else "0 0 0 0 0.0\n"
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
    See https://arxiv.org/abs/2110.03098
    """
    assert 0 in tokens, "We assume 0 is ID of the blank symbol"

    selfloops_shift = int(with_self_loops)
    blank_num = 1
    num_states = len(tokens) + selfloops_shift
    final_state = num_states
    arcs = f"0 {selfloops_shift} {blank_num} 0 0.0\n"
    for i in range(blank_num + selfloops_shift, num_states):
        arcs += f"0 {i} {tokens[i - selfloops_shift] + 1} {tokens[i - selfloops_shift] + 1} 0.0\n"
    arcs += f"0 {final_state} -1 -1 0.0\n"
    for i in range(blank_num, num_states):
        arcs += f"{i} 0 0 0 0.0\n"
        if with_self_loops:
            arcs += f"{i} {i} {tokens[i - selfloops_shift] + 1} 0 0.0\n"
    arcs += f"{final_state}"
    ans = k2.Fsa.from_str(arcs, num_aux_labels=1)
    ans = k2.arc_sort(ans)
    return ans


def build_shared_blank_topo(tokens: List[int], with_self_loops: bool = True) -> 'k2.Fsa':
    """Build the shared blank CTC topology.
    See https://github.com/k2-fsa/k2/issues/746#issuecomment-856421616
    """
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
    See https://arxiv.org/abs/2110.03098
    """
    assert 0 in tokens, "We assume 0 is ID of the blank symbol"

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
