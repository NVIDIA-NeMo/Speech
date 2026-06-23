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

from nemo.core.utils.optional_libs import GRAPHVIZ_AVAILABLE, graphviz_required

if GRAPHVIZ_AVAILABLE:
    import graphviz


@graphviz_required
def draw_linear(
    labels: list[str], title="", merges: list[tuple[int, int, str]] | None = None, case_insensitive: bool = False
):
    """
    Visualize linear graph from labels with optional merges

    Args:
        labels: list of labels (minimal units, usually characters)
        title: title of the graph
        merges: list of triples containing start state, end state, label (merged)
        case_insensitive: if the graph should show case-insensitive approach
    """
    graph_attr = {
        "rankdir": "LR",
        "size": "8.5,11",
        "center": "1",
        "orientation": "Portrait",
        "ranksep": "0.4",
        "nodesep": "0.25",
    }
    if title:
        graph_attr["label"] = title

    default_node_attr = {
        "shape": "circle",
        "style": "bold",
        "fontsize": "14",
    }

    final_state_attr = {
        "shape": "doublecircle",
        "style": "bold",
        "fontsize": "14",
    }

    dot = graphviz.Digraph(name="Context Graph", graph_attr=graph_attr)
    for i in range(len(labels) + 1):
        is_final = i == len(labels)
        if is_final:
            dot.node(str(i), label=str(i), **final_state_attr)
        else:
            dot.node(str(i), label=str(i), **default_node_attr)
        if i > 0:
            if case_insensitive:
                label = labels[i - 1].lower()
                dot.edge(str(i - 1), str(i), label=f"{label}")
                if label.upper() != label:
                    dot.edge(str(i - 1), str(i), label=f"{label.upper()}")
            else:
                label = labels[i - 1]
                dot.edge(str(i - 1), str(i), label=f"{label}")
    if merges:
        for b, e, label in merges:
            dot.edge(str(b), str(e), label=f"{label}")
    return dot
