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

# Copyright    2023  Xiaomi Corp.        (authors: Wei Kang)
#
# See ../LICENSE for clarification regarding multiple authors
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

# The script was obtained and modified from Icefall repo:
# https://github.com/k2-fsa/icefall/blob/11d816d174076ec9485ab8b1d36af2592514e348/icefall/context_graph.py

from collections import deque
from typing import Dict, List, Optional


try:
    import graphviz

    _GRAPHVIZ_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    _GRAPHVIZ_AVAILABLE = False


class ContextState:
    """The state in ContextGraph"""

    def __init__(
        self,
        index: int,
        is_end: bool = False,
        word: Optional[str] = None,
    ):
        """Create a ContextState.
        Args:
          index:
            The node index, only for visualization now. A node is in [0, graph.num_nodes).
            The index of the root node is always 0.
          is_end:
            True if current node is the end of a context biasing word.
          word:
            The word of coresponding transcription (not None only for end states).
        """
        self.index = index
        self.is_end = is_end
        self.word = word
        # dict of next token transitions to next states (key: token, value: next state)
        self.next = {}
        # the best token on current state (needed for state pruning during word spotter work)
        self.best_token = None


class ContextGraphCTC:
    """
    Context-biasing graph (based on prefix tree) according to the CTC transition topology (with blank nodes).
    A ContextGraph contains some words / phrases that we expect to boost their recognition accuracy.
    """

    def __init__(self, blank_id: int = 1024):
        """
        Initialize the ContextGraphCTC based on given blank_id.

        Args:
            blank_id: the id of blank token in ASR model
        """

        self.num_nodes = 0
        self.root = ContextState(index=self.num_nodes, is_end=False)
        self.blank_token = blank_id

    def add_to_graph(self, word_items: List[tuple[str, List[List[int]]]]):
        """
        Adding nodes to the context graph based on given word_items.

        Args:
            word_items: a list of word items, each word item is a tuple of (word, tokenizations)
                        word: the word to be inserted into the context graph
                        tokenizations: a list of possible BPE word tokenizations
                        (each word can have several tokenizations to improve the recognition accuracy)

        """
        # process context biasing words with tokenizations
        for word_item in word_items:
            for tokens in word_item[1]:
                prev_node = self.root
                prev_token = None
                for i, token in enumerate(tokens):
                    if token not in prev_node.next:
                        self.num_nodes += 1
                        is_end = i == len(tokens) - 1
                        word = word_item[0] if is_end else None
                        node = ContextState(index=self.num_nodes, is_end=is_end, word=word)
                        node.next[token] = node
                        prev_node.next[token] = node

                        # add blank node:
                        if prev_node is not self.root:
                            if self.blank_token in prev_node.next:
                                # blank node already exists
                                prev_node.next[self.blank_token].next[token] = node
                            else:
                                # create new blank node
                                self.num_nodes += 1
                                blank_node = ContextState(index=self.num_nodes, is_end=False)
                                blank_node.next[self.blank_token] = blank_node
                                blank_node.next[token] = node
                                prev_node.next[self.blank_token] = blank_node

                    # in case of two consecutive equal tokens
                    if token == prev_token:
                        # if token already in prev_node.next[balnk_token].next
                        if self.blank_token in prev_node.next and token in prev_node.next[self.blank_token].next:
                            prev_node = prev_node.next[self.blank_token].next[token]
                            prev_token = token
                            continue
                        # create new token
                        self.num_nodes += 1
                        is_end = i == len(tokens) - 1
                        word = word_item[0] if is_end else None
                        node = ContextState(index=self.num_nodes, is_end=is_end, word=word)
                        # add blank
                        if self.blank_token in prev_node.next:
                            prev_node.next[self.blank_token].next[token] = node
                            node.next[token] = node
                        else:
                            # create new blank node
                            self.num_nodes += 1
                            blank_node = ContextState(index=self.num_nodes, is_end=False)
                            blank_node.next[self.blank_token] = blank_node
                            blank_node.next[token] = node
                            prev_node.next[self.blank_token] = blank_node
                    # rewrite previous node
                    if prev_node.index != prev_node.next[token].index:
                        prev_node = prev_node.next[token]
                    else:
                        prev_node = prev_node.next[self.blank_token].next[token]
                    prev_token = token

    def draw(
        self,
        title: Optional[str] = None,
        symbol_table: Optional[Dict[int, str]] = None,
    ) -> "graphviz.Digraph":
        """Visualize a ContextGraph via graphviz.

        Render ContextGraph as an image via graphviz, and return the Digraph object

        Note:
          You need to install graphviz to use this function:
            pip install graphviz

        Args:
           title:
              Title to be displayed in image, e.g. 'A simple FSA example'
           symbol_table:
              Map the token ids to symbols.
        Returns:
          A Diagraph from grahpviz.
        """
        if _GRAPHVIZ_AVAILABLE is False:
            raise ImportError("graphviz is not installed")

        graph_attr = {
            "rankdir": "LR",
            "size": "8.5,11",
            "center": "1",
            "orientation": "Portrait",
            "ranksep": "0.30",
            "nodesep": "0.25",
        }
        if title is not None:
            graph_attr["label"] = title

        default_edge_attr = {
            "fontsize": "12",
        }

        default_node_attr = {
            "shape": "circle",
            "style": "bold",
            "fontsize": "12",
        }

        final_state_attr = {
            "shape": "doublecircle",
            "style": "bold",
            "fontsize": "12",
        }

        dot = graphviz.Digraph(name="Context Graph", graph_attr=graph_attr)

        seen = set()
        queue = deque()
        queue.append(self.root)
        # root id is always 0
        dot.node("0", label="0", **default_node_attr)
        seen.add(0)
        printed_arcs = set()

        while len(queue):
            current_node = queue.popleft()
            for token, node in current_node.next.items():
                if node.index not in seen:
                    label = f"{node.index}"
                    if node.is_end:
                        dot.node(str(node.index), label=label, **final_state_attr)
                    else:
                        dot.node(str(node.index), label=label, **default_node_attr)
                    seen.add(node.index)
                label = str(token) if symbol_table is None else symbol_table[token]
                if node.index != current_node.index:
                    output, input, arc = str(current_node.index), str(node.index), f"{label}"
                    if (output, input, arc) not in printed_arcs:
                        if arc == self.blank_token:
                            dot.edge(output, input, label=self.blank_token, color="blue", **default_edge_attr)
                        else:
                            dot.edge(output, input, label=arc)
                        queue.append(node)
                else:
                    output, input, arc = str(current_node.index), str(current_node.index), f"{label}"
                    if (output, input, arc) not in printed_arcs:
                        if arc == self.blank_token:
                            dot.edge(output, input, label=self.blank_token, color="blue", **default_edge_attr)
                        else:
                            dot.edge(output, input, label=arc, color="green")
                printed_arcs.add((output, input, arc))

        return dot
