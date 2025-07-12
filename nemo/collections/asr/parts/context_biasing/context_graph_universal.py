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
# https://github.com/k2-fsa/icefall/blob/aac7df064a6d1529f3bf4acccc6c550bd260b7b3/icefall/context_graph.py


import os
import shutil
from collections import deque
from typing import Dict, List, Optional
import numpy as np


class ContextState:
    """The state in ContextGraph"""

    def __init__(
        self,
        id: int,
        token: int,
        token_score: float,
        node_score: float,
        output_score: float,
        is_end: bool,
        level: int,
        phrase: str = "",
        ac_threshold: float = 1.0,
    ):
        """Create a ContextState.

        Args:
          id:
            The node id, only for visualization now. A node is in [0, graph.num_nodes).
            The id of the root node is always 0.
          token:
            The token id.
          token_score:
            The bonus for each token during decoding, which will hopefully
            boost the token up to survive beam search.
          node_score:
            The accumulated bonus from root of graph to current node, it will be
            used to calculate the score for fail arc.
          output_score:
            The total scores of matched phrases, sum of the node_score of all
            the output node for current node.
          is_end:
            True if current token is the end of a context.
          level:
            The distance from current node to root.
          phrase:
            The context phrase of current state, the value is valid only when
            current state is end state (is_end == True).
          ac_threshold:
            The acoustic threshold (probability) of current context phrase, the
            value is valid only when current state is end state (is_end == True).
            Note: ac_threshold only used in keywords spotting.
        """
        self.id = id
        self.token = token
        self.token_score = token_score
        self.node_score = node_score
        self.output_score = output_score
        self.is_end = is_end
        self.level = level
        self.next = {}
        self.phrase = phrase
        self.ac_threshold = ac_threshold
        self.fail = None
        self.output = None


class ContextGraph:
    """The ContextGraph is modified from Aho-Corasick which is mainly
    a Trie with a fail arc for each node.
    See https://en.wikipedia.org/wiki/Aho%E2%80%93Corasick_algorithm for more details
    of Aho-Corasick algorithm.

    A ContextGraph contains some words / phrases that we expect to boost their
    scores during decoding. If the substring of a decoded sequence matches the word / phrase
    in the ContextGraph, we will give the decoded sequence a bonus to make it survive
    beam search.
    """

    def __init__(self, context_score: float, depth_scaling: float = 1.0, ac_threshold: float = 1.0):
        """Initialize a ContextGraph with the given ``context_score``.

        A root node will be created (**NOTE:** the token of root is hardcoded to -1).

        Args:
          context_score:
            The bonus score for each token(note: NOT for each word/phrase, it means longer
            word/phrase will have larger bonus score, they have to be matched though).
            Note: This is just the default score for each token, the users can manually
            specify the context_score for each word/phrase (i.e. different phrase might
            have different token score).
          depth_scaling:
            The depth scaling factor for each token [1, inf), it is used to give a larger score for all tokens after the first one.
          ac_threshold:
            The acoustic threshold (probability) to trigger the word/phrase, this argument
            is used only when applying the graph to keywords spotting system.
        """
        self.context_score = context_score
        self.depth_scaling = depth_scaling
        self.ac_threshold = ac_threshold
        self.num_nodes = 0
        self.root = ContextState(
            id=self.num_nodes,
            token=-1,
            token_score=0,
            node_score=0,
            output_score=0,
            is_end=False,
            level=0,
        )
        self.root.fail = self.root

    def _fill_fail_output(self):
        """This function fills the fail arc for each trie node, it can be computed
        in linear time by performing a breadth-first search starting from the root.
        See https://en.wikipedia.org/wiki/Aho%E2%80%93Corasick_algorithm for the
        details of the algorithm.
        """
        queue = deque()
        for token, node in self.root.next.items():
            node.fail = self.root
            queue.append(node)
        while queue:
            current_node = queue.popleft()
            for token, node in current_node.next.items():
                fail = current_node.fail
                if token in fail.next:
                    fail = fail.next[token]
                else:
                    fail = fail.fail
                    while token not in fail.next:
                        fail = fail.fail
                        if fail.token == -1:  # root
                            break
                    if token in fail.next:
                        fail = fail.next[token]
                node.fail = fail
                # fill the output arc
                output = node.fail
                while not output.is_end:
                    output = output.fail
                    if output.token == -1:  # root
                        output = None
                        break
                node.output = output
                node.output_score += 0 if output is None else output.output_score
                queue.append(node)

    def build(
        self,
        token_ids: List[List[int]],
        phrases: Optional[List[str]] = None,
        scores: Optional[List[float]] = None,
        ac_thresholds: Optional[List[float]] = None,
        uniform_weights: Optional[bool] = False,
    ):
        """Build the ContextGraph from a list of token list.
        It first build a trie from the given token lists, then fill the fail arc
        for each trie node.

        See https://en.wikipedia.org/wiki/Trie for how to build a trie.

        Args:
          token_ids:
            The given token lists to build the ContextGraph, it is a list of
            token list, the token list contains the token ids
            for a word/phrase. The token id could be an id of a char
            (modeling with single Chinese char) or an id of a BPE
            (modeling with BPEs).
          phrases:
            The given phrases, they are the original text of the token_ids, the
            length of `phrases` MUST be equal to the length of `token_ids`.
          scores:
            The customize boosting score(token level) for each word/phrase,
            0 means using the default value (i.e. self.context_score).
            It is a list of floats, and the length of `scores` MUST be equal to
            the length of `token_ids`.
          ac_thresholds:
            The customize trigger acoustic threshold (probability) for each phrase,
            0 means using the default value (i.e. self.ac_threshold). It is
            used only when this graph applied for the keywords spotting system.
            The length of `ac_threshold` MUST be equal to the length of `token_ids`.
          uniform_weights:
            If True, the weights will be distributed uniformly for all tokens as in Icefall.

        Note: The phrases would have shared states, the score of the shared states is
              the MAXIMUM value among all the tokens sharing this state.
        """
        num_phrases = len(token_ids)
        if phrases is not None:
            assert len(phrases) == num_phrases, (len(phrases), num_phrases)
        if scores is not None:
            assert len(scores) == num_phrases, (len(scores), num_phrases)
        if ac_thresholds is not None:
            assert len(ac_thresholds) == num_phrases, (len(ac_thresholds), num_phrases)

        for index, tokens in enumerate(token_ids):
            phrase = "" if phrases is None else phrases[index]
            score = 0.0 if scores is None else scores[index]
            ac_threshold = 0.0 if ac_thresholds is None else ac_thresholds[index]
            node = self.root
            # If has customized score using the customized token score, otherwise
            # using the default score
            context_score = self.context_score if score == 0.0 else score
            threshold = self.ac_threshold if ac_threshold == 0.0 else ac_threshold
            for i, token in enumerate(tokens):
                if token not in node.next:
                    if i > 0 and not uniform_weights:
                        token_score = context_score * self.depth_scaling + np.log(
                            i + 1
                        )  # depth scaling is used to give a larger score for all tokens after the first one
                    else:
                        token_score = context_score
                    self.num_nodes += 1
                    is_end = i == len(tokens) - 1
                    node_score = node.node_score + token_score
                    node.next[token] = ContextState(
                        id=self.num_nodes,
                        token=token,
                        token_score=token_score,
                        node_score=node_score,
                        output_score=node_score if is_end else 0,
                        is_end=is_end,
                        level=i + 1,
                        phrase=phrase if is_end else "",
                        ac_threshold=threshold if is_end else 0.0,
                    )
                else:
                    # node exists, get the score of shared state.
                    token_score = max(context_score, node.next[token].token_score)
                    node.next[token].token_score = token_score
                    node_score = node.node_score + token_score
                    node.next[token].node_score = node_score
                    is_end = i == len(tokens) - 1 or node.next[token].is_end
                    node.next[token].output_score = node_score if is_end else 0
                    node.next[token].is_end = is_end
                    if i == len(tokens) - 1:
                        node.next[token].phrase = phrase
                        node.next[token].ac_threshold = threshold
                node = node.next[token]
        self._fill_fail_output()

    def draw(
        self,
        title: Optional[str] = None,
        filename: Optional[str] = "",
        symbol_table: Optional[Dict[int, str]] = None,
    ) -> "Digraph":  # noqa
        """Visualize a ContextGraph via graphviz.

        Render ContextGraph as an image via graphviz, and return the Digraph object;
        and optionally save to file `filename`.
        `filename` must have a suffix that graphviz understands, such as
        `pdf`, `svg` or `png`.

        Note:
          You need to install graphviz to use this function::

            pip install graphviz

        Args:
           title:
              Title to be displayed in image, e.g. 'A simple FSA example'
           filename:
              Filename to (optionally) save to, e.g. 'foo.png', 'foo.svg',
              'foo.png'  (must have a suffix that graphviz understands).
           symbol_table:
              Map the token ids to symbols.
        Returns:
          A Diagraph from grahpviz.
        """

        try:
            import graphviz
        except Exception:
            print("You cannot use `to_dot` unless the graphviz package is installed.")
            raise

        graph_attr = {
            "rankdir": "LR",
            "size": "8.5,11",
            "center": "1",
            "orientation": "Portrait",
            "ranksep": "0.4",
            "nodesep": "0.25",
        }
        if title is not None:
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

        seen = set()
        queue = deque()
        queue.append(self.root)
        # root id is always 0
        dot.node("0", label="0", **default_node_attr)
        dot.edge("0", "0", color="red")
        seen.add(0)

        while len(queue):
            current_node = queue.popleft()
            for token, node in current_node.next.items():
                if node.id not in seen:
                    node_score = f"{node.node_score:.2f}".rstrip("0").rstrip(".")
                    output_score = f"{node.output_score:.2f}".rstrip("0").rstrip(".")
                    label = f"{node.id}/({node_score}, {output_score})"
                    if node.is_end:
                        dot.node(str(node.id), label=label, **final_state_attr)
                    else:
                        dot.node(str(node.id), label=label, **default_node_attr)
                    seen.add(node.id)
                weight = f"{node.token_score:.2f}".rstrip("0").rstrip(".")
                label = str(token) if symbol_table is None else symbol_table[token]
                dot.edge(str(current_node.id), str(node.id), label=f"{label}/{weight}")
                dot.edge(
                    str(node.id),
                    str(node.fail.id),
                    color="red",
                )
                if node.output is not None:
                    dot.edge(
                        str(node.id),
                        str(node.output.id),
                        color="green",
                    )
                queue.append(node)

        if filename:
            _, extension = os.path.splitext(filename)
            if extension == "" or extension[0] != ".":
                raise ValueError("Filename needs to have a suffix like .png, .pdf, .svg: {}".format(filename))

            import tempfile

            with tempfile.TemporaryDirectory() as tmp_dir:
                temp_fn = dot.render(
                    filename="temp",
                    directory=tmp_dir,
                    format=extension[1:],
                    cleanup=True,
                )

                shutil.move(temp_fn, filename)

        return dot
