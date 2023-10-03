# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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


try:
    import pynini
    from nemo_text_processing.text_normalization.en.graph_utils import GraphFst, delete_extra_space, delete_space
    from nemo_text_processing.text_normalization.en.verbalizers.word import WordFst
    from nn_wfst.en.electronic.verbalize import VerbalizeFst
    from pynini.lib import pynutil
except (ImportError, ModuleNotFoundError):
    raise ModuleNotFoundError(
        "The package `nemo_text_processing` was not installed in this environment. Please refer to"
        " https://github.com/NVIDIA/NeMo-text-processing and install this package before using "
        "this script"
    )


class VerbalizeFinalFst(GraphFst):
    """
    Finite state transducer that verbalizes an entire sentence.

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple options (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="verbalize_final", kind="verbalize", deterministic=deterministic)
        verbalize = VerbalizeFst(deterministic=deterministic).fst
        word = WordFst(deterministic=deterministic).fst
        types = verbalize | word

        if deterministic:
            graph = (
                pynutil.delete("tokens")
                + delete_space
                + pynutil.delete("{")
                + delete_space
                + types
                + delete_space
                + pynutil.delete("}")
            )
        else:
            graph = delete_space + types + delete_space
        graph = delete_space + pynini.closure(graph + delete_extra_space) + graph + delete_space
        self.fst = graph
