# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
from collections import defaultdict, deque
from dataclasses import InitVar, dataclass, field
from pathlib import Path
from typing import NamedTuple, Optional

import numpy as np
import sentencepiece as spm
import torch
from lightning.pytorch import Trainer
from omegaconf import MISSING, DictConfig, OmegaConf

from nemo.collections.asr.parts.context_biasing.context_graph_universal import ContextGraph, ContextState
from nemo.collections.asr.parts.submodules.ngram_lm import DEFAULT_TOKEN_OFFSET, NGramGPULanguageModel
from nemo.collections.common.tokenizers import AggregateTokenizer
from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.utils import logging
from nemo.utils.exceptions import NeMoBaseException


@dataclass
class PhraseItem:
    phrase: str  # phrase itself
    lang: str  # per-phrase language (for aggregate tokenizer)
    # custom weight can be further added


@dataclass
class PhraseItem:
    phrase: str  # phrase itself
    lang: str  # per-phrase language (for aggregate tokenizer)
    # custom weight can be further added


class TokenWithLength(NamedTuple):
    token_id: int
    length: int = 1


@dataclass
class TokenizerExtras:
    vocab: list[str]
    token2id: dict[str, int]
    token_id2canonical_id: dict[int, int]
    canonical_id2alternatives: dict[int, list[int]]
    token_id2canonical_split: dict[int, tuple[int, ...]]
    max_token_len: int

    def __init__(self, tokenizer):
        # vocab + inversed vocab (id2token/token2id)
        vocab = tokenizer.vocab
        vocab_len_no_special = tokenizer.tokenizer.get_piece_size()
        self.vocab = vocab[:vocab_len_no_special]
        self.token2id = dict(zip(vocab, range(len(vocab))))

        # mapping: token_id -> canonical_id (lowercase)
        self.token_id2canonical_id = dict()
        for token, token_id in self.token2id.items():
            if token.lower() != token and token.lower() in self.token2id:
                self.token_id2canonical_id[token_id] = self.token2id[token.lower()]
            else:
                self.token_id2canonical_id[token_id] = token_id

        # inverse mapping: canonical_id -> list of token_id alternatives
        # TODO: fix (not necessary exists!)
        self.canonical_id2alternatives = defaultdict(list)
        for token_id, canonical_id in self.token_id2canonical_id.items():
            self.canonical_id2alternatives[canonical_id].append(token_id)

        # mapping: token_id -> canonical split + inverse
        self.max_token_len = 1
        self.token_id2canonical_split = dict()
        for token, token_id in self.token2id.items():
            if len(token) == 1:
                canonical_split = (self.token_id2canonical_id[token_id],)
            else:
                try:
                    canonical_split = tuple(
                        self.token_id2canonical_id[self.token2id[sub_token]] for sub_token in token
                    )
                except KeyError:
                    print(f"No split for token {token}:{token_id}")
                    canonical_split = (self.token_id2canonical_id[token_id],)
            self.token_id2canonical_split[token_id] = canonical_split
            self.max_token_len = max(self.max_token_len, len(canonical_split))
        # self.canonical_split2token_id = dict(zip(self.token_id2canonical_split.values(), self.token_id2canonical_split.keys()))
        self.canonical_split2token_ids = defaultdict(list)
        for token_id, canonical_split in self.token_id2canonical_split.items():
            self.canonical_split2token_ids[canonical_split].append(token_id)
            # if canonical_split in self.canonical_split2token_id: continue
            # token_id = self.token_id2canonical_id[token_id]
            # self.canonical_split2token_id[canonical_split] = token_id

    def ids_to_canonical(self, token_ids: list[int]) -> list[int]:
        canonical_ids = []
        for token_id in token_ids:
            canonical_ids.extend(self.token_id2canonical_split[token_id])
        return canonical_ids

    def ids_to_all_representations(self, token_ids: list[int]) -> tuple[list[int], list[list[TokenWithLength]]]:
        orig_len = [len(self.token_id2canonical_split[token_id]) for token_id in token_ids]
        canonical_ids = self.ids_to_canonical(token_ids=token_ids)
        if len(canonical_ids) != sum(orig_len):
            logging.warning(f"unequal len: {len(canonical_ids)} != {sum(orig_len)}")
        representations: list[list[TokenWithLength]] = [[] for _ in range(len(canonical_ids))]
        for i, canonical_token_id in enumerate(canonical_ids):
            # add alternatives to canonical tokens
            for token_id in self.canonical_id2alternatives[canonical_token_id]:
                representations[i].append(TokenWithLength(token_id=token_id, length=1))
            # add merges
            # TODO: can be optimized with tree
            for start in range(max(0, i - self.max_token_len), i):
                if tuple(canonical_ids[start : i + 1]) in self.canonical_split2token_ids:
                    for merged_token_id in self.canonical_split2token_ids[tuple(canonical_ids[start : i + 1])]:
                        representations[i].append(TokenWithLength(token_id=merged_token_id, length=i - start + 1))
        return orig_len, representations


@dataclass
class BoostingTreeModelConfig:
    """
    Boosting tree model config
    """

    model_path: Optional[str] = None  # The path to builded '.nemo' boosting tree model
    key_phrases_file: Optional[str] = None  # The path to the context-biasing list file (one phrase per line)
    key_phrases_list: Optional[list[str]] = (
        None  # The list of context-biasing phrases ['word1', 'word2', 'word3', ...]
    )
    # The list of context-biasing phrases with custom options:
    # [PhraseItem("word1", lang="en"), PhraseItem("frase dos", lang="es"), ...]
    # in CLI: key_phrase_items_list='[{phrase:"word1",lang:en},{phrase:"frase dos",lang:es}]'
    key_phrase_items_list: list[PhraseItem] | None = None
    context_score: float = 1.0  # The score for each arc transition in the context graph
    depth_scaling: float = (
        2.0  # The scaling factor for the depth of the context graph (2.0 for CTC, RNN-T and TDT, 1.0 for Canary)
    )
    unk_score: float = (
        0.0  # The score for unknown tokens (tokens that are not presented in the beginning of context-biasing phrases)
    )
    final_eos_score: float = (
        1.0  # The score for eos token after detected end of context phrase to prevent hallucination for AED models
    )
    score_per_phrase: float = 0.0  # Custom score for each phrase in the context graph
    source_lang: str = "en"  # The source language of the context-biasing phrases (for aggregate tokenizer)
    use_triton: bool = True  # Whether to use Triton for inference.
    uniform_weights: bool = False  # Whether to use uniform weights for the context-biasing tree as in Icefall
    use_variative_bpe: bool = False
    use_bpe_dropout: bool = False  # Whether to use BPE dropout for generating alternative transcriptions
    num_of_transcriptions: int = (
        5  # The number of alternative transcriptions to generate for each context-biasing phrase
    )
    bpe_alpha: float = 0.3  # The alpha parameter for BPE dropout

    @staticmethod
    def is_empty(cfg: "BoostingTreeModelConfig") -> bool:
        return (
            cfg.model_path is None
            and cfg.key_phrases_file is None
            and (not cfg.key_phrases_list)
            and (not cfg.key_phrase_items_list)
        )


class TBranch(NamedTuple):
    """Structure (tuple) to represent a branch in the boosting tree"""

    symbol: int  # token id
    start_node: ContextState  # start node of the branch
    next_node: ContextState  # next node of the branch


@dataclass
class BoostingTreeStorage:
    """
    NumPy-based storage for suffix tree (weighted acceptor) for phrase boosting
    """

    num_states_max: InitVar[int]
    num_arcs_max: InitVar[int]

    vocab_size: int
    max_order: int

    arcs: np.ndarray = field(init=False)
    states: np.ndarray = field(init=False)

    _node_cache: dict[int, int] = field(default_factory=dict)

    unk_score: float = 0.0
    final_eos_score: float = 0.0
    num_states: int = 0
    num_arcs: int = 0
    start_state: int = 0
    bos_state: int = 0

    _state_mapping: dict[int, int] = field(default_factory=dict)

    def __post_init__(self, num_states_max: int, num_arcs_max: int):
        if max(num_states_max, num_arcs_max) < np.iinfo(np.int32).max:
            int_np_dtype = np.int32
        else:
            int_np_dtype = np.int64
        self.arcs = np.zeros(
            [num_arcs_max],
            dtype=[("from", int_np_dtype), ("to", int_np_dtype), ("ilabel", int_np_dtype), ("weight", np.float32)],
        )
        self.states = np.zeros(
            [num_states_max],
            dtype=[
                ("arcs_start", int_np_dtype),
                ("arcs_end", int_np_dtype),
                ("order", int_np_dtype),
                ("backoff_to", int_np_dtype),
                ("backoff_w", np.float32),
                ("final", np.float32),
            ],
        )
        self.states["final"] = self.final_eos_score
        self._node_cache[0] = 0
        self.separate_bos_state = False

    def _add_tbranches_first_order(self, tbranches: list):
        """Add all first order tbranches to the model (similar with unigrams for N-Gram LM)"""

        tbranches = sorted(tbranches, key=lambda x: (x.start_node.id, x.symbol))

        self.num_states = 1
        self.num_arcs = 0
        # state: start_arcs, end_arcs, order, backoff_to, backoff_weight
        self.states[self.start_state] = (0, self.vocab_size, 1, self.start_state, 0.0, 0.0)
        added_symbols = set()
        num_vocab_labels = 0
        for tbranch in tbranches:
            assert tbranch.start_node.id == 0, "should be root"
            ilabel = tbranch.symbol
            assert ilabel < self.vocab_size
            arc_id = ilabel
            added_symbols.add(ilabel)
            if tbranch.next_node.id in self._state_mapping:
                next_state = self._state_mapping[tbranch.next_node.id]
            else:
                next_state = self.num_states
                self.num_states += 1
                self._state_mapping[tbranch.next_node.id] = next_state
            self.arcs[arc_id] = (
                self.start_state,
                next_state,
                ilabel,
                tbranch.next_node.node_score - tbranch.start_node.node_score,
            )
            self.num_arcs += 1

            if tbranch.next_node.is_end:
                # we do not penalize transitions from final nodes in case of non-uniform weights
                backoff_weight = 0.0
            else:
                backoff_weight = tbranch.next_node.fail.node_score - tbranch.next_node.node_score

            # state order
            self.states[next_state] = (
                0,
                0,
                self.states[self.start_state]["order"] + 1,
                self.start_state,
                backoff_weight,
                self.final_eos_score if tbranch.next_node.is_end else 0.0,
            )
            num_vocab_labels += 1
            self._node_cache[tbranch.next_node.id] = next_state

        for ilabel in range(self.vocab_size):
            if ilabel not in added_symbols:
                self.arcs[ilabel] = (self.start_state, self.start_state, ilabel, self.unk_score)
                self.num_arcs += 1

    def _add_tbranches_next_order(self, tbranches: list):
        """Add tbranches for the order > 1; should be called after adding first order tokens (unigrams), using increasing order"""
        tbranches = sorted(tbranches, key=lambda x: (x.start_node.id, x.symbol))

        for tbranch in tbranches:
            assert tbranch.start_node.id in self._state_mapping
            ilabel = tbranch.symbol
            from_state = self._node_cache[tbranch.start_node.id]
            assert ilabel < self.vocab_size
            backoff_state = self._node_cache[tbranch.next_node.fail.id]

            if tbranch.next_node.is_end and not self.uniform_weights:
                # we do not penalize transitions from final nodes in case of non-uniform weights
                backoff_weight = 0.0
            else:
                backoff_weight = tbranch.next_node.fail.node_score - tbranch.next_node.node_score

            arc_id = self.num_arcs
            self.num_arcs += 1
            if tbranch.next_node.id in self._state_mapping:
                next_state = self._state_mapping[tbranch.next_node.id]
            else:
                next_state = self.num_states
                self.num_states += 1
                self._state_mapping[tbranch.next_node.id] = next_state
            # token_score = tbranch.next_node.token_score
            token_score = tbranch.next_node.node_score - tbranch.start_node.node_score
            if self.uniform_weights and tbranch.next_node.is_end:
                token_score += tbranch.next_node.node_score

            self.arcs[arc_id] = (from_state, next_state, ilabel, token_score)

            self.states[next_state] = (
                0,
                0,
                self.states[from_state]["order"] + 1,
                backoff_state,
                backoff_weight,
                self.final_eos_score if tbranch.next_node.is_end else 0.0,
            )

            self._node_cache[tbranch.next_node.id] = next_state

            if self.states[from_state]["arcs_start"] == 0:
                self.states[from_state]["arcs_start"] = arc_id
                self.states[from_state]["arcs_end"] = arc_id + 1
            else:
                assert self.states[from_state]["arcs_end"] == arc_id
                self.states[from_state]["arcs_end"] = arc_id + 1

    def _start_adding_tbranches_for_order(self, order: int):
        """Prepare for adding tbranches for the given order: initialize temporary storage"""
        self._start_arcs = self.num_arcs
        self._cur_order = order
        self._tbranches = []
        self._tbranches_cnt = 0

    def _end_adding_tbranches_for_order(self, order: int):
        """Finish adding tbranches for the given order"""
        if order == 1:
            assert len(self._tbranches) == self._tbranches_cnt
            self._add_tbranches_first_order(tbranches=self._tbranches)
            self._tbranches = None
            self._tbranches_cnt = 0
        else:
            assert len(self._tbranches) == self._tbranches_cnt
            self._add_tbranches_next_order(tbranches=self._tbranches)
            self._tbranches = None
            self._tbranches_cnt = 0

    def sanity_check(self):
        """Sanity check for the model"""
        assert (self.arcs["ilabel"][: self.num_arcs] < self.vocab_size).all()
        assert (self.arcs["ilabel"][: self.num_arcs] >= 0).all()


@dataclass
class BoostingTreeConfig:
    """
    N-Gram LM Config
    """

    num_states: int = MISSING
    num_arcs: int = MISSING
    max_order: int = MISSING
    vocab_size: int = MISSING
    separate_bos_state: bool = False
    use_triton: bool | None = None


class GPUBoostingTreeModel(NGramGPULanguageModel):
    """
    GPU-accelerated boosting tree supporting batched queries.
    Fast implementation for parallel queries for full vocabulary.
    Supports autograd (differentiable weights).
    """

    START_STATE = 0

    def __init__(
        self,
        cfg: DictConfig,
        trainer: Trainer = None,
    ):
        """
        Stubs for constructor that does not initialize the structure.
        This constructor can be useful when storing/loading module using native torch serialization mechanism
        instead of directly reading ARPA model -> converting to Torch, which can be slow for large N-Gram models
        (of several GBs).

        Args:
            cfg:
                num_states: number of states in graph
                num_arcs: number of arcs (transitions) in graph
                max_order: maximum order of n-gram LM (maximum possible nubmer of transitions without backoffs)
                vocab_size: vocabulary size (existing vocabulary units in LM; should not include blank etc.)
                separate_bos_state: separate Begin-of-Sentence state (default: True - for n-gram LM)
                use_triton: allow using Triton implementation;
                    None (default) means "auto" (used if available), True means forced mode
                    (will crash if Triton is unavailable)
            trainer: Lightning trainer (optional)
        """
        super().__init__(cfg=cfg, trainer=trainer)
        self.bos_state = self.START_STATE  # Always START_STATE for gpu boosting tree

    @classmethod
    def _read_context_graph(
        cls,
        context_graph: ContextGraph,
    ) -> tuple[dict[int, int], list[TBranch]]:
        """
        Read context-biasing tree from python structure and return branches in TBranch format.

        Args:
            context_graph: python context-biasing graph
        """

        seen = set()
        queue = deque()
        queue.append(context_graph.root)
        # seen.add(0)
        order2cnt = {}
        tbranches_list = []

        # read context graph tree in breadth-first order to add branches for boosting tree generation
        while len(queue):
            current_node = queue.popleft()
            if current_node.id in seen:
                continue
            for token, node in current_node.next.items():
                tbranches_list.append(TBranch(symbol=token, start_node=current_node, next_node=node))
                order2cnt[current_node.level + 1] = order2cnt.get(current_node.level + 1, 0) + 1
                queue.append(node)
            seen.add(current_node.id)
        tbranches_list.sort(key=lambda t: (t.start_node.level, t.start_node.id, t.next_node.level, t.next_node.id))

        return order2cnt, tbranches_list

    @classmethod
    def from_context_graph(
        cls,
        context_graph: ContextGraph,
        vocab_size: int,
        unk_score: float = 0.0,
        final_eos_score: float = 0.0,
        use_triton: bool | None = None,
        uniform_weights: bool | None = None,
    ) -> "GPUBoostingTreeModel":
        """
        Constructor from Icefall context graph (dict-based tree).

        Args:
            context_graph: context-biasing graph
            vocab_size: vocabulary size (existing vocabulary units in LM; should not include blank etc.)
            unk_score: score for unknown tokens
            final_eos_score: score for eos token after detected end of context phrase
            use_triton: allow using Triton implementation;
                None (default) means "auto" (used if available), True means forced mode
                (will crash if Triton is unavailable)
            uniform_weights: whether to use uniform weights for the context-biasing tree as in Icefall

        Returns:
            GPUBoostingTreeModel instance
        """
        logging.info(f"{cls.__name__}: reading boosting tree from {context_graph}")

        order2cnt, tbranches_list = cls._read_context_graph(context_graph=context_graph)

        # init suffix tree storage
        max_states = context_graph.num_nodes + 1  # + 1 for root state
        boosting_tree_np = BoostingTreeStorage(
            num_states_max=max_states,
            num_states=0,
            num_arcs=0,
            num_arcs_max=sum(order2cnt.values()) + vocab_size * 2 + 1,
            unk_score=unk_score,
            final_eos_score=final_eos_score,
            vocab_size=vocab_size,
            max_order=max(order2cnt) + 1,
        )

        boosting_tree_np.uniform_weights = uniform_weights
        # convert context-biasing graph to np boosting tree
        tbranch_cur_order_i = 0
        cur_order = 1

        for tbranch in tbranches_list:

            if tbranch_cur_order_i == 0:
                boosting_tree_np._start_adding_tbranches_for_order(order=cur_order)
            tbranch_cur_order_i += 1

            # add tbranch
            boosting_tree_np._tbranches.append(tbranch)
            boosting_tree_np._tbranches_cnt += 1

            if tbranch_cur_order_i == order2cnt[cur_order]:
                boosting_tree_np._end_adding_tbranches_for_order(order=cur_order)
                logging.debug(f"Processed {order2cnt[cur_order]} n-grams of order {cur_order}")
                cur_order += 1
                tbranch_cur_order_i = 0

        assert tbranch_cur_order_i == 0
        boosting_tree_np.sanity_check()
        logging.debug(f"Loaded boosting model with {len(tbranches_list)} arcs")
        return GPUBoostingTreeModel.from_boosting_tree_np(boosting_tree_np=boosting_tree_np, use_triton=use_triton)

    @classmethod
    def from_boosting_tree_np(
        cls, boosting_tree_np: BoostingTreeStorage, use_triton: bool | None = None
    ) -> "GPUBoostingTreeModel":
        """
        Constructor from suffix tree storage.

        Args:
            suffix_tree_np: suffix tree
            use_triton: allow using Triton implementation;
                None (default) means "auto" (used if available), True means forced mode
                (will crash if Triton is unavailable)

        Returns:
            GPUBoostingTreeModel instance
        """
        model = GPUBoostingTreeModel(
            OmegaConf.structured(
                BoostingTreeConfig(
                    num_states=boosting_tree_np.num_states,
                    num_arcs=boosting_tree_np.num_arcs,
                    max_order=boosting_tree_np.max_order,
                    vocab_size=boosting_tree_np.vocab_size,
                    use_triton=use_triton,
                )
            )
        )
        model._init_from_suffix_tree_np(suffix_tree_np=boosting_tree_np)
        model._resolve_final()
        return model

    def advance(self, states: torch.Tensor, eos_id: Optional[int] = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Advance `states` [B]: return scores [B, V] and next states [B, V] for full vocab
        Args:
            states: batch of states
            eos_id: if not None, for eos symbol use final state weight

        Returns:
            tuple with next states and scores
        """
        if self.use_triton and states.device.type == "cuda":
            scores, next_states = self._advance_triton(states=states)
        else:
            scores, next_states = self._advance_pytorch(states=states)

        # replace eos_id score with maximum state weight to prevent from hallucinating in case of AED models (e.g. Canary)
        if eos_id is not None:
            # 1. replace eos score with maximum boosting value at each step
            scores[:, eos_id] = torch.clamp(torch.max(scores, dim=1).values, min=0.0)

            # 2. increase eos score after detected end of context phrase
            scores[:, eos_id] += self.get_final(states)

            next_states[:, eos_id] = states
        return scores, next_states

    def get_final(self, states: torch.Tensor) -> torch.Tensor:
        """
        Get final weights for states

        Args:
            states: batch of states

        Returns:
            tensor [B] with final weights for each state
        """

        return self.final_weights[states]

    @classmethod
    def dummy_boosting_tree(
        cls,
        vocab_size: int,
        use_triton: bool | None = None,
    ) -> "GPUBoostingTreeModel":
        """
        Constructs a trivial boosting tree with only one context phrase without scores.
        Useful for testing purposes (e.g., decoding).

        Returns:
            GPUBoostingTreeModel instance
        """

        context_graph_trivial = ContextGraph(context_score=0.0, depth_scaling=0.0)
        context_graph_trivial.build(token_ids=[[1]], phrases=["c"], scores=[0.0], uniform_weights=False)

        boosting_tree_trivial = GPUBoostingTreeModel.from_context_graph(
            context_graph=context_graph_trivial,
            vocab_size=vocab_size,
            unk_score=0.0,
            final_eos_score=0.0,
            use_triton=use_triton,
            uniform_weights=False,
        )
        return boosting_tree_trivial

    @classmethod
    def get_alternative_transcripts(
        cls, cfg: BoostingTreeModelConfig, tokenizer: TokenizerSpec, phrase: str
    ) -> list[list[int]]:
        """
        Get alternative transcriptions for a key phrase using BPE dropout
        """
        i = 1
        cur_step = 1
        transcripts_set = set()
        transcripts_list = [tokenizer.text_to_ids(phrase)]
        while i < cfg.num_of_transcriptions and cur_step < cfg.num_of_transcriptions * 5:
            cur_step += 1
            transcript = tokenizer.tokenizer.encode(phrase, enable_sampling=True, alpha=cfg.bpe_alpha, nbest_size=-1)
            transcript_text = tokenizer.ids_to_tokens(transcript)
            if transcript_text[0] == "▁":  # skip the case of empty first token
                continue
            transcript_text = " ".join(transcript_text)
            if transcript_text not in transcripts_set:
                transcripts_list.append(transcript)
                transcripts_set.add(transcript_text)
                i += 1
        return transcripts_list

    @classmethod
    def from_arpa(
        cls,
        lm_path: Path | str,
        vocab_size: int,
        normalize_unk: bool = True,
        use_triton: bool | None = None,
        token_offset: int = DEFAULT_TOKEN_OFFSET,
    ) -> "NGramGPULanguageModel":
        raise NeMoBaseException("Boosting tree cannot be loaded from ARPA file")

    @classmethod
    def from_config(cls, cfg: BoostingTreeModelConfig, tokenizer: TokenizerSpec) -> "GPUBoostingTreeModel":
        """
        Constructor boosting tree model from config file
        """
        # load boosting tree from already built model path
        if cfg.model_path is not None and os.path.exists(cfg.model_path):
            return cls.from_nemo(lm_path=cfg.model_path, vocab_size=tokenizer.vocab_size)

        # 1. read key phrases from file or list
        phrase_items_list: list[PhraseItem]
        if cfg.key_phrases_file is not None and bool(cfg.key_phrases_list or cfg.key_phrase_items_list):
            raise ValueError("Both file and phrases specified, use only one")
        elif cfg.key_phrases_file:
            with open(cfg.key_phrases_file, "r", encoding="utf-8") as f:
                phrase_items_list = [PhraseItem(line.strip(), cfg.source_lang) for line in f]
        elif cfg.key_phrases_list or cfg.key_phrase_items_list:
            phrase_items_list = []
            if cfg.key_phrases_list:
                phrase_items_list = [PhraseItem(phrase, cfg.source_lang) for phrase in cfg.key_phrases_list]
            if cfg.key_phrase_items_list:
                phrase_items_list += cfg.key_phrase_items_list
        else:
            raise ValueError("No key phrases file or list specified")

        # 2. tokenize key phrases
        phrases_dict = {}
        is_aggregate_tokenizer = isinstance(tokenizer, AggregateTokenizer)

        use_bpe_dropout = cfg.use_bpe_dropout
        if cfg.use_variative_bpe:
            use_bpe_dropout = False
            assert not is_aggregate_tokenizer
            tokenizer_extras = TokenizerExtras(tokenizer)
        if use_bpe_dropout:
            if is_aggregate_tokenizer:
                logging.warning(
                    "Aggregated tokenizer does not support BPE dropout, only one default transcription will be used..."
                )
                use_bpe_dropout = False
            spm.set_random_generator_seed(1234)  # fix random seed for reproducibility of BPE dropout

        for phrase_item in phrase_items_list:
            phrase = phrase_item.phrase
            if use_bpe_dropout:
                phrases_dict[phrase] = cls.get_alternative_transcripts(cfg, tokenizer, phrase)
            else:
                if is_aggregate_tokenizer:
                    phrases_dict[phrase] = tokenizer.text_to_ids(phrase, phrase_item.lang)
                else:
                    token_ids = tokenizer.text_to_ids(phrase)
                    if cfg.use_variative_bpe:
                        phrases_dict[phrase] = tokenizer_extras.ids_to_all_representations(token_ids)
                    else:
                        phrases_dict[phrase] = token_ids

        # 3. build python context graph
        contexts, scores, phrases = [], [], []
        for phrase in phrases_dict:
            if use_bpe_dropout:
                for transcript in phrases_dict[phrase]:
                    contexts.append(transcript)
                    scores.append(round(cfg.score_per_phrase / len(phrase), 2))
                    phrases.append(phrase)
            else:
                contexts.append(phrases_dict[phrase])
                scores.append(round(cfg.score_per_phrase / len(phrase), 2))
                phrases.append(phrase)

        context_graph = ContextGraph(context_score=cfg.context_score, depth_scaling=cfg.depth_scaling)
        if cfg.use_variative_bpe:
            context_graph.build_from_variative_bpe(
                token_ids=contexts,
                scores=scores,
                phrases=phrases,
                uniform_weights=cfg.uniform_weights,
            )
        else:
            context_graph.build(
                token_ids=contexts,
                scores=scores,
                phrases=phrases,
                uniform_weights=cfg.uniform_weights,
            )

        # graph_name = "no_var"
        # context_graph.draw(title=f"graph_{graph_name}", symbol_table=tokenizer.vocab, filename=f"/Users/vbataev/code/nemo/.sandbox/graph_{graph_name}.pdf")

        # 4. build GPU boosting tree model from python context graph
        boosting_tree_model = GPUBoostingTreeModel.from_context_graph(
            context_graph=context_graph,
            vocab_size=tokenizer.vocab_size,
            unk_score=cfg.unk_score,
            final_eos_score=cfg.final_eos_score,
            use_triton=cfg.use_triton,
            uniform_weights=cfg.uniform_weights,
        )

        # 5. save model
        if cfg.model_path is not None:
            boosting_tree_model.save_to(cfg.model_path)

        return boosting_tree_model
