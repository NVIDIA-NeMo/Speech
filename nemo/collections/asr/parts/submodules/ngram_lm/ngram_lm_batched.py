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

import re
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import InitVar, dataclass, field
from pathlib import Path
from typing import NamedTuple, Optional, Union, cast

import numpy as np
import torch
import torch.nn as nn
from lightning.pytorch import Trainer
from omegaconf import MISSING, DictConfig, OmegaConf
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm

from nemo.collections.asr.parts.submodules.ngram_lm.constants import DEFAULT_TOKEN_OFFSET
from nemo.collections.common.parts import NEG_INF
from nemo.core import ModelPT, PretrainedModelInfo
from nemo.core.utils.optional_libs import KENLM_AVAILABLE, TRITON_AVAILABLE, kenlm_required, triton_required
from nemo.utils import logging

if KENLM_AVAILABLE:
    import kenlm

if TRITON_AVAILABLE:
    import triton

    from nemo.collections.asr.parts.submodules.ngram_lm.ngram_lm_triton import ngram_advance_triton_kernel

# Define constants for parsing ARPA
_BOS_ID = -1  # Begin-of-Sentence
_EOS_ID = -2  # End-of-Sentence
_UNK_ID = -3  # Unk
_SPECIAL_SYMBOLS_MAP = {"<s>": _BOS_ID, "</s>": _EOS_ID, "<unk>": _UNK_ID}


def _log_10_to_e(score):
    """Convert logarithm with base 10 to natural"""
    return score / np.log10(np.e)


class KenLMBatchedWrapper:
    """
    KenLM model wrapper for single element and batched queries (slow) for reference decoding and testing purposes.
    """

    @kenlm_required
    def __init__(self, lm_path: Path | str, vocab_size: int, token_offset: int = DEFAULT_TOKEN_OFFSET):
        """
        Constructor from KenLM (binary) or ARPA (text) model

        Args:
            lm_path: path to the LM file (binary KenLM or text ARPA model)
            vocab_size: full vocabulary size for the LM
            token_offset: offset for the tokens used for building LM
        """
        self.ngram_lm = kenlm.Model(str(lm_path))
        self.token_offset = token_offset
        self.vocab_size = vocab_size

    @classmethod
    def from_file(
        cls, lm_path: Path | str, vocab_size: int, token_offset: int = DEFAULT_TOKEN_OFFSET
    ) -> "KenLMBatchedWrapper":
        """
        Constructor from KenLM (binary) or ARPA (text) model (same as `__init__`).
        Useful for fast switching between NGramGPULanguageModel and this class.

        Args:
            lm_path: path to .nemo checkpoint or ARPA (text) file
            vocab_size: model vocabulary size:
            token_offset: offset for the tokens used for building ARPA LM

        Returns:
            KenLMBatchedWrapper instance
        """
        return cls(lm_path=lm_path, vocab_size=vocab_size, token_offset=token_offset)

    def get_init_state(self, bos=True) -> "kenlm.State":
        """
        Get initial state for the LM (KenLM)

        Args:
            bos: use begin-of-sentence (start-of-sentence) state, default True

        Returns:
            initial state
        """
        init_lm_state = kenlm.State()

        if not bos:
            return init_lm_state

        self.ngram_lm.BeginSentenceWrite(init_lm_state)
        return init_lm_state

    def get_init_states(self, batch_size: int, bos=True) -> list["kenlm.State"]:
        """
        Get initial states for the LM (KenLM) for batched queries

        Args:
            batch_size: batch size
            bos: use begin-of-sentence (start-of-sentence) state, default True

        Returns:
            batch (list) of states
        """
        return [self.get_init_state(bos=bos) for _ in range(batch_size)]

    def advance(self, states: list["kenlm.State"]) -> tuple[torch.Tensor, list[list["kenlm.State"]]]:
        """
        Advance `states` [B]: return scores [B, V] and next states [B, V] for full vocab
        Args:
            states: batch of states

        Returns:
            tuple containing next states and scores
        """
        batch_size = len(states)
        new_states = [[] for _ in range(len(states))]
        scores = torch.zeros(batch_size, self.vocab_size)
        for i, state in enumerate(states):
            for label in range(self.vocab_size):
                score, new_state = self.advance_single(state, label)
                scores[i, label] = score
                new_states[i].append(new_state)

        return scores, new_states

    def advance_single(self, state: "kenlm.State", label: int) -> tuple[float, "kenlm.State"]:
        """
        Computes the score with KenLM N-gram language model for `label` given `state`
        Args:
            state: KenLM state
            label: text token

        Returns:
            tuple: score, next state
        """
        if self.token_offset:
            label = chr(label + self.token_offset)
        else:
            label = str(label)

        next_state = kenlm.State()
        lm_score = self.ngram_lm.BaseScore(state, label, next_state)
        lm_score /= np.log10(np.e)

        return lm_score, next_state

    def score_sentence(self, sentence: list[int], bos: bool = True, eos: bool = False) -> torch.Tensor:
        """
        Compute log-probabilities for all labels in the sentence using N-Gram LM.

        Args:
            sentence: list of tokens
            bos: start with BOS symbol

        Returns:
            Tensor with scores for the sentence. Size: [L+1] if eos else [L]
        """
        state = self.get_init_state(bos=bos)
        scores = []
        for label in sentence:
            score, state = self.advance_single(state=state, label=label)
            scores.append(score)
        if eos:
            scores.append(self.get_final_single(state=state))
        return torch.FloatTensor(scores)

    def score_sentences(self, sentences: list[list[int]], bos: bool = True, eos: bool = False) -> torch.Tensor:
        """
        Compute log-probabilities for all labels in sentences using N-Gram LM.

        Args:
            sentences: list of sequences of tokens
            bos: start with BOS symbol

        Returns:
            Tensor with scores for each sentence. Size: [B, L+1] if eos else [B, L]
        """
        return pad_sequence(
            [self.score_sentence(sentence, bos=bos, eos=eos) for sentence in sentences], batch_first=True
        )

    def get_final_single(self, state: "kenlm.State") -> float:
        """
        Get final score for the state

        Args:
            state: state

        Returns:
            final score
        """
        new_state = kenlm.State()  # needed for query, but we ignore it further since not needed in decoding
        return _log_10_to_e(self.ngram_lm.BaseScore(state, "</s>", new_state))

    def get_final(self, states: list["kenlm.State"]) -> torch.Tensor:
        """
        Get final scores for the states

        Args:
            states: list of states

        Returns:
            Tensor [B] with final scores
        """
        final_scores = torch.zeros(len(states))
        for i, state in enumerate(states):
            final_scores[i] = self.get_final_single(state)

        return final_scores


class NGram(NamedTuple):
    """Structure (tuple) to represent N-Gram element (symbols, weight, backoff)"""

    symbols: tuple[int, ...]
    weight: float
    backoff: float


class Arc(NamedTuple):
    """Structure (tuple) to represent arc in the weighted acceptor"""

    weight: float
    ilabel: int
    to: int


@dataclass
class SuffixTreeStorage:
    """
    NumPy-based storage for suffix tree (weighted acceptor) for N-Gram LM
    """

    num_states_max: InitVar[int]
    num_arcs_max: InitVar[int]

    vocab_size: int
    max_order: int

    separate_bos_state: InitVar[bool] = True

    arcs: np.ndarray = field(init=False)
    states: np.ndarray = field(init=False)

    _arc_cache: dict[tuple[int, ...], int] = field(default_factory=dict)

    unk_prob: float = float("-inf")
    normalize_unk: bool = True

    num_states: int = 0
    num_arcs: int = 0

    start_state: int = 0
    bos_state: int = 1

    def __post_init__(self, num_states_max: int, num_arcs_max: int, separate_bos_state: bool = True):
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
        self.states["final"] = NEG_INF
        self.bos_state = 1 if separate_bos_state else self.start_state

    def _add_unigrams(self, ngrams: np.ndarray, bos_id: int, unk_id: int):
        """Add all unigrams"""
        assert bos_id < 0 and unk_id < 0
        bos_unigram = None

        ngrams.sort(order="symbols")
        for ngram in ngrams:
            assert len(ngram["symbols"]) == 1  # unigrams
            symbol = ngram["symbols"][-1]
            if symbol == unk_id:
                self.unk_prob = ngram["weight"]
            elif symbol == bos_id:
                bos_unigram = ngram
        assert bos_unigram is not None

        self.num_states = 2  # SOS + BOS
        self.num_arcs = 0
        # state: start_arcs, end_arcs, order, backoff_to, backoff_weight
        self.states[self.start_state] = (0, self.vocab_size, 1, self.start_state, 0.0, NEG_INF)
        added_symbols = set()
        num_vocab_labels = 0
        for ngram in ngrams:
            ilabel = ngram["symbols"][-1]
            if ilabel < 0:
                # special symbol
                if ilabel == _EOS_ID:
                    self.states[self.start_state]["final"] = ngram["weight"]
                continue
            assert ilabel < self.vocab_size
            arc_id = ilabel
            added_symbols.add(ilabel)
            next_state = self.num_states
            self.num_states += 1
            self.arcs[arc_id] = (self.start_state, next_state, ilabel, ngram["weight"])
            self.num_arcs += 1
            # state order
            self.states[next_state] = (
                0,
                0,
                self.states[self.start_state]["order"] + 1,
                self.start_state,
                ngram["backoff"],
                NEG_INF,
            )
            num_vocab_labels += 1

        if self.normalize_unk:
            num_unk_labels = self.vocab_size - num_vocab_labels
            if num_unk_labels > 1:
                self.unk_prob -= np.log(num_unk_labels)
        for ilabel in range(self.vocab_size):
            if ilabel not in added_symbols:
                self.arcs[ilabel] = (self.start_state, self.start_state, ilabel, self.unk_prob)
                self.num_arcs += 1

        # add BOS unigram
        assert self.bos_state == 1
        # NB: we do not add BOS unigram to the arcs, but only to the states
        self.states[self.bos_state] = (
            0,
            0,
            self.states[self.start_state]["order"] + 1,
            self.start_state,
            bos_unigram["backoff"],
            NEG_INF,
        )

    def _find_state(self, symbols: tuple[int, ...], bos_id: int) -> int:
        """
        Find the state given sequence of symbols
        Args:
            symbols: sequence of symbols
            bos_id: ID of the Begin-of-Sentence symbol

        Returns:
            state in tree for the last symbol
        """
        if len(symbols) > 1:
            return self._arc_cache[tuple(symbols)]
        assert len(symbols) == 1
        label = symbols[0]
        if label == bos_id:
            return 1
        elif label >= 0:
            return self.arcs[label]["to"]
        raise ValueError(f"Invalid symbol {label}")

    def _add_ngrams_next_order(self, ngrams: np.ndarray, bos_id: int):
        """Add ngrams for the order > 1; should be called after adding unigrams, using increasing order"""
        ngrams.sort(order="symbols")
        new_arc_cache = dict()
        for ngram in tqdm(ngrams):
            symbols = ngram["symbols"].item()
            ilabel = symbols[-1]
            from_state = self._find_state(symbols[:-1], bos_id=bos_id)
            if ilabel < 0:
                assert ilabel == _EOS_ID
                self.states[from_state]["final"] = ngram["weight"]
                continue
            assert ilabel < self.vocab_size
            backoff_state = self._find_state(symbols[1:], bos_id=bos_id)

            arc_id = self.num_arcs
            next_state = self.num_states
            self.num_arcs += 1
            self.num_states += 1
            self.arcs[arc_id] = (from_state, next_state, ilabel, ngram["weight"])
            # state: start_arcs, end_arcs, order, backoff_to, backoff_weight
            self.states[next_state] = (
                0,
                0,
                self.states[from_state]["order"] + 1,
                backoff_state,
                ngram["backoff"],
                NEG_INF,
            )

            if self.states[from_state]["arcs_start"] == 0:
                self.states[from_state]["arcs_start"] = arc_id
                self.states[from_state]["arcs_end"] = arc_id + 1
            else:
                assert self.states[from_state]["arcs_end"] == arc_id
                self.states[from_state]["arcs_end"] = arc_id + 1
            # cache state
            new_arc_cache[symbols] = next_state
        self._arc_cache = new_arc_cache  # replace arc cache, previous is not needed

    def _start_adding_ngrams_for_order(self, order: int, max_ngrams: int):
        """Prepare for adding ngrams for the given order: initialize temporary storage"""
        self._start_arcs = self.num_arcs
        self._cur_order = order
        if order < self.max_order:
            dtype = [
                ("symbols", [(f"{i}", np.int32) for i in range(order)]),
                ("weight", np.float32),
                ("backoff", np.float32),
            ]
            self._ngrams = np.zeros([max_ngrams], dtype=dtype)
            self._ngrams_cnt = 0
        # for max order - no need in accumulator

    def _add_ngram(self, ngram: NGram, bos_id: int):
        """Helper to add ngram"""
        assert len(ngram.symbols) == self._cur_order
        if self._cur_order == self.max_order:
            self._add_ngram_max_order(ngram=ngram, bos_id=bos_id)
            return
        self._ngrams[self._ngrams_cnt] = (ngram.symbols, ngram.weight, ngram.backoff)
        self._ngrams_cnt += 1

    def _end_adding_ngrams_for_order(self, order: int, bos_id: int, unk_id: int):
        """Finish adding ngrams for the given order"""
        if order == 1:
            assert self._ngrams.shape[0] == self._ngrams_cnt
            self._add_unigrams(ngrams=self._ngrams, bos_id=bos_id, unk_id=unk_id)
            self._ngrams = None
            self._ngrams_cnt = 0
        elif order < self.max_order:
            assert self._ngrams.shape[0] == self._ngrams_cnt
            self._add_ngrams_next_order(ngrams=self._ngrams, bos_id=bos_id)
            self._ngrams = None
            self._ngrams_cnt = 0
        else:
            self._end_adding_ngrams_max_order()

    def _add_ngram_max_order(self, ngram: NGram, bos_id: int):
        """Add ngram for the maximum order"""
        ilabel = ngram.symbols[-1]
        from_state = self._find_state(ngram.symbols[:-1], bos_id=bos_id)
        if ilabel < 0:
            assert ilabel == _EOS_ID
            self.states[from_state]["final"] = ngram.weight
            return
        backoff_state = self._find_state(ngram.symbols[1:], bos_id=bos_id)

        arc_id = self.num_arcs
        self.num_arcs += 1
        self.arcs[arc_id] = (from_state, backoff_state, ilabel, ngram.weight)

    def _end_adding_ngrams_max_order(self):
        """Finish adding ngrams for the maximum order"""
        self.arcs[self._start_arcs : self.num_arcs].sort(order=["from", "ilabel"])
        for arc_i in range(self._start_arcs, self.num_arcs):
            from_state = self.arcs[arc_i]["from"]
            if self.states[from_state]["arcs_start"] == 0:
                self.states[from_state]["arcs_start"] = arc_i
            self.states[from_state]["arcs_end"] = arc_i + 1

    def sanity_check(self):
        """Sanity check for the model"""
        assert (self.arcs["ilabel"][: self.num_arcs] < self.vocab_size).all()
        assert (self.arcs["ilabel"][: self.num_arcs] >= 0).all()


@dataclass
class NGramLMConfig:
    """
    N-Gram LM Config
    """

    num_states: int = MISSING
    num_arcs: int = MISSING
    max_order: int = MISSING
    vocab_size: int = MISSING
    separate_bos_state: bool = True
    use_triton: bool | None = None


class NGramGPULanguageModel(ModelPT):
    """
    N-Gram GPU-accelerated Language Model (NGPU-LM) supporting batched queries.
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
        cfg = cast(NGramLMConfig, cfg)
        self.use_triton = cfg.use_triton if cfg.use_triton is not None else TRITON_AVAILABLE
        if not self.use_triton:
            logging.warning(
                "Triton is disabled. Version without Triton is not compatible with Cuda graphs; decoding can be slow"
            )

        self.bos_state: int = 1 if cfg.separate_bos_state else self.START_STATE
        self.vocab_size: int = cfg.vocab_size
        self.num_states: int = cfg.num_states
        self.num_arcs: int = cfg.num_arcs
        self.max_order: int = cfg.max_order
        self.num_arcs_extended: int = cfg.num_arcs + self.vocab_size  # + extra padding

        # parameters: weights (forward/backoff/final)
        self.arcs_weights = nn.Parameter(torch.zeros([self.num_arcs_extended]))
        self.backoff_weights = nn.Parameter(torch.zeros([self.num_states]))
        self.final_weights = nn.Parameter(torch.zeros([self.num_states]))

        if max(self.num_states, self.num_arcs_extended) < torch.iinfo(torch.int32).max:
            int_dtype = torch.int32
        else:
            int_dtype = torch.int64
        # buffers: LM (suffix tree) structure
        # arcs data
        self.from_states = nn.Buffer(torch.zeros([self.num_arcs_extended], dtype=int_dtype))
        self.to_states = nn.Buffer(torch.zeros([self.num_arcs_extended], dtype=int_dtype))
        self.ilabels = nn.Buffer(torch.zeros([self.num_arcs_extended], dtype=int_dtype))

        # states data
        self.backoff_to_states = nn.Buffer(torch.zeros([self.num_states], dtype=int_dtype))
        self.start_end_arcs = nn.Buffer(torch.zeros([self.num_states, 2], dtype=int_dtype))
        self.state_order = nn.Buffer(torch.zeros([self.num_states], dtype=int_dtype))

        self._final_resolved = False

    @classmethod
    def list_available_models(cls) -> list[PretrainedModelInfo]:
        """Stub necessary to create the ModelPT. Not used for LM"""
        return []

    def setup_training_data(self, train_data_config: Union[DictConfig, dict]):
        """Stub necessary to create the ModelPT. Not used for LM"""
        pass

    def setup_validation_data(self, val_data_config: Union[DictConfig, dict]):
        """Stub necessary to create the ModelPT. Not used for LM"""
        pass

    def compatible_with_cuda_graphs(self) -> bool:
        """True if model can be compiled as a part of CUDA graph, False otherwise"""
        return self.use_triton

    @classmethod
    def from_nemo(
        cls,
        lm_path: Path | str,
        vocab_size: int,
        use_triton: bool | None = None,
    ) -> "NGramGPULanguageModel":
        """
        Constructor from Nemo checkpoint (state dict).

        Args:
            lm_path: path to .nemo checkpoint
            vocab_size: model vocabulary size
            use_triton: allow using Triton implementation; None (default) means "auto" (used if available)
        """
        model = cls.restore_from(restore_path=str(lm_path), map_location="cpu")
        model._resolve_final()
        assert model.vocab_size == vocab_size
        model.use_triton = use_triton if use_triton is not None else TRITON_AVAILABLE
        if not model.use_triton:
            logging.warning(
                "Triton is disabled. Version without Triton is not compatible with Cuda graphs; decoding can be slow"
            )
        return model

    @classmethod
    def from_file(
        cls,
        lm_path: Path | str,
        vocab_size: int,
        normalize_unk: bool = True,
        use_triton: bool | None = None,
        token_offset: int = DEFAULT_TOKEN_OFFSET,
    ) -> "NGramGPULanguageModel":
        """
        Constructor from ARPA or Nemo (`.nemo`) checkpoint.

        Args:
            lm_path: path to .nemo checkpoint or ARPA (text) file
            vocab_size: model vocabulary size:
            normalize_unk: normalize unk probabilities (for tokens missing in LM) to make
                all unigram probabilities sum to 1.0 (default: True)
            use_triton: allow using Triton implementation; None (default) means "auto" (used if available)
            token_offset: offset for the tokens used for building ARPA LM

        Returns:
            NGramGPULanguageModel instance
        """
        if not isinstance(lm_path, Path):
            lm_path = Path(lm_path)
        if lm_path.suffix == ".nemo":
            return cls.from_nemo(lm_path=lm_path, vocab_size=vocab_size, use_triton=use_triton)
        return cls.from_arpa(
            lm_path=lm_path,
            vocab_size=vocab_size,
            normalize_unk=normalize_unk,
            token_offset=token_offset,
            use_triton=use_triton,
        )

    @classmethod
    def from_arpa(
        cls,
        lm_path: Path | str,
        vocab_size: int,
        normalize_unk: bool = True,
        use_triton: bool | None = None,
        token_offset: int = DEFAULT_TOKEN_OFFSET,
    ) -> "NGramGPULanguageModel":
        """
        Constructor from ARPA LM (text format).

        Args:
            lm_path: path to ARPA model (human-readable)
            vocab_size: vocabulary size (existing vocabulary units in LM; should not include blank etc.)
            normalize_unk: unk normalization to make all output probabilities sum to 1.0 (default: True).
                Setting to False can be useful for one-to-one comparison with KenLM (tests, etc.).
            use_triton: allow using Triton implementation;
                None (default) means "auto" (used if available), True means forced mode
                (will crash if Triton is unavailable)
            token_offset: offset for the tokens used for building ARPA LM

        Returns:
            NGramGPULanguageModel instance
        """
        logging.info(f"{cls.__name__}: reading LM from {lm_path}")
        with open(lm_path, "r", encoding="utf-8") as f:
            order2cnt = cls._read_header(f=f)
            # init suffix tree storage
            max_order = max(order2cnt.keys())
            total_ngrams = sum(order2cnt.values())
            max_states = 2 + vocab_size + sum(order2cnt[o] for o in range(2, max_order))  # without last!
            suffix_tree_np = SuffixTreeStorage(
                num_states_max=max_states,
                num_states=0,
                num_arcs=0,
                num_arcs_max=total_ngrams + vocab_size * 2 + 1,
                normalize_unk=normalize_unk,
                vocab_size=vocab_size,
                max_order=max_order,
            )
            # add ngrams to suffix tree
            ngram_cur_order_i = 0
            cur_order = 1
            for ngram in tqdm(cls._read_ngrams(f=f, token_offset=token_offset), total=total_ngrams):
                if ngram_cur_order_i == 0:
                    suffix_tree_np._start_adding_ngrams_for_order(order=cur_order, max_ngrams=order2cnt[cur_order])
                ngram_cur_order_i += 1
                suffix_tree_np._add_ngram(ngram=ngram, bos_id=_BOS_ID)

                if ngram_cur_order_i == order2cnt[cur_order]:
                    suffix_tree_np._end_adding_ngrams_for_order(order=cur_order, bos_id=_BOS_ID, unk_id=_UNK_ID)
                    logging.debug(f"Processed {order2cnt[cur_order]} n-grams of order {cur_order}")
                    cur_order += 1
                    ngram_cur_order_i = 0

            assert ngram_cur_order_i == 0
            suffix_tree_np.sanity_check()
        return NGramGPULanguageModel.from_suffix_tree(suffix_tree_np=suffix_tree_np, use_triton=use_triton)

    @classmethod
    def dummy_unigram_lm(
        cls,
        vocab_size: int,
        use_triton: bool | None = None,
    ) -> "NGramGPULanguageModel":
        """
        Constructs a trivial unigram LM with uniform distribution over the vocabulary.
        Useful for testing purposes (e.g., decoding).

        Returns:
            NGramGPULanguageModel instance
        """
        model = NGramGPULanguageModel(
            OmegaConf.structured(
                NGramLMConfig(
                    num_states=2,
                    num_arcs=vocab_size,
                    max_order=1,
                    vocab_size=vocab_size,
                    use_triton=use_triton,
                )
            )
        )
        unigram_weight = -np.log(vocab_size)

        # start state
        model.backoff_weights.data[0] = 0.0
        model.final_weights.data[0] = unigram_weight
        model.backoff_to_states.data[0] = 0
        model.start_end_arcs.data[0, :] = torch.tensor([0, vocab_size], dtype=model.start_end_arcs.dtype)
        model.state_order.data[0] = 1

        # BOS state
        model.backoff_weights.data[1] = 0.0
        model.final_weights.data[1] = unigram_weight
        model.backoff_to_states.data[1] = 0  # to start state
        model.start_end_arcs.data[1, :] = torch.tensor(
            [vocab_size, vocab_size], dtype=model.start_end_arcs.dtype
        )  # no arcs
        model.state_order.data[1] = 2

        # all tokens - unigrams from start to start state (cycles)
        model.arcs_weights.data.fill_(unigram_weight)
        model.from_states.data.fill_(model.START_STATE)
        model.to_states.data.fill_(model.START_STATE)
        model.ilabels.data[:vocab_size].copy_(torch.arange(vocab_size, dtype=model.ilabels.dtype))

        model._resolve_final()
        return model

    @classmethod
    def from_suffix_tree(
        cls, suffix_tree_np: SuffixTreeStorage, use_triton: bool | None = None
    ) -> "NGramGPULanguageModel":
        """
        Constructor from suffix tree storage.

        Args:
            suffix_tree_np: suffix tree
            use_triton: allow using Triton implementation;
                None (default) means "auto" (used if available), True means forced mode
                (will crash if Triton is unavailable)

        Returns:
            NGramGPULanguageModel instance
        """
        model = NGramGPULanguageModel(
            OmegaConf.structured(
                NGramLMConfig(
                    num_states=suffix_tree_np.num_states,
                    num_arcs=suffix_tree_np.num_arcs,
                    max_order=suffix_tree_np.max_order,
                    vocab_size=suffix_tree_np.vocab_size,
                    use_triton=use_triton,
                )
            )
        )
        model._init_from_suffix_tree_np(suffix_tree_np=suffix_tree_np)
        model._resolve_final()
        return model

    @classmethod
    def _read_header(cls, f) -> dict[int, int]:
        """
        Parse ARPA header

        Args:
            f: file object

        Returns:
            dictionary with order -> number of ngrams
        """
        is_start = True
        order2cnt: dict[int, int] = defaultdict(int)
        for line in f:
            line = line.strip()
            if is_start:
                assert line == "\\data\\"
                is_start = False
                continue

            if line.startswith("ngram"):
                ngram_order, cnt = line.split("=")
                order = int(ngram_order.split()[-1])
                cnt = int(cnt)
                order2cnt[order] = cnt
                continue
            else:
                assert not line, "empty line expected after header"
                break
        return order2cnt

    @classmethod
    def _read_ngrams(cls, f, token_offset: int) -> Iterator[NGram]:
        special_words_pattern = '|'.join(re.escape(symbol) for symbol in _SPECIAL_SYMBOLS_MAP)
        pattern = re.compile(rf'({special_words_pattern}|.)\s?')
        for line in f:
            if line.endswith("\n"):
                line = line[:-1]

            if not line:
                continue

            if line.startswith("\\end\\"):
                break

            if line.startswith("\\"):
                continue

            ngram = cls._line_to_ngram(line=line, pattern=pattern, token_offset=token_offset)
            yield ngram

    @staticmethod
    def _line_to_ngram(line: str, pattern: re.Pattern, token_offset: int) -> NGram:
        """Parse ARPA line to N-Gram structure"""
        weight, symbols_str, *backoff_opt = line.split("\t")
        if backoff_opt:
            assert len(backoff_opt) == 1
            backoff = _log_10_to_e(float(backoff_opt[0]))
        else:
            backoff = 0.0
        weight = _log_10_to_e(float(weight))
        symbols_re = pattern.findall(symbols_str)

        symbols = tuple(
            (ord(symbol) - token_offset if symbol not in _SPECIAL_SYMBOLS_MAP else _SPECIAL_SYMBOLS_MAP[symbol])
            for symbol in symbols_re
        )
        return NGram(symbols=symbols, weight=weight, backoff=backoff)

    def _init_from_suffix_tree_np(self, suffix_tree_np: SuffixTreeStorage):
        """Helper function to init params from suffix tree params"""
        # parameters: weights
        self.arcs_weights.data.copy_(torch.from_numpy(suffix_tree_np.arcs["weight"][: self.num_arcs_extended]))
        self.backoff_weights.data.copy_(torch.from_numpy(suffix_tree_np.states["backoff_w"][: self.num_states]))
        self.final_weights.data.copy_(torch.from_numpy(suffix_tree_np.states["final"][: self.num_states]))

        # buffers: LM (suffix tree) structure
        self.from_states.data.copy_(torch.from_numpy(suffix_tree_np.arcs["from"][: self.num_arcs_extended]))
        self.to_states.data.copy_(torch.from_numpy(suffix_tree_np.arcs["to"][: self.num_arcs_extended]))
        self.ilabels.data.copy_(torch.from_numpy(suffix_tree_np.arcs["ilabel"][: self.num_arcs_extended]))
        self.backoff_to_states.data.copy_(torch.from_numpy(suffix_tree_np.states["backoff_to"][: self.num_states]))

        self.start_end_arcs.data[:, 0].copy_(torch.from_numpy(suffix_tree_np.states["arcs_start"][: self.num_states]))
        self.start_end_arcs.data[:, 1].copy_(torch.from_numpy(suffix_tree_np.states["arcs_end"][: self.num_states]))
        self.state_order.data.copy_(torch.from_numpy(suffix_tree_np.states["order"][: self.num_states]))

        # sanity check
        assert self.state_order.min().item() == 1
        assert self.state_order.max().item() <= self.max_order

    def get_init_states(self, batch_size: int, bos=True) -> torch.Tensor:
        """
        Get batch of the initial states

        Args:
            batch_size: batch size
            bos: use begin-of-sentence state

        Returns:
            tensor [B] of initial states
        """
        device = self.arcs_weights.device
        return torch.full(
            [batch_size], fill_value=self.bos_state if bos else self.START_STATE, device=device, dtype=torch.long
        )

    def forward(
        self,
        labels: torch.Tensor,
        labels_lengths: Optional[torch.Tensor] = None,
        bos: bool = True,
        eos: bool = False,
    ) -> torch.Tensor:
        """
        Compute log-probabilities for all labels in utterances using N-Gram LM.

        Args:
            labels: label sequences [B x L] if eos=False, [B x (L+1)] if eos=True
            labels_lengths (optional): lengths of the label sequences
            bos: start with BOS symbol
            eos: add EOS score after the sentence

        Returns:
            Tensor [B x L] with scores for each label in the utterance
        """
        return self.score_sentences(labels=labels, labels_lengths=labels_lengths, bos=bos, eos=eos)

    def score_sentences(
        self,
        labels: torch.Tensor,
        labels_lengths: Optional[torch.Tensor] = None,
        bos: bool = True,
        eos: bool = False,
    ) -> torch.Tensor:
        """
        Compute log-probabilities for all labels in utterances using N-Gram LM.

        Args:
            labels: label sequences [B x L] if eos=False, [B x (L+1)] if eos=True
            labels_lengths (optional): lengths of the label sequences
            bos: start with BOS symbol
            eos: add EOS score after the sentence

        Returns:
            Tensor [B x (L + 1) if eos else B x L] with scores for each label in the utterance
        """
        device = labels.device
        batch_size, max_length = labels.shape
        if labels_lengths is None:
            labels_lengths = torch.full([batch_size], fill_value=max_length, dtype=torch.int32, device=device)
        batch_size, max_length = labels.shape
        scores = torch.zeros([batch_size, max_length + (1 if eos else 0)], device=device)
        states = self.get_init_states(batch_size=batch_size, bos=bos)
        # NB: It is possible to speedup this algorithm with a custom kernel (no need to retrieve all weights/labels)
        for i in range(max_length):
            # NB: _advance_triton is not differentiable (need to implement backward manually);
            # for training _advance_pytorch only can be used
            prev_states = states
            step_scores, states = self._advance_pytorch(states)
            scores[:, i] = step_scores.gather(dim=1, index=labels[:, i].unsqueeze(-1)).squeeze(-1) * (
                i < labels_lengths
            )
            # get next states, preserve last state if the utterance ended
            states = torch.where(
                i < labels_lengths, states.gather(dim=1, index=labels[:, i].unsqueeze(-1)).squeeze(-1), prev_states
            )
        if eos:
            final_weights = self.get_final(states)
            scores.scatter_(dim=1, index=labels_lengths.unsqueeze(-1).to(torch.int64), src=final_weights.unsqueeze(-1))
        return scores

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

        # replace weight corresponding to eos_id with final state weight
        if eos_id is not None:
            scores[:, eos_id] = self.get_final(states=states)
            next_states[:, eos_id] = states
        return scores, next_states

    def _advance_pytorch(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Advance `states` [B]: return scores [B, V] and next states [B, V] for full vocab.
        PyTorch implementation (slow, differentiable).

        Args:
            states: batch of states

        Returns:
            tuple of scores and next states
        """
        batch_size = states.shape[0]
        device = states.device
        current_states = states.clone()
        states_dtype = current_states.dtype

        # init output tensors
        out_scores = torch.zeros(batch_size, self.vocab_size, device=device)
        out_states = torch.full([batch_size, self.vocab_size], fill_value=-1, dtype=states_dtype, device=device)

        # helper ranges
        vocab_range = torch.arange(self.vocab_size, device=device)
        batch_indices = torch.arange(batch_size, device=device)

        # backoff weight accumulator
        accumulated_backoff = torch.zeros(batch_size, device=device)
        # loop condition
        start_state_not_processed = torch.full([batch_size], fill_value=True, dtype=torch.bool, device=device)

        num_iterations = 0
        while start_state_not_processed.any():
            assert num_iterations <= self.max_order, "Infinite loop in LM advance"
            num_iterations += 1
            # get arc boundaries
            start, end = self.start_end_arcs[current_states].unbind(dim=1)
            # number of arcs for each state cannot be larger than vocab size
            indices = start[:, None] + vocab_range[None, :]
            mask = indices < end[:, None]
            mask &= start_state_not_processed[:, None]
            mask_flat = mask.view(-1)
            indices_flat = indices.view(-1)
            # map indices outside the mask to vocab_size + 1
            scores_add = torch.zeros([batch_size, self.vocab_size + 1], device=device, dtype=out_scores.dtype)
            out_states_add = torch.full(
                [batch_size, self.vocab_size + 1], fill_value=-1, device=device, dtype=states_dtype
            )
            ilabels = self.ilabels[indices_flat] * mask_flat + ~mask_flat * self.vocab_size
            scores_add[batch_indices.repeat_interleave(self.vocab_size), ilabels] = self.arcs_weights[indices_flat]
            out_states_add[batch_indices.repeat_interleave(self.vocab_size), ilabels] = self.to_states[
                indices_flat
            ].to(states_dtype)
            # fill out_scores and out_states with new values where state is not found yet
            state_found = out_states != -1
            out_scores = torch.where(
                state_found, out_scores, accumulated_backoff.unsqueeze(-1) + scores_add[:, : self.vocab_size]
            )
            out_states = torch.where(state_found, out_states, out_states_add[:, : self.vocab_size])
            # update loop condition; process backoffs
            start_state_not_processed &= current_states != self.START_STATE
            accumulated_backoff += self.backoff_weights[current_states] * start_state_not_processed
            torch.where(
                start_state_not_processed, self.backoff_to_states[current_states], current_states, out=current_states
            )
        return out_scores, out_states

    @triton_required
    def _advance_triton(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Advance `states` [B]: return scores [B, V] and next states [B, V] for full vocab.
        Triton implementation. Currently not differentiable.

        Args:
            states: batch of states

        Returns:
            tuple of scores and next states
        """
        batch_size = states.shape[0]
        device = states.device
        scores = torch.empty([batch_size, self.vocab_size], device=device, dtype=self.arcs_weights.dtype)
        new_states = torch.empty([batch_size, self.vocab_size], dtype=torch.long, device=device)

        ngram_advance_triton_kernel[batch_size,](
            vocab_size=self.vocab_size,
            states_ptr=states,
            new_states_ptr=new_states,
            scores_ptr=scores,
            start_state=self.START_STATE,
            to_states_ptr=self.to_states,
            ilabels_ptr=self.ilabels,
            arcs_weights_ptr=self.arcs_weights,
            start_end_arcs_ptr=self.start_end_arcs,
            backoff_to_states_ptr=self.backoff_to_states,
            backoff_weights_ptr=self.backoff_weights,
            BLOCK_SIZE=triton.next_power_of_2(self.vocab_size),
        )

        return scores, new_states

    def get_final(self, states: torch.Tensor) -> torch.Tensor:
        """
        Get final weights for states

        Args:
            states: batch of states

        Returns:
            tensor [B] with final weights for each state
        """
        if self._final_resolved:
            return self.final_weights[states]
        logging.warning("Final weights are not resolved; using slow implementation")
        return self._get_final_pytorch(states=states)

    def _resolve_final(self):
        """Resolve final weights for all states by iterating over backoffs"""
        if self._final_resolved:
            return
        with torch.no_grad():
            self.final_weights.data.copy_(
                self._get_final_pytorch(states=torch.arange(self.num_states, device=self.final_weights.device))
            )
        self._final_resolved = True

    def _get_final_pytorch(self, states: torch.Tensor) -> torch.Tensor:
        """
        Get final weights for states, resolving backoffs

        Args:
            states: batch of states

        Returns:
            batch of final weights
        """
        cur_states = states.clone().detach()
        out_scores = self.final_weights[cur_states]
        accumulated_backoff = torch.zeros_like(out_scores)
        while (out_scores <= NEG_INF).any() and (cur_states != self.START_STATE).any():
            accumulated_backoff += self.backoff_weights[cur_states]
            cur_states = self.backoff_to_states[cur_states]
            cur_final = self.final_weights[cur_states]
            out_scores = torch.where(
                (out_scores > NEG_INF) | (cur_final <= NEG_INF), out_scores, accumulated_backoff + cur_final
            )
        return out_scores
