# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
import re
import tempfile
from abc import ABC, abstractmethod, abstractproperty
from collections import defaultdict, namedtuple
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

from nemo.utils import logging


TW_BREAK = "‡"


try:
    import kaldifst

    # check that kaldifst package is not empty
    # Note: pytorch_lightning.utilities.imports.package_available may not help here
    kaldifst.StdVectorFst()
    _KALDIFST_AVAILABLE = True
except (ImportError, ModuleNotFoundError, AttributeError):
    _KALDIFST_AVAILABLE = False


try:
    import graphviz

    _GRAPHVIZ_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    _GRAPHVIZ_AVAILABLE = False


try:
    import kaldilm

    _KALDILM_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    _KALDILM_AVAILABLE = False


KALDIFST_INSTALLATION_MESSAGE = (
    "kaldifst is not installed or is installed incorrectly.\n"
    "please run `pip install kaldifst` or `bash scripts/installers/install_riva_decoder.sh` to install."
)


GRAPHVIZ_INSTALLATION_MESSAGE = (
    "graphviz is not installed.\n" "please run `bash scripts/installers/install_graphviz.sh` to install."
)


KALDILM_INSTALLATION_MESSAGE = (
    "kaldilm is not installed.\n"
    "please run `pip install kaldilm` or `bash scripts/installers/install_riva_decoder.sh` to install."
)


def _kaldifst_maybe_raise():
    if _KALDIFST_AVAILABLE is False:
        raise ImportError(KALDIFST_INSTALLATION_MESSAGE)


def kaldifst_importer():
    """Import helper function that returns kaldifst package or raises ImportError exception."""
    _kaldifst_maybe_raise()
    return kaldifst


def _graphviz_maybe_raise():
    if _GRAPHVIZ_AVAILABLE is False:
        raise ImportError(GRAPHVIZ_INSTALLATION_MESSAGE)


def graphviz_importer():
    """Import helper function that returns graphviz package or raises ImportError exception."""
    _graphviz_maybe_raise()
    return graphviz


def _kaldilm_maybe_raise():
    if _KALDILM_AVAILABLE is False:
        raise ImportError(KALDILM_INSTALLATION_MESSAGE)


def kaldilm_importer():
    """Import helper function that returns kaldifst package or raises ImportError exception."""
    _kaldilm_maybe_raise()
    return kaldilm


@dataclass
class LexiconUnit:
    """A dataclass encapsulating the name of the language unit (e.g. wordpiece) and its mark (e.g. word begin)."""

    name: str
    mark: str = ""


class Lexicon:
    def __init__(
        self,
        wordid2tokenid: Dict[int, List[List[int]]],
        id2word: Union[Dict[int, str], Dict[int, LexiconUnit]],
        id2token: Union[Dict[int, str], Dict[int, LexiconUnit]],
        disambig_pattern: str = re.compile(r"^#\d+$"),
    ):
        """
        Lexicon class which contains word-to-token-sequence, word-to-id, and token-to-id mappings.

        Args:
          wordid2tokenid:
            Lexicon.
            Mapping from word_id to token1_id token2_id ... tokenN_id.

          id2word:
            Word index.
            Mapping from word_id to word_str.

          id2token:
            Token index.
            Mapping from token_id to token_str.

          disambig_pattern:
            Pattern for disambiguation symbols.
        """
        is_id2token_str = not isinstance(list(id2token.values())[0], LexiconUnit)
        self.id2token = {k: LexiconUnit(v) for k, v in id2token.items()} if is_id2token_str else id2token
        self.token2id = {v.name: k for k, v in self.id2token.items()}
        is_id2word_str = not isinstance(list(id2word.values())[0], LexiconUnit)
        self.id2word = {k: LexiconUnit(v) for k, v in id2word.items()} if is_id2word_str else id2word
        self.word2id = {v.name: k for k, v in self.id2word.items()}
        self.wordid2tokenid = wordid2tokenid
        word2tokens = defaultdict(list)
        for k, v in self.wordid2tokenid.items():
            word2tokens[self.id2word[k].name] += [[self.id2token[i].name for i in vp] for vp in v]
        self.word2tokens = word2tokens
        self.disambig_pattern = disambig_pattern

        max_disambig_id = -1
        num_disambigs = 0
        self.has_epsilon = False
        self._default_disambig_mark = "disambig"
        self._default_epsilon_mark = "epsilon"
        self._default_epsilon_name = "<eps>"
        for i, s in self.id2token.items():
            if self.disambig_pattern.match(s.name):
                if is_id2token_str or not s.mark.startswith(self._default_disambig_mark):
                    s.mark = self._default_disambig_mark
                if i > max_disambig_id:
                    max_disambig_id = i
                    num_disambigs += 1
            if s.name == self._default_epsilon_name or s.mark == self._default_epsilon_mark:
                assert i == 0
                self.has_epsilon = True
        self.max_disambig_id = max_disambig_id
        self.num_disambigs = num_disambigs

        if is_id2word_str:
            for i, s in self.id2word.items():
                if self.disambig_pattern.match(s.name):
                    s.mark = self._default_disambig_mark
                elif s.name == self._default_epsilon_name:
                    s.mark == self._default_epsilon_mark

    def __iter__(self) -> Tuple[str, List[str]]:
        for wordid, tokenid_list in self.wordid2tokenid.items():
            for tokenids in tokenid_list:
                yield wordid, tokenids

    def __str__(self):
        return str(self.word2tokens)

    @property
    def token_ids(self) -> List[int]:
        """Return a list of token IDs excluding those from
        disambiguation symbols.
        """
        ans = []
        for i, s in self.id2token.items():
            if not s.mark.startswith(self._default_epsilon_mark) and (not self.has_epsilon or i != 0):
                ans.append(i)
        ans.sort()
        return ans


def arpa2fst(lm_path: str, attach_symbol_table: bool = True) -> 'kaldifst.StdVectorFst':
    """
    Compiles an ARPA LM file into a grammar WFST (G.fst).

    Args:
      lm_path:
        Path to the ARPA LM file.

      attach_symbol_table:
        Whether to attach the words for indices of the returned WFST.

    Returns:
      Kaldi-type grammar WFST.
    """
    _kaldifst_maybe_raise()
    _kaldilm_maybe_raise()

    with tempfile.TemporaryDirectory() as tempdirname:
        output_fst = os.path.join(tempdirname, "output.fst")
        words_txt = os.path.join(tempdirname, "words.txt")
        #         with suppress_stdout_stderr():
        kaldilm.arpa2fst(
            input_arpa=lm_path,
            output_fst=output_fst,
            disambig_symbol="#0",
            write_symbol_table=words_txt,
        )

        G = kaldifst.StdVectorFst.read(output_fst)

        if attach_symbol_table:
            osym = kaldifst.SymbolTable()
            with open(words_txt, encoding="utf-8") as f:
                for line in f:
                    w, i = line.strip().split()
                    osym.add_symbol(symbol=w, key=int(i))
            G.output_symbols = osym

    kaldifst.arcsort(G, sort_type="ilabel")
    return G


def add_tokenwords_(
    g_fst: 'kaldifst.StdVectorFst',
    tokens: List[str],
    word_weight: float = 2.0,
    token_unigram_weight: float = 4.0,
    token_oov: str = "<unk>",
) -> int:
    """
    Adds special words representing individual tokens (tokenwords).
    In-place operation.

    Args:
      g_fst:
        Kaldi-type grammar WFST.
        Will be augmented with the tokenwords.

      tokens:
        Token vocabulary.

      word_weight:
        The weight of an Out Of Vocabulary (OOV) word emission.

      token_unigram_weight:
        The weight of a tokenword emission.

      token_oov:
        OOV token.

    Returns:
        The id of the tokenword disambiguation token.
    """
    _kaldifst_maybe_raise()

    unigram_state = 0
    # check if 0 is the unigram state (has no outgoing epsilon arcs)
    assert kaldifst.ArcIterator(g_fst, unigram_state).value.ilabel not in (0, g_fst.output_symbols.find("#0"))

    # we put tokenword self-loops in a separate state wrapped with a tokenword_disambig token
    tokenword_disambig_id = g_fst.output_symbols.available_key()
    tokenword_disambig = "#1"
    g_fst.output_symbols.add_symbol(tokenword_disambig, tokenword_disambig_id)
    tokenword_state = g_fst.add_state()
    # we keep olabel !=0 to mark tokenword segments in the recognition results
    g_fst.add_arc(
        state=unigram_state,
        arc=kaldifst.StdArc(
            ilabel=tokenword_disambig_id,
            olabel=tokenword_disambig_id,
            weight=word_weight,
            nextstate=tokenword_state,
        ),
    )
    g_fst.add_arc(
        state=tokenword_state,
        arc=kaldifst.StdArc(
            ilabel=tokenword_disambig_id,
            olabel=tokenword_disambig_id,
            weight=0.0,
            nextstate=unigram_state,
        ),
    )
    label = tokenword_disambig_id + 1
    for t in tokens:
        if t != token_oov:
            g_fst.add_arc(
                state=tokenword_state,
                arc=kaldifst.StdArc(
                    ilabel=label,
                    olabel=label,
                    weight=token_unigram_weight,
                    nextstate=tokenword_state,
                ),
            )
            g_fst.output_symbols.add_symbol(f"{t}{TW_BREAK}{tokenword_disambig}", label)
            label += 1

    return tokenword_disambig_id


def generate_lexicon_sentencepiece(
    tokenizer: 'TokenizerSpec',
    id2word: Dict[int, str],
    oov: str = "<unk>",
    add_epsilon: bool = False,
    first_tokenword_id: int = -1,
    disambig_pattern: str = re.compile(r"^#\d+$"),
) -> Lexicon:
    """
    Generate a Lexicon using a SentencePiece tokenizer.

    Args:
      tokenizer:
        NeMo SentencePiece tokenizer.

      id2word:
        Word index.
        Mapping from word_id to word_str.

      oov:
        Out Of Vocabulary word in lexicon.

    Returns:
      Lexicon object.
    """
    word2id = {v: k for k, v in id2word.items()}
    backoff_disambig = "#0"
    tokenword_disambig = "#1"
    word_begin_mark = "▁"

    tokenword_mode = first_tokenword_id != -1
    if tokenword_mode:
        words, tokenwords = [], []
        for k, v in id2word.items():
            if disambig_pattern.match(v):
                continue
            words.append(v) if k < first_tokenword_id else tokenwords.append(v)
    else:
        words, tokenwords = [v for v in id2word.values() if not disambig_pattern.match(v)], []

    # Use encode to avoid OOV tokens
    words_piece_ids = tokenizer.encode(words, out_type=int)

    # tokenizer.get_vocab() gives indices starting with 1
    maybe_add_one = int(add_epsilon)
    maybe_subtract_one = int(not add_epsilon)
    vocab = tokenizer.get_vocab()
    id2token = {
        v - maybe_subtract_one: LexiconUnit(k, "begin" if k.startswith(word_begin_mark) else "")
        for k, v in vocab.items()
    }

    # Introduce unk, blank, and the first disambig ids
    unk_id = tokenizer.piece_to_id(oov) + maybe_add_one
    id2token[unk_id] = LexiconUnit(oov, "unk")
    # We assume blank to have the last output id of the neural network output
    max_token_id = max(id2token.keys())
    id2token[max_token_id + 1] = LexiconUnit("<blk>", "blank")
    id2token[max_token_id + 2] = LexiconUnit(backoff_disambig, "disambig_backoff")
    if tokenword_mode:
        id2token[max_token_id + 3] = LexiconUnit(tokenword_disambig, "disambig_tokenword")
    if add_epsilon:
        # insert first
        id2token[0] = LexiconUnit("<eps>", "epsilon")
        id2token = {k: v for k, v in sorted(id2token.items(), key=lambda item: item[0])}

    if tokenword_mode:
        words += tokenwords
        words_piece_ids += [[vocab[tw.rstrip(f"{TW_BREAK}{tokenword_disambig}")] - maybe_add_one] for tw in tokenwords]

    wordid2tokenid = defaultdict(list)

    for word, piece_ids in zip(words, words_piece_ids):
        if word.startswith("<") and word != "<eps>":  # not a real word, probably some tag
            continue
        elif word == "<eps>":  # we do not need to tokelize <eps>
            continue
        else:
            wordid2tokenid[word2id[word]].append([p + maybe_add_one for p in piece_ids])

    lexicon = Lexicon(wordid2tokenid, id2word, id2token)
    # state disambig purpose explicitly for further use
    lexicon.id2word[lexicon.word2id[backoff_disambig]].mark = "disambig_backoff"
    if tokenword_mode:
        lexicon.id2word[lexicon.word2id[tokenword_disambig]].mark = "disambig_tokenword"
        for tw in tokenwords:
            lexicon.id2word[lexicon.word2id[tw]].mark = "tokenword"
    return lexicon


def add_disambig_symbols(lexicon: Lexicon) -> Lexicon:
    """
    Adds pseudo-token disambiguation symbols #1, #2 and so on
    at the ends of tokens to ensure that all pronunciations are different,
    and that none is a prefix of another.

    See also add_lex_disambig.pl from kaldi.

    Args:
      lexicon:
        Lexicon object.

    Returns:
      Return Lexicon augmented with subseqence disambiguation symbols.
    """

    tokenword_mode = "#1" in lexicon.word2id
    if tokenword_mode:
        first_tokenword_id = lexicon.word2id["#1"] + 1
        last_used_disambig_id = lexicon.token2id["#1"]
    else:
        last_used_disambig_id = lexicon.token2id["#0"]

    # (1) Work out the count of each token-sequence in the lexicon.
    count = defaultdict(int)
    for _, token_ids in lexicon:
        count[tuple(token_ids)] += 1

    # (2) For each left sub-sequence of each token-sequence, note down
    # that it exists (for identifying prefixes of longer strings).
    issubseq = defaultdict(int)
    for word_id, token_ids in lexicon:
        if tokenword_mode and word_id >= first_tokenword_id:
            continue
        token_ids = token_ids.copy()
        token_ids.pop()
        while token_ids:
            issubseq[tuple(token_ids)] = 1
            token_ids.pop()

    # (3) For each entry in the lexicon:
    # if the token sequence is unique and is not a
    # prefix of another word, no disambig symbol.
    # Else output #1, or #2, #3, ... if the same token-seq
    # has already been assigned a disambig symbol.
    wordid2tokenid = defaultdict(list)
    id2token = lexicon.id2token.copy()

    first_allowed_disambig = lexicon.num_disambigs
    first_allowed_disambig_id = last_used_disambig_id + 1
    max_disambig = first_allowed_disambig - 1
    last_used_disambig_id_of = defaultdict(int)

    for word_id, token_ids in lexicon:
        token_key = tuple(token_ids)
        assert len(token_key) > 0
        if issubseq[token_key] == 0 and count[token_key] == 1 or tokenword_mode and word_id >= first_tokenword_id:
            wordid2tokenid[word_id].append(token_ids)
            continue

        cur_disambig_id = last_used_disambig_id_of[token_key]
        if cur_disambig_id == 0:
            cur_disambig = first_allowed_disambig
            cur_disambig_id = first_allowed_disambig_id
        else:
            cur_disambig = int(id2token[cur_disambig_id].name.lstrip("#")) + 1

        if cur_disambig > max_disambig:
            max_disambig = cur_disambig
            cur_disambig_id = max(id2token.keys()) + 1
            id2token[cur_disambig_id] = LexiconUnit(f"#{max_disambig}", "disambig_subsequence")
        last_used_disambig_id_of[token_key] = cur_disambig_id
        wordid2tokenid[word_id].append(token_ids + [cur_disambig_id])
    return Lexicon(wordid2tokenid, lexicon.id2word, id2token)


def make_lexicon_fst_no_silence(
    lexicon: Lexicon,
    attach_symbol_table: bool = True,
) -> 'kaldifst.StdVectorFst':
    """
    Compiles a Lexicon into a lexicon WFST (L.fst).

    See also make_lexicon_fst.py from kaldi.

    Args:
      lexicon:
        Lexicon object.

    Returns:
      Kaldi-type lexicon WFST.
    """
    _kaldifst_maybe_raise()

    backoff_disambig = "#0"
    tokenword_disambig = "#1"
    tokenword_mode = tokenword_disambig in lexicon.word2id
    if tokenword_mode:
        first_tokenword_id = lexicon.word2id[tokenword_disambig] + 1

    fst = kaldifst.StdVectorFst()
    start_state = fst.add_state()
    fst.start = start_state
    fst.set_final(state=start_state, weight=0)
    fst.add_arc(
        state=start_state,
        arc=kaldifst.StdArc(
            ilabel=lexicon.token2id[backoff_disambig],
            olabel=lexicon.word2id[backoff_disambig],
            weight=0,
            nextstate=start_state,
        ),
    )
    if tokenword_mode:
        tokenword_state_begin = fst.add_state()
        fst.add_arc(
            state=start_state,
            arc=kaldifst.StdArc(
                ilabel=lexicon.token2id[tokenword_disambig],
                olabel=lexicon.word2id[tokenword_disambig],
                weight=0,
                nextstate=tokenword_state_begin,
            ),
        )

    for word_id, token_ids in lexicon:
        cur_state = start_state

        if not tokenword_mode or word_id < first_tokenword_id - 1:
            for i, token_id in enumerate(token_ids[:-1]):
                next_state = fst.add_state()
                fst.add_arc(
                    state=cur_state,
                    arc=kaldifst.StdArc(
                        ilabel=token_id,
                        olabel=word_id if i == 0 else 0,
                        weight=0,
                        nextstate=next_state,
                    ),
                )
                cur_state = next_state
            i = len(token_ids) - 1  # note: i == -1 if tokens is empty.
            fst.add_arc(
                state=cur_state,
                arc=kaldifst.StdArc(
                    ilabel=token_ids[-1] if i >= 0 else 0,
                    olabel=word_id if i <= 0 else 0,
                    weight=0,
                    nextstate=start_state,
                ),
            )
    if tokenword_mode:
        tokenword_begin, tokenword_other = [], []
        for word_id in range(first_tokenword_id, max(lexicon.id2word) + 1):
            token_id = lexicon.token2id[lexicon.id2word[word_id].name.rstrip(f"{TW_BREAK}{tokenword_disambig}")]
            token_unit = lexicon.id2token[token_id]
            if token_unit.mark.startswith("begin"):
                tokenword_begin.append((token_id, word_id))
            elif token_unit.mark == "":
                tokenword_other.append((token_id, word_id))
            else:
                raise RuntimeError(f"Unexpected mark `{token_unit.mark}` for tokenword `{token_unit.name}`")

        tokenword_state_main = fst.add_state()
        for token_id, word_id in tokenword_begin:
            fst.add_arc(
                state=tokenword_state_begin,
                arc=kaldifst.StdArc(
                    ilabel=token_id,
                    olabel=word_id,
                    weight=0,
                    nextstate=tokenword_state_main,
                ),
            )
        tokenword_state_end = fst.add_state()
        for token_id, word_id in tokenword_other:
            fst.add_arc(
                state=tokenword_state_main,
                arc=kaldifst.StdArc(
                    ilabel=token_id,
                    olabel=word_id,
                    weight=0,
                    nextstate=tokenword_state_main,
                ),
            )
            fst.add_arc(
                state=tokenword_state_main,
                arc=kaldifst.StdArc(
                    ilabel=token_id,
                    olabel=word_id,
                    weight=0,
                    nextstate=tokenword_state_end,
                ),
            )
        fst.add_arc(
            state=tokenword_state_end,
            arc=kaldifst.StdArc(
                ilabel=lexicon.token2id[tokenword_disambig],
                olabel=lexicon.word2id[tokenword_disambig],
                weight=0,
                nextstate=start_state,
            ),
        )

    if attach_symbol_table:
        isym = kaldifst.SymbolTable()
        for p, i in lexicon.token2id.items():
            isym.add_symbol(symbol=p, key=i)
        fst.input_symbols = isym

        osym = kaldifst.SymbolTable()
        for w, i in lexicon.word2id.items():
            osym.add_symbol(symbol=w, key=i)
        fst.output_symbols = osym

    kaldifst.arcsort(fst, sort_type="ilabel")
    return fst


def build_topo(
    name: str, token2id: Dict[str, int], with_self_loops: bool = True, attach_symbol_table: bool = True
) -> 'kaldifst.StdVectorFst':
    """Helper function to build a topology WFST (T.fst).

    Args:
      name:
        Topology name. Choices: default, compact, minimal

      token2id:
        Token index.
        Mapping from token_str to token_id.

      with_self_loops:
        Whether to add token-to-epsilon self-loops to the topology.

      attach_symbol_table:
        Whether to attach the token names for indices of the returned WFST.

    Returns:
      Kaldi-type topology WFST.
    """
    _kaldifst_maybe_raise()

    if name == "default":
        fst = build_default_topo(token2id, with_self_loops)
    elif name == "compact":
        fst = build_compact_topo(token2id, with_self_loops)
    elif name == "minimal":
        fst = build_minimal_topo(token2id)
    else:
        raise ValueError(f"Unknown topo name: {name}")

    if attach_symbol_table:
        isym = kaldifst.SymbolTable()
        for t, i in token2id.items():
            isym.add_symbol(symbol=t, key=i)
        fst.input_symbols = isym
        fst.output_symbols = fst.input_symbols.copy()
    return fst


def build_default_topo(token2id: Dict[str, int], with_self_loops: bool = True) -> 'kaldifst.StdVectorFst':
    """Build the default (correct) CTC topology."""
    _kaldifst_maybe_raise()

    disambig_pattern = re.compile(r"^#\d+$")
    blank_id = token2id["<blk>"]
    fst = kaldifst.StdVectorFst()
    start_state = fst.add_state()
    fst.start = start_state
    fst.set_final(state=start_state, weight=0)
    fst.add_arc(
        state=start_state,
        arc=kaldifst.StdArc(
            ilabel=blank_id,
            olabel=0,
            weight=0,
            nextstate=start_state,  # token2id["<eps>"] is always 0
        ),
    )

    disambig_ids = []
    token_ids = {}
    for s, i in token2id.items():
        if s == "<eps>" or s == "<blk>":
            continue
        elif disambig_pattern.match(s):
            disambig_ids.append(i)
        else:
            state = fst.add_state()
            fst.set_final(state=state, weight=0)
            token_ids[state] = i
            fst.add_arc(
                state=start_state,
                arc=kaldifst.StdArc(
                    ilabel=i,
                    olabel=i,
                    weight=0,
                    nextstate=state,
                ),
            )
            if with_self_loops:
                fst.add_arc(
                    state=state,
                    arc=kaldifst.StdArc(
                        ilabel=i,
                        olabel=0,
                        weight=0,
                        nextstate=state,  # token2id["<eps>"] is always 0
                    ),
                )
            fst.add_arc(
                state=state,
                arc=kaldifst.StdArc(
                    ilabel=blank_id,
                    olabel=0,
                    weight=0,
                    nextstate=start_state,  # token2id["<eps>"] is always 0
                ),
            )

    for istate in kaldifst.StateIterator(fst):
        if istate > 0:
            for ostate in kaldifst.StateIterator(fst):
                if ostate > 0 and istate != ostate:
                    label = token_ids[ostate]
                    fst.add_arc(
                        state=istate,
                        arc=kaldifst.StdArc(
                            ilabel=label,
                            olabel=label,
                            weight=0,
                            nextstate=ostate,
                        ),
                    )
        for disambig_id in disambig_ids:
            fst.add_arc(
                state=istate,
                arc=kaldifst.StdArc(
                    ilabel=0,
                    olabel=disambig_id,
                    weight=0,
                    nextstate=istate,  # token2id["<eps>"] is always 0
                ),
            )

    return fst


def build_compact_topo(token2id: Dict[str, int], with_self_loops: bool = True) -> 'kaldifst.StdVectorFst':
    """Build the Compact CTC topology."""
    _kaldifst_maybe_raise()

    disambig_pattern = re.compile(r"^#\d+$")
    blank_id = token2id["<blk>"]
    fst = kaldifst.StdVectorFst()
    start_state = fst.add_state()
    fst.start = start_state
    fst.set_final(state=start_state, weight=0)
    fst.add_arc(
        state=start_state,
        arc=kaldifst.StdArc(
            ilabel=blank_id,
            olabel=0,
            weight=0,
            nextstate=start_state,  # token2id["<eps>"] is always 0
        ),
    )

    for s, i in token2id.items():
        if s == "<eps>" or s == "<blk>":
            continue
        elif disambig_pattern.match(s):
            fst.add_arc(
                state=start_state,
                arc=kaldifst.StdArc(
                    ilabel=0,
                    olabel=i,
                    weight=0,
                    nextstate=start_state,  # token2id["<eps>"] is always 0
                ),
            )
        else:
            state = fst.add_state()
            fst.add_arc(
                state=start_state,
                arc=kaldifst.StdArc(
                    ilabel=i,
                    olabel=i,
                    weight=0,
                    nextstate=state,
                ),
            )
            if with_self_loops:
                fst.add_arc(
                    state=state,
                    arc=kaldifst.StdArc(
                        ilabel=i,
                        olabel=0,
                        weight=0,
                        nextstate=state,  # token2id["<eps>"] is always 0
                    ),
                )
            fst.add_arc(
                state=state,
                arc=kaldifst.StdArc(
                    ilabel=0,  # token2id["<eps>"] is always 0
                    olabel=0,  # token2id["<eps>"] is always 0
                    weight=0,
                    nextstate=start_state,
                ),
            )

    return fst


def build_minimal_topo(token2id: Dict[str, int]) -> 'kaldifst.StdVectorFst':
    """Build the Minimal CTC topology."""
    _kaldifst_maybe_raise()

    disambig_pattern = re.compile(r"^#\d+$")
    blank_id = token2id["<blk>"]
    fst = kaldifst.StdVectorFst()
    start_state = fst.add_state()
    fst.start = start_state
    fst.set_final(state=start_state, weight=0)
    fst.add_arc(
        state=start_state,
        arc=kaldifst.StdArc(
            ilabel=blank_id,
            olabel=0,
            weight=0,
            nextstate=start_state,  # token2id["<eps>"] is always 0
        ),
    )

    for s, i in token2id.items():
        if s == "<eps>" or s == "<blk>":
            continue
        elif disambig_pattern.match(s):
            fst.add_arc(
                state=start_state,
                arc=kaldifst.StdArc(
                    ilabel=0,
                    olabel=i,
                    weight=0,
                    nextstate=start_state,  # token2id["<eps>"] is always 0
                ),
            )
        else:
            fst.add_arc(
                state=start_state,
                arc=kaldifst.StdArc(
                    ilabel=i,
                    olabel=i,
                    weight=0,
                    nextstate=start_state,
                ),
            )

    return fst


def mkgraph_ctc_ov(
    tokenizer: 'TokenizerSpec',
    lm_path: Union[Path, str],
    topology_name: str = "default",
    write_tlg_path: Optional[Union[Path, str]] = None,
    open_vocabulary: bool = False,
    open_vocabulary_weights: Tuple[float, float] = (2.0, 4.0),
    target: str = "kaldi",  # "kaldi", "k2"
) -> Tuple[Union['kaldifst.StdVectorFst', 'k2.Fsa'], int]:
    """
    Builds a decoding WFST (TLG.fst or TLG.pt).

    See also mkgraph.sh from kaldi.

    Args:
      tokenizer:
        NeMo SentencePiece tokenizer.

      lm_path:
        Path to the ARPA LM file.

      topology_name:
        Topology name. Choices: default, compact, minimal.

      write_tlg_path:
        Where to buffer the TLG.

      open_vocabulary:
        Whether to build a decoding WFST suitable for the open vocabulary decoding.

      open_vocabulary_weights:
        Pair of weights (oov_word_weight, token_unigram_weight).

      target:
        What type to build the WFST for. Choices: kaldi, k2.

    Returns:
      A pair of kaldi- or k2-type decoding WFST and its id of the tokenword disambiguation token.
    """
    _kaldifst_maybe_raise()

    logging.info("Compiling G.fst ...")
    G = arpa2fst(lm_path)
    if open_vocabulary:
        # in-place for g_fst
        tokenword_disambig_id = add_tokenwords_(
            g_fst=G,
            tokens=tokenizer.tokenizer.get_vocab().keys(),
            word_weight=open_vocabulary_weights[0],
            token_unigram_weight=open_vocabulary_weights[1],
        )
    else:
        tokenword_disambig_id = -1

    logging.info("Building L.fst ...")
    id2word = {int(line.split("\t")[1]): line.split("\t")[0] for line in str(G.output_symbols).strip().split("\n")}
    lexicon = generate_lexicon_sentencepiece(
        tokenizer.tokenizer, id2word, add_epsilon=True, first_tokenword_id=tokenword_disambig_id
    )
    lexicon_disambig = add_disambig_symbols(lexicon)

    L = make_lexicon_fst_no_silence(lexicon_disambig)
    kaldifst.arcsort(L, sort_type="olabel")

    logging.info("Building LG.fst ...")
    LG = kaldifst.compose(L, G)
    kaldifst.determinize_star(LG)
    kaldifst.minimize_encoded(LG)
    kaldifst.arcsort(LG, sort_type="ilabel")

    logging.info("Building TLG.fst ...")
    T = build_topo(topology_name, lexicon_disambig.token2id)
    kaldifst.arcsort(T, sort_type="olabel")
    TLG = kaldifst.compose(T, LG)

    if target == "kaldi":
        if write_tlg_path:
            logging.info(f"Buffering TLG.fst into {write_tlg_path} ...")
            TLG.write(write_tlg_path)
    elif target == "k2":
        logging.info("Converting TLG.fst to k2 ...")
        import torch

        from nemo.core.utils.k2_guard import k2

        blank_id = [i for i, t in lexicon_disambig.id2token.items() if t.mark == "blank"][0]
        first_token_disambig_id = [i for i, t in lexicon_disambig.id2token.items() if t.mark == "disambig_backoff"][0]
        word_disambig_id = lexicon_disambig.word2id[lexicon_disambig.id2token[first_token_disambig_id].name]
        assert lexicon_disambig.id2word[word_disambig_id].mark == "disambig_backoff"
        input_symbols = "\n".join(
            [f"{k} {v - 1}" for k, v in lexicon_disambig.token2id.items() if 0 < v < first_token_disambig_id]
        )
        output_symbols = str(TLG.output_symbols)
        TLG.input_symbols = None
        TLG.output_symbols = None
        # k2 does not support torch.inference_mode enabled
        with torch.inference_mode(False):
            TLG = k2.Fsa.from_openfst(TLG.to_str(show_weight_one=True), acceptor=False)
            TLG.labels[TLG.labels >= first_token_disambig_id] = blank_id
            TLG.aux_labels[TLG.aux_labels.values == word_disambig_id] = 0
            TLG.__dict__["_properties"] = None
            TLG = k2.arc_sort(k2.connect(k2.remove_epsilon(TLG)))
            TLG.labels[TLG.labels > 0] = TLG.labels[TLG.labels > 0] - 1
            TLG.__dict__["_properties"] = None
            TLG.labels_sym = k2.SymbolTable.from_str(input_symbols)
            TLG.aux_labels_sym = k2.SymbolTable.from_str(output_symbols)
            TLG = k2.arc_sort(TLG)
            if write_tlg_path:
                logging.info(f"Buffering TLG.pt into {write_tlg_path} ...")
                torch.save(TLG.as_dict(), write_tlg_path)
    else:
        raise ValueError(f"Unsupported target: `{target}`")

    return TLG, tokenword_disambig_id


class KaldiFstMask(Enum):
    Acceptor = 65536
    Error = 4
    TopSorted = 274877906944
    Acyclic = 34359738368
    IlabelSorted = 268435456
    OlabelSorted = 1073741824
    IlabelDeterministic = 262144
    OlabelDeterministic = 1048576
    HasEpsilons = 4194304
    HasIEpsilons = 16777216
    Accessible = 1099511627776
    Coaccessible = 4398046511104
    Weighted = 4294967296


class LatticeProperties(NamedTuple):
    Acceptor: bool
    Valid: bool
    Nonempty: bool
    TopSorted: bool
    Acyclic: bool
    ArcSorted: bool
    Deterministic: bool
    EpsilonFree: bool
    InputEpsilonFree: bool
    Connected: bool
    Weighted: bool


class AbstractLattice(ABC):
    """A lattice wrapper with high-level capabilities."""

    def __init__(self, lattice: Any):
        self._lattice = lattice
        self._properties = None

    @abstractmethod
    def as_tensor(self) -> 'torch.Tensor':
        """Represents the lattice as a tensor.

        Returns:
          torch.Tensor
        """
        pass

    @abstractmethod
    def draw(
        self, filename: Optional[Union[Path, str]] = None, title: Optional[Union[Path, str]] = None, zoom: float = 1.0
    ) -> Union['graphviz.Digraph', 'IPython.display.HTML']:
        """Render FSA as an image via graphviz, and return the Digraph object; and optionally save to file filename.
        filename must have a suffix that graphviz understands, such as pdf, svg or png.

        Note:
          You need to install graphviz to use this function::

            ./scripts/installers/install_graphviz.sh

        Args:
          filename:
            Filename to (optionally) save to, e.g. ‘foo.png’, ‘foo.svg’, ‘foo.png’.

          title:
            Title to be displayed in image, e.g. ‘A simple lattice example’.

          zoom:
            Zoom-in lattice in IPython notebook (needed for large lattices).

        Returns:
          graphviz.Digraph or IPython.display.HTML
        """
        pass

    @abstractmethod
    def edit_distance(self, reference_sequence: List[int]) -> int:
        """Get the edit distance from a reference sequence to the lattice.

        Args:
          reference_sequence:
            List of word- or token-ids.

        Returns:
          Number of edits.
        """

    @property
    def lattice(self):
        self._properties = None
        return self._lattice

    @abstractproperty
    def properties(self) -> LatticeProperties:
        pass

    @abstractproperty
    def symbol_table(self) -> Optional[Dict[int, str]]:
        pass

    @abstractproperty
    def auxiliary_tables(self) -> Optional[Tuple[Any]]:
        pass


class KaldiWordLattice(AbstractLattice):
    """A Kaldi lattice wrapper with high-level capabilities."""

    def __init__(
        self,
        lattice: 'kaldifst.Lattice',
        symbol_table: Optional[Dict[int, str]] = None,
        auxiliary_tables: Optional[Dict[str, Any]] = None,
    ):
        _kaldifst_maybe_raise()

        if not isinstance(lattice, kaldifst.Lattice):
            raise ValueError(f"Wrong lattice type: `{type(lattice)}`")
        super().__init__(lattice)

        kaldi_symbols2dict = lambda symbols: {
            int(line.split("\t")[1]): line.split("\t")[0] for line in str(symbols).strip().split("\n")
        }
        self._symbol_table = None
        # most likely lattice will have empty input_symbols
        if symbol_table is not None:
            self._symbol_table = symbol_table
        elif self._lattice.output_symbols is not None:
            # we suppose that lattice.input_symbols will not be changed
            self._symbol_table = kaldi_symbols2dict(self._lattice.output_symbols)

        self._auxiliary_tables = None
        if auxiliary_tables is not None:
            attributes, values = list(auxiliary_tables.keys()), list(auxiliary_tables.values())
            if "input_symbols" not in attributes and self._lattice.input_symbols is not None:
                # rare but possible case
                attributes.append("input_symbols")
                values.append(kaldi_symbols2dict(self._lattice.input_symbols))
            self._auxiliary_tables = namedtuple("KaldiAuxiliaryTables", attributes)(*values)
        elif self._lattice.input_symbols is not None:
            self._auxiliary_tables = namedtuple("KaldiAuxiliaryTables", "input_symbols")(
                kaldi_symbols2dict(self._lattice.input_symbols)
            )

    @property
    def properties(self) -> LatticeProperties:
        if self._properties is None:
            acceptor = self._lattice.properties(KaldiFstMask.Acceptor.value, True) == KaldiFstMask.Acceptor.value
            valid = self._lattice.properties(KaldiFstMask.Error.value, True) != KaldiFstMask.Error.value
            nonempty = self._lattice.num_states > 0
            top_sorted = self._lattice.properties(KaldiFstMask.TopSorted.value, True) == KaldiFstMask.TopSorted.value
            acyclic = self._lattice.properties(KaldiFstMask.Acyclic.value, True) == KaldiFstMask.Acyclic.value
            arc_sorted = (
                self._lattice.properties(KaldiFstMask.IlabelSorted.value, True) == KaldiFstMask.IlabelSorted.value
                and self._lattice.properties(KaldiFstMask.OlabelSorted.value, True) == KaldiFstMask.OlabelSorted.value
            )
            deterministic = (
                self._lattice.properties(KaldiFstMask.IlabelDeterministic.value, True)
                == KaldiFstMask.IlabelDeterministic.value
                and self._lattice.properties(KaldiFstMask.OlabelDeterministic.value, True)
                == KaldiFstMask.OlabelDeterministic.value
            )
            epsilon_free = (
                self._lattice.properties(KaldiFstMask.HasEpsilons.value, True) != KaldiFstMask.HasEpsilons.value
            )
            input_epsilon_free = (
                self._lattice.properties(KaldiFstMask.HasIEpsilons.value, True) != KaldiFstMask.HasIEpsilons.value
            )
            connected = (
                self._lattice.properties(KaldiFstMask.Accessible.value, True) == KaldiFstMask.Accessible.value
                and self._lattice.properties(KaldiFstMask.Coaccessible.value, True) == KaldiFstMask.Coaccessible.value
            )
            weighted = self._lattice.properties(KaldiFstMask.Weighted.value, True) == KaldiFstMask.Weighted.value
            self._properties = LatticeProperties(
                Acceptor=acceptor,
                Valid=valid,
                Nonempty=nonempty,
                TopSorted=top_sorted,
                Acyclic=acyclic,
                ArcSorted=arc_sorted,
                Deterministic=deterministic,
                EpsilonFree=epsilon_free,
                InputEpsilonFree=input_epsilon_free,
                Connected=connected,
                Weighted=weighted,
            )
        return self._properties

    @property
    def symbol_table(self) -> Optional[Dict[int, str]]:
        return self._symbol_table

    @property
    def auxiliary_tables(self) -> Optional[Tuple[Any]]:
        return self._auxiliary_tables

    def as_tensor(self) -> 'torch.Tensor':
        """Represents the lattice as a tensor.

        Returns:
          torch.Tensor
        """
        raise NotImplementedError("Tensor representation is not supported yet.")

    def edit_distance(self, reference_sequence: List[int]) -> int:
        """Get the edit distance from a reference sequence to the lattice.

        Args:
          reference_sequence:
            List of word- or token-ids.

        Returns:
          Number of edits.
        """
        _kaldifst_maybe_raise()

        if not self.properties.InputEpsilonFree:
            logging.warning(f"Lattice contains input epsilons. Edit distance calculations may not be accurate.")
        if not all(reference_sequence):
            raise ValueError(f"reference_sequence contains zeros, which is not allowed.")
        ref = levenshtein_graph_kaldi(kaldifst.make_linear_acceptor(reference_sequence))
        hyp = levenshtein_graph_kaldi(self._lattice)
        kaldifst.invert(hyp)
        ali_fst = kaldifst.compose(hyp, ref)
        succeeded, _, _, total_weight = kaldifst.get_linear_symbol_sequence(kaldifst.shortest_path(ali_fst))
        if not succeeded:
            raise RuntimeError("Something went wrong while calculating edit_distance. Please check input manually.")
        return round(total_weight.value)

    def draw(
        self, filename: Optional[Union[Path, str]] = None, title: Optional[Union[Path, str]] = None, zoom: float = 1.0
    ) -> Union['graphviz.Digraph', 'IPython.display.HTML']:
        """Render FSA as an image via graphviz, and return the Digraph object; and optionally save to file filename.
        filename must have a suffix that graphviz understands, such as pdf, svg or png.

        Note:
          You need to install graphviz to use this function::

            ./scripts/installers/install_graphviz.sh

        Args:
          filename:
            Filename to (optionally) save to, e.g. ‘foo.png’, ‘foo.svg’, ‘foo.png’.

          title:
            Title to be displayed in image, e.g. ‘A simple lattice example’.

          zoom:
            Zoom-in lattice in IPython notebook (needed for large lattices).

        Returns:
          graphviz.Digraph or IPython.display.HTML
        """
        _kaldifst_maybe_raise()
        _graphviz_maybe_raise()

        isym, osym = None, None
        if self._symbol_table:
            osym = kaldifst.SymbolTable()
            for i, w in self._symbol_table.items():
                osym.add_symbol(symbol=w, key=i)

        if (
            self._auxiliary_tables
            and hasattr(self._auxiliary_tables, "input_symbols")
            and self._auxiliary_tables.input_symbols
        ):
            isym = kaldifst.SymbolTable()
            for i, t in self._auxiliary_tables.input_symbols.items():
                isym.add_symbol(symbol=t, key=i)

        fst_dot = kaldifst.draw(
            self._lattice, acceptor=False, portrait=True, isymbols=isym, osymbols=osym, show_weight_one=True
        )
        source = graphviz.Source(fst_dot)
        source_lines = str(source).splitlines()
        # Remove 'digraph tree {'
        source_lines.pop(0)
        # Remove the closing brackets '}'
        source_lines.pop(-1)
        graph_attr = {
            'rankdir': 'LR',
            'size': '8.5,11',
            'center': '1',
            'orientation': 'Portrait',
            'ranksep': '0.4',
            'nodesep': '0.25',
            'margin': '0.0',
        }
        if title is not None:
            graph_attr['label'] = title
        digraph = graphviz.Digraph(graph_attr=graph_attr)
        digraph.body += source_lines
        if filename:
            _, extension = os.path.splitext(filename)
            if extension == '' or extension[0] != '.':
                raise ValueError(f"Filename needs to have a suffix like .png, .pdf, .svg, or .gv: `{filename}`")
            with tempfile.TemporaryDirectory() as tmp_dir:
                temp_fn = digraph.render(filename='temp', directory=tmp_dir, format=extension[1:], cleanup=True)

                shutil.move(temp_fn, filename)
        if _is_notebook():
            import warnings

            from IPython.display import HTML

            with tempfile.TemporaryDirectory() as tmp_dir:
                temp_fn = digraph.render(filename='temp', directory=tmp_dir, format="svg", cleanup=True)
                svg, (width, height) = _svg_srcdoc_resize(temp_fn, zoom)
            # IFrame requires src file to be present when rendering
            # so we use HTML with iframe srcdoc instead
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return HTML(
                    f"""<iframe srcdoc='{svg}' width="100%" height="{round(height * zoom) * 2}px" frameborder="0" allowfullscreen></iframe>"""
                )
        return digraph


def _is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell' or 'Shell':
            return True  # Jupyter notebook, Google Colab notebook, or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type
    except NameError:
        return False  # Probably standard Python interpreter


def _svg_srcdoc_resize(filename: Union[Path, str], zoom: float) -> Tuple[str, Tuple[int, int]]:
    with open(filename, "rt", encoding="utf-8") as f:
        line = f.readline()
        while not line.startswith("<svg"):
            line = f.readline()
        width, height = re.findall('\d+', line)
        width, height = int(width), int(height)
        return f'<svg width="{round(width * zoom)}pt" height="{round(height * zoom)}pt"\n' + f.read(), (width, height)


def levenshtein_graph_kaldi(
    fst: Union['kaldifst.StdFst', 'kaldifst.Lattice'], ins_del_score: float = 0.501
) -> 'kaldifst.StdFst':
    """Construct the levenshtein graph from a kaldi-type WFST or a lattice.

    See also levenshtein_graph from k2.

    Args:
      fst:
        Kaldi-type source WFST or lattice.

      ins_del_score:
        Insertion and deletion penalty.
        Should be more than 0.5 for substitutions to be preferred over insertions/deletions, or less otherwise.

    Returns:
      Kaldi-type levenshtein WFST.
    """
    _kaldifst_maybe_raise()

    if fst.properties(KaldiFstMask.Acceptor.value, True) != KaldiFstMask.Acceptor.value:
        logging.warning(
            "Levenshtein graph construction is not safe for WFSTs with different input and output symbols."
        )
    if fst.properties(KaldiFstMask.Acyclic.value, True) != KaldiFstMask.Acyclic.value:
        raise ValueError("Levenshtein graph is not defined for WFSTs with cycles.")
    if isinstance(fst, kaldifst.StdFst):
        lfst = fst.copy(safe=True)
    elif isinstance(fst, kaldifst.Lattice):
        # dropping lattice weights
        lfst = kaldifst.compile(re.sub("[-\d.]+,[-\d.]+", "0", fst.to_str(show_weight_one=True)))
    else:
        raise ValueError(f"Levenshtein graph building is not supported for the type `{type(fst)}`.")
    sub_score = 0.5
    eps = 0
    for state in kaldifst.StateIterator(lfst):
        # epsilon self-loop for insertions and deletions
        arcs_to_add = [
            kaldifst.StdArc(
                ilabel=eps,
                olabel=eps,
                weight=ins_del_score,
                nextstate=state,
            )
        ]
        for arc in kaldifst.ArcIterator(lfst, state):
            # epsilon-to-ilabel arc for substitutions
            arcs_to_add.append(
                kaldifst.StdArc(
                    ilabel=eps,
                    olabel=arc.ilabel,
                    weight=sub_score,
                    nextstate=arc.nextstate,
                )
            )
            # zero weight for correct ids (redundant for lattices)
            arc.weight = 0.0
        for arc in arcs_to_add:
            lfst.add_arc(state=state, arc=arc)
    kaldifst.arcsort(lfst)
    return lfst


def load_word_lattice(
    lat_filename: Union[Path, str], id2word: Optional[Dict[int, str]] = None, id2token: Optional[Dict[int, str]] = None
) -> Dict[str, KaldiWordLattice]:
    """Helper function to load riva-decoder recognition lattices.

    Args:
      lat_filename:
        Path to the riva-decoder recognition lattice file.

      id2word:
        Word index.
        Mapping from word_id to word_str.

      id2token:
        Token index.
        Mapping from token_id to token_str.

    Returns:
      Dictionary with lattice names and corresponding lattices in KaldiWordLattice format.
    """
    _kaldifst_maybe_raise()

    lattice_dict = {}
    lattice = None
    max_state = 0
    token_seq_list = []
    with open(lat_filename, "rt") as f:
        for line in f.readlines():
            line_items = line.strip().split()
            line_len = len(line_items)
            if line_len == 0:  # end of lattice
                token_seq_list = []
                lattice = None
                max_state = 0
            elif line_len == 1:  # lattice identifier
                assert lattice is None
                assert max_state == 0
                assert len(token_seq_list) == 0
                lat_id = line_items[0]
                lattice = kaldifst.Lattice()
                lattice_dict[lat_id] = KaldiWordLattice(
                    lattice=lattice,
                    symbol_table=id2word,
                    auxiliary_tables={"token_seq_list": token_seq_list, "input_symbols": id2token},
                )
                start = lattice.add_state()
                lattice.start = start
                max_state += 1
            elif line_len in (3, 4):  # arc
                if line_len == 4:  # regular arc
                    state, next_state, label = [int(i) for i in line_items[:-1]]
                    trunk = line_items[-1].split(',')
                    graph_cost, acoustic_cost = [float(i) for i in trunk[:-1]]
                else:  # arc without weight
                    logging.warning(
                        f"""An arc without weight is detected for lattice `{lat_id}`.
                                    Weights and token sequences will be set trivially."""
                    )
                    state, next_state, label = [int(i) for i in line_items]
                    trunk = [""]
                    graph_cost, acoustic_cost = 0.0, 0.0
                if next_state >= max_state:
                    for i in range(max_state, next_state + 1):
                        lattice.add_state()
                    max_state = next_state + 1
                ark = kaldifst.LatticeArc(
                    ilabel=label,
                    olabel=label,
                    weight=kaldifst.LatticeWeight(graph_cost=graph_cost, acoustic_cost=acoustic_cost),
                    nextstate=next_state,
                )
                lattice.add_arc(state=state, arc=ark)
                token_seq_list.append((ark, [int(i) for i in trunk[-1].split(TW_BREAK)] if trunk[-1] != "" else []))
            elif line_len == 2:  # final state
                state = int(line_items[0])
                trunk = line_items[-1].split(',')
                graph_cost, acoustic_cost = [float(i) for i in trunk[:-1]]
                lattice.set_final(
                    state=state, weight=kaldifst.LatticeWeight(graph_cost=graph_cost, acoustic_cost=acoustic_cost)
                )
            else:
                raise RuntimeError(f"Broken line: `{line}`")
    return lattice_dict
