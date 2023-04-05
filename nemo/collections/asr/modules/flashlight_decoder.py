# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import itertools
import math
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
import torch

from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.core.classes import NeuralModule, typecheck
from nemo.core.neural_types import LengthsType, LogprobsType, NeuralType, PredictionsType


class _TokensWrapper:
    def __init__(self, vocabulary: List[str], tokenizer: TokenizerSpec):
        self.vocabulary = vocabulary
        self.tokenizer = tokenizer

        if tokenizer is None:
            self.reverse_map = {self.vocabulary[i]: i for i in range(len(self.vocabulary))}

        self.vocab_len = len(self.vocabulary)

        if (self.tokenizer is not None) and hasattr(self.tokenizer, 'unk_id') and self.tokenizer.unk_id is not None:
            self.unknown_id = self.tokenizer.unk_id
        elif ' ' in self.vocabulary:
            self.unknown_id = self.token_to_id(' ')
        elif '<unk>' in self.vocabulary:
            self.unknown_id = self.token_to_id('<unk>')
        else:
            self.unknown_id = -1

    @property
    def blank(self):
        return self.vocab_len

    @property
    def unk_id(self):
        return self.unknown_id

    @property
    def vocab(self):
        return self.vocabulary

    @property
    def vocab_size(self):
        # the +1 is because we add the blank id
        return self.vocab_len + 1

    def token_to_id(self, token: str):
        if token == self.blank:
            return -1

        if self.tokenizer is not None:
            return self.tokenizer.token_to_id(token)
        else:
            return self.reverse_map[token]

    def text_to_tokens(self, text: str):
        if self.tokenizer is not None:
            return self.tokenizer.text_to_tokens(text)
        else:
            return list(text)


class FlashLightKenLMBeamSearchDecoder(NeuralModule):
    '''
    @property
    def input_types(self):
        """Returns definitions of module input ports.
        """
        return {
            "log_probs": NeuralType(('B', 'T', 'D'), LogprobsType()),
        }

    @property
    def output_types(self):
        """Returns definitions of module output ports.
        """
        return {"hypos": NeuralType(('B'), PredictionsType())}
    '''

    def __init__(
        self,
        lm_path: str,
        vocabulary: List[str],
        tokenizer: Optional[TokenizerSpec] = None,
        lexicon_path: Optional[str] = None,
        boost_path: Optional[str] = None,
        beam_size: int = 32,
        beam_size_token: int = 32,
        beam_threshold: float = 25.0,
        lm_weight: float = 2.0,
        word_score: float = -1.0,
        unk_weight: float = -math.inf,
        sil_weight: float = 0.0,
    ):

        try:
            from flashlight.lib.text.decoder import (
                LM,
                CriterionType,
                KenLM,
                LexiconDecoder,
                LexiconDecoderOptions,
                SmearingMode,
                Trie,
            )
            from flashlight.lib.text.dictionary import create_word_dict, load_words
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "FlashLightKenLMBeamSearchDecoder requires the installation of flashlight python bindings "
                "from https://github.com/flashlight/text. Please follow the build instructions there."
            )

        super().__init__()

        self.criterion_type = CriterionType.CTC
        self.tokenizer_wrapper = _TokensWrapper(vocabulary, tokenizer)
        self.vocab_size = self.tokenizer_wrapper.vocab_size
        self.blank = self.tokenizer_wrapper.blank
        self.silence = self.tokenizer_wrapper.unk_id

        if lexicon_path is not None:
            self.lexicon = load_words(lexicon_path)
            self.word_dict = create_word_dict(self.lexicon)
            self.unk_word = self.word_dict.get_index("<unk>")

            # loads in the boosted words if given via a file
            if boost_path is not None:
                with open(boost_path, 'r', encoding='utf_8') as fr:
                    boost_words = [line.strip().split('\t') for line in fr]
                    boost_words = {w[0]: w[1] for w in boost_words}
            else:
                boost_words = {}

            # add OOV boosted words to word_dict so it gets picked up in LM obj creation
            for word in boost_words.keys():
                if word not in self.lexicon:
                    self.word_dict.add_entry(word)

            # loads in the kenlm binary and combines in with the dictionary object from the lexicon
            # this gives a mapping between each entry in the kenlm binary and its mapping to whatever
            # numeraire is used by the AM, which is explicitly mapped via the lexicon
            # this information is ued to build a vocabulary trie for decoding
            self.lm = KenLM(lm_path, self.word_dict)
            self.trie = Trie(self.vocab_size, self.silence)

            start_state = self.lm.start(False)
            for i, (word, spellings) in enumerate(self.lexicon.items()):
                word_idx = self.word_dict.get_index(word)
                _, score = self.lm.score(start_state, word_idx)
                for spelling in spellings:
                    spelling_idxs = [self.tokenizer_wrapper.token_to_id(token) for token in spelling]
                    if self.tokenizer_wrapper.unk_id in spelling_idxs:
                        print(f'tokenizer has unknown id for word[ {word} ] {spelling} {spelling_idxs}', flush=True)
                        continue
                    self.trie.insert(
                        spelling_idxs, word_idx, score if word not in boost_words else float(boost_words[word])
                    )
            # handle OOV boosted words
            for word, boost in boost_words.items():
                if word not in self.lexicon:
                    word_idx = self.word_dict.get_index(word)
                    spelling = self.tokenizer_wrapper.text_to_tokens(word)
                    spelling_idxs = [self.tokenizer_wrapper.token_to_id(token) for token in spelling]
                    if self.tokenizer_wrapper.unk_id in spelling_idxs:
                        print(f'tokenizer has unknown id for word[ {word} ] {spelling} {spelling_idxs}', flush=True)
                        continue
                    self.trie.insert(spelling_idxs, word_idx, float(boost))
            self.trie.smear(SmearingMode.MAX)

            self.decoder_opts = LexiconDecoderOptions(
                beam_size=beam_size,
                beam_size_token=int(beam_size_token),
                beam_threshold=beam_threshold,
                lm_weight=lm_weight,
                word_score=word_score,
                unk_score=unk_weight,
                sil_score=sil_weight,
                log_add=False,
                criterion_type=self.criterion_type,
            )

            self.decoder = LexiconDecoder(
                self.decoder_opts, self.trie, self.lm, self.silence, self.blank, self.unk_word, [], False,
            )
        else:
            from flashlight.lib.text.decoder import LexiconFreeDecoder, LexiconFreeDecoderOptions

            d = {
                w: [[w]]
                for w in self.tokenizer_wrapper.vocab + ([] if '<unk>' in self.tokenizer_wrapper.vocab else ['<unk>'])
            }
            self.word_dict = create_word_dict(d)
            self.lm = KenLM(lm_path, self.word_dict)
            self.decoder_opts = LexiconFreeDecoderOptions(
                beam_size=beam_size,
                beam_size_token=int(beam_size_token),
                beam_threshold=beam_threshold,
                lm_weight=lm_weight,
                sil_score=sil_weight,
                log_add=False,
                criterion_type=self.criterion_type,
            )
            self.decoder = LexiconFreeDecoder(self.decoder_opts, self.lm, self.silence, self.blank, [])

    def _get_tokens(self, idxs: List[int]):
        """Normalize tokens by handling CTC blank, ASG replabels, etc."""

        idxs = (g[0] for g in itertools.groupby(idxs))
        if self.silence < 0:
            idxs = filter(lambda x: x != self.blank and x != self.silence, idxs)
        else:
            idxs = filter(lambda x: x != self.blank, idxs)
        idxs = list(idxs)
        if idxs[0] == self.silence:
            idxs = idxs[1:]
        if idxs[-1] == self.silence:
            idxs = idxs[:-1]

        return torch.LongTensor(idxs)

    def _get_timesteps(self, token_idxs: List[int]):
        """Returns frame numbers corresponding to every non-blank token.
        Parameters
        ----------
        token_idxs : List[int]
            IDs of decoded tokens.
        Returns
        -------
        List[int]
            Frame numbers corresponding to every non-blank token.
        """

        timesteps = []
        for i, token_idx in enumerate(token_idxs):
            if token_idx == self.blank:
                continue
            if i == 0 or token_idx != token_idxs[i - 1]:
                timesteps.append(i)

        return timesteps

    # @typecheck(ignore_collections=True)
    @torch.no_grad()
    def forward(self, log_probs: Union[np.ndarray, torch.Tensor]):
        if isinstance(log_probs, np.ndarray):
            log_probs = torch.from_numpy(log_probs).float()
        if log_probs.dim() == 2:
            log_probs = log_probs.unsqueeze(0)

        emissions = log_probs.cpu().contiguous()

        B, T, N = emissions.size()
        hypos = []
        # we iterate over the batch dimension of our input tensor log probabilities
        for b in range(B):
            # the flashlight C++ expects a C style pointer, so the memory address
            # which is what we obtain here. Then we pass it to pybinding method which
            # is bound to the underlying C++ code
            emissions_ptr = emissions.data_ptr() + 4 * b * emissions.stride(0)
            results = self.decoder.decode(emissions_ptr, T, N)

            hypos.append(
                [
                    {
                        "tokens": self._get_tokens(result.tokens),
                        "score": result.score,
                        "timesteps": self._get_timesteps(result.tokens),
                        "words": [self.word_dict.get_entry(x) for x in result.words if x >= 0],
                    }
                    for result in results
                ]
            )

        return hypos
