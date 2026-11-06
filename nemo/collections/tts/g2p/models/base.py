# SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
import random
from abc import ABC, abstractmethod
from typing import Optional

from nemo.collections.common.data.lhotse.stateful_rng import process_seeded_rng


class BaseG2p(ABC):
    def __init__(
        self,
        phoneme_dict=None,
        word_tokenize_func=lambda x: x,
        apply_to_oov_word=None,
        mapping_file: Optional[str] = None,
    ):
        """Abstract class for creating an arbitrary module to convert grapheme words
        to phoneme sequences, leave unchanged, or use apply_to_oov_word.
        Args:
            phoneme_dict: Arbitrary representation of dictionary (phoneme -> grapheme) for known words.
            word_tokenize_func: Function for tokenizing text to words.
            apply_to_oov_word: Function that will be applied to out of phoneme_dict word.
        """
        self.phoneme_dict = phoneme_dict
        self.word_tokenize_func = word_tokenize_func
        self.apply_to_oov_word = apply_to_oov_word
        self.mapping_file = mapping_file

    @property
    def rng(self) -> random.Random:
        """
        Random generator for stochastic G2P choices (e.g. ``phoneme_probability``).

        A training dataset that owns a checkpointed generator (see
        :class:`~nemo.collections.common.data.lhotse.stateful_rng.StatefulRandomMixin`) assigns it here in its
        dataloading process, so G2P draws are restored on resume together with the dataset's own draws.
        Otherwise, the G2P uses a generator seeded for the current process from ``torch.initial_seed()``
        and the data-parallel rank. A generator assigned or created in another process (e.g. before the G2P was
        copied into a dataloader worker) is replaced by a fresh per-process one, so workers never share a sequence.
        """
        rng = self.__dict__.get("_rng")
        if rng is None or self.__dict__.get("_rng_pid") != os.getpid():
            self.rng = rng = process_seeded_rng("g2p:")
        return rng

    @rng.setter
    def rng(self, value: random.Random) -> None:
        self._rng = value
        self._rng_pid = os.getpid()

    @abstractmethod
    def __call__(self, text: str) -> str:
        pass
