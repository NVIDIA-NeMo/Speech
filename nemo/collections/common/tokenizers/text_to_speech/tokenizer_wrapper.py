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

from nemo.collections.common.tokenizers import TokenizerSpec
from nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers import EnglishPhonemesTokenizer
from nemo.collections.tts.g2p.models.en_us_arpabet import EnglishG2p

__all__ = ['TextToSpeechTokenizer']


class TextToSpeechTokenizer(TokenizerSpec):
    def __init__(self, phoneme_dict, heteronyms):
        self.g2p = EnglishG2p(phoneme_dict=phoneme_dict, heteronyms=heteronyms)
        self.tokenizer = EnglishPhonemesTokenizer(
            self.g2p, stresses=True, chars=True, pad_with_space=True, add_blank_at=True
        )
        self.vocab_size = len(self.tokenizer.tokens)

    def text_to_ids(self, text):
        return self.tokenizer.encode(text)

    def text_to_tokens(self, text):
        return self.g2p(text)

    def tokens_to_text(self, tokens):
        pass

    def tokens_to_ids(self, tokens):
        pass

    def ids_to_tokens(self, ids):
        pass

    def ids_to_text(self, ids):
        pass

    @property
    def pad_id(self):
        return self.tokenizer.pad

    @property
    def bos_id(self):
        return self.tokenizer.pad

    @property
    def eos_id(self):
        return self.tokenizer.pad
