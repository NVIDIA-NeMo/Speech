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


from nemo.collections.asr.inference.streaming.decoders.greedy.greedy_ctc_decoder import CTCGreedyDecoder
from nemo.collections.asr.inference.streaming.decoders.greedy.greedy_rnnt_decoder import RNNTGreedyDecoder


class TokenClassifier:
    """Per-token semantics used by the endpointers (silence / word-start / absorb).

    Composed by the endpointer classes instead of subclassing per decoder: it wraps the matching greedy
    decoder so the `is_token_silent` / `is_token_start_of_word` logic (incl. the out-of-vocabulary bounds
    check) has a single source of truth in `decoders/`. `absorb_token_ids` are tokens (e.g. punctuation /
    language tokens) that should be absorbed into the text preceding an EoU rather than starting a new
    utterance.
    """

    def __init__(self, decoder, absorb_token_ids: set[int] | None = None) -> None:
        self._decoder = decoder
        self.vocabulary = decoder.vocabulary
        self.blank_id = decoder.blank_id
        self.absorb_token_ids = set(absorb_token_ids) if absorb_token_ids else set()

    @classmethod
    def for_ctc(cls, vocabulary: list[str], absorb_token_ids: set[int] | None = None) -> "TokenClassifier":
        """Build a classifier backed by the CTC greedy decoder (provides `get_labels` for emission scans)."""
        return cls(CTCGreedyDecoder(vocabulary, conf_func=None), absorb_token_ids)

    @classmethod
    def for_rnnt(cls, vocabulary: list[str], absorb_token_ids: set[int] | None = None) -> "TokenClassifier":
        """Build a classifier backed by the RNNT greedy decoder."""
        return cls(RNNTGreedyDecoder(vocabulary, conf_func=None), absorb_token_ids)

    def is_token_silent(self, token_id: int) -> bool:
        """True if the token is silence (the blank token)."""
        return self._decoder.is_token_silent(token_id)

    def is_token_start_of_word(self, token_id: int) -> bool:
        """True if the token starts a new word (out-of-vocabulary ids are not word starts)."""
        return self._decoder.is_token_start_of_word(token_id)

    def is_token_to_absorb(self, token_id: int) -> bool:
        """True if the token should be absorbed into the text preceding the EoU."""
        return token_id in self.absorb_token_ids

    def get_labels(self, log_probs) -> list[int]:
        """Greedy argmax labels from log-probs (CTC only; used by the buffered-CTC emission scan)."""
        return self._decoder.get_labels(log_probs)
