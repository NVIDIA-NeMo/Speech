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


from nemo.collections.asr.inference.streaming.decoders.greedy.greedy_rnnt_decoder import RNNTGreedyDecoder
from nemo.collections.asr.inference.streaming.endpointing.greedy.greedy_endpointing import GreedyEndpointing


class RNNTGreedyEndpointing(GreedyEndpointing):
    """Greedy endpointing for the streaming RNNT pipeline"""

    def __init__(
        self,
        vocabulary: list[str],
        ms_per_timestep: int,
        effective_buffer_size_in_secs: float = None,
        stop_history_eou: int = -1,
        residue_tokens_at_end: int = 0,
    ) -> None:
        """
        Initialize the RNNTGreedyEndpointing class
        Args:
            vocabulary: (list[str]) List of vocabulary
            ms_per_timestep: (int) Number of milliseconds per timestep
            effective_buffer_size_in_secs: (float, optional) Effective buffer size for VAD-based EOU detection for stateless and stateful RNNT. If None, VAD functionality is disabled.
            stop_history_eou: (int) Number of silent tokens to trigger a EOU, if -1 then it is disabled
            residue_tokens_at_end: (int) Number of residue tokens at the end, if 0 then it is disabled
        """
        super().__init__(
            vocabulary, ms_per_timestep, effective_buffer_size_in_secs, stop_history_eou, residue_tokens_at_end
        )
        self.greedy_rnnt_decoder = RNNTGreedyDecoder(self.vocabulary, conf_func=None)

    def is_token_start_of_word(self, token_id: int) -> bool:
        """
        Check if the token is the start of a word
        Args:
            token_id (int): token id
        Returns:
            bool: True if the token is the start of a word, False otherwise
        """
        return self.greedy_rnnt_decoder.is_token_start_of_word(token_id=token_id)

    def is_token_silent(self, token_id: int) -> bool:
        """
        Check if the token is silent
        Args:
            token_id (int): token id
        Returns:
            bool: True if the token is silent, False otherwise
        """
        return self.greedy_rnnt_decoder.is_token_silent(token_id=token_id)
