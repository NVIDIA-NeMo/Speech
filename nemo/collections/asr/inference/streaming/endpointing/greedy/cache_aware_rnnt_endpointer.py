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


from nemo.collections.asr.inference.streaming.endpointing.greedy.cache_aware_endpointer import CacheAwareEndpointer
from nemo.collections.asr.inference.streaming.endpointing.greedy.token_classifier import TokenClassifier


class CacheAwareRNNTEndpointer(CacheAwareEndpointer):
    """Cache-aware endpointing for the streaming RNNT pipeline (RNNT token semantics)."""

    def __init__(
        self,
        vocabulary: list[str],
        ms_per_timestep: int,
        stop_history_eou: int = -1,
        stop_history_eou_end: int | None = None,
        absorb_token_ids: set[int] | None = None,
    ) -> None:
        """
        Args:
            vocabulary: (list[str]) List of vocabulary tokens.
            ms_per_timestep: (int) Number of milliseconds per timestep.
            stop_history_eou: (int) Mid-buffer silence threshold (ms); -1 disables EoU detection.
            stop_history_eou_end: (int | None) End-of-buffer silence threshold (ms); see CacheAwareEndpointer.
            absorb_token_ids: (set[int] | None) Token ids absorbed into the text preceding the EoU.
        """
        super().__init__(
            TokenClassifier.for_rnnt(vocabulary, absorb_token_ids),
            ms_per_timestep,
            stop_history_eou=stop_history_eou,
            stop_history_eou_end=stop_history_eou_end,
        )
