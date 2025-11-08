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

from typing import List, Optional

from vllm.config import TokenizerPoolConfig
from vllm.lora.request import LoRARequest
from vllm.transformers_utils.tokenizer_group import TokenizerGroup

from nemo.export.sentencepiece_tokenizer import SentencePieceTokenizer


class NemoTokenizerGroup(TokenizerGroup):
    """
    Implements a custom tokenizer for vLLM, based on SentencePieceTokenizer.
    """

    def __init__(self, tokenizer: SentencePieceTokenizer, add_bos_token: bool = False):
        self.tokenizer = tokenizer
        self.add_bos_token = add_bos_token

    @classmethod
    def from_config(cls, tokenizer_pool_config: Optional[TokenizerPoolConfig] = None, **init_kwargs):
        """Create a tokenizer group from a config."""
        raise NotImplementedError

    def ping(self) -> bool:
        """Check if the tokenizer group is alive."""
        return True

    def get_max_input_len(self, lora_request: Optional[LoRARequest] = None) -> Optional[int]:
        """Get the maximum input length for the LoRA request."""
        return None

    def encode(
        self,
        prompt: str,
        request_id: Optional[str] = None,
        lora_request: Optional[LoRARequest] = None,
        add_special_tokens: Optional[bool] = None,
    ) -> List[int]:
        """Tokenizes the prompt."""
        ids = self.tokenizer.encode(prompt)
        if self.add_bos_token:
            ids = [self.tokenizer.bos_token_id] + ids
        return ids

    async def encode_async(
        self,
        prompt: str,
        request_id: Optional[str] = None,
        lora_request: Optional[LoRARequest] = None,
        add_special_tokens: Optional[bool] = None,
    ) -> List[int]:
        """Encode a prompt using the tokenizer group."""
        return self.tokenizer.encode(prompt)  # TODO: not sure how this is supposed to work

    def get_lora_tokenizer(self, lora_request: Optional[LoRARequest] = None) -> SentencePieceTokenizer:
        """Get a tokenizer for a LoRA request."""
        return self.tokenizer

    async def get_lora_tokenizer_async(self, lora_request: Optional[LoRARequest] = None) -> SentencePieceTokenizer:
        """Get a tokenizer for a LoRA request."""
        return self.tokenizer
