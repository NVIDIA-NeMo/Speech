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

from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from nemo.collections.llm.gpt.data.core import create_sft_dataset
from nemo.collections.llm.gpt.data.fine_tuning import FineTuningDataModule

if TYPE_CHECKING:
    from nemo.collections.common.tokenizers import TokenizerSpec
    from nemo.collections.llm.gpt.data.packed_sequence import PackedSequenceSpecs


class ChatDataModule(FineTuningDataModule):
    """
    Base class for fine-tuning an LLM on chat datasets.
    This class calls `GPTSFTChatDataset` for chat template processing

    See base class `FineTuningDataModule` for more details.
    """

    def __init__(
        self,
        dataset_root: Union[str, Path],
        seq_length: int = 2048,
        tokenizer: Optional["TokenizerSpec"] = None,
        micro_batch_size: int = 4,
        global_batch_size: int = 8,
        rampup_batch_size: Optional[List[int]] = None,
        seed: int = 1234,
        memmap_workers: int = 1,
        num_workers: int = 8,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        packed_sequence_specs: Optional["PackedSequenceSpecs"] = None,
        dataset_kwargs: Optional[Dict[str, Any]] = None,
        use_hf_tokenizer_chat_template: bool = False,
    ):
        """Data module for finetuning on chat datasets.
        See base class `FineTuningDataModule` for more details of the arguments.

        Args:
            use_hf_tokenizer_chat_template: Whether to use the chat template from the HuggingFace tokenizer. If True,
                uses the tokenizer's built-in chat template. If False, uses default chat template from
                GPTSFTChatDataset.  Defaults to False.
        """
        super().__init__(
            dataset_root,
            seq_length,
            tokenizer,
            micro_batch_size,
            global_batch_size,
            rampup_batch_size,
            seed,
            memmap_workers,
            num_workers,
            pin_memory,
            persistent_workers,
            packed_sequence_specs,
            dataset_kwargs,
        )
        self.use_hf_tokenizer_chat_template = use_hf_tokenizer_chat_template

    @lru_cache
    def _create_dataset(self, path, pack_metadata_path=None, is_test=False, **kwargs):
        # pylint: disable=C0115,C0116
        return create_sft_dataset(
            path,
            tokenizer=self.tokenizer,
            seq_length=(self.seq_length if is_test or self.packed_sequence_size <= 0 else self.packed_sequence_size),
            memmap_workers=self.memmap_workers,
            seed=self.seed,
            chat=True,
            is_test=is_test,
            pack_metadata_file_path=None,  # packing is not supported
            pad_cu_seqlens=False,
            use_hf_tokenizer_chat_template=self.use_hf_tokenizer_chat_template,
            **kwargs,
        )
