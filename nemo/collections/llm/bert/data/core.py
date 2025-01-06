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

from pathlib import Path
from typing import TYPE_CHECKING, Optional

from nemo.collections.nlp.data.information_retrieval.bert_embedding_dataset import BertEmbeddingDataset
from nemo.lightning.base import NEMO_DATASETS_CACHE

if TYPE_CHECKING:
    from nemo.collections.common.tokenizers import TokenizerSpec


def get_dataset_root(name: str) -> Path:
    """Retrieve the root path for the dataset. Create the folder if not exists."""
    output = Path(NEMO_DATASETS_CACHE) / name
    output.mkdir(parents=True, exist_ok=True)

    return output


def create_sft_dataset(
    path: Path,
    tokenizer: "TokenizerSpec",
    seq_length: int = 2048,
    add_bos: bool = False,
    add_eos: bool = True,
    seed: int = 1234,
    index_mapping_dir: Optional[str] = None,
    truncation_method: str = 'right',
    memmap_workers: int = 2,
    data_type: str = 'train',
    num_hard_negatives: int = 1,
    **kwargs,
) -> "BertEmbeddingDataset":
    """Create BertEmbeddingDataset for SFT training."""

    return BertEmbeddingDataset(
        file_path=str(path),
        tokenizer=tokenizer,
        max_seq_length=seq_length,
        add_bos=add_bos,
        add_eos=add_eos,
        memmap_workers=memmap_workers,
        seed=seed,
        index_mapping_dir=index_mapping_dir,
        truncation_method=truncation_method,
        data_type=data_type,
        num_hard_negatives=num_hard_negatives,
        **kwargs,
    )
