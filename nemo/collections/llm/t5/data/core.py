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
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from datasets import load_dataset

from nemo.collections.common.tokenizers import TokenizerSpec
from nemo.collections.llm.gpt.data.core import _JSONLMemMapDataset
from nemo.core.classes import Dataset
from nemo.lightning.base import NEMO_DATASETS_CACHE


def get_dataset_root(name: str) -> Path:
    """ """

    output = Path(NEMO_DATASETS_CACHE) / name
    output.mkdir(parents=True, exist_ok=True)

    return output


def create_sft_dataset(
    path: Path,
    tokenizer: "TokenizerSpec",
    seq_length: int = 512,
    seq_length_dec: int = 128,
    add_bos: bool = True,
    add_eos: bool = True,
    replace_bos_with_pad: bool = False,
    seed: int = 1234,
    index_mapping_dir: Optional[str] = None,
    memmap_workers: int = 2,
    hf_dataset: bool = False,
    **kwargs,
) -> "T5SFTDataset":
    """ """

    return T5SFTDataset(
        file_path=str(path),
        src_tokenizer=tokenizer,
        tgt_tokenizer=tokenizer,
        max_src_seq_length=seq_length,
        max_tgt_seq_length=seq_length_dec,
        memmap_workers=memmap_workers,
        hf_dataset=hf_dataset,
        add_bos_to_input=add_bos,
        add_eos_to_input=add_eos,
        replace_bos_with_pad=replace_bos_with_pad,
        index_mapping_dir=index_mapping_dir,
    )


class T5SFTDataset(Dataset):
    """Sequence to Sequence Dataset in memory.
    Similar to SequenceToSequenceDataset but with the same input format as GPTSFTDataset
    """

    def __init__(
        self,
        file_path: str,
        src_tokenizer: TokenizerSpec,
        tgt_tokenizer: TokenizerSpec,
        max_src_seq_length: int,
        max_tgt_seq_length: int,
        add_bos_to_input: bool = True,
        add_eos_to_input: bool = True,
        replace_bos_with_pad: bool = False,
        index_mapping_dir: str = None,
        memmap_workers: Optional[int] = None,
        hf_dataset: bool = False,
    ):
        """
        index_mapping_dir: Directory to save the index mapping to.
            If None, will write to the same folder as the dataset.
        hf_dataset: Whether to load the json file with the HuggingFace dataset.
            Otherwise, will load the jsonl file with the JSONLMemMapDataset.
        """
        super().__init__()
        self.file_path = file_path
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_src_seq_length = max_src_seq_length
        self.max_tgt_seq_length = max_tgt_seq_length
        self.add_bos_to_input = add_bos_to_input
        self.add_eos_to_input = add_eos_to_input
        self.replace_bos_with_pad = replace_bos_with_pad
        assert self.max_src_seq_length > 0
        assert self.max_tgt_seq_length > 0

        # check file exists
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Data file {self.file_path} not found")

        if hf_dataset:
            self.indexed_dataset = load_dataset(
                'json', data_files=file_path, cache_dir=index_mapping_dir, num_proc=memmap_workers, split='train'
            )
        else:
            self.indexed_dataset = _JSONLMemMapDataset(
                dataset_paths=[file_path],
                tokenizer=None,
                header_lines=0,
                index_mapping_dir=index_mapping_dir,
                workers=memmap_workers,
            )

    def _process_src(self, src):
        src = self.src_tokenizer.text_to_ids(src.strip())
        if self.add_bos_to_input:
            src = [self.src_tokenizer.pad_id if self.replace_bos_with_pad else self.src_tokenizer.bos_id] + src
        if self.add_eos_to_input:
            src = src + [self.src_tokenizer.eos_id]
        if len(src) > self.max_src_seq_length:
            src = src[-self.max_src_seq_length + 1 :]
        return src

    def _process_tgt(self, tgt):
        tgt = (
            [self.tgt_tokenizer.pad_id if self.replace_bos_with_pad else self.tgt_tokenizer.bos_id]
            + self.tgt_tokenizer.text_to_ids(tgt.strip())
            + [self.tgt_tokenizer.eos_id]
        )
        if len(tgt) > self.max_tgt_seq_length:
            tgt = tgt[-self.max_tgt_seq_length + 1 :]
        return tgt

    def __len__(self):
        return len(self.indexed_dataset)

    def __getitem__(self, idx):
        example = self.indexed_dataset[idx]
        text_enc = self._process_src(example['input'])
        tgt = self._process_tgt(example['output'])
        text_dec = tgt[:-1]
        labels = tgt[1:]
        return {'text_enc': text_enc, 'text_dec': text_dec, 'labels': labels}

    def collate_fn(self, batch):
        text_enc = [item['text_enc'] for item in batch]
        text_dec = [item['text_dec'] for item in batch]
        labels = [item['labels'] for item in batch]

        if isinstance(text_enc[0], np.ndarray):
            text_enc = [x.tolist() for x in text_enc]

        if isinstance(text_dec[0], np.ndarray):
            text_dec = [x.tolist() for x in text_dec]

        if isinstance(labels[0], np.ndarray):
            labels = [x.tolist() for x in labels]

        max_dec_input_length = max([len(item) for item in text_dec]) if text_dec else 0
        max_enc_input_length = max([len(item) for item in text_enc]) if text_enc else 0
        max_label_length = max([len(item) for item in labels]) if labels else 0

        loss_mask = [([1] * (len(item))) + ([0] * (max_label_length - len(item))) for item in labels]
        text_enc = [item + [self.src_tokenizer.pad_id] * (max_enc_input_length - len(item)) for item in text_enc]
        text_dec = [item + [self.tgt_tokenizer.pad_id] * (max_dec_input_length - len(item)) for item in text_dec]
        labels = [item + [self.tgt_tokenizer.pad_id] * (max_label_length - len(item)) for item in labels]

        text_enc = torch.LongTensor(text_enc)
        text_dec = torch.LongTensor(text_dec)
        labels = torch.LongTensor(labels)
        loss_mask = torch.LongTensor(loss_mask)

        enc_mask = (text_enc != self.src_tokenizer.pad_id).long()
        dec_mask = (text_dec != self.tgt_tokenizer.pad_id).long()

        return {
            'text_enc': text_enc,
            'text_dec': text_dec,
            'labels': labels,
            'loss_mask': loss_mask,
            'enc_mask': enc_mask,
            'dec_mask': dec_mask,
        }
