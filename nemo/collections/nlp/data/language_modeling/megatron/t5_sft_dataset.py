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
import json
import os
from typing import Optional

import numpy as np
import torch
from datasets import load_dataset

from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.collections.nlp.data.language_modeling.text_memmap_dataset import JSONLMemMapDataset
from nemo.core.classes import Dataset
from nemo.utils import logging

__all__ = ['T5SFTDataset']


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
        index_mapping_dir: Directory to save the index mapping to. If None, will write to the same folder as the dataset.
        hf_dataset: Whether to load the json file with the HuggingFace dataset. otherwise, will load the jsonl file with the JSONLMemMapDataset.
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
            self.indexed_dataset = JSONLMemMapDataset(
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


def convert_data_file_format(src_file_name, tgt_file_name, output_file_name):
    """
    Converts the old two-file format used by SequenceToSequenceDataset to the new JSONL format used by T5SFTDataset
    """
    output_lines = []
    with open(src_file_name, encoding='utf8') as f_src, open(tgt_file_name, encoding='utf8') as f_tgt:
        for i, (src, tgt) in enumerate(zip(f_src, f_tgt)):
            if i % 10000 == 0 and i != 0:
                logging.info(f"Read {i} lines from {src_file_name} & {tgt_file_name}")
            output_lines.append({'input': src, 'output': tgt})

    logging.info(f'Dataset Length : {len(output_lines)}')

    with open(output_file_name, "w") as f_json:
        for line in output_lines:
            f_json.write(json.dumps(line) + '\n')
