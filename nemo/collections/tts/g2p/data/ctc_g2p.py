# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
from typing import List

import torch
from transformers import PreTrainedTokenizerBase

from nemo.core import Dataset
from nemo.utils import logging


class CTCG2PBPEDataset(Dataset):
    def __init__(
        self,
        manifest_filepath: str,
        tokenizer_graphemes: PreTrainedTokenizerBase,
        tokenizer_phonemes: PreTrainedTokenizerBase,
        do_lower: bool = True,
        labels: List[str] = None,
        max_source_len: int = 512,
        phoneme_field: str = "text",
        grapheme_field: str = "text_graphemes",
        with_labels: bool = True,
    ):
        """
        Creates a dataset to train a CTC-based G2P models.

        Args:
            manifest_filepath: path to a .json manifest that contains "phoneme_field" and "grapheme_field"
            tokenizer_graphemes: tokenizer for graphemes
            tokenizer_phonemes: tokenizer for phonemes
            do_lower: set to True to lower case input graphemes
            labels: output labels (tokenizer_phonemes vocabulary)
            max_source_len: max length of the grapheme input sequence (examples exceeding len will be dropped)
            phoneme_field: name of the field in manifest_filepath for ground truth phonemes
            grapheme_field: name of the field in manifest_filepath for input grapheme text
            with_labels: set to True for training and False for inference
        """
        super().__init__()

        if not os.path.exists(manifest_filepath):
            raise ValueError(f"{manifest_filepath} not found")

        self.manifest = manifest_filepath
        self.tokenizer_graphemes = tokenizer_graphemes
        self.tokenizer_phonemes = tokenizer_phonemes
        self.max_source_len = max_source_len
        self.labels = labels
        self.labels_tkn2id = {l: i for i, l in enumerate(labels)}
        self.data = []
        self.pad_token = 0
        self.with_labels = with_labels

        removed_ctc_max = 0
        removed_source_max = 0
        with open(manifest_filepath, "r") as f_in:
            logging.debug(f"Loading dataset from: {manifest_filepath}")
            for i, line in enumerate(f_in):
                item = json.loads(line)

                if do_lower:
                    item[grapheme_field] = item[grapheme_field].lower()

                if isinstance(self.tokenizer_graphemes, PreTrainedTokenizerBase):
                    grapheme_tokens = self.tokenizer_graphemes(item[grapheme_field])
                    grapheme_tokens_len = len(grapheme_tokens["input_ids"])
                else:
                    grapheme_tokens = self.tokenizer_graphemes.text_to_ids(item[grapheme_field])
                    grapheme_tokens_len = len(grapheme_tokens)

                if with_labels:
                    target_tokens = self.tokenizer_phonemes.text_to_ids(item[phoneme_field])
                    target_len = len(target_tokens)

                    if target_len > grapheme_tokens_len:
                        removed_ctc_max += 1
                        continue

                    if grapheme_tokens_len > max_source_len:
                        removed_source_max += 1
                        continue

                    self.data.append(
                        {
                            "graphemes": item[grapheme_field],
                            "phonemes": item[phoneme_field],
                            "target": target_tokens,
                            "target_len": target_len,
                        }
                    )
                else:
                    if len(grapheme_tokens) > max_source_len:
                        item[grapheme_field] = item[grapheme_field][:max_source_len]
                        removed_source_max += 1
                    self.data.append(
                        {"graphemes": item[grapheme_field],}
                    )

        logging.info(
            f"Removed {removed_ctc_max} examples on CTC constraint, {removed_source_max} examples based on max_source_len from {manifest_filepath}"
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def map(self, text: str) -> List[int]:
        """ Creates a mapping from target labels to ids."""
        tokens = []
        for word_id, word in enumerate(text.split()):
            tokens.append(self.labels_tkn2id[word])
        return tokens

    def _collate_fn(self, batch):
        graphemes_batch = [entry["graphemes"] for entry in batch]

        # Encode inputs (graphemes)
        # for ByT5 encoder
        if isinstance(self.tokenizer_graphemes, PreTrainedTokenizerBase):
            input_encoding = self.tokenizer_graphemes(
                graphemes_batch,
                padding='longest',
                max_length=self.max_source_len,
                truncation=True,
                return_tensors='pt',
            )
            input_ids, attention_mask = input_encoding.input_ids, input_encoding.attention_mask
            input_len = torch.sum(attention_mask, 1) - 1
        else:
            # for Conformer encoder
            input_ids = [self.tokenizer_graphemes.text_to_ids(sentence) for sentence in graphemes_batch]
            input_len = [len(entry) for entry in input_ids]
            max_len = max(input_len)
            input_ids = [entry + [0] * (max_len - entry_len) for entry, entry_len in zip(input_ids, input_len)]
            attention_mask = None  # not used with Conformer encoder
            input_ids = torch.tensor(input_ids)
            input_len = torch.tensor(input_len)

        # inference
        if not self.with_labels:
            output = (input_ids, attention_mask, input_len)
        # Encode targets (phonemes)
        else:
            targets = [torch.tensor(entry["target"]) for entry in batch]
            target_lengths = [torch.tensor(entry["target_len"]) for entry in batch]
            max_target_len = max(target_lengths)

            padded_targets = []
            for target, target_len in zip(targets, target_lengths):
                pad = (0, max_target_len - target_len)
                target_pad = torch.nn.functional.pad(target, pad, value=len(self.labels))
                padded_targets.append(target_pad)

            padded_targets = torch.stack(padded_targets)
            target_lengths = torch.stack(target_lengths)
            output = (input_ids, attention_mask, input_len, padded_targets, target_lengths)
        return output
