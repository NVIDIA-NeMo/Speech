# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import PreTrainedTokenizerBase

from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.core.classes import Dataset
from nemo.utils import logging

__all__ = ["HeteronymClassificationDataset", "CTCG2PBPEDataset", "T5G2PDataset"]


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


class T5G2PDataset(Dataset):
    """
    Creates a dataset to train a T5G2P model.
    """

    def __init__(
        self,
        manifest_filepath: str,
        tokenizer: PreTrainedTokenizerBase,
        max_source_len: int = 512,
        max_target_len: int = 512,
        do_lower: bool = False,
        grapheme_field: str = "text_graphemes",
        phoneme_field: str = "text",
        with_labels: bool = True,
    ):
        """
        Dataset to train T5-based G2P generative model.

        Args:
            manifest_filepath: path to a .json manifest that contains "phoneme_field" and "grapheme_field"
            tokenizer: pretrained T5 tokenizer
            max_source_len: max length of the grapheme input sequence (examples exceeding len will be dropped)
            max_target_len: max length of the phoneme sequence (examples exceeding len will be dropped)
            do_lower: a flag that indicates whether to lower case input grapheme sequence
            phoneme_field: name of the field in manifest_filepath for ground truth phonemes
            grapheme_field: name of the field in manifest_filepath for input grapheme text
            with_labels: set to True for training and False for inference
        """
        super().__init__()

        if not os.path.exists(manifest_filepath):
            raise ValueError(f"{manifest_filepath} not found")

        self.tokenizer = tokenizer
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len
        self.do_lower = do_lower
        self.with_labels = with_labels
        self.data = []

        num_filtered = 0

        # Load grapheme/phoneme sequence pairs into self.data
        with open(manifest_filepath, 'r') as f_in:
            logging.info(f"Loading dataset from: {manifest_filepath}")
            for line in f_in:
                item = json.loads(line)
                graphemes = item[grapheme_field]
                if do_lower:
                    graphemes = graphemes.lower()

                if with_labels:
                    graphemes_len = len(self.tokenizer.tokenize(graphemes))
                    if graphemes_len > max_source_len:
                        num_filtered += 1
                        logging.debug(f"dropping {graphemes_len} longer max_source_len")
                        continue

                    target_len = len(self.tokenizer.tokenize(item[phoneme_field]))
                    if max_target_len > 0 and target_len > max_target_len:
                        num_filtered += 1
                        logging.debug(f"dropping {target_len} longer max_target_len")
                        continue
                    self.data.append({"graphemes": graphemes, "phonemes": item[phoneme_field]})
                else:
                    # truncate input graphemes for inference if the length exceeds max_source_len
                    graphemes_tokenized = self.tokenizer(graphemes)["input_ids"]
                    if len(graphemes_tokenized) > max_source_len:
                        # -1 for special token at the end, <\s>
                        graphemes_tokenized_truncated = graphemes_tokenized[: max_source_len - 1]
                        graphemes = self.tokenizer.batch_decode([graphemes_tokenized_truncated])[0]
                    self.data.append({"graphemes": graphemes})

        logging.info(f"Filtered {num_filtered} too long entries from {manifest_filepath}.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def _collate_fn(self, batch):
        graphemes_batch = [entry["graphemes"] for entry in batch]

        # Encode inputs (graphemes)
        input_encoding = self.tokenizer(
            graphemes_batch, padding='longest', max_length=self.max_source_len, truncation=True, return_tensors='pt',
        )
        input_ids, attention_mask = input_encoding.input_ids, input_encoding.attention_mask
        output = (input_ids, attention_mask)

        # labels are available during training and evaluation but not inference
        if self.with_labels:
            # Encode targets (phonemes)
            phonemes_batch = [entry["phonemes"] for entry in batch]
            target_encoding = self.tokenizer(
                phonemes_batch, padding='longest', max_length=self.max_target_len, truncation=True,
            )
            labels = target_encoding.input_ids

            # Need to replace padding tokens w/ -100 for loss to ignore them
            labels = [
                [(label if label != self.tokenizer.pad_token_id else -100) for label in labels_example]
                for labels_example in labels
            ]
            labels = torch.tensor(labels)
            output = (input_ids, attention_mask, labels)  # grapheme IDs, attention mask, phoneme IDs

        return output


class HeteronymClassificationDataset(Dataset):
    def __init__(
        self,
        manifest: str,
        tokenizer: TokenizerSpec,
        heteronym_dict: Dict[str, Dict[str, str]],
        wordid_to_idx: Dict[str, int],
        max_seq_len: int = 512,
        grapheme_field: str = "text_graphemes",
        with_labels: bool = True,
    ):
        """
        Creates dataset to use to run training and inference on G2PClassificationModel.
        Processes WikiHomograph raw data files:
        https://github.com/google-research-datasets/WikipediaHomographData/tree/master/data

        Args:
            manifest: path to manifest with "heteronym_span", "start_end", "text_graphemes"
                and (optional) "word_id" fields. "word_id" is required for model training.
            tokenizer: pretrained tokenizer
            heteronym_dict: a dictionary where each grapheme contains word_id to ipa_form mappings, e.g.,
                {'use': {'use_nou': "'juːs", 'use_vrb': "'juːz"}}
            wordid_to_idx: mapping from word id to index
            max_seq_len: maximum input sequence length
            grapheme_field: name of the field in the .json manifest with grapheme input
            with_labels: indicates whether labels are provided in the manifest. False for inference, True for training
        """
        super().__init__()

        if not os.path.exists(manifest):
            raise ValueError(f"{manifest} not found")

        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.data = []
        self.pad_token = 0
        self.with_labels = with_labels
        self.heteronym_dict = heteronym_dict
        self.wordid_to_idx = wordid_to_idx
        self.LOSS_PAD_TOKEN = -100
        self.PAD_TOKEN = 0

        num_skipped = 0
        with open(manifest, "r") as f:
            for line in f:
                line = json.loads(line)
                cur_start_end, cur_heteronyms = (line["start_end"], line["heteronym_span"])

                # during inference word_id is not present in the manifest
                if "word_id" in line:
                    cur_word_ids = line["word_id"]
                else:
                    if isinstance(cur_heteronyms, str):
                        cur_word_ids = None
                    else:
                        cur_word_ids = [None] * len(cur_heteronyms)

                if isinstance(cur_heteronyms, str):
                    cur_start_end, cur_heteronyms, cur_word_ids = [cur_start_end], [cur_heteronyms], [cur_word_ids]

                example = self._prepare_sample(line[grapheme_field], cur_start_end, cur_heteronyms, cur_word_ids)
                if example is None:
                    num_skipped += 1
                else:
                    example_dict = {
                        "input_ids": example[0],
                        "subtokens_mask": example[1],
                        "target": example[2],  # None if self.with_labels is False
                    }
                    self.data.append(example_dict)

        logging.info(f"Number of samples in {manifest}: {len(self.data)}, remove {num_skipped} lines")

    def _prepare_sample(
        self,
        sentence: str,
        start_end: List[Tuple[int, int]],
        heteronyms: List[str],
        word_ids: Optional[List[str]] = None,
    ):
        """
        Prepares a single training sample

        Args:
            sentence: input sentence in grapheme form
            start_end: start and end indices of the heteronym spans, start_end indices should be in increasing order
            heteronyms: heteronyms present in the sentence
            word_ids: [Optional] target word_ids, use None for inference, e.g. ['diffuse_adj']
        """
        # drop example where sequence length exceeds max sequence length, +2 for special tokens
        length = len(self.tokenizer.text_to_tokens(sentence)) + 2
        if length > self.max_seq_len:
            logging.debug(f"Sequence length exceeds max sequence length ({self.max_seq_len}): {sentence}.")
            return None

        # check the correctness on start-end indices
        for heteronym_, start_end_ in zip(heteronyms, start_end):
            if heteronym_.lower() != sentence[start_end_[0] : start_end_[1]].lower():
                logging.debug(f"Span for {heteronym_} is incorrect. Skipping example.")
                return None

        input_ids, subtokens_mask, target_word_ids = [], [], []
        # add bos token
        if hasattr(self.tokenizer, "bos_id"):
            input_ids.append(self.tokenizer.bos_id)
            subtokens_mask.append(
                self.PAD_TOKEN
            )  # the first tokens of heteronym spans are 1s, the rest of the tokens are 0s

            if self.with_labels:
                target_word_ids.append(self.LOSS_PAD_TOKEN)  # -100 to pad plain tokens
            else:
                target_word_ids = None  # for inference when labels are not available

        heteronym_span_idx = 0
        # split sentence by space and keep track of word boundaries
        # we assume heteronym is a standalone word
        matches = [(m.group(0), (m.start(), m.end() - 1)) for m in re.finditer(r'\S+', sentence)]
        for match in matches:
            word, word_start_end = match
            # check if the start of the next heteronym span is within the word indices
            if (
                heteronym_span_idx < len(start_end)
                and word_start_end[0] <= start_end[heteronym_span_idx][0] < word_start_end[1]
            ):
                heteronym_start_end = start_end[heteronym_span_idx]
                prefix = ""
                prefix_ids = []
                # for cases when word also includes punctuation marks at the beginning or a prefix,
                # e.g. "diffuse" vs. diffuse vs. pre-diffuse for heteronym {diffuse}
                if word_start_end[0] < heteronym_start_end[0]:
                    prefix = sentence[word_start_end[0] : heteronym_start_end[0]]
                    prefix_ids = self.tokenizer.text_to_ids(prefix)
                    subtokens_mask.extend([self.PAD_TOKEN] * len(prefix_ids))

                word = word[word.index(prefix) + len(prefix) :]
                word_input_ids = self.tokenizer.text_to_ids(word)
                input_ids.extend(prefix_ids + word_input_ids)

                subtokens_mask.extend([1] + [self.PAD_TOKEN] * (len(word_input_ids) - 1))

                if self.with_labels:
                    cur_target_word_id = self.wordid_to_idx[word_ids[heteronym_span_idx]]
                    target_word_ids.extend(
                        [self.LOSS_PAD_TOKEN] * len(prefix_ids)
                        + [cur_target_word_id]
                        + [self.LOSS_PAD_TOKEN] * (len(word_input_ids) - 1)
                    )

                heteronym = sentence.lower()[heteronym_start_end[0] : heteronym_start_end[1]]
                if heteronym not in self.heteronym_dict:
                    logging.debug(f"{heteronym} is not supported. Skipping example.")
                    return None

                heteronym_span_idx += 1
            else:
                ids = self.tokenizer.text_to_ids(word)
                input_ids.extend(ids)
                subtokens_mask.extend([self.PAD_TOKEN] * len(ids))
                if self.with_labels:
                    target_word_ids.extend([self.LOSS_PAD_TOKEN] * len(ids))

        if heteronym_span_idx < len(start_end):
            logging.info("Not all heteronym spans were processed. Skipping example.")
            return None

        # add eos token
        if hasattr(self.tokenizer, "eos_id"):
            input_ids.append(self.tokenizer.eos_id)
            subtokens_mask.append(self.PAD_TOKEN)
            if self.with_labels:
                target_word_ids.append(self.LOSS_PAD_TOKEN)

        # target_word_ids are None for inference when labels are not available
        return input_ids, subtokens_mask, target_word_ids

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def _collate_fn(self, batch):
        """
        Args:
            batch:  A list of tuples of (input_ids, subtokens_mask, [Optional] target_word_ids).
        """
        max_length = max([len(entry["input_ids"]) for entry in batch])

        padded_input_ids = []
        padded_subtokens_mask = []
        padded_attention_mask = []

        if self.with_labels:
            padded_targets = []

        for item in batch:
            input_ids = item["input_ids"]
            if len(input_ids) < max_length:
                pad_width = max_length - len(input_ids)
                padded_attention_mask.append([1] * len(input_ids) + [0] * pad_width)
                padded_input_ids.append(np.pad(input_ids, pad_width=[0, pad_width], constant_values=self.PAD_TOKEN))
                padded_subtokens_mask.append(
                    np.pad(item["subtokens_mask"], pad_width=[0, pad_width], constant_values=self.PAD_TOKEN)
                )

                if self.with_labels:
                    padded_targets.append(
                        np.pad(item["target"], pad_width=[0, pad_width], constant_values=self.LOSS_PAD_TOKEN)
                    )
            else:
                padded_attention_mask.append([1] * len(input_ids))
                padded_input_ids.append(input_ids)
                padded_subtokens_mask.append(item["subtokens_mask"])
                if self.with_labels:
                    padded_targets.append(item["target"])

        output = {
            "input_ids": torch.LongTensor(np.array(padded_input_ids)),
            "attention_mask": torch.LongTensor(np.array(padded_attention_mask)),
            "subtokens_mask": torch.LongTensor(np.array(padded_subtokens_mask)),
        }
        if self.with_labels:
            output["targets"] = torch.LongTensor(padded_targets)
        return output
