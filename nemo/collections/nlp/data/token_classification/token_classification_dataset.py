# Copyright 2018 The Google AI Language Team Authors and
# The HuggingFace Inc. team.
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

"""
Utility functions for Token Classification NLP tasks
Some parts of this code were adapted from the HuggingFace library at
https://github.com/huggingface/pytorch-pretrained-BERT
"""

import os
import pickle
import tempfile
import time
from typing import Dict, List, Optional

import numpy as np

from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.collections.nlp.data.data_utils.data_preprocessing import get_stats
from nemo.core.classes import Dataset
from nemo.core.neural_types import ChannelType, LabelsType, MaskType, NeuralType
from nemo.utils import logging
from nemo.utils.get_rank import is_global_rank_zero

__all__ = ['BertTokenClassificationDataset', 'BertTokenClassificationInferDataset']


def get_features(
    queries: List[str],
    tokenizer: TokenizerSpec,
    max_seq_length: int = -1,
    label_ids: dict = None,
    pad_label: str = 'O',
    raw_labels: List[str] = None,
    ignore_extra_tokens: bool = False,
    ignore_start_end: bool = False,
):
    """
    Processes the data and returns features.
    Args:
        queries: text sequences
        tokenizer: such as AutoTokenizer
        max_seq_length: max sequence length minus 2 for [CLS] and [SEP], when -1 - use the max len from the data
        pad_label: pad value use for labels. By default, it's the neutral label.
        raw_labels: list of labels for every word in a sequence
        label_ids: dict to map labels to label ids.
            Starts with pad_label->0 and then increases in alphabetical order.
            Required for training and evaluation, not needed for inference.
        ignore_extra_tokens: whether to ignore extra tokens in the loss_mask
        ignore_start_end: whether to ignore bos and eos tokens in the loss_mask
    """
    all_subtokens = []
    all_loss_mask = []
    all_subtokens_mask = []
    all_segment_ids = []
    all_input_ids = []
    all_input_mask = []
    sent_lengths = []
    all_labels = []
    with_label = False

    if raw_labels is not None:
        with_label = True

    for i, query in enumerate(queries):
        words = query.strip().split()

        # add bos token
        subtokens = [tokenizer.cls_token]
        loss_mask = [1 - ignore_start_end]
        subtokens_mask = [0]
        if with_label:
            pad_id = label_ids[pad_label]
            labels = [pad_id]
            query_labels = [label_ids[lab] for lab in raw_labels[i]]

        for j, word in enumerate(words):
            word_tokens = tokenizer.text_to_tokens(word)

            # to handle emojis that could be neglected during tokenization
            if len(word.strip()) > 0 and len(word_tokens) == 0:
                word_tokens = [tokenizer.ids_to_tokens(tokenizer.unk_id)]

            subtokens.extend(word_tokens)

            loss_mask.append(1)
            loss_mask.extend([int(not ignore_extra_tokens)] * (len(word_tokens) - 1))

            subtokens_mask.append(1)
            subtokens_mask.extend([0] * (len(word_tokens) - 1))

            if with_label:
                labels.extend([query_labels[j]] * len(word_tokens))
        # add eos token
        subtokens.append(tokenizer.sep_token)
        loss_mask.append(1 - ignore_start_end)
        subtokens_mask.append(0)
        sent_lengths.append(len(subtokens))
        all_subtokens.append(subtokens)
        all_loss_mask.append(loss_mask)
        all_subtokens_mask.append(subtokens_mask)
        all_input_mask.append([1] * len(subtokens))

        if with_label:
            labels.append(pad_id)
            all_labels.append(labels)

    max_seq_length_data = max(sent_lengths)
    max_seq_length = min(max_seq_length, max_seq_length_data) if max_seq_length > 0 else max_seq_length_data
    logging.info(f'Setting Max Seq length to: {max_seq_length}')
    get_stats(sent_lengths)
    too_long_count = 0

    for i, subtokens in enumerate(all_subtokens):
        if len(subtokens) > max_seq_length:
            subtokens = [tokenizer.cls_token] + subtokens[-max_seq_length + 1 :]
            all_input_mask[i] = [1] + all_input_mask[i][-max_seq_length + 1 :]
            all_loss_mask[i] = [int(not ignore_start_end)] + all_loss_mask[i][-max_seq_length + 1 :]
            all_subtokens_mask[i] = [0] + all_subtokens_mask[i][-max_seq_length + 1 :]

            if with_label:
                all_labels[i] = [pad_id] + all_labels[i][-max_seq_length + 1 :]
            too_long_count += 1

        all_input_ids.append(tokenizer.tokens_to_ids(subtokens))

        if len(subtokens) < max_seq_length:
            extra = max_seq_length - len(subtokens)
            all_input_ids[i] = all_input_ids[i] + [0] * extra
            all_loss_mask[i] = all_loss_mask[i] + [0] * extra
            all_subtokens_mask[i] = all_subtokens_mask[i] + [0] * extra
            all_input_mask[i] = all_input_mask[i] + [0] * extra

            if with_label:
                all_labels[i] = all_labels[i] + [pad_id] * extra

        all_segment_ids.append([0] * max_seq_length)

    logging.warning(f'{too_long_count} are longer than {max_seq_length}')

    for i in range(min(len(all_input_ids), 1)):
        logging.info("*** Example ***")
        logging.info("i: %s", i)
        logging.info("subtokens: %s", " ".join(list(map(str, all_subtokens[i]))))
        logging.info("loss_mask: %s", " ".join(list(map(str, all_loss_mask[i]))))
        logging.info("input_mask: %s", " ".join(list(map(str, all_input_mask[i]))))
        logging.info("subtokens_mask: %s", " ".join(list(map(str, all_subtokens_mask[i]))))
        if with_label:
            logging.info("labels: %s", " ".join(list(map(str, all_labels[i]))))
    return (all_input_ids, all_segment_ids, all_input_mask, all_subtokens_mask, all_loss_mask, all_labels)


class BertTokenClassificationDataset(Dataset):
    """
    Creates dataset to use during training for token classification tasks with a pretrained model.

    Converts from raw data to an instance that can be used by Dataloader.
    For dataset to use during inference without labels, see BertTokenClassificationInferDataset.

    Args:
        text_file: file to sequences, each line should a sentence, no header.
        label_file: file to labels, each line corresponds to word labels for a sentence in the text_file. No header.
        max_seq_length: max sequence length minus 2 for [CLS] and [SEP]
        tokenizer: such as AutoTokenizer
        num_samples: number of samples you want to use for the dataset.
            If -1, use all dataset. Useful for testing.
        pad_label: pad value use for labels. By default, it's the neutral label.
        label_ids: label_ids (dict): dict to map labels to label ids.
            Starts with pad_label->0 and then increases in alphabetical order
            For dev set use label_ids generated during training to support
            cases when not all labels are present in the dev set.
            For training set label_ids should be None.
        ignore_extra_tokens: whether to ignore extra tokens in the loss_mask
        ignore_start_end: whether to ignore bos and eos tokens in the loss_mask
        use_cache: whether to use processed data cache or not
    """

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports.
               """
        return {
            'input_ids': NeuralType(('B', 'T'), ChannelType()),
            'segment_ids': NeuralType(('B', 'T'), ChannelType()),
            'input_mask': NeuralType(('B', 'T'), MaskType()),
            'subtokens_mask': NeuralType(('B', 'T'), MaskType()),
            'loss_mask': NeuralType(('B', 'T'), MaskType()),
            'labels': NeuralType(('B', 'T'), LabelsType()),
        }

    def __init__(
        self,
        text_file: str,
        label_file: str,
        max_seq_length: int,
        tokenizer: TokenizerSpec,
        num_samples: int = -1,
        pad_label: str = 'O',
        label_ids: Dict[str, int] = None,
        ignore_extra_tokens: bool = False,
        ignore_start_end: bool = False,
        use_cache: bool = True,
    ):
        """ Initializes BertTokenClassificationDataset. """

        data_dir = os.path.dirname(text_file)
        text_filename = os.path.basename(text_file)
        lbl_filename = os.path.basename(label_file)

        if not text_filename.endswith('.txt'):
            raise ValueError("{text_file} should have extension .txt")

        vocab_size = getattr(tokenizer, "vocab_size", 0)
        features_pkl = os.path.join(
            data_dir,
            f"cached__{text_filename}__{lbl_filename}__{tokenizer.name}_{max_seq_length}_{vocab_size}_{num_samples}",
        )

        master_device = is_global_rank_zero()
        features = None
        if master_device and (not use_cache or not os.path.exists(features_pkl)):
            if num_samples == 0:
                raise ValueError("num_samples has to be positive", num_samples)

            with open(text_file, 'r') as f:
                text_lines = f.readlines()

            labels_lines = []
            with open(label_file, 'r') as f:
                for line in f:
                    line = line.strip().split()
                    labels_lines.append(line)

            if len(labels_lines) != len(text_lines):
                raise ValueError("Labels file should contain labels for every word")

            if num_samples > 0:
                dataset = list(zip(text_lines, labels_lines))
                dataset = dataset[:num_samples]

                dataset = list(zip(*dataset))
                text_lines = dataset[0]
                labels_lines = dataset[1]

            features = get_features(
                queries=text_lines,
                max_seq_length=max_seq_length,
                tokenizer=tokenizer,
                pad_label=pad_label,
                raw_labels=labels_lines,
                label_ids=label_ids,
                ignore_extra_tokens=ignore_extra_tokens,
                ignore_start_end=ignore_start_end,
            )

            # save features to a temp file first to make sure that non-master processes don't start reading the file
            # until the master process is done with writing
            ofd, tmp_features_pkl = tempfile.mkstemp(
                suffix='.pkl', prefix=os.path.basename(features_pkl), dir=os.path.dirname(features_pkl)
            )
            with os.fdopen(ofd, 'wb') as temp_f:
                pickle.dump(features, temp_f)

            os.rename(tmp_features_pkl, features_pkl)
            logging.info(f'features saved to {features_pkl}')

        # wait until the master process writes to the processed data files
        if not master_device:
            while features is None and not os.path.exists(features_pkl):
                time.sleep(10)

        if features is None:
            features = pickle.load(open(features_pkl, 'rb'))
            logging.info(f'features restored from {features_pkl}')

        self.all_input_ids = features[0]
        self.all_segment_ids = features[1]
        self.all_input_mask = features[2]
        self.all_subtokens_mask = features[3]
        self.all_loss_mask = features[4]
        self.all_labels = features[5]

    def __len__(self):
        return len(self.all_input_ids)

    def __getitem__(self, idx):
        return (
            np.array(self.all_input_ids[idx]),
            np.array(self.all_segment_ids[idx]),
            np.array(self.all_input_mask[idx], dtype=np.longlong),
            np.array(self.all_subtokens_mask[idx]),
            np.array(self.all_loss_mask[idx]),
            np.array(self.all_labels[idx]),
        )


class BertTokenClassificationInferDataset(Dataset):
    """
    Creates dataset to use during inference for token classification tasks with a pretrained model.
    For dataset to use during training with labels, see BertTokenClassificationDataset.
    """

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports.
               """
        return {
            'input_ids': NeuralType(('B', 'T'), ChannelType()),
            'segment_ids': NeuralType(('B', 'T'), ChannelType()),
            'input_mask': NeuralType(('B', 'T'), MaskType()),
            'subtokens_mask': NeuralType(('B', 'T'), MaskType()),
        }

    def __init__(
        self, queries: List[str], max_seq_length: int, tokenizer: TokenizerSpec,
    ):
        """
        Initializes BertTokenClassificationInferDataset
        Args:
            queries: text sequences
            max_seq_length: max sequence length minus 2 for [CLS] and [SEP]
            tokenizer: such as AutoTokenizer
        """
        features = get_features(queries=queries, max_seq_length=max_seq_length, tokenizer=tokenizer)

        self.all_input_ids = features[0]
        self.all_segment_ids = features[1]
        self.all_input_mask = features[2]
        self.all_subtokens_mask = features[3]

    def __len__(self):
        return len(self.all_input_ids)

    def __getitem__(self, idx):
        return (
            np.array(self.all_input_ids[idx]),
            np.array(self.all_segment_ids[idx]),
            np.array(self.all_input_mask[idx], dtype=np.longlong),
            np.array(self.all_subtokens_mask[idx]),
        )
