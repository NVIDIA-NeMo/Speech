# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

import io
import os
import pickle
import random
from collections import OrderedDict
from typing import List, Optional, Tuple

import braceexpand
import numpy as np
import torch
import webdataset as wd
from torch.utils.data import IterableDataset
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

from nemo.collections.common.tokenizers.moses_tokenizers import MosesProcessor
from nemo.collections.nlp.data.text_normalization import constants
from nemo.collections.nlp.data.text_normalization.utils import read_data_file
from nemo.core.classes import Dataset
from nemo.utils import logging

__all__ = ['TextNormalizationDecoderDataset', 'TarredTextNormalizationDecoderDataset']


class TextNormalizationDecoderDataset(Dataset):
    """
    Creates dataset to use to train a DuplexDecoderModel.
    Converts from raw data to an instance that can be used by Dataloader.
    For dataset to use to do end-to-end inference, see TextNormalizationTestDataset.

    Args:
        input_file: path to the raw data file (e.g., train.tsv).
            For more info about the data format, refer to the
            `text_normalization doc <https://github.com/NVIDIA/NeMo/blob/main/docs/source/nlp/text_normalization/nn_text_normalization.rst>`.
        raw_instances: processed raw instances in the Google TN dataset format (used for tarred dataset)
        tokenizer: tokenizer of the model that will be trained on the dataset
        tokenizer_name: name of the tokenizer,
        mode: should be one of the values ['tn', 'itn', 'joint'].  `tn` mode is for TN only.
            `itn` mode is for ITN only. `joint` is for training a system that can do both TN and ITN at the same time.
        max_len: maximum length of sequence in tokens. The code will discard any training instance whose input or
            output is longer than the specified max_len.
        decoder_data_augmentation (bool): a flag indicates whether to augment the dataset with additional data
            instances that may help the decoder become more robust against the tagger's errors.
            Refer to the doc for more info.
        lang: language of the dataset
        use_cache: Enables caching to use pickle format to store and read data from
        max_insts: Maximum number of instances (-1 means no limit)
        do_tokenize: Tokenize each instance (set to False for Tarred dataset)
        initial_shuffle: Set to True to shuffle the data
    """

    def __init__(
        self,
        input_file: str,
        tokenizer: PreTrainedTokenizerBase,
        tokenizer_name: str,
        raw_instances: Optional[List[List[str]]] = None,
        mode: str = "joint",
        max_len: int = 512,
        decoder_data_augmentation: bool = False,
        lang: str = "en",
        use_cache: bool = False,
        max_insts: int = -1,
        do_tokenize: bool = True,
        initial_shuffle: bool = False,
    ):
        assert mode in constants.MODES
        assert lang in constants.SUPPORTED_LANGS
        self.mode = mode
        self.lang = lang
        self.use_cache = use_cache
        self.max_insts = max_insts
        self.tokenizer = tokenizer
        self.max_seq_len = max_len
        self.mode = mode

        # Get cache path
        data_dir, filename = os.path.split(input_file)
        tokenizer_name_normalized = tokenizer_name.replace('/', '_')
        cached_data_file = os.path.join(
            data_dir, f'cached_decoder_{filename}_{tokenizer_name_normalized}_{lang}_{max_insts}_{mode}_{max_len}.pkl',
        )

        if use_cache and os.path.exists(cached_data_file):
            logging.warning(
                f"Processing of {input_file} is skipped as caching is enabled and a cache file "
                f"{cached_data_file} already exists."
            )
            with open(cached_data_file, 'rb') as f:
                data = pickle.load(f)
                self.insts, self.inputs, self.examples, self.tn_count, self.itn_count, self.label_ids_semiotic = data
        else:
            if raw_instances is None:
                raw_instances = read_data_file(fp=input_file, lang=self.lang, max_insts=max_insts)
            else:
                raw_instances = raw_instances[:max_insts]

            if initial_shuffle:
                random.shuffle(raw_instances)

            logging.debug(f"Converting raw instances to DecoderDataInstance for {input_file}...")
            self.insts, all_semiotic_classes = self.__process_raw_entries(
                raw_instances, decoder_data_augmentation=decoder_data_augmentation
            )
            logging.debug(
                f"Extracted {len(self.insts)} DecoderDateInstances out of {len(raw_instances)} raw instances."
            )
            self.label_ids_semiotic = OrderedDict({l: idx for idx, l in enumerate(all_semiotic_classes)})
            logging.debug(f'Label_ids: {self.label_ids_semiotic}')
            # save labels list from the training file to the input_file to the file
            dir_name, file_name = os.path.split(input_file)
            if 'train' in file_name:
                with open(os.path.join(dir_name, f"label_ids_{file_name}"), 'w') as f:
                    f.write('\n'.join(self.label_ids_semiotic.keys()))

            if do_tokenize:
                logging.debug(f'Processing samples, total number: {len(self.insts)}')
                self.__tokenize_samples(use_cache=use_cache, cached_data_file=cached_data_file)

    def __process_raw_entries(self, raw_instances: List[Tuple[str]], decoder_data_augmentation):
        """
        Converts raw instances to DecoderDataInstance

        raw_instances: raw entries: (semiotic class, written words, spoken words)
        decoder_data_augmentation (bool): a flag indicates whether to augment the dataset with additional data
            instances that may help the decoder become more robust against the tagger's errors.
            Refer to the doc for more info.

        Returns:
            converted instances and all semiotic classes present in the data
        """
        all_semiotic_classes = set([])
        insts = []
        for (classes, w_words, s_words) in tqdm(raw_instances):
            for ix, (_class, w_word, s_word) in enumerate(zip(classes, w_words, s_words)):
                all_semiotic_classes.update([_class])
                if s_word in constants.SPECIAL_WORDS:
                    continue
                for inst_dir in constants.INST_DIRECTIONS:
                    if inst_dir == constants.INST_BACKWARD and self.mode == constants.TN_MODE:
                        continue
                    if inst_dir == constants.INST_FORWARD and self.mode == constants.ITN_MODE:
                        continue
                    # Create a DecoderDataInstance
                    inst = DecoderDataInstance(
                        w_words, s_words, inst_dir, start_idx=ix, end_idx=ix + 1, lang=self.lang, semiotic_class=_class
                    )
                    insts.append(inst)

                    if decoder_data_augmentation:
                        noise_left = random.randint(1, 2)
                        noise_right = random.randint(1, 2)
                        inst = DecoderDataInstance(
                            w_words,
                            s_words,
                            inst_dir,
                            start_idx=ix - noise_left,
                            end_idx=ix + 1 + noise_right,
                            semiotic_class=_class,
                            lang=self.lang,
                        )
                        insts.append(inst)

        all_semiotic_classes = list(all_semiotic_classes)
        all_semiotic_classes.sort()
        return insts, all_semiotic_classes

    def __tokenize_samples(self, use_cache: bool = False, cached_data_file: str = None):
        """
        Tokenizes the entries, samples longer than max_seq_len are discarded

        Args:
            use_cache: Enables caching to use pickle format to store and read data from
            cached_data_file: path the cache file
        """
        inputs = [inst.input_str.strip() for inst in self.insts]
        inputs_center = [inst.input_center_str.strip() for inst in self.insts]
        targets = [inst.output_str.strip() for inst in self.insts]
        classes = [self.label_ids_semiotic[inst.semiotic_class] for inst in self.insts]
        directions = [constants.DIRECTIONS_TO_ID[inst.direction] for inst in self.insts]

        # Tokenization
        self.inputs, self.examples, _inputs_center = [], [], []
        self.tn_count, self.itn_count, long_examples_filtered = 0, 0, 0
        input_max_len, target_max_len = 0, 0
        for idx in tqdm(range(len(inputs))):
            # Input
            _input = self.tokenizer([inputs[idx]])
            input_len = len(_input['input_ids'][0])
            if input_len > self.max_seq_len:
                long_examples_filtered += 1
                continue

            # Target
            _target = self.tokenizer([targets[idx]])
            target_len = len(_target['input_ids'][0])
            if target_len > self.max_seq_len:
                long_examples_filtered += 1
                continue

            # Update
            self.inputs.append(inputs[idx])
            _input['labels'] = _target['input_ids']
            _input['semiotic_class_id'] = [[classes[idx]]]
            _input['direction'] = [[directions[idx]]]
            _inputs_center.append(inputs_center[idx])

            self.examples.append(_input)
            if inputs[idx].startswith(constants.TN_PREFIX):
                self.tn_count += 1
            if inputs[idx].startswith(constants.ITN_PREFIX):
                self.itn_count += 1
            input_max_len = max(input_max_len, input_len)
            target_max_len = max(target_max_len, target_len)
        logging.info(f'long_examples_filtered: {long_examples_filtered}')
        logging.info(f'input_max_len: {input_max_len} | target_max_len: {target_max_len}')

        # we need to pad input_center, so we first collect all values, and then batch_tokenize with padding
        _input_centers = self.tokenizer(_inputs_center, padding=True)

        for idx in range(len(self.examples)):
            self.examples[idx]['input_center'] = [_input_centers['input_ids'][idx]]

        # Write to cache (if use_cache)
        if use_cache:
            with open(cached_data_file, 'wb') as out_file:
                data = (
                    self.insts,
                    self.inputs,
                    self.examples,
                    self.tn_count,
                    self.itn_count,
                    self.label_ids_semiotic,
                )
                pickle.dump(data, out_file, protocol=pickle.HIGHEST_PROTOCOL)

    def __getitem__(self, idx):
        """
        Returns a dataset item

        Args:
            idx: ID of the item
        Returns:
            A dictionary that represents the item, the dictionary contains the following fields:
            input_ids: input ids
            attention_mask: attention mask
            labels: ground truth labels
            semiotic_class_id: id of the semiotic class of the example
            direction: id of the TN/ITN tast (see constants for the values)
            inputs_center: ids of input center (only semiotic span, no special tokens and context)
        """
        example = self.examples[idx]
        item = {key: val[0] for key, val in example.items()}
        return item

    def __len__(self):
        return len(self.examples)

    def batchify(self, batch_size: int):
        """
        Creates a batch

        Args:
            batch_size: the size of the batch
        """
        logging.info("Padding the data and creating batches...")

        long_examples_filtered = 0
        inputs_all = [inst.input_str.strip() for inst in self.insts]
        targets_all = [inst.output_str.strip() for inst in self.insts]
        batch, batches = [], []
        for idx in tqdm(range(len(self.insts))):
            # exclude examples that are longer than maximum sequence length
            # Input
            _input = self.tokenizer([inputs_all[idx]])
            input_len = len(_input['input_ids'][0])
            if input_len > self.max_seq_len:
                long_examples_filtered += 1
                continue

            # Target
            _target = self.tokenizer([targets_all[idx]])
            target_len = len(_target['input_ids'][0])
            if target_len > self.max_seq_len:
                long_examples_filtered += 1
                continue

            batch.append(self.insts[idx])

            if len(batch) == batch_size:
                inputs = [inst.input_str.strip() for inst in batch]
                inputs_center = [inst.input_center_str.strip() for inst in batch]
                targets = [inst.output_str.strip() for inst in batch]

                # Here we assume that every input_file contains examples from every semiotic class
                classes = [[self.label_ids_semiotic[inst.semiotic_class]] for inst in batch]
                directions = [[constants.DIRECTIONS_TO_ID[inst.direction]] for inst in batch]

                batch = self.tokenizer(inputs, padding=True)
                batch['input_center'] = self.tokenizer(inputs_center, padding=True)['input_ids']
                batch['direction'] = directions
                batch['semiotic_class_id'] = classes

                labels = self.tokenizer(targets, padding=True)['input_ids']
                batch['decoder_input_ids'] = np.insert(
                    [x[:-1] for x in labels], 0, self.tokenizer.pad_token_id, axis=-1
                )

                # use LABEL_PAD_TOKEN_ID to disregard padded values for the loss calculations
                batch['labels'] = [[x if x != 0 else constants.LABEL_PAD_TOKEN_ID for x in l] for l in labels]
                batches.append(batch)
                batch = []

        logging.info(f'long_examples_filtered: {long_examples_filtered}')
        self.batches = batches


class DecoderDataInstance:
    """
    This class represents a data instance in a TextNormalizationDecoderDataset.

    Intuitively, each data instance can be thought as having the following form:
        Input:  <Left Context of Input> <Input Span> <Right Context of Input>
        Output: <Output Span>
    where the context size is determined by the constant DECODE_CTX_SIZE.

    Args:
        w_words: List of words in the written form
        s_words: List of words in the spoken form
        inst_dir: Indicates the direction of the instance (i.e., INST_BACKWARD for ITN or INST_FORWARD for TN).
        start_idx: The starting index of the input span in the original input text
        end_idx: The ending index of the input span (exclusively)
        lang: Language of the instance
        semiotic_class: The semiotic class of the input span (can be set to None if not available)
    """

    def __init__(
        self,
        w_words: List[str],
        s_words: List[str],
        inst_dir: str,
        start_idx: int,
        end_idx: int,
        lang: str,
        semiotic_class: str = None,
    ):
        processor = MosesProcessor(lang_id=lang)
        start_idx = max(start_idx, 0)
        end_idx = min(end_idx, len(w_words))
        ctx_size = constants.DECODE_CTX_SIZE
        extra_id_0 = constants.EXTRA_ID_0
        extra_id_1 = constants.EXTRA_ID_1

        # Extract center words
        c_w_words = w_words[start_idx:end_idx]
        c_s_words = s_words[start_idx:end_idx]

        # Extract context
        w_left = w_words[max(0, start_idx - ctx_size) : start_idx]
        w_right = w_words[end_idx : end_idx + ctx_size]
        s_left = s_words[max(0, start_idx - ctx_size) : start_idx]
        s_right = s_words[end_idx : end_idx + ctx_size]

        # Process sil words and self words
        for jx in range(len(s_left)):
            if s_left[jx] == constants.SIL_WORD:
                s_left[jx] = ''
            if s_left[jx] == constants.SELF_WORD:
                s_left[jx] = w_left[jx]
        for jx in range(len(s_right)):
            if s_right[jx] == constants.SIL_WORD:
                s_right[jx] = ''
            if s_right[jx] == constants.SELF_WORD:
                s_right[jx] = w_right[jx]
        for jx in range(len(c_s_words)):
            if c_s_words[jx] == constants.SIL_WORD:
                c_s_words[jx] = c_w_words[jx]
                if inst_dir == constants.INST_BACKWARD:
                    c_w_words[jx] = ''
                    c_s_words[jx] = ''
            if c_s_words[jx] == constants.SELF_WORD:
                c_s_words[jx] = c_w_words[jx]

        # Extract input_words and output_words
        c_w_words = processor.tokenize(' '.join(c_w_words)).split()
        c_s_words = processor.tokenize(' '.join(c_s_words)).split()

        # for cases when nearby words are actually multiple tokens, e.g. '1974,'
        w_left = processor.tokenize(' '.join(w_left)).split()[-constants.DECODE_CTX_SIZE :]
        w_right = processor.tokenize(' '.join(w_right)).split()[: constants.DECODE_CTX_SIZE]

        w_input = w_left + [extra_id_0] + c_w_words + [extra_id_1] + w_right
        s_input = s_left + [extra_id_0] + c_s_words + [extra_id_1] + s_right

        if inst_dir == constants.INST_BACKWARD:
            input_center_words = c_s_words
            input_words = [constants.ITN_PREFIX] + s_input
            output_words = c_w_words
        if inst_dir == constants.INST_FORWARD:
            input_center_words = c_w_words
            input_words = [constants.TN_PREFIX] + w_input
            output_words = c_s_words
        # Finalize
        self.input_str = ' '.join(input_words)
        self.input_center_str = ' '.join(input_center_words)
        self.output_str = ' '.join(output_words)
        self.direction = inst_dir
        self.semiotic_class = semiotic_class


class TarredTextNormalizationDecoderDataset(IterableDataset):
    """
    A similar Dataset to the TextNormalizationDecoderDataset, but which loads tarred tokenized pickle files.
    Accepts a single JSON metadata file containing the total number of batches
    as well as the path(s) to the tarball(s) containing the pickled dataset batch files.
    Valid formats for the text_tar_filepaths argument include:
    (1) a single string that can be brace-expanded, e.g. 'path/to/text.tar' or 'path/to/text_{1..100}.tar', or
    (2) a list of file paths that will not be brace-expanded, e.g. ['text_1.tar', 'text_2.tar', ...].
    Note: For brace expansion in (1), there may be cases where `{x..y}` syntax cannot be used due to shell interference.
    This occurs most commonly inside SLURM scripts. Therefore we provide a few equivalent replacements.
    Supported opening braces - { <=> (, [, < and the special tag _OP_.
    Supported closing braces - } <=> ), ], > and the special tag _CL_.
    For SLURM based tasks, we suggest the use of the special tags for ease of use.
    See the WebDataset documentation for more information about accepted data and input formats.
    If using multiple processes the number of shards should be divisible by the number of workers to ensure an
    even split among workers. If it is not divisible, logging will give a warning but training will proceed.
    Additionally, please note that the len() of this DataLayer is assumed to be the number of tokens
    of the text data. An incorrect manifest length may lead to some DataLoader issues down the line.

    Args:
        text_tar_filepaths: Either a list of tokenized text tarball filepaths, or a string (can be brace-expandable).
        num_batches: total number of batches
        shuffle_n: How many samples to look ahead and load to be shuffled.See WebDataset documentation for more details.
        shard_strategy: Tarred dataset shard distribution strategy chosen as a str value during ddp.
            -   `scatter`: The default shard strategy applied by WebDataset, where each node gets
                a unique set of shards, which are permanently pre-allocated and never changed at runtime.
            -   `replicate`: Optional shard strategy, where each node gets all of the set of shards
                available in the tarred dataset, which are permanently pre-allocated and never changed at runtime.
                The benefit of replication is that it allows each node to sample data points from the entire
                dataset independently of other nodes, and reduces dependence on value of `shuffle_n`.

                .. warning::
                    Replicated strategy allows every node to sample the entire set of available tarfiles,
                    and therefore more than one node may sample the same tarfile, and even sample the same
                    data points! As such, there is no assured guarantee that all samples in the dataset will be
                    sampled at least once during 1 epoch. Scattered strategy, on the other hand, on specific
                    occasions (when the number of shards is not divisible with ``world_size``), will not sample
                    the entire dataset. For these reasons it is not advisable to use tarred datasets as validation
                    or test datasets.
        global_rank: Worker rank, used for partitioning shards.
        world_size: Total number of processes, used for partitioning shards.
    """

    def __init__(
        self,
        text_tar_filepaths: str,
        num_batches: int,
        shuffle_n: int = 0,
        shard_strategy: str = "scatter",
        global_rank: int = 0,
        world_size: int = 1,
    ):
        super(TarredTextNormalizationDecoderDataset, self).__init__()

        valid_shard_strategies = ['scatter', 'replicate']
        if shard_strategy not in valid_shard_strategies:
            raise ValueError(
                f"Invalid shard strategy of type {type(shard_strategy)} "
                f"{repr(shard_strategy) if len(repr(shard_strategy)) < 100 else repr(shard_strategy)[:100] + '...'}! "
                f"Allowed values are: {valid_shard_strategies}."
            )

        if isinstance(text_tar_filepaths, str):
            # Replace '(', '[', '<' and '_OP_' with '{'
            brace_keys_open = ['(', '[', '<', '_OP_']
            for bkey in brace_keys_open:
                if bkey in text_tar_filepaths:
                    text_tar_filepaths = text_tar_filepaths.replace(bkey, "{")

            # Replace ')', ']', '>' and '_CL_' with '}'
            brace_keys_close = [')', ']', '>', '_CL_']
            for bkey in brace_keys_close:
                if bkey in text_tar_filepaths:
                    text_tar_filepaths = text_tar_filepaths.replace(bkey, "}")

        if isinstance(text_tar_filepaths, str):
            # Brace expand
            text_tar_filepaths = list(braceexpand.braceexpand(text_tar_filepaths))

        if shard_strategy == 'scatter':
            logging.info("Tarred dataset shards will be scattered evenly across all nodes.")
            if len(text_tar_filepaths) % world_size != 0:
                logging.warning(
                    f"Number of shards in tarred dataset ({len(text_tar_filepaths)}) is not divisible "
                    f"by number of distributed workers ({world_size}). "
                    f"Some shards will not be used ({len(text_tar_filepaths) % world_size})."
                )
            batches_per_tar = num_batches // len(text_tar_filepaths)
            begin_idx = (len(text_tar_filepaths) // world_size) * global_rank
            end_idx = begin_idx + (len(text_tar_filepaths) // world_size)
            logging.info('Begin Index : %d' % (begin_idx))
            logging.info('End Index : %d' % (end_idx))
            text_tar_filepaths = text_tar_filepaths[begin_idx:end_idx]
            logging.info(
                "Partitioning tarred dataset: process (%d) taking shards [%d, %d)", global_rank, begin_idx, end_idx
            )
            self.length = batches_per_tar * len(text_tar_filepaths) * world_size

        elif shard_strategy == 'replicate':
            logging.info("All tarred dataset shards will be replicated across all nodes.")
            self.length = num_batches

        else:
            raise ValueError(f"Invalid shard strategy! Allowed values are: {valid_shard_strategies}")

        # Put together WebDataset
        self._dataset = wd.WebDataset(urls=text_tar_filepaths, nodesplitter=None)
        if shuffle_n > 0:
            self._dataset = self._dataset.shuffle(shuffle_n)
        else:
            logging.info("WebDataset will not shuffle files within the tar files.")

        self._dataset = self._dataset.rename(pkl='pkl', key='__key__').to_tuple('pkl', 'key').map(f=self._build_sample)

    def _build_sample(self, fname):
        # Load file
        pkl_file, _ = fname
        pkl_file = io.BytesIO(pkl_file)
        data = pickle.load(pkl_file)  # loads np.int64 vector
        pkl_file.close()
        data = {k: torch.tensor(v) for k, v in data.items()}
        return data

    def __iter__(self):
        return self._dataset.__iter__()

    def __len__(self):
        return self.length
