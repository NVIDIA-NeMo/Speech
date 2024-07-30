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

import math
import re
from typing import List, Mapping, Optional

import datasets
import numpy as np
import torch

# hack to avoid the "not enough disk space" error in some slurm cluster
datasets.builder.has_sufficient_disk_space = lambda needed_bytes, directory='.': True
from datasets import load_dataset

from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.collections.nlp.data.language_modeling.megatron.dataset_utils import get_samples_mapping
from nemo.collections.nlp.data.language_modeling.text_memmap_dataset import JSONLMemMapDataset, OnlineSampleMapping
from nemo.core.classes import Dataset
from nemo.utils import logging

__all__ = ['GPTSFTDataset']


class GPTSFTDataset(Dataset):
    def __init__(
        self,
        file_path: str,
        tokenizer: TokenizerSpec,
        max_seq_length: int = 1024,
        min_seq_length: int = 1,
        pad_seq_length_to_mult: int = 16,
        add_bos: bool = False,
        add_eos: bool = True,
        add_sep: bool = False,
        sep_id: int = None,
        max_num_samples: int = None,
        seed: int = 1234,
        label_key: str = "answer",
        answer_only_loss: bool = True,
        truncation_field: str = "text",
        pad_to_max_length: bool = False,  # (@adithyare) allows for much faster training especially in PEFT settings.
        index_mapping_dir: str = None,
        prompt_template: str = None,
        virtual_tokens: int = 0,
        tokens_to_generate: int = 0,
        memmap_workers: Optional[int] = None,
        hf_dataset: bool = False,
        global_sample_mapping: bool = False,
        truncation_method: str = 'right',
        special_tokens: Optional[Mapping[str, str]] = None,  # special tokens, a dictory of {token_type: token}
        is_test: bool = False,
        output_original_text: bool = False,
        ceil_to_power_2: bool = False,
        get_attention_mask_from_fusion: bool = False,
    ):
        """
        file_path: Path to a JSONL GPT supervised fine-tuning dataset. Data is formatted as multiple JSON lines with each line formatted as follows. {'input': 'John von Neumann\nVon Neumann made fundamental contributions .... Q: What did the math of artificial viscosity do?', 'output': 'smoothed the shock transition without sacrificing basic physics'}
        tokenizer: Tokenizer for the dataset. Instance of a class that inherits TokenizerSpec (ex: SentencePiece).
        max_seq_length (int): maximum sequence length for each dataset examples. Examples will either be truncated to fit this length or dropped if they cannot be truncated.
        min_seq_length (int): min length of each data example in the dataset. Data examples will be dropped if they do not meet the min length requirements.
        add_bos (bool): Whether to add a beginning of sentence token to each data example
        add_eos (bool): Whether to add an end of sentence token to each data example
        add_sep (bool): Whether to add a separation token to each data example (goes between prompt and answer)
        tokens_to_generate (int): (inference only) Number of tokens to generate during inference
        seed: Random seed for data shuffling.
        max_num_samples: Maximum number of samples to load. This can be > dataset length if you want to oversample data. If None, all samples will be loaded.
        seed: int = 1234,
        label_key: Key to use for the label in your JSONL file
        answer_only_loss: If True, will compute the loss only on the answer part of the input. If False, will compute the loss on the entire input.
        truncation_field: Field to use for truncation. (Options: keys in prompt_template). Field to be used for truncation if the combined length exceeds the max sequence length.
        pad_to_max_length: Whether to pad the input to the max sequence length. If False, will pad to the max length of the current batch.
        index_mapping_dir: Directory to save the index mapping to. If None, will write to the same folder as the dataset.
        prompt_template: Prompt template to inject via an fstring. Formatted like Q: {context_key}\n\nA: {label_key}
        hf_dataset: Whether to load the json file with the HuggingFace dataset. otherwise, will load the jsonl file with the JSONLMemMapDataset.
        global_sample_mapping: Whether to shuffle all data together, or shuffle the dataset within each epoch
        truncation_method: Truncation from which position. Options: ['left', 'right']
        special_tokens: special tokens for the chat prompts, a dictionary of {token_type: token}. Default: {'system_turn_start': '<extra_id_0>', 'turn_start': '<extra_id_1>', 'label_start': '<extra_id_2>', 'end_of_turn': '\n', "end_of_name": "\n"}
        is_test: Whether this dataset is the test split.
        output_original_text (bool): if true, will keep the original text in the output alongside the tokenized ids.
        """
        self.tokenizer = tokenizer
        self.file_path = file_path
        self.max_seq_length = max_seq_length
        self.min_seq_length = min_seq_length
        self.pad_seq_length_to_mult = pad_seq_length_to_mult
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.add_sep = add_sep
        self.sep_id = sep_id
        self.max_num_samples = max_num_samples
        self.seed = seed
        self.label_key = label_key
        self.answer_only_loss = answer_only_loss
        self.truncation_fields = truncation_field.split(',') if truncation_field is not None else []
        self.pad_to_max_length = pad_to_max_length
        self.index_mapping_dir = index_mapping_dir
        self.prompt_template = prompt_template
        self.virtual_tokens = virtual_tokens
        self.tokens_to_generate = tokens_to_generate
        self.memmap_workers = memmap_workers
        self.hf_dataset = hf_dataset
        self.global_sample_mapping = global_sample_mapping
        self.truncation_method = truncation_method
        self.is_test = is_test
        self.output_original_text = output_original_text
        self.ceil_to_power_2 = ceil_to_power_2
        self.get_attention_mask_from_fusion = get_attention_mask_from_fusion

        if special_tokens is None:
            self.special_tokens = {
                "system_turn_start": "<extra_id_0>",
                "turn_start": "<extra_id_1>",
                "label_start": "<extra_id_2>",
                "end_of_turn": "\n",
                "end_of_name": "\n",
            }
        else:
            self.special_tokens = special_tokens

        self._load_dataset()

        # Validate prompt template
        self._maybe_validate_prompt_template()

        # Will be None after this call if `max_num_samples` is None
        self._build_samples_mapping()

    def _load_dataset(self):
        if self.hf_dataset:
            self.indexed_dataset = load_dataset(
                'json',
                data_files=self.file_path,
                cache_dir=self.index_mapping_dir,
                num_proc=self.memmap_workers,
                split='train',
            )
        else:
            self.indexed_dataset = JSONLMemMapDataset(
                dataset_paths=[self.file_path],
                tokenizer=None,
                header_lines=0,
                index_mapping_dir=self.index_mapping_dir,
                workers=self.memmap_workers,
            )

    def _maybe_validate_prompt_template(self):
        assert (
            self.prompt_template is not None
        ), f'we need prompt_template to combine contexts and label {self.label_key}'
        # When providing things like newlines in the prompt template via the CLI, they are escaped. This line unescapes them.
        self.prompt_template = self.prompt_template.encode('utf-8').decode('unicode_escape')
        self.prompt_template_keys = re.findall(r'{(.*?)}', self.prompt_template)

        label_placeholder = f'{{{self.label_key}}}'
        assert (
            self.prompt_template[-len(label_placeholder) :] == label_placeholder
        ), f'{label_placeholder} must be at the end of prompt_template.'

        # Legacy checkpoints has self.truncation_fields = ['context'] and self.prompt_template_keys = ['input', 'output']
        if len(self.truncation_fields) > 0:
            if self.prompt_template_keys[0] == 'input' and self.truncation_fields[0] == 'context':
                self.truncation_fields[0] = self.prompt_template_keys[0]

        assert set(self.truncation_fields).issubset(
            self.prompt_template_keys
        ), f'truncation_fields {self.truncation_fields} must in {self.prompt_template_keys}'

    def _build_samples_mapping(self):
        if self.max_num_samples is not None:
            osm = (
                OnlineSampleMapping(dataset_size=len(self.indexed_dataset), num_samples=self.max_num_samples)
                if not self.global_sample_mapping
                else None
            )
            self.samples_mapping = get_samples_mapping(
                indexed_dataset=self.indexed_dataset,
                data_prefix=self.file_path,
                num_epochs=None,
                max_num_samples=self.max_num_samples,
                max_seq_length=self.max_seq_length - 2,
                short_seq_prob=0,
                seed=self.seed,
                name=self.file_path.split('/')[-1],
                binary_head=False,
                index_mapping_dir=self.index_mapping_dir,
                samples_mapping=osm,
            )
        else:
            self.samples_mapping = None

    def __len__(self):
        if self.max_num_samples is None:
            return len(self.indexed_dataset)
        else:
            return len(self.samples_mapping)

    def __getitem__(self, idx):
        if isinstance(idx, np.int64):
            idx = idx.item()

        if self.samples_mapping is not None:
            assert idx < len(self.samples_mapping)
            idx, _, _ = self.samples_mapping[idx]
            if isinstance(idx, np.uint32):
                idx = idx.item()

        assert idx < len(self.indexed_dataset)
        # idx may < 0 because we pad_samples_to_global_batch_size, e.g. id = -1
        if idx < 0:
            idx = len(self) + idx
            auto_gen_idx = True
        else:
            auto_gen_idx = False
        try:
            example = self.indexed_dataset[idx]
            if auto_gen_idx:
                example['__AUTOGENERATED__'] = True
        except Exception as e:
            logging.error(f"Error while loading example {idx} from dataset {self.file_path}")
            raise e
        return self._process_example(example)

    def _separate_template(self, prompt_template_values: List[str]):
        """
        Combine contexts and label based on prompt_template into a list of strings and a list of keys.

        Args:
            prompt_template_values (List[str]): the list of context and label strings extrated from jsonl file with prompt_template_keys.

        Returns:
            template_strings (List[str]): separated prompt_template with contexts/label placeholder filled with corresponding strings
            template_strings_keys (List[str]): strings point to placeholder keys or <template>

        Examples:
            prompt_template = 'Context:  {context} Question: {question} Answer: {label}'
            prompt_template_values = ['xxx', 'yyy', 'zzz']

            # tokenizer.space_sensitive = True
            template_strings = ['Context:', '  xxx', ' Question:', ' yyy', ' Answer:', ' zzz']

            # tokenizer.space_sensitive = False
            template_strings = ['Context:', ' xxx', 'Question:', 'yyy', 'Answer:', 'zzz']

            template_strings_keys = ['<template>', 'context', '<template>', 'question', '<template>', 'label']
        """
        placeholders = [f'{{{k}}}' for k in self.prompt_template_keys]

        # placeholder to string
        ph_to_s = {ph: s for ph, s in zip(placeholders, prompt_template_values)}
        # placeholder to key
        ph_to_k = {ph: k for ph, k in zip(placeholders, self.prompt_template_keys)}

        # separate prompt_template based on '<space>{placeholder}'
        # examples:
        #   self.prompt_template = "Context:{context}  Passage: {passage}\n\nQuestion:{question} {label}"
        #   template_with_placeholder_separated = ['Context:', '{context}', '  Passage:', ' {passage}', '\n\nQuestion:', '{question}', ' {label}']
        template_with_placeholder_separated = re.split('( *?{.+?})', self.prompt_template)
        template_with_placeholder_separated = [s for s in template_with_placeholder_separated if len(s) > 0]

        # remove space if we have leading space and tokenizer is not space_sensitive
        # space_sensitive = True : tokenizer.text_to_tokens('A{num_spaces}B') = tokenizer.text_to_tokens('A') + tokenizer.text_to_tokens('{num_spaces}B')
        # space_sensitive = False: tokenizer.text_to_tokens('A{num_spaces}B') = tokenizer.text_to_tokens('A') + tokenizer.text_to_tokens('{num_spaces-1}B')
        space_sensitive = getattr(self.tokenizer, 'space_sensitive', False)
        template_with_space_reduced = [
            s[1:] if not space_sensitive and s[0] == ' ' else s for s in template_with_placeholder_separated
        ]

        # convert placeholder to the corresponding string (preserve left spaces) and key
        template_strings, template_strings_keys = [], []
        for t in template_with_space_reduced:
            placeholder = t.lstrip(' ')
            left_spaces = ' ' * (len(t) - len(placeholder))
            template_strings.append(left_spaces + ph_to_s.get(placeholder, placeholder))
            template_strings_keys.append(ph_to_k.get(placeholder, '<template>'))

        return template_strings, template_strings_keys

    def _multiple_truncation(self, template_ids: List[List[int]], template_ids_keys: List[str]):
        """
        Calculate total tokens and truncate multiple contexts in truncation_fields.

        Args:
            template_ids (List[List[int]]): the list of separate prompt_template ids.
            template_ids_keys (List[str]): the list of placeholder keys or <template> (used to check key in truncation_fields).

        Returns:
            context_ids (List[int]): all context ids.
            label_ids (List[int]): all label ids.
        """
        context_ids = template_ids[:-1]
        label_ids = template_ids[-1]
        total_ids = (
            self.virtual_tokens
            + sum(len(ids) for ids in context_ids)
            + max(len(label_ids), self.tokens_to_generate)
            + self.add_bos
            + self.add_sep
            + self.add_eos  # Only training need to consider eos token
        )

        if total_ids > self.max_seq_length:
            truncation_length_total = total_ids - self.max_seq_length
            num_fields = len(self.truncation_fields)
            if num_fields > 0:
                # sorted equal divide length to each field
                # examples:
                #   truncation_length_total = 3
                #   num_fields = 11
                #   truncation_length_list = [3,4,4]
                truncation_length_list = [
                    truncation_length_total // num_fields + (1 if i < truncation_length_total % num_fields else 0)
                    for i in range(num_fields)[::-1]
                ]

                for i, (ids, key) in enumerate(zip(template_ids, template_ids_keys)):
                    if key in self.truncation_fields:
                        truncation_length = truncation_length_list.pop()
                        if len(ids) < truncation_length:
                            logging.warning(f'{key} is not long enough to truncate.')
                            truncation_length = len(ids)

                        truncation_length_total -= truncation_length
                        template_ids[i] = self._truncation(ids, len(ids) - truncation_length)

            if truncation_length_total > 0:
                template_ids_lengths = [len(ids) for ids in template_ids]
                if self.truncation_method == 'left':
                    iters = range(0, len(template_ids_lengths), 1)
                elif self.truncation_method == 'right':
                    iters = range(len(template_ids_lengths) - 1, -1, -1)
                    # We need to truncate more to let context_ids + tokens_to_generate < self.max_seq_length
                    truncation_length_total += min(len(label_ids), self.tokens_to_generate)
                else:
                    raise ValueError(f'{self.truncation_method} is not supported')

                # Iterate all lengths of template_ids.
                for i in iters:
                    if template_ids_lengths[i] >= truncation_length_total:
                        template_ids_lengths[i] -= truncation_length_total
                        template_ids[i] = self._truncation(template_ids[i], template_ids_lengths[i])
                        break
                    else:
                        truncation_length_total -= template_ids_lengths[i]
                        template_ids_lengths[i] = 0
                        template_ids[i] = self._truncation(template_ids[i], template_ids_lengths[i])

        context_ids = [i for ids in template_ids[:-1] for i in ids]
        label_ids = template_ids[-1]
        return context_ids, label_ids

    def _truncation(self, ids, expect_length):
        if expect_length == 0:
            return []
        elif self.truncation_method == 'left':
            return ids[-expect_length:]
        elif self.truncation_method == 'right':
            return ids[:expect_length]
        else:
            raise ValueError(f'{self.truncation_method} is not supported')

    def _process_example(self, example):
        """
        Create an example by concatenating text and answer.
        Truncation is carried out when needed, but it is performed only on the prompt side.
        BOS, EOS, and SEP, are added if specified.
        """
        prompt_template_values = []
        for c in self.prompt_template_keys:
            try:
                prompt_template_values.append(example[c].strip(' '))
            except KeyError as e:
                if c == self.label_key and self.is_test:
                    # allow missing label during testing, if user only wants to do inference without calculating metrics
                    prompt_template_values.append("")
                else:
                    raise e

        template_strings, template_strings_keys = self._separate_template(prompt_template_values)
        template_ids = [self.tokenizer.text_to_ids(s) for s in template_strings]
        context_ids, answer_ids = self._multiple_truncation(template_ids, template_strings_keys)

        if self.virtual_tokens:
            # (@adithyare) we are going to insert "pad/eos" tokens in the beginning of the text and context
            # these pad/eos tokens are placeholders for virtual tokens
            context_ids = [self.tokenizer.eos_id] * self.virtual_tokens + context_ids

        # Adds bos token in the start
        if self.add_bos:
            context_ids = [self.tokenizer.bos_id] + context_ids

        # Adds sep token between text/prompt and answer
        if self.add_sep:
            context_ids = context_ids + [self.sep_id]

        input_ids = context_ids + answer_ids

        # Only training need to consider eos token
        if self.add_eos:
            input_ids = input_ids + [self.tokenizer.eos_id]

        # store metadata in dataset, in case user may have keys required in the prediction json files
        metadata = {k: v for k, v in example.items() if k not in self.prompt_template_keys}
        if self.output_original_text:
            for orig_text, text_key in zip(template_strings, template_strings_keys):
                metadata[text_key] = orig_text

        processed_example = {
            'input_ids': input_ids,
            'answer_start_idx': len(context_ids),
            'context_ids': context_ids,
            'context_length': len(context_ids),
            'answer_ids': answer_ids,
            'metadata': metadata,
            'token_count': len(input_ids),
        }

        return processed_example

    def _maybe_cast_to_list(self, x):
        if isinstance(x, np.ndarray):
            return [item.tolist() for item in x]
        return x

    def _ceil_to_nearest(self, n, m):
        if self.ceil_to_power_2:
            # Reccurent Gemma (AKA Griffin) requires seq length to be a power of 2 for parallel scan
            return 2 ** math.ceil(math.log2(n))
        else:
            return (n + m - 1) // m * m

    def _collate_item(self, item, max_length, pad_id):
        item = self._maybe_cast_to_list(item)
        # max_length = max([len(x) for x in item]) if item else 0
        # here [0] should be tokenizer.pad_id
        item = [x + [pad_id] * (max_length - len(x)) for x in item]
        return item

    def _build_loss_mask(self, processed_example):
        """Pad input_ids in batch to max batch length while building loss mask"""
        input_ids = processed_example['input_ids']
        answer_start_idx = processed_example['answer_start_idx']
        if self.answer_only_loss:
            loss_mask = [float(idx >= answer_start_idx) for idx in range(len(input_ids))]
        else:
            loss_mask = [1.0] * len(input_ids)

        return loss_mask

    @torch.no_grad()
    def _create_attention_mask(self, max_length):
        """Create `attention_mask`.
        Args:
            input_ids: A 1D tensor that holds the indices of tokens.
        """
        # seq_length = len(input_ids)
        # `attention_mask` has the shape of [1, seq_length, seq_length]
        attention_mask = torch.tril(torch.ones((max_length, max_length))).unsqueeze(0)
        attention_mask = attention_mask < 0.5
        return attention_mask

    def collate_fn(self, batch):
        input_ids = [item['input_ids'][:-1] for item in batch]
        labels = [item['input_ids'][1:] for item in batch]
        contexts = [item['context_ids'] for item in batch]
        context_lengths = torch.LongTensor([item['context_length'] for item in batch])
        answers = [item['answer_ids'] for item in batch]
        loss_mask = [self._build_loss_mask(item)[1:] for item in batch]
        metadata = [item['metadata'] for item in batch]
        token_count = [item['token_count'] for item in batch]

        max_length = max(max([len(x) for x in input_ids]), max([len(x) for x in contexts]) + self.tokens_to_generate)
        # increase max length to nearest multiple of 4 or 8
        if self.pad_to_max_length:
            max_length = self.max_seq_length
        else:
            max_length = min(self.max_seq_length, self._ceil_to_nearest(max_length, self.pad_seq_length_to_mult))
        assert max_length <= self.max_seq_length

        if not self.get_attention_mask_from_fusion:
            attention_mask = [self._create_attention_mask(max_length) for _ in batch]
            attention_mask = torch.stack(attention_mask)
        position_ids = [list(range(max_length)) for _ in batch]
        position_ids = torch.LongTensor(position_ids)
        input_ids = torch.LongTensor(
            self._collate_item(input_ids, max_length=max_length, pad_id=self.tokenizer.eos_id)
        )
        labels = torch.LongTensor(self._collate_item(labels, max_length=max_length, pad_id=self.tokenizer.eos_id))
        loss_mask = torch.LongTensor(self._collate_item(loss_mask, max_length=max_length, pad_id=0))
        contexts = torch.LongTensor(self._collate_item(contexts, max_length=max_length, pad_id=self.tokenizer.eos_id))
        answers = torch.LongTensor(self._collate_item(answers, max_length=max_length, pad_id=self.tokenizer.eos_id))

        processed_batch = {
            'tokens': input_ids,
            'labels': labels,
            'loss_mask': loss_mask,
            'position_ids': position_ids,
            'contexts': contexts,
            'context_lengths': context_lengths,
            'answers': answers,
            'metadata': metadata,
            'token_count': token_count,
        }

        if not self.get_attention_mask_from_fusion:
            processed_batch['attention_mask'] = attention_mask

        return processed_batch


class GPTSFTPackedDataset(GPTSFTDataset):
    def __init__(self, file_path: str, tokenizer: TokenizerSpec, return_cu_seqlen: bool = True, **kwargs):
        np.random.seed(kwargs.get('seed', 1234))
        super().__init__(file_path, tokenizer, **kwargs)
        assert self.virtual_tokens == 0, "P-Tuning with packed sequence is not supported."

        # Whether to return `cu_seqlen` to pass to model. This should be true for almost all use cases.
        self.return_cu_seqlen = return_cu_seqlen

    def __getitem__(self, idx):
        if self.samples_mapping is not None:
            # assert idx < len(self.samples_mapping)
            idx = self.samples_mapping[idx]

        input_ids = self.indexed_dataset[idx]['input_ids']
        seq_boundaries = self.indexed_dataset[idx]['seq_start_id'] + [len(input_ids)]
        loss_mask = self.indexed_dataset[idx]['loss_mask']
        if idx < 0:
            loss_mask = [0] * len(loss_mask)
        return {'input_ids': input_ids, 'seq_boundaries': seq_boundaries, 'loss_mask': loss_mask}

    def _load_dataset(self):
        try:
            self.indexed_dataset = np.load(self.file_path, allow_pickle=True)
        except Exception as e:
            logging.error(
                f"Failed to load packed dataset. The dataset should be a `.npy` file. "
                f"Please check if the packed dataset was prepared correctly. The original error was:\n {e}",
            )
            exit(1)

    def _build_samples_mapping(self):
        if self.max_num_samples is not None:
            # custom samples mapping logic, following the format for unpacked sft dataset
            # Note: this is epoch-level shuffling, i.e. sampling without replacement until end of epoch, then repeat.
            # Unpacked dataset shuffles by sampling with replacement indefinitely.
            dataset_len = len(self.indexed_dataset)
            max_num_epochs = np.ceil(self.max_num_samples / dataset_len)
            indices = np.arange(dataset_len)[None, :].repeat(max_num_epochs, axis=0)
            [np.random.shuffle(x) for x in indices]
            self.samples_mapping = indices.reshape(1, -1).squeeze()[: self.max_num_samples]
        else:
            self.samples_mapping = None

    def _build_loss_mask(self, processed_example):
        if self.answer_only_loss:
            seq_boundaries = processed_example['seq_boundaries']
            return np.concatenate(
                [
                    processed_example['loss_mask'][seq_boundaries[i] + 1 : seq_boundaries[i + 1]]
                    for i in range(len(seq_boundaries) - 1)
                ]
            )
        return [1.0] * (len(processed_example['input_ids']) - len(processed_example['seq_boundaries']) + 1)

    def _maybe_cast_to_list(self, x):
        return [item.tolist() if isinstance(item, np.ndarray) else item for item in x]

    def collate_fn(self, batch):
        input_ids = [
            np.concatenate(
                [
                    item['input_ids'][item['seq_boundaries'][i] : item['seq_boundaries'][i + 1] - 1]
                    for i in range(len(item['seq_boundaries']) - 1)
                ]
            )
            for item in batch
        ]
        labels = [
            np.concatenate(
                [
                    item['input_ids'][item['seq_boundaries'][i] + 1 : item['seq_boundaries'][i + 1]]
                    for i in range(len(item['seq_boundaries']) - 1)
                ]
            )
            for item in batch
        ]

        loss_mask = [self._build_loss_mask(item) for item in batch]

        token_count = [item.shape[0] for item in input_ids]

        if self.pad_to_max_length:
            max_length = self.max_seq_length
        else:
            # pad to the nearest multiple of 16 for FP8 training
            # for many datasets in practice, all packed sequence lengths are very close to the
            # target length (2048, 4096, 8192), so there is very minimal padding
            max_length = max(len(l) for l in input_ids)
            max_length = min(self.max_seq_length, self._ceil_to_nearest(max_length, self.pad_seq_length_to_mult))
        assert max_length <= self.max_seq_length

        position_ids: List[List[int]] = []
        cu_seqlens: List[List[int]] = []
        for item in batch:
            position_ids.append([])
            cu_seqlens.append([0])
            seqlens = np.array(item['seq_boundaries'][1:]) - np.array(item['seq_boundaries'][:-1])
            for l in seqlens:
                # length minus 1 because input_ids is truncated by 1 for labels
                position_ids[-1].extend(list(range(l - 1)))
                cu_seqlens[-1].append(cu_seqlens[-1][-1] + l - 1)
            # set last seq to the max seq len because rope and attn kernels expect no padding
            cu_seqlens[-1][-1] = max_length

        assert len(input_ids[0]) == len(
            position_ids[0]
        ), "Dataset problem: input_ids and position_ids lengths don't match"

        input_ids = self._collate_item(input_ids, max_length=max_length, pad_id=self.tokenizer.eos_id)
        labels = self._collate_item(labels, max_length=max_length, pad_id=self.tokenizer.eos_id)
        loss_mask = self._collate_item(loss_mask, max_length=max_length, pad_id=0)
        position_ids = self._collate_item(position_ids, max_length=max_length, pad_id=0)

        processed_batch = {
            'tokens': torch.LongTensor(input_ids),
            'labels': torch.LongTensor(labels),
            'loss_mask': torch.LongTensor(loss_mask),
            'position_ids': torch.LongTensor(position_ids),
            'token_count': token_count,
        }

        if self.return_cu_seqlen:
            cu_seqlens = self._collate_item(cu_seqlens, max_length=max(len(l) for l in cu_seqlens) + 1, pad_id=-1)

            # Pre-generate `cu_seqlens_argmin` and `max_seqlen` as CPU tensor to avoid device-to-host copies.
            cu_seqlens = torch.IntTensor(cu_seqlens)
            cu_seqlens_argmin = torch.argmin(cu_seqlens, dim=1, keepdim=True)
            seqlens = cu_seqlens[:, 1:] - cu_seqlens[:, :-1]
            max_seqlen, _ = seqlens.max(dim=1, keepdim=True)

            processed_batch.update(
                {
                    'attention_mask': torch.LongTensor(
                        [1] * len(input_ids)
                    ),  # no attention mask is needed for packed seq
                    'cu_seqlens': torch.IntTensor(cu_seqlens),  # cu_seqlens_q must be in dtype torch.int32
                    'cu_seqlens_argmin': cu_seqlens_argmin,  # only required for perf
                    'max_seqlen': max_seqlen,  # only required for perf
                }
            )
        else:
            attention_mask = [self._create_attention_mask(max_length) for _ in batch]
            processed_batch.update(
                {
                    'attention_mask': torch.stack(attention_mask),
                }
            )

        return processed_batch
