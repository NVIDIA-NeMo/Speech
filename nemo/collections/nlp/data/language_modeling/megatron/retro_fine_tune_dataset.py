# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

from random import sample

import numpy as np
import torch

from nemo.collections.nlp.data.language_modeling.megatron.dataset_utils import get_samples_mapping
from nemo.collections.nlp.data.language_modeling.text_memmap_dataset import JSONLMemMapDataset
from nemo.core import Dataset
from nemo.utils import logging

__all__ = ['RetroQAFineTuneDataset']


class RetroQAFineTuneDataset(Dataset):
    """
    The dataset class for fine tune RETRO models.

    Args:
        data (list[strings], list[dicts]): (1) paths to .jsonl or .json files, (2) dict objects corresponding to each input example
        tokenizer (tokenizer): Tokenizer from frozen language model
        task_templates (dict): Dictionary containing all task template information needed to format prompts. Created in the GPTPromptLearningModel class.
        pad_token_id (int): ID of pad token from tokenizer
        max_seq_length (int): maximum sequence length for each dataset examples. Examples will either be truncated to fit this length or dropped if they cannot be truncated.
        min_seq_length (int): min length of each data example in the dataset. Data examples will be dropped if they do not meet the min length requirements. 
        add_bos (bool): Whether to add a beginning of sentence token to each data example
        add_eos (bool): Whether to add an end of sentence token to each data example
        for_train (bool): Whether you're creating a dataset for training or inference
        tokens_to_generate (int): (inference only) Number of tokens to generate during inference
    """

    def __init__(
        self,
        data,
        tokenizer,
        answer_only_loss: bool,
        pad_token_id: int,
        max_seq_length: int,
        add_bos: bool = False,
        add_eos: bool = True,
        max_num_samples: int = None,
        seed: int = 1234,
        neighbors: int = 20,
    ):
        self.tokenizer = tokenizer
        self.pad_token_id = pad_token_id
        self.max_seq_length = max_seq_length
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.answer_only_loss = answer_only_loss
        self.max_num_samples = max_num_samples
        self.seed = seed
        self.neighbors = neighbors

        assert self.max_seq_length > 0, "Max sequence length should be greater than 0"

        logging.info("Loading and tokenizing dataset ... ")

        self.indexed_dataset = JSONLMemMapDataset(dataset_paths=[data], tokenizer=None, header_lines=0, workers=12)
        # Will be None after this call if `max_num_samples` is None
        self._build_samples_mapping(data)

    def _build_samples_mapping(self, file_path):
        if self.max_num_samples is not None:
            self.samples_mapping = get_samples_mapping(
                indexed_dataset=self.indexed_dataset,
                data_prefix=file_path,
                num_epochs=None,
                max_num_samples=self.max_num_samples,
                max_seq_length=self.max_seq_length - 2,
                short_seq_prob=0,
                seed=self.seed,
                name=file_path.split('/')[-1],
                binary_head=False,
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
        example = self.indexed_dataset[idx]
        return self._process_example(example)

    def _process_example(self, example):
        """
        Process a single example from the dataset into IDs and other T0-related metadata.
        """
        question = example['question'].strip()
        tokenized_input = self.tokenizer.text_to_ids(f"question: {question}\n")
        # add a space between input and output
        if 'answers' in example:
            # sample one answer from answers
            answer = sample(example['answers'], 1)[0].strip()
            tokenized_output = self.tokenizer.text_to_ids(f"answer: {answer}")
        else:
            tokenized_output = self.tokenizer.text_to_ids('answer: ')

        bos_id = self.tokenizer.bos_id
        if self.add_bos:
            tokenized_input = [bos_id] + tokenized_input
        if self.add_eos:
            target = tokenized_output + [self.tokenizer.eos_id]
        else:
            target = tokenized_output

        # pad the question so 'answer:' coincides with the end of the first chunk of 64
        if len(tokenized_input) < 64:
            padding_length = 64 - len(tokenized_input)
            tokenized_input = [self.pad_token_id] * padding_length + tokenized_input

        if len(tokenized_input) + len(target) > self.max_seq_length:
            cut_tokens = len(tokenized_input) + len(target) - self.max_seq_length
            if len(tokenized_input) - cut_tokens > 0:
                # cut the input by default
                tokenized_input = tokenized_input[: len(tokenized_input) - cut_tokens]
            elif len(target) - cut_tokens > 0:
                # cut the output
                target = target[: len(target) - cut_tokens]
            else:
                # cut both the input and output
                cut_input_tokens = len(tokenized_input) - 1  # retain at least one token
                cut_output_tokens = cut_tokens - cut_input_tokens
                tokenized_input = tokenized_input[: len(tokenized_input) - cut_input_tokens]
                target = target[: len(target) - cut_output_tokens]

        chunks = []
        contexts = example['ctxs']
        assert self.neighbors <= len(
            contexts
        ), f"specify {self.neighbors}, but only provide {len(contexts)} neighbors in the dataset"
        for neighbor in contexts[: self.neighbors]:
            tokens = self.tokenizer.text_to_ids(neighbor)
            tokens = tokens[:128]
            if len(tokens) < 128:
                tokens = tokens + [self.pad_token_id] * (128 - len(tokens))
            chunks.append(tokens)

        answer_start_idx = len(tokenized_input)
        input_ids = tokenized_input + target
        assert len(input_ids) <= 128, "cannot handle more than two chunks yet"
        chunks = np.array(chunks).reshape(1, self.neighbors, -1).astype(np.int64)
        results = (input_ids, answer_start_idx, chunks)
        return results

    def collate_fn(self, batch, tp_workers=0):
        """ Prepares input_ids, labels, loss mask, attention_mask, and position ids for global batch """
        input_ids, answer_starts, chunks = zip(*batch)
        # convert chunks into torch tensors
        chunks = torch.tensor(chunks)

        # Get max sequence length of batch
        batch_max = max(len(ids) for ids in input_ids)

        if tp_workers > 1:
            # make sure the sequence length is multiply of number of tp_workers, needed for sequence parallel.
            resi_padding = (tp_workers - (batch_max - 1) % tp_workers) % tp_workers
        else:
            resi_padding = 0
        batch_max += resi_padding
        input_ids, loss_mask = self.pad_batch_and_build_loss_mask(input_ids, batch_max, answer_starts)
        # Should be a label for every token in batch, label is the next token
        labels = input_ids[:, 1:].contiguous()
        input_ids = input_ids[:, :-1].contiguous()
        batch_max -= 1

        # Loss mask should align with labels
        loss_mask = loss_mask[:, 1:].contiguous()

        hidden_mask = input_ids != self.pad_token_id
        context_mask = chunks != self.pad_token_id

        # Using causal attention mask for whole input

        return {
            'tokens': input_ids,
            'labels': labels,
            'tokens_mask': hidden_mask,
            'loss_mask': loss_mask,
            'retrieved_emb_mask': context_mask,
            'retrieved_ids': chunks,
        }

    def pad_batch_and_build_loss_mask(self, input_ids, batch_max, answer_starts):
        """ Pad input_ids in batch to max batch length while building loss mask """
        batch_loss_masks = []
        padded_input_ids = []
        for ids, answer_start_idx in zip(input_ids, answer_starts):
            if self.answer_only_loss and answer_start_idx is not None:
                # Loss mask where answer tokens are 1.0 and all other tokens are 0.0
                loss_mask = [float(idx >= answer_start_idx) for idx in range(len(ids))]
            else:
                # Loss mask where virtual tokens are 0.0 and all other tokens are 1.0
                loss_mask = [1.0] * len(ids)
            # Pad to max length
            input_length = len(ids)
            padding_length = batch_max - input_length
            ids = ids + [self.pad_token_id] * padding_length
            padded_input_ids.append(ids)

            # Account for padding in loss mask
            loss_mask.extend([0.0] * padding_length)
            batch_loss_masks.append(torch.tensor(loss_mask, dtype=torch.float))

        # Make into torch tensors
        padded_input_ids = torch.tensor(padded_input_ids, dtype=torch.long)
        batch_loss_masks = torch.stack(batch_loss_masks)

        return padded_input_ids, batch_loss_masks
