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

import json
import os
import pickle

import torch
from tqdm.auto import tqdm

from nemo.collections.nlp.modules.common import VirtualPromptSource
from nemo.collections.nlp.modules.common.megatron.utils import build_position_ids
from nemo.core import Dataset
from nemo.utils import AppState, logging

__all__ = ['GPTPromptLearningDataset']


class GPTPromptLearningDataset(Dataset):
    """
    The dataset class for prompt-tuning or p-tuning pretrained GPT models.
    
    Args:
        data (list[strings], list[dicts]): (1) paths to .jsonl or .json files, (2) dict objects corresponding to each input example
        tokenizer (tokenizer): Tokenizer from frozen language model
        virtual_prompt_source (Enum): Either VirtualPromptSource.PROMPT_TABLE or VirtualPromptSource.PROMPT_ENCODER
        task_templates (dict): Dictionary containing all task template information needed to format prompts. Created in the GPTPromptLearningModel class.
        pseudo_tokens (list[strings]): A list of virtual prompt token placeholders e.g [<prompt_1>, <prompt_2>, ...] up to max num virtual tokens
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
        virtual_prompt_source: VirtualPromptSource,
        task_templates: dict,
        pseudo_tokens,
        pad_token_id: int,
        max_seq_length: int,
        min_seq_length: int = 1,
        add_bos: bool = False,
        add_eos: bool = True,
        for_train: bool = True,
        tokens_to_generate=None,
        cache_data_path: str = None,  # the cache file
        load_cache: bool = True,  # whether to load from the cache if it is available
    ):
        self.tokenizer = tokenizer
        self.virtual_prompt_source = virtual_prompt_source
        self.task_templates = task_templates
        self.pseudo_tokens = pseudo_tokens
        self.pseudo_token_ids = set(self.tokenizer.tokens_to_ids(self.pseudo_tokens))
        self.pad_token_id = pad_token_id
        self.max_seq_length = max_seq_length
        self.min_seq_length = min_seq_length
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.for_train = for_train
        self.examples = []

        if not self.for_train:
            self.tokens_to_generate = tokens_to_generate

        assert self.min_seq_length <= max_seq_length, "Min sequence length should be less than or equal to max"
        assert self.max_seq_length > 0, "Max sequence length should be greater than 0"

        logging.info("Loading and tokenizing dataset ... ")

        if load_cache and cache_data_path is not None and os.path.exists(cache_data_path):
            # load it from the cache
            logging.info(f'load the data from the cache file {cache_data_path}')
            with open(cache_data_path, 'rb') as f:
                self.examples = pickle.load(f)
        else:
            # Data is just a list of dicts already loaded from a json file or passed in directly as a dict
            if isinstance(data[0], dict):
                self.load_data(data)

            # Datasets are a list of file path strings to .json or .jsonl files
            elif isinstance(data[0], str):
                for path in data:
                    dataset = open(path, 'r', encoding='utf-8')
                    self.load_data(dataset)
            else:
                raise ValueError("Datasets must be a list of filepath strings or a list of data example dicts")
            if cache_data_path is not None:
                # the first worker save the results into the cache file
                app_state = AppState()
                if app_state._global_rank == 0:
                    with open(cache_data_path, 'wb') as f:
                        pickle.dump(self.examples, f)
                    logging.info(f'save the data to the cache file {cache_data_path}')

    def load_data(self, dataset):
        """
        Loads a dataset by filling in the task templates specified in the config file
        with the information from each training/inference example. Converts all input 
        text into token ids. Also replaces the <|VIRTUAL_PROMPT_#|> placeholders in 
        the task templates with the actual virtual prompt token ids. 

        params:
            dataset: A list of json objects or a dictionary objects each
                     containing the information needed for a training example
        """
        skipped = 0

        for json_line in tqdm(dataset):

            # Read example dict or load the information for a single example from .json file
            if type(json_line) == dict:
                doc = json_line
            else:
                doc = json.loads(json_line)

            taskname = doc["taskname"]
            prompt_template = self.task_templates[taskname]["prompt_template"]
            prompt_template_fields = self.task_templates[taskname]["prompt_template_fields"]
            total_virtual_tokens = self.task_templates[taskname]["total_virtual_tokens"]
            virtual_token_splits = self.task_templates[taskname]["virtual_token_splits"]
            truncation_field = self.task_templates[taskname]['truncate_field']
            answer_only_loss = self.task_templates[taskname]["answer_only_loss"]
            answer_field = self.task_templates[taskname]["answer_field"]

            input_example = prompt_template

            self._input_sanity_checks(
                total_virtual_tokens,
                virtual_token_splits,
                prompt_template,
                prompt_template_fields,
                truncation_field,
                answer_only_loss,
                answer_field,
                doc,
            )

            # Format the input example according to the template
            input_example = self._insert_text_in_template(input_example, prompt_template_fields, doc)
            input_example = self._insert_virtual_token_placeholders(input_example, virtual_token_splits)
            input_ids = self.tokenizer.text_to_ids(input_example)

            # Add BOS/EOS if desired, adds EOS by default
            if self.add_bos:
                input_ids = [self.tokenizer.bos_id] + input_ids
            if self.add_eos:
                input_ids = input_ids + [self.tokenizer.eos_id]

            # Try to truncate input text to fit into the max sequence length
            if len(input_ids) > self.max_seq_length:
                input_ids = self._truncate_input(truncation_field, input_ids, taskname, doc)

            # Skip example if the final length doesn't fit length requirements even after truncation
            if self.min_seq_length <= len(input_ids) <= self.max_seq_length:
                if self.virtual_prompt_source == VirtualPromptSource.PROMPT_ENCODER:
                    taskname_id = self.tokenizer.text_to_ids(taskname)

                elif self.virtual_prompt_source == VirtualPromptSource.PROMPT_TABLE:
                    taskname_id = self.task_templates[taskname]["task_id_num"]

                elif self.virtual_prompt_source == VirtualPromptSource.NO_PROMPT:
                    taskname_id = -1
                else:
                    raise ValueError("Invalid virtual prompt source specified")

                # Find answer field indices if training and answer_only_loss is True
                answer_start_idx = None
                if answer_only_loss and self.for_train:
                    answer_start_idx = self._find_answer_start(taskname, input_ids, answer_field, doc)

                self.examples.append((taskname_id, input_ids, answer_start_idx))
            else:
                skipped += 1

        logging.info(f'Skipped {skipped} sentences, sequence length too short or too long even after truncation')

    def _input_sanity_checks(
        self,
        total_virtual_tokens,
        virtual_token_splits,
        prompt_template,
        prompt_template_fields,
        truncation_field,
        answer_only_loss,
        answer_field,
        doc,
    ):
        # Sanity check amount of virtual token
        assert (
            total_virtual_tokens < self.max_seq_length
        ), "virtual prompt tokens should not exceed max sequence length"

        # Make sure virtual token splits add up to the total number of virtual tokens
        assert (
            sum(virtual_token_splits) == total_virtual_tokens
        ), "Sum of prompt token split values must equal total number of prompt tokens"

        # Make sure number of virtual prompt locations match the number of virtual prompt splits
        assert prompt_template.count('<|VIRTUAL_PROMPT_') == len(
            virtual_token_splits
        ), "The number of '<|VIRTUAL_PROMPT_n|>' markers and the number of prompt token splits must match"

        # Check if input example has fields not present in template
        keys_not_in_template = list(set(doc.keys()) - set(prompt_template_fields) - set(['taskname']))
        assert (
            len(keys_not_in_template) == 0
        ), f"Examples in your dataset contain the fields: {keys_not_in_template} that are not in the task template."

        # Answer field checks
        if answer_only_loss and self.for_train:
            assert answer_field is not None, "If answer_only_loss=True, an answer_field must be given"
            assert (
                answer_field in doc.keys()
            ), f"answer_only_loss=True but the given answer_field '{answer_field}' is not in data json"
            assert truncation_field != answer_field, "Answer field and truncation field should not match"

            answer_placeholder = "{" + answer_field + "}"
            answer_placeholder_len = len(answer_placeholder)
            placeholder_start = len(prompt_template) - answer_placeholder_len
            assert prompt_template[placeholder_start:] == answer_placeholder, "Answer field must be at prompt end"

    def _insert_text_in_template(self, input_example, prompt_template_fields, doc):
        """ Format the input example according to the template """
        for field in prompt_template_fields:
            if field in doc.keys():
                field_text = doc[field]
                input_example = input_example.replace('{' + field + '}', field_text)

            # If some fields from the template aren't present, e.g. {answer} during inference
            # just remove that field from the template, leaving the space blank
            else:
                input_example = input_example.replace('{' + field + '}', "")
                input_example = input_example.strip()

        return input_example

    def _insert_virtual_token_placeholders(self, input_example, virtual_token_splits):
        """ Insert the correct number of pseudo tokens at the <|VIRTUAL_PROMPT_n|> markers """
        total_inserted_tokens = 0

        for idx in range(len(virtual_token_splits)):
            split_start = total_inserted_tokens
            split_end = total_inserted_tokens + virtual_token_splits[idx]
            pseudo_tokens_for_split = "".join(self.pseudo_tokens[split_start:split_end])
            input_example = input_example.replace(f'<|VIRTUAL_PROMPT_{idx}|>', pseudo_tokens_for_split)
            total_inserted_tokens = split_end

        return input_example

    def _truncate_input(self, truncation_field, input_ids, taskname, doc):
        """ Try to truncate input text to fit into the max sequence length """
        logging.info(
            f"Input greater than max sequence length. Attempting to truncate: '{truncation_field}' in task: '{taskname}'"
        )

        # Truncate the text ids in this part of input to try and fit max sequence length
        if truncation_field is not None and truncation_field in doc.keys():
            truncation_length = len(input_ids) - self.max_seq_length
            field_text = doc[truncation_field]
            field_text = self._add_leading_space(taskname, truncation_field, field_text)

            # Truncate field text
            field_text_ids = self.tokenizer.text_to_ids(field_text)
            truncated_text_ids = field_text_ids[: -min(truncation_length, len(field_text_ids))]

            # Replace original text ids with truncated text ids
            field_start, field_end = find_subsequence_location(input_ids, field_text_ids)
            input_ids = input_ids[:field_start] + truncated_text_ids + input_ids[field_end + 1 :]

        return input_ids

    def _find_answer_start(self, taskname, input_ids, answer_field, doc):
        """ Find the token ids corresponding to the answer start, for loss masking purposes.
            Assumes the answer is always at the end of the prompt.
        """
        answer_text = doc[answer_field]
        answer_text = self._add_leading_space(taskname, answer_field, answer_text)
        answer_text_ids = self.tokenizer.text_to_ids(answer_text)
        num_answer_text_ids = len(answer_text_ids)

        if self.add_eos:
            num_answer_text_ids += 1

        answer_start_idx = len(input_ids) - num_answer_text_ids

        return answer_start_idx

    def _add_leading_space(self, taskname, field_name, field_text):
        """ Add leading space to text if there is a space before it in the template """
        prompt_template = self.task_templates[taskname]["prompt_template"]
        field_text_start = prompt_template.find("{" + field_name + "}")
        if field_text_start != 0 and prompt_template[field_text_start - 1] == " ":
            field_text = " " + field_text

        return field_text

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

    def collate_fn(self, batch, tp_workers=0):
        """ Prepares input_ids, labels, loss mask, attention_mask, and position ids for global batch """
        taskname_ids, input_ids, answer_starts = zip(*batch)

        # Pad taskname_ids to be the same length for the prompt encoder
        if self.virtual_prompt_source == VirtualPromptSource.PROMPT_ENCODER:
            max_taskname_length = max(len(ids) for ids in taskname_ids)
            taskname_ids = [ids + [self.pad_token_id] * (max_taskname_length - len(ids)) for ids in taskname_ids]
            taskname_ids = torch.tensor(taskname_ids)

        # Task ids are just used for a look up embeddings for prompt-table
        elif self.virtual_prompt_source in [VirtualPromptSource.PROMPT_TABLE, VirtualPromptSource.NO_PROMPT]:
            taskname_ids = torch.tensor(taskname_ids)

        # Get max sequence length of batch
        batch_max = max(len(ids) for ids in input_ids)

        if tp_workers > 1:
            # more sure the sequence length is multiply of number of tp_workers, needed for sequence parallel.
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

        # Using causal attention mask for whole input
        batch_size = len(input_ids)
        attention_mask = torch.tril(torch.ones((batch_size, batch_max, batch_max))).view(
            batch_size, 1, batch_max, batch_max
        )

        # Convert attention mask from float to bool
        attention_mask = attention_mask < 0.5
        position_ids = build_position_ids(input_ids)

        return input_ids, labels, loss_mask, position_ids, attention_mask, taskname_ids

    def pad_batch_and_build_loss_mask(self, input_ids, batch_max, answer_starts):
        """ Pad input_ids in batch to max batch length while building loss mask """
        batch_loss_masks = []
        padded_input_ids = []
        for ids, answer_start_idx in zip(input_ids, answer_starts):
            if answer_start_idx is not None:
                # Loss mask where answer tokens are 1.0 and all other tokens are 0.0
                loss_mask = [float(idx >= answer_start_idx) for idx in range(len(ids))]
            else:
                # Loss mask where virtual tokens are 0.0 and all other tokens are 1.0
                loss_mask = [float(token_id not in self.pseudo_token_ids) for token_id in ids]

            # Pad to max length
            input_length = len(ids)
            padding_length = batch_max - input_length
            pad_extend = [self.pad_token_id] * padding_length
            ids = ids + pad_extend
            padded_input_ids.append(ids)

            # Account for padding in loss mask
            loss_mask.extend([0.0] * padding_length)
            batch_loss_masks.append(torch.tensor(loss_mask, dtype=torch.float))

        # Make into torch tensors
        padded_input_ids = torch.tensor(padded_input_ids, dtype=torch.long)
        batch_loss_masks = torch.stack(batch_loss_masks)

        return padded_input_ids, batch_loss_masks

    def inference_collate_fn(self, batch):
        """
        Used for loading inference data. 
        """
        task_id_nums, input_ids, answer_starts = zip(*batch)
        input_lengths = torch.cuda.LongTensor([len(inputs) for inputs in input_ids])
        task_id_nums = torch.cuda.LongTensor(task_id_nums)
        batch_max = input_lengths.max().item()
        batch_max += self.tokens_to_generate

        input_ids, _ = self.pad_batch_and_build_loss_mask(input_ids, batch_max, answer_starts)
        input_ids = input_ids.cuda()
        input_ids = torch.cuda.LongTensor(input_ids)

        return task_id_nums, (input_ids, input_lengths)


def find_subsequence_location(sequence, subsequence):
    """ Finds the start and end index of the first occurance 
        of a given subsequence within a larger list. Returns 
        the two indices corresponding to the postition of 
        the first and last token of the subseqeunce.
        Assumes subsequence is known to be in sequence. 
    """
    assert len(sequence) >= len(subsequence), "subsequence too long"

    start_idx = None
    next_subseq_token = subsequence[0]
    next_subsequence_idx = 1

    for seq_idx, token in enumerate(sequence):
        if token == next_subseq_token:
            if start_idx is None:
                start_idx = seq_idx

            if next_subsequence_idx == len(subsequence):
                end_idx = seq_idx
                return start_idx, end_idx
            else:
                next_subseq_token = subsequence[next_subsequence_idx]
                next_subsequence_idx += 1
        else:
            start_idx = None
            next_subseq_token = subsequence[0]
            next_subsequence_idx = 1

    raise ValueError("Subsequence not found in sequence")
