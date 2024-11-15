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

import omegaconf
import torch

from nemo.collections.nlp.modules.common import VirtualPromptSource
from nemo.core import Dataset
from nemo.utils import logging
from nemo.utils.decorators import deprecated_warning

__all__ = ['BasePromptLearningDataset']


class BasePromptLearningDataset(Dataset):
    """
    The base dataset class for prompt-tuning or p-tuning.
    TODO: (@adithyare) should be merged into GPTPromptLearningDataset
    """

    def __init__(
        self,
        datasets,
        tokenizer,
        virtual_prompt_source: VirtualPromptSource,
        task_templates: dict,
        pseudo_tokens,
        pad_token_id: str,
        max_seq_length: int,
        min_seq_length: int = 1,
        add_bos: bool = False,
        add_eos: bool = True,
        for_train: bool = True,
    ):
        # deprecation warning
        deprecated_warning("BasePromptLearningDataset")

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

        assert self.min_seq_length <= max_seq_length, "Min sequence length should be less than or equal to max"
        assert self.max_seq_length > 0, "Max sequence length should be greater than 0"

        logging.info("Loading and tokenizing dataset ... ")

        # Datasets is just a list of json dicts
        if isinstance(datasets[0], dict):
            self.load_data(datasets)

        # Datasets are a list of file path strings to .json or .jsonl files
        elif isinstance(datasets[0], str):
            for path in datasets:
                with open(path, 'r', encoding='utf-8') as dataset:
                    dataset_examples = self.load_data(dataset)
                self.examples.extend(dataset_examples)
        elif isinstance(datasets[0], omegaconf.ListConfig) or isinstance(datasets[0], list):
            # Dataset is a list of tuples with the first element being the probability of sampling from the dataset
            # This code repeates the smaller datasets to approximately match the target probabilities
            total_examples = 0
            dataset_lengths = []
            target_probs = []
            datasets_examples_list = []
            for prob_and_path in datasets:
                prob = prob_and_path[0]
                path = prob_and_path[1]
                with open(path, 'r', encoding='utf-8') as dataset:
                    dataset_examples = self.load_data(dataset)
                datasets_examples_list.append(dataset_examples)
                dataset_lengths.append(len(dataset_examples))
                total_examples += len(dataset_examples)
                target_probs.append(prob)

            # Normalize the target probs
            target_probs = [prob / sum(target_probs) for prob in target_probs]
            current_probs = [dataset_lengths[i] / total_examples for i in range(len(dataset_lengths))]

            # Increase number of examples needed without reducing the larger datasets with low target probs
            new_total_examples = total_examples
            for dataset_idx in range(len(datasets)):
                if target_probs[dataset_idx] < current_probs[dataset_idx]:
                    target_total_examples = int(dataset_lengths[dataset_idx] / target_probs[dataset_idx])
                    new_total_examples = max(new_total_examples, target_total_examples)

            final_total_examples = 0
            final_dataset_lengths = []
            for dataset_idx in range(len(datasets)):
                num_samples_required = int(new_total_examples * target_probs[dataset_idx])
                num_repeat = max(
                    int(round(num_samples_required // dataset_lengths[dataset_idx])), 1
                )  # At least 1 repeat
                logging.info("dataset idx {}, num_repeat {}".format(dataset_idx, num_repeat))
                dataset_examples_repeated = datasets_examples_list[dataset_idx] * num_repeat
                final_dataset_lengths.append(len(dataset_examples_repeated))
                final_total_examples += len(dataset_examples_repeated)
                self.examples.extend(dataset_examples_repeated)

            final_probs = [final_dataset_lengths[i] / final_total_examples for i in range(len(final_dataset_lengths))]
            logging.info("Target probs: {}".format(target_probs))
            logging.info("Final probs: {}".format(final_probs))
            logging.info("Initial total examples: {}".format(total_examples))
            logging.info("Final total examples: {}".format(final_total_examples))
        else:
            raise ValueError("Datasets must be a list of dicts or a list of filepath strings")

    def _insert_virtual_token_placeholders(self, input_example, virtual_token_splits):
        """Insert the correct number of pseudo tokens at the <|VIRTUAL_PROMPT_n|> markers"""
        total_inserted_tokens = 0

        for idx in range(len(virtual_token_splits)):
            split_start = total_inserted_tokens
            split_end = total_inserted_tokens + virtual_token_splits[idx]
            pseudo_tokens_for_split = "".join(self.pseudo_tokens[split_start:split_end])
            input_example = input_example.replace(f'<|VIRTUAL_PROMPT_{idx}|>', pseudo_tokens_for_split)
            total_inserted_tokens = split_end

        return input_example

    def _truncate_input(self, truncation_field, input_ids, taskname, doc, total_virtual_tokens=0):
        """Try to truncate input text to fit into the max sequence length"""
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
        else:
            if not self.for_train:
                # Hack alert! Slash and burn
                #  @TODO (@adithyare) need a more graceful truncation here, we should not skip examples in test
                input_ids = (
                    input_ids[:total_virtual_tokens]
                    + input_ids[total_virtual_tokens:][-self.max_seq_length + total_virtual_tokens :]
                )

        return input_ids

    def _add_leading_space(self, taskname, field_name, field_text):
        """Add leading space to text if there is a space before it in the template"""
        prompt_template = self.task_templates[taskname]["prompt_template"]
        field_text_start = prompt_template.find("{" + field_name + "}")
        if field_text_start != 0 and prompt_template[field_text_start - 1] == " ":
            field_text = " " + field_text

        return field_text

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

    def _input_sanity_checks(
        self,
        total_virtual_tokens,
        virtual_token_splits,
        prompt_template,
        prompt_template_fields,
        truncation_field,
        answer_field,
        doc,
        answer_only_loss=None,
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

        # Check answer field
        if self.for_train:
            assert answer_field is not None, "An answer_field must be given"
            assert answer_field in doc.keys(), f"The given answer_field '{answer_field}' is not in data json"
            assert truncation_field != answer_field, "Answer field and truncation field should not match"

            answer_placeholder = "{" + answer_field + "}"
            answer_placeholder_len = len(answer_placeholder)
            placeholder_start = len(prompt_template) - answer_placeholder_len
            assert prompt_template[placeholder_start:] == answer_placeholder, "Answer field must be at prompt end"

    def pad_taskname_ids(self, taskname_ids):
        # Pad taskname_ids to be the same length for the prompt encoder
        if self.virtual_prompt_source == VirtualPromptSource.PROMPT_ENCODER:
            max_taskname_length = max(len(ids) for ids in taskname_ids)
            taskname_ids = [ids + [self.pad_token_id] * (max_taskname_length - len(ids)) for ids in taskname_ids]
            taskname_ids = torch.tensor(taskname_ids)

        # Task ids are just used for a look up embeddings for prompt-table
        elif self.virtual_prompt_source == VirtualPromptSource.NO_PROMPT:
            taskname_ids = torch.tensor(taskname_ids)

        return taskname_ids


def find_subsequence_location(sequence, subsequence):
    """Finds the start and end index of the first occurance
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
