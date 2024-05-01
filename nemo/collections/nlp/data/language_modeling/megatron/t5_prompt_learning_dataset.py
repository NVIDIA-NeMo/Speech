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

import torch
from tqdm.auto import tqdm

from nemo.collections.nlp.data.language_modeling.megatron.base_prompt_learning_dataset import BasePromptLearningDataset
from nemo.collections.nlp.models.language_modeling.megatron_t5_model import T5Sentinel
from nemo.collections.nlp.modules.common import VirtualPromptSource
from nemo.collections.nlp.modules.common.megatron.utils import build_position_ids
from nemo.utils import logging

__all__ = ['T5PromptLearningDataset']


class T5PromptLearningDataset(BasePromptLearningDataset):
    """
    The dataset class for prompt-tuning or p-tuning pretrained T5 models.
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
        decoder_starts_with_pad: bool = False,
        add_eos_to_decoder_output: bool = True,
        add_sentinel_to_input: bool = True,
        ul2_prompt_token: str = None,
    ):
        # These two variables need to be set before calling super().__init__() because the parent class calls `load_data()` which requires these attributes.
        self.decoder_starts_with_pad = decoder_starts_with_pad
        self.add_eos_to_decoder_output = add_eos_to_decoder_output
        self.add_sentinel_to_input = add_sentinel_to_input
        self.ul2_prompt_token = ul2_prompt_token
        super().__init__(
            datasets=datasets,
            tokenizer=tokenizer,
            virtual_prompt_source=virtual_prompt_source,
            task_templates=task_templates,
            pseudo_tokens=pseudo_tokens,
            pad_token_id=pad_token_id,
            max_seq_length=max_seq_length,
            min_seq_length=min_seq_length,
            add_bos=add_bos,
            add_eos=add_eos,
            for_train=for_train,
        )

    def load_data(self, dataset):
        """
        Loads a dataset by filling in the task templates specified in the config file
        with the information from each training/inference example. Converts all input 
        text into token ids. Also replaces the ``<|VIRTUAL_PROMPT_#|>`` placeholders in
        the task templates with the actual virtual prompt token ids. 

        Args:
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
            answer_field = self.task_templates[taskname]["answer_field"]

            input_example = prompt_template

            self._input_sanity_checks(
                total_virtual_tokens=total_virtual_tokens,
                virtual_token_splits=virtual_token_splits,
                prompt_template=prompt_template,
                prompt_template_fields=prompt_template_fields,
                truncation_field=truncation_field,
                answer_field=answer_field,
                doc=doc,
            )

            # Format the input example according to the template
            input_example = self._insert_text_in_template(input_example, prompt_template_fields, doc, answer_field)
            input_example = self._insert_virtual_token_placeholders(input_example, virtual_token_splits)

            # a trick to align with the data format in t5 pretraining
            input_ids = self.tokenizer.text_to_ids(input_example)
            if self.add_sentinel_to_input:
                input_ids = input_ids + self.tokenizer.text_to_ids(T5Sentinel.FIRST.value)

            # Add BOS/EOS to the input of encoder if desired, adds EOS by default
            if self.ul2_prompt_token is not None:
                ul2_prompt_token_id = self.tokenizer.text_to_ids(self.ul2_prompt_token)
                assert len(ul2_prompt_token_id) == 1
                input_ids = ul2_prompt_token_id + input_ids
            if self.add_bos:
                input_ids = [self.tokenizer.bos_id] + input_ids
            if self.add_eos:
                input_ids = input_ids + [self.tokenizer.eos_id]

            # Try to truncate input text to fit into the max sequence length
            if len(input_ids) > self.max_seq_length:
                input_ids = self._truncate_input(truncation_field, input_ids, taskname, doc, total_virtual_tokens)

            # get answer ids
            if answer_field in doc.keys():  # training and validation
                answer_text = doc[answer_field]

                if self.decoder_starts_with_pad:
                    answer_text_ids = [self.tokenizer.pad_id]
                else:
                    answer_text_ids = [self.tokenizer.bos_id]
                # a trick to align with the data format in t5 pretraining
                if self.add_sentinel_to_input:
                    answer_text_ids += self.tokenizer.text_to_ids(T5Sentinel.FIRST.value)
                answer_text_ids += self.tokenizer.text_to_ids(answer_text)
                if self.add_eos_to_decoder_output:
                    answer_text_ids += [self.tokenizer.eos_id]
                else:
                    answer_text_ids += self.tokenizer.text_to_ids(T5Sentinel.END.value)

            # Skip example if the final length doesn't fit length requirements even after truncation
            if self.min_seq_length <= len(input_ids) <= self.max_seq_length:
                if self.virtual_prompt_source == VirtualPromptSource.PROMPT_ENCODER:
                    taskname_id = self.tokenizer.text_to_ids(taskname)
                elif (
                    self.virtual_prompt_source == VirtualPromptSource.NO_PROMPT
                ):  # TODO (@adithyare) this class and GPTPromptLearningDataset should be merged.
                    taskname_id = -1
                else:
                    raise ValueError("Invalid virtual prompt source specified")

                dec_input = None
                dec_labels = None

                if answer_field in doc.keys():  # training and validation
                    dec_input = answer_text_ids[:-1]
                    dec_labels = answer_text_ids[1:]

                self.examples.append((taskname_id, input_ids, dec_input, dec_labels))
            else:
                skipped += 1

        logging.info(f'Skipped {skipped} sentences, sequence length too short or too long even after truncation')

    def _insert_text_in_template(self, input_example, prompt_template_fields, doc, answer_field):
        """ Format the input example according to the template """
        for field in prompt_template_fields:
            # discard the last one, {label} / {answer}
            # Or if some fields from the template aren't present, e.g. {answer} during inference
            # just remove that field from the template, leaving the space blank
            if field == answer_field or field not in doc.keys():
                input_example = input_example.replace('{' + field + '}', "")

            else:
                field_text = doc[field]
                input_example = input_example.replace('{' + field + '}', field_text)

        return input_example.strip(" ")

    def collate_fn(self, batch):
        """ Prepares enc_input, dec_input, labels, loss_mask, enc_mask, dec_mask, position_ids, taskname_ids for global batch """

        taskname_ids, enc_input, dec_input, dec_labels = zip(*batch)

        taskname_ids = self.pad_taskname_ids(taskname_ids)

        enc_input, dec_input, labels, loss_mask, enc_mask, dec_mask = self.pad_batch_and_build_loss_mask(
            enc_input, dec_input, dec_labels
        )

        position_ids = build_position_ids(enc_input).contiguous()

        return enc_input, dec_input, labels, loss_mask, enc_mask, dec_mask, position_ids, taskname_ids

    def pad_batch_and_build_loss_mask(self, enc_input, dec_input, dec_labels):
        """ Pad enc_input, dec_input, labels in batch to max batch length while building loss_mask, enc_mask, and dec_mask """

        # have labels (during training and validation)
        if dec_input[0] and dec_labels[0]:
            max_dec_input_length = max([len(item) for item in dec_input]) if dec_input[0] else 0
            max_label_length = max([len(item) for item in dec_labels]) if dec_labels[0] else 0

            loss_mask = [([1] * (len(item))) + ([0] * (max_label_length - len(item))) for item in dec_labels]
            dec_input = [item + [self.tokenizer.pad_id] * (max_dec_input_length - len(item)) for item in dec_input]
            labels = [item + [self.tokenizer.pad_id] * (max_label_length - len(item)) for item in dec_labels]

            dec_input = torch.LongTensor(dec_input).contiguous()
            labels = torch.LongTensor(labels).contiguous()
            loss_mask = torch.LongTensor(loss_mask).contiguous()

            dec_mask = (dec_input != self.tokenizer.pad_id).long().contiguous()

        # during inference
        else:
            dec_input, labels, loss_mask, dec_mask = None, None, None, None

        # for all training, validation, and inference
        max_enc_query_length = max([len(item) for item in enc_input]) if enc_input[0] else 0
        enc_input = [item + [self.tokenizer.pad_id] * (max_enc_query_length - len(item)) for item in enc_input]
        enc_input = torch.LongTensor(enc_input).contiguous()

        enc_mask = (enc_input != self.tokenizer.pad_id).long().contiguous()

        return enc_input, dec_input, labels, loss_mask, enc_mask, dec_mask
