# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# Copyright 2019 The Google Research Authors.
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
This file contains code artifacts adapted from the original implementation:
https://github.com/google-research/google-research/blob/master/schema_guided_dst
"""

import os
import re
from typing import List

import numpy as np

from nemo.collections.nlp.data.dialogue.dataset.dialogue_dataset import DialogueDataset
from nemo.collections.nlp.data.dialogue.input_example.sgd_input_example import SGDInputExample

__all__ = ['DialogueSGDBERTDataset']


class DialogueSGDBERTDataset(DialogueDataset):
    '''
    Dataset Class 
        1. Performs Model-dependent (but Data-independent) operations (tokenization etc)
        2. This can allow the same model preprocessing for multiple datasources
        3. Users can configurate which labels to use for modelling 
            (e.g. intent classification, slot filling or both together etc)
    '''

    def __init__(self, dataset_split: str, dialogues_processor: object, tokenizer, schemas, schema_config, cfg):
        """ Constructor
        Args:
            dataset_split: dataset split
            dialogues_processor: Data generator for SGD dialogues
            tokenizer: tokenizer
            schemas: SGD schema for domain, intent and slots
            schema_config: config dict for schemas
            cfg: cfg container for dataset
        """
        self.dataset_split = dataset_split
        self.tokenizer = tokenizer
        self.schemas = schemas
        self.schema_config = schema_config
        self.dialogues_processor = dialogues_processor
        self.cfg = cfg
        self.subsample = self.dialogues_processor._subsample

        dial_file = f"{dialogues_processor._task_name}_{dataset_split}_examples_bert.processed"
        self.dial_file = os.path.join(self.cfg.data_dir, dial_file)
        if self.cfg.use_cache and os.path.exists(self.dial_file):
            self.load_features()
        else:
            self.process_features()
            self.save_features()

    def load_features(self):
        with open(self.dial_file, "rb") as f:
            self.features = np.load(f, allow_pickle=True)

    def process_features(self):
        self.features = []
        self.raw_features = self.dialogues_processor.get_dialog_examples(self.dataset_split)
        for idx in range(len(self.raw_features)):
            self.bert_process_one_sample(idx)

    def save_features(self):
        with open(self.dial_file, "wb") as f:
            np.save(f, self.features)

    def _tokenize(self, utterance: str):
        """
        Tokenize the utterance

        Args:
            utterance: A string containing the utterance to be tokenized.

        Returns:
            bert_tokens: A list of tokens obtained by word-piece tokenization of the
                utterance.
            alignments: A dict mapping indices of characters corresponding to start
                and end positions of words (not subwords) to corresponding indices in
                bert_tokens list.
            inverse_alignments: A list of size equal to bert_tokens. Each element is a
                tuple containing the index of the starting and inclusive ending
                character of the word corresponding to the subword. This list is used
                during inference to map word-piece indices to spans in the original
                utterance.
        """
        # utterance = tokenization.convert_to_unicode(utterance)

        # After _naive_tokenize, spaces and punctuation marks are all retained, i.e.
        # direct concatenation of all the tokens in the sequence will be the
        # original string.
        tokens = DialogueSGDBERTDataset._naive_tokenize(utterance)
        # ['I', ' ', 'am', ' ', 'feeling', ' ', 'hungry', ' ', 'so', ' ', 'I', ' ', 'would', ' ', 'like', ' ', 'to', ' ', 'find', ' ', 'a', ' ', 'place', ' ', 'to', ' ', 'eat', '.']
        # Filter out empty tokens and obtain aligned character index for each token.
        alignments = {}
        char_index = 0
        bert_tokens = (
            []
        )  # ['I', 'am', 'feeling', 'hungry', 'so', 'I', 'would', 'like', 'to', 'find', 'a', 'place', 'to', 'eat', '.']
        # These lists store inverse alignments to be used during inference.
        bert_tokens_start_chars = []
        bert_tokens_end_chars = []
        for token in tokens:
            if token.strip():
                subwords = self.tokenizer.text_to_tokens(token)
                # Store the alignment for the index of starting character and the
                # inclusive ending character of the token.
                alignments[char_index] = len(bert_tokens)
                bert_tokens_start_chars.extend([char_index] * len(subwords))
                bert_tokens.extend(subwords)
                # The inclusive ending character index corresponding to the word.
                inclusive_char_end = char_index + len(token) - 1
                alignments[inclusive_char_end] = len(bert_tokens) - 1
                bert_tokens_end_chars.extend([inclusive_char_end] * len(subwords))
            char_index += len(token)
        inverse_alignments = list(zip(bert_tokens_start_chars, bert_tokens_end_chars))
        return bert_tokens, alignments, inverse_alignments

    @classmethod
    def _naive_tokenize(cls, s: str):
        """
        Tokenizes a string, separating words, spaces and punctuations.
        Args:
            s: a string
        Returns:
            seq_tok: list of words, spaces and punctuations from the string
        """
        # Spaces and punctuation marks are all retained, i.e. direct concatenation
        # of all the tokens in the sequence will be the original string.
        seq_tok = [tok for tok in re.split(r"([^a-zA-Z0-9])", s) if tok]
        return seq_tok

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx: int):
        ex = self.features[idx]

        return (
            np.array(ex.example_id_num),
            np.array(ex.example_id_num[-1]),  # service_id
            np.array(ex.utterance_ids),
            np.array(ex.utterance_segment),
            np.array(ex.utterance_mask, dtype=np.longlong),
            np.array(ex.intent_status, dtype=np.float32),
            np.array(ex.requested_slot_status, dtype=np.float32),
            np.array(ex.categorical_slot_status),
            np.array(ex.categorical_slot_value_status, dtype=np.float32),
            np.array(ex.noncategorical_slot_status),
            np.array(ex.noncategorical_slot_value_start),
            np.array(ex.noncategorical_slot_value_end),
            np.array(ex.start_char_idx),  # noncat_alignment_start
            np.array(ex.end_char_idx),  # noncat_alignment_end
            np.array(ex.task_mask),  # noncat_alignment_end
        )

    def bert_process_one_sample(self, idx):
        """
        Creates an example for each frame in the user turn.
        Args:
            turn_id: turn number
            system_utterance: last system utterance
            user_utterance: lst user utterance
            system_frames: all system utterances and slot - slot value pairs
            user_frames: all user utterances and slot - slot value pairs
            prev_states: slot - slot value pairs from the previous turns
            schemas: schema for all services of all datasets
            subsample: whether to balance postive and negative samples in the dataset
        Returns:
            examples: a list of `InputExample`s.
            prev_states: updated dialogue state e.g. {'Restaurants_1': {'city': ['San Jose'], 'cuisine': ['American']}}
        """

        ex = self.raw_features[idx].data
        example_id_num = ex["example_id_num"]
        example_id = ex["example_id"]
        user_utterance = ex["utterance"]
        system_utterance = ex["system_utterance"]
        service = ex["labels"]["service"]
        schemas = self.schemas
        state_update = ex["labels"]["slots"]
        system_slots = ex["system_slots"]

        user_tokens, user_alignments, user_inv_alignments = self._tokenize(user_utterance)
        system_tokens, system_alignments, system_inv_alignments = self._tokenize(system_utterance)
        system_user_utterance = system_utterance + ' ' + user_utterance
        system_user_tokens, system_user_alignments, system_user_inv_alignments = self._tokenize(system_user_utterance)
        examples = []

        base_example = SGDInputExample(schema_config=self.schema_config, tokenizer=self.tokenizer)
        base_example.service_schema = self.schemas.get_service_schema(service)
        base_example.service_id = example_id_num[-1]

        base_example.example_id = example_id
        base_example.example_id_num = example_id_num

        for model_task in range(self.schema_config["NUM_TASKS"]):
            if model_task == 0:
                for intent_id, intent in enumerate(schemas.get_service_schema(service).intents):
                    task_example = base_example.make_copy()
                    task_example.task_mask[model_task] = 1
                    task_example.intent_id = intent_id
                    task_example.example_id += f"-{model_task}-{intent_id}-0"
                    task_example.example_id_num.extend([model_task, intent_id, 0])
                    intent_description = (
                        intent + " " + self.schemas.get_service_schema(service).intent_descriptions[intent]
                    )
                    intent_tokens, intent_alignments, intent_inv_alignments = self._tokenize(intent_description)
                    task_example.add_utterance_features(
                        intent_tokens,
                        intent_inv_alignments,
                        system_user_tokens,
                        system_user_inv_alignments,
                        intent_description,
                        system_user_utterance,
                    )

                    task_example.add_intents(ex)
                    examples.append(task_example)

            if model_task == 1:
                for slot_id, slot in enumerate(schemas.get_service_schema(service).slots):
                    task_example = base_example.make_copy()
                    task_example.task_mask[model_task] = 1
                    task_example.requested_slot_id = slot_id
                    task_example.example_id += f"-{model_task}-{slot_id}-0"
                    task_example.example_id_num.extend([model_task, slot_id, 0])
                    slot_description = slot + " " + self.schemas.get_service_schema(service).slot_descriptions[slot]
                    slot_tokens, slot_alignments, slot_inv_alignments = self._tokenize(slot_description)
                    task_example.add_utterance_features(
                        slot_tokens,
                        slot_inv_alignments,
                        user_tokens,
                        user_inv_alignments,
                        slot_description,
                        user_utterance,
                    )

                    task_example.add_requested_slots(ex)
                    examples.append(task_example)

            if model_task == 2:
                off_slots = []
                on_slots = []
                for slot_id, slot in enumerate(schemas.get_service_schema(service).categorical_slots):
                    task_example = base_example.make_copy()
                    task_example.task_mask[model_task] = 1

                    # assert task_example.task_mask == [0, 0, 1, 0, 0, 0]
                    task_example.categorical_slot_id = slot_id
                    task_example.example_id += f"-{model_task}-{slot_id}-0"
                    task_example.example_id_num.extend([model_task, slot_id, 0])
                    slot_description = slot + " " + schemas.get_service_schema(service).slot_descriptions[slot]
                    slot_tokens, slot_alignments, slot_inv_alignments = self._tokenize(slot_description)
                    task_example.add_utterance_features(
                        slot_tokens,
                        slot_inv_alignments,
                        system_user_tokens,
                        system_user_inv_alignments,
                        slot_description,
                        system_user_utterance,
                    )
                    task_example.add_categorical_slots(state_update)

                    if task_example.categorical_slot_status == 0:
                        off_slots.append(task_example)
                    else:
                        on_slots.append(task_example)
                        examples.append(task_example)
                    old_example = task_example

                    for value_id, value in enumerate(
                        schemas.get_service_schema(service).get_categorical_slot_values(slot)
                    ):
                        if self.dataset_split != 'train' or task_example.categorical_slot_status == 1:
                            task_example = old_example.make_copy_of_categorical_features()
                            task_example.task_mask[3] = 1
                            # assert task_example.task_mask == [0, 0, 0, 1, 0, 0]
                            task_example.categorical_slot_id = slot_id
                            task_example.categorical_slot_value_id = value_id
                            task_example.example_id = base_example.example_id + f"-3-{slot_id}-{value_id}"
                            task_example.example_id_num = base_example.example_id_num + [3, slot_id, value_id]
                            slot_description = slot + " " + value  # add slot description
                            slot_tokens, slot_alignments, slot_inv_alignments = self._tokenize(slot_description)
                            task_example.add_utterance_features(
                                slot_tokens,
                                slot_inv_alignments,
                                system_user_tokens,
                                system_user_inv_alignments,
                                slot_description,
                                system_user_utterance,
                            )
                            task_example.add_categorical_slots(state_update)
                            assert task_example.categorical_slot_status == old_example.categorical_slot_status
                            examples.append(task_example)

                if self.dataset_split == 'train' and self.subsample:
                    num_on_slots = len(on_slots)
                    examples.extend(
                        np.random.choice(off_slots, replace=False, size=min(max(num_on_slots, 1), len(off_slots)))
                    )
                else:
                    examples.extend(off_slots)

            if model_task == 4:  # noncat slot status
                off_slots = []
                on_slots = []
                for slot_id, slot in enumerate(schemas.get_service_schema(service).non_categorical_slots):
                    task_example = base_example.make_copy()
                    task_example.task_mask[model_task] = 1
                    # assert task_example.task_mask == [0, 0, 0, 0, 1, 0]
                    task_example.noncategorical_slot_id = slot_id
                    task_example.example_id += f"-{model_task}-{slot_id}-0"
                    task_example.example_id_num.extend([model_task, slot_id, 0])
                    slot_description = slot + " " + schemas.get_service_schema(service).slot_descriptions[slot]
                    slot_tokens, slot_alignments, slot_inv_alignments = self._tokenize(slot_description)
                    task_example.add_utterance_features(
                        slot_tokens,
                        slot_inv_alignments,
                        system_user_tokens,
                        system_user_inv_alignments,
                        slot_description,
                        system_user_utterance,
                    )

                    user_span_boundaries = self._find_subword_indices(
                        state_update,
                        user_utterance,
                        ex["label_positions"]["slots"],
                        user_alignments,
                        user_tokens,
                        2 + len(slot_tokens) + len(system_tokens),
                    )

                    if system_slots is not None:
                        system_span_boundaries = self._find_subword_indices(
                            state_update,
                            system_utterance,
                            system_slots,
                            system_alignments,
                            system_tokens,
                            2 + len(slot_tokens),
                        )
                    else:
                        system_span_boundaries = {}

                    task_example.add_noncategorical_slots(state_update, user_span_boundaries, system_span_boundaries)
                    if task_example.noncategorical_slot_status == 0:
                        off_slots.append(task_example)
                    else:
                        on_slots.append(task_example)
                        examples.append(task_example)

                    if self.dataset_split != 'train' or task_example.noncategorical_slot_status == 1:
                        task_example = task_example.make_copy_of_non_categorical_features()
                        task_example.task_mask[5] = 1
                        # assert task_example.task_mask == [0, 0, 0, 0, 0, 1]
                        task_example.example_id = base_example.example_id + f"-5-{slot_id}-0"
                        task_example.example_id_num = base_example.example_id_num + [5, slot_id, 0]
                        examples.append(task_example)

                if self.dataset_split == 'train' and self.subsample:
                    num_on_slots = len(on_slots)
                    examples.extend(
                        np.random.choice(off_slots, replace=False, size=min(max(num_on_slots, 1), len(off_slots)))
                    )
                else:
                    examples.extend(off_slots)

        for example in examples:
            self.features.append(example)

    def _find_subword_indices(
        self,
        slot_values: dict,
        utterance: str,
        char_slot_spans: dict,
        alignments: List[int],
        subwords: List[str],
        bias: int,
    ) -> dict:
        """
        Find indices for subwords corresponding to slot values.
        Args:
            slot_values: slot - slot value pairs
            utterance: utterance
            char_slot_spans: char - slot spans
            alignments: alignments
            subwords: subtokens mapping
            bias: offset
        Returns:
            span_boundaries: span boundaries
        """
        span_boundaries = {}
        for slot, values in slot_values.items():
            # Get all values present in the utterance for the specified slot.
            value_char_spans = {}
            for key, slot_span in char_slot_spans.items():
                # print(key, slot, slot_span, char_slot_spans)
                if slot_span["slot"] == slot:
                    value = utterance[slot_span["start"] : slot_span["exclusive_end"]]
                    start_tok_idx = alignments[slot_span["start"]]
                    end_tok_idx = alignments[slot_span["exclusive_end"] - 1]
                    if 0 <= start_tok_idx < len(subwords):
                        end_tok_idx = min(end_tok_idx, len(subwords) - 1)
                        value_char_spans[value] = (start_tok_idx + bias, end_tok_idx + bias)
            for v in values:
                if v in value_char_spans:
                    span_boundaries[slot] = value_char_spans[v]
                    break
        return span_boundaries
