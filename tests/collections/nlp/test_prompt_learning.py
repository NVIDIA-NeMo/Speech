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

import pytest
import torch

from nemo.collections.nlp.data.language_modeling.megatron.gpt_prompt_learning_dataset import GPTPromptLearningDataset
from nemo.collections.nlp.models.language_modeling.megatron_gpt_prompt_learning_model import get_pseudo_tokens
from nemo.collections.nlp.modules.common import VirtualPromptSource
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.core import Dataset


def get_prompt_tuning_dataset(
    dataset_path, tokenizer, virtual_prompt_source, task_templates, pseudo_tokens,
):
    dataset = GPTPromptLearningDataset(
        data=[dataset_path],
        tokenizer=tokenizer,
        virtual_prompt_source=virtual_prompt_source,
        task_templates=task_templates,
        pseudo_tokens=pseudo_tokens,
        pad_token_id=tokenizer.unk_id,
        max_seq_length=512,
        min_seq_length=1,
    )

    return dataset


def create_temp_dataset():
    example_dataset_a = [
        {'taskname': 'task name A', 'text': 'Test sentence one, Answer: ', 'answer': 'test'} for i in range(24)
    ]
    example_dataset_b = [
        {'taskname': 'task name B', 'question': 'This is a question', 'answer': 'test'} for i in range(13)
    ]
    example_dataset = example_dataset_a + example_dataset_b
    temp_file_name = 'temp_dataset_file.jsonl'

    with open(temp_file_name, 'w') as temp:
        for example in example_dataset:
            temp.write(json.dumps(example) + '\n')

    return temp_file_name


def get_task_templates():
    task_templates = {}
    task_templates['task name A'] = {
        "prompt_template": "<|VIRTUAL_PROMPT_0|>{text}{answer}",
        "prompt_template_fields": ['text', 'answer'],
        "total_virtual_tokens": 5,
        "virtual_token_splits": [5],
        "truncate_field": None,
        "answer_only_loss": True,
        "answer_field": "answer",
        "task_id_num": 0,
    }
    task_templates['task name B'] = {
        "prompt_template": "<|VIRTUAL_PROMPT_0|>{question}<|VIRTUAL_PROMPT_1|>{answer}{extra}",
        "prompt_template_fields": ['question', 'answer', 'extra'],
        "total_virtual_tokens": 10,
        "virtual_token_splits": [7, 3],
        "truncate_field": None,
        "answer_only_loss": False,
        "answer_field": None,
        "task_id_num": 1,
    }
    return task_templates


class TestMegatronGPTPromptLearningDataset:
    @pytest.mark.run_only_on('GPU')
    @pytest.mark.unit
    def test_init_prompt_learning_dataset(self):
        tokenizer = get_nmt_tokenizer(library='megatron', model_name='GPT2BPETokenizer')
        task_templates = get_task_templates()
        dataset_path = create_temp_dataset()

        # Setup virtual token place holders
        total_virtual_tokens = 10
        pseudo_tokens = get_pseudo_tokens(total_virtual_tokens)
        tokenizer.add_special_tokens({'additional_special_tokens': pseudo_tokens})

        dataset = get_prompt_tuning_dataset(
            dataset_path, tokenizer, VirtualPromptSource.PROMPT_ENCODER, task_templates, pseudo_tokens,
        )

        print(type(dataset))

        assert isinstance(dataset, Dataset)

        os.remove(dataset_path)

    @pytest.mark.run_only_on('GPU')
    @pytest.mark.unit
    def test_prompt_learning_dataset_collate_fn_prompt_encoder(self):
        tokenizer = get_nmt_tokenizer(library='megatron', model_name='GPT2BPETokenizer')
        task_templates = get_task_templates()
        dataset_path = create_temp_dataset()

        # Setup virtual token place holders
        total_virtual_tokens = 10
        pseudo_tokens = get_pseudo_tokens(total_virtual_tokens)
        tokenizer.add_special_tokens({'additional_special_tokens': pseudo_tokens})

        dataset = get_prompt_tuning_dataset(
            dataset_path, tokenizer, VirtualPromptSource.PROMPT_ENCODER, task_templates, pseudo_tokens,
        )

        batch = [dataset[i] for i in range(8)]
        batch = dataset.collate_fn(batch)

        assert len(batch) == 6

        _, _, _, _, _, taskname_ids = batch

        assert list(taskname_ids[0].numpy()) == tokenizer.text_to_ids("task name A")

        os.remove(dataset_path)


if __name__ == "__main__":
    t = TestMegatronGPTPromptLearningDataset()
    t.test_init_prompt_learning_dataset()
    t.test_prompt_learning_dataset_collate_fn_prompt_encoder()
    print('-' * 50 + '\nALL PROMPT TUNING UNIT TESTS PASS!\n' + '-' * 50)
