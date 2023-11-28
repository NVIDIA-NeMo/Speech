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


import json
import os
import random
from functools import partial

import pytest

from nemo.collections.nlp.data.language_modeling.megatron.gpt_sft_chat_dataset import (
    GPTSFTChatDataset,
    get_prompt_template_example,
)
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer

TOKENIZER_FILE_43B = '/home/TestData/nlp/megatron_sft/tokenizer.model'
TOKENIZER_FILE_Llama2 = '/home/TestData/nlp/megatron_sft/llama2_tokenizer.model'
MERGE_FILE = '/home/TestData/nlp/megatron_sft/merges.txt'
VOCAB_FILE = '/home/TestData/nlp/megatron_sft/vocab.json'


def ids_to_text(tokenizer, ids):
    tokens = tokenizer.ids_to_tokens(ids)
    text = tokenizer.tokens_to_text(tokens)
    return text


def get_random_sentence():
    nouns = ("puppy", "car", "rabbit", "girl", "monkey")
    verbs = ("runs", "hits", "jumps", "drives", "barfs")
    adv = ("crazily.", "dutifully.", "foolishly.", "merrily.", "occasionally.")
    num1 = random.randrange(0, 5)
    num2 = random.randrange(0, 5)
    num3 = random.randrange(0, 5)
    return nouns[num1] + ' ' + verbs[num2] + ' ' + adv[num3]


def get_random_label():
    keys = ["quality", "toxicity", "humor", "creativity", "violence", "helpfulness", "not_appropriate"]
    values = [random.randrange(0, 5) for i in range(len(keys))]
    return ",".join([k + ":" + str(v) for k, v in zip(keys, values)])


def create_data_points(mask_user, turn_num, records, temp_file, t2v, label=True):
    data_points = []
    with open(temp_file, 'w', encoding='utf-8') as f:
        for r in range(records):
            record = {}
            record['system'] = 'a chat'
            record['type'] = 'TEXT_TO_VALUE' if t2v else 'VALUE_TO_TEXT'
            record['mask'] = 'User' if mask_user else 'Assistant'
            turns = []
            record['conversations'] = turns
            for i in range(turn_num):
                turn = {}
                turn['from'] = 'User' if i % 2 == 0 else 'Assistant'
                turn['value'] = get_random_sentence()
                if label:
                    turn['label'] = get_random_label()
                turns.append(turn)
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
            data_points.append(record)
    return data_points


@pytest.mark.skipif(not os.path.exists('/home/TestData'), reason='Not a Jenkins machine')
class TestGPTSFTChatDataset:
    @classmethod
    def setup_class(cls):
        cls.special_tokens = {
            "system_turn_start": "<extra_id_0>",
            "turn_start": "<extra_id_1>",
            "label_start": "<extra_id_2>",
            "end_of_turn": "\n",
            "end_of_name": "\n",
        }
        cls.suffix = cls.special_tokens['end_of_turn'] + cls.special_tokens['turn_start']
        cls.label_suffix = cls.special_tokens['end_of_name'] + cls.special_tokens['turn_start']

    def _mask_user_test(self, tokenizer, ids_to_text):
        random.seed(5)
        temp_file = '/tmp/test_file.jsonl'
        turn_num = 5
        records = 5
        try:
            data_points = create_data_points(True, turn_num, records, temp_file, t2v=False)
            d = GPTSFTChatDataset(
                temp_file,
                tokenizer,
                4096,
                1,
                index_mapping_dir='/tmp/',
                hf_dataset=True,
                special_tokens=self.special_tokens,
            )
            for i in range(len(d)):
                result = d[i]
                input_ids = result['input_ids']
                mask = result['mask']
                text = ids_to_text(input_ids[mask].tolist())
                expected_text = ''
                for j in range(1, turn_num, 2):
                    expected_text += data_points[i]['conversations'][j]['value'] + self.suffix
                assert text == expected_text
        finally:
            os.remove(temp_file)

    def _mask_assistant_test(self, tokenizer, ids_to_text):
        random.seed(3)
        temp_file = '/tmp/test_file.jsonl'
        turn_num = 5
        records = 5
        try:
            data_points = create_data_points(False, turn_num, records, temp_file, t2v=False)
            d = GPTSFTChatDataset(
                temp_file,
                tokenizer,
                4096,
                1,
                index_mapping_dir='/tmp/',
                hf_dataset=True,
                special_tokens=self.special_tokens,
            )
            for i in range(len(d)):
                result = d[i]
                input_ids = result['input_ids']
                mask = result['mask']
                text = ids_to_text(input_ids[mask].tolist())
                expected_text = ''
                for j in range(2, turn_num, 2):
                    expected_text += data_points[i]['conversations'][j]['value'] + self.suffix
                assert text == expected_text
        finally:
            os.remove(temp_file)

    def _mask_user_t2v_test(self, tokenizer, ids_to_text):
        random.seed(5)
        temp_file = '/tmp/test_file.jsonl'
        turn_num = 5
        records = 5
        try:
            data_points = create_data_points(True, turn_num, records, temp_file, t2v=True)
            d = GPTSFTChatDataset(
                temp_file,
                tokenizer,
                4096,
                1,
                index_mapping_dir='/tmp/',
                hf_dataset=True,
                special_tokens=self.special_tokens,
            )
            for i in range(len(d)):
                result = d[i]
                input_ids = result['input_ids']
                mask = result['mask']
                text = ids_to_text(input_ids[mask].tolist())
                expected_text = ''
                for j in range(1, turn_num, 2):
                    expected_text += data_points[i]['conversations'][j]['label'] + self.label_suffix
                assert text == expected_text
        finally:
            os.remove(temp_file)

    def _mask_assistant_t2v_test(self, tokenizer, ids_to_text):
        random.seed(5)
        temp_file = '/tmp/test_file.jsonl'
        turn_num = 5
        records = 5
        try:
            data_points = create_data_points(False, turn_num, records, temp_file, t2v=True)
            d = GPTSFTChatDataset(
                temp_file,
                tokenizer,
                4096,
                1,
                index_mapping_dir='/tmp/',
                hf_dataset=True,
                special_tokens=self.special_tokens,
            )
            for i in range(len(d)):
                result = d[i]
                input_ids = result['input_ids']
                mask = result['mask']
                text = ids_to_text(input_ids[mask].tolist())
                expected_text = ''
                for j in range(0, turn_num, 2):
                    expected_text += data_points[i]['conversations'][j]['label'] + self.label_suffix
                assert text == expected_text
        finally:
            os.remove(temp_file)

    def _mask_user_nolabel_test(self, tokenizer, ids_to_text):
        random.seed(5)
        temp_file = '/tmp/test_file.jsonl'
        turn_num = 5
        records = 5
        try:
            data_points = create_data_points(True, turn_num, records, temp_file, t2v=False, label=False)
            d = GPTSFTChatDataset(
                temp_file,
                tokenizer,
                4096,
                1,
                index_mapping_dir='/tmp/',
                hf_dataset=True,
                special_tokens=self.special_tokens,
            )
            for i in range(len(d)):
                result = d[i]
                input_ids = result['input_ids']
                mask = result['mask']
                text = ids_to_text(input_ids[mask].tolist())
                expected_text = ''
                for j in range(1, turn_num, 2):
                    expected_text += data_points[i]['conversations'][j]['value'] + self.suffix
                assert text == expected_text
        finally:
            os.remove(temp_file)

    def _mask_assistant_nolabel_test(self, tokenizer, ids_to_text):
        random.seed(3)
        temp_file = '/tmp/test_file.jsonl'
        turn_num = 5
        records = 5
        try:
            data_points = create_data_points(False, turn_num, records, temp_file, t2v=False, label=False)
            d = GPTSFTChatDataset(
                temp_file,
                tokenizer,
                4096,
                1,
                index_mapping_dir='/tmp/',
                hf_dataset=True,
                special_tokens=self.special_tokens,
            )
            for i in range(len(d)):
                result = d[i]
                input_ids = result['input_ids']
                mask = result['mask']
                text = ids_to_text(input_ids[mask].tolist())
                expected_text = ''
                for j in range(2, turn_num, 2):
                    expected_text += data_points[i]['conversations'][j]['value'] + self.suffix
                assert text == expected_text
        finally:
            os.remove(temp_file)

    def _test_example_prompt(self, tokenizer):
        random.seed(5)
        conv = get_prompt_template_example(self.special_tokens)
        expected = (
            self.special_tokens['system_turn_start']
            + 'System'
            + self.special_tokens['end_of_name']
            + '{system message}'
            + self.special_tokens['end_of_turn']
        )
        for turn in range(2):
            expected += (
                self.special_tokens['turn_start']
                + 'User'
                + self.special_tokens['end_of_name']
                + f'{{turn {turn + 1} user message}}'
                + self.special_tokens['end_of_turn']
            )
            expected += self.special_tokens['turn_start'] + 'Assistant' + self.special_tokens['end_of_name']
            expected += (
                self.special_tokens['label_start']
                + f'{{turn {turn + 1} assistant label}}'
                + self.special_tokens['end_of_name']
            )
            expected += f'{{turn {turn + 1} assistant message}}' + self.special_tokens['end_of_turn']
        expected += self.special_tokens['turn_start']
        assert conv == expected

    @pytest.mark.unit
    def test_43B_example_prompt(self):
        tokenizer = get_nmt_tokenizer(library='sentencepiece', tokenizer_model=TOKENIZER_FILE_43B)
        self._test_example_prompt(tokenizer)

    @pytest.mark.unit
    def test_43B_tokenizer_mask_user(self):
        tokenizer = get_nmt_tokenizer(library='sentencepiece', tokenizer_model=TOKENIZER_FILE_43B)
        self._mask_user_test(tokenizer, tokenizer.ids_to_text)

    @pytest.mark.unit
    def test_43B_tokenizer_mask_assistant(self):
        tokenizer = get_nmt_tokenizer(library='sentencepiece', tokenizer_model=TOKENIZER_FILE_43B)
        self._mask_assistant_test(tokenizer, tokenizer.ids_to_text)

    @pytest.mark.unit
    def test_43B_tokenizer_mask_user_t2v(self):
        tokenizer = get_nmt_tokenizer(library='sentencepiece', tokenizer_model=TOKENIZER_FILE_43B)
        self._mask_user_t2v_test(tokenizer, tokenizer.ids_to_text)

    @pytest.mark.unit
    def test_43B_tokenizer_mask_assistant_t2v(self):
        tokenizer = get_nmt_tokenizer(library='sentencepiece', tokenizer_model=TOKENIZER_FILE_43B)
        self._mask_assistant_t2v_test(tokenizer, tokenizer.ids_to_text)

    @pytest.mark.unit
    def test_43B_tokenizer_mask_user_nolabel(self):
        tokenizer = get_nmt_tokenizer(library='sentencepiece', tokenizer_model=TOKENIZER_FILE_43B)
        self._mask_user_nolabel_test(tokenizer, tokenizer.ids_to_text)

    @pytest.mark.unit
    def test_43B_tokenizer_mask_assistant_nolabel(self):
        tokenizer = get_nmt_tokenizer(library='sentencepiece', tokenizer_model=TOKENIZER_FILE_43B)
        self._mask_assistant_nolabel_test(tokenizer, tokenizer.ids_to_text)

    @pytest.mark.unit
    def test_mpt_tokenizer_mask_user(self):
        tokenizer = get_nmt_tokenizer(
            library='huggingface', model_name='gpt2', merges_file=MERGE_FILE, vocab_file=VOCAB_FILE, use_fast=True
        )
        tokenizer.add_special_tokens({'additional_special_tokens': ['<extra_id_0>', '<extra_id_1>', '<extra_id_2>']})
        self._mask_user_test(tokenizer, partial(ids_to_text, tokenizer))

    @pytest.mark.unit
    def test_mpt_tokenizer_mask_assistant(self):
        tokenizer = get_nmt_tokenizer(
            library='huggingface', model_name='gpt2', merges_file=MERGE_FILE, vocab_file=VOCAB_FILE, use_fast=True
        )
        tokenizer.add_special_tokens({'additional_special_tokens': ['<extra_id_0>', '<extra_id_1>', '<extra_id_2>']})
        self._mask_assistant_test(tokenizer, partial(ids_to_text, tokenizer))

    @pytest.mark.unit
    def test_mpt_tokenizer_mask_user_t2v(self):
        tokenizer = get_nmt_tokenizer(
            library='huggingface', model_name='gpt2', merges_file=MERGE_FILE, vocab_file=VOCAB_FILE, use_fast=True
        )
        tokenizer.add_special_tokens({'additional_special_tokens': ['<extra_id_0>', '<extra_id_1>', '<extra_id_2>']})
        self._mask_user_t2v_test(tokenizer, partial(ids_to_text, tokenizer))

    @pytest.mark.unit
    def test_mpt_tokenizer_mask_assistant_t2v(self):
        tokenizer = get_nmt_tokenizer(
            library='huggingface', model_name='gpt2', merges_file=MERGE_FILE, vocab_file=VOCAB_FILE, use_fast=True
        )
        tokenizer.add_special_tokens({'additional_special_tokens': ['<extra_id_0>', '<extra_id_1>', '<extra_id_2>']})
        self._mask_assistant_t2v_test(tokenizer, partial(ids_to_text, tokenizer))

    @pytest.mark.unit
    def test_mpt_tokenizer_mask_user_nolabel(self):
        tokenizer = get_nmt_tokenizer(
            library='huggingface', model_name='gpt2', merges_file=MERGE_FILE, vocab_file=VOCAB_FILE, use_fast=True
        )
        tokenizer.add_special_tokens({'additional_special_tokens': ['<extra_id_0>', '<extra_id_1>', '<extra_id_2>']})
        self._mask_user_nolabel_test(tokenizer, partial(ids_to_text, tokenizer))

    @pytest.mark.unit
    def test_mpt_tokenizer_mask_assistant_nolabel(self):
        tokenizer = get_nmt_tokenizer(
            library='huggingface', model_name='gpt2', merges_file=MERGE_FILE, vocab_file=VOCAB_FILE, use_fast=True
        )
        tokenizer.add_special_tokens({'additional_special_tokens': ['<extra_id_0>', '<extra_id_1>', '<extra_id_2>']})
        self._mask_assistant_nolabel_test(tokenizer, partial(ids_to_text, tokenizer))

    @pytest.mark.unit
    def test_llama2_tokenizer_mask_user(self):
        tokenizer = get_nmt_tokenizer(library='sentencepiece', tokenizer_model=TOKENIZER_FILE_Llama2)
        self._mask_user_test(tokenizer, tokenizer.ids_to_text)

    @pytest.mark.unit
    def test_llama2_tokenizer_mask_assistant(self):
        tokenizer = get_nmt_tokenizer(library='sentencepiece', tokenizer_model=TOKENIZER_FILE_Llama2)
        self._mask_assistant_test(tokenizer, tokenizer.ids_to_text)

    @pytest.mark.unit
    def test_llama2_tokenizer_mask_user_t2v(self):
        tokenizer = get_nmt_tokenizer(library='sentencepiece', tokenizer_model=TOKENIZER_FILE_Llama2)
        self._mask_user_t2v_test(tokenizer, tokenizer.ids_to_text)

    @pytest.mark.unit
    def test_llama2_tokenizer_mask_assistant_t2v(self):
        tokenizer = get_nmt_tokenizer(library='sentencepiece', tokenizer_model=TOKENIZER_FILE_Llama2)
        self._mask_assistant_t2v_test(tokenizer, tokenizer.ids_to_text)

    @pytest.mark.unit
    def test_llama2_tokenizer_mask_user_nolabel(self):
        tokenizer = get_nmt_tokenizer(library='sentencepiece', tokenizer_model=TOKENIZER_FILE_Llama2)
        self._mask_user_nolabel_test(tokenizer, tokenizer.ids_to_text)

    @pytest.mark.unit
    def test_llama2_tokenizer_mask_assistant_nolabel(self):
        tokenizer = get_nmt_tokenizer(library='sentencepiece', tokenizer_model=TOKENIZER_FILE_Llama2)
        self._mask_assistant_nolabel_test(tokenizer, tokenizer.ids_to_text)

    @pytest.mark.unit
    def test_normal_mpt_tokenizer_mask_user(self):
        tokenizer = get_nmt_tokenizer(
            library='huggingface', model_name='gpt2', merges_file=MERGE_FILE, vocab_file=VOCAB_FILE, use_fast=True
        )
        self._mask_user_test(tokenizer, tokenizer.ids_to_text)

    @pytest.mark.unit
    def test_normal_mpt_tokenizer_mask_assistant(self):
        tokenizer = get_nmt_tokenizer(
            library='huggingface', model_name='gpt2', merges_file=MERGE_FILE, vocab_file=VOCAB_FILE, use_fast=True
        )
        self._mask_assistant_test(tokenizer, tokenizer.ids_to_text)

    @pytest.mark.unit
    def test_normal_mpt_tokenizer_mask_user_t2v(self):
        tokenizer = get_nmt_tokenizer(
            library='huggingface', model_name='gpt2', merges_file=MERGE_FILE, vocab_file=VOCAB_FILE, use_fast=True
        )
        self._mask_user_t2v_test(tokenizer, tokenizer.ids_to_text)

    @pytest.mark.unit
    def test_normal_mpt_tokenizer_mask_assistant_t2v(self):
        tokenizer = get_nmt_tokenizer(
            library='huggingface', model_name='gpt2', merges_file=MERGE_FILE, vocab_file=VOCAB_FILE, use_fast=True
        )
        self._mask_assistant_t2v_test(tokenizer, tokenizer.ids_to_text)

    @pytest.mark.unit
    def test_normal_mpt_tokenizer_mask_user_nolabel(self):
        tokenizer = get_nmt_tokenizer(
            library='huggingface', model_name='gpt2', merges_file=MERGE_FILE, vocab_file=VOCAB_FILE, use_fast=True
        )
        self._mask_user_nolabel_test(tokenizer, tokenizer.ids_to_text)

    @pytest.mark.unit
    def test_normal_mpt_tokenizer_mask_assistant_nolabel(self):
        tokenizer = get_nmt_tokenizer(
            library='huggingface', model_name='gpt2', merges_file=MERGE_FILE, vocab_file=VOCAB_FILE, use_fast=True
        )
        self._mask_assistant_nolabel_test(tokenizer, tokenizer.ids_to_text)


class TestDifferentGPTSFTChatDataset(TestGPTSFTChatDataset):
    @classmethod
    def setup_class(cls):
        cls.special_tokens = {
            "system_turn_start": "<|im_start|>",
            "turn_start": "<|im_start|>",
            "label_start": "<|label|>",
            "end_of_turn": "<|im_end|>\n",
            "end_of_name": "\n",
        }
        cls.suffix = cls.special_tokens['end_of_turn'] + cls.special_tokens['turn_start']
        cls.label_suffix = cls.special_tokens['end_of_name'] + cls.special_tokens['turn_start']
