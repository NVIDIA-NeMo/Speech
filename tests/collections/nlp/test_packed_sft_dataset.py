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


import os

import numpy as np
import pytest

from nemo.collections.nlp.data.language_modeling.megatron.gpt_sft_dataset import GPTSFTPackedDataset
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer

TOKENIZER_FILE_Llama2 = '/home/TestData/nlp/megatron_sft/llama2_tokenizer.model'
TEMP_FILE = '/tmp/test_file.npy'


def create_data_points(temp_file, answer_only_loss=True):
    '''
    Saves mock data in the temp_file, and return the expected output of the collate function
    '''
    # explanation of data:
    # input_ids: concatenated (packed) sequences that each end in EOS (index = 2)
    # loss_mask: 0 denotes context, 1 denotes answer
    # seq_start_id: start index of each sequence in the pack
    output_data = [
        {
            'input_ids': [10, 11, 12, 2, 20, 21, 22, 23, 2, 30, 31, 32, 2],
            'loss_mask': [0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1],
            'seq_start_id': [0, 4, 9],
        },
        {
            'input_ids': [10, 11, 12, 2, 20, 21, 22, 23, 2, 30, 31, 32, 33, 34, 35, 2],
            'loss_mask': [0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1],
            'seq_start_id': [0, 4, 9],
        },
    ]

    expected_output = [
        {
            "tokens": [10, 11, 12, 20, 21, 22, 23, 30, 31, 32],
            "labels": [11, 12, 2, 21, 22, 23, 2, 31, 32, 2],
            "loss_mask": [0, 1, 1, 0, 0, 1, 1, 1, 1, 1] if answer_only_loss else [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "position_ids": [0, 1, 2, 0, 1, 2, 3, 0, 1, 2],
            "cu_seqlens": [0, 3, 7, 10],
        },
        {
            "tokens": [10, 11, 12, 20, 21, 22, 23, 30, 31, 32, 33, 34, 35],
            "labels": [11, 12, 2, 21, 22, 23, 2, 31, 32, 33, 34, 35, 2],
            "loss_mask": [0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
            if answer_only_loss
            else [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "position_ids": [0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 4, 5],
            "cu_seqlens": [0, 3, 7, 13],
        },
    ]

    np.save(temp_file, output_data)
    return expected_output


@pytest.mark.skipif(not os.path.exists('/home/TestData'), reason='Not a Jenkins machine')
class TestGPTSFTPackedDataset:
    def _packed_seq_test(self, tokenizer, answer_only_loss=True):
        try:
            expected_output = create_data_points(TEMP_FILE, answer_only_loss)
            test_dataset = GPTSFTPackedDataset(
                TEMP_FILE,
                tokenizer,
                label_key="output",
                truncation_field="input",
                prompt_template="{input} {output}",
                answer_only_loss=answer_only_loss,
            )
            batch = [data for data in test_dataset]
            model_input = test_dataset.collate_fn(batch)

            for i in range(len(batch)):
                for output_key in ('tokens', 'labels', 'loss_mask', 'position_ids', 'cu_seqlens'):
                    expected = expected_output[i][output_key]
                    # remove padding in the actual results
                    actual = model_input[output_key][i][: len(expected)].tolist()
                    print('expected', expected, 'actual', actual)
                    if output_key == 'cu_seqlens':
                        # skip the last value of cu_seqlens since it was changed to the padded length for FP8 training
                        actual[-1] = expected[-1]
                    assert expected == actual, f"{output_key} does not match"

        finally:
            os.remove(TEMP_FILE)

    @pytest.mark.unit
    def test_full_loss(self):
        tokenizer = get_nmt_tokenizer(library='sentencepiece', tokenizer_model=TOKENIZER_FILE_Llama2)
        self._packed_seq_test(tokenizer, answer_only_loss=False)

    @pytest.mark.unit
    def test_answer_only_loss(self):
        tokenizer = get_nmt_tokenizer(library='sentencepiece', tokenizer_model=TOKENIZER_FILE_Llama2)
        self._packed_seq_test(tokenizer, answer_only_loss=True)
