# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import pytest

import nemo.lightning as nl
from nemo.collections.llm.gpt.data.pre_training import PreTrainingDataModule
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer

DATA_PATH = "/home/TestData/nlp/megatron_gpt/data/gpt/simple_wiki_gpt_preproc_text_document"
VOCAB_PATH = "/home/TestData/nlp/megatron_gpt/data/gpt/vocab.json"
MERGES_PATH = "/home/TestData/nlp/megatron_gpt/data/gpt/merges.txt"


@pytest.fixture
def tokenizer():
    return get_nmt_tokenizer(
        "megatron",
        "GPT2BPETokenizer",
        vocab_file=VOCAB_PATH,
        merges_file=MERGES_PATH,
    )


@pytest.fixture
def trainer():
    return nl.Trainer(
        accelerator="cpu",
        max_steps=1,
    )


def test_single_data_distribution(tokenizer, trainer):

    data = PreTrainingDataModule(
        paths=[DATA_PATH],
        seq_length=512,
        micro_batch_size=2,
        global_batch_size=2,
        tokenizer=tokenizer,
    )
    data.trainer = trainer

    ## AssertioneError because we are trying to do eval on the whole
    ## dataset with just a single distribution
    with pytest.raises(AssertionError):
        data.setup(stage="dummy")

    trainer.limit_val_batches = 5
    ## this should succeed
    data.setup(stage="dummy")


def test_multiple_data_distributions(tokenizer, trainer):
    data = PreTrainingDataModule(
        paths={
            "train": ['1', DATA_PATH],
            "validation": [DATA_PATH, DATA_PATH],
            "test": ['1', DATA_PATH],
        },
        seq_length=512,
        micro_batch_size=2,
        global_batch_size=2,
        tokenizer=tokenizer,
    )
    data.trainer = trainer

    ## this should succeed
    data.setup(stage="dummy")
