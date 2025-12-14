# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
from nemo.collections.common.tokenizers.tokenizer_utils import get_nmt_tokenizer
from nemo.collections.llm.gpt.data.pre_training import PreTrainingDataModule

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


def test_validate_dataset_asset_accessibility_file_does_not_exist(tokenizer, trainer):
    raised_exception = False
    try:
        data = PreTrainingDataModule(
            paths=["/this/path/should/not/exist/"],
            seq_length=512,
            micro_batch_size=2,
            global_batch_size=2,
            tokenizer=tokenizer,
        )
        data.trainer = trainer
    except FileNotFoundError:
        raised_exception = True

    assert raised_exception == True, "Expected to raise a FileNotFoundError"


def test_validate_dataset_asset_accessibility_file_is_none(tokenizer, trainer):
    raised_exception = False
    try:
        data = PreTrainingDataModule(
            paths=None,
            seq_length=512,
            micro_batch_size=2,
            global_batch_size=2,
            tokenizer=tokenizer,
        )
        data.trainer = trainer
    except ValueError:
        raised_exception = True

    assert raised_exception == True, "Expected to raise a ValueError"


def test_object_storage_cache_path(tokenizer):
    data = PreTrainingDataModule(
        paths=[f"msc://default{DATA_PATH}"],
        seq_length=512,
        micro_batch_size=2,
        global_batch_size=2,
        tokenizer=tokenizer,
        object_storage_cache_path="/tmp/object_storage_cache_path",
    )

    assert data.build_kwargs["object_storage_cache_path"] == "/tmp/object_storage_cache_path"
    assert data.build_kwargs["mmap_bin_files"] == False
