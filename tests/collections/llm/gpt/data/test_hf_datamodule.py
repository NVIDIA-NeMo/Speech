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

from nemo.collections import llm

DATA_PATH = "/home/TestData/lite/hf_cache/squad/"


def test_load_single_split():
    ds = llm.HFDatasetDataModule(
        path=DATA_PATH,
        split='train',
        seq_length=512,
        micro_batch_size=2,
        global_batch_size=2,
    )
    from datasets.arrow_dataset import Dataset

    assert isinstance(ds.dataset_splits, dict)
    assert len(ds.dataset_splits) == 3
    assert 'train' in ds.dataset_splits
    assert ds.dataset_splits['train'] is not None
    assert ds.train is not None
    assert isinstance(ds.dataset_splits['train'], Dataset)
    assert 'val' in ds.dataset_splits
    assert ds.dataset_splits['val'] is None
    assert ds.val is None
    assert 'test' in ds.dataset_splits
    assert ds.dataset_splits['test'] is None
    assert ds.test is None


def test_load_nonexistent_split():
    exception_msg = ''
    expected_msg = '''Unknown split "this_split_name_should_not_exist". Should be one of ['train', 'validation'].'''
    try:
        llm.HFDatasetDataModule(
            path=DATA_PATH,
            split='this_split_name_should_not_exist',
            seq_length=512,
            micro_batch_size=2,
            global_batch_size=2,
        )
    except ValueError as e:
        exception_msg = str(e)
    assert exception_msg == expected_msg, exception_msg


def test_load_multiple_split():
    ds = llm.HFDatasetDataModule(
        path=DATA_PATH,
        split=['train', 'validation'],
        seq_length=512,
        micro_batch_size=2,
        global_batch_size=2,
    )
    from datasets.arrow_dataset import Dataset

    assert isinstance(ds.dataset_splits, dict)
    assert len(ds.dataset_splits) == 3
    assert 'train' in ds.dataset_splits
    assert ds.dataset_splits['train'] is not None
    assert ds.train is not None
    assert isinstance(ds.dataset_splits['train'], Dataset)
    assert isinstance(ds.train, Dataset)
    assert 'val' in ds.dataset_splits
    assert ds.dataset_splits['val'] is not None
    assert ds.val is not None
    assert isinstance(ds.dataset_splits['val'], Dataset)
    assert isinstance(ds.val, Dataset)
    assert 'test' in ds.dataset_splits
    assert ds.dataset_splits['test'] is None
    assert ds.test is None


def test_validate_dataset_asset_accessibility_file_does_not_exist():
    raised_exception = False
    try:
        llm.HFDatasetDataModule(
            path="/this/path/should/not/exist/",
            seq_length=512,
            micro_batch_size=2,
            global_batch_size=2,
        )
    except FileNotFoundError:
        raised_exception = True

    assert raised_exception == True, "Expected to raise a FileNotFoundError"


def test_validate_dataset_asset_accessibility_file_is_none():  # tokenizer, trainer):
    raised_exception = False
    try:
        llm.HFDatasetDataModule(
            path=None,
            seq_length=512,
            micro_batch_size=2,
            global_batch_size=2,
        )
    except TypeError:
        raised_exception = True

    assert raised_exception == True, "Expected to raise a ValueError"
