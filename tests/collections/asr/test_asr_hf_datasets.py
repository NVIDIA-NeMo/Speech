# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

import datasets as hf_datasets
import numpy as np
import pytest
import torch
from omegaconf import DictConfig

from nemo.collections.asr.data.huggingface.hf_audio_to_text import HFAudioToCharDataset, HFIterableAudioToCharDataset

LABELS = list(" abcdefghijklmnopqrstuvwxyz'")
SAMPLE_RATE = 16000


@pytest.fixture
def parquet_with_raw_samples(tmp_path):
    """A Parquet dataset that stores audio samples as plain float lists, so `datasets` does no audio decoding."""
    rng = np.random.default_rng(0)
    data = {
        "audio": [
            {"array": rng.uniform(-0.5, 0.5, n).astype(np.float32).tolist(), "sampling_rate": SAMPLE_RATE}
            for n in (8000, 4000)
        ],
        "text": ["hello world", "yes"],
    }
    path = tmp_path / "data.parquet"
    hf_datasets.Dataset.from_dict(data).to_parquet(str(path))
    return str(path)


def _dataset_kwargs(path):
    return dict(
        labels=LABELS,
        audio_key="audio.array",
        text_key="text",
        sample_rate_key="audio.sampling_rate",
        hf_data_cfg=DictConfig({"path": "parquet", "data_files": path, "split": "train"}),
        sample_rate=SAMPLE_RATE,
    )


@pytest.mark.unit
def test_map_style_dataset_accepts_audio_stored_as_lists(parquet_with_raw_samples):
    dataset = HFAudioToCharDataset(**_dataset_kwargs(parquet_with_raw_samples))

    audio, audio_len, tokens, tokens_len = dataset[0]

    assert audio.dtype == torch.float32
    assert audio_len.item() == audio.shape[0] == 8000
    assert tokens_len.item() == len("hello world")


@pytest.mark.unit
def test_iterable_dataset_accepts_audio_stored_as_lists(parquet_with_raw_samples):
    dataset = HFIterableAudioToCharDataset(**_dataset_kwargs(parquet_with_raw_samples), global_rank=0, world_size=1)

    lengths = [int(sample["a_sig_length"]) for sample in dataset]

    assert lengths == [8000, 4000]
