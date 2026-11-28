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

"""Batch-size validation of `AudioCodecModel._get_lhotse_dataloader`."""

import lhotse
import numpy as np
import pytest
import soundfile as sf
from lhotse import CutSet
from omegaconf import DictConfig

from nemo.collections.tts.models import AudioCodecModel
from tests.collections.tts.models.test_audio_codec import create_codec_config

SEGMENT_DURATION = 0.5


@pytest.fixture(scope="module")
def codec_model():
    return AudioCodecModel(cfg=create_codec_config())


@pytest.fixture()
def cuts_path(tmp_path):
    """Six cuts of varying duration (all longer than SEGMENT_DURATION) with `target_audio` wavs on disk."""
    sample_rate = 24000
    cuts = []
    for i, duration in enumerate([1.0, 1.0, 1.5, 1.5, 2.0, 2.0]):
        path = tmp_path / f"cut_{i}.wav"
        samples = np.random.default_rng(i).uniform(-0.5, 0.5, int(duration * sample_rate)).astype(np.float32)
        sf.write(str(path), samples, sample_rate, format="WAV")
        recording = lhotse.Recording.from_file(path, recording_id=f"rec_{i}")
        cut = recording.to_cut()
        cut.id = f"cut_{i}"
        cut.custom = {"target_audio": recording}
        cuts.append(cut)
    path = tmp_path / "cuts.jsonl.gz"
    CutSet.from_cuts(cuts).to_file(path)
    return str(path)


def _train_ds_cfg(cuts_path, **dataloader_params):
    return DictConfig(
        {
            "dataloader_params": {
                "use_lhotse": True,
                "cuts_path": cuts_path,
                "shuffle": True,
                "seed": 0,
                "num_workers": 0,
                **dataloader_params,
            },
            "dataset_args": {"segment_duration": SEGMENT_DURATION},
        }
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "dataloader_params",
    [
        pytest.param({"batch_size": 2}, id="batch_size"),
        pytest.param(
            {
                "batch_size": 2,
                "use_bucketing": True,
                "num_buckets": 2,
                "bucket_duration_bins": [1.25],
                "bucket_buffer_size": 6,
                "concurrent_bucketing": False,
            },
            id="batch_size_bucketing",
        ),
    ],
)
def test_lhotse_dataloader_accepts_batch_size(codec_model, cuts_path, dataloader_params):
    dl = codec_model._get_lhotse_dataloader(_train_ds_cfg(cuts_path, **dataloader_params))
    batches = list(dl)
    assert sum(b["audio"].shape[0] for b in batches) == 6
    for batch in batches:
        assert batch["audio"].shape == (2, int(SEGMENT_DURATION * codec_model.output_sample_rate))


@pytest.mark.unit
@pytest.mark.parametrize(
    "dataloader_params",
    [
        pytest.param({"batch_duration": 3.0}, id="batch_duration"),
        pytest.param({"batch_duration": 3.0, "batch_size": 2}, id="batch_duration_and_batch_size"),
        pytest.param(
            {
                "batch_duration": 3.0,
                "use_bucketing": True,
                "num_buckets": 2,
                "bucket_duration_bins": [1.25],
                "concurrent_bucketing": False,
            },
            id="batch_duration_bucketing",
        ),
    ],
)
def test_lhotse_dataloader_rejects_batch_duration(codec_model, cuts_path, dataloader_params):
    with pytest.raises(ValueError, match="`batch_duration` is not supported.*set `batch_size` instead"):
        codec_model._get_lhotse_dataloader(_train_ds_cfg(cuts_path, **dataloader_params))


@pytest.mark.unit
def test_lhotse_dataloader_requires_batch_size(codec_model, cuts_path):
    with pytest.raises(ValueError, match="`batch_size` must be set.*fixed-length crops"):
        codec_model._get_lhotse_dataloader(_train_ds_cfg(cuts_path))
