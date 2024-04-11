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
import os
import tempfile

import numpy as np
import pytest
import soundfile as sf
import torch

from nemo.collections.asr.data import audio_to_text
from nemo.collections.asr.parts.utils.asr_batching import SemiSortBatchSampler
from nemo.collections.asr.parts.utils.manifest_utils import write_manifest


class TestASRSamplers:
    labels = [
        " ",
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "j",
        "k",
        "l",
        "m",
        "n",
        "o",
        "p",
        "q",
        "r",
        "s",
        "t",
        "u",
        "v",
        "w",
        "x",
        "y",
        "z",
        "'",
    ]

    @pytest.mark.unit
    def test_ssb_sampler(self):
        # Generate random signals
        data_min_duration = 0.1
        data_max_duration = 16.7

        random_seed = 42
        sample_rate = 16000

        _rng = np.random.default_rng(seed=random_seed)

        def generate_samples(num_examples: int) -> list:
            data_duration = np.round(_rng.uniform(low=data_min_duration, high=data_max_duration, size=num_examples), 3)
            data_duration_samples = np.floor(data_duration * sample_rate).astype(int)
            samples = []
            for data_duration_sample in data_duration_samples:
                samples.append(_rng.uniform(low=-0.5, high=0.5, size=(data_duration_sample)))
            return samples

        with tempfile.TemporaryDirectory() as test_dir:
            # Build metadata for manifest
            metadata = []

            # Test size of dataloader with and without ssb
            for num_samples in np.concatenate([np.array([1, 2]), _rng.integers(3, 10, 2), _rng.integers(10, 1000, 2)]):
                samples = generate_samples(num_samples)

                for n, sample in enumerate(samples):
                    meta = dict()
                    signal_filename = f'{n:04d}.wav'
                    # write audio files
                    sf.write(os.path.join(test_dir, signal_filename), sample, sample_rate)
                    # update metadata
                    meta['audio_filepath'] = os.path.join(test_dir, signal_filename)
                    meta['duration'] = len(sample) / sample_rate
                    meta['text'] = 'non empty'
                    metadata.append(meta)

                # Save manifest
                manifest_filepath = os.path.join(test_dir, 'manifest.json')
                write_manifest(manifest_filepath, metadata)

                # Make dataset
                dataset = audio_to_text.AudioToCharDataset(
                    manifest_filepath=manifest_filepath,
                    labels=self.labels,
                    sample_rate=sample_rate,
                    max_duration=data_max_duration,
                    min_duration=data_min_duration,
                )
                durations = [sample.duration for sample in dataset.manifest_processor.collection.data]

                # Compare two dataloader
                for batch_size in _rng.integers(1, n + 20, 5):
                    batch_size = int(batch_size)
                    drop_last = True if _rng.integers(0, 2) else False
                    sampler = SemiSortBatchSampler(
                        global_rank=0,
                        world_size=1,
                        durations=durations,
                        batch_size=batch_size,
                        batch_shuffle=True,
                        drop_last=drop_last,
                        randomization_factor=0.1,
                        seed=random_seed,
                    )
                    dataloader_with_ssb = torch.utils.data.DataLoader(
                        dataset=dataset,
                        batch_size=None,
                        sampler=sampler,
                        batch_sampler=None,
                        collate_fn=lambda x: audio_to_text._speech_collate_fn(x, pad_id=0),
                    )
                    dataloader = torch.utils.data.DataLoader(
                        dataset=dataset,
                        batch_size=batch_size,
                        collate_fn=lambda x: audio_to_text._speech_collate_fn(x, pad_id=0),
                        drop_last=drop_last,
                        shuffle=True,
                    )

                    assert abs(len(dataloader) - len(dataloader_with_ssb)) == 0, (
                        "Different num of batches with batch! Num of batches with ssb is "
                        f"{len(dataloader_with_ssb)} and without ssb is {len(dataloader)}!"
                    )

                    dataloader_with_ssb_exception, dataloader_exception = False, False

                    try:
                        list(dataloader_with_ssb)
                    except:
                        dataloader_with_ssb_exception = True

                    try:
                        list(dataloader)
                    except:
                        dataloader_exception = True

                    assert dataloader_with_ssb_exception == dataloader_exception
