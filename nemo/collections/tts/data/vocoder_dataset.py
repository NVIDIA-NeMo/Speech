# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch.utils.data

from nemo.collections.asr.parts.utils.manifest_utils import read_manifest
from nemo.collections.tts.parts.preprocessing.feature_processors import FeatureProcessor
from nemo.collections.tts.parts.utils.tts_dataset_utils import (
    filter_dataset_by_duration,
    get_weighted_sampler,
    load_audio,
    sample_audio,
    stack_tensors,
)
from nemo.core.classes import Dataset
from nemo.utils import logging
from nemo.utils.decorators import experimental


@dataclass
class DatasetMeta:
    manifest_path: Path
    audio_dir: Path
    sample_weight: float = 1.0


@dataclass
class DatasetSample:
    dataset_name: str
    manifest_entry: dict
    audio_dir: Path


@experimental
class VocoderDataset(Dataset):
    """
    Class for processing and loading Vocoder training examples.

    Args:
        dataset_meta: Dict of dataset names (string) to dataset metadata.
        sample_rate: Sample rate to load audio as. If the audio is stored at a different sample rate, then it will
            be resampled.
        n_samples: Optional int, if provided then n_samples samples will be randomly sampled from the full
            audio file.
        weighted_sampling_steps_per_epoch: Optional int, If provided, then data will be sampled (with replacement) based on
            the sample weights provided in the dataset metadata. If None, then sample weights will be ignored.
        feature_processors: Optional, list of feature processors to run on training examples.
        min_duration: Optional float, if provided audio files in the training manifest shorter than 'min_duration'
            will be ignored.
        max_duration: Optional float, if provided audio files in the training manifest longer than 'max_duration'
            will be ignored.
        trunc_duration: Optional int, if provided audio will be truncated to at most 'trunc_duration' seconds.
        volume_norm: Whether to apply volume normalization to loaded audio.
    """

    def __init__(
        self,
        dataset_meta: Dict,
        sample_rate: int,
        n_samples: Optional[int] = None,
        weighted_sampling_steps_per_epoch: Optional[int] = None,
        feature_processors: Optional[Dict[str, FeatureProcessor]] = None,
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
        trunc_duration: Optional[float] = None,
        volume_norm: bool = False,
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.n_samples = n_samples
        self.trunc_duration = trunc_duration
        self.volume_norm = volume_norm
        self.weighted_sampling_steps_per_epoch = weighted_sampling_steps_per_epoch
        self.load_precomputed_mel = False

        if feature_processors:
            logging.info(f"Found feature processors {feature_processors.keys()}")
            self.feature_processors = list(feature_processors.values())
        else:
            self.feature_processors = []

        self.data_samples = []
        self.sample_weights = []
        for dataset_name, dataset_info in dataset_meta.items():
            dataset = DatasetMeta(**dataset_info)
            samples, weights = self._preprocess_manifest(
                dataset_name=dataset_name, dataset=dataset, min_duration=min_duration, max_duration=max_duration,
            )
            self.data_samples += samples
            self.sample_weights += weights

    def get_sampler(self, batch_size: int) -> Optional[torch.utils.data.Sampler]:
        if not self.weighted_sampling_steps_per_epoch:
            return None

        sampler = get_weighted_sampler(
            sample_weights=self.sample_weights, batch_size=batch_size, num_steps=self.weighted_sampling_steps_per_epoch
        )
        return sampler

    @staticmethod
    def _preprocess_manifest(
        dataset_name: str, dataset: DatasetMeta, min_duration: float, max_duration: float,
    ):
        entries = read_manifest(dataset.manifest_path)
        filtered_entries, total_hours, filtered_hours = filter_dataset_by_duration(
            entries=entries, min_duration=min_duration, max_duration=max_duration
        )

        logging.info(dataset_name)
        logging.info(f"Original # of files: {len(entries)}")
        logging.info(f"Filtered # of files: {len(filtered_entries)}")
        logging.info(f"Original duration: {total_hours:.2f} hours")
        logging.info(f"Filtered duration: {filtered_hours:.2f} hours")

        samples = []
        sample_weights = []
        for entry in filtered_entries:
            sample = DatasetSample(dataset_name=dataset_name, manifest_entry=entry, audio_dir=Path(dataset.audio_dir))
            samples.append(sample)
            sample_weights.append(dataset.sample_weight)

        return samples, sample_weights

    def __len__(self):
        return len(self.data_samples)

    def __getitem__(self, index):
        data = self.data_samples[index]

        if self.n_samples:
            audio_array, _, audio_filepath_rel = sample_audio(
                manifest_entry=data.manifest_entry,
                audio_dir=data.audio_dir,
                sample_rate=self.sample_rate,
                n_samples=self.n_samples,
                volume_norm=self.volume_norm,
            )
        else:
            audio_array, _, audio_filepath_rel = load_audio(
                manifest_entry=data.manifest_entry,
                audio_dir=data.audio_dir,
                sample_rate=self.sample_rate,
                max_duration=self.trunc_duration,
                volume_norm=self.volume_norm,
            )
        audio = torch.tensor(audio_array, dtype=torch.float32)
        audio_len = audio.shape[0]

        example = {
            "dataset_name": data.dataset_name,
            "audio_filepath": audio_filepath_rel,
            "audio": audio,
            "audio_len": audio_len,
        }

        for processor in self.feature_processors:
            processor.process(example)

        return example

    def collate_fn(self, batch: List[dict]):
        dataset_name_list = []
        audio_filepath_list = []
        audio_list = []
        audio_len_list = []

        for example in batch:
            dataset_name_list.append(example["dataset_name"])
            audio_filepath_list.append(example["audio_filepath"])
            audio_list.append(example["audio"])
            audio_len_list.append(example["audio_len"])

        batch_audio_len = torch.IntTensor(audio_len_list)
        audio_max_len = int(batch_audio_len.max().item())

        batch_audio = stack_tensors(audio_list, max_lens=[audio_max_len])

        batch_dict = {
            "dataset_names": dataset_name_list,
            "audio_filepaths": audio_filepath_list,
            "audio": batch_audio,
            "audio_lens": batch_audio_len,
        }

        return batch_dict
