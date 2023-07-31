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

import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import torch.utils.data

from nemo.collections.asr.parts.preprocessing.segment import AudioSegment
from nemo.collections.asr.parts.utils.manifest_utils import read_manifest
from nemo.collections.tts.parts.preprocessing.feature_processors import FeatureProcessor
from nemo.collections.tts.parts.utils.tts_dataset_utils import (
    filter_dataset_by_duration,
    get_abs_rel_paths,
    get_weighted_sampler,
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
        num_audio_retries: Number of read attempts to make when sampling audio file, to avoid training failing
            from sporadic IO errors.
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
        num_audio_retries: int = 5,
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.n_samples = n_samples
        self.weighted_sampling_steps_per_epoch = weighted_sampling_steps_per_epoch
        self.num_audio_retries = num_audio_retries
        self.load_precomputed_mel = False

        if trunc_duration:
            self.trunc_samples = int(trunc_duration * self.sample_rate)
        else:
            self.trunc_samples = None

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

    def _segment_audio(self, audio_filepath: Path) -> AudioSegment:
        # Retry file read multiple times as file seeking can produce random IO errors.
        for _ in range(self.num_audio_retries):
            try:
                audio_segment = AudioSegment.segment_from_file(
                    audio_filepath, target_sr=self.sample_rate, n_segments=self.n_samples,
                )
                return audio_segment
            except Exception:
                traceback.print_exc()

        raise ValueError(f"Failed to read audio {audio_filepath}")

    def _sample_audio(self, audio_filepath: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.n_samples:
            audio_array, _ = librosa.load(audio_filepath, sr=self.sample_rate)
        else:
            audio_segment = self._segment_audio(audio_filepath)
            audio_array = audio_segment.samples

        if self.trunc_samples:
            audio_array = audio_array[: self.trunc_samples]

        audio = torch.tensor(audio_array)
        audio_len = torch.tensor(audio.shape[0])
        return audio, audio_len

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
            sample = DatasetSample(manifest_entry=entry, audio_dir=Path(dataset.audio_dir),)
            samples.append(sample)
            sample_weights.append(dataset.sample_weight)

        return samples, sample_weights

    def __len__(self):
        return len(self.data_samples)

    def __getitem__(self, index):
        data = self.data_samples[index]

        audio_filepath = Path(data.manifest_entry["audio_filepath"])
        audio_filepath_abs, audio_filepath_rel = get_abs_rel_paths(input_path=audio_filepath, base_path=data.audio_dir)

        audio, audio_len = self._sample_audio(audio_filepath_abs)

        example = {"audio_filepath": audio_filepath_rel, "audio": audio, "audio_len": audio_len}

        for processor in self.feature_processors:
            processor.process(example)

        return example

    def collate_fn(self, batch: List[dict]):
        audio_filepath_list = []
        audio_list = []
        audio_len_list = []

        for example in batch:
            audio_filepath_list.append(example["audio_filepath"])
            audio_list.append(example["audio"])
            audio_len_list.append(example["audio_len"])

        batch_audio_len = torch.IntTensor(audio_len_list)
        audio_max_len = int(batch_audio_len.max().item())

        batch_audio = stack_tensors(audio_list, max_lens=[audio_max_len])

        batch_dict = {
            "audio_filepaths": audio_filepath_list,
            "audio": batch_audio,
            "audio_lens": batch_audio_len,
        }

        return batch_dict
