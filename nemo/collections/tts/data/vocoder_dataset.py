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

import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import librosa
import soundfile as sf
import torch.utils.data
import webdataset as wd

from nemo.collections.asr.data.audio_to_text import expand_sharded_filepaths
from nemo.collections.asr.parts.preprocessing.segment import available_formats as valid_sf_formats
from nemo.collections.asr.parts.utils.manifest_utils import read_manifest
from nemo.collections.tts.parts.preprocessing.feature_processors import FeatureProcessor
from nemo.collections.tts.parts.utils.tts_dataset_utils import (
    filter_dataset_by_duration,
    get_weighted_sampler,
    load_audio,
    sample_audio,
    stack_tensors,
)
from nemo.core.classes import Dataset, IterableDataset
from nemo.utils import logging
from nemo.utils.decorators import experimental

VALID_FILE_FORMATS = ';'.join(['wav', 'mp3', 'flac'] + [fmt.lower() for fmt in valid_sf_formats.keys()])


@dataclass
class DatasetMeta:
    manifest_path: Path
    audio_dir: Path
    sample_weight: float = 1.0
    audio_tar_filepaths: Optional[List[str]] = None


@dataclass
class DatasetSample:
    dataset_name: str
    manifest_entry: dict
    audio_dir: Path


def audio_collate_fn(batch: List[dict]):
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


def preprocess_manifest(
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
            samples, weights = preprocess_manifest(
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

    def collate_fn(self, batch):
        return audio_collate_fn(batch)


class TarredVocoderDataset(IterableDataset):
    """
    A similar Dataset to the VocoderDataset, but loads tarred audio files.

    Accepts a single comma-separated JSON manifest file (in the same style as for the VocoderDataset),
    as well as the path(s) to the tarball(s) containing the wav files. Each line of the manifest should
    contain the information for one audio file, and duration of audio.

    Valid formats for the audio_tar_filepaths argument include:
    (1) a single string that can be brace-expanded, e.g. 'path/to/audio.tar' or 'path/to/audio_{1..100}.tar.gz', or
    (2) a list of file paths that will not be brace-expanded, e.g. ['audio_1.tar', 'audio_2.tar', ...].

    See the WebDataset documentation for more information about accepted data and input formats.

    If using multiple processes the number of shards should be divisible by the number of workers to ensure an
    even split among workers. If it is not divisible, logging will give a warning but training will proceed.
    In addition, if using mutiprocessing, each shard MUST HAVE THE SAME NUMBER OF ENTRIES after filtering
    is applied. We currently do not check for this, but your program may hang if the shards are uneven!

    Additionally, please note that the len() of this DataLayer is assumed to be the length of the manifest
    after filtering. An incorrect manifest length may lead to some DataLoader issues down the line.

    Args:        
        dataset_meta: Dict of dataset names (string) to dataset metadata.
        audio_tar_filepaths: Either a list of audio tarball filepaths, or a
            string (can be brace-expandable).
        sample_rate: Sample rate to load audio as. If the audio is stored at a different sample rate, then it will
            be resampled.
        n_samples: Optional int, if provided then n_samples samples will be randomly sampled from the full
            audio file.
        shuffle_n (int): How many samples to look ahead and load to be shuffled.
            See WebDataset documentation for more details.
            Defaults to 0.
        min_duration: Optional float, if provided audio files in the training manifest shorter than 'min_duration'
            will be ignored.
        max_duration: Optional float, if provided audio files in the training manifest longer than 'max_duration'
            will be ignored.
        trunc_duration: Optional int, if provided audio will be truncated to at most 'trunc_duration' seconds.
        feature_processors: Optional, list of feature processors to run on training examples.
        shard_strategy (str): Tarred dataset shard distribution strategy chosen as a str value during ddp.
            -   `scatter`: The default shard strategy applied by WebDataset, where each node gets
                a unique set of shards, which are permanently pre-allocated and never changed at runtime.
            -   `replicate`: Optional shard strategy, where each node gets all of the set of shards
                available in the tarred dataset, which are permanently pre-allocated and never changed at runtime.
                The benefit of replication is that it allows each node to sample data points from the entire
                dataset independently of other nodes, and reduces dependence on value of `shuffle_n`.

                .. warning::
                    Replicated strategy allows every node to sample the entire set of available tarfiles,
                    and therefore more than one node may sample the same tarfile, and even sample the same
                    data points! As such, there is no assured guarantee that all samples in the dataset will be
                    sampled at least once during 1 epoch. Scattered strategy, on the other hand, on specific
                    occasions (when the number of shards is not divisible with ``world_size``), will not sample
                    the entire dataset. For these reasons it is not advisable to use tarred datasets as validation
                    or test datasets.
        global_rank (int): Worker rank, used for partitioning shards. Defaults to 0.
        world_size (int): Total number of processes, used for partitioning shards. Defaults to 0.
    """

    def __init__(
        self,
        dataset_meta: Dict,
        sample_rate: int,
        n_samples: Optional[int] = None,
        shuffle_n: int = 0,
        min_duration: float = 0.1,
        max_duration: Optional[float] = None,
        trunc_duration: Optional[float] = None,
        feature_processors: Optional[Dict[str, FeatureProcessor]] = None,
        shard_strategy: str = "scatter",
        global_rank: int = 0,
        world_size: int = 2,
        **kwargs,
    ):
        super().__init__()

        if len(kwargs) > 0:
            logging.warning(
                f"Arguments {kwargs.keys()} does not support for TarredVocoderDataset, they will be ignored."
            )

        self.sample_rate = sample_rate
        self.n_samples = n_samples

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
        self.audio_tar_filepaths = []
        for dataset_name, dataset_info in dataset_meta.items():
            audio_tar_filepaths = dataset_info.audio_tar_filepaths
            self.audio_tar_filepaths += [audio_tar_filepaths]
            dataset = DatasetMeta(**dataset_info)
            samples, _ = preprocess_manifest(
                dataset_name=dataset_name, dataset=dataset, min_duration=min_duration, max_duration=max_duration,
            )
            self.data_samples += samples

        self.file_id_to_sample_map = {}
        for sample in self.data_samples:
            file_id = os.path.splitext(os.path.basename(sample.manifest_entry["audio_filepath"]))[0]
            if file_id not in self.file_id_to_sample_map:
                self.file_id_to_sample_map[file_id] = sample
            else:
                raise ValueError(
                    f"Duplicate file_id {file_id} found in manifest {sample.manifest_entry['audio_filepath']}"
                )

        logging.info(f"world size: {world_size}")
        audio_tar_filepaths = expand_sharded_filepaths(
            sharded_filepaths=audio_tar_filepaths,
            global_rank=global_rank,
            world_size=world_size,
            shard_strategy=shard_strategy,
        )

        self._dataset = wd.WebDataset(audio_tar_filepaths, nodesplitter=None)

        if shuffle_n > 0:
            self._dataset = self._dataset.shuffle(shuffle_n, initial=shuffle_n)
        else:
            logging.info("WebDataset will not shuffle data. Consider setting shuffle_n > 0.")

        self._dataset = (
            self._dataset.rename(audio=VALID_FILE_FORMATS, key='__key__')
            .to_tuple('audio', 'key')
            .pipe(self._filter)
            .map(f=self._build_sample)
        )

    def _filter(self, iterator):
        class FilteredIterator:
            def __init__(self, file_id_to_sample_map):
                self.iterator = iterator
                self.file_id_to_sample_map = file_id_to_sample_map

            def __iter__(self):
                return self

            def __next__(self):
                while True:
                    audio_bytes, audio_filename = next(self.iterator)
                    file_id = os.path.splitext(os.path.basename(audio_filename))[0]
                    if file_id in self.file_id_to_sample_map:
                        return audio_bytes, audio_filename

        return FilteredIterator(self.file_id_to_sample_map)

    def _build_sample(self, tup):
        audio_bytes, audio_filename = tup
        file_id = os.path.splitext(os.path.basename(audio_filename))[0]
        data = self.file_id_to_sample_map[file_id]

        audio_array, sr = sf.read(file=io.BytesIO(audio_bytes), dtype='float32')
        if sr != self.sample_rate:
            logging.warning(
                f"Sample rate of {sr} does not match target sample rate of {self.sample_rate}. Resampling audio."
            )
            audio_array = librosa.core.resample(audio_array, orig_sr=sr, target_sr=self.sample_rate)

        audio_array = torch.from_numpy(audio_array)
        if self.n_samples:
            len_audio = audio_array.shape[0]
            if len_audio > self.n_samples:
                start = torch.randint(0, len_audio - self.n_samples, (1,))
                audio_array = audio_array[start : start + self.n_samples]
            else:
                audio_array = audio_array[: self.n_samples]

        if self.trunc_samples:
            audio_array = audio_array[: self.trunc_samples]

        audio_len = torch.tensor(audio_array.shape[0])

        example = {
            "dataset_name": data.dataset_name,
            "audio_filepath": audio_filename,
            "audio": audio_array,
            "audio_len": audio_len,
        }

        for processor in self.feature_processors:
            processor.process(example)

        return example

    def get_sampler(self, batch_size: int = 16):
        """
        Currently sampler is not supported for tarred dataset.
        """
        return None

    def collate_fn(self, batch):
        return audio_collate_fn(batch)

    def __iter__(self):
        return self._dataset.__iter__()

    def __len__(self):
        return len(self.file_id_to_sample_map)
