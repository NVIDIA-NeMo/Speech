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

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import librosa
import torch.utils.data

from nemo.collections.asr.parts.utils.manifest_utils import read_manifest
from nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers import BaseTokenizer
from nemo.collections.tts.parts.preprocessing.feature_processors import FeatureProcessor
from nemo.collections.tts.parts.preprocessing.features import Featurizer
from nemo.collections.tts.parts.utils.tts_dataset_utils import (
    beta_binomial_prior_distribution,
    filter_dataset_by_duration,
    get_weighted_sampler,
    load_audio,
    stack_tensors,
)
from nemo.core.classes import Dataset
from nemo.utils import logging
from nemo.utils.decorators import experimental


@dataclass
class DatasetMeta:
    manifest_path: Path
    audio_dir: Path
    feature_dir: Path
    sample_weight: float = 1.0


@dataclass
class DatasetSample:
    dataset_name: str
    manifest_entry: Dict[str, Any]
    audio_dir: Path
    feature_dir: Path
    text: str
    speaker: str
    speaker_index: int = None


@experimental
class TextToSpeechDataset(Dataset):
    """
    Class for processing and loading text to speech training examples.

    Args:
        dataset_meta: Dict of dataset names (string) to dataset metadata.
        sample_rate: Sample rate to load audio as. If the audio is stored at a different sample rate, then it will
            be resampled.
        text_tokenizer: Tokenizer to apply to the text field.
        weighted_sampling_steps_per_epoch: Optional int, If provided, then data will be sampled (with replacement) based on
            the sample weights provided in the dataset metadata. If None, then sample weights will be ignored.
        speaker_path: Optional, path to JSON file with speaker indices, for multi-speaker training. Can be created with
            scripts.dataset_processing.tts.create_speaker_map.py
        featurizers: Optional, list of featurizers to load feature data from. Should be the same config provided
            when running scripts.dataset_processing.tts.compute_features.py before training.
        feature_processors: Optional, list of feature processors to run on training examples.
        align_prior_hop_length: Optional int, hop length of audio features.
            If provided alignment prior will be calculated and included in batch output. Must match hop length
            of audio features used for training.
        min_duration: Optional float, if provided audio files in the training manifest shorter than 'min_duration'
            will be ignored.
        max_duration: Optional float, if provided audio files in the training manifest longer than 'max_duration'
            will be ignored.
        volume_norm: Whether to apply volume normalization to loaded audio.
    """

    def __init__(
        self,
        dataset_meta: Dict,
        sample_rate: int,
        text_tokenizer: BaseTokenizer,
        weighted_sampling_steps_per_epoch: Optional[int] = None,
        speaker_path: Optional[Path] = None,
        featurizers: Optional[Dict[str, Featurizer]] = None,
        feature_processors: Optional[Dict[str, FeatureProcessor]] = None,
        align_prior_hop_length: Optional[int] = None,
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
        volume_norm: bool = True,
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.text_tokenizer = text_tokenizer
        self.weighted_sampling_steps_per_epoch = weighted_sampling_steps_per_epoch
        self.align_prior_hop_length = align_prior_hop_length
        self.include_align_prior = self.align_prior_hop_length is not None
        self.volume_norm = volume_norm

        if speaker_path:
            self.include_speaker = True
            with open(speaker_path, 'r', encoding="utf-8") as speaker_f:
                speaker_index_map = json.load(speaker_f)
        else:
            self.include_speaker = False
            speaker_index_map = None

        if featurizers:
            logging.info(f"Found featurizers {featurizers.keys()}")
            self.featurizers = list(featurizers.values())
        else:
            self.featurizers = []

        if feature_processors:
            logging.info(f"Found featurize processors {feature_processors.keys()}")
            self.feature_processors = list(feature_processors.values())
        else:
            self.feature_processors = []

        self.data_samples = []
        self.sample_weights = []
        for dataset_name, dataset_info in dataset_meta.items():
            dataset = DatasetMeta(**dataset_info)
            samples, weights = self._preprocess_manifest(
                dataset_name=dataset_name,
                dataset=dataset,
                min_duration=min_duration,
                max_duration=max_duration,
                speaker_index_map=speaker_index_map,
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

    def _preprocess_manifest(
        self,
        dataset_name: str,
        dataset: DatasetMeta,
        min_duration: float,
        max_duration: float,
        speaker_index_map: Dict[str, int],
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

            if "normalized_text" in entry:
                text = entry["normalized_text"]
            else:
                text = entry["text"]

            if self.include_speaker:
                speaker = entry["speaker"]
                speaker_index = speaker_index_map[speaker]
            else:
                speaker = None
                speaker_index = 0

            sample = DatasetSample(
                dataset_name=dataset_name,
                manifest_entry=entry,
                audio_dir=Path(dataset.audio_dir),
                feature_dir=Path(dataset.feature_dir),
                text=text,
                speaker=speaker,
                speaker_index=speaker_index,
            )
            samples.append(sample)
            sample_weights.append(dataset.sample_weight)

        return samples, sample_weights

    def __len__(self):
        return len(self.data_samples)

    def __getitem__(self, index):
        data = self.data_samples[index]

        audio_array, _, audio_filepath_rel = load_audio(
            manifest_entry=data.manifest_entry,
            audio_dir=data.audio_dir,
            sample_rate=self.sample_rate,
            volume_norm=self.volume_norm,
        )
        audio = torch.tensor(audio_array, dtype=torch.float32)
        audio_len = audio.shape[0]

        tokens = self.text_tokenizer(data.text)
        tokens = torch.tensor(tokens, dtype=torch.int32)
        text_len = tokens.shape[0]

        example = {
            "dataset_name": data.dataset_name,
            "audio_filepath": audio_filepath_rel,
            "audio": audio,
            "audio_len": audio_len,
            "tokens": tokens,
            "text_len": text_len,
        }

        if data.speaker is not None:
            example["speaker"] = data.speaker
            example["speaker_index"] = data.speaker_index

        if self.include_align_prior:
            spec_len = 1 + librosa.core.samples_to_frames(audio_len, hop_length=self.align_prior_hop_length)
            align_prior = beta_binomial_prior_distribution(phoneme_count=text_len, mel_count=spec_len)
            align_prior = torch.tensor(align_prior, dtype=torch.float32)
            example["align_prior"] = align_prior

        for featurizer in self.featurizers:
            feature_dict = featurizer.load(
                manifest_entry=data.manifest_entry, audio_dir=data.audio_dir, feature_dir=data.feature_dir
            )
            example.update(feature_dict)

        for processor in self.feature_processors:
            processor.process(example)

        return example

    def collate_fn(self, batch: List[dict]):
        dataset_name_list = []
        audio_filepath_list = []
        audio_list = []
        audio_len_list = []
        token_list = []
        token_len_list = []
        speaker_list = []
        prior_list = []

        for example in batch:
            dataset_name_list.append(example["dataset_name"])
            audio_filepath_list.append(example["audio_filepath"])

            audio_list.append(example["audio"])
            audio_len_list.append(example["audio_len"])

            token_list.append(example["tokens"])
            token_len_list.append(example["text_len"])

            if self.include_speaker:
                speaker_list.append(example["speaker_index"])

            if self.include_align_prior:
                prior_list.append(example["align_prior"])

        batch_audio_len = torch.IntTensor(audio_len_list)
        audio_max_len = int(batch_audio_len.max().item())

        batch_token_len = torch.IntTensor(token_len_list)
        token_max_len = int(batch_token_len.max().item())

        batch_audio = stack_tensors(audio_list, max_lens=[audio_max_len])
        batch_tokens = stack_tensors(token_list, max_lens=[token_max_len], pad_value=self.text_tokenizer.pad)

        batch_dict = {
            "dataset_names": dataset_name_list,
            "audio_filepaths": audio_filepath_list,
            "audio": batch_audio,
            "audio_lens": batch_audio_len,
            "text": batch_tokens,
            "text_lens": batch_token_len,
        }

        if self.include_speaker:
            batch_dict["speaker_id"] = torch.IntTensor(speaker_list)

        if self.include_align_prior:
            spec_max_len = max([prior.shape[0] for prior in prior_list])
            text_max_len = max([prior.shape[1] for prior in prior_list])
            batch_dict["align_prior_matrix"] = stack_tensors(prior_list, max_lens=[text_max_len, spec_max_len],)

        for featurizer in self.featurizers:
            feature_dict = featurizer.collate_fn(batch)
            batch_dict.update(feature_dict)

        return batch_dict
