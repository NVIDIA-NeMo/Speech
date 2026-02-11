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
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import librosa
import numpy as np
import torch.utils.data

from nemo.collections.asr.parts.utils.manifest_utils import read_manifest
from nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers import BaseTokenizer
from nemo.collections.tts.parts.preprocessing.feature_processors import FeatureProcessor
from nemo.collections.tts.parts.preprocessing.features import Featurizer
from nemo.collections.tts.parts.utils.tts_dataset_utils import (
    _read_audio,
    beta_binomial_prior_distribution,
    chunk_and_tokenize_text_by_sentence,
    filter_dataset_by_duration,
    get_weighted_sampler,
    load_audio,
    stack_tensors,
)
from nemo.core.classes import Dataset
from nemo.utils import logging


@dataclass
class DatasetMeta:
    manifest_path: Path
    audio_dir: Path
    feature_dir: Path = None
    sample_weight: float = 1.0
    tokenizer_names: List[str] = None


@dataclass
class DatasetSample:
    dataset_name: str
    manifest_entry: Dict[str, Any]
    audio_dir: Optional[Path]
    feature_dir: Optional[Path]
    text: str
    speaker: str
    speaker_index: int = None
    tokenizer_names: List[str] = None


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

    def get_sampler(self, batch_size: int, world_size: int) -> Optional[torch.utils.data.Sampler]:
        if not self.weighted_sampling_steps_per_epoch:
            return None

        sampler = get_weighted_sampler(
            sample_weights=self.sample_weights,
            batch_size=batch_size,
            world_size=world_size,
            num_steps=self.weighted_sampling_steps_per_epoch,
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
                audio_dir=None if dataset.audio_dir is None else Path(dataset.audio_dir),
                feature_dir=None if dataset.feature_dir is None else Path(dataset.feature_dir),
                text=text,
                speaker=speaker,
                speaker_index=speaker_index,
                tokenizer_names=dataset.tokenizer_names,
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
            batch_dict["align_prior_matrix"] = stack_tensors(
                prior_list,
                max_lens=[text_max_len, spec_max_len],
            )

        for featurizer in self.featurizers:
            feature_dict = featurizer.collate_fn(batch)
            batch_dict.update(feature_dict)

        return batch_dict


class MagpieTTSDataset(TextToSpeechDataset):
    """
    Class for processing and loading text to speech training examples for Magpie-TTS model.
    In addition to the manifest structure for TextToSpeechDataset, we can have the following keys:
    context_audio_filepath, context_audio_duration, target_audio_codes_path, context_audio_codes_path.
    Note: target_audio_codes_path, context_audio_codes_path are absolute paths to the cached audio codes.
    If they are not present in the manifest or if load_cached_codes_if_available=False, then the audio will be loaded
    and codes will be computed on the fly in the model class.

    Args:
        dataset_meta: Dict of dataset names (string) to dataset metadata.
        sample_rate: Sample rate to load audio as. If the audio is stored at a different sample rate, then it will
            be resampled.
        weighted_sampling_steps_per_epoch: Optional int, If provided, then data will be sampled (with replacement) based on
            the sample weights provided in the dataset metadata. If None, then sample weights will be ignored.
        min_duration: Optional float, if provided audio files in the training manifest shorter than 'min_duration'
            will be ignored.
        max_duration: Optional float, if provided audio files in the training manifest longer than 'max_duration'
            will be ignored.
        volume_norm: Whether to apply volume normalization to loaded audio.
        codec_model_samples_per_frame: Num samples in waveform per codec frame (codec downsample factor).
        bos_id: Text BOS token id.
        eos_id: Text EOS token id.
        num_audio_codebooks: Number of audio codebooks.
        prior_scaling_factor: Scaling factor for the beta binomial prior distribution.
        load_cached_codes_if_available: Whether to load cached audio codes if available *_codes_path keys are available in the manifest.
        dataset_type: Dataset type (train, dev, test).
        tokenizer_config: Config of the tokenzizer used in worker_init_fn to setup the tokenizer (See Magpie-TTS yamls)
        load_16khz_audio: Whether to load 16khz audio for SV model.
        use_text_conditioning_tokenizer: Set True for text context conditioning.
        pad_context_text_to_max_duration: Whether to pad context text to max context audio frames.
        context_duration_min: Minimum duration of context audio in seconds.
        context_duration_max: Maximum duration of context audio in seconds.
        text_context_remapping: Dict defining mapping of multiple text contexts to a single text context.
        text_context_remapping_prob: Probability of remapping the original text context to a remapped text context.
    """

    def __init__(
        self,
        dataset_meta: Dict,
        sample_rate: int,
        weighted_sampling_steps_per_epoch: Optional[int] = None,
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
        volume_norm: bool = True,
        codec_model_samples_per_frame: int = None,
        bos_id: int = None,
        eos_id: int = None,
        num_audio_codebooks: int = None,
        prior_scaling_factor: float = None,
        load_cached_codes_if_available: bool = True,
        dataset_type: str = 'train',
        tokenizer_config=None,
        load_16khz_audio: bool = True,
        use_text_conditioning_tokenizer: bool = False,
        text_conditioning_tokenizer_name: str = None,
        pad_context_text_to_max_duration: bool = False,
        context_duration_min: float = 3.0,
        context_duration_max: float = 10.0,
        text_context_remapping: Dict[str, str] = None,
        text_context_remapping_prob: float = 0.0,
    ):
        super().__init__(
            dataset_meta=dataset_meta,
            sample_rate=sample_rate,
            text_tokenizer=None,
            weighted_sampling_steps_per_epoch=weighted_sampling_steps_per_epoch,
            speaker_path=None,
            featurizers=None,
            feature_processors=None,
            align_prior_hop_length=None,
            min_duration=min_duration,
            max_duration=max_duration,
            volume_norm=volume_norm,
        )
        self.bos_id = bos_id  # TODO @xueyang: this should be removed since no other places used it.
        self.eos_id = eos_id
        self.num_audio_codebooks = num_audio_codebooks
        self.codec_model_samples_per_frame = codec_model_samples_per_frame
        self.include_align_prior = prior_scaling_factor is not None
        self.prior_scaling_factor = prior_scaling_factor
        self.load_cached_codes_if_available = load_cached_codes_if_available
        self.dataset_type = dataset_type
        self.tokenizer_config = tokenizer_config
        self.text_tokenizer = None  # Assigned in worker_init_fn in model file
        self.load_16khz_audio = load_16khz_audio
        self.use_text_conditioning_tokenizer = use_text_conditioning_tokenizer
        self.text_conditioning_tokenizer_name = text_conditioning_tokenizer_name
        self.pad_context_text_to_max_duration = pad_context_text_to_max_duration
        self.context_duration_min = context_duration_min
        self.context_duration_max = context_duration_max
        self.text_context_remapping = text_context_remapping
        self.text_context_remapping_prob = text_context_remapping_prob

    def get_num_audio_samples_to_slice(self, duration, sample_rate):
        num_codec_frames = int(duration * sample_rate / self.codec_model_samples_per_frame)
        num_audio_samples = num_codec_frames * self.codec_model_samples_per_frame
        return num_audio_samples

    def __getitem__(self, index):
        data = self.data_samples[index]
        tokenizer_name = "english_phoneme"  # Default to english phoneme tokenizer
        if data.tokenizer_names is not None:
            # Pick a random tokenizer from the list of tokenizers
            tokenizer_name = random.choice(data.tokenizer_names)
        tokens = self.text_tokenizer.encode(text=data.text, tokenizer_name=tokenizer_name)
        tokens = tokens + [self.eos_id]  # Not adding BOS id
        tokens = torch.tensor(tokens, dtype=torch.int32)
        text_len = tokens.shape[0]

        example = {
            "dataset_name": data.dataset_name,
            "tokens": tokens,
            "text_len": text_len,
        }

        if self.load_cached_codes_if_available and 'target_audio_codes_path' in data.manifest_entry:
            audio_codes_path = data.manifest_entry['target_audio_codes_path']
            audio_codes = torch.load(audio_codes_path)  # (C, T)
            spec_len = audio_codes.shape[1] + 1  # +1 for EOS
            audio_codes_len = audio_codes.shape[1]
            example['audio_codes'] = audio_codes
            example['audio_codes_len'] = audio_codes_len
            example['audio_filepath'] = audio_codes_path
            if 'audio_filepath' in data.manifest_entry:
                # If audio_filepath is available, then use the actual audio file path.
                example['audio_filepath'] = data.manifest_entry['audio_filepath']
        elif 'audio_filepath' in data.manifest_entry:
            # Only load audio if codes are not available
            audio_array, _, audio_filepath_rel = load_audio(
                manifest_entry=data.manifest_entry,
                audio_dir=data.audio_dir,
                sample_rate=self.sample_rate,
                volume_norm=self.volume_norm,
            )
            audio = torch.tensor(audio_array, dtype=torch.float32)
            # Pad audio to be multiple of downsample factor
            audio = torch.nn.functional.pad(
                audio,
                (0, self.codec_model_samples_per_frame - (audio.shape[0] % self.codec_model_samples_per_frame)),
                value=0,
            )
            audio_len = audio.shape[0]
            example['audio_filepath'] = data.manifest_entry['audio_filepath']
            example['audio'] = audio
            example['audio_len'] = audio_len
            spec_len = int(audio_len / self.codec_model_samples_per_frame) + 1  # +1 for EOS

        if self.load_cached_codes_if_available and 'context_audio_codes_path' in data.manifest_entry:
            context_audio_codes_path = data.manifest_entry['context_audio_codes_path']
            context_audio_codes = torch.load(context_audio_codes_path)  # (8, T)
            # Sample random duration between self.context_duration_min and self.context_duration_max
            _context_duration_to_slice = random.uniform(self.context_duration_min, self.context_duration_max)
            _num_frames_to_slice = int(
                _context_duration_to_slice * self.sample_rate / self.codec_model_samples_per_frame
            )
            if _num_frames_to_slice < context_audio_codes.shape[1]:
                start_idx = random.randint(0, context_audio_codes.shape[1] - _num_frames_to_slice)
                context_audio_codes = context_audio_codes[:, start_idx : start_idx + _num_frames_to_slice]
            else:
                # Repeaet the audio if it is shorter than the desired duration
                _num_repeats = int(np.ceil(_num_frames_to_slice / context_audio_codes.shape[1]))
                # context_audio_codes is a tensor of shape (num_codebooks, T)
                context_audio_codes_repeated = context_audio_codes.repeat(1, _num_repeats)
                context_audio_codes = context_audio_codes_repeated[:, :_num_frames_to_slice]

            context_audio_codes_len = context_audio_codes.shape[1]
            example['context_audio_codes'] = context_audio_codes
            example['context_audio_codes_len'] = context_audio_codes_len
        elif 'context_audio_filepath' in data.manifest_entry:
            context_audio_filepath = os.path.join(data.audio_dir, data.manifest_entry['context_audio_filepath'])
            context_duration = data.manifest_entry['context_audio_duration']
            context_audio_array = _read_audio(
                audio_filepath=context_audio_filepath,
                sample_rate=self.sample_rate,
                offset=0,
                duration=context_duration,
            )
            context_audio_array = context_audio_array.samples
            _context_duration_to_slice = random.uniform(self.context_duration_min, self.context_duration_max)
            _num_samples_to_slice = self.get_num_audio_samples_to_slice(_context_duration_to_slice, self.sample_rate)
            if _num_samples_to_slice < len(context_audio_array):
                start_idx = random.randint(0, len(context_audio_array) - _num_samples_to_slice)
                context_audio_array = context_audio_array[start_idx : start_idx + _num_samples_to_slice]
            else:
                # Repeaet the audio if it is shorter than the desired duration
                _num_repeats = int(np.ceil(_num_samples_to_slice / len(context_audio_array)))
                context_audio_array = np.tile(context_audio_array, _num_repeats)
                context_audio_array = context_audio_array[:_num_samples_to_slice]
            context_audio = torch.tensor(context_audio_array, dtype=torch.float32)
            context_audio_len = context_audio.shape[0]
            example['context_audio'] = context_audio
            example['context_audio_len'] = context_audio_len
        else:
            # We always want to have context_audio_codes if available for multi-encoder model. These are ignored for singlencoder model.
            # If context audio is not available, just use a dummy context_audio_codes
            # (Will be used in text context scenario)
            if self.load_cached_codes_if_available:
                context_audio_codes = torch.zeros([self.num_audio_codebooks, 0], dtype=torch.int32)
                context_audio_codes_len = 0
                example['context_audio_codes'] = context_audio_codes
                example['context_audio_codes_len'] = context_audio_codes_len
            else:
                # @shehzeenh: Added this condition so that a batch does not have a mix of context_audio and context_audio_codes
                # @blisc: Added a +1. If we send in exactly 882 samples, then a conv layer complains about padding.
                #         Adding 883 works. This occurs when we use text context during inference.
                context_audio = torch.zeros(self.codec_model_samples_per_frame + 1, dtype=torch.float32)
                context_audio_len = context_audio.shape[0]
                example['context_audio'] = context_audio
                example['context_audio_len'] = context_audio_len

        # 16kHz audio is used for SV model
        if self.load_16khz_audio:
            if 'context_audio_filepath' in data.manifest_entry:
                # If context_audio_filepath is available, then use that for 16khz audio for SV model
                context_audio_filepath = os.path.join(data.audio_dir, data.manifest_entry['context_audio_filepath'])
                context_duration = data.manifest_entry['context_audio_duration']
                audio_array_16khz = _read_audio(
                    audio_filepath=context_audio_filepath, sample_rate=16000, offset=0, duration=context_duration
                ).samples
            else:
                # Otherwise, load the target audio file.
                audio_array_16khz, _, _ = load_audio(
                    manifest_entry=data.manifest_entry,
                    audio_dir=data.audio_dir,
                    sample_rate=16000,
                    volume_norm=self.volume_norm,
                )
            _context_duration_to_slice = random.uniform(self.context_duration_min, self.context_duration_max)
            _num_samples_to_slice = int(_context_duration_to_slice * 16000)
            if _num_samples_to_slice < len(audio_array_16khz):
                start_idx = random.randint(0, len(audio_array_16khz) - _num_samples_to_slice)
                audio_array_16khz = audio_array_16khz[start_idx : start_idx + _num_samples_to_slice]
            audio_16khz = torch.tensor(audio_array_16khz, dtype=torch.float32)
            audio_len_16khz = audio_16khz.shape[0]
            example['audio_16khz'] = audio_16khz
            example['audio_len_16khz'] = audio_len_16khz

        if self.use_text_conditioning_tokenizer:
            if 'context_text' in data.manifest_entry:
                context_text = data.manifest_entry['context_text']
                if self.text_context_remapping is not None and context_text in self.text_context_remapping:
                    if self.dataset_type == 'train' and random.random() < self.text_context_remapping_prob:
                        # Only remap during training. Give the exact text context during inference.
                        context_text = self.text_context_remapping[context_text]
                context_tokens = self.text_tokenizer.encode(context_text, self.text_conditioning_tokenizer_name)
                example['has_text_context'] = True
            else:
                context_tokens = self.text_tokenizer.encode("[NO TEXT CONTEXT]", self.text_conditioning_tokenizer_name)
                example['has_text_context'] = False
            if self.pad_context_text_to_max_duration:
                _required_len = (
                    int(self.context_duration_max * self.sample_rate / self.codec_model_samples_per_frame) + 2
                )  # +2 for BOS and EOS
                if len(context_tokens) < _required_len:
                    _pad_id = self.text_tokenizer.tokenizer_pad_ids[self.text_conditioning_tokenizer_name]
                    context_tokens += [_pad_id] * (_required_len - len(context_tokens))
                else:
                    context_tokens = context_tokens[:_required_len]

            context_tokens = torch.tensor(context_tokens, dtype=torch.int32)
            context_text_len = context_tokens.shape[0]
            example['context_text_tokens'] = context_tokens
            example['context_text_len'] = context_text_len

        if self.include_align_prior:
            align_prior = beta_binomial_prior_distribution(
                phoneme_count=text_len, mel_count=spec_len, scaling_factor=self.prior_scaling_factor
            )
            align_prior = torch.tensor(align_prior, dtype=torch.float32)
            example["align_prior"] = align_prior

        if "original_text" in data.manifest_entry:
            # Raw Text is used as the GT for CER/WER computation in DPO pref data generation
            # and GRPO reward setup. For manifests in which the 'text' field is phonemized,
            # we use the 'original_text' field as the raw text. Otherwise, we use the regular text field.
            example['raw_text'] = data.manifest_entry['original_text']
        else:
            example['raw_text'] = data.text

        example['language'] = data.manifest_entry.get('language', 'en')

        if "reward" in data.manifest_entry:
            example["reward"] = data.manifest_entry["reward"]

        # Get speaker index if available (for models with baked context embeddings)
        if 'speaker_index' in data.manifest_entry:
            example['speaker_index'] = data.manifest_entry['speaker_index']

        return example

    def collate_fn(self, batch: List[dict]):
        dataset_name_list = []
        audio_filepath_list = []
        audio_list = []
        audio_len_list = []
        audio_list_16khz = []
        audio_len_list_16khz = []
        token_list = []
        token_len_list = []
        prior_list = []
        audio_codes_list = []
        audio_codes_len_list = []
        context_audio_list = []
        context_audio_len_list = []
        context_audio_codes_list = []
        context_audio_codes_len_list = []
        context_text_tokens_list = []
        context_text_tokens_len_list = []
        context_has_text_context_list = []
        reward_list = []
        raw_text_list = []
        language_list = []
        speaker_indices_list = []
        for example in batch:
            dataset_name_list.append(example["dataset_name"])
            raw_text_list.append(example["raw_text"])
            language_list.append(example["language"])

            token_list.append(example["tokens"])
            token_len_list.append(example["text_len"])

            if 'audio_filepath' in example:
                audio_filepath_list.append(example["audio_filepath"])

            if 'audio' in example:
                audio_list.append(example["audio"])
                audio_len_list.append(example["audio_len"])

            if 'audio_16khz' in example:
                audio_list_16khz.append(example["audio_16khz"])
                audio_len_list_16khz.append(example["audio_len_16khz"])

            if 'audio_codes' in example:
                audio_codes_list.append(example['audio_codes'])
                audio_codes_len_list.append(example['audio_codes_len'])

            if 'context_audio' in example:
                context_audio_list.append(example['context_audio'])
                context_audio_len_list.append(example['context_audio_len'])

            if 'context_audio_codes' in example:
                context_audio_codes_list.append(example['context_audio_codes'])
                context_audio_codes_len_list.append(example['context_audio_codes_len'])

            if 'context_text_tokens' in example:
                context_text_tokens_list.append(example['context_text_tokens'])
                context_text_tokens_len_list.append(example['context_text_len'])
                context_has_text_context_list.append(example['has_text_context'])

            if 'reward' in example:
                reward_list.append(example['reward'])

            if 'speaker_index' in example:
                speaker_indices_list.append(example['speaker_index'])

            if self.include_align_prior:
                prior_list.append(example["align_prior"])

        batch_token_len = torch.IntTensor(token_len_list)
        token_max_len = int(batch_token_len.max().item())
        batch_tokens = stack_tensors(token_list, max_lens=[token_max_len], pad_value=self.text_tokenizer.pad)

        batch_dict = {
            "dataset_names": dataset_name_list,
            "raw_texts": raw_text_list,
            "languages": language_list,
            "audio_filepaths": audio_filepath_list,
            "text": batch_tokens,
            "text_lens": batch_token_len,
        }

        if len(audio_list) > 0:
            batch_audio_len = torch.IntTensor(audio_len_list)
            audio_max_len = int(batch_audio_len.max().item())
            batch_audio = stack_tensors(audio_list, max_lens=[audio_max_len])
            batch_dict['audio'] = batch_audio
            batch_dict['audio_lens'] = batch_audio_len

        if len(audio_list_16khz) > 0:
            batch_audio_len_16khz = torch.IntTensor(audio_len_list_16khz)
            audio_max_len_16khz = int(batch_audio_len_16khz.max().item())
            batch_audio_16khz = stack_tensors(audio_list_16khz, max_lens=[audio_max_len_16khz])
            batch_dict['audio_16khz'] = batch_audio_16khz
            batch_dict['audio_lens_16khz'] = batch_audio_len_16khz

        if len(audio_codes_list) > 0:
            batch_audio_codes_len = torch.IntTensor(audio_codes_len_list)
            audio_codes_max_len = int(batch_audio_codes_len.max().item())
            batch_audio_codes = stack_tensors(audio_codes_list, max_lens=[audio_codes_max_len])
            batch_dict['audio_codes'] = batch_audio_codes
            batch_dict['audio_codes_lens'] = batch_audio_codes_len

        if len(context_audio_list) > 0:
            batch_context_audio_len = torch.IntTensor(context_audio_len_list)
            context_audio_max_len = int(batch_context_audio_len.max().item())
            batch_context_audio = stack_tensors(context_audio_list, max_lens=[context_audio_max_len])
            batch_dict['context_audio'] = batch_context_audio
            batch_dict['context_audio_lens'] = batch_context_audio_len

        if len(context_audio_codes_list) > 0:
            batch_context_audio_codes_len = torch.IntTensor(context_audio_codes_len_list)
            context_audio_codes_max_len = int(batch_context_audio_codes_len.max().item())
            # TODO @xueyang: verify if batch_context_audio_codes are integer.
            batch_context_audio_codes = stack_tensors(context_audio_codes_list, max_lens=[context_audio_codes_max_len])
            batch_dict['context_audio_codes'] = batch_context_audio_codes
            batch_dict['context_audio_codes_lens'] = batch_context_audio_codes_len

        if self.use_text_conditioning_tokenizer:
            batch_context_text_tokens_len = torch.IntTensor(context_text_tokens_len_list)
            context_text_tokens_max_len = int(batch_context_text_tokens_len.max().item())
            # TODO @xueyang: potential bugs if self.tokenizer.pad is not 0.0. verify if batch_context_text_tokens are integer.
            batch_context_text_tokens = stack_tensors(context_text_tokens_list, max_lens=[context_text_tokens_max_len])
            batch_dict['context_text_tokens'] = batch_context_text_tokens
            batch_dict['context_text_tokens_lens'] = batch_context_text_tokens_len
            batch_dict['has_text_context'] = torch.BoolTensor(context_has_text_context_list)

        if self.include_align_prior:
            spec_max_len = max([prior.shape[0] for prior in prior_list])
            text_max_len = max([prior.shape[1] for prior in prior_list])
            batch_dict["align_prior_matrix"] = stack_tensors(
                prior_list,
                max_lens=[text_max_len, spec_max_len],
            )

        if len(reward_list) > 0:
            batch_dict['rewards'] = torch.FloatTensor(reward_list)

        if len(speaker_indices_list) > 0:
            batch_dict['speaker_indices'] = torch.tensor(speaker_indices_list, dtype=torch.int64)

        # Assert no more than one of audio or audio_codes in the batch
        if 'audio' in batch_dict:
            assert 'audio_codes' not in batch_dict

        # Assert no more than one of context_audio or context_audio_codes in the batch
        if 'context_audio' in batch_dict:
            assert 'context_audio_codes' not in batch_dict

        return batch_dict


class MagpieTTSDatasetDPO(MagpieTTSDataset):
    """
    This class is meant to be used with the DPO model. To generate manifests for this dataset, please use
        - scripts/magpietts/dpo/create_text_contextpairs.py
        - scripts/magpietts/dpo/create_preference_pairs.py
    in sequence to generate samples and create preference pairs.
    """

    def __len__(self):
        return len(self.data_samples) // 2

    def __getitem__(self, index):
        chosen_example = super().__getitem__(index * 2)
        rejected_example = super().__getitem__(index * 2 + 1)
        assert chosen_example['reward'] == 1.0
        assert rejected_example['reward'] < 1.0
        return {"chosen": chosen_example, "rejected": rejected_example}

    def collate_fn(self, batch: List[dict]):
        chosen_batch = [example['chosen'] for example in batch]
        rejected_batch = [example['rejected'] for example in batch]
        chosen_collated = super().collate_fn(chosen_batch)
        rejected_collated = super().collate_fn(rejected_batch)
        return {"chosen": chosen_collated, "rejected": rejected_collated}


class LongFormTTSInferenceDataset(MagpieTTSDataset):
    """
    Dataset for longform TTS inference with sentence-level text chunking.

    Inherits from MagpieTTSDataset to reuse context audio loading, text conditioning,
    and other preprocessing logic. Adds sentence-level text chunking on top.

    Args:
        dataset_meta: Dataset metadata dictionary (same format as MagpieTTSDataset).
        sample_rate: Audio sample rate.
        tokenizer_name: Name of the tokenizer to use for sentence chunking.
        codec_model_samples_per_frame: Samples per codec frame.
        eos_id: End-of-sequence token ID.
        audio_bos_id: Audio BOS token ID (for target audio).
        audio_eos_id: Audio EOS token ID (for target audio).
        context_audio_bos_id: Context audio BOS token ID.
        context_audio_eos_id: Context audio EOS token ID.
        num_audio_codebooks: Number of audio codebooks.
        context_duration_min: Minimum context duration in seconds.
        context_duration_max: Maximum context duration in seconds.
        use_text_conditioning_tokenizer: Whether model uses text conditioning encoder.
        text_conditioning_tokenizer_name: Name of text conditioning tokenizer.
        pad_context_text_to_max_duration: Whether to pad context text.
        load_16khz_audio: Whether to load 16kHz audio for SV model.
    """

    def __init__(
        self,
        dataset_meta: Dict[str, Any],
        sample_rate: int,
        tokenizer_name: str,
        codec_model_samples_per_frame: int,
        eos_id: int,
        num_audio_codebooks: int,
        context_duration_min: float = 3.0,
        context_duration_max: float = 10.0,
        use_text_conditioning_tokenizer: bool = False,
        text_conditioning_tokenizer_name: str = None,
        pad_context_text_to_max_duration: bool = False,
        load_16khz_audio: bool = False,
        **kwargs,
    ):
        # Initialize parent - handles manifest reading and context audio loading
        super().__init__(
            dataset_meta=dataset_meta,
            sample_rate=sample_rate,
            codec_model_samples_per_frame=codec_model_samples_per_frame,
            eos_id=eos_id,
            num_audio_codebooks=num_audio_codebooks,
            context_duration_min=context_duration_min,
            context_duration_max=context_duration_max,
            use_text_conditioning_tokenizer=use_text_conditioning_tokenizer,
            text_conditioning_tokenizer_name=text_conditioning_tokenizer_name,
            pad_context_text_to_max_duration=pad_context_text_to_max_duration,
            load_16khz_audio=load_16khz_audio,
            load_cached_codes_if_available=True,  # Prefer codes for inference
            dataset_type='test',
            **kwargs,
        )
        self.tokenizer_name = tokenizer_name

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Add sentence chunking on top of parent's __getitem__.

        Returns:
            Dictionary containing all parent fields plus:
                - idx: Sample index
                - chunked_tokens: List of tokenized text chunks (per sentence)
                - chunked_tokens_len: List of token lengths
                - entry: Original manifest entry
        """
        # Get text for sentence chunking
        data = self.data_samples[idx]
        text = data.text  # entry.get("normalized_text", entry.get("text", ""))

        # Call parent to get ALL the context audio, text conditioning, etc.
        example = super().__getitem__(idx)

        # Sentence chunking (longform-specific)
        chunked_tokens, chunked_tokens_len, _ = chunk_and_tokenize_text_by_sentence(
            text,
            self.tokenizer_name,
            self.text_tokenizer,
            self.eos_id,
            language=example['language'],
        )

        # Handle empty text edge case
        if not chunked_tokens:
            chunked_tokens = [torch.tensor([self.eos_id], dtype=torch.int32)]
            chunked_tokens_len = [1]

        # Add longform-specific fields
        example['idx'] = idx
        example['chunked_tokens'] = chunked_tokens
        example['chunked_tokens_len'] = chunked_tokens_len

        return example

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate function for batching longform samples.

        Calls parent's collate_fn to handle context audio, text conditioning, etc.,
        then adds longform-specific fields (chunked_tokens).
        """
        # Call parent's collate_fn to handle all standard fields
        batch_dict = super().collate_fn(batch)

        # Add longform-specific fields
        indices = []
        chunked_tokens_list = []
        chunked_tokens_lens_list = []

        # Find max number of chunks across batch
        max_num_chunks = max(len(sample['chunked_tokens']) for sample in batch)

        for sample in batch:
            indices.append(sample['idx'])

            # Pad chunked tokens to max_num_chunks with single EOS token
            num_padding = max_num_chunks - len(sample['chunked_tokens'])
            padded_tokens = sample['chunked_tokens'] + [
                torch.tensor([self.eos_id], dtype=torch.int32) for _ in range(num_padding)
            ]
            padded_lens = sample['chunked_tokens_len'] + [1] * num_padding

            chunked_tokens_list.append(padded_tokens)
            chunked_tokens_lens_list.append(padded_lens)

        # Add longform-specific fields to batch_dict
        batch_dict['idx'] = indices
        batch_dict['chunked_tokens'] = chunked_tokens_list
        batch_dict['chunked_tokens_lens'] = chunked_tokens_lens_list

        return batch_dict
