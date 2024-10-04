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

import copy
import io
import json
import os
from dataclasses import dataclass
from math import isclose
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from lhotse.dataset import AudioSamples
from omegaconf import DictConfig, ListConfig, open_dict

from nemo.collections.asr.data import audio_to_text, audio_to_text_dataset
from nemo.collections.asr.parts.preprocessing.perturb import WhiteNoisePerturbation, process_augmentations
from nemo.collections.asr.parts.preprocessing.segment import AudioSegment
from nemo.collections.asr.parts.utils.manifest_utils import read_manifest
from nemo.collections.common.data.dataset import ConcatDataset
from nemo.collections.common.parts.preprocessing.manifest import get_full_path
from nemo.core.classes import Serialization
from nemo.utils import logging


@dataclass
class AudioNoiseItem:
    sample_id: str | None = None
    audio: torch.Tensor | None = None
    audio_len: torch.Tensor | None = None
    noise: torch.Tensor | None = None
    noise_len: torch.Tensor | None = None
    noisy_audio: torch.Tensor | None = None
    noisy_audio_len: torch.Tensor | None = None


@dataclass
class AudioNoiseBatch:
    sample_id: list | None = None
    audio: torch.Tensor | None = None
    audio_len: torch.Tensor | None = None
    noise: torch.Tensor | None = None
    noise_len: torch.Tensor | None = None
    noisy_audio: torch.Tensor | None = None
    noisy_audio_len: torch.Tensor | None = None


def _parse_manifest_item(line: str, manifest_file: str) -> Dict[str, Any]:
    """
    Specialized function to parse the manifest file by ignoring text,
    such that nemo dataset can save time on tokenizing text.
    """
    item = json.loads(line)

    # Audio file
    if 'audio_filename' in item:
        item['audio_file'] = item.pop('audio_filename')
    elif 'audio_filepath' in item:
        item['audio_file'] = item.pop('audio_filepath')
    else:
        raise KeyError(f"No 'audio_filename' or 'audio_filepath' in manifest item: {item}")

    item['audio_file'] = get_full_path(audio_file=item['audio_file'], manifest_file=manifest_file)

    # Duration.
    if 'duration' not in item:
        item['duration'] = None

    # dummy text
    item['text'] = ""

    item = dict(
        audio_file=item['audio_file'],
        duration=item['duration'],
        text=item['text'],
        offset=item.get('offset', None),
        speaker=item.get('speaker', None),
        orig_sr=item.get('orig_sample_rate', None),
        token_labels=item.get('token_labels', None),
        lang=item.get('lang', None),
    )
    return item


def _audio_noise_collate_fn(batch: List[AudioNoiseItem], batch_augmentor: Any = None) -> AudioNoiseBatch:
    audios = [x.audio for x in batch]
    audio_lengths = [x.audio_len for x in batch]
    max_audio_len = max(audio_lengths).item()

    noises = [x.noise for x in batch]
    noise_lengths = [x.noise_len for x in batch]

    noisy_audios = [x.noisy_audio for x in batch]
    noisy_audio_lengths = [x.noisy_audio_len for x in batch]

    audio_signal_list = []
    noise_signal_list = []
    noisy_audio_signal_list = []
    for i, audio in enumerate(audios):
        audio_len = audio.size(0)
        if audio_len < max_audio_len:
            pad = (0, max_audio_len - audio_len)
            audio = torch.nn.functional.pad(audio, pad)
        audio_signal_list.append(audio)

        noise = noises[i]
        noise_len = noise.size(0)
        if noise_len < max_audio_len:
            pad = (0, max_audio_len - noise_len)
            noise = torch.nn.functional.pad(noise, pad)
        noise_signal_list.append(noise[:max_audio_len])

        noisy_audio = noisy_audios[i]
        noisy_audio_len = noisy_audio.size(0)
        if noisy_audio_len < max_audio_len:
            pad = (0, max_audio_len - noisy_audio_len)
            noisy_audio = torch.nn.functional.pad(noisy_audio, pad)
        noisy_audio_signal_list.append(noisy_audio[:max_audio_len])

    audio_signal = torch.stack(audio_signal_list).float()
    audio_lengths = torch.stack(audio_lengths).long()
    noise_signal = torch.stack(noise_signal_list).float()
    noise_lengths = torch.stack(noise_lengths).long()
    noisy_audio_signal = torch.stack(noisy_audio_signal_list).float()
    noisy_audio_lengths = torch.stack(noisy_audio_lengths).long()

    output = AudioNoiseBatch(
        audio=audio_signal,
        audio_len=audio_lengths,
        noise=noise_signal,
        noise_len=noise_lengths,
        noisy_audio=noisy_audio_signal,
        noisy_audio_len=noisy_audio_lengths,
    )

    if batch_augmentor is not None:
        output = batch_augmentor(output)

    return output


def load_noise_manifest(noise_manifest: str | ListConfig | None):
    """
    load noise manifest from a single or a list of manifest files
    """
    if noise_manifest is None:
        return []

    if isinstance(noise_manifest, str):
        noise_manifest = noise_manifest.split(',')

    noise_data = []
    for manifest in noise_manifest:
        curr_data = read_manifest(manifest)
        for i in range(len(curr_data)):
            curr_data[i]['audio_filepath'] = get_full_path(curr_data[i]['audio_filepath'], manifest)
        noise_data.extend(curr_data)
    return noise_data


def load_noise_audio(
    sample: Dict[str, Any],
    sample_rate: int,
    max_audio_len: Optional[int] = None,
    pad_to_max: bool = True,
    min_white_noise_db: int = -90,
    max_white_noise_db: int = -46,
    max_trial: int = 100,
):
    """
    Load noise audio from the manifest item, and apply white noise if the loaded noise audio is empty.
    Args:
        sample: a sample from the noise manifest
        sample_rate: target sample rate to resample the noise audio
        max_audio_len: the maximum audio length to load
        pad_to_max: whether to pad the audio to max_audio_len
        min_white_noise_db: the minimum white noise level in dB
        max_white_noise_db: the maximum white noise level in dB
        max_trial: the maximum number of trials to load noise audio before giving up
    Returns:
        noise: the loaded noise audio
        noise_len: the length of the loaded noise audio
    """
    max_dur = None if max_audio_len is None else max_audio_len / sample_rate
    duration = sample.get('duration', None)
    offset = sample.get('offset', 0.0)

    if max_dur is not None and duration is not None and duration > max_dur:
        cnt = 0
        while cnt < max_trial:
            # randomly sample a segment of the noise
            offset = np.random.uniform(0, duration - max_dur)

            audio_segment = AudioSegment.from_file(
                audio_file=sample['audio_filepath'],
                offset=offset,
                duration=max_dur,
                target_sr=sample_rate,
            )

            if sum(audio_segment.samples) > 0:
                # break if the segment is not empty
                break
            cnt += 1
    else:
        audio_segment = AudioSegment.from_file(
            audio_file=sample['audio_filepath'],
            offset=offset,
            duration=duration,
            target_sr=sample_rate,
        )

    if sum(audio_segment.samples) == 0:
        logging.warning(
            f"Loaded noise audio is empty: {sample}, with sampled offset={offset}, duration={max_dur}. Adding white noise."
        )
        WhiteNoisePerturbation(min_level=min_white_noise_db, max_level=max_white_noise_db).perturb(audio_segment)

    noise = torch.tensor(audio_segment.samples, dtype=torch.float)
    noise_len = torch.tensor(noise.size(0)).long()
    # pad to max_audio_len if necessary
    if max_audio_len is not None and pad_to_max:
        if noise.size(0) < max_audio_len:
            pad = (0, max_audio_len - noise.size(0))
            noise = torch.nn.functional.pad(noise, pad)
        else:
            noise = noise[:max_audio_len]
            noise_len = torch.tensor(max_audio_len).long()
    return noise, noise_len


def sample_noise(noise_data: List[Dict], sample_rate: int, max_audio_len: int | None = None, max_trial: int = 20):
    """
    Randomly sample noise audio from the noise manifest.
    Args:
        noise_data: the noise manifest data
        sample_rate: target sample rate to resample the noise audio
        max_audio_len: the maximum audio length to load
        max_trial: the maximum number of trials to load noise audio before giving up
    Returns:
        noise_audio: the sampled noise audio
        noise_len: the length of the sampled noise audio
    """
    cnt = 0
    noise_audio = torch.zeros(max_audio_len).float()
    noise_len = torch.tensor(max_audio_len).long()
    while cnt < max_trial and len(noise_data) > 0:
        try:
            noise_sample = noise_data[np.random.randint(len(noise_data))]
            noise_audio, noise_len = load_noise_audio(noise_sample, sample_rate, max_audio_len)
            break
        except Exception as e:
            logging.warning(f"Error loading noise audio with config {noise_sample} and exception: {e}, retrying.")
            cnt += 1
            if cnt == max_trial:
                logging.warning(f"Failed to load noise audio after {max_trial} attempts, returning zero noise.")
                return torch.zeros(max_audio_len).float(), torch.tensor(max_audio_len).long()
    return noise_audio, noise_len


def pad_audio(audio: torch.Tensor, min_len: int, pad_audio_mode) -> torch.Tensor:
    """
    Pad audio to min_len with the specified mode
    Args:
        audio: the input audio tensor
        min_len: the minimum length to pad to
        pad_audio_mode: the padding mode, either 'repeat' or 'zero'
    Returns:
        audio: the padded audio tensor
    """
    allowed_mode = ['repeat', 'zero']
    if audio.size(0) < min_len:
        if pad_audio_mode == 'repeat' and audio.size(0) > 0:
            num_repeats = int(np.ceil(min_len / audio.size(0)))
            audio = audio.repeat(num_repeats)[:min_len]
        elif pad_audio_mode == 'zero' or audio.size(0) == 0:
            audio = torch.nn.functional.pad(audio, (0, min_len - audio.size(0)))
        else:
            raise ValueError(f"Unsupported pad_audio_mode: {pad_audio_mode}, must be one of {allowed_mode}")
    return audio


class AudioNoiseDataset(audio_to_text.AudioToCharDataset):
    @property
    def output_types(self):
        # disable type checking for since it doesn't support dataclass
        return None

    def __init__(
        self,
        noise_manifest: str | None = None,
        batch_augmentor: Any | None = None,
        min_audio_len_secs: float = 1.0,
        pad_audio_mode: str = 'repeat',
        **kwargs,
    ):
        # add bos_id=0 to avoid empty text token
        super().__init__(bos_id=0, manifest_parse_func=_parse_manifest_item, **kwargs)
        self.noise_manifest = noise_manifest
        self.batch_augmentor = batch_augmentor
        self.noise_data = load_noise_manifest(noise_manifest)
        self.min_audio_len_secs = min_audio_len_secs
        self.pad_audio_mode = pad_audio_mode

    def __getitem__(self, index) -> AudioNoiseItem:
        sample = self.manifest_processor.collection[index]
        offset = sample.offset

        if offset is None:
            offset = 0

        audio = self.featurizer.process(
            sample.audio_file,
            offset=offset,
            duration=sample.duration,
            trim=self.trim,
            orig_sr=sample.orig_sr,
            channel_selector=self.channel_selector,
        )
        if audio.size(0) == 0:
            logging.warning(f"Loaded audio has zero length: {sample}.")

        min_len = int(self.min_audio_len_secs * self.featurizer.sample_rate)
        audio = pad_audio(audio, min_len, self.pad_audio_mode)
        audio_len = torch.tensor(audio.shape[0]).long()
        noise, noise_len = sample_noise(self.noise_data, self.featurizer.sample_rate, audio_len.item())

        item = AudioNoiseItem(
            sample_id=str(index),
            audio=audio,
            audio_len=audio_len,
            noise=noise,
            noise_len=noise_len,
            noisy_audio=audio + noise,
            noisy_audio_len=audio_len,
        )
        return item

    def _collate_fn(self, batch: List[AudioNoiseItem]) -> AudioNoiseBatch:
        return _audio_noise_collate_fn(batch, self.batch_augmentor)


class TarredAudioNoiseDataset(audio_to_text.TarredAudioToCharDataset):
    @property
    def output_types(self):
        # disable type checking for since it doesn't support dataclass
        return None

    def __init__(
        self,
        noise_manifest: str | None = None,
        batch_augmentor: Any | None = None,
        min_audio_len_secs: float = 1.0,
        pad_audio_mode: str = 'repeat',
        **kwargs,
    ):
        """
        Args:
            noise_manifest: the noise manifest file
            batch_augmentor: the batch augmentor
            min_audio_len_secs: the minimum audio length in seconds, audios shorter than this will be padded
            pad_audio_mode: the padding mode for audios shorter than min_audio_len_secs, either 'repeat' or 'zero'
            **kwargs: other arguments for TarredAudioToCharDataset

        """
        super().__init__(bos_id=0, manifest_parse_func=_parse_manifest_item, **kwargs)
        self.noise_manifest = noise_manifest
        self.batch_augmentor = batch_augmentor
        self.noise_data = load_noise_manifest(noise_manifest)
        self.min_audio_len_secs = min_audio_len_secs
        self.pad_audio_mode = pad_audio_mode

    def _build_sample(self, tup):
        """Builds the training sample by combining the data from the WebDataset with the manifest info."""
        audio_bytes, audio_filename, offset_id = tup

        # Grab manifest entry from self.manifest_preprocessor.collection
        file_id, _ = os.path.splitext(os.path.basename(audio_filename))
        manifest_idx = self.manifest_processor.collection.mapping[file_id][offset_id]
        manifest_entry = self.manifest_processor.collection[manifest_idx]

        offset = manifest_entry.offset
        if offset is None:
            offset = 0

        try:
            # Convert audio bytes to IO stream for processing (for SoundFile to read)
            audio_filestream = io.BytesIO(audio_bytes)
            audio = self.featurizer.process(
                audio_filestream,
                offset=offset,
                duration=manifest_entry.duration,
                trim=self.trim,
                orig_sr=manifest_entry.orig_sr,
            )
            audio_filestream.close()
        except Exception as e:
            raise RuntimeError(f"Error reading audio sample: {manifest_entry}, with exception: {e}.")

        min_len = int(self.min_audio_len_secs * self.featurizer.sample_rate)
        audio = pad_audio(audio, min_len, self.pad_audio_mode)
        audio_len = torch.tensor(audio.shape[0]).long()
        noise, noise_len = sample_noise(self.noise_data, self.featurizer.sample_rate, audio_len.item())

        item = AudioNoiseItem(
            sample_id=str(manifest_idx),
            audio=audio,
            audio_len=audio_len,
            noise=noise,
            noise_len=noise_len,
            noisy_audio=audio + noise,
            noisy_audio_len=audio_len,
        )
        return item

    def _pad_audio(self, audio: torch.Tensor) -> torch.Tensor:
        min_len = int(self.min_audio_len_secs * self.featurizer.sample_rate)
        if audio.size(0) < min_len:
            if self.pad_audio_mode == 'repeat':
                num_repeats = int(np.ceil(min_len / audio.size(0)))
                audio = audio.repeat(num_repeats)[:min_len]
            elif self.pad_audio_mode == 'zero':
                audio = torch.nn.functional.pad(audio, (0, min_len - audio.size(0)))
            else:
                raise ValueError(
                    f"Unsupported pad_audio_mode: {self.pad_audio_mode}, must be one of ['repeat', 'zero']"
                )
        return audio

    def _collate_fn(self, batch: List[AudioNoiseItem]) -> AudioNoiseBatch:
        return _audio_noise_collate_fn(batch, self.batch_augmentor)


class LhotseAudioNoiseDataset(torch.utils.data.Dataset):
    def __init__(self, noise_manifest: str | None = None, batch_augmentor_cfg: DictConfig = None):
        super().__init__()

        if batch_augmentor_cfg:
            batch_augmentor = Serialization.from_config_dict(batch_augmentor_cfg)
        else:
            batch_augmentor = None

        self.batch_augmentor = batch_augmentor
        self.noise_data = load_noise_manifest(noise_manifest)
        self.load_audio = AudioSamples(fault_tolerant=True)

    def __getitem__(self, cuts):

        audios, audio_lens, cuts = self.load_audio(cuts)
        sampled_noises = [sample_noise(self.noise_data, cut.sampling_rate, cut.num_samples) for cut in cuts]

        items = [
            AudioNoiseItem(
                sample_id=str(cuts[i].id),
                audio=audios[i],
                audio_len=audio_lens[i],
                noise=sampled_noises[i][0],
                noise_len=sampled_noises[i][1],
                noisy_audio=audios[i] + sampled_noises[i][0],
                noisy_audio_len=audio_lens[i],
            )
            for i in range(len(cuts))
        ]
        return _audio_noise_collate_fn(items, self.batch_augmentor)


def get_audio_noise_dataset(
    config: Dict[str, Any], augmentor: Any = None, batch_augmentor: Any = None
) -> AudioNoiseDataset:
    dataset = AudioNoiseDataset(
        noise_manifest=config.get('noise_manifest', None),
        batch_augmentor=batch_augmentor,
        manifest_filepath=config['manifest_filepath'],
        labels=config.get('labels', None),
        sample_rate=config['sample_rate'],
        int_values=config.get('int_values', False),
        augmentor=augmentor,
        max_duration=config.get('max_duration', None),
        min_duration=config.get('min_duration', None),
        trim=config.get('trim_silence', False),
        channel_selector=config.get('channel_selector', None),
    )
    return dataset


def get_concat_audio_noise_dataset(
    config: Dict[str, Any], global_rank: int, world_size: int, augmentor: Any = None, batch_augmentor: Any = None
) -> ConcatDataset:
    manifest_filepaths = config['manifest_filepath']
    datasets = []

    # needed to support validation Concat Datasets that arrive here as
    # [[dataset1,dataset2]] otherwise ModelPT would interfere
    if len(manifest_filepaths) == 1 and not isinstance(manifest_filepaths[0], str):
        logging.info(f"removing an extra nesting level from {manifest_filepaths}")
        manifest_filepaths = config['manifest_filepath'][0]

    for manifest_filepath in manifest_filepaths:
        conf = copy.deepcopy(config)
        conf['manifest_filepath'] = manifest_filepath

        dataset = get_audio_noise_dataset(config=conf, augmentor=augmentor)
        datasets.append(dataset)

    dataset = ConcatDataset(
        datasets,
        sampling_technique=config.get('concat_sampling_technique', 'temperature'),
        sampling_temperature=config.get('concat_sampling_temperature', 5),
        sampling_scale=config.get('concat_sampling_scale', 1),
        sampling_probabilities=config.get('concat_sampling_probabilities', None),
        shuffle=config.get('concat_shuffle', True),
        seed=config.get('concat_sampling_seed', None),
        global_rank=global_rank,
        world_size=world_size,
    )
    return dataset


def get_tarred_audio_noise_dataset(config, shuffle_n, global_rank, world_size, augmentor, batch_augmentor: Any = None):
    tarred_audio_filepaths = config['tarred_audio_filepaths']
    manifest_filepaths = config['manifest_filepath']
    datasets = []
    tarred_audio_filepaths = audio_to_text_dataset.convert_to_config_list(tarred_audio_filepaths)
    manifest_filepaths = audio_to_text_dataset.convert_to_config_list(manifest_filepaths)

    bucketing_weights = config.get('bucketing_weights', None)  # For upsampling buckets
    if bucketing_weights:
        for idx, weight in enumerate(bucketing_weights):
            if not isinstance(weight, int) or weight <= 0:
                raise ValueError(f"bucket weights must be positive integers")

    if len(manifest_filepaths) != len(tarred_audio_filepaths):
        raise ValueError(
            f"manifest_filepaths (length={len(manifest_filepaths)}) and tarred_audio_filepaths (length={len(tarred_audio_filepaths)}) need to have the same number of buckets."
        )

    for dataset_idx, (tarred_audio_filepath, manifest_filepath) in enumerate(
        zip(tarred_audio_filepaths, manifest_filepaths)
    ):
        if len(tarred_audio_filepath) == 1:
            tarred_audio_filepath = tarred_audio_filepath[0]
        if len(manifest_filepath) == 1:
            manifest_filepath = manifest_filepath[0]

        is_sharded_manifest = True if "_OP_" in manifest_filepath and "_CL_" in manifest_filepath else False
        logging.info(
            f"Loading TarredAudioNoiseDataset from {tarred_audio_filepath} and {manifest_filepath}, shard={is_sharded_manifest}"
        )
        dataset = TarredAudioNoiseDataset(
            noise_manifest=config.get('noise_manifest', None),
            batch_augmentor=batch_augmentor,
            audio_tar_filepaths=tarred_audio_filepath,
            manifest_filepath=manifest_filepath,
            labels=config.get('labels', None),
            sample_rate=config['sample_rate'],
            int_values=config.get('int_values', False),
            augmentor=augmentor,
            shuffle_n=shuffle_n,
            max_duration=config.get('max_duration', None),
            min_duration=config.get('min_duration', None),
            trim=config.get('trim_silence', False),
            shard_strategy=config.get('tarred_shard_strategy', 'scatter'),
            shard_manifests=is_sharded_manifest,
            global_rank=global_rank,
            world_size=world_size,
        )
        if bucketing_weights:
            [datasets.append(dataset) for _ in range(bucketing_weights[dataset_idx])]
        else:
            datasets.append(dataset)

    return audio_to_text_dataset.get_chain_dataset(datasets=datasets, ds_config=config, rank=global_rank)


def get_concat_tarred_audio_noise_dataset(
    config, shuffle_n, global_rank, world_size, augmentor, batch_augmentor: Any = None
):
    tarred_audio_filepaths = config['tarred_audio_filepaths']
    manifest_filepaths = config['manifest_filepath']
    datasets = []
    for dataset_idx, (tarred_audio_filepath, manifest_filepath) in enumerate(
        zip(tarred_audio_filepaths, manifest_filepaths)
    ):
        conf = copy.deepcopy(config)
        conf['manifest_filepath'] = manifest_filepath
        conf['tarred_audio_filepaths'] = tarred_audio_filepath
        dataset = get_tarred_audio_noise_dataset(
            config=conf,
            shuffle_n=shuffle_n,
            global_rank=global_rank,
            world_size=world_size,
            augmentor=augmentor,
            batch_augmentor=batch_augmentor,
        )
        datasets.append(dataset)

    dataset = ConcatDataset(
        datasets,
        sampling_technique=config.get('concat_sampling_technique', 'temperature'),
        sampling_temperature=config.get('concat_sampling_temperature', 5),
        sampling_scale=config.get('concat_sampling_scale', 1),
        sampling_probabilities=config.get('concat_sampling_probabilities', None),
        shuffle=config.get('concat_shuffle', True),
        seed=config.get('concat_sampling_seed', None),
        global_rank=global_rank,
        world_size=world_size,
    )
    return dataset


def get_audio_noise_dataset_from_config(
    config,
    global_rank: int,
    world_size: int,
):
    if 'augmentor' in config:
        augmentor = process_augmentations(config['augmentor'], global_rank=global_rank, world_size=world_size)
    else:
        augmentor = None

    if 'batch_augmentor' in config:
        batch_augmentor = Serialization.from_config_dict(config['batch_augmentor'])
    else:
        batch_augmentor = None

    is_concat = config.get('is_concat', False)
    if is_concat:
        if config.get('concat_sampling_technique', None) is None:
            logging.warning(
                f"Concat dataset requires `concat_sampling_technique` but it was not provided, using round-robin. Config: {config}"
            )
            config['concat_sampling_technique'] = 'round-robin'

        if config['concat_sampling_technique'] == 'random':
            if not 'concat_sampling_probabilities' in config:
                logging.warning(
                    f"Concat dataset requires `concat_sampling_probabilities` list, using uniform weights. Config: {config}"
                )
                with open_dict(config):
                    config['concat_sampling_probabilities'] = [1 / len(config['manifest_filepath'])] * len(
                        config['manifest_filepath']
                    )
            elif not isclose(sum(config['concat_sampling_probabilities']), 1, abs_tol=1e-6):
                raise ValueError(
                    f"`concat_sampling_probabilities` need to sum to 1 with 1e-6 tolerance. Config: {config}"
                )

    shuffle = config['shuffle']
    if config.get('is_tarred', False):
        if ('tarred_audio_filepaths' in config and config['tarred_audio_filepaths'] is None) or (
            'manifest_filepath' in config and config['manifest_filepath'] is None
        ):
            logging.warning(
                "Could not load dataset as `manifest_filepath` was None or "
                f"`tarred_audio_filepaths` is None. Provided config : {config}"
            )
            return None

        shuffle_n = config.get('shuffle_n', 4 * config['batch_size']) if shuffle else 0
        if is_concat:
            dataset = get_concat_tarred_audio_noise_dataset(
                config=config,
                shuffle_n=shuffle_n,
                global_rank=global_rank,
                world_size=world_size,
                augmentor=augmentor,
                batch_augmentor=batch_augmentor,
            )
        else:
            dataset = get_tarred_audio_noise_dataset(
                config=config,
                shuffle_n=shuffle_n,
                global_rank=global_rank,
                world_size=world_size,
                augmentor=augmentor,
                batch_augmentor=batch_augmentor,
            )
    else:
        if 'manifest_filepath' in config and config['manifest_filepath'] is None:
            logging.warning(f"Could not load dataset as `manifest_filepath` was None. Provided config : {config}")
            return None
        if is_concat:
            dataset = get_concat_audio_noise_dataset(
                config=config,
                global_rank=global_rank,
                world_size=world_size,
                augmentor=augmentor,
                batch_augmentor=batch_augmentor,
            )
        else:
            dataset = get_audio_noise_dataset(config=config, augmentor=augmentor, batch_augmentor=batch_augmentor)
    return dataset
