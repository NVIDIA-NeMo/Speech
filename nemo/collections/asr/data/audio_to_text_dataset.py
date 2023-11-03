# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
import json
import random
from math import isclose
from typing import Any, List, Optional, Union

import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from omegaconf.listconfig import ListConfig
from pytorch_lightning.callbacks import BasePredictionWriter
from torch.utils.data import ChainDataset

from nemo.collections.asr.data import audio_to_text, audio_to_text_dali
from nemo.collections.asr.data.huggingface.hf_audio_to_text_dataset import (
    get_hf_audio_to_text_bpe_dataset,
    get_hf_audio_to_text_char_dataset,
)
from nemo.collections.asr.parts.preprocessing.perturb import process_augmentations
from nemo.collections.common.data.dataset import CodeSwitchedDataset, ConcatDataset
from nemo.utils import logging


def inject_dataloader_value_from_model_config(model_cfg: dict, dataloader_cfg: DictConfig, key: str):
    """
    Extracts the label set provided at the top level of the model, and propagates it to the dataloader
    config.

    Args:
        model_cfg: A DictConfig representing the model's config.
        dataloader_cfg: A DictConfig representing the individual data loader
        key: A str value representing a key in the model_cfg whose value will be propagated to the
            dataloader config.
    """
    if key not in model_cfg:
        logging.info(
            f"Model level config does not contain `{key}`, please explicitly provide `{key}` to the dataloaders."
        )
        return

    if not isinstance(dataloader_cfg, DictConfig):
        dataloader_cfg = DictConfig(dataloader_cfg)

    # If key exists in the data loader config (either set explicitly or as a placeholder (via None))
    if key in dataloader_cfg:
        # Dataloader `labels` is provided and is non-null
        if dataloader_cfg[key] is not None and model_cfg[key] != dataloader_cfg[key]:
            # Model level `labels` dont match Dataloader level `labels`
            logging.warning(
                f'`{key}` is explicitly provided to the data loader, and is different from '
                f'the `{key}` provided at the model level config.\n'
                f'If this is incorrect, please set the dataloader\'s `{key}` to None.'
            )

        else:
            # Dataloader `key` is None or values match
            # Propagate from model level `key` (even if they match)
            with open_dict(dataloader_cfg):
                dataloader_cfg[key] = model_cfg[key]

    else:
        # If key key doesnt even exist in dataloader_cfg, inject it explicitly
        with open_dict(dataloader_cfg):
            dataloader_cfg[key] = model_cfg[key]


def get_concat_char_dataset(
    config: dict, global_rank: int, world_size: int, augmentor: Optional['AudioAugmentor'] = None
) -> ConcatDataset:
    """
    Instantiates an instance of ConcatDataset containing one or more intances of
    Character Encoding based AudioToCharDataset.

    Args:
        config: Config of the AudioToCharDataset.
        global_rank: Global rank of this device.
        world_size: Global world size in the training method.
        augmentor: Optional AudioAugmentor object for augmentations on audio data.

    Returns:
        An instance of ConcatDataset containing one or more instances of AudioToCharDataset.
    """
    if 'labels' not in config:
        logging.warning(f"dataset does not have explicitly defined labels")

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

        dataset = get_char_dataset(config=conf, augmentor=augmentor)
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


def get_char_dataset(config: dict, augmentor: Optional['AudioAugmentor'] = None) -> audio_to_text.AudioToCharDataset:
    """
    Instantiates a Character Encoding based AudioToCharDataset.

    Args:
        config: Config of the AudioToCharDataset.
        augmentor: Optional AudioAugmentor object for augmentations on audio data.

    Returns:
        An instance of AudioToCharDataset.
    """
    if 'labels' not in config:
        logging.warning(f"dataset does not have explicitly defined labels")

    dataset = audio_to_text.AudioToCharDataset(
        manifest_filepath=config['manifest_filepath'],
        labels=config.get('labels', None),
        sample_rate=config['sample_rate'],
        int_values=config.get('int_values', False),
        augmentor=augmentor,
        max_duration=config.get('max_duration', None),
        min_duration=config.get('min_duration', None),
        max_utts=config.get('max_utts', 0),
        blank_index=config.get('blank_index', -1),
        unk_index=config.get('unk_index', -1),
        normalize=config.get('normalize_transcripts', False),
        trim=config.get('trim_silence', False),
        parser=config.get('parser', 'en'),
        return_sample_id=config.get('return_sample_id', False),
        channel_selector=config.get('channel_selector', None),
    )
    return dataset


def get_concat_bpe_dataset(
    config: dict,
    tokenizer: 'TokenizerSpec',
    global_rank: int,
    world_size: int,
    augmentor: Optional['AudioAugmentor'] = None,
) -> ConcatDataset:
    """
    Instantiates a ContactDataset based on several Byte Pair Encoding / Word Piece Encoding based AudioToBPEDatasets.

    Args:
        config: Config of the AudioToBPEDataset.
        tokenizer: An instance of a TokenizerSpec object.
        global_rank: Global rank of this device.
        world_size: Global world size in the training method.
        augmentor: Optional AudioAugmentor object for augmentations on audio data.

    Returns:
        An instance of ConcatDataset containing several instances of AudioToBPEDataset.
    """
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
        dataset = get_bpe_dataset(config=conf, tokenizer=tokenizer, augmentor=augmentor)
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


def get_bpe_dataset(
    config: dict, tokenizer: 'TokenizerSpec', augmentor: Optional['AudioAugmentor'] = None
) -> audio_to_text.AudioToBPEDataset:
    """
    Instantiates a Byte Pair Encoding / Word Piece Encoding based AudioToBPEDataset.

    Args:
        config: Config of the AudioToBPEDataset.
        tokenizer: An instance of a TokenizerSpec object.
        augmentor: Optional AudioAugmentor object for augmentations on audio data.

    Returns:
        An instance of AudioToBPEDataset.
    """
    dataset = audio_to_text.AudioToBPEDataset(
        manifest_filepath=config['manifest_filepath'],
        tokenizer=tokenizer,
        sample_rate=config['sample_rate'],
        int_values=config.get('int_values', False),
        augmentor=augmentor,
        max_duration=config.get('max_duration', None),
        min_duration=config.get('min_duration', None),
        max_utts=config.get('max_utts', 0),
        trim=config.get('trim_silence', False),
        use_start_end_token=config.get('use_start_end_token', True),
        return_sample_id=config.get('return_sample_id', False),
        channel_selector=config.get('channel_selector', None),
    )
    return dataset


def get_concat_tarred_dataset(
    config: dict,
    shuffle_n: int,
    global_rank: int,
    world_size: int,
    tokenizer: Optional['TokenizerSpec'] = None,
    augmentor: Optional['AudioAugmentor'] = None,
) -> ConcatDataset:
    """
    Instantiates a ConcatDataset containing multiple Word Piece/BPE Encoding based TarredAudioToBPEDataset or a char based TarredAudioToCharDataset.

    Args:
        config: Config of the TarredAudioToBPEDataset or TarredAudioToCharDataset.
        shuffle_n: How many samples to look ahead and load to be shuffled.
            See WebDataset documentation for more details.
        tokenizer: An instance of a TokenizerSpec object if BPE dataset is needed.
        global_rank: Global rank of this device.
        world_size: Global world size in the training method.
            Passsing None would return a char-based dataset.
        augmentor: Optional AudioAugmentor object for augmentations on audio data.

    Returns:
        An instance of ConcatDataset containing one or more TarredAudioToBPEDatasets or TarredAudioToCharDatasets.
    """

    tarred_audio_filepaths = config['tarred_audio_filepaths']
    manifest_filepaths = config['manifest_filepath']
    datasets = []
    for dataset_idx, (tarred_audio_filepath, manifest_filepath) in enumerate(
        zip(tarred_audio_filepaths, manifest_filepaths)
    ):
        conf = copy.deepcopy(config)
        conf['manifest_filepath'] = manifest_filepath
        conf['tarred_audio_filepaths'] = tarred_audio_filepath
        dataset = get_tarred_dataset(
            config=conf,
            tokenizer=tokenizer,
            shuffle_n=shuffle_n,
            global_rank=global_rank,
            world_size=world_size,
            augmentor=augmentor,
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


def get_tarred_dataset(
    config: dict,
    shuffle_n: int,
    global_rank: int,
    world_size: int,
    tokenizer: Optional['TokenizerSpec'] = None,
    augmentor: Optional['AudioAugmentor'] = None,
) -> Union[audio_to_text.TarredAudioToBPEDataset, audio_to_text.TarredAudioToCharDataset]:
    """
    Instantiates a Word Piece/BPE Encoding based TarredAudioToBPEDataset or a char based TarredAudioToCharDataset.

    Args:
        config: Config of the TarredAudioToBPEDataset or TarredAudioToCharDataset.
        shuffle_n: How many samples to look ahead and load to be shuffled.
            See WebDataset documentation for more details.
        tokenizer: An instance of a TokenizerSpec object if BPE dataset is needed.
        global_rank: Global rank of this device.
        world_size: Global world size in the training method.
            Passsing None would return a char-based dataset.
        augmentor: Optional AudioAugmentor object for augmentations on audio data.

    Returns:
        An instance of TarredAudioToBPEDataset or TarredAudioToCharDataset.
    """
    tarred_audio_filepaths = config['tarred_audio_filepaths']
    manifest_filepaths = config['manifest_filepath']
    datasets = []
    tarred_audio_filepaths = convert_to_config_list(tarred_audio_filepaths)
    manifest_filepaths = convert_to_config_list(manifest_filepaths)

    bucketing_weights = config.get('bucketing_weights', None)  # For upsampling buckets
    if bucketing_weights:
        for idx, weight in enumerate(bucketing_weights):
            if not isinstance(weight, int) or weight <= 0:
                raise ValueError(f"bucket weights must be positive integers")

    if len(manifest_filepaths) != len(tarred_audio_filepaths):
        raise ValueError(
            f"manifest_filepaths (length={len(manifest_filepaths)}) and tarred_audio_filepaths (length={len(tarred_audio_filepaths)}) need to have the same number of buckets."
        )

    if 'labels' not in config:
        logging.warning(f"dataset does not have explicitly defined labels")

    if 'max_utts' in config:
        raise ValueError('"max_utts" parameter is not supported for tarred datasets')

    for dataset_idx, (tarred_audio_filepath, manifest_filepath) in enumerate(
        zip(tarred_audio_filepaths, manifest_filepaths)
    ):
        if len(tarred_audio_filepath) == 1:
            tarred_audio_filepath = tarred_audio_filepath[0]
        if len(manifest_filepath) == 1:
            manifest_filepath = manifest_filepath[0]

        if tokenizer is None:
            dataset = audio_to_text.TarredAudioToCharDataset(
                audio_tar_filepaths=tarred_audio_filepath,
                manifest_filepath=manifest_filepath,
                labels=config.get('labels', None),
                sample_rate=config['sample_rate'],
                int_values=config.get('int_values', False),
                augmentor=augmentor,
                shuffle_n=shuffle_n,
                max_duration=config.get('max_duration', None),
                min_duration=config.get('min_duration', None),
                blank_index=config.get('blank_index', -1),
                unk_index=config.get('unk_index', -1),
                normalize=config.get('normalize_transcripts', False),
                trim=config.get('trim_silence', False),
                parser=config.get('parser', 'en'),
                shard_strategy=config.get('tarred_shard_strategy', 'scatter'),
                shard_manifests=config.get('shard_manifests', False),
                global_rank=global_rank,
                world_size=world_size,
                return_sample_id=config.get('return_sample_id', False),
            )
        else:
            dataset = audio_to_text.TarredAudioToBPEDataset(
                audio_tar_filepaths=tarred_audio_filepath,
                manifest_filepath=manifest_filepath,
                tokenizer=tokenizer,
                sample_rate=config['sample_rate'],
                int_values=config.get('int_values', False),
                augmentor=augmentor,
                shuffle_n=shuffle_n,
                max_duration=config.get('max_duration', None),
                min_duration=config.get('min_duration', None),
                trim=config.get('trim_silence', False),
                use_start_end_token=config.get('use_start_end_token', True),
                shard_strategy=config.get('tarred_shard_strategy', 'scatter'),
                shard_manifests=config.get('shard_manifests', False),
                global_rank=global_rank,
                world_size=world_size,
                return_sample_id=config.get('return_sample_id', False),
            )
        if bucketing_weights:
            [datasets.append(dataset) for _ in range(bucketing_weights[dataset_idx])]
        else:
            datasets.append(dataset)

    return get_chain_dataset(datasets=datasets, ds_config=config, rank=global_rank)


def get_code_switched_dataset(
    config: dict,
    shuffle_n: int,
    global_rank: int,
    world_size: int,
    tokenizer: Optional['TokenizerSpec'] = None,
    augmentor: Optional['AudioAugmentor'] = None,
) -> CodeSwitchedDataset:

    if 'manifest_filepath' not in config:
        raise ValueError("`manifest_filepath` must be provided in the dataset config if `is_code_switched=True`")
    if 'code_switched' not in config:
        raise ValueError("`code_switched` param group must be in the dataset config if `is_code_switched=True`")

    manifest_filepaths = config['manifest_filepath']
    tarred_audio_filepaths = config.get('tarred_audio_filepaths', None)

    cs_config = OmegaConf.to_container(config['code_switched'])

    # needed to support validation Datasets that arrive here as
    # [[dataset1,dataset2]] otherwise ModelPT would interfere
    if len(manifest_filepaths) == 1 and not isinstance(manifest_filepaths[0], str):
        manifest_filepaths = config['manifest_filepath'][0]
    if tarred_audio_filepaths is None:
        tarred_audio_filepaths = [None] * len(manifest_filepaths)

    if len(manifest_filepaths) != len(tarred_audio_filepaths):
        raise ValueError(
            f"manifest_filepaths (length={len(manifest_filepaths)}) and tarred_audio_filepaths (length={len(tarred_audio_filepaths)}) need to have the same number of items."
        )

    datasets = []
    for dataset_idx, (tarred_audio_filepath, manifest_filepath) in enumerate(
        zip(tarred_audio_filepaths, manifest_filepaths)
    ):
        conf = copy.deepcopy(config)
        conf['manifest_filepath'] = manifest_filepath
        with open_dict(conf):
            conf['tarred_audio_filepaths'] = tarred_audio_filepath
        if tarred_audio_filepath is None or len(tarred_audio_filepath) == 0:
            if tokenizer is None:
                dataset = get_char_dataset(config=conf, augmentor=None)
            else:
                dataset = get_bpe_dataset(config=conf, tokenizer=tokenizer, augmentor=None)
        else:
            dataset = get_tarred_dataset(
                config=conf,
                tokenizer=tokenizer,
                shuffle_n=shuffle_n,
                global_rank=global_rank,
                world_size=world_size,
                augmentor=None,
            )
        datasets.append(dataset)

    config = OmegaConf.to_container(config)

    dataset = CodeSwitchedDataset(
        datasets,
        shuffle=cs_config.get('shuffle', True),
        min_duration=cs_config.get('min_duration', 4),
        max_duration=cs_config.get('max_duration', 20),
        min_monolingual=cs_config.get('min_monolingual', 0.3),
        lang_probs=cs_config.get('probs', None),
        db_norm=cs_config.get('db_norm', -25.0),
        pause_start=cs_config.get('pause_start', 0),
        pause_join=cs_config.get('pause_join', 0),
        pause_end=cs_config.get('pause_end', 0),
        sampling_scales=cs_config.get('sampling_scales', None),
        seed=cs_config.get('seed', None),
        global_rank=global_rank,
        world_size=world_size,
        pure_random=cs_config.get('pure_random', False),
        force_monochannel=cs_config.get('force_monochannel', True),
        infinity_mode=cs_config.get('infinity_mode', False),
        sample_rate=config['sample_rate'],
        augmentor=augmentor,
    )

    return dataset


def get_dali_char_dataset(
    config: dict,
    shuffle: bool,
    device_id: int,
    global_rank: int,
    world_size: int,
    preprocessor_cfg: Optional[DictConfig] = None,
) -> audio_to_text_dali.AudioToCharDALIDataset:
    """
    Instantiates a Character Encoding based AudioToCharDALIDataset.

    Args:
        config: Config of the AudioToCharDALIDataset.
        shuffle: Bool flag whether to shuffle the dataset.
        device_id: Index of the GPU to be used (local_rank). Only applicable when device == 'gpu'. Defaults to 0.
        global_rank: Global rank of this device.
        world_size: Global world size in the training method.
        augmentor: Optional AudioAugmentor object for augmentations on audio data.
        preprocessor_cfg: Preprocessor configuration. Supports AudioToMelSpectrogramPreprocessor and AudioToMFCCPreprocessor.

    Returns:
        An instance of AudioToCharDALIDataset.
    """
    device = 'gpu' if torch.cuda.is_available() else 'cpu'
    dataset = audio_to_text_dali.AudioToCharDALIDataset(
        manifest_filepath=config['manifest_filepath'],
        device=device,
        batch_size=config['batch_size'],
        labels=config['labels'],
        sample_rate=config['sample_rate'],
        audio_tar_filepaths=config.get('tarred_audio_filepaths', None),
        audio_tar_index_filepaths=config.get('tarred_audio_index_filepaths', None),
        max_duration=config.get('max_duration', None),
        min_duration=config.get('min_duration', None),
        blank_index=config.get('blank_index', -1),
        unk_index=config.get('unk_index', -1),
        normalize=config.get('normalize_transcripts', False),
        trim=config.get('trim_silence', False),
        parser=config.get('parser', 'en'),
        shuffle=shuffle,
        shard_strategy=config.get('tarred_shard_strategy', 'scatter'),
        device_id=device_id,
        global_rank=global_rank,
        world_size=world_size,
        preprocessor_cfg=preprocessor_cfg,
        return_sample_id=config.get('return_sample_id', False),
    )
    return dataset


def get_dali_bpe_dataset(
    config: dict,
    tokenizer,
    shuffle: bool,
    device_id: int,
    global_rank: int,
    world_size: int,
    preprocessor_cfg: Optional[DictConfig] = None,
) -> audio_to_text_dali.AudioToCharDALIDataset:
    """
    Instantiates a Subword Encoding based AudioToBPEDALIDataset.

    Args:
        config: Config of the AudioToBPEDALIDataset.
        tokenizer: An implementation of NeMo TokenizerSpec.
        shuffle: Bool flag whether to shuffle the dataset.
        device_id: Index of the GPU to be used (local_rank). Only applicable when device == 'gpu'. Defaults to 0.
        global_rank: Global rank of this device.
        world_size: Global world size in the training method.
        preprocessor_cfg: Preprocessor configuration. Supports AudioToMelSpectrogramPreprocessor and AudioToMFCCPreprocessor.

    Returns:
        An instance of AudioToCharDALIDataset.
    """
    device = 'gpu' if torch.cuda.is_available() else 'cpu'
    dataset = audio_to_text_dali.AudioToBPEDALIDataset(
        manifest_filepath=config['manifest_filepath'],
        tokenizer=tokenizer,
        device=device,
        batch_size=config['batch_size'],
        sample_rate=config['sample_rate'],
        audio_tar_filepaths=config.get('tarred_audio_filepaths', None),
        audio_tar_index_filepaths=config.get('tarred_audio_index_filepaths', None),
        max_duration=config.get('max_duration', None),
        min_duration=config.get('min_duration', None),
        trim=config.get('trim_silence', False),
        use_start_end_token=config.get('use_start_end_token', True),
        shuffle=shuffle,
        shard_strategy=config.get('tarred_shard_strategy', 'scatter'),
        device_id=device_id,
        global_rank=global_rank,
        world_size=world_size,
        preprocessor_cfg=preprocessor_cfg,
        return_sample_id=config.get('return_sample_id', False),
    )
    return dataset


def get_audio_to_text_char_dataset_from_config(
    config, local_rank: int, global_rank: int, world_size: int, preprocessor_cfg: Optional[DictConfig] = None
):
    """
    Construct Audio-To-Text Char dataset from a config.
    Args:
        config: dataset config
        local_rank: model local rank
        global_rank: model global rand
        world_size: world size
        preprocessor_cfg: preprocessor config, for DALI dataset

    Returns:
        constructed dataset or None if dataset config is invalid or nothing to load
    """
    if 'augmentor' in config:
        augmentor = process_augmentations(config['augmentor'], global_rank=global_rank, world_size=world_size)
    else:
        augmentor = None

    if 'hf_data_cfg' in config:
        return get_hf_audio_to_text_char_dataset(
            config=config, global_rank=global_rank, world_size=world_size, augmentor=augmentor
        )

    is_concat = config.get('is_concat', False)
    if is_concat:
        if 'concat_sampling_technique' in config and config['concat_sampling_technique'] is None:
            logging.warning(
                f"Concat dataset requires `concat_sampling_technique` but it was not provided. Config: {config}"
            )
            return None
        if config['concat_sampling_technique'] == 'random':
            if not 'concat_sampling_probabilities' in config:
                logging.warning(f"Concat dataset requires `concat_sampling_probabilities` list. Config: {config}")
                return None
            else:
                if not isclose(sum(config['concat_sampling_probabilities']), 1, abs_tol=1e-6):
                    logging.warning(f"`concat_sampling_probabilities` need to sum to 1. Config: {config}")
                    return None

    shuffle = config['shuffle']
    device = 'gpu' if torch.cuda.is_available() else 'cpu'
    if config.get('use_dali', False):
        device_id = local_rank if device == 'gpu' else None
        dataset = get_dali_char_dataset(
            config=config,
            shuffle=shuffle,
            device_id=device_id,
            global_rank=global_rank,
            world_size=world_size,
            preprocessor_cfg=preprocessor_cfg,
        )
        return dataset

    # Instantiate a code-switched dataset if config is present
    if config.get('is_code_switched', False):
        if 'manifest_filepath' in config and config['manifest_filepath'] is None:
            logging.warning(f"Could not load dataset as `manifest_filepath` was None. Provided config : {config}")
            return None
        if not ('code_switched' in config and config['code_switched'] is not None):
            logging.warning(
                f"Code switched dataset requires `*_ds.code_switched.*` dict but it was not provided. Config: {config}"
            )
            return None
        if (
            ('probs' in config['code_switched'])
            and (config['code_switched']['probs'] is not None)
            and (not isclose(sum(config['code_switched']['probs']), 1, abs_tol=1e-6))
        ):
            logging.warning(f"`.code_switched.probs` need to sum to 1. Config: {config['code_switched']}")
            return None

        shuffle_n = config.get('shuffle_n', 4 * config['batch_size']) if shuffle else 0
        dataset = get_code_switched_dataset(
            config=config,
            shuffle_n=shuffle_n,
            global_rank=global_rank,
            world_size=world_size,
            tokenizer=None,
            augmentor=augmentor,
        )
    # Instantiate tarred dataset loader or normal dataset loader
    elif config.get('is_tarred', False):
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
            dataset = get_concat_tarred_dataset(
                config=config,
                shuffle_n=shuffle_n,
                global_rank=global_rank,
                world_size=world_size,
                augmentor=augmentor,
            )
        else:
            dataset = get_tarred_dataset(
                config=config,
                shuffle_n=shuffle_n,
                global_rank=global_rank,
                world_size=world_size,
                augmentor=augmentor,
            )
    else:
        if 'manifest_filepath' in config and config['manifest_filepath'] is None:
            logging.warning(f"Could not load dataset as `manifest_filepath` was None. Provided config : {config}")
            return None
        if is_concat:
            dataset = get_concat_char_dataset(
                config=config, global_rank=global_rank, world_size=world_size, augmentor=augmentor
            )
        else:
            dataset = get_char_dataset(config=config, augmentor=augmentor)
    return dataset


def get_audio_to_text_bpe_dataset_from_config(
    config,
    local_rank: int,
    global_rank: int,
    world_size: int,
    tokenizer,
    preprocessor_cfg: Optional[DictConfig] = None,
):
    """
    Construct Audio-To-Text BPE dataset from a config.
    Args:
        config: BPE dataset config
        local_rank: model local rank
        global_rank: model global rand
        world_size: world size
        tokenizer: BPE tokenizer
        preprocessor_cfg: preprocessor config, for DALI BPE dataset

    Returns:
        constructed dataset or None if dataset config is invalid or nothing to load
    """
    if 'augmentor' in config:
        augmentor = process_augmentations(config['augmentor'], global_rank=global_rank, world_size=world_size)
    else:
        augmentor = None

    if 'hf_data_cfg' in config:
        return get_hf_audio_to_text_bpe_dataset(
            config=config, global_rank=global_rank, world_size=world_size, tokenizer=tokenizer, augmentor=augmentor
        )

    is_concat = config.get('is_concat', False)
    if is_concat:
        if 'concat_sampling_technique' in config and config['concat_sampling_technique'] is None:
            logging.warning(
                f"Concat dataset requires `concat_sampling_technique` but it was not provided. Config: {config}"
            )
            return None

        if config['concat_sampling_technique'] == 'random':
            if not 'concat_sampling_probabilities' in config:
                logging.warning(f"Concat dataset requires `concat_sampling_probabilities` list. Config: {config}")
                return None
            else:
                if not isclose(sum(config['concat_sampling_probabilities']), 1, abs_tol=1e-6):
                    logging.warning(f"`concat_sampling_probabilities` need to sum to 1. Config: {config}")
                    return None

    shuffle = config['shuffle']
    device = 'gpu' if torch.cuda.is_available() else 'cpu'
    if config.get('use_dali', False):
        device_id = local_rank if device == 'gpu' else None
        dataset = get_dali_bpe_dataset(
            config=config,
            tokenizer=tokenizer,
            shuffle=shuffle,
            device_id=device_id,
            global_rank=global_rank,
            world_size=world_size,
            preprocessor_cfg=preprocessor_cfg,
        )
        return dataset

    # Instantiate a code-switched dataset if config is present
    if config.get('is_code_switched', False):
        if 'manifest_filepath' in config and config['manifest_filepath'] is None:
            logging.warning(f"Could not load dataset as `manifest_filepath` was None. Provided config : {config}")
            return None
        if not ('code_switched' in config and config['code_switched'] is not None):
            logging.warning(
                f"Code switched dataset requires `*_ds.code_switched.*` dict but it was not provided. Config: {config}"
            )
            return None
        if (
            ('probs' in config['code_switched'])
            and (config['code_switched']['probs'] is not None)
            and (not isclose(sum(config['code_switched']['probs']), 1, abs_tol=1e-6))
        ):
            logging.warning(f"`.code_switched.probs` need to sum to 1. Config: {config['code_switched']}")
            return None

        shuffle_n = config.get('shuffle_n', 4 * config['batch_size']) if shuffle else 0
        dataset = get_code_switched_dataset(
            config=config,
            shuffle_n=shuffle_n,
            global_rank=global_rank,
            world_size=world_size,
            tokenizer=tokenizer,
            augmentor=augmentor,
        )
    # Instantiate tarred dataset loader or normal dataset loader
    elif config.get('is_tarred', False):
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
            dataset = get_concat_tarred_dataset(
                config=config,
                tokenizer=tokenizer,
                shuffle_n=shuffle_n,
                global_rank=global_rank,
                world_size=world_size,
                augmentor=augmentor,
            )
        else:
            dataset = get_tarred_dataset(
                config=config,
                tokenizer=tokenizer,
                shuffle_n=shuffle_n,
                global_rank=global_rank,
                world_size=world_size,
                augmentor=augmentor,
            )
    else:
        if 'manifest_filepath' in config and config['manifest_filepath'] is None:
            logging.warning(f"Could not load dataset as `manifest_filepath` was None. Provided config : {config}")
            return None
        if is_concat:
            dataset = get_concat_bpe_dataset(
                config=config,
                global_rank=global_rank,
                world_size=world_size,
                tokenizer=tokenizer,
                augmentor=augmentor,
            )
        else:
            dataset = get_bpe_dataset(config=config, tokenizer=tokenizer, augmentor=augmentor)
    return dataset


class ASRPredictionWriter(BasePredictionWriter):
    def __init__(self, dataset, output_file: str):
        super().__init__(write_interval="batch")
        self.outf = open(output_file, 'w', encoding='utf-8')
        self.dataset = dataset
        self.samples_num = 0

    def write_on_batch_end(
        self,
        trainer,
        pl_module: 'LightningModule',
        prediction: Any,
        batch_indices: List[int],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ):
        for sample_id, transcribed_text in prediction:
            item = {}
            sample = self.dataset.get_manifest_sample(sample_id)
            item["audio_filepath"] = sample.audio_file
            item["offset"] = sample.offset
            item["duration"] = sample.duration
            item["text"] = sample.text_raw
            item["pred_text"] = transcribed_text
            self.outf.write(json.dumps(item) + "\n")
            self.samples_num += 1
        return

    def close_output_file(self):
        self.outf.close()
        return self.samples_num


def convert_to_config_list(initial_list):
    if type(initial_list) is str:
        initial_list = initial_list.split(",")
    if initial_list is None or initial_list == []:
        raise ValueError("manifest_filepaths and tarred_audio_filepaths must not be empty.")
    if not isinstance(initial_list, ListConfig):
        initial_list = ListConfig([initial_list])

    for list_idx, list_val in enumerate(initial_list):
        if type(list_val) != type(initial_list[0]):
            raise ValueError(
                "manifest_filepaths and tarred_audio_filepaths need to be a list of lists for bucketing or just a list of strings"
            )
    if type(initial_list[0]) is not ListConfig:
        initial_list = ListConfig([initial_list])
    return initial_list


def get_chain_dataset(datasets, ds_config, rank=0):
    if len(datasets) > 1:
        if ds_config.get('bucketing_batch_size', None) is not None:
            bucketing_batch_sizes = calc_bucketing_batch_sizes(ds_config, len(datasets))
            logging.info(
                f"Batch bucketing is enabled for {len(datasets)} buckets with adaptive batch sizes of {bucketing_batch_sizes}!"
            )
            for idx, dataset in enumerate(datasets):
                datasets[idx] = audio_to_text.BucketingDataset(
                    dataset=dataset, bucketing_batch_size=bucketing_batch_sizes[idx]
                )
        else:
            logging.info(
                f"Batch bucketing is enabled for {len(datasets)} buckets with fixed batch size of {ds_config['batch_size']}!"
            )

    if len(datasets) == 1:
        return datasets[0]
    bucketing_strategy = ds_config.get('bucketing_strategy', 'synced_randomized')
    if bucketing_strategy == 'fixed_order':
        return ChainDataset(datasets)
    elif bucketing_strategy == 'synced_randomized':
        return audio_to_text.RandomizedChainDataset(datasets=datasets, rnd_seed=0)
    elif bucketing_strategy == 'fully_randomized':
        return audio_to_text.RandomizedChainDataset(datasets=datasets, rnd_seed=random.randint(0, 30000) + rank)
    else:
        raise ValueError(
            f'bucketing_strategy={bucketing_strategy} is not supported! Supported strategies are [fixed_order, fully_randomized, synced_randomized].'
        )


def calc_bucketing_batch_sizes(ds_config, datasets_len):
    bucketing_batch_size = ds_config['bucketing_batch_size']
    bucketing_weights = ds_config.get('bucketing_weights', None)  # To adjust for upsampled buckets

    bucketing_batch_sizes = []

    if ds_config['batch_size'] != 1:
        raise ValueError(
            f"batch_size should be set to one when bucketing_batch_size is set and adaptive bucketing is enabled (batch_size={ds_config['batch_size']}!"
        )
    if type(bucketing_batch_size) == int:  # linear scaling
        if bucketing_weights:  # Want same batchsize for the same duplicated bucket
            for idx, weight in enumerate(bucketing_weights):
                scale_factor = datasets_len - idx
                [bucketing_batch_sizes.append(scale_factor * bucketing_batch_size) for _ in range(weight)]
        else:
            for idx in range(datasets_len):
                scale_factor = datasets_len - idx
                bucketing_batch_sizes.append(scale_factor * bucketing_batch_size)
    elif isinstance(bucketing_batch_size, ListConfig) or isinstance(
        bucketing_batch_size, list
    ):  # assigned bucket sizes
        if bucketing_weights:  # Want same batchsize for same duplicated bucket
            for idx, weight in enumerate(bucketing_weights):
                [bucketing_batch_sizes.append(bucketing_batch_size[idx]) for _ in range(weight)]
        else:
            bucketing_batch_sizes = bucketing_batch_size
    else:
        raise ValueError(
            f"bucketing_batch_size should be an integer or a list (bucketing_batch_size={bucketing_batch_size})!"
        )

    if len(bucketing_batch_sizes) != datasets_len:
        raise ValueError(
            f"batch_size should have the same length as the number of buckets ({len(bucketing_batch_sizes)}!={datasets_len}) "
        )
    return bucketing_batch_sizes
