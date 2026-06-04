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

from omegaconf import DictConfig

from nemo.collections.asr.data import audio_to_label
from nemo.collections.asr.data.audio_to_text_dataset import convert_to_config_list, get_chain_dataset
from nemo.collections.asr.parts.preprocessing.perturb import process_augmentations
from nemo.collections.common.data.dataset import ConcatDataset


def get_classification_label_dataset(featurizer, config: dict) -> audio_to_label.AudioToClassificationLabelDataset:
    """
    Instantiates a Classification AudioLabelDataset.

    Args:
        config: Config of the AudioToClassificationLabelDataset.

    Returns:
        An instance of AudioToClassificationLabelDataset.
    """
    dataset = audio_to_label.AudioToClassificationLabelDataset(
        manifest_filepath=config['manifest_filepath'],
        labels=config['labels'],
        featurizer=featurizer,
        max_duration=config.get('max_duration', None),
        min_duration=config.get('min_duration', None),
        trim=config.get('trim_silence', False),
        is_regression_task=config.get('is_regression_task', False),
        cal_labels_occurrence=config.get('cal_labels_occurrence', False),
    )
    return dataset


def get_speech_label_dataset(featurizer, config: dict) -> audio_to_label.AudioToSpeechLabelDataset:
    """
    Instantiates a Speech Label (e.g. VAD, speaker recognition) AudioLabelDataset.

    Args:
        config: Config of the AudioToSpeechLabelDataSet.

    Returns:
        An instance of AudioToSpeechLabelDataset.
    """
    dataset = audio_to_label.AudioToSpeechLabelDataset(
        manifest_filepath=config['manifest_filepath'],
        labels=config['labels'],
        featurizer=featurizer,
        max_duration=config.get('max_duration', None),
        min_duration=config.get('min_duration', None),
        trim=config.get('trim_silence', False),
        window_length_in_sec=config.get('window_length_in_sec', 0.31),
        shift_length_in_sec=config.get('shift_length_in_sec', 0.01),
        normalize_audio=config.get('normalize_audio', False),
        cal_labels_occurrence=config.get('cal_labels_occurrence', False),
    )
    return dataset


def get_tarred_classification_label_dataset(
    featurizer, config: dict, shuffle_n: int, global_rank: int, world_size: int
) -> audio_to_label.TarredAudioToClassificationLabelDataset:
    """
    Instantiates a Classification TarredAudioLabelDataset.

    Args:
        config: Config of the TarredAudioToClassificationLabelDataset.
        shuffle_n: How many samples to look ahead and load to be shuffled.
            See WebDataset documentation for more details.
        global_rank: Global rank of this device.
        world_size: Global world size in the training method.

    Returns:
        An instance of TarredAudioToClassificationLabelDataset.
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
                raise ValueError("bucket weights must be positive integers")

    if len(manifest_filepaths) != len(tarred_audio_filepaths):
        raise ValueError(
            f"manifest_filepaths (length={len(manifest_filepaths)}) and tarred_audio_filepaths (length={len(tarred_audio_filepaths)}) need to have the same number of buckets."
        )

    for dataset_idx, (tarred_audio_filepath, manifest_filepath) in enumerate(
        zip(tarred_audio_filepaths, manifest_filepaths)
    ):
        if len(tarred_audio_filepath) == 1:
            tarred_audio_filepath = tarred_audio_filepath[0]
        dataset = audio_to_label.TarredAudioToClassificationLabelDataset(
            audio_tar_filepaths=tarred_audio_filepath,
            manifest_filepath=manifest_filepath,
            labels=config['labels'],
            featurizer=featurizer,
            shuffle_n=shuffle_n,
            max_duration=config.get('max_duration', None),
            min_duration=config.get('min_duration', None),
            trim=config.get('trim_silence', False),
            shard_strategy=config.get('tarred_shard_strategy', 'scatter'),
            global_rank=global_rank,
            world_size=world_size,
            is_regression_task=config.get('is_regression_task', False),
        )

        if bucketing_weights:
            [datasets.append(dataset) for _ in range(bucketing_weights[dataset_idx])]
        else:
            datasets.append(dataset)

    return get_chain_dataset(datasets=datasets, ds_config=config, rank=global_rank)


def get_concat_tarred_speech_label_dataset(
    featurizer,
    config: dict,
    shuffle_n: int,
    global_rank: int,
    world_size: int,
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
        dataset = get_tarred_speech_label_dataset(
            config=conf,
            featurizer=featurizer,
            shuffle_n=shuffle_n,
            global_rank=global_rank,
            world_size=world_size,
        )
        datasets.append(dataset)

    dataset = ConcatDataset(
        datasets,
        sampling_technique=config.get('concat_sampling_technique', 'temperature'),
        sampling_temperature=config.get('concat_sampling_temperature', 5),
        sampling_probabilities=config.get('concat_sampling_probabilities', None),
        global_rank=global_rank,
        world_size=world_size,
        shuffle=config['shuffle'],
    )
    return dataset


def get_tarred_speech_label_dataset(
    featurizer,
    config: dict,
    shuffle_n: int,
    global_rank: int,
    world_size: int,
) -> audio_to_label.TarredAudioToSpeechLabelDataset:
    """
    InInstantiates a Speech Label (e.g. VAD, speaker recognition) TarredAudioLabelDataset.

    Args:
        config: Config of the TarredAudioToSpeechLabelDataset.
        shuffle_n: How many samples to look ahead and load to be shuffled.
            See WebDataset documentation for more details.
        global_rank: Global rank of this device.
        world_size: Global world size in the training method.

    Returns:
        An instance of TarredAudioToSpeechLabelDataset.
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
                raise ValueError("bucket weights must be positive integers")

    if len(manifest_filepaths) != len(tarred_audio_filepaths):
        raise ValueError(
            f"manifest_filepaths (length={len(manifest_filepaths)}) and tarred_audio_filepaths (length={len(tarred_audio_filepaths)}) need to have the same number of buckets."
        )

    for dataset_idx, (tarred_audio_filepath, manifest_filepath) in enumerate(
        zip(tarred_audio_filepaths, manifest_filepaths)
    ):
        if len(tarred_audio_filepath) == 1:
            tarred_audio_filepath = tarred_audio_filepath[0]
        dataset = audio_to_label.TarredAudioToSpeechLabelDataset(
            audio_tar_filepaths=tarred_audio_filepath,
            manifest_filepath=manifest_filepath,
            labels=config['labels'],
            featurizer=featurizer,
            shuffle_n=shuffle_n,
            max_duration=config.get('max_duration', None),
            min_duration=config.get('min_duration', None),
            trim=config.get('trim_silence', False),
            window_length_in_sec=config.get('window_length_in_sec', 8),
            shift_length_in_sec=config.get('shift_length_in_sec', 0.075),
            normalize_audio=config.get('normalize_audio', False),
            shard_strategy=config.get('tarred_shard_strategy', 'scatter'),
            global_rank=global_rank,
            world_size=world_size,
        )

        if bucketing_weights:
            [datasets.append(dataset) for _ in range(bucketing_weights[dataset_idx])]
        else:
            datasets.append(dataset)

    return get_chain_dataset(datasets=datasets, ds_config=config, rank=global_rank)


def get_audio_multi_label_dataset(cfg: DictConfig) -> audio_to_label.AudioToMultiLabelDataset:
    if "augmentor" in cfg:
        augmentor = process_augmentations(cfg.augmentor)
    else:
        augmentor = None

    dataset = audio_to_label.AudioToMultiLabelDataset(
        manifest_filepath=cfg.get("manifest_filepath"),
        sample_rate=cfg.get("sample_rate"),
        labels=cfg.get("labels", None),
        int_values=cfg.get("int_values", False),
        augmentor=augmentor,
        min_duration=cfg.get("min_duration", None),
        max_duration=cfg.get("max_duration", None),
        trim_silence=cfg.get("trim_silence", False),
        is_regression_task=cfg.get("is_regression_task", False),
        cal_labels_occurrence=cfg.get("cal_labels_occurrence", False),
        delimiter=cfg.get("delimiter", None),
        normalize_audio_db=cfg.get("normalize_audio_db", None),
    )
    return dataset


def get_tarred_audio_multi_label_dataset(
    cfg: DictConfig, shuffle_n: int, global_rank: int, world_size: int
) -> audio_to_label.TarredAudioToMultiLabelDataset:

    if "augmentor" in cfg:
        augmentor = process_augmentations(cfg.augmentor)
    else:
        augmentor = None

    tarred_audio_filepaths = cfg['tarred_audio_filepaths']
    manifest_filepaths = cfg['manifest_filepath']
    datasets = []
    tarred_audio_filepaths = convert_to_config_list(tarred_audio_filepaths)
    manifest_filepaths = convert_to_config_list(manifest_filepaths)

    bucketing_weights = cfg.get('bucketing_weights', None)  # For upsampling buckets
    if bucketing_weights:
        for idx, weight in enumerate(bucketing_weights):
            if not isinstance(weight, int) or weight <= 0:
                raise ValueError("bucket weights must be positive integers")

    if len(manifest_filepaths) != len(tarred_audio_filepaths):
        raise ValueError(
            f"manifest_filepaths (length={len(manifest_filepaths)}) and tarred_audio_filepaths (length={len(tarred_audio_filepaths)}) need to have the same number of buckets."
        )

    for dataset_idx, (tarred_audio_filepath, manifest_filepath) in enumerate(
        zip(tarred_audio_filepaths, manifest_filepaths)
    ):
        if len(tarred_audio_filepath) == 1:
            tarred_audio_filepath = tarred_audio_filepath[0]

        dataset = audio_to_label.TarredAudioToMultiLabelDataset(
            audio_tar_filepaths=tarred_audio_filepath,
            manifest_filepath=manifest_filepath,
            sample_rate=cfg["sample_rate"],
            labels=cfg['labels'],
            shuffle_n=shuffle_n,
            int_values=cfg.get("int_values", False),
            augmentor=augmentor,
            min_duration=cfg.get('min_duration', None),
            max_duration=cfg.get('max_duration', None),
            trim_silence=cfg.get('trim_silence', False),
            is_regression_task=cfg.get('is_regression_task', False),
            delimiter=cfg.get("delimiter", None),
            shard_strategy=cfg.get('tarred_shard_strategy', 'scatter'),
            global_rank=global_rank,
            world_size=world_size,
            normalize_audio_db=cfg.get("normalize_audio_db", None),
        )

        if bucketing_weights:
            [datasets.append(dataset) for _ in range(bucketing_weights[dataset_idx])]
        else:
            datasets.append(dataset)

    return get_chain_dataset(datasets=datasets, ds_config=cfg, rank=global_rank)
