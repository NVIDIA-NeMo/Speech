# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

from nemo.collections.asr.data import audio_to_audio


def get_audio_to_target_dataset(config: dict) -> audio_to_audio.AudioToTargetDataset:
    """Instantiates an audio-to-audio dataset.

    Args:
        config: Config of AudioToTargetDataset.

    Returns:
        An instance of AudioToTargetDataset
    """
    dataset = audio_to_audio.AudioToTargetDataset(
        manifest_filepath=config['manifest_filepath'],
        sample_rate=config['sample_rate'],
        input_key=config['input_key'],
        target_key=config['target_key'],
        audio_duration=config.get('audio_duration', None),
        random_offset=config.get('random_offset', False),
        max_duration=config.get('max_duration', None),
        min_duration=config.get('min_duration', None),
        max_utts=config.get('max_utts', 0),
        input_channel_selector=config.get('input_channel_selector', None),
        target_channel_selector=config.get('target_channel_selector', None),
        normalization_signal=config.get('normalization_signal', None),
    )
    return dataset


def get_audio_to_target_with_reference_dataset(config: dict) -> audio_to_audio.AudioToTargetWithReferenceDataset:
    """Instantiates an audio-to-audio dataset.

    Args:
        config: Config of AudioToTargetWithReferenceDataset.

    Returns:
        An instance of AudioToTargetWithReferenceDataset
    """
    dataset = audio_to_audio.AudioToTargetWithReferenceDataset(
        manifest_filepath=config['manifest_filepath'],
        sample_rate=config['sample_rate'],
        input_key=config['input_key'],
        target_key=config['target_key'],
        reference_key=config['reference_key'],
        audio_duration=config.get('audio_duration', None),
        random_offset=config.get('random_offset', False),
        max_duration=config.get('max_duration', None),
        min_duration=config.get('min_duration', None),
        max_utts=config.get('max_utts', 0),
        input_channel_selector=config.get('input_channel_selector', None),
        target_channel_selector=config.get('target_channel_selector', None),
        reference_channel_selector=config.get('reference_channel_selector', None),
        reference_is_synchronized=config.get('reference_is_synchronized', True),
        reference_duration=config.get('reference_duration', None),
        normalization_signal=config.get('normalization_signal', None),
    )
    return dataset


def get_audio_to_target_with_embedding_dataset(config: dict) -> audio_to_audio.AudioToTargetWithEmbeddingDataset:
    """Instantiates an audio-to-audio dataset.

    Args:
        config: Config of AudioToTargetWithEmbeddingDataset.

    Returns:
        An instance of AudioToTargetWithEmbeddingDataset
    """
    dataset = audio_to_audio.AudioToTargetWithEmbeddingDataset(
        manifest_filepath=config['manifest_filepath'],
        sample_rate=config['sample_rate'],
        input_key=config['input_key'],
        target_key=config['target_key'],
        embedding_key=config['embedding_key'],
        audio_duration=config.get('audio_duration', None),
        random_offset=config.get('random_offset', False),
        max_duration=config.get('max_duration', None),
        min_duration=config.get('min_duration', None),
        max_utts=config.get('max_utts', 0),
        input_channel_selector=config.get('input_channel_selector', None),
        target_channel_selector=config.get('target_channel_selector', None),
        normalization_signal=config.get('normalization_signal', None),
    )
    return dataset
