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

from typing import Optional

from nemo.collections.asr.data.feature_to_text import FeatureToBPEDataset, FeatureToCharDataset
from nemo.utils import logging


def get_char_dataset(config: dict, augmentor: Optional['FeatureAugmentor'] = None) -> FeatureToCharDataset:
    """
    Instantiates a Character Encoding based FeatureToCharDataset.

    Args:
        config: Config of the FeatureToCharDataset.
        augmentor: Optional AudioAugmentor object for augmentations on audio data.

    Returns:
        An instance of FeatureToCharDataset.
    """
    if 'labels' not in config:
        logging.warning(f"dataset does not have explicitly defined labels")

    dataset = FeatureToCharDataset(
        manifest_filepath=config['manifest_filepath'],
        labels=config.get('labels', None),
        normalize=config.get('normalize', 'post_norm'),
        normalize_type=config.get('normalize_type', 'per_feature'),
        use_rttm=config.get('use_rttm', False),
        rttm_mode=config.get('rttm_mode', 'mask'),
        feat_min_len=config.get('feat_min_len', 4),
        feat_mask_val=config.get('feat_mask_val', None),
        frame_unit_time_secs=config.get('frame_unit_time_secs', 0.01),
        sample_rate=config.get('sample_rate', 16000),
        augmentor=augmentor,
        max_duration=config.get('max_duration', None),
        min_duration=config.get('min_duration', None),
        max_utts=config.get('max_utts', 0),
        blank_index=config.get('blank_index', -1),
        unk_index=config.get('unk_index', -1),
        trim=config.get('trim_silence', False),
        parser=config.get('parser', 'en'),
        return_sample_id=config.get('return_sample_id', False),
        channel_selector=config.get('channel_selector', None),
    )
    return dataset


def get_bpe_dataset(
    config: dict, tokenizer: 'TokenizerSpec', augmentor: Optional['FeatureAugmentor'] = None
) -> FeatureToBPEDataset:
    """
    Instantiates a Byte Pair Encoding / Word Piece Encoding based FeatureoToBPEDataset.

    Args:
        config: Config of the FeatureToBPEDataset.
        tokenizer: An instance of a TokenizerSpec object.
        augmentor: Optional FeatureAugmentor object for augmentations on audio features.

    Returns:
        An instance of FeatureToBPEDataset.
    """
    dataset = FeatureToBPEDataset(
        manifest_filepath=config['manifest_filepath'],
        tokenizer=tokenizer,
        normalize=config.get('normalize', 'post_norm'),
        normalize_type=config.get('normalize_type', 'per_feature'),
        use_rttm=config.get('use_rttm', False),
        rttm_mode=config.get('rttm_mode', 'mask'),
        feat_min_len=config.get('feat_min_len', 4),
        feat_mask_val=config.get('feat_mask_val', None),
        frame_unit_time_secs=config.get('frame_unit_time_secs', 0.01),
        sample_rate=config.get('sample_rate', 16000),
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
