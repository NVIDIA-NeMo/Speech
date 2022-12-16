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
from typing import Optional

from nemo.collections.asr.data import feature_to_label


def get_feature_seq_speakerlabel_dataset(
    feature_loader, config: dict
) -> feature_to_label.FeatureToSeqSpeakerLabelDataset:
    """
    Instantiates a FeatureSeqSpeakerLabelDataset.
    Args:
        config: Config of the FeatureToSeqSpeakerLabelDataset.

    Returns:
        An instance of FeatureToSeqSpeakerLabelDataset.
    """
    dataset = feature_to_label.FeatureToSeqSpeakerLabelDataset(
        manifest_filepath=config['manifest_filepath'], labels=config['labels'], feature_loader=feature_loader,
    )
    return dataset


def get_feature_label_dataset(
    config: dict, augmentor: Optional['AudioAugmentor'] = None
) -> feature_to_label.FeatureToLabelDataset:
    dataset = feature_to_label.FeatureToLabelDataset(
        manifest_filepath=config['manifest_filepath'],
        labels=config['labels'],
        augmentor=augmentor,
        window_length_in_sec=config.get("window_length_in_sec", 0.63),
        shift_length_in_sec=config.get("shift_length_in_sec", 0.01),
    )
    return dataset
