# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

from nemo.collections.asr.parts.preprocessing.feature_loader import ExternalFeatureLoader
from nemo.collections.asr.parts.preprocessing.features import FeaturizerFactory, FilterbankFeatures, WaveformFeaturizer
from nemo.collections.asr.parts.preprocessing.perturb import (
    AudioAugmentor,
    AugmentationDataset,
    GainPerturbation,
    ImpulsePerturbation,
    NoisePerturbation,
    NoisePerturbationWithNormalization,
    Perturbation,
    RirAndNoisePerturbation,
    ShiftPerturbation,
    SilencePerturbation,
    SpeedPerturbation,
    TimeStretchPerturbation,
    TranscodePerturbation,
    WhiteNoisePerturbation,
    perturbation_types,
    process_augmentations,
    register_perturbation,
)
from nemo.collections.asr.parts.preprocessing.segment import AudioSegment
