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

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from omegaconf import MISSING

import nemo.core.classes.dataset
from nemo.collections.asr.metrics.wer import CTCDecodingConfig
from nemo.collections.asr.modules.audio_preprocessing import (
    AudioToMelSpectrogramPreprocessorConfig,
    SpectrogramAugmentationConfig,
)
from nemo.collections.asr.modules.conv_asr import ConvASRDecoderConfig, ConvASREncoderConfig
from nemo.core.config import modelPT as model_cfg


@dataclass
class ASRDatasetConfig(nemo.core.classes.dataset.DatasetConfig):
    manifest_filepath: Optional[Any] = None
    sample_rate: int = MISSING
    labels: List[str] = MISSING
    trim_silence: bool = False

    # Tarred dataset support
    is_tarred: bool = False
    tarred_audio_filepaths: Optional[Any] = None
    tarred_shard_strategy: str = "scatter"
    shard_manifests: bool = False
    shuffle_n: int = 0

    # Optional
    int_values: Optional[int] = None
    augmentor: Optional[Dict[str, Any]] = None
    max_duration: Optional[float] = None
    min_duration: Optional[float] = None
    max_utts: int = 0
    blank_index: int = -1
    unk_index: int = -1
    normalize: bool = False
    trim: bool = True
    parser: Optional[str] = 'en'
    eos_id: Optional[int] = None
    bos_id: Optional[int] = None
    pad_id: int = 0
    use_start_end_token: bool = False
    return_sample_id: Optional[bool] = False

    # bucketing params
    bucketing_strategy: str = "synced_randomized"
    bucketing_batch_size: Optional[Any] = None
    bucketing_weights: Optional[List[int]] = None


@dataclass
class EncDecCTCConfig(model_cfg.ModelConfig):
    # Model global arguments
    sample_rate: int = 16000
    repeat: int = 1
    dropout: float = 0.0
    separable: bool = False
    labels: List[str] = MISSING

    # Dataset configs
    train_ds: ASRDatasetConfig = ASRDatasetConfig(manifest_filepath=None, shuffle=True)
    validation_ds: ASRDatasetConfig = ASRDatasetConfig(manifest_filepath=None, shuffle=False)
    test_ds: ASRDatasetConfig = ASRDatasetConfig(manifest_filepath=None, shuffle=False)

    # Optimizer / Scheduler config
    optim: Optional[model_cfg.OptimConfig] = model_cfg.OptimConfig(sched=model_cfg.SchedConfig())

    # Model component configs
    preprocessor: AudioToMelSpectrogramPreprocessorConfig = AudioToMelSpectrogramPreprocessorConfig()
    spec_augment: Optional[SpectrogramAugmentationConfig] = SpectrogramAugmentationConfig()
    encoder: ConvASREncoderConfig = ConvASREncoderConfig()
    decoder: ConvASRDecoderConfig = ConvASRDecoderConfig()
    decoding: CTCDecodingConfig = CTCDecodingConfig()


@dataclass
class EncDecCTCModelConfig(model_cfg.NemoConfig):
    model: EncDecCTCConfig = EncDecCTCConfig()


@dataclass
class CacheAwareStreamingConfig:
    chunk_size: int = 0  # the size of each chunk at each step, it can be a list of two integers to specify different chunk sizes for the first step and others
    shift_size: int = 0  # the size of the shift in each step, it can be a list of two integers to specify different shift sizes for the first step and others

    cache_drop_size: int = 0  # the number of steps to drop from the cache
    last_channel_cache_size: int = 0  # the size of the needed cache for last channel layers

    valid_out_len: int = 0  # the number of the steps in the final output which are valid (have the same value as in the offline mode)

    pre_encode_cache_size: int = 0  # the size of the needed cache for the pre-encoding part of the model to avoid caching inside the pre-encoding layers
    drop_extra_pre_encoded: int = 0  # the number of steps to get dropped after the pre-encoding layer

    last_channel_num: int = 0  # number of the last channel layers (like MHA layers) which need caching in the model
    last_time_num: int = 0  # number of the last time layers (like convolutions) which need caching in the model
