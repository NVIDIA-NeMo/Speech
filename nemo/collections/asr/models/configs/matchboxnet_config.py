# SPDX-FileCopyrightText: Copyright (c) 2020, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

from omegaconf import MISSING

from nemo.collections.asr.models.configs import classification_models_config as clf_cfg
from nemo.collections.asr.modules.audio_preprocessing import (
    AudioToMFCCPreprocessorConfig,
    CropOrPadSpectrogramAugmentationConfig,
    SpectrogramAugmentationConfig,
)
from nemo.collections.asr.modules.conv_asr import (
    ConvASRDecoderClassificationConfig,
    ConvASREncoderConfig,
    JasperEncoderConfig,
)
from nemo.core.config import modelPT as model_cfg


# fmt: off
def matchboxnet_3x1x64():
    config = [
        JasperEncoderConfig(filters=128, repeat=1, kernel=[11], stride=[1], dilation=[1], dropout=0.0,
                            residual=False, groups=1, separable=True, heads=-1, residual_mode='add',
                            residual_dense=False, se=False, se_reduction_ratio=8, se_context_size=-1,
                            se_interpolation_mode='nearest', kernel_size_factor=1.0, stride_last=False),
        JasperEncoderConfig(filters=64, repeat=1, kernel=[13], stride=[1], dilation=[1], dropout=0.0,
                            residual=True, groups=1, separable=True, heads=-1, residual_mode='add',
                            residual_dense=False, se=False, se_reduction_ratio=8, se_context_size=-1,
                            se_interpolation_mode='nearest', kernel_size_factor=1.0, stride_last=False),
        JasperEncoderConfig(filters=64, repeat=1, kernel=[15], stride=[1], dilation=[1], dropout=0.0,
                            residual=True, groups=1, separable=True, heads=-1, residual_mode='add',
                            residual_dense=False, se=False, se_reduction_ratio=8, se_context_size=-1,
                            se_interpolation_mode='nearest', kernel_size_factor=1.0, stride_last=False),
        JasperEncoderConfig(filters=64, repeat=1, kernel=[17], stride=[1], dilation=[1], dropout=0.0,
                            residual=True, groups=1, separable=True, heads=-1, residual_mode='add',
                            residual_dense=False, se=False, se_reduction_ratio=8, se_context_size=-1,
                            se_interpolation_mode='nearest', kernel_size_factor=1.0, stride_last=False),
        JasperEncoderConfig(filters=128, repeat=1, kernel=[29], stride=[1], dilation=[2], dropout=0.0,
                            residual=False, groups=1, separable=True, heads=-1, residual_mode='add',
                            residual_dense=False, se=False, se_reduction_ratio=8, se_context_size=-1,
                            se_interpolation_mode='nearest', kernel_size_factor=1.0, stride_last=False),
        JasperEncoderConfig(filters=128, repeat=1, kernel=[1], stride=[1], dilation=[1], dropout=0.0,
                            residual=False, groups=1, separable=False, heads=-1, residual_mode='add',
                            residual_dense=False, se=False, se_reduction_ratio=8, se_context_size=-1,
                            se_interpolation_mode='nearest', kernel_size_factor=1.0, stride_last=False)
    ]
    return config


def matchboxnet_3x1x64_vad():
    config = [
        JasperEncoderConfig(filters=128, repeat=1, kernel=[11], stride=[1], dilation=[1], dropout=0.0,
                            residual=False, groups=1, separable=True, heads=-1, residual_mode='add',
                            residual_dense=False, se=False, se_reduction_ratio=8, se_context_size=-1,
                            se_interpolation_mode='nearest', kernel_size_factor=1.0, stride_last=False),
        JasperEncoderConfig(filters=64, repeat=1, kernel=[13], stride=[1], dilation=[1], dropout=0.0,
                            residual=True, groups=1, separable=True, heads=-1, residual_mode='add',
                            residual_dense=False, se=False, se_reduction_ratio=8, se_context_size=-1,
                            se_interpolation_mode='nearest', kernel_size_factor=1.0, stride_last=False),
        JasperEncoderConfig(filters=64, repeat=1, kernel=[15], stride=[1], dilation=[1], dropout=0.0,
                            residual=True, groups=1, separable=True, heads=-1, residual_mode='add',
                            residual_dense=False, se=False, se_reduction_ratio=8, se_context_size=-1,
                            se_interpolation_mode='nearest', kernel_size_factor=1.0, stride_last=False),
        JasperEncoderConfig(filters=64, repeat=1, kernel=[17], stride=[1], dilation=[1], dropout=0.0,
                            residual=True, groups=1, separable=True, heads=-1, residual_mode='add',
                            residual_dense=False, se=False, se_reduction_ratio=8, se_context_size=-1,
                            se_interpolation_mode='nearest', kernel_size_factor=1.0, stride_last=False),
        JasperEncoderConfig(filters=128, repeat=1, kernel=[29], stride=[1], dilation=[2], dropout=0.0,
                            residual=False, groups=1, separable=True, heads=-1, residual_mode='add',
                            residual_dense=False, se=False, se_reduction_ratio=8, se_context_size=-1,
                            se_interpolation_mode='nearest', kernel_size_factor=1.0, stride_last=False),
        JasperEncoderConfig(filters=128, repeat=1, kernel=[1], stride=[1], dilation=[1], dropout=0.0,
                            residual=False, groups=1, separable=False, heads=-1, residual_mode='add',
                            residual_dense=False, se=False, se_reduction_ratio=8, se_context_size=-1,
                            se_interpolation_mode='nearest', kernel_size_factor=1.0, stride_last=False)
    ]
    return config


# fmt: on


@dataclass
class MatchboxNetModelConfig(clf_cfg.EncDecClassificationConfig):
    # Model global arguments
    sample_rate: int = 16000
    repeat: int = 1
    dropout: float = 0.0
    separable: bool = True
    kernel_size_factor: float = 1.0
    timesteps: int = 128
    labels: List[str] = MISSING

    # Dataset configs
    train_ds: clf_cfg.EncDecClassificationDatasetConfig = field(
        default_factory=lambda: clf_cfg.EncDecClassificationDatasetConfig(
            manifest_filepath=None, shuffle=True, trim_silence=False
        )
    )
    validation_ds: clf_cfg.EncDecClassificationDatasetConfig = field(
        default_factory=lambda: clf_cfg.EncDecClassificationDatasetConfig(manifest_filepath=None, shuffle=False)
    )
    test_ds: clf_cfg.EncDecClassificationDatasetConfig = field(
        default_factory=lambda: clf_cfg.EncDecClassificationDatasetConfig(manifest_filepath=None, shuffle=False)
    )

    # Optimizer / Scheduler config
    optim: Optional[model_cfg.OptimConfig] = field(
        default_factory=lambda: model_cfg.OptimConfig(sched=model_cfg.SchedConfig())
    )

    # Model general component configs
    preprocessor: AudioToMFCCPreprocessorConfig = field(
        default_factory=lambda: AudioToMFCCPreprocessorConfig(window_size=0.025)
    )
    spec_augment: Optional[SpectrogramAugmentationConfig] = field(
        default_factory=lambda: SpectrogramAugmentationConfig(
            freq_masks=2, time_masks=2, freq_width=15, time_width=25, rect_masks=5, rect_time=25, rect_freq=15
        )
    )
    crop_or_pad_augment: Optional[CropOrPadSpectrogramAugmentationConfig] = field(
        default_factory=lambda: CropOrPadSpectrogramAugmentationConfig(audio_length=128)
    )

    encoder: ConvASREncoderConfig = field(default_factory=lambda: ConvASREncoderConfig(activation="relu"))
    decoder: ConvASRDecoderClassificationConfig = field(default_factory=lambda: ConvASRDecoderClassificationConfig())


@dataclass
class MatchboxNetVADModelConfig(MatchboxNetModelConfig):
    timesteps: int = 64
    labels: List[str] = field(default_factory=lambda: ['background', 'speech'])

    crop_or_pad_augment: Optional[CropOrPadSpectrogramAugmentationConfig] = None


class EncDecClassificationModelConfigBuilder(model_cfg.ModelConfigBuilder):
    VALID_CONFIGS = ['matchboxnet_3x1x64', 'matchboxnet_3x1x64_vad']

    def __init__(self, name: str = 'matchboxnet_3x1x64', encoder_cfg_func: Optional[Callable[[], List[Any]]] = None):
        if name not in EncDecClassificationModelConfigBuilder.VALID_CONFIGS:
            raise ValueError("`name` must be one of : \n" f"{EncDecClassificationModelConfigBuilder.VALID_CONFIGS}")

        self.name = name

        if 'matchboxnet_3x1x64_vad' in name:
            if encoder_cfg_func is None:
                encoder_cfg_func = matchboxnet_3x1x64_vad

            model_cfg = MatchboxNetVADModelConfig(
                repeat=1,
                separable=True,
                encoder=ConvASREncoderConfig(jasper=encoder_cfg_func(), activation="relu"),
                decoder=ConvASRDecoderClassificationConfig(),
            )

        elif 'matchboxnet_3x1x64' in name:
            if encoder_cfg_func is None:
                encoder_cfg_func = matchboxnet_3x1x64

            model_cfg = MatchboxNetModelConfig(
                repeat=1,
                separable=False,
                spec_augment=SpectrogramAugmentationConfig(rect_masks=5, rect_freq=50, rect_time=120),
                encoder=ConvASREncoderConfig(jasper=encoder_cfg_func(), activation="relu"),
                decoder=ConvASRDecoderClassificationConfig(),
            )

        else:
            raise ValueError(f"Invalid config name submitted to {self.__class__.__name__}")

        super(EncDecClassificationModelConfigBuilder, self).__init__(model_cfg)
        self.model_cfg: clf_cfg.EncDecClassificationConfig = model_cfg  # enable type hinting

    def set_labels(self, labels: List[str]):
        self.model_cfg.labels = labels

    def set_separable(self, separable: bool):
        self.model_cfg.separable = separable

    def set_repeat(self, repeat: int):
        self.model_cfg.repeat = repeat

    def set_sample_rate(self, sample_rate: int):
        self.model_cfg.sample_rate = sample_rate

    def set_dropout(self, dropout: float = 0.0):
        self.model_cfg.dropout = dropout

    def set_timesteps(self, timesteps: int):
        self.model_cfg.timesteps = timesteps

    def set_is_regression_task(self, is_regression_task: bool):
        self.model_cfg.is_regression_task = is_regression_task

    # Note: Autocomplete for users wont work without these overrides
    # But practically it is not needed since python will infer at runtime

    # def set_train_ds(self, cfg: Optional[clf_cfg.EncDecClassificationDatasetConfig] = None):
    #     super().set_train_ds(cfg)
    #
    # def set_validation_ds(self, cfg: Optional[clf_cfg.EncDecClassificationDatasetConfig] = None):
    #     super().set_validation_ds(cfg)
    #
    # def set_test_ds(self, cfg: Optional[clf_cfg.EncDecClassificationDatasetConfig] = None):
    #     super().set_test_ds(cfg)

    def _finalize_cfg(self):
        # propagate labels
        self.model_cfg.train_ds.labels = self.model_cfg.labels
        self.model_cfg.validation_ds.labels = self.model_cfg.labels
        self.model_cfg.test_ds.labels = self.model_cfg.labels
        self.model_cfg.decoder.vocabulary = self.model_cfg.labels

        # propagate num classes
        self.model_cfg.decoder.num_classes = len(self.model_cfg.labels)

        # propagate sample rate
        self.model_cfg.sample_rate = self.model_cfg.sample_rate
        self.model_cfg.preprocessor.sample_rate = self.model_cfg.sample_rate
        self.model_cfg.train_ds.sample_rate = self.model_cfg.sample_rate
        self.model_cfg.validation_ds.sample_rate = self.model_cfg.sample_rate
        self.model_cfg.test_ds.sample_rate = self.model_cfg.sample_rate

        # propagate filters
        self.model_cfg.encoder.feat_in = self.model_cfg.preprocessor.features
        self.model_cfg.decoder.feat_in = self.model_cfg.encoder.jasper[-1].filters

        # propagate timeteps
        if self.model_cfg.crop_or_pad_augment is not None:
            self.model_cfg.crop_or_pad_augment.audio_length = self.model_cfg.timesteps

        # propagate separable
        for layer in self.model_cfg.encoder.jasper[:-1]:  # type: JasperEncoderConfig
            layer.separable = self.model_cfg.separable

        # propagate repeat
        for layer in self.model_cfg.encoder.jasper[1:-2]:  # type: JasperEncoderConfig
            layer.repeat = self.model_cfg.repeat

        # propagate dropout
        for layer in self.model_cfg.encoder.jasper:  # type: JasperEncoderConfig
            layer.dropout = self.model_cfg.dropout

    def build(self) -> clf_cfg.EncDecClassificationConfig:
        return super().build()
