# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
from typing import Optional

import torch
import torch.nn as nn
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from omegaconf import OmegaConf
from transformers import AutoConfig, AutoFeatureExtractor, AutoModelForSpeechSeq2Seq

import nemo.collections.asr as nemo_asr
from nemo.collections.speechlm.utils import get_nested_attr, to_dict_config
from nemo.core.classes.common import Serialization
from nemo.core.classes.module import NeuralModule
from nemo.lightning import io
from nemo.utils import logging, model_utils


class MCoreASRModule(MegatronModule):
    """
    Wrapper class for ASR encoder from `nemo.collections.asr.models.ASRModel`.

    `TransformerConfig` is a dummy config to satisfy the `MegatronModule` constructor.
    `num_attention_heads` is set to 16 such that it's divisible by the value of TP.
    `num_layers` and `hidden_size` are set to 1 since not used.
    """

    def __init__(
        self,
        encoder: NeuralModule,
        preprocessor: Optional[nn.Module] = None,
        spec_augment: Optional[nn.Module] = None,
    ):
        super().__init__(config=TransformerConfig(num_layers=1, hidden_size=1, num_attention_heads=16))
        self.encoder = encoder
        self.preprocessor = preprocessor
        self.spec_augmentation = spec_augment

    def maybe_preprocess_audio(
        self,
        input_signal=None,
        input_signal_length=None,
        processed_signal=None,
        processed_signal_length=None,
    ):
        has_input_signal = input_signal is not None and input_signal_length is not None
        has_processed_signal = processed_signal is not None and processed_signal_length is not None
        if (has_input_signal ^ has_processed_signal) is False:
            raise ValueError(
                f"{self.__class__} Arguments ``input_signal`` and ``input_signal_length`` are mutually exclusive "
                " with ``processed_signal`` and ``processed_signal_len`` arguments."
            )

        if not has_processed_signal:
            processed_signal, processed_signal_length = self.preprocessor(
                input_signal=input_signal,
                length=input_signal_length,
            )
        return processed_signal, processed_signal_length

    def forward(
        self,
        input_signal: Optional[torch.Tensor],
        input_signal_length: Optional[torch.Tensor],
        processed_signal: Optional[torch.Tensor],
        processed_signal_length: Optional[torch.Tensor],
    ):
        processed_signal, processed_signal_length = self.maybe_preprocess_audio(
            input_signal, input_signal_length, processed_signal, processed_signal_length
        )

        # Spec augment is not applied during evaluation/testing
        if self.spec_augmentation is not None and self.training:
            processed_signal = self.spec_augmentation(input_spec=processed_signal, length=processed_signal_length)

        encoded, encoded_len = self.encoder(audio_signal=processed_signal, length=processed_signal_length)
        return encoded, encoded_len


class HFWrappedPreprocessor(nn.Module):
    def __init__(self, preprocessor: AutoFeatureExtractor, sample_rate: int):
        super().__init__()
        self.preprocessor = preprocessor
        self.sample_rate = sample_rate

    def forward(self, input_signal: torch.Tensor, length: torch.Tensor):
        processed = self.preprocessor(
            input_signal.cpu().numpy(), sampling_rate=self.sample_rate, return_tensors="pt"
        )  # type: transformers.feature_extraction_utils.BatchFeature
        processed_signal = processed["input_features"]  # type: torch.Tensor # [batch, hidden, time]
        processed_signal = processed_signal.to(input_signal.device).type_as(input_signal)
        processed_signal_len = torch.tensor(
            [processed_signal.shape[2]] * processed_signal.shape[0], device=length.device, dtype=length.dtype
        )  # [batch]
        return processed_signal, processed_signal_len

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class HFWrappedEncoder(nn.Module):
    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder

    def forward(self, audio_signal: torch.Tensor, length: torch.Tensor):
        # no input length required for models like Whisper
        output = self.encoder(
            audio_signal.type(self.encoder.dtype)
        )  # type: transformers.modeling_outputs.BaseModelOutput
        encoded = output["last_hidden_state"]  # [batch, time, hidden]
        encoded = encoded.transpose(1, 2)  # [batch, hidden, time]
        encoded_len = torch.tensor([encoded.shape[2]] * encoded.shape[0], device=encoded.device).long()  # [batch]
        return encoded, encoded_len

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


@dataclass
class ASRModuleConfig(ModelParallelConfig, io.IOMixin):
    _target_: Optional[str] = None
    pretrained_model: Optional[str] = "nvidia/canary-1b"
    hidden_size: Optional[int] = None
    config: Optional[dict] = None
    preprocessor_config: Optional[dict] = None
    spec_augment_config: Optional[dict] = None
    init_from_pretrained_model: Optional[str] = None
    init_from_nemo_model: Optional[str] = None
    init_from_ptl_ckpt: Optional[str] = None
    target_module: Optional[str] = "encoder"
    sample_rate: Optional[int] = 16000
    use_hf_auto_model: Optional[bool] = False
    hf_trust_remote_code: Optional[bool] = False
    hf_load_pretrained_weights: Optional[bool] = True

    def configure_nemo_asr_model(self):
        if self._target_ is not None:
            imported_cls = model_utils.import_class_by_path(self._target_)
        else:
            imported_cls = nemo_asr.models.ASRModel
        if self.pretrained_model is not None and self.config is None:
            if str(self.pretrained_model).endswith(".nemo"):
                asr_model = imported_cls.restore_from(self.pretrained_model)  # type: nemo_asr.models.ASRModel
            else:
                asr_model = imported_cls.from_pretrained(
                    model_name=self.pretrained_model
                )  # type: nemo_asr.models.ASRModel
            self.config = OmegaConf.to_container(asr_model.cfg, resolve=True)
        else:
            cfg = OmegaConf.create(self.config)
            asr_model = imported_cls(cfg=cfg)  # type: nemo_asr.models.ASRModel
            init_cfg = OmegaConf.create(
                {
                    "init_from_pretrained_model": self.init_from_pretrained_model,
                    "init_from_nemo_model": self.init_from_nemo_model,
                    "init_from_ptl_ckpt": self.init_from_ptl_ckpt,
                }
            )
            asr_model.maybe_init_from_pretrained_checkpoint(init_cfg)

        model = asr_model
        if self.target_module is not None:
            model = get_nested_attr(asr_model, self.target_module)

        if model is None:
            raise ValueError(f"Model {self._target_} does not have attribute {self.target_module}")

        if self.preprocessor_config is not None:
            preprocessor = Serialization.from_config_dict(to_dict_config(self.preprocessor_config))
        elif hasattr(asr_model, "preprocessor"):
            preprocessor = asr_model.preprocessor  # type: nemo_asr.modules.AudioToMelSpectrogramPreprocessor
        else:
            preprocessor = None
            logging.warning(f"Model {self._target_} does not have a preprocessor, use with caution.")

        if self.sample_rate != preprocessor._sample_rate:
            raise ValueError(
                f"Sample rate mismatch: ASRModuleConfig ({self.sample_rate}) != preprocessor ({preprocessor._sample_rate}). "
                "Please provide a preprocessor config with the correct sample rate."
            )
        return model, preprocessor

    def configure_hf_auto_model(self):
        hf_preprocessor = AutoFeatureExtractor.from_pretrained(
            self.pretrained_model, trust_remote_code=self.hf_trust_remote_code
        )
        preprocessor = HFWrappedPreprocessor(hf_preprocessor, self.sample_rate)

        if self.hf_load_pretrained_weights:
            asr_model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.pretrained_model,
                torch_dtype=torch.bfloat16,
                trust_remote_code=self.hf_trust_remote_code,
                use_safetensors=True,
            )
        else:
            config = AutoConfig.from_pretrained(self.pretrained_model, trust_remote_code=self.hf_trust_remote_code)
            asr_model = AutoModelForSpeechSeq2Seq.from_config(config, trust_remote_code=self.hf_trust_remote_code)

        model = asr_model
        if self.target_module is not None:
            model = get_nested_attr(asr_model, self.target_module)

        model = HFWrappedEncoder(model)
        return model, preprocessor

    def configure_model(self):
        if self.use_hf_auto_model:
            model, preprocessor = self.configure_hf_auto_model()
        else:
            model, preprocessor = self.configure_nemo_asr_model()

        # add attribute "tensor_parallel_grad_reduce" to the model for TP grad all-reduce
        model.tensor_parallel_grad_reduce = True

        if self.spec_augment_config is not None:
            spec_augment = Serialization.from_config_dict(to_dict_config(self.spec_augment_config))
        else:
            spec_augment = None

        return MCoreASRModule(encoder=model, preprocessor=preprocessor, spec_augment=spec_augment)
