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
import logging
from abc import ABC, abstractmethod
from typing import List

import torch

from nemo.core.classes import ModelPT
from nemo.core.classes.common import PretrainedModelInfo
from nemo.core.classes.exportable import Exportable
from nemo.core.classes.mixins import AccessMixin
from nemo.core.utils.neural_type_utils import get_io_names
from nemo.utils import logging, model_utils
from nemo.utils.cast_utils import cast_all

__all__ = ['ASRModel']


class ASRModel(ModelPT, ABC):
    @abstractmethod
    def transcribe(self, paths2audio_files: List[str], batch_size: int = 4, verbose: bool = True) -> List[str]:
        """
        Takes paths to audio files and returns text transcription
        Args:
            paths2audio_files: paths to audio fragment to be transcribed
            verbose: (bool) whether to display tqdm progress bar

        Returns:
            transcription texts
        """
        pass

    def multi_validation_epoch_end(self, outputs, dataloader_idx: int = 0):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        wer_num = torch.stack([x['val_wer_num'] for x in outputs]).sum()
        wer_denom = torch.stack([x['val_wer_denom'] for x in outputs]).sum()
        tensorboard_logs = {'val_loss': val_loss_mean, 'val_wer': wer_num / wer_denom}
        return {'val_loss': val_loss_mean, 'log': tensorboard_logs}

    def multi_test_epoch_end(self, outputs, dataloader_idx: int = 0):
        val_loss_mean = torch.stack([x['test_loss'] for x in outputs]).mean()
        wer_num = torch.stack([x['test_wer_num'] for x in outputs]).sum()
        wer_denom = torch.stack([x['test_wer_denom'] for x in outputs]).sum()
        tensorboard_logs = {'test_loss': val_loss_mean, 'test_wer': wer_num / wer_denom}
        return {'test_loss': val_loss_mean, 'log': tensorboard_logs}

    @classmethod
    def list_available_models(cls) -> 'List[PretrainedModelInfo]':
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.
        Returns:
            List of available pre-trained models.
        """
        # recursively walk the subclasses to generate pretrained model info
        list_of_models = model_utils.resolve_subclass_pretrained_model_info(cls)
        return list_of_models

    def add_auxiliary_losses(self, loss: torch.Tensor, reset_registry: bool = False) -> torch.Tensor:
        """
        Utility method to enable calculation of auxiliary losses for ASR training.

        Args:
            loss: The output loss value prior to addition with auxiliary losses.
            reset_registry: Bool, whether to reset the AccessMixin registry after adding auxiliary losses.

        Returns:
            Loss tensor used for back propagation.
        """
        # Add adapter auxiliary losses, if registered
        if AccessMixin.is_access_enabled():
            registry = AccessMixin.get_module_registry(self)
            log_dict = {}

            for loss_key, loss_registry in registry.items():
                # Add auxiliary loss to total loss
                if 'adapter_loss' in loss_registry:
                    loss_list = loss_registry['adapter_loss']
                    loss_value = sum(loss_list)
                    loss += loss_value

                    # Log current loss name and value
                    keys = loss_key.split(".")
                    key = "/".join(keys)
                    key = "adapter_loss/" + key
                    log_dict[key] = loss_value.detach()

            if len(log_dict) > 0:
                self.log_dict(log_dict)

        if reset_registry:
            AccessMixin.reset_registry(self)

        # return total loss
        return loss

    def setup_optimization_flags(self):
        """
        Utility method that must be explicitly called by the subclass in order to support optional optimization flags.
        This method is the only valid place to access self.cfg prior to DDP training occurs.

        The subclass may chose not to support this method, therefore all variables here must be checked via hasattr()
        """
        # Skip update if nan/inf grads appear on any rank.
        self._skip_nan_grad = False
        if "skip_nan_grad" in self._cfg and self._cfg["skip_nan_grad"]:
            self._skip_nan_grad = self._cfg["skip_nan_grad"]

    def on_after_backward(self):
        """
        zero-out the gradients which any of them is NAN or INF
        """
        super().on_after_backward()

        if hasattr(self, '_skip_nan_grad') and self._skip_nan_grad:
            device = next(self.parameters()).device
            valid_gradients = torch.tensor([1], device=device, dtype=torch.float32)

            # valid_gradients = True
            for param_name, param in self.named_parameters():
                if param.grad is not None:
                    is_not_nan_or_inf = not (torch.isnan(param.grad).any() or torch.isinf(param.grad).any())
                    if not is_not_nan_or_inf:
                        valid_gradients = valid_gradients * 0
                        break

            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(valid_gradients, op=torch.distributed.ReduceOp.MIN)

            if valid_gradients < 1:
                logging.warning(f'detected inf or nan values in gradients! Setting gradients to zero.')
                self.zero_grad()


class ExportableEncDecModel(Exportable):
    """
    Simple utiliy mix-in to export models that consist of encoder/decoder pair
    plus pre/post processor, but have to be exported as encoder/decoder pair only
    (covers most ASR classes)
    """

    @property
    def input_module(self):
        return self.encoder

    @property
    def output_module(self):
        return self.decoder

    @property
    def output_names(self):
        otypes = self.output_module.output_types
        if getattr(self.input_module, 'export_cache_support', False):
            in_types = self.input_module.output_types
            otypes = {n: t for (n, t) in list(otypes.items())[:1]}
            for (n, t) in list(in_types.items())[1:]:
                otypes[n] = t
        return get_io_names(otypes, self.disabled_deployment_output_names)

    def forward_for_export(
        self, input, length=None, cache_last_channel=None, cache_last_time=None, cache_last_channel_len=None
    ):
        """
        This forward is used when we need to export the model to ONNX format.
        Inputs cache_last_channel and cache_last_time are needed to be passed for exporting streaming models.
        Args:
            input: Tensor that represents a batch of raw audio signals,
                of shape [B, T]. T here represents timesteps.
            length: Vector of length B, that contains the individual lengths of the audio sequences.
            cache_last_channel: Tensor of shape [N, B, T, H] which contains the cache for last channel layers
            cache_last_time: Tensor of shape [N, B, H, T] which contains the cache for last time layers
                N is the number of such layers which need caching, B is batch size, H is the hidden size of activations,
                and T is the length of the cache

        Returns:
            the output of the model
        """
        enc_fun = getattr(self.input_module, 'forward_for_export', self.input_module.forward)
        if cache_last_channel is None:
            encoder_output = enc_fun(audio_signal=input, length=length)
            if isinstance(encoder_output, tuple):
                encoder_output = encoder_output[0]
        else:
            encoder_output, length, cache_last_channel, cache_last_time, cache_last_channel_len = enc_fun(
                audio_signal=input,
                length=length,
                cache_last_channel=cache_last_channel,
                cache_last_time=cache_last_time,
                cache_last_channel_len=cache_last_channel_len,
            )

        dec_fun = getattr(self.output_module, 'forward_for_export', self.output_module.forward)
        ret = dec_fun(encoder_output=encoder_output)
        if isinstance(ret, tuple):
            ret = ret[0]
        if cache_last_channel is not None:
            ret = (ret, length, cache_last_channel, cache_last_time, cache_last_channel_len)
        return cast_all(ret, from_dtype=torch.float16, to_dtype=torch.float32)

    @property
    def disabled_deployment_input_names(self):
        return self.encoder.disabled_deployment_input_names

    @property
    def disabled_deployment_output_names(self):
        return self.encoder.disabled_deployment_output_names

    def set_export_config(self, args):
        if 'cache_support' in args:
            enable = bool(args['cache_support'])
            self.encoder.export_cache_support = enable
            logging.info(f"Caching support enabled: {enable}")
            self.encoder.setup_streaming_params()
        super().set_export_config(args)
