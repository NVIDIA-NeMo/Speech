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
from abc import ABC
from typing import List, Optional

import torch

from nemo.collections.common.parts.optional_cuda_graphs import WithOptionalCudaGraphs
from nemo.core.classes import ModelPT
from nemo.core.classes.common import PretrainedModelInfo
from nemo.core.classes.exportable import Exportable
from nemo.core.classes.mixins import AccessMixin
from nemo.core.neural_types import AudioSignal, LabelsType, LengthsType, NeuralType
from nemo.core.utils.neural_type_utils import get_io_names
from nemo.utils import logging, model_utils
from nemo.utils.cast_utils import cast_all

__all__ = ['ASRModel']


class ASRModel(ModelPT, WithOptionalCudaGraphs, ABC):
    def multi_validation_epoch_end(self, outputs, dataloader_idx: int = 0):
        val_loss = {}
        tensorboard_logs = {}

        if 'val_loss' in outputs[0]:
            val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
            val_loss = {'val_loss': val_loss_mean}

            tensorboard_logs.update(val_loss)

        if "val_wer_num" in outputs[0]:
            wer_num = torch.stack([x['val_wer_num'] for x in outputs]).sum()
            wer_denom = torch.stack([x['val_wer_denom'] for x in outputs]).sum()
            val_wer = {'val_wer': wer_num / wer_denom}

            tensorboard_logs.update(val_wer)

        if "val_bleu_num" in outputs[0]:
            bleu_pred_len = torch.stack([x["val_bleu_pred_len"] for x in outputs]).sum()
            bleu_target_len = torch.stack([x["val_bleu_target_len"] for x in outputs]).sum()
            bleu_num = torch.stack([x["val_bleu_num"] for x in outputs]).sum(dim=0)
            bleu_denom = torch.stack([x["val_bleu_denom"] for x in outputs]).sum(dim=0)
            val_bleu = {"val_bleu": self.bleu._compute_bleu(bleu_pred_len, bleu_target_len, bleu_num, bleu_denom)}

            tensorboard_logs.update(val_bleu)

        return {**val_loss, 'log': tensorboard_logs}

    def multi_test_epoch_end(self, outputs, dataloader_idx: int = 0):
        val_loss = {}
        tensorboard_logs = {}

        if 'test_loss' in outputs[0]:
            val_loss_mean = torch.stack([x['test_loss'] for x in outputs]).mean()
            val_loss = {'test_loss': val_loss_mean}

            tensorboard_logs.update(val_loss)

        if "test_wer_num" in outputs[0]:
            wer_num = torch.stack([x['test_wer_num'] for x in outputs]).sum()
            wer_denom = torch.stack([x['test_wer_denom'] for x in outputs]).sum()
            val_wer = {'test_wer': wer_num / wer_denom}

            tensorboard_logs.update(val_wer)

        if "test_bleu_num" in outputs[0]:
            bleu_pred_len = torch.stack([x["test_bleu_pred_len"] for x in outputs]).sum()
            bleu_target_len = torch.stack([x["test_bleu_target_len"] for x in outputs]).sum()
            bleu_num = torch.stack([x["test_bleu_num"] for x in outputs]).sum(dim=0)
            bleu_denom = torch.stack([x["test_bleu_denom"] for x in outputs]).sum(dim=0)
            val_bleu = {"test_bleu": self.bleu._compute_bleu(bleu_pred_len, bleu_target_len, bleu_num, bleu_denom)}

            tensorboard_logs.update(val_bleu)

        return {**val_loss, 'log': tensorboard_logs}

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
        if AccessMixin.is_access_enabled(self.model_guid):
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
                logging.warning('detected inf or nan values in gradients! Setting gradients to zero.')
                self.zero_grad()

    def maybe_enable_cuda_graphs(self, force_reinit=False) -> bool:
        """Enable CUDA graphs if all conditions met, return True if state changed"""
        state_changed = False
        if force_reinit:
            state_changed = self.disable_cuda_graphs()
        # check that self.decoding.decoding exists and is instance of WithOptionalCudaGraphs
        if isinstance(getattr(getattr(self, "decoding", None), "decoding", None), WithOptionalCudaGraphs):
            state_changed |= self.decoding.decoding.maybe_enable_cuda_graphs()
        if state_changed:
            logging.info(
                f"CUDA graphs enabled for {type(self).__name__}::{type(self.decoding).__name__}::"
                f"{type(self.decoding.decoding).__name__}"
            )
        return state_changed

    def disable_cuda_graphs(self) -> bool:
        """Disable (maybe temporary) CUDA graphs, return True if state changed"""
        state_changed = False
        # check that self.decoding.decoding exists and is instance of WithOptionalCudaGraphs
        if isinstance(getattr(getattr(self, "decoding", None), "decoding", None), WithOptionalCudaGraphs):
            state_changed = self.decoding.decoding.disable_cuda_graphs()
        if state_changed:
            logging.info(
                f"CUDA graphs disabled for {type(self).__name__}::{type(self.decoding).__name__}::"
                f"{type(self.decoding.decoding).__name__}"
            )
        return state_changed

    def on_train_epoch_start(self) -> None:
        """
        Decoder with CUDA graphs does not release memory, thus we disable it for training epoch.
        EncDecRNNTModel.decoding.decoding is the inference class with CUDA graphs
        """
        self.disable_cuda_graphs()

    def on_train_epoch_end(self) -> None:
        """
        After training, we can enable the decoder with CUDA graphs.
        EncDecRNNTModel.decoding.decoding is the inference class with CUDA graphs
        """
        self.maybe_enable_cuda_graphs()

    def on_validation_epoch_start(self) -> None:
        """
        For validation, we enable CUDA graphs to speedup validation.
        Force re-enabling required due to issues with trainer and mixed precision.
        EncDecRNNTModel.decoding.decoding is the inference class with CUDA graphs.
        """
        self.maybe_enable_cuda_graphs(force_reinit=True)

    def on_validation_epoch_end(self) -> Optional[dict[str, dict[str, torch.Tensor]]]:
        """
        After validation, we disable CUDA graphs, since `validation` can be called in training loop, and
        training will continue after validation
        EncDecRNNTModel.decoding.decoding is the inference class with CUDA graphs.
        """
        self.disable_cuda_graphs()
        return super().on_validation_epoch_end(sync_metrics=True)

    def on_test_epoch_start(self) -> None:
        """
        For testing, we enable CUDA graphs to speedup validation.
        Force re-enabling required due to issues with trainer and mixed precision.
        We do not need to disable CUDA graphs after testing, since `test` cannot be called in training loop.
        EncDecRNNTModel.decoding.decoding is the inference class with CUDA graphs.
        """
        self.maybe_enable_cuda_graphs(force_reinit=True)

    def on_predict_epoch_start(self) -> None:
        """
        For predicting, we enable CUDA graphs to speedup validation.
        Force re-enabling required due to issues with trainer and mixed precision.
        We do not need to disable CUDA graphs after predicting, since `predict` cannot be called in training loop.
        EncDecRNNTModel.decoding.decoding is the inference class with CUDA graphs
        """
        self.maybe_enable_cuda_graphs(force_reinit=True)

    @property
    def oomptimizer_schema(self) -> dict:
        """
        Return a typing schema for optimal batch size calibration for various
        sequence lengths using OOMptimizer.
        """
        return {
            "cls": tuple,
            "inputs": [
                {"type": NeuralType(("B", "T"), AudioSignal()), "seq_length": "input"},
                {"type": NeuralType(("B",), LengthsType()), "seq_length": "input"},
                {
                    "type": NeuralType(("B", "T"), LabelsType()),
                    "seq_length": "output",
                    "vocab_size": self.tokenizer.vocab_size,
                },
                {"type": NeuralType(("B",), LengthsType()), "seq_length": "output"},
            ],
        }


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
            for n, t in list(in_types.items())[1:]:
                otypes[n] = t
        return get_io_names(otypes, self.disabled_deployment_output_names)

    def forward_for_export(
        self, audio_signal, length=None, cache_last_channel=None, cache_last_time=None, cache_last_channel_len=None
    ):
        """
        This forward is used when we need to export the model to ONNX format.
        Inputs cache_last_channel and cache_last_time are needed to be passed for exporting streaming models.

        Args:
            input: Tensor that represents a batch of raw audio signals of shape [B, T]. T here represents timesteps.
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
            encoder_output = enc_fun(audio_signal=audio_signal, length=length)
            if isinstance(encoder_output, tuple):
                encoder_output = encoder_output[0]
        else:
            encoder_output, length, cache_last_channel, cache_last_time, cache_last_channel_len = enc_fun(
                audio_signal=audio_signal,
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
