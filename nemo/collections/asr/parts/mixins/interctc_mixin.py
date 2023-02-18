# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
from typing import Dict, List, Tuple

import torch

from nemo.core.classes.mixins import AccessMixin


class InterCTCMixin:
    """Adds utilities for computing interCTC loss from https://arxiv.org/abs/2102.03216.

    To use, make sure encoder accesses ``interctc['capture_layers']``
    property in the AccessMixin and registers ``interctc/layer_output_X`` and
    ``interctc/layer_length_X`` for all layers that we want to get loss from.
    Additionally, specify the following config parameters to set up loss::

        interctc:
            # can use different values
            loss_weights: [0.3]
            apply_at_layers: [8]

    Then call

        * ``self.setup_interctc`` in the init method
        * ``self.add_interctc_losses`` after computing regular loss.
        * ``self.finalize_interctc_metrics(metrics, outputs, prefix="val_")``
          in the `multi_validation_epoch_end` method.
        * ``self.finalize_interctc_metrics(metrics, outputs, prefix="test_")``
          in the `multi_test_epoch_end` method.
    """

    def _process_config_values(self, loss_weights: List[float], apply_at_layers: List[int]):
        self._intermediate_loss_weights = loss_weights
        self._apply_at_layers = apply_at_layers
        self._main_loss_weight = 1.0 - sum(self._intermediate_loss_weights)
        if self._main_loss_weight <= 0.0:
            raise ValueError(
                "Make sure that sum of intermediate loss weights is < 1.0. "
                "Note that we don't do any normalization and assign "
                "remaining weight to the regular model loss. "
                "E.g., if interctc.loss_weights = [0.1, 0.3], regular "
                "loss will have weight of 0.6"
            )
        self._interctc_enabled = len(self._intermediate_loss_weights) > 0

        if len(self._apply_at_layers) != len(self._intermediate_loss_weights):
            raise ValueError('Length of interctc.apply_at_layers has to match interctc.loss_weights')

        # setting up config for AccessMixin that will be checked in encoders to
        # log the layers we need
        AccessMixin.update_access_cfg({'interctc': {'capture_layers': self._apply_at_layers}})

    def setup_interctc(self):
        """Sets up all interctc-specific parameters and checks config consistency."""
        interctc_config = self.cfg.get("interctc")
        if interctc_config is not None:
            # if interctc is in the config, we want to check that it indeed defines
            # the required keys and nothing else - that's automatically done by
            # matching with keyword arguments in self._process_config_values
            self._process_config_values(**interctc_config)
        else:
            self._interctc_enabled = False

    def _verify_setup_was_called(self):
        """Can be used to verify if setup_interctc was called."""
        if not hasattr(self, '_interctc_enabled'):
            raise RuntimeError('self.setup_interctc() has to be called before InterCTC loss can be used!')

    def is_interctc_enabled(self) -> bool:
        """Returns whether interCTC loss is enabled."""
        self._verify_setup_was_called()
        return self._interctc_enabled

    def set_interctc_enabled(self, enabled: bool):
        """Can be used to enable/disable InterCTC manually."""
        self._verify_setup_was_called()
        if enabled:  # checking if proper config parameters were specified
            if len(self._intermediate_loss_weights) == 0:
                raise RuntimeError(
                    'InterCTC cannot be enabled since interctc.loss_weights was not specified in the config.'
                )
            if len(self._apply_at_layers) != len(self._intermediate_loss_weights):
                raise RuntimeError(
                    'InterCTC cannot be enabled, since length of "loss_weights" does not match "apply_at_layers".'
                )
        self._interctc_enabled = enabled

    def finalize_interctc_metrics(self, metrics: Dict, outputs: List[Dict], prefix: str):
        """Finalizes InterCTC WER and loss metrics for logging purposes.

        Should be called inside ``multi_validation_epoch_end`` (with ``prefix="val_"``) or
        ``multi_test_epoch_end`` (with ``prefix="test_"``).

        Note that ``metrics`` argument is going to be updated in-place.
        """
        if self.is_interctc_enabled():
            for layer_idx in self._apply_at_layers:
                loss = torch.stack([x[f"{prefix}inter_ctc_loss_l{layer_idx}"] for x in outputs]).mean()
                wer_num = torch.stack([x[f"{prefix}inter_wer_num_l{layer_idx}"] for x in outputs]).sum()
                wer_denom = torch.stack([x[f"{prefix}inter_wer_denom_l{layer_idx}"] for x in outputs]).sum()
                metrics["log"].update(
                    {
                        f"{prefix}inter_ctc_loss_l{layer_idx}": loss,
                        f"{prefix}inter_wer_l{layer_idx}": wer_num / wer_denom,
                    }
                )
            metrics["log"][f"{prefix}final_ctc_loss"] = torch.stack(
                [x[f"{prefix}final_ctc_loss"] for x in outputs]
            ).mean()

    def get_captured_interctc_tensors(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Returns a list of captured tensors from encoder: tuples of (output, length).

        Will additionally apply ``self.decoder`` to the outputs.
        """
        if not self.is_interctc_enabled():
            return []

        # note that we have a loop here, because tensors can be defined from
        # submodules of encoder (e.g., that's the case in Jasper)
        total_registry = {}
        for module_registry in AccessMixin.get_module_registry(self.encoder).values():
            for key, value in module_registry.items():
                if key.startswith("interctc/") and key in total_registry:
                    raise RuntimeError(f"layer {key} has been logged multiple times!")
            total_registry.update(module_registry)
        # if intermediate_loss_weights was set, the encoder has to register
        # interctc/layer_output_X and interctc/layer_length_X tensors.
        # We need to apply decoder to each of them and compute CTC loss.
        captured_tensors = []
        for layer_idx in self._apply_at_layers:
            try:
                layer_outputs = total_registry[f"interctc/layer_output_{layer_idx}"]
                layer_lengths = total_registry[f"interctc/layer_length_{layer_idx}"]
            except KeyError:
                raise RuntimeError(
                    f"Intermediate layer {layer_idx} was not captured! "
                    "Check if length of model.encoder.captured_layer_outputs matches "
                    "length of model.intermediate_loss_weights properties."
                )
            if len(layer_outputs) > 1 or len(layer_lengths) > 1:
                raise RuntimeError(
                    "Make sure encoder.forward is called exactly one time before interCTC loss is computed."
                )
            captured_tensors.append((self.decoder(encoder_output=layer_outputs[0]), layer_lengths[0]))
        return captured_tensors

    def add_interctc_losses(
        self,
        loss_value: torch.Tensor,
        transcript: torch.Tensor,
        transcript_len: torch.Tensor,
        compute_wer: bool,
        log_wer_num_denom: bool = False,
        log_prefix: str = "",
    ) -> Tuple[torch.Tensor, Dict]:
        """Adding interCTC losses if required.

        Will also register loss/wer metrics in the returned dictionary.

        Args:
            loss_value (torch.Tensor): regular loss tensor (will add interCTC loss to it).
            transcript (torch.Tensor): current utterance transcript.
            transcript_len (torch.Tensor): current utterance transcript length.
            compute_wer (bool): whether to compute WER for the current utterance.
                Should typically be True for validation/test and only True for
                training if current batch WER should be logged.
            log_wer_num_denom (bool): if True, will additionally log WER num/denom
                in the returned metrics dictionary. Should always be True for
                validation/test to allow correct metrics aggregation. Should
                always be False for training. Defaults to False.
            log_prefix (str): prefix added to all log values. Should be ``""`` for
                training and ``"val_"`` for validation.

        Returns:
            tuple[torch.Tensor, Dict]: tuple of new loss tensor and dictionary with logged metrics.
        """
        if not self.is_interctc_enabled() or not AccessMixin.is_access_enabled():
            return loss_value, {}
        metrics = {f"{log_prefix}final_ctc_loss": loss_value}
        captured_tensors = self.get_captured_interctc_tensors()

        for layer_idx, intermediate_result, loss_weight in zip(
            self._apply_at_layers, captured_tensors, self._intermediate_loss_weights
        ):
            inter_loss_value = self.loss(
                log_probs=intermediate_result[0],
                targets=transcript,
                target_lengths=transcript_len,
                input_lengths=intermediate_result[1],
            )
            metrics[f"{log_prefix}inter_ctc_loss_l{layer_idx}"] = loss_value.detach()
            loss_value += inter_loss_value * loss_weight
            if compute_wer:
                self._wer.update(
                    predictions=intermediate_result[0],
                    targets=transcript,
                    target_lengths=transcript_len,
                    predictions_lengths=intermediate_result[1],
                )
                wer, wer_num, wer_denom = self._wer.compute()
                self._wer.reset()
                metrics.update({f'{log_prefix}inter_wer_l{layer_idx}': wer})
                if log_wer_num_denom:
                    metrics.update(
                        {
                            f'{log_prefix}inter_wer_num_l{layer_idx}': wer_num,
                            f'{log_prefix}inter_wer_denom_l{layer_idx}': wer_denom,
                        }
                    )

        # return total loss
        return loss_value, metrics
