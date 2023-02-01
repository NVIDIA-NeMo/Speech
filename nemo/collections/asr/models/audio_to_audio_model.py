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

from abc import ABC, abstractmethod
from typing import List, Union

import hydra
import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer

from nemo.collections.asr.metrics.audio import AudioMetricWrapper
from nemo.core.classes import ModelPT
from nemo.utils import logging, model_utils

__all__ = ['AudioToAudioModel']


class AudioToAudioModel(ModelPT, ABC):
    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        super().__init__(cfg=cfg, trainer=trainer)

        self._setup_loss()

    def _setup_loss(self):
        """Setup loss for this model.
        """
        self.loss = AudioToAudioModel.from_config_dict(self._cfg.loss)

    def _get_num_dataloaders(self, tag: str = 'val'):
        if tag == 'val':
            num_dataloaders = len(self._validation_dl) if isinstance(self._validation_dl, List) else 1
        elif tag == 'test':
            num_dataloaders = len(self._test_dl) if isinstance(self._test_dl, List) else 1
        else:
            raise ValueError(f'Unexpected tag {tag}.')

        return num_dataloaders

    def _setup_metrics(self, tag: str = 'val'):
        """Setup metrics for this model for all available dataloaders.

        When using multiple DataLoaders, it is recommended to initialize separate modular
        metric instances for each DataLoader and use them separately.

        Reference:
            - https://torchmetrics.readthedocs.io/en/stable/pages/lightning.html#common-pitfalls
        """
        # Number of currently configured dataloaders
        num_dataloaders = self._get_num_dataloaders(tag)
        logging.debug('Found %d dataloaders for %s', num_dataloaders, tag)

        if hasattr(self, 'metrics'):
            if tag in self.metrics and len(self.metrics[tag]) == num_dataloaders:
                # Exact number of metrics have already been configured, nothing else to do
                logging.debug('Found %d metrics for tag %s, not necesary to initialize again', num_dataloaders, tag)
                return

        if 'metrics' not in self._cfg or tag not in self._cfg['metrics']:
            # Metrics are not available in the configuration, nothing to do
            logging.debug('No metrics configured for %s in model.metrics.%s', tag, tag)
            return

        metrics_cfg = self._cfg['metrics'][tag]

        if 'loss' in metrics_cfg:
            raise ValueError(
                f'Loss is automatically included in the metrics, it should not be specified in model.metrics.{tag}.'
            )

        # Initialize metrics
        if not hasattr(self, 'metrics'):
            self.metrics = torch.nn.ModuleDict()

        # Setup metrics for each dataloader
        self.metrics[tag] = torch.nn.ModuleList()
        for dataloader_idx in range(num_dataloaders):
            metrics_dataloader_idx = torch.nn.ModuleDict(
                {
                    name: AudioMetricWrapper(
                        metric=hydra.utils.instantiate(cfg),
                        channel=cfg.get('channel'),
                        metric_using_batch_averaging=cfg.get('metric_using_batch_averaging'),
                    )
                    for name, cfg in metrics_cfg.items()
                }
            )
            self.metrics[tag].append(metrics_dataloader_idx.to(self.device))

            logging.info(
                'Setup metrics for %s, dataloader %d: %s', tag, dataloader_idx, ', '.join(metrics_dataloader_idx)
            )

    @abstractmethod
    def evaluation_step(self, batch, batch_idx, dataloader_idx: int = 0, tag: str = 'val'):
        pass

    def on_validation_start(self):
        self._setup_metrics('val')
        return super().on_validation_start()

    def on_test_start(self):
        self._setup_metrics('test')
        return super().on_test_start()

    def validation_step(self, batch, batch_idx, dataloader_idx: int = 0):
        return self.evaluation_step(batch, batch_idx, dataloader_idx, 'val')

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self.evaluation_step(batch, batch_idx, dataloader_idx, 'test')

    def multi_evaluation_epoch_end(self, outputs, dataloader_idx: int = 0, tag: str = 'val'):
        # Handle loss
        loss_mean = torch.stack([x[f'{tag}_loss'] for x in outputs]).mean()
        output_dict = {f'{tag}_loss': loss_mean}
        tensorboard_logs = {f'{tag}_loss': loss_mean}

        # Handle metrics for this tag and dataloader_idx
        if hasattr(self, 'metrics') and tag in self.metrics:
            for name, metric in self.metrics[tag][dataloader_idx].items():
                # Compute & reset the metric
                value = metric.compute()
                metric.reset()
                # Store for logs
                tensorboard_logs[f'{tag}_{name}'] = value

        output_dict['log'] = tensorboard_logs

        return output_dict

    def multi_validation_epoch_end(self, outputs, dataloader_idx: int = 0):
        return self.multi_evaluation_epoch_end(outputs, dataloader_idx, 'val')

    def multi_test_epoch_end(self, outputs, dataloader_idx: int = 0):
        return self.multi_evaluation_epoch_end(outputs, dataloader_idx, 'test')

    @abstractmethod
    def process(
        self, paths2audio_files: List[str], output_dir: str, batch_size: int = 4
    ) -> List[Union[str, List[str]]]:
        """
        Takes paths to audio files and returns a list of paths to processed
        audios.

        Args:
            paths2audio_files: paths to audio files to be processed
            output_dir: directory to save processed files
            batch_size: batch size for inference

        Returns:
            Paths to processed audio signals.
        """
        pass

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
