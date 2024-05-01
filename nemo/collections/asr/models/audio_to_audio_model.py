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

import json
import os
import tempfile
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union

import hydra
import librosa
import soundfile as sf
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from tqdm import tqdm

from nemo.collections.asr.data import audio_to_audio_dataset
from nemo.collections.asr.data.audio_to_audio_lhotse import LhotseAudioToTargetDataset
from nemo.collections.asr.data.audio_to_text_dataset import inject_dataloader_value_from_model_config
from nemo.collections.asr.metrics.audio import AudioMetricWrapper
from nemo.collections.asr.parts.utils.audio_utils import ChannelSelectorType
from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config
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

        if self.cfg.get('metrics') is None:
            # Metrics are not available in the configuration, nothing to do
            logging.debug('No metrics configured in model.metrics')
            return

        if (metrics_cfg := self.cfg['metrics'].get(tag)) is None:
            # Metrics configuration is not available in the configuration, nothing to do
            logging.debug('No metrics configured for %s in model.metrics', tag)
            return

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
            metrics_dataloader_idx = {}
            for name, cfg in metrics_cfg.items():
                logging.debug('Initialize %s for dataloader_idx %s', name, dataloader_idx)
                cfg_dict = OmegaConf.to_container(cfg)
                cfg_channel = cfg_dict.pop('channel', None)
                cfg_batch_averaging = cfg_dict.pop('metric_using_batch_averaging', None)
                metrics_dataloader_idx[name] = AudioMetricWrapper(
                    metric=hydra.utils.instantiate(cfg_dict),
                    channel=cfg_channel,
                    metric_using_batch_averaging=cfg_batch_averaging,
                )

            metrics_dataloader_idx = torch.nn.ModuleDict(metrics_dataloader_idx)
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
        output_dict = self.evaluation_step(batch, batch_idx, dataloader_idx, 'val')
        if isinstance(self.trainer.val_dataloaders, (list, tuple)) and len(self.trainer.val_dataloaders) > 1:
            self.validation_step_outputs[dataloader_idx].append(output_dict)
        else:
            self.validation_step_outputs.append(output_dict)
        return output_dict

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        output_dict = self.evaluation_step(batch, batch_idx, dataloader_idx, 'test')
        if isinstance(self.trainer.test_dataloaders, (list, tuple)) and len(self.trainer.test_dataloaders) > 1:
            self.test_step_outputs[dataloader_idx].append(output_dict)
        else:
            self.test_step_outputs.append(output_dict)
        return output_dict

    def multi_evaluation_epoch_end(self, outputs, dataloader_idx: int = 0, tag: str = 'val'):
        # Handle loss
        loss_mean = torch.stack([x[f'{tag}_loss'] for x in outputs]).mean()
        tensorboard_logs = {f'{tag}_loss': loss_mean}

        # Handle metrics for this tag and dataloader_idx
        if hasattr(self, 'metrics') and tag in self.metrics:
            for name, metric in self.metrics[tag][dataloader_idx].items():
                # Compute & reset the metric
                value = metric.compute()
                metric.reset()
                # Store for logs
                tensorboard_logs[f'{tag}_{name}'] = value

        return {f'{tag}_loss': loss_mean, 'log': tensorboard_logs}

    def multi_validation_epoch_end(self, outputs, dataloader_idx: int = 0):
        return self.multi_evaluation_epoch_end(outputs, dataloader_idx, 'val')

    def multi_test_epoch_end(self, outputs, dataloader_idx: int = 0):
        return self.multi_evaluation_epoch_end(outputs, dataloader_idx, 'test')

    @torch.no_grad()
    def process(
        self,
        paths2audio_files: List[str],
        output_dir: str,
        batch_size: int = 1,
        num_workers: Optional[int] = None,
        input_channel_selector: Optional[ChannelSelectorType] = None,
    ) -> List[str]:
        """
        Process audio files provided in paths2audio_files.
        Processed signals will be saved in output_dir.

        Args:
            paths2audio_files: (a list) of paths to audio files. \
                Recommended length per file is between 5 and 25 seconds. \
                But it is possible to pass a few hours long file if enough GPU memory is available.
            output_dir: 
            batch_size: (int) batch size to use during inference.
                Bigger will result in better throughput performance but would use more memory.
            num_workers: Number of workers for the dataloader
            input_channel_selector (int | Iterable[int] | str): select a single channel or a subset of channels from multi-channel audio. If set to `'average'`, it performs averaging across channels. Disabled if set to `None`. Defaults to `None`.

        Returns:
        """
        if paths2audio_files is None or len(paths2audio_files) == 0:
            return {}

        if num_workers is None:
            num_workers = min(batch_size, os.cpu_count() - 1)

        # Output
        paths2processed_files = []

        # Model's mode and device
        mode = self.training
        device = next(self.parameters()).device

        try:
            # Switch model to evaluation mode
            self.eval()
            # Freeze weights
            self.freeze()

            logging_level = logging.get_verbosity()
            logging.set_verbosity(logging.WARNING)

            # Processing
            with tempfile.TemporaryDirectory() as tmpdir:
                # Save temporary manifest
                temporary_manifest_filepath = os.path.join(tmpdir, 'manifest.json')
                with open(temporary_manifest_filepath, 'w', encoding='utf-8') as fp:
                    for audio_file in paths2audio_files:
                        entry = {'input_filepath': audio_file, 'duration': librosa.get_duration(path=audio_file)}
                        fp.write(json.dumps(entry) + '\n')

                config = {
                    'manifest_filepath': temporary_manifest_filepath,
                    'input_key': 'input_filepath',
                    'input_channel_selector': input_channel_selector,
                    'batch_size': min(batch_size, len(paths2audio_files)),
                    'num_workers': num_workers,
                }

                # Create output dir if necessary
                if not os.path.isdir(output_dir):
                    os.makedirs(output_dir)

                # DataLoader for the input files
                temporary_dataloader = self._setup_process_dataloader(config)

                # Indexing of the original files, used to form the output file name
                file_idx = 0

                # Process batches
                for test_batch in tqdm(temporary_dataloader, desc="Processing"):
                    input_signal = test_batch[0]
                    input_length = test_batch[1]

                    # Expand channel dimension, if necessary
                    # For consistency, the model uses multi-channel format, even if the channel dimension is 1
                    if input_signal.ndim == 2:
                        input_signal = input_signal.unsqueeze(1)

                    processed_batch, _ = self.forward(
                        input_signal=input_signal.to(device), input_length=input_length.to(device)
                    )

                    for example_idx in range(processed_batch.size(0)):
                        # This assumes the data loader is not shuffling files
                        file_name = os.path.basename(paths2audio_files[file_idx])
                        # Prepare output file
                        output_file = os.path.join(output_dir, f'processed_{file_name}')
                        # Crop the output signal to the actual length
                        output_signal = processed_batch[example_idx, :, : input_length[example_idx]].cpu().numpy()
                        # Write audio
                        sf.write(output_file, output_signal.T, self.sample_rate, 'float')
                        # Update the file counter
                        file_idx += 1
                        # Save processed file
                        paths2processed_files.append(output_file)

                    del test_batch
                    del processed_batch

        finally:
            # set mode back to its original value
            self.train(mode=mode)
            if mode is True:
                self.unfreeze()
            logging.set_verbosity(logging_level)

        return paths2processed_files

    def _setup_dataloader_from_config(self, config: Optional[Dict]):

        if config.get("use_lhotse", False):
            return get_lhotse_dataloader_from_config(
                config, global_rank=self.global_rank, world_size=self.world_size, dataset=LhotseAudioToTargetDataset()
            )

        is_concat = config.get('is_concat', False)
        if is_concat:
            raise NotImplementedError('Concat not implemented')

        # TODO: Consider moving `inject` from `audio_to_text_dataset` to a utility module?
        # Automatically inject args from model config to dataloader config
        inject_dataloader_value_from_model_config(self.cfg, config, key='sample_rate')

        # Instantiate tarred dataset loader or normal dataset loader
        if config.get('is_tarred', False):
            raise NotImplementedError('Tarred datasets not supported')

        if 'manifest_filepath' in config and config['manifest_filepath'] is None:
            logging.warning(f"Could not load dataset as `manifest_filepath` was None. Provided config : {config}")
            return None

        dataset = audio_to_audio_dataset.get_audio_to_target_dataset(config=config)

        if hasattr(dataset, 'collate_fn'):
            collate_fn = dataset.collate_fn
        elif hasattr(dataset.datasets[0], 'collate_fn'):
            # support datasets that are lists of entries
            collate_fn = dataset.datasets[0].collate_fn
        else:
            # support datasets that are lists of lists
            collate_fn = dataset.datasets[0].datasets[0].collate_fn

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=config['batch_size'],
            collate_fn=collate_fn,
            drop_last=config.get('drop_last', False),
            shuffle=config['shuffle'],
            num_workers=config.get('num_workers', 0),
            pin_memory=config.get('pin_memory', False),
        )

    def setup_training_data(self, train_data_config: Optional[Union[DictConfig, Dict]]):
        """
        Sets up the training data loader via a Dict-like object.

        Args:
            train_data_config: A config that contains the information regarding construction
                of a training dataset.

        Supported Datasets:
            -   :class:`~nemo.collections.asr.data.audio_to_audio.AudioToTargetDataset`
        """
        if 'shuffle' not in train_data_config:
            train_data_config['shuffle'] = True

        # preserve config
        self._update_dataset_config(dataset_name='train', config=train_data_config)

        self._train_dl = self._setup_dataloader_from_config(config=train_data_config)

        if 'is_tarred' in train_data_config and train_data_config['is_tarred']:
            raise NotImplementedError('Tarred datasets not supported')

    def setup_validation_data(self, val_data_config: Optional[Union[DictConfig, Dict]]):
        """
        Sets up the validation data loader via a Dict-like object.

        Args:
            val_data_config: A config that contains the information regarding construction
                of a validation dataset.

        Supported Datasets:
            -   :class:`~nemo.collections.asr.data.audio_to_audio.AudioToTargetDataset`
        """
        if 'shuffle' not in val_data_config:
            val_data_config['shuffle'] = False

        # preserve config
        self._update_dataset_config(dataset_name='validation', config=val_data_config)

        self._validation_dl = self._setup_dataloader_from_config(config=val_data_config)

    def setup_test_data(self, test_data_config: Optional[Union[DictConfig, Dict]]):
        """
        Sets up the test data loader via a Dict-like object.

        Args:
            test_data_config: A config that contains the information regarding construction
                of a test dataset.

        Supported Datasets:
            -   :class:`~nemo.collections.asr.data.audio_to_audio.AudioToTargetDataset`
        """
        if 'shuffle' not in test_data_config:
            test_data_config['shuffle'] = False

        # preserve config
        self._update_dataset_config(dataset_name='test', config=test_data_config)

        self._test_dl = self._setup_dataloader_from_config(config=test_data_config)

    def _setup_process_dataloader(self, config: Dict) -> 'torch.utils.data.DataLoader':
        """Prepare a dataloader for processing files.

        Args:
            config: A python dictionary which contains the following keys:
                manifest_filepath: path to a manifest file
                input_key: key with audio filepaths in the manifest
                input_channel_selector: Optional, used to select a subset of channels from input audio files
                batch_size: batch size for the dataloader
                num_workers: number of workers for the dataloader

        Returns:
            A pytorch DataLoader for the given manifest filepath.
        """
        dl_config = {
            'manifest_filepath': config['manifest_filepath'],
            'sample_rate': self.sample_rate,
            'input_key': config['input_key'],
            'input_channel_selector': config.get('input_channel_selector', None),
            'target_key': None,
            'target_channel_selector': None,
            'batch_size': config['batch_size'],
            'shuffle': False,
            'num_workers': config.get('num_workers', min(config['batch_size'], os.cpu_count() - 1)),
            'pin_memory': True,
        }

        temporary_dataloader = self._setup_dataloader_from_config(config=DictConfig(dl_config))
        return temporary_dataloader

    @staticmethod
    def match_batch_length(input: torch.Tensor, batch_length: int) -> torch.Tensor:
        """Trim or pad the output to match the batch length.

        Args:
            input: tensor with shape (B, C, T)
            batch_length: int

        Returns:
            Tensor with shape (B, C, T), where T matches the
            batch length.
        """
        input_length = input.size(-1)
        pad_length = batch_length - input_length
        pad = (0, pad_length)
        # pad with zeros or crop
        return torch.nn.functional.pad(input, pad, 'constant', 0)

    @torch.no_grad()
    def process(
        self,
        paths2audio_files: List[str],
        output_dir: str,
        batch_size: int = 1,
        num_workers: Optional[int] = None,
        input_channel_selector: Optional[ChannelSelectorType] = None,
    ) -> List[str]:
        """
        Takes paths to audio files and returns a list of paths to processed
        audios.

        Args:
            paths2audio_files: paths to audio files to be processed
            output_dir: directory to save the processed files
            batch_size: (int) batch size to use during inference.
            num_workers: Number of workers for the dataloader
            input_channel_selector (int | Iterable[int] | str): select a single channel or a subset of channels from multi-channel audio.
                            If set to `'average'`, it performs averaging across channels. Disabled if set to `None`. Defaults to `None`.

        Returns:
            Paths to processed audio signals.
        """
        if paths2audio_files is None or len(paths2audio_files) == 0:
            return {}

        if num_workers is None:
            num_workers = min(batch_size, os.cpu_count() - 1)

        # Output
        paths2processed_files = []

        # Model's mode and device
        mode = self.training
        device = next(self.parameters()).device

        try:
            # Switch model to evaluation mode
            self.eval()
            # Freeze weights
            self.freeze()

            logging_level = logging.get_verbosity()
            logging.set_verbosity(logging.WARNING)

            # Processing
            with tempfile.TemporaryDirectory() as tmpdir:
                # Save temporary manifest
                temporary_manifest_filepath = os.path.join(tmpdir, 'manifest.json')
                with open(temporary_manifest_filepath, 'w', encoding='utf-8') as fp:
                    for audio_file in paths2audio_files:
                        entry = {'input_filepath': audio_file, 'duration': librosa.get_duration(path=audio_file)}
                        fp.write(json.dumps(entry) + '\n')

                config = {
                    'manifest_filepath': temporary_manifest_filepath,
                    'input_key': 'input_filepath',
                    'input_channel_selector': input_channel_selector,
                    'batch_size': min(batch_size, len(paths2audio_files)),
                    'num_workers': num_workers,
                }

                # Create output dir if necessary
                if not os.path.isdir(output_dir):
                    os.makedirs(output_dir)

                # DataLoader for the input files
                temporary_dataloader = self._setup_process_dataloader(config)

                # Indexing of the original files, used to form the output file name
                file_idx = 0

                # Process batches
                for test_batch in tqdm(temporary_dataloader, desc="Processing"):
                    input_signal = test_batch[0]
                    input_length = test_batch[1]

                    # Expand channel dimension, if necessary
                    # For consistency, the model uses multi-channel format, even if the channel dimension is 1
                    if input_signal.ndim == 2:
                        input_signal = input_signal.unsqueeze(1)

                    processed_batch, _ = self.forward(
                        input_signal=input_signal.to(device), input_length=input_length.to(device)
                    )

                    for example_idx in range(processed_batch.size(0)):
                        # This assumes the data loader is not shuffling files
                        file_name = os.path.basename(paths2audio_files[file_idx])
                        # Prepare output file
                        output_file = os.path.join(output_dir, f'processed_{file_name}')
                        # Crop the output signal to the actual length
                        output_signal = processed_batch[example_idx, :, : input_length[example_idx]].cpu().numpy()
                        # Write audio
                        sf.write(output_file, output_signal.T, self.sample_rate, 'float')
                        # Update the file counter
                        file_idx += 1
                        # Save processed file
                        paths2processed_files.append(output_file)

                    del test_batch
                    del processed_batch

        finally:
            # set mode back to its original value
            self.train(mode=mode)
            if mode is True:
                self.unfreeze()
            logging.set_verbosity(logging_level)

        return paths2processed_files

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
