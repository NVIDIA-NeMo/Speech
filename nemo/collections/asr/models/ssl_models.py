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

from math import ceil
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from omegaconf import DictConfig
from pytorch_lightning import Trainer

from nemo.collections.asr.data import audio_to_text_dataset
from nemo.collections.asr.data.audio_to_text_dali import DALIOutputs
from nemo.collections.asr.parts.mixins import ASRModuleMixin
from nemo.collections.asr.parts.preprocessing.perturb import process_augmentations
from nemo.core.classes import ModelPT
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.classes.mixins import AccessMixin, set_access_cfg
from nemo.core.neural_types import (
    AcousticEncodedRepresentation,
    AudioSignal,
    LabelsType,
    LengthsType,
    NeuralType,
    SpectrogramType,
)
from nemo.utils import logging

__all__ = ['SpeechEncDecSelfSupervisedModel']


class SpeechEncDecSelfSupervisedModel(ModelPT, ASRModuleMixin, AccessMixin):
    """Base class for encoder-decoder models used for self-supervised encoder pre-training"""

    @classmethod
    def list_available_models(cls) -> List[PretrainedModelInfo]:
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.

        Returns:
            List of available pre-trained models.
        """
        results = []

        model = PretrainedModelInfo(
            pretrained_model_name="ssl_en_conformer_large",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:ssl_en_conformer_large",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/ssl_en_conformer_large/versions/1.10.1/files/ssl_en_conformer_large.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="ssl_en_conformer_xlarge",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:ssl_en_conformer_xlarge",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/ssl_en_conformer_xlarge/versions/1.10.0/files/ssl_en_conformer_xlarge.nemo",
        )
        results.append(model)

        return results

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        # Get global rank and total number of GPU workers for IterableDataset partitioning, if applicable
        # Global_rank and local_rank is set by LightningModule in Lightning 1.2.0
        self.world_size = 1
        if trainer is not None:
            self.world_size = trainer.world_size

        super().__init__(cfg=cfg, trainer=trainer)
        self.preprocessor = SpeechEncDecSelfSupervisedModel.from_config_dict(self._cfg.preprocessor)
        self.encoder = SpeechEncDecSelfSupervisedModel.from_config_dict(self._cfg.encoder)

        self.decoder_losses = None

        if "loss_list" in self._cfg:

            self.decoder_losses = {}
            self.loss_alphas = {}
            self.start_step = {}
            self.output_from_layer = {}
            self.transpose_encoded = {}
            self.targets_from_loss = {}
            self.decoder_losses_active = {}
            # need to be separate for moduledict

            for decoder_loss_name, decoder_loss_cfg in self._cfg.loss_list.items():
                if not decoder_loss_cfg.get("is_active", True):  # active by default
                    continue

                new_decoder_loss = {
                    'decoder': SpeechEncDecSelfSupervisedModel.from_config_dict(decoder_loss_cfg.decoder),
                    'loss': SpeechEncDecSelfSupervisedModel.from_config_dict(decoder_loss_cfg.loss),
                }
                new_decoder_loss = nn.ModuleDict(new_decoder_loss)
                self.decoder_losses[decoder_loss_name] = new_decoder_loss
                self.loss_alphas[decoder_loss_name] = decoder_loss_cfg.get("loss_alpha", 1.0)
                self.output_from_layer[decoder_loss_name] = decoder_loss_cfg.get("output_from_layer", None)
                self.targets_from_loss[decoder_loss_name] = decoder_loss_cfg.get("targets_from_loss", None)
                self.start_step[decoder_loss_name] = decoder_loss_cfg.get("start_step", 0)
                self.transpose_encoded[decoder_loss_name] = decoder_loss_cfg.get("transpose_encoded", False)
                self.decoder_losses_active[decoder_loss_name] = True

            self.decoder_losses = nn.ModuleDict(self.decoder_losses)

        else:
            self.decoder_ssl = SpeechEncDecSelfSupervisedModel.from_config_dict(self._cfg.decoder)
            self.loss = SpeechEncDecSelfSupervisedModel.from_config_dict(self._cfg.loss)

        self.spec_augmentation = SpeechEncDecSelfSupervisedModel.from_config_dict(self._cfg.spec_augment)

        # dropout for features/spectrograms (applied before masking)
        self.dropout_features = (
            torch.nn.Dropout(self._cfg.dropout_features) if "dropout_features" in self._cfg else None
        )

        # dropout for targets (applied before quantization)
        self.dropout_features_q = (
            torch.nn.Dropout(self._cfg.dropout_features_q) if "dropout_features_q" in self._cfg else None
        )

        # Feature penalty for preprocessor encodings (for Wav2Vec training)
        if "feature_penalty" in self._cfg:
            self.feat_pen, self.pen_factor = 0.0, self._cfg.feature_penalty
        else:
            self.feat_pen, self.pen_factor = None, None

        if "access" in self._cfg:
            set_access_cfg(self._cfg.access)

        self.apply_masking = True

    def _setup_dataloader_from_config(self, config: Optional[Dict]):
        if 'augmentor' in config:
            augmentor = process_augmentations(config['augmentor'])
        else:
            augmentor = None

        # Automatically inject args from model config to dataloader config
        audio_to_text_dataset.inject_dataloader_value_from_model_config(self.cfg, config, key='sample_rate')

        shuffle = config['shuffle']
        device = 'gpu' if torch.cuda.is_available() else 'cpu'
        if config.get('use_dali', False):
            device_id = self.local_rank if device == 'gpu' else None
            dataset = audio_to_text_dataset.get_dali_char_dataset(
                config=config,
                shuffle=shuffle,
                device_id=device_id,
                global_rank=self.global_rank,
                world_size=self.world_size,
                preprocessor_cfg=self._cfg.preprocessor,
            )
            return dataset

        # Instantiate tarred dataset loader or normal dataset loader
        if config.get('is_tarred', False):
            if ('tarred_audio_filepaths' in config and config['tarred_audio_filepaths'] is None) or (
                'manifest_filepath' in config and config['manifest_filepath'] is None
            ):
                logging.warning(
                    "Could not load dataset as `manifest_filepath` was None or "
                    f"`tarred_audio_filepaths` is None. Provided config : {config}"
                )
                return None

            shuffle_n = config.get('shuffle_n', 4 * config['batch_size']) if shuffle else 0
            dataset = audio_to_text_dataset.get_tarred_dataset(
                config=config,
                shuffle_n=shuffle_n,
                global_rank=self.global_rank,
                world_size=self.world_size,
                augmentor=augmentor,
            )
            shuffle = False
        else:
            if 'manifest_filepath' in config and config['manifest_filepath'] is None:
                logging.warning(f"Could not load dataset as `manifest_filepath` was None. Provided config : {config}")
                return None

            dataset = audio_to_text_dataset.get_char_dataset(config=config, augmentor=augmentor)

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
            shuffle=shuffle,
            num_workers=config.get('num_workers', 0),
            pin_memory=config.get('pin_memory', False),
        )

    def setup_training_data(self, train_data_config: Optional[Union[DictConfig, Dict]]):
        """
        Sets up the training data loader via a Dict-like object.

        Args:
            train_data_config: A config that contains the information regarding construction
                of an ASR Training dataset.

        Supported Datasets:
            -   :class:`~nemo.collections.asr.data.audio_to_text.AudioToCharDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.AudioToBPEDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.TarredAudioToCharDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.TarredAudioToBPEDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text_dali.AudioToCharDALIDataset`
        """
        if 'shuffle' not in train_data_config:
            train_data_config['shuffle'] = True

        # preserve config
        self._update_dataset_config(dataset_name='train', config=train_data_config)

        self._train_dl = self._setup_dataloader_from_config(config=train_data_config)

        # Need to set this because if using an IterableDataset, the length of the dataloader is the total number
        # of samples rather than the number of batches, and this messes up the tqdm progress bar.
        # So we set the number of steps manually (to the correct number) to fix this.
        if (
            self._train_dl is not None
            and hasattr(self._train_dl, 'dataset')
            and isinstance(self._train_dl.dataset, torch.utils.data.IterableDataset)
        ):
            # We also need to check if limit_train_batches is already set.
            # If it's an int, we assume that the user has set it to something sane, i.e. <= # training batches,
            # and don't change it. Otherwise, adjust batches accordingly if it's a float (including 1.0).
            if isinstance(self._trainer.limit_train_batches, float):
                self._trainer.limit_train_batches = int(
                    self._trainer.limit_train_batches
                    * ceil((len(self._train_dl.dataset) / self.world_size) / train_data_config['batch_size'])
                )

    def setup_validation_data(self, val_data_config: Optional[Union[DictConfig, Dict]]):
        """
        Sets up the validation data loader via a Dict-like object.

        Args:
            val_data_config: A config that contains the information regarding construction
                of an ASR Training dataset.

        Supported Datasets:
            -   :class:`~nemo.collections.asr.data.audio_to_text.AudioToCharDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.AudioToBPEDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.TarredAudioToCharDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.TarredAudioToBPEDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text_dali.AudioToCharDALIDataset`
        """
        if 'shuffle' not in val_data_config:
            val_data_config['shuffle'] = False

        # preserve config
        self._update_dataset_config(dataset_name='validation', config=val_data_config)

        self._validation_dl = self._setup_dataloader_from_config(config=val_data_config)

        # Need to set this because if using an IterableDataset, the length of the dataloader is the total number
        # of samples rather than the number of batches, and this messes up the tqdm progress bar.
        # So we set the number of steps manually (to the correct number) to fix this.
        if (
            self._validation_dl is not None
            and hasattr(self._validation_dl, 'dataset')
            and isinstance(self._validation_dl.dataset, torch.utils.data.IterableDataset)
        ):
            # We also need to check if limit_train_batches is already set.
            # If it's an int, we assume that the user has set it to something sane, i.e. <= # training batches,
            # and don't change it. Otherwise, adjust batches accordingly if it's a float (including 1.0).
            if isinstance(self._trainer.limit_val_batches, float):
                self._trainer.limit_val_batches = int(
                    self._trainer.limit_val_batches
                    * ceil((len(self._validation_dl.dataset) / self.world_size) / val_data_config['batch_size'])
                )

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        if hasattr(self.preprocessor, '_sample_rate'):
            input_signal_eltype = AudioSignal(freq=self.preprocessor._sample_rate)
        else:
            input_signal_eltype = AudioSignal()
        return {
            "input_signal": NeuralType(('B', 'T'), input_signal_eltype, optional=True),
            "input_signal_length": NeuralType(tuple('B'), LengthsType(), optional=True),
            "processed_signal": NeuralType(('B', 'D', 'T'), SpectrogramType(), optional=True),
            "processed_signal_length": NeuralType(tuple('B'), LengthsType(), optional=True),
            "targets": NeuralType(('B', 'T'), LabelsType(), optional=True),
            "target_lengths": NeuralType(tuple('B'), LengthsType(), optional=True),
        }

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "spectrograms": NeuralType(('B', 'D', 'T'), SpectrogramType()),
            "spec_masks": NeuralType(('B', 'D', 'T'), SpectrogramType()),
            "encoded": NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation()),
            "encoded_len": NeuralType(tuple('B'), LengthsType()),
        }

    @typecheck()
    def forward(
        self, input_signal=None, input_signal_length=None, processed_signal=None, processed_signal_length=None,
    ):
        """
        Forward pass of the model.

        Args:
            input_signal: Tensor that represents a batch of raw audio signals,
                of shape [B, T]. T here represents timesteps, with 1 second of audio represented as
                `self.sample_rate` number of floating point values.
            input_signal_length: Vector of length B, that contains the individual lengths of the audio
                sequences.
            processed_signal: Tensor that represents a batch of processed audio signals,
                of shape (B, D, T) that has undergone processing via some DALI preprocessor.
            processed_signal_length: Vector of length B, that contains the individual lengths of the
                processed audio sequences.

        Returns:
            A tuple of 4 elements -
            1) Processed spectrograms of shape [B, D, T].
            2) Masks applied to spectrograms of shape [B, D, T].
            3) The encoded features tensor of shape [B, D, T].
            2) The lengths of the acoustic sequence after propagation through the encoder, of shape [B].
        """
        # Reset access registry
        if self.is_access_enabled():
            self.reset_registry()

        # Check for special flag for validation step
        if hasattr(self, '_in_validation_step'):
            in_validation_step = self._in_validation_step
        else:
            in_validation_step = False

        # reset module registry from AccessMixin
        if (
            (self.training or in_validation_step)
            and self.decoder_losses is not None
            and self.output_from_layer is not None
            and len(self.output_from_layer) > 0
        ):
            layer_names = list(self.output_from_layer.values())
            register_layer = any([name is not None for name in layer_names])

            if register_layer:
                self.access_cfg['save_encoder_tensors'] = True
                self.set_access_enabled(access_enabled=True)

        has_input_signal = input_signal is not None and input_signal_length is not None
        has_processed_signal = processed_signal is not None and processed_signal_length is not None
        if (has_input_signal ^ has_processed_signal) == False:
            raise ValueError(
                f"{self} Arguments ``input_signal`` and ``input_signal_length`` are mutually exclusive "
                " with ``processed_signal`` and ``processed_signal_len`` arguments."
            )

        if not has_processed_signal:
            processed_signal, processed_signal_length = self.preprocessor(
                input_signal=input_signal, length=input_signal_length,
            )

        if self.pen_factor:
            self.feat_pen = processed_signal.float().pow(2).mean() * self.pen_factor
        spectrograms = processed_signal.detach().clone()

        if self.dropout_features:
            processed_signal = self.dropout_features(processed_signal)
        if self.dropout_features_q:
            spectrograms = self.dropout_features_q(spectrograms)

        if self.apply_masking:
            processed_signal = self.spec_augmentation(input_spec=processed_signal, length=processed_signal_length)

        masked_spectrograms = processed_signal.detach()
        spec_masks = torch.logical_and(masked_spectrograms < 1e-5, masked_spectrograms > -1e-5).float()
        for idx, proc_len in enumerate(processed_signal_length):
            spec_masks[idx, :, proc_len:] = 0.0

        encoded, encoded_len = self.encoder(audio_signal=processed_signal, length=processed_signal_length)

        return spectrograms, spec_masks, encoded, encoded_len

    def decoder_loss_step(self, spectrograms, spec_masks, encoded, encoded_len, targets=None, target_lengths=None):
        """
        Forward pass through all decoders and calculate corresponding losses.
        Args:
            spectrograms: Processed spectrograms of shape [B, D, T].
            spec_masks: Masks applied to spectrograms of shape [B, D, T].
            encoded: The encoded features tensor of shape [B, D, T].
            encoded_len: The lengths of the acoustic sequence after propagation through the encoder, of shape [B].
            targets: Optional target labels of shape [B, T]
            target_lengths: Optional target label lengths of shape [B]

        Returns:
            A tuple of 2 elements -
            1) Total sum of losses weighted by corresponding loss_alphas
            2) Dictionary of unweighted losses
        """
        loss_val_dict = {}

        if self.decoder_losses is None:
            if hasattr(self.decoder_ssl, "needs_labels") and self.decoder_ssl.needs_labels:
                outputs = self.decoder_ssl(encoder_output=encoded, targets=targets, target_lengths=target_lengths)
            else:
                outputs = self.decoder_ssl(encoder_output=encoded)
            if self.loss.needs_labels:
                loss_value = self.loss(
                    spec_masks=spec_masks,
                    decoder_outputs=outputs,
                    targets=targets,
                    decoder_lengths=encoded_len,
                    target_lengths=target_lengths,
                )
            else:
                loss_value = self.loss(spectrograms=spectrograms, spec_masks=spec_masks, decoder_outputs=outputs)
        else:

            loss_value = encoded.new_zeros(1)
            outputs = {}
            registry = self.get_module_registry(self.encoder)

            for dec_loss_name, dec_loss in self.decoder_losses.items():
                # loop through decoders and corresponding losses
                if not self.decoder_losses_active[dec_loss_name]:
                    continue

                if self.output_from_layer[dec_loss_name] is None:
                    dec_input = encoded
                else:
                    # extract output from specified layer using AccessMixin registry
                    dec_input = registry[self.output_from_layer[dec_loss_name]]['encoder'][-1]
                if self.transpose_encoded[dec_loss_name]:
                    dec_input = dec_input.transpose(-2, -1)

                if self.targets_from_loss[dec_loss_name] is not None:
                    # extract targets from specified loss
                    target_loss = self.targets_from_loss[dec_loss_name]
                    targets = self.decoder_losses[target_loss]['loss'].target_ids
                    target_lengths = self.decoder_losses[target_loss]['loss'].target_lengths
                    if target_lengths is None:
                        target_lengths = encoded_len

                if hasattr(dec_loss['decoder'], "needs_labels") and dec_loss['decoder'].needs_labels:
                    # if we are using a decoder which needs labels, provide them
                    outputs[dec_loss_name] = dec_loss['decoder'](
                        encoder_output=dec_input, targets=targets, target_lengths=target_lengths
                    )
                else:
                    outputs[dec_loss_name] = dec_loss['decoder'](encoder_output=dec_input)

                current_loss = dec_loss['loss']
                if current_loss.needs_labels:
                    # if we are using a loss which needs labels, provide them
                    current_loss_value = current_loss(
                        spec_masks=spec_masks,
                        decoder_outputs=outputs[dec_loss_name],
                        targets=targets,
                        decoder_lengths=encoded_len,
                        target_lengths=target_lengths,
                    )
                else:
                    current_loss_value = current_loss(
                        spectrograms=spectrograms,
                        spec_masks=spec_masks,
                        decoder_outputs=outputs[dec_loss_name],
                        decoder_lengths=encoded_len,
                    )
                loss_value = loss_value + current_loss_value * self.loss_alphas[dec_loss_name]
                loss_val_dict[dec_loss_name] = current_loss_value

        return loss_value, loss_val_dict

    # PTL-specific methods
    def training_step(self, batch, batch_nb):
        signal, signal_len, targets, target_lengths = batch
        if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
            spectrograms, spec_masks, encoded, encoded_len = self.forward(
                processed_signal=signal, processed_signal_length=signal_len,
            )
        else:
            spectrograms, spec_masks, encoded, encoded_len = self.forward(
                input_signal=signal, input_signal_length=signal_len,
            )

        if self.decoder_losses is not None:
            for dec_loss_name, dec_loss in self.decoder_losses.items():
                self.decoder_losses_active[dec_loss_name] = self.trainer.global_step >= self.start_step[dec_loss_name]
                loss = dec_loss['loss']
                if hasattr(loss, "set_num_updates"):
                    loss.set_num_updates(self.trainer.global_step)
        else:
            if hasattr(self.loss, "set_num_updates"):
                self.loss.set_num_updates(self.trainer.global_step)

        loss_value, loss_val_dict = self.decoder_loss_step(
            spectrograms, spec_masks, encoded, encoded_len, targets, target_lengths
        )

        tensorboard_logs = {
            'learning_rate': self._optimizer.param_groups[0]['lr'],
            'global_step': self.trainer.global_step,
        }

        for loss_name, loss_val in loss_val_dict.items():
            tensorboard_logs['train_' + loss_name] = loss_val

        if self.feat_pen:
            loss_value += self.feat_pen

        # Reset access registry
        self.reset_registry()

        return {'loss': loss_value, 'log': tensorboard_logs}

    def validation_pass(self, batch, batch_idx, dataloader_idx=0):
        # Set flag to register tensors
        self._in_validation_step = True

        signal, signal_len, targets, target_lengths = batch
        if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
            spectrograms, spec_masks, encoded, encoded_len = self.forward(
                processed_signal=signal, processed_signal_length=signal_len,
            )
        else:
            spectrograms, spec_masks, encoded, encoded_len = self.forward(
                input_signal=signal, input_signal_length=signal_len,
            )

        if self.decoder_losses is not None:
            for dec_loss_name, dec_loss in self.decoder_losses.items():
                self.decoder_losses_active[dec_loss_name] = self.trainer.global_step >= self.start_step[dec_loss_name]

        loss_value, _ = self.decoder_loss_step(spectrograms, spec_masks, encoded, encoded_len, targets, target_lengths)

        if self.feat_pen:
            loss_value += self.feat_pen

        # reset access registry
        self.reset_registry()
        del self._in_validation_step

        metrics = {'val_loss': loss_value}

        return metrics

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        metrics = self.validation_pass(batch, batch_idx, dataloader_idx)
        if type(self.trainer.val_dataloaders) == list and len(self.trainer.val_dataloaders) > 1:
            self.validation_step_outputs[dataloader_idx].append(metrics)
        else:
            self.validation_step_outputs.append(metrics)
        return metrics

    def multi_validation_epoch_end(self, outputs, dataloader_idx: int = 0):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': val_loss_mean}
        return {'val_loss': val_loss_mean, 'log': tensorboard_logs}
