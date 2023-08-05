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

import copy
import json
import os
import tempfile
from abc import abstractmethod
from math import ceil, floor
from typing import Dict, List, Optional, Union

import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from pytorch_lightning import Trainer
from torchmetrics import Accuracy
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError

from nemo.collections.asr.data import audio_to_label_dataset, feature_to_label_dataset
from nemo.collections.asr.models.asr_model import ASRModel, ExportableEncDecModel
from nemo.collections.asr.parts.preprocessing.features import WaveformFeaturizer
from nemo.collections.asr.parts.preprocessing.perturb import process_augmentations
from nemo.collections.common.losses import CrossEntropyLoss, MSELoss
from nemo.collections.common.metrics import TopKClassificationAccuracy
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.neural_types import *
from nemo.utils import logging, model_utils
from nemo.utils.cast_utils import cast_all

__all__ = ['EncDecClassificationModel', 'EncDecRegressionModel']


class _EncDecBaseModel(ASRModel, ExportableEncDecModel):
    """Encoder decoder Classification models."""

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        # Get global rank and total number of GPU workers for IterableDataset partitioning, if applicable
        # Global_rank and local_rank is set by LightningModule in Lightning 1.2.0
        self.world_size = 1
        if trainer is not None:
            self.world_size = trainer.num_nodes * trainer.num_devices

        # Convert config to a DictConfig
        cfg = model_utils.convert_model_config_to_dict_config(cfg)

        # Convert config to support Hydra 1.0+ instantiation
        cfg = model_utils.maybe_update_config_version(cfg)

        self.is_regression_task = cfg.get('is_regression_task', False)
        # Change labels if needed
        self._update_decoder_config(cfg.labels, cfg.decoder)
        super().__init__(cfg=cfg, trainer=trainer)

        if hasattr(self._cfg, 'spec_augment') and self._cfg.spec_augment is not None:
            self.spec_augmentation = ASRModel.from_config_dict(self._cfg.spec_augment)
        else:
            self.spec_augmentation = None
        if hasattr(self._cfg, 'crop_or_pad_augment') and self._cfg.crop_or_pad_augment is not None:
            self.crop_or_pad = ASRModel.from_config_dict(self._cfg.crop_or_pad_augment)
        else:
            self.crop_or_pad = None

        self.preprocessor = self._setup_preprocessor()
        self.encoder = self._setup_encoder()
        self.decoder = self._setup_decoder()
        self.loss = self._setup_loss()
        self._setup_metrics()

    @abstractmethod
    def _setup_preprocessor(self):
        """
        Setup preprocessor for audio data
        Returns: Preprocessor

        """
        pass

    @abstractmethod
    def _setup_encoder(self):
        """
        Setup encoder for the Encoder-Decoder network
        Returns: Encoder
        """
        pass

    @abstractmethod
    def _setup_decoder(self):
        """
        Setup decoder for the Encoder-Decoder network
        Returns: Decoder
        """
        pass

    @abstractmethod
    def _setup_loss(self):
        """
        Setup loss function for training
        Returns: Loss function

        """
        pass

    @abstractmethod
    def _setup_metrics(self):
        """
        Setup metrics to be tracked in addition to loss
        Returns: void

        """
        pass

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        if hasattr(self.preprocessor, '_sample_rate'):
            audio_eltype = AudioSignal(freq=self.preprocessor._sample_rate)
        else:
            audio_eltype = AudioSignal()
        return {
            "input_signal": NeuralType(('B', 'T'), audio_eltype, optional=True),
            "input_signal_length": NeuralType(tuple('B'), LengthsType(), optional=True),
            "processed_signal": NeuralType(('B', 'D', 'T'), SpectrogramType(), optional=True),
            "processed_signal_length": NeuralType(tuple('B'), LengthsType(), optional=True),
        }

    @property
    @abstractmethod
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        pass

    def forward(
        self, input_signal=None, input_signal_length=None, processed_signal=None, processed_signal_length=None
    ):
        has_input_signal = input_signal is not None and input_signal_length is not None
        has_processed_signal = processed_signal is not None and processed_signal_length is not None
        if (has_input_signal ^ has_processed_signal) == False:
            raise ValueError(
                f"{self} Arguments ``input_signal`` and ``input_signal_length`` are mutually exclusive "
                " with ``processed_signal`` and ``processed_signal_length`` arguments."
            )

        if not has_processed_signal:
            processed_signal, processed_signal_length = self.preprocessor(
                input_signal=input_signal, length=input_signal_length,
            )
        # Crop or pad is always applied
        if self.crop_or_pad is not None:
            processed_signal, processed_signal_length = self.crop_or_pad(
                input_signal=processed_signal, length=processed_signal_length
            )
        # Spec augment is not applied during evaluation/testing
        if self.spec_augmentation is not None and self.training:
            processed_signal = self.spec_augmentation(input_spec=processed_signal, length=processed_signal_length)
        encoded, encoded_len = self.encoder(audio_signal=processed_signal, length=processed_signal_length)
        logits = self.decoder(encoder_output=encoded)
        return logits

    def setup_training_data(self, train_data_config: Optional[Union[DictConfig, Dict]]):
        if 'shuffle' not in train_data_config:
            train_data_config['shuffle'] = True
        # preserve config
        self._update_dataset_config(dataset_name='train', config=train_data_config)

        self._train_dl = self._setup_dataloader_from_config(config=DictConfig(train_data_config))

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
        if 'shuffle' not in val_data_config:
            val_data_config['shuffle'] = False

        # preserve config
        self._update_dataset_config(dataset_name='validation', config=val_data_config)

        self._validation_dl = self._setup_dataloader_from_config(config=DictConfig(val_data_config))

    def setup_test_data(self, test_data_config: Optional[Union[DictConfig, Dict]], use_feat: bool = False):
        if 'shuffle' not in test_data_config:
            test_data_config['shuffle'] = False

        # preserve config
        self._update_dataset_config(dataset_name='test', config=test_data_config)

        if use_feat and hasattr(self, '_setup_feature_label_dataloader'):
            self._test_dl = self._setup_feature_label_dataloader(config=DictConfig(test_data_config))
        else:
            self._test_dl = self._setup_dataloader_from_config(config=DictConfig(test_data_config))

    def test_dataloader(self):
        if self._test_dl is not None:
            return self._test_dl

    def _setup_dataloader_from_config(self, config: DictConfig):

        OmegaConf.set_struct(config, False)
        config.is_regression_task = self.is_regression_task
        OmegaConf.set_struct(config, True)

        if 'augmentor' in config:
            augmentor = process_augmentations(config['augmentor'])
        else:
            augmentor = None

        featurizer = WaveformFeaturizer(
            sample_rate=config['sample_rate'], int_values=config.get('int_values', False), augmentor=augmentor
        )
        shuffle = config['shuffle']

        # Instantiate tarred dataset loader or normal dataset loader
        if config.get('is_tarred', False):
            if ('tarred_audio_filepaths' in config and config['tarred_audio_filepaths'] is None) or (
                'manifest_filepath' in config and config['manifest_filepath'] is None
            ):
                logging.warning(
                    "Could not load dataset as `manifest_filepath` is None or "
                    f"`tarred_audio_filepaths` is None. Provided config : {config}"
                )
                return None

            if 'vad_stream' in config and config['vad_stream']:
                logging.warning("VAD inference does not support tarred dataset now")
                return None

            shuffle_n = config.get('shuffle_n', 4 * config['batch_size']) if shuffle else 0
            dataset = audio_to_label_dataset.get_tarred_classification_label_dataset(
                featurizer=featurizer,
                config=config,
                shuffle_n=shuffle_n,
                global_rank=self.global_rank,
                world_size=self.world_size,
            )
            shuffle = False
            batch_size = config['batch_size']
            if hasattr(dataset, 'collate_fn'):
                collate_fn = dataset.collate_fn
            elif hasattr(dataset.datasets[0], 'collate_fn'):
                # support datasets that are lists of entries
                collate_fn = dataset.datasets[0].collate_fn
            else:
                # support datasets that are lists of lists
                collate_fn = dataset.datasets[0].datasets[0].collate_fn

        else:
            if 'manifest_filepath' in config and config['manifest_filepath'] is None:
                logging.warning(f"Could not load dataset as `manifest_filepath` is None. Provided config : {config}")
                return None

            if 'vad_stream' in config and config['vad_stream']:
                logging.info("Perform streaming frame-level VAD")
                dataset = audio_to_label_dataset.get_speech_label_dataset(featurizer=featurizer, config=config)
                batch_size = 1
                collate_fn = dataset.vad_frame_seq_collate_fn
            else:
                dataset = audio_to_label_dataset.get_classification_label_dataset(featurizer=featurizer, config=config)
                batch_size = config['batch_size']
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
            batch_size=batch_size,
            collate_fn=collate_fn,
            drop_last=config.get('drop_last', False),
            shuffle=shuffle,
            num_workers=config.get('num_workers', 0),
            pin_memory=config.get('pin_memory', False),
        )

    def _setup_feature_label_dataloader(self, config: DictConfig) -> torch.utils.data.DataLoader:
        """
        setup dataloader for VAD inference with audio features as input
        """

        OmegaConf.set_struct(config, False)
        config.is_regression_task = self.is_regression_task
        OmegaConf.set_struct(config, True)

        if 'augmentor' in config:
            augmentor = process_augmentations(config['augmentor'])
        else:
            augmentor = None
        if 'manifest_filepath' in config and config['manifest_filepath'] is None:
            logging.warning(f"Could not load dataset as `manifest_filepath` is None. Provided config : {config}")
            return None

        dataset = feature_to_label_dataset.get_feature_label_dataset(config=config, augmentor=augmentor)
        if 'vad_stream' in config and config['vad_stream']:
            collate_func = dataset._vad_segment_collate_fn
            batch_size = 1
            shuffle = False
        else:
            collate_func = dataset._collate_fn
            batch_size = config['batch_size']
            shuffle = config['shuffle']

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            collate_fn=collate_func,
            drop_last=config.get('drop_last', False),
            shuffle=shuffle,
            num_workers=config.get('num_workers', 0),
            pin_memory=config.get('pin_memory', False),
        )

    @torch.no_grad()
    def transcribe(self, paths2audio_files: List[str], batch_size: int = 4, logprobs=False) -> List[str]:
        """
        Generate class labels for provided audio files. Use this method for debugging and prototyping.

        Args:
            paths2audio_files: (a list) of paths to audio files. \
                Recommended length per file is approximately 1 second.
            batch_size: (int) batch size to use during inference. \
                Bigger will result in better throughput performance but would use more memory.
            logprobs: (bool) pass True to get log probabilities instead of class labels.

        Returns:

            A list of transcriptions (or raw log probabilities if logprobs is True) in the same order as paths2audio_files
        """
        if paths2audio_files is None or len(paths2audio_files) == 0:
            return []
        # We will store transcriptions here
        labels = []
        # Model's mode and device
        mode = self.training
        device = next(self.parameters()).device

        if hasattr(self.preprocessor.featurizer, 'dither'):
            dither_value = self.preprocessor.featurizer.dither
        if hasattr(self.preprocessor.featurizer, 'pad_to'):
            pad_to_value = self.preprocessor.featurizer.pad_to

        try:
            if hasattr(self.preprocessor.featurizer, 'dither'):
                self.preprocessor.featurizer.dither = 0.0
            if hasattr(self.preprocessor.featurizer, 'pad_to'):
                self.preprocessor.featurizer.pad_to = 0
            # Switch model to evaluation mode
            self.eval()
            logging_level = logging.get_verbosity()
            logging.set_verbosity(logging.WARNING)
            # Work in tmp directory - will store manifest file there
            with tempfile.TemporaryDirectory() as tmpdir:
                with open(os.path.join(tmpdir, 'manifest.json'), 'w', encoding='utf-8') as fp:
                    for audio_file in paths2audio_files:
                        label = 0.0 if self.is_regression_task else self.cfg.labels[0]
                        entry = {'audio_filepath': audio_file, 'duration': 100000.0, 'label': label}
                        fp.write(json.dumps(entry) + '\n')

                config = {'paths2audio_files': paths2audio_files, 'batch_size': batch_size, 'temp_dir': tmpdir}

                temporary_datalayer = self._setup_transcribe_dataloader(config)
                for test_batch in temporary_datalayer:
                    logits = self.forward(
                        input_signal=test_batch[0].to(device), input_signal_length=test_batch[1].to(device)
                    )
                    if logprobs:
                        # dump log probs per file
                        for idx in range(logits.shape[0]):
                            lg = logits[idx]
                            labels.append(lg.cpu().numpy())
                    else:
                        labels_k = []
                        top_ks = self._accuracy.top_k
                        for top_k_i in top_ks:
                            # replace top k value with current top k
                            self._accuracy.top_k = top_k_i
                            labels_k_i = self._accuracy.top_k_predicted_labels(logits)
                            labels_k_i = labels_k_i.cpu()
                            labels_k.append(labels_k_i)

                        # convenience: if only one top_k, pop out the nested list
                        if len(top_ks) == 1:
                            labels_k = labels_k[0]

                        labels += labels_k
                        # reset top k to orignal value
                        self._accuracy.top_k = top_ks
                    del test_batch
        finally:
            # set mode back to its original value
            self.train(mode=mode)

            if hasattr(self.preprocessor.featurizer, 'dither'):
                self.preprocessor.featurizer.dither = dither_value
            if hasattr(self.preprocessor.featurizer, 'pad_to'):
                self.preprocessor.featurizer.pad_to = pad_to_value
            logging.set_verbosity(logging_level)
        return labels

    def _setup_transcribe_dataloader(self, config: Dict) -> 'torch.utils.data.DataLoader':
        """
        Setup function for a temporary data loader which wraps the provided audio file.

        Args:
            config: A python dictionary which contains the following keys:

        Returns:
            A pytorch DataLoader for the given audio file(s).
        """
        dl_config = {
            'manifest_filepath': os.path.join(config['temp_dir'], 'manifest.json'),
            'sample_rate': self.preprocessor._sample_rate,
            'labels': self.cfg.labels,
            'batch_size': min(config['batch_size'], len(config['paths2audio_files'])),
            'trim_silence': False,
            'shuffle': False,
        }

        temporary_datalayer = self._setup_dataloader_from_config(config=DictConfig(dl_config))
        return temporary_datalayer

    @abstractmethod
    def _update_decoder_config(self, labels, cfg):
        pass


class EncDecClassificationModel(_EncDecBaseModel):
    """Encoder decoder Classification models."""

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):

        if cfg.get("is_regression_task", False):
            raise ValueError(f"EndDecClassificationModel requires the flag is_regression_task to be set as false")

        super().__init__(cfg=cfg, trainer=trainer)

    def _setup_preprocessor(self):
        return EncDecClassificationModel.from_config_dict(self._cfg.preprocessor)

    def _setup_encoder(self):
        return EncDecClassificationModel.from_config_dict(self._cfg.encoder)

    def _setup_decoder(self):
        return EncDecClassificationModel.from_config_dict(self._cfg.decoder)

    def _setup_loss(self):
        return CrossEntropyLoss()

    def _setup_metrics(self):
        self._accuracy = TopKClassificationAccuracy(dist_sync_on_step=True)

    @classmethod
    def list_available_models(cls) -> Optional[List[PretrainedModelInfo]]:
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.

        Returns:
            List of available pre-trained models.
        """
        results = []

        model = PretrainedModelInfo(
            pretrained_model_name="vad_multilingual_marblenet",
            description="For details about this model, please visit https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/vad_multilingual_marblenet",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/vad_multilingual_marblenet/versions/1.10.0/files/vad_multilingual_marblenet.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="vad_telephony_marblenet",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:vad_telephony_marblenet",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/vad_telephony_marblenet/versions/1.0.0rc1/files/vad_telephony_marblenet.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="vad_marblenet",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:vad_marblenet",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/vad_marblenet/versions/1.0.0rc1/files/vad_marblenet.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="commandrecognition_en_matchboxnet3x1x64_v1",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:commandrecognition_en_matchboxnet3x1x64_v1",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/commandrecognition_en_matchboxnet3x1x64_v1/versions/1.0.0rc1/files/commandrecognition_en_matchboxnet3x1x64_v1.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="commandrecognition_en_matchboxnet3x2x64_v1",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:commandrecognition_en_matchboxnet3x2x64_v1",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/commandrecognition_en_matchboxnet3x2x64_v1/versions/1.0.0rc1/files/commandrecognition_en_matchboxnet3x2x64_v1.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="commandrecognition_en_matchboxnet3x1x64_v2",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:commandrecognition_en_matchboxnet3x1x64_v2",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/commandrecognition_en_matchboxnet3x1x64_v2/versions/1.0.0rc1/files/commandrecognition_en_matchboxnet3x1x64_v2.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="commandrecognition_en_matchboxnet3x2x64_v2",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:commandrecognition_en_matchboxnet3x2x64_v2",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/commandrecognition_en_matchboxnet3x2x64_v2/versions/1.0.0rc1/files/commandrecognition_en_matchboxnet3x2x64_v2.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="commandrecognition_en_matchboxnet3x1x64_v2_subset_task",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:commandrecognition_en_matchboxnet3x1x64_v2_subset_task",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/commandrecognition_en_matchboxnet3x1x64_v2_subset_task/versions/1.0.0rc1/files/commandrecognition_en_matchboxnet3x1x64_v2_subset_task.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="commandrecognition_en_matchboxnet3x2x64_v2_subset_task",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:commandrecognition_en_matchboxnet3x2x64_v2_subset_task",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/commandrecognition_en_matchboxnet3x2x64_v2_subset_task/versions/1.0.0rc1/files/commandrecognition_en_matchboxnet3x2x64_v2_subset_task.nemo",
        )
        results.append(model)
        return results

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {"outputs": NeuralType(('B', 'D'), LogitsType())}

    # PTL-specific methods
    def training_step(self, batch, batch_nb):
        audio_signal, audio_signal_len, labels, labels_len = batch
        logits = self.forward(input_signal=audio_signal, input_signal_length=audio_signal_len)
        loss_value = self.loss(logits=logits, labels=labels)

        self.log('train_loss', loss_value)
        self.log('learning_rate', self._optimizer.param_groups[0]['lr'])
        self.log('global_step', self.trainer.global_step)

        self._accuracy(logits=logits, labels=labels)
        topk_scores = self._accuracy.compute()
        self._accuracy.reset()

        for top_k, score in zip(self._accuracy.top_k, topk_scores):
            self.log('training_batch_accuracy_top_{}'.format(top_k), score)

        return {
            'loss': loss_value,
        }

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        audio_signal, audio_signal_len, labels, labels_len = batch
        logits = self.forward(input_signal=audio_signal, input_signal_length=audio_signal_len)
        loss_value = self.loss(logits=logits, labels=labels)
        acc = self._accuracy(logits=logits, labels=labels)
        correct_counts, total_counts = self._accuracy.correct_counts_k, self._accuracy.total_counts_k
        loss = {
            'val_loss': loss_value,
            'val_correct_counts': correct_counts,
            'val_total_counts': total_counts,
            'val_acc': acc,
        }
        if type(self.trainer.val_dataloaders) == list and len(self.trainer.val_dataloaders) > 1:
            self.validation_step_outputs[dataloader_idx].append(loss)
        else:
            self.validation_step_outputs.append(loss)
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        audio_signal, audio_signal_len, labels, labels_len = batch
        logits = self.forward(input_signal=audio_signal, input_signal_length=audio_signal_len)
        loss_value = self.loss(logits=logits, labels=labels)
        acc = self._accuracy(logits=logits, labels=labels)
        correct_counts, total_counts = self._accuracy.correct_counts_k, self._accuracy.total_counts_k
        loss = {
            'test_loss': loss_value,
            'test_correct_counts': correct_counts,
            'test_total_counts': total_counts,
            'test_acc': acc,
        }
        if type(self.trainer.test_dataloaders) == list and len(self.trainer.test_dataloaders) > 1:
            self.test_step_outputs[dataloader_idx].append(loss)
        else:
            self.test_step_outputs.append(loss)
        return loss

    def multi_validation_epoch_end(self, outputs, dataloader_idx: int = 0):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        correct_counts = torch.stack([x['val_correct_counts'] for x in outputs]).sum(axis=0)
        total_counts = torch.stack([x['val_total_counts'] for x in outputs]).sum(axis=0)

        self._accuracy.correct_counts_k = correct_counts
        self._accuracy.total_counts_k = total_counts
        topk_scores = self._accuracy.compute()
        self._accuracy.reset()

        tensorboard_log = {'val_loss': val_loss_mean}
        for top_k, score in zip(self._accuracy.top_k, topk_scores):
            tensorboard_log['val_epoch_top@{}'.format(top_k)] = score

        return {'log': tensorboard_log}

    def multi_test_epoch_end(self, outputs, dataloader_idx: int = 0):
        test_loss_mean = torch.stack([x['test_loss'] for x in outputs]).mean()
        correct_counts = torch.stack([x['test_correct_counts'].unsqueeze(0) for x in outputs]).sum(axis=0)
        total_counts = torch.stack([x['test_total_counts'].unsqueeze(0) for x in outputs]).sum(axis=0)

        self._accuracy.correct_counts_k = correct_counts
        self._accuracy.total_counts_k = total_counts
        topk_scores = self._accuracy.compute()
        self._accuracy.reset()

        tensorboard_log = {'test_loss': test_loss_mean}
        for top_k, score in zip(self._accuracy.top_k, topk_scores):
            tensorboard_log['test_epoch_top@{}'.format(top_k)] = score

        return {'log': tensorboard_log}

    @typecheck()
    def forward(
        self, input_signal=None, input_signal_length=None, processed_signal=None, processed_signal_length=None
    ):
        logits = super().forward(
            input_signal=input_signal,
            input_signal_length=input_signal_length,
            processed_signal=processed_signal,
            processed_signal_length=processed_signal_length,
        )
        return logits

    def change_labels(self, new_labels: List[str]):
        """
        Changes labels used by the decoder model. Use this method when fine-tuning on from pre-trained model.
        This method changes only decoder and leaves encoder and pre-processing modules unchanged. For example, you would
        use it if you want to use pretrained encoder when fine-tuning on a data in another dataset.

        If new_labels == self.decoder.vocabulary then nothing will be changed.

        Args:

            new_labels: list with new labels. Must contain at least 2 elements. Typically, \
            this is set of labels for the dataset.

        Returns: None

        """
        if new_labels is not None and not isinstance(new_labels, ListConfig):
            new_labels = ListConfig(new_labels)

        if self._cfg.labels == new_labels:
            logging.warning(
                f"Old labels ({self._cfg.labels}) and new labels ({new_labels}) match. Not changing anything"
            )
        else:
            if new_labels is None or len(new_labels) == 0:
                raise ValueError(f'New labels must be non-empty list of labels. But I got: {new_labels}')

            # Update config
            self._cfg.labels = new_labels

            decoder_config = self.decoder.to_config_dict()
            new_decoder_config = copy.deepcopy(decoder_config)
            self._update_decoder_config(new_labels, new_decoder_config)
            del self.decoder
            self.decoder = EncDecClassificationModel.from_config_dict(new_decoder_config)

            OmegaConf.set_struct(self._cfg.decoder, False)
            self._cfg.decoder = new_decoder_config
            OmegaConf.set_struct(self._cfg.decoder, True)

            if 'train_ds' in self._cfg and self._cfg.train_ds is not None:
                self._cfg.train_ds.labels = new_labels

            if 'validation_ds' in self._cfg and self._cfg.validation_ds is not None:
                self._cfg.validation_ds.labels = new_labels

            if 'test_ds' in self._cfg and self._cfg.test_ds is not None:
                self._cfg.test_ds.labels = new_labels

            logging.info(f"Changed decoder output to {self.decoder.num_classes} labels.")

    def _update_decoder_config(self, labels, cfg):
        """
        Update the number of classes in the decoder based on labels provided.

        Args:
            labels: The current labels of the model
            cfg: The config of the decoder which will be updated.
        """
        OmegaConf.set_struct(cfg, False)

        if 'params' in cfg:
            cfg.params.num_classes = len(labels)
        else:
            cfg.num_classes = len(labels)

        OmegaConf.set_struct(cfg, True)


class EncDecRegressionModel(_EncDecBaseModel):
    """Encoder decoder class for speech regression models.
    Model class creates training, validation methods for setting up data
    performing model forward pass.
    """

    @classmethod
    def list_available_models(cls) -> List[PretrainedModelInfo]:
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.
        Returns:
            List of available pre-trained models.
        """
        result = []

        return result

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        if not cfg.get('is_regression_task', False):
            raise ValueError(f"EndDecRegressionModel requires the flag is_regression_task to be set as true")
        super().__init__(cfg=cfg, trainer=trainer)

    def _setup_preprocessor(self):
        return EncDecRegressionModel.from_config_dict(self._cfg.preprocessor)

    def _setup_encoder(self):
        return EncDecRegressionModel.from_config_dict(self._cfg.encoder)

    def _setup_decoder(self):
        return EncDecRegressionModel.from_config_dict(self._cfg.decoder)

    def _setup_loss(self):
        return MSELoss()

    def _setup_metrics(self):
        self._mse = MeanSquaredError()
        self._mae = MeanAbsoluteError()

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {"preds": NeuralType(tuple('B'), RegressionValuesType())}

    @typecheck()
    def forward(self, input_signal, input_signal_length):
        logits = super().forward(input_signal=input_signal, input_signal_length=input_signal_length)
        return logits.view(-1)

    # PTL-specific methods
    def training_step(self, batch, batch_idx):
        audio_signal, audio_signal_len, targets, targets_len = batch
        logits = self.forward(input_signal=audio_signal, input_signal_length=audio_signal_len)
        loss = self.loss(preds=logits, labels=targets)
        train_mse = self._mse(preds=logits, target=targets)
        train_mae = self._mae(preds=logits, target=targets)

        self.log_dict(
            {
                'train_loss': loss,
                'train_mse': train_mse,
                'train_mae': train_mae,
                'learning_rate': self._optimizer.param_groups[0]['lr'],
            },
        )

        return {'loss': loss}

    def validation_step(self, batch, batch_idx, dataloader_idx: int = 0):
        audio_signal, audio_signal_len, targets, targets_len = batch
        logits = self.forward(input_signal=audio_signal, input_signal_length=audio_signal_len)
        loss_value = self.loss(preds=logits, labels=targets)
        val_mse = self._mse(preds=logits, target=targets)
        val_mae = self._mae(preds=logits, target=targets)

        return {'val_loss': loss_value, 'val_mse': val_mse, 'val_mae': val_mae}

    def test_step(self, batch, batch_idx, dataloader_idx: int = 0):
        logs = self.validation_step(batch, batch_idx, dataloader_idx)

        return {'test_loss': logs['val_loss'], 'test_mse': logs['test_mse'], 'test_mae': logs['val_mae']}

    def multi_validation_epoch_end(self, outputs, dataloader_idx: int = 0):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        val_mse = self._mse.compute()
        self._mse.reset()
        val_mae = self._mae.compute()
        self._mae.reset()

        tensorboard_logs = {'val_loss': val_loss_mean, 'val_mse': val_mse, 'val_mae': val_mae}

        return {'val_loss': val_loss_mean, 'val_mse': val_mse, 'val_mae': val_mae, 'log': tensorboard_logs}

    def multi_test_epoch_end(self, outputs, dataloader_idx: int = 0):
        test_loss_mean = torch.stack([x['test_loss'] for x in outputs]).mean()
        test_mse = self._mse.compute()
        self._mse.reset()
        test_mae = self._mae.compute()
        self._mae.reset()

        tensorboard_logs = {'test_loss': test_loss_mean, 'test_mse': test_mse, 'test_mae': test_mae}

        return {'test_loss': test_loss_mean, 'test_mse': test_mse, 'test_mae': test_mae, 'log': tensorboard_logs}

    @torch.no_grad()
    def transcribe(self, paths2audio_files: List[str], batch_size: int = 4) -> List[float]:
        """
        Generate class labels for provided audio files. Use this method for debugging and prototyping.

        Args:
            paths2audio_files: (a list) of paths to audio files. \
                Recommended length per file is approximately 1 second.
            batch_size: (int) batch size to use during inference. \
                Bigger will result in better throughput performance but would use more memory.

        Returns:

            A list of predictions in the same order as paths2audio_files
        """
        predictions = super().transcribe(paths2audio_files, batch_size, logprobs=True)
        return [float(pred) for pred in predictions]

    def _update_decoder_config(self, labels, cfg):

        OmegaConf.set_struct(cfg, False)

        if 'params' in cfg:
            cfg.params.num_classes = 1
        else:
            cfg.num_classes = 1

        OmegaConf.set_struct(cfg, True)


class EncDecFrameClassificationModel(EncDecClassificationModel):
    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {"outputs": NeuralType(('B', 'T', 'C'), LogitsType())}

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        self.num_classes = len(cfg.labels)
        self.eval_loop_cnt = 0
        self.ratio_threshold = cfg.get('ratio_threshold', 0.2)
        super().__init__(cfg=cfg, trainer=trainer)
        self.decoder.output_types = self.output_types
        self.decoder.output_types_for_export = self.output_types

    @classmethod
    def list_available_models(cls) -> Optional[List[PretrainedModelInfo]]:
        results = []
        model = PretrainedModelInfo(
            pretrained_model_name="vad_multilingual_frame_marblenet",
            description="For details about this model, please visit https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/vad_multilingual_frame_marblenet",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/vad_multilingual_frame_marblenet/versions/1.20.0/files/vad_multilingual_frame_marblenet.nemo",
        )
        results.append(model)
        return results

    def _setup_metrics(self):
        self._accuracy = TopKClassificationAccuracy(dist_sync_on_step=True)
        self._macro_accuracy = Accuracy(num_classes=self.num_classes, average='macro', task="multiclass")

    def _setup_loss(self):
        if "loss" in self.cfg:
            weight = self.cfg.loss.get("weight", None)
            if weight in [None, "none", "None"]:
                weight = [1.0] * self.num_classes
            logging.info(f"Using cross-entropy with weights: {weight}")
        else:
            weight = [1.0] * self.num_classes
        return CrossEntropyLoss(logits_ndim=3, weight=weight)

    def _setup_dataloader_from_config(self, config: DictConfig):
        OmegaConf.set_struct(config, False)
        config.is_regression_task = self.is_regression_task
        OmegaConf.set_struct(config, True)
        shuffle = config.get('shuffle', False)

        if config.get('is_tarred', False):
            if ('tarred_audio_filepaths' in config and config['tarred_audio_filepaths'] is None) or (
                'manifest_filepath' in config and config['manifest_filepath'] is None
            ):
                raise ValueError(
                    "Could not load dataset as `manifest_filepath` is None or "
                    f"`tarred_audio_filepaths` is None. Provided cfg : {config}"
                )

            shuffle_n = config.get('shuffle_n', 4 * config['batch_size']) if shuffle else 0
            dataset = audio_to_label_dataset.get_tarred_audio_multi_label_dataset(
                cfg=config, shuffle_n=shuffle_n, global_rank=self.global_rank, world_size=self.world_size,
            )
            shuffle = False
            if hasattr(dataset, 'collate_fn'):
                collate_func = dataset.collate_fn
            else:
                collate_func = dataset.datasets[0].collate_fn
        else:
            if 'manifest_filepath' in config and config['manifest_filepath'] is None:
                raise ValueError(f"Could not load dataset as `manifest_filepath` is None. Provided cfg : {config}")
            dataset = audio_to_label_dataset.get_audio_multi_label_dataset(config)
            collate_func = dataset.collate_fn

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=config.get("batch_size", 1),
            collate_fn=collate_func,
            drop_last=config.get('drop_last', False),
            shuffle=shuffle,
            num_workers=config.get('num_workers', 0),
            pin_memory=config.get('pin_memory', False),
        )

    def _setup_feature_label_dataloader(self, config: DictConfig) -> torch.utils.data.DataLoader:
        """
        setup dataloader for VAD inference with audio features as input
        """

        OmegaConf.set_struct(config, False)
        config.is_regression_task = self.is_regression_task
        OmegaConf.set_struct(config, True)

        if 'augmentor' in config:
            augmentor = process_augmentations(config['augmentor'])
        else:
            augmentor = None
        if 'manifest_filepath' in config and config['manifest_filepath'] is None:
            logging.warning(f"Could not load dataset as `manifest_filepath` is None. Provided config : {config}")
            return None

        dataset = feature_to_label_dataset.get_feature_multi_label_dataset(config=config, augmentor=augmentor)

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=config.get("batch_size", 1),
            collate_fn=dataset.collate_fn,
            drop_last=config.get('drop_last', False),
            shuffle=config.get('shuffle', False),
            num_workers=config.get('num_workers', 0),
            pin_memory=config.get('pin_memory', False),
        )

    def get_label_masks(self, labels, labels_len):
        mask = torch.arange(labels.size(1))[None, :].to(labels.device) < labels_len[:, None]
        return mask.to(labels.device, dtype=bool)

    @typecheck()
    def forward(
        self, input_signal=None, input_signal_length=None, processed_signal=None, processed_signal_length=None
    ):
        has_input_signal = input_signal is not None and input_signal_length is not None
        has_processed_signal = processed_signal is not None and processed_signal_length is not None
        if (has_input_signal ^ has_processed_signal) == False:
            raise ValueError(
                f"{self} Arguments ``input_signal`` and ``input_signal_length`` are mutually exclusive "
                " with ``processed_signal`` and ``processed_signal_length`` arguments."
            )

        if not has_processed_signal:
            processed_signal, processed_signal_length = self.preprocessor(
                input_signal=input_signal, length=input_signal_length,
            )

        # Crop or pad is always applied
        if self.crop_or_pad is not None:
            processed_signal, processed_signal_length = self.crop_or_pad(
                input_signal=processed_signal, length=processed_signal_length
            )
        # Spec augment is not applied during evaluation/testing
        if self.spec_augmentation is not None and self.training:
            processed_signal = self.spec_augmentation(input_spec=processed_signal, length=processed_signal_length)
        encoded, encoded_len = self.encoder(audio_signal=processed_signal, length=processed_signal_length)
        logits = self.decoder(encoded.transpose(1, 2))
        return logits

    # PTL-specific methods
    def training_step(self, batch, batch_idx):
        audio_signal, audio_signal_len, labels, labels_len = batch
        logits = self.forward(input_signal=audio_signal, input_signal_length=audio_signal_len)
        labels, labels_len = self.reshape_labels(logits, labels, audio_signal_len, labels_len)
        masks = self.get_label_masks(labels, labels_len)

        loss_value = self.loss(logits=logits, labels=labels, loss_mask=masks)

        tensorboard_logs = {
            'train_loss': loss_value,
            'learning_rate': self._optimizer.param_groups[0]['lr'],
            'global_step': torch.tensor(self.trainer.global_step, dtype=torch.float32),
        }

        metric_logits, metric_labels = self.get_metric_logits_labels(logits, labels, masks)
        self._accuracy(logits=metric_logits, labels=metric_labels)
        topk_scores = self._accuracy.compute()
        self._accuracy.reset()

        for top_k, score in zip(self._accuracy.top_k, topk_scores):
            tensorboard_logs[f'training_batch_accuracy_top@{top_k}'] = score

        return {'loss': loss_value, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx, dataloader_idx: int = 0, tag: str = 'val'):
        audio_signal, audio_signal_len, labels, labels_len = batch
        logits = self.forward(input_signal=audio_signal, input_signal_length=audio_signal_len)
        labels, labels_len = self.reshape_labels(logits, labels, audio_signal_len, labels_len)
        masks = self.get_label_masks(labels, labels_len)

        loss_value = self.loss(logits=logits, labels=labels, loss_mask=masks)

        metric_logits, metric_labels = self.get_metric_logits_labels(logits, labels, masks)

        acc = self._accuracy(logits=metric_logits, labels=metric_labels)
        correct_counts, total_counts = self._accuracy.correct_counts_k, self._accuracy.total_counts_k

        self._macro_accuracy.update(preds=metric_logits, target=metric_labels)
        stats = self._macro_accuracy._final_state()

        return {
            f'{tag}_loss': loss_value,
            f'{tag}_correct_counts': correct_counts,
            f'{tag}_total_counts': total_counts,
            f'{tag}_acc_micro': acc,
            f'{tag}_acc_stats': stats,
        }

    def multi_validation_epoch_end(self, outputs, dataloader_idx: int = 0, tag: str = 'val'):
        val_loss_mean = torch.stack([x[f'{tag}_loss'] for x in outputs]).mean()
        correct_counts = torch.stack([x[f'{tag}_correct_counts'] for x in outputs]).sum(axis=0)
        total_counts = torch.stack([x[f'{tag}_total_counts'] for x in outputs]).sum(axis=0)

        self._accuracy.correct_counts_k = correct_counts
        self._accuracy.total_counts_k = total_counts
        topk_scores = self._accuracy.compute()

        self._macro_accuracy.tp = torch.stack([x[f'{tag}_acc_stats'][0] for x in outputs]).sum(axis=0)
        self._macro_accuracy.fp = torch.stack([x[f'{tag}_acc_stats'][1] for x in outputs]).sum(axis=0)
        self._macro_accuracy.tn = torch.stack([x[f'{tag}_acc_stats'][2] for x in outputs]).sum(axis=0)
        self._macro_accuracy.fn = torch.stack([x[f'{tag}_acc_stats'][3] for x in outputs]).sum(axis=0)
        macro_accuracy_score = self._macro_accuracy.compute()

        self._accuracy.reset()
        self._macro_accuracy.reset()

        tensorboard_log = {
            f'{tag}_loss': val_loss_mean,
            f'{tag}_acc_macro': macro_accuracy_score,
        }

        for top_k, score in zip(self._accuracy.top_k, topk_scores):
            tensorboard_log[f'{tag}_acc_micro_top@{top_k}'] = score

        self.log_dict(tensorboard_log, sync_dist=True)
        return tensorboard_log

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self.validation_step(batch, batch_idx, dataloader_idx, tag='test')

    def multi_test_epoch_end(self, outputs, dataloader_idx: int = 0):
        return self.multi_validation_epoch_end(outputs, dataloader_idx, tag='test')

    def reshape_labels(self, logits, labels, logits_len, labels_len):
        """
        Reshape labels to match logits shape. For example, each label is expected to cover a 40ms frame, while each frme prediction from the
        model covers 20ms. If labels are shorter than logits, labels are repeated, otherwise labels are folded and argmax is applied to obtain 
        the label of each frame. When lengths of labels and logits are not factors of each other, labels are truncated or padded with zeros.
        The ratio_threshold=0.2 is used to determine whether to pad or truncate labels, where the value 0.2 is not important as in real cases the ratio
        is very close to either ceil(ratio) or floor(ratio). We use 0.2 here for easier unit-testing. This implementation does not allow frame length
        and label length that are not multiples of each other.
        Args:
            logits: logits tensor with shape [B, T1, C]
            labels: labels tensor with shape [B, T2]
            logits_len: logits length tensor with shape [B]
            labels_len: labels length tensor with shape [B]
        Returns:
            labels: labels tensor with shape [B, T1]
            labels_len: labels length tensor with shape [B]
        """
        logits_max_len = logits.size(1)
        labels_max_len = labels.size(1)
        batch_size = logits.size(0)
        if logits_max_len < labels_max_len:
            ratio = labels_max_len // logits_max_len
            res = labels_max_len % logits_max_len
            if ceil(ratio) - ratio < self.ratio_threshold:  # e.g., ratio is 1.99
                # pad labels with zeros until labels_max_len is a multiple of logits_max_len
                labels = labels.cpu().tolist()
                if len(labels) % ceil(ratio) != 0:
                    labels += [0] * (ceil(ratio) - len(labels) % ceil(ratio))
                labels = torch.tensor(labels).long().to(logits.device)
                labels = labels.view(-1, ceil(ratio)).amax(1)
                return self.reshape_labels(logits, labels, logits_len, labels_len)
            else:
                # truncate additional labels until labels_max_len is a multiple of logits_max_len
                if res > 0:
                    labels = labels[:, :-res]
                    mask = labels_len > (labels_max_len - res)
                    labels_len = labels_len - mask * (labels_len - (labels_max_len - res))
                labels = labels.view(batch_size, ratio, -1).amax(1)
                labels_len = torch.div(labels_len, ratio, rounding_mode="floor")
                labels_len = torch.min(torch.cat([logits_len[:, None], labels_len[:, None]], dim=1), dim=1)[0]
                return labels.contiguous(), labels_len.contiguous()
        elif logits_max_len > labels_max_len:
            ratio = logits_max_len / labels_max_len
            res = logits_max_len % labels_max_len
            if ceil(ratio) - ratio < self.ratio_threshold:  # e.g., ratio is 1.99
                # repeat labels for ceil(ratio) times, and DROP additional labels based on logits_max_len
                labels = labels.repeat_interleave(ceil(ratio), dim=1).long()
                labels = labels[:, :logits_max_len]
                labels_len = labels_len * ceil(ratio)
                mask = labels_len > logits_max_len
                labels_len = labels_len - mask * (labels_len - logits_max_len)
            else:  # e.g., ratio is 2.01
                # repeat labels for floor(ratio) times, and ADD padding labels based on logits_max_len
                labels = labels.repeat_interleave(floor(ratio), dim=1).long()
                labels_len = labels_len * floor(ratio)
                if res > 0:
                    labels = torch.cat([labels, labels[:, -res:]], dim=1)
                    # no need to update `labels_len` since we ignore additional "res" padded labels
            labels_len = torch.min(torch.cat([logits_len[:, None], labels_len[:, None]], dim=1), dim=1)[0]
            return labels.contiguous(), labels_len.contiguous()
        else:
            labels_len = torch.min(torch.cat([logits_len[:, None], labels_len[:, None]], dim=1), dim=1)[0]
            return labels, labels_len

    def get_metric_logits_labels(self, logits, labels, masks):
        """
        Computes valid logits and labels for metric computation.
        Args:
           logits: tensor of shape [B, T, C]
           labels: tensor of shape [B, T]
           masks: tensor of shape [B, T]
        Returns:
           logits of shape [N, C]
           labels of shape [N,]
        """
        C = logits.size(2)
        logits = logits.view(-1, C)  # [BxT, C]
        labels = labels.view(-1).contiguous()  # [BxT,]
        masks = masks.view(-1)  # [BxT,]
        idx = masks.nonzero()  # [BxT, 1]

        logits = logits.gather(dim=0, index=idx.repeat(1, 2))
        labels = labels.gather(dim=0, index=idx.view(-1))

        return logits, labels

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
        ret = dec_fun(hidden_states=encoder_output.transpose(1, 2))
        if isinstance(ret, tuple):
            ret = ret[0]
        if cache_last_channel is not None:
            ret = (ret, length, cache_last_channel, cache_last_time, cache_last_channel_len)
        return cast_all(ret, from_dtype=torch.float16, to_dtype=torch.float32)
