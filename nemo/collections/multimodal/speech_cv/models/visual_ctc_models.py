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

import copy
import json
import os
import tempfile
from math import ceil
from typing import Dict, List, Optional, Union

import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning import Trainer
from tqdm.auto import tqdm

from nemo.collections.asr.data import audio_to_text_dataset
from nemo.collections.asr.losses.ctc import CTCLoss
from nemo.collections.asr.metrics.wer import WER, CTCDecoding, CTCDecodingConfig
from nemo.collections.asr.models.asr_model import ASRModel, ExportableEncDecModel
from nemo.collections.asr.parts.mixins import ASRModuleMixin, InterCTCMixin
from nemo.collections.asr.parts.utils.audio_utils import ChannelSelectorType
from nemo.collections.multimodal.speech_cv.data import video_to_text_dataset
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.classes.mixins import AccessMixin
from nemo.core.neural_types import LabelsType, LengthsType, LogprobsType, NeuralType, VideoSignal
from nemo.utils import logging

__all__ = ['VisualEncDecCTCModel']


class VisualEncDecCTCModel(ASRModel, ExportableEncDecModel, ASRModuleMixin, InterCTCMixin):
    """Base class for encoder decoder CTC-based models."""

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):

        # Get global rank and total number of GPU workers for IterableDataset partitioning, if applicable
        # Global_rank and local_rank is set by LightningModule in Lightning 1.2.0
        self.world_size = 1
        if trainer is not None:
            self.world_size = trainer.world_size

        # Init
        super().__init__(cfg=cfg, trainer=trainer)

        # Preprocessor, video transforms
        self.video_preprocessor = VisualEncDecCTCModel.from_config_dict(self._cfg.video_preprocessor)

        # Augmentation, video augmentations
        self.video_augmentation = VisualEncDecCTCModel.from_config_dict(self._cfg.video_augment)

        # Front-end Network, learned module that transform videos to temporal sequence
        self.video_front_end = VisualEncDecCTCModel.from_config_dict(self._cfg.video_front_end)

        # Encoder Network
        self.encoder = VisualEncDecCTCModel.from_config_dict(self._cfg.encoder)

        with open_dict(self._cfg):
            if "feat_in" not in self._cfg.decoder or (
                not self._cfg.decoder.feat_in and hasattr(self.encoder, '_feat_out')
            ):
                self._cfg.decoder.feat_in = self.encoder._feat_out
            if "feat_in" not in self._cfg.decoder or not self._cfg.decoder.feat_in:
                raise ValueError("param feat_in of the decoder's config is not set!")

            if self.cfg.decoder.num_classes < 1 and self.cfg.decoder.vocabulary is not None:
                logging.info(
                    "\nReplacing placeholder number of classes ({}) with actual number of classes - {}".format(
                        self.cfg.decoder.num_classes, len(self.cfg.decoder.vocabulary)
                    )
                )
                cfg.decoder["num_classes"] = len(self.cfg.decoder.vocabulary)

        # Decoder
        self.decoder = VisualEncDecCTCModel.from_config_dict(self._cfg.decoder)

        # CTC Loss
        self.loss = CTCLoss(
            num_classes=self.decoder.num_classes_with_blank - 1,
            zero_infinity=True,
            reduction=self._cfg.get("ctc_reduction", "mean_batch"),
        )

        # Setup decoding objects
        decoding_cfg = self.cfg.get('decoding', None)

        # In case decoding config not found, use default config
        if decoding_cfg is None:
            decoding_cfg = OmegaConf.structured(CTCDecodingConfig)
            with open_dict(self.cfg):
                self.cfg.decoding = decoding_cfg

        # Decoding
        self.decoding = CTCDecoding(self.cfg.decoding, vocabulary=self.decoder.vocabulary)

        # Setup metric with decoding strategy
        self._wer = WER(
            decoding=self.decoding,
            use_cer=self._cfg.get('use_cer', False),
            dist_sync_on_step=True,
            log_prediction=self._cfg.get("log_prediction", False),
        )

        # Setup optional Optimization flags
        self.setup_optimization_flags()

        # setting up interCTC loss (from InterCTCMixin)
        self.setup_interctc(decoder_name='decoder', loss_name='loss', wer_name='_wer')

        # Adapter modules setup (from ASRAdapterModelMixin)
        self.setup_adapters()

    @torch.no_grad()
    def transcribe(
        self,
        paths2video_files: List[str],
        batch_size: int = 4,
        logprobs: bool = False,
        return_hypotheses: bool = False,
        num_workers: int = 0,
        channel_selector: Optional[ChannelSelectorType] = None,
        augmentor: DictConfig = None,
    ) -> List[str]:
        """
        If modify this function, please remember update transcribe_partial_audio() in
        nemo/collections/asr/parts/utils/trancribe_utils.py

        Uses greedy decoding to transcribe video files. Use this method for debugging and prototyping.

        Args:
            paths2video_files: (a list) of paths to video files.
            batch_size: (int) batch size to use during inference.
                Bigger will result in better throughput performance but would use more memory.
            logprobs: (bool) pass True to get log probabilities instead of transcripts.
            return_hypotheses: (bool) Either return hypotheses or text
                With hypotheses can do some postprocessing like getting timestamp or rescoring
            num_workers: (int) number of workers for DataLoader
            channel_selector (int | Iterable[int] | str): select a single channel or a subset of channels from multi-channel audio. If set to `'average'`, it performs averaging across channels. Disabled if set to `None`. Defaults to `None`.
            augmentor: (DictConfig): Augment audio samples during transcription if augmentor is applied.
        Returns:
            A list of transcriptions (or raw log probabilities if logprobs is True) in the same order as paths2video_files
        """
        if paths2video_files is None or len(paths2video_files) == 0:
            return {}

        if return_hypotheses and logprobs:
            raise ValueError(
                "Either `return_hypotheses` or `logprobs` can be True at any given time."
                "Returned hypotheses will contain the logprobs."
            )

        if num_workers is None:
            num_workers = min(batch_size, os.cpu_count() - 1)

        # We will store transcriptions here
        hypotheses = []
        all_hypotheses = []

        # Model's mode and device
        mode = self.training
        device = next(self.parameters()).device

        try:
            # Switch model to evaluation mode
            self.eval()
            # Freeze the visual front-end, encoder and decoder modules
            self.video_front_end.freeze()
            self.encoder.freeze()
            self.decoder.freeze()
            logging_level = logging.get_verbosity()
            logging.set_verbosity(logging.WARNING)
            # Work in tmp directory - will store manifest file there
            with tempfile.TemporaryDirectory() as tmpdir:
                with open(os.path.join(tmpdir, 'manifest.json'), 'w', encoding='utf-8') as fp:
                    for video_file in paths2video_files:
                        entry = {'video_filepath': video_file, 'duration': 100000, 'text': ''}
                        fp.write(json.dumps(entry) + '\n')

                config = {
                    'paths2video_files': paths2video_files,
                    'batch_size': batch_size,
                    'temp_dir': tmpdir,
                    'num_workers': num_workers,
                    'channel_selector': channel_selector,
                }

                if augmentor:
                    config['augmentor'] = augmentor

                temporary_datalayer = self._setup_transcribe_dataloader(config)
                for test_batch in tqdm(temporary_datalayer, desc="Transcribing"):
                    logits, logits_len, greedy_predictions = self.forward(
                        input_signal=test_batch[0].to(device), input_signal_length=test_batch[1].to(device)
                    )
                    if logprobs:
                        # dump log probs per file
                        for idx in range(logits.shape[0]):
                            lg = logits[idx][: logits_len[idx]]
                            hypotheses.append(lg.cpu().numpy())
                    else:
                        current_hypotheses, all_hyp = self.decoding.ctc_decoder_predictions_tensor(
                            logits, decoder_lengths=logits_len, return_hypotheses=return_hypotheses,
                        )

                        if return_hypotheses:
                            # dump log probs per file
                            for idx in range(logits.shape[0]):
                                current_hypotheses[idx].y_sequence = logits[idx][: logits_len[idx]]
                                if current_hypotheses[idx].alignments is None:
                                    current_hypotheses[idx].alignments = current_hypotheses[idx].y_sequence

                        if all_hyp is None:
                            hypotheses += current_hypotheses
                        else:
                            hypotheses += all_hyp

                    del greedy_predictions
                    del logits
                    del test_batch
        finally:
            # set mode back to its original value
            self.train(mode=mode)
            if mode is True:
                self.video_front_end.unfreeze()
                self.encoder.unfreeze()
                self.decoder.unfreeze()
            logging.set_verbosity(logging_level)

        return hypotheses

    def change_vocabulary(self, new_vocabulary: List[str], decoding_cfg: Optional[DictConfig] = None):
        """
        Changes vocabulary used during CTC decoding process. Use this method when fine-tuning on from pre-trained model.
        This method changes only decoder and leaves encoder and pre-processing modules unchanged. For example, you would
        use it if you want to use pretrained encoder when fine-tuning on a data in another language, or when you'd need
        model to learn capitalization, punctuation and/or special characters.

        If new_vocabulary == self.decoder.vocabulary then nothing will be changed.

        Args:

            new_vocabulary: list with new vocabulary. Must contain at least 2 elements. Typically, \
            this is target alphabet.

        Returns: None

        """
        if self.decoder.vocabulary == new_vocabulary:
            logging.warning(f"Old {self.decoder.vocabulary} and new {new_vocabulary} match. Not changing anything.")
        else:
            if new_vocabulary is None or len(new_vocabulary) == 0:
                raise ValueError(f'New vocabulary must be non-empty list of chars. But I got: {new_vocabulary}')
            decoder_config = self.decoder.to_config_dict()
            new_decoder_config = copy.deepcopy(decoder_config)
            new_decoder_config['vocabulary'] = new_vocabulary
            new_decoder_config['num_classes'] = len(new_vocabulary)

            del self.decoder
            self.decoder = VisualEncDecCTCModel.from_config_dict(new_decoder_config)
            del self.loss
            self.loss = CTCLoss(
                num_classes=self.decoder.num_classes_with_blank - 1,
                zero_infinity=True,
                reduction=self._cfg.get("ctc_reduction", "mean_batch"),
            )

            if decoding_cfg is None:
                # Assume same decoding config as before
                decoding_cfg = self.cfg.decoding

            # Assert the decoding config with all hyper parameters
            decoding_cls = OmegaConf.structured(CTCDecodingConfig)
            decoding_cls = OmegaConf.create(OmegaConf.to_container(decoding_cls))
            decoding_cfg = OmegaConf.merge(decoding_cls, decoding_cfg)

            self.decoding = CTCDecoding(decoding_cfg=decoding_cfg, vocabulary=self.decoder.vocabulary)

            self._wer = WER(
                decoding=self.decoding,
                use_cer=self._cfg.get('use_cer', False),
                dist_sync_on_step=True,
                log_prediction=self._cfg.get("log_prediction", False),
            )

            # Update config
            with open_dict(self.cfg.decoder):
                self._cfg.decoder = new_decoder_config

            with open_dict(self.cfg.decoding):
                self.cfg.decoding = decoding_cfg

            ds_keys = ['train_ds', 'validation_ds', 'test_ds']
            for key in ds_keys:
                if key in self.cfg:
                    with open_dict(self.cfg[key]):
                        self.cfg[key]['labels'] = OmegaConf.create(new_vocabulary)

            logging.info(f"Changed decoder to output to {self.decoder.vocabulary} vocabulary.")

    def change_decoding_strategy(self, decoding_cfg: DictConfig):
        """
        Changes decoding strategy used during CTC decoding process.

        Args:
            decoding_cfg: A config for the decoder, which is optional. If the decoding type
                needs to be changed (from say Greedy to Beam decoding etc), the config can be passed here.
        """
        if decoding_cfg is None:
            # Assume same decoding config as before
            logging.info("No `decoding_cfg` passed when changing decoding strategy, using internal config")
            decoding_cfg = self.cfg.decoding

        # Assert the decoding config with all hyper parameters
        decoding_cls = OmegaConf.structured(CTCDecodingConfig)
        decoding_cls = OmegaConf.create(OmegaConf.to_container(decoding_cls))
        decoding_cfg = OmegaConf.merge(decoding_cls, decoding_cfg)

        self.decoding = CTCDecoding(decoding_cfg=decoding_cfg, vocabulary=self.decoder.vocabulary)

        self._wer = WER(
            decoding=self.decoding,
            use_cer=self._wer.use_cer,
            log_prediction=self._wer.log_prediction,
            dist_sync_on_step=True,
        )

        # Update config
        with open_dict(self.cfg.decoding):
            self.cfg.decoding = decoding_cfg

        logging.info(f"Changed decoding strategy to \n{OmegaConf.to_yaml(self.cfg.decoding)}")

    def _setup_dataloader_from_config(self, config: Optional[Dict]):
        # Automatically inject args from model config to dataloader config
        audio_to_text_dataset.inject_dataloader_value_from_model_config(self.cfg, config, key='sample_rate')
        audio_to_text_dataset.inject_dataloader_value_from_model_config(self.cfg, config, key='labels')
        dataset = video_to_text_dataset.get_video_to_text_char_dataset_from_config(
            config=config,
            local_rank=self.local_rank,
            global_rank=self.global_rank,
            world_size=self.world_size,
            preprocessor_cfg=self._cfg.get("preprocessor", None),
        )

        if dataset is None:
            return None

        shuffle = config['shuffle']
        if config.get('is_tarred', False):
            shuffle = False

        if hasattr(dataset, 'collate_fn'):
            collate_fn = dataset.collate_fn
        else:
            collate_fn = dataset.datasets[0].collate_fn

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
            -   :class:`~nemo.collections.multimodal.speech_cv.data.video_to_text.VideoToCharDataset`
            -   :class:`~nemo.collections.asr.data.video_to_text.VideoToBPEDataset`
            -   :class:`~nemo.collections.asr.data.video_to_text.TarredVideoToBPEDataset`
        """
        if 'shuffle' not in train_data_config:
            train_data_config['shuffle'] = True

        # preserve config
        self._update_dataset_config(dataset_name='train', config=train_data_config)

        self._train_dl = self._setup_dataloader_from_config(config=train_data_config)

        # Need to set this because if using an IterableDataset, the length of the dataloader is the total number
        # of samples rather than the number of batches, and this messes up the tqdm progress bar.
        # So we set the number of steps manually (to the correct number) to fix this.
        if 'is_tarred' in train_data_config and train_data_config['is_tarred']:
            # We also need to check if limit_train_batches is already set.
            # If it's an int, we assume that the user has set it to something sane, i.e. <= # training batches,
            # and don't change it. Otherwise, adjust batches accordingly if it's a float (including 1.0).
            if self._trainer is not None and isinstance(self._trainer.limit_train_batches, float):
                self._trainer.limit_train_batches = int(
                    self._trainer.limit_train_batches
                    * ceil((len(self._train_dl.dataset) / self.world_size) / train_data_config['batch_size'])
                )
            elif self._trainer is None:
                logging.warning(
                    "Model Trainer was not set before constructing the dataset, incorrect number of "
                    "training batches will be used. Please set the trainer and rebuild the dataset."
                )

    def setup_validation_data(self, val_data_config: Optional[Union[DictConfig, Dict]]):
        """
        Sets up the validation data loader via a Dict-like object.

        Args:
            val_data_config: A config that contains the information regarding construction
                of an ASR Training dataset.

        Supported Datasets:
            -   :class:`~nemo.collections.multimodal.speech_cv.data.video_to_text.VideoToCharDataset`
            -   :class:`~nemo.collections.asr.data.video_to_text.VideoToBPEDataset`
            -   :class:`~nemo.collections.asr.data.video_to_text.TarredVideoToBPEDataset`
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
                of an ASR Training dataset.

        Supported Datasets:
            -   :class:`~nemo.collections.multimodal.speech_cv.data.video_to_text.VideoToCharDataset`
            -   :class:`~nemo.collections.asr.data.video_to_text.VideoToBPEDataset`
            -   :class:`~nemo.collections.asr.data.video_to_text.TarredVideoToBPEDataset`
        """
        if 'shuffle' not in test_data_config:
            test_data_config['shuffle'] = False

        # preserve config
        self._update_dataset_config(dataset_name='test', config=test_data_config)

        self._test_dl = self._setup_dataloader_from_config(config=test_data_config)

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "input_video_signal": NeuralType(('B', 'C', 'T', 'H', 'W'), VideoSignal(), optional=True),
            "input_video_signal_length": NeuralType(tuple('B'), LengthsType(), optional=True),
            "sample_id": NeuralType(tuple('B'), LengthsType(), optional=True),
        }

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "outputs": NeuralType(('B', 'T', 'D'), LogprobsType()),
            "encoded_lengths": NeuralType(tuple('B'), LengthsType()),
            "greedy_predictions": NeuralType(('B', 'T'), LabelsType()),
        }

    @typecheck()
    def forward(self, input_video_signal=None, input_video_signal_length=None):
        """
        Forward pass of the model.

        Args:
            input_video_signal: Tensor that represents a batch of video signals,
                of shape [B, T, H, W, C]. T here represents timesteps, H height, W width and C channels
            input_video_signal_length: Vector of length B, that contains the individual lengths of the video
                sequences.

        Returns:
            A tuple of 3 elements -
            1) The log probabilities tensor of shape [B, T, D].
            2) The lengths of the acoustic sequence after propagation through the encoder, of shape [B].
            3) The greedy token predictions of the model of shape [B, T] (via argmax)
        """

        # Preprocessing
        processed_video_signal, processed_video_signal_length = self.video_preprocessor(
            input_signal=input_video_signal, length=input_video_signal_length
        )

        # Augmentation
        processed_video_signal = self.video_augmentation(
            input_signal=processed_video_signal, length=processed_video_signal_length
        )

        # Front-end Networks
        processed_video_signal, processed_video_signal_length = self.video_front_end(
            input_signal=processed_video_signal, length=processed_video_signal_length
        )

        # Back-end Networks
        encoded, encoded_len = self.encoder(audio_signal=processed_video_signal, length=processed_video_signal_length)

        log_probs = self.decoder(encoder_output=encoded)
        greedy_predictions = log_probs.argmax(dim=-1, keepdim=False)

        return (
            log_probs,
            encoded_len,
            greedy_predictions,
        )

    # PTL-specific methods
    def training_step(self, batch, batch_nb):
        # Reset access registry
        if AccessMixin.is_access_enabled():
            AccessMixin.reset_registry(self)

        if self.is_interctc_enabled():
            AccessMixin.set_access_enabled(access_enabled=True)

        video_signal, video_signal_len, transcript, transcript_len = batch
        log_probs, encoded_len, predictions = self.forward(
            input_video_signal=video_signal, input_video_signal_length=video_signal_len
        )

        if hasattr(self, '_trainer') and self._trainer is not None:
            log_every_n_steps = self._trainer.log_every_n_steps
        else:
            log_every_n_steps = 1

        loss_value = self.loss(
            log_probs=log_probs, targets=transcript, input_lengths=encoded_len, target_lengths=transcript_len
        )

        # Add auxiliary losses, if registered
        loss_value = self.add_auxiliary_losses(loss_value)
        # only computing WER when requested in the logs (same as done for final-layer WER below)
        loss_value, tensorboard_logs = self.add_interctc_losses(
            loss_value, transcript, transcript_len, compute_wer=((batch_nb + 1) % log_every_n_steps == 0)
        )

        # Reset access registry
        if AccessMixin.is_access_enabled():
            AccessMixin.reset_registry(self)

        tensorboard_logs.update(
            {
                'train_loss': loss_value,
                'learning_rate': self._optimizer.param_groups[0]['lr'],
                'global_step': torch.tensor(self.trainer.global_step, dtype=torch.float32),
            }
        )

        if (batch_nb + 1) % log_every_n_steps == 0:
            self._wer.update(
                predictions=log_probs,
                targets=transcript,
                target_lengths=transcript_len,
                predictions_lengths=encoded_len,
            )
            wer, _, _ = self._wer.compute()
            self._wer.reset()
            tensorboard_logs.update({'training_batch_wer': wer})

        return {'loss': loss_value, 'log': tensorboard_logs}

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        video_signal, video_signal_len, transcript, transcript_len, sample_id = batch
        log_probs, encoded_len, predictions = self.forward(
            input_video_signal=video_signal, input_video_signal_length=video_signal_len
        )

        transcribed_texts, _ = self._wer.decoding.ctc_decoder_predictions_tensor(
            decoder_outputs=log_probs, decoder_lengths=encoded_len, return_hypotheses=False,
        )

        sample_id = sample_id.cpu().detach().numpy()
        return list(zip(sample_id, transcribed_texts))

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if self.is_interctc_enabled():
            AccessMixin.set_access_enabled(access_enabled=True)

        video_signal, video_signal_len, transcript, transcript_len = batch
        log_probs, encoded_len, predictions = self.forward(
            input_video_signal=video_signal, input_video_signal_length=video_signal_len
        )

        loss_value = self.loss(
            log_probs=log_probs, targets=transcript, input_lengths=encoded_len, target_lengths=transcript_len
        )
        loss_value, metrics = self.add_interctc_losses(
            loss_value, transcript, transcript_len, compute_wer=True, log_wer_num_denom=True, log_prefix="val_",
        )

        self._wer.update(
            predictions=log_probs, targets=transcript, target_lengths=transcript_len, predictions_lengths=encoded_len
        )
        wer, wer_num, wer_denom = self._wer.compute()
        self._wer.reset()
        metrics.update({'val_loss': loss_value, 'val_wer_num': wer_num, 'val_wer_denom': wer_denom, 'val_wer': wer})

        self.log('global_step', torch.tensor(self.trainer.global_step, dtype=torch.float32))

        # Reset access registry
        if AccessMixin.is_access_enabled():
            AccessMixin.reset_registry(self)

        return metrics

    def multi_validation_epoch_end(self, outputs, dataloader_idx: int = 0):
        metrics = super().multi_validation_epoch_end(outputs, dataloader_idx)
        self.finalize_interctc_metrics(metrics, outputs, prefix="val_")
        return metrics

    def multi_test_epoch_end(self, outputs, dataloader_idx: int = 0):
        metrics = super().multi_test_epoch_end(outputs, dataloader_idx)
        self.finalize_interctc_metrics(metrics, outputs, prefix="test_")
        return metrics

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        logs = self.validation_step(batch, batch_idx, dataloader_idx=dataloader_idx)
        test_logs = {name.replace("val_", "test_"): value for name, value in logs.items()}
        if type(self.trainer.test_dataloaders) == list and len(self.trainer.test_dataloaders) > 1:
            self.test_step_outputs[dataloader_idx].append(test_logs)
        else:
            self.test_step_outputs.append(test_logs)
        return test_logs

    def test_dataloader(self):
        if self._test_dl is not None:
            return self._test_dl

    def _setup_transcribe_dataloader(self, config: Dict) -> 'torch.utils.data.DataLoader':
        """
        Setup function for a temporary data loader which wraps the provided video file.

        Args:
            config: A python dictionary which contains the following keys:
            paths2video_files: (a list) of paths to video files. The files should be relatively short fragments. \
                Recommended length per file is between 5 and 25 seconds.
            batch_size: (int) batch size to use during inference. \
                Bigger will result in better throughput performance but would use more memory.
            temp_dir: (str) A temporary directory where the video manifest is temporarily
                stored.
            num_workers: (int) number of workers. Depends of the batch_size and machine. \
                0 - only the main process will load batches, 1 - one worker (not main process)

        Returns:
            A pytorch DataLoader for the given video file(s).
        """
        if 'manifest_filepath' in config:
            manifest_filepath = config['manifest_filepath']
            batch_size = config['batch_size']
        else:
            manifest_filepath = os.path.join(config['temp_dir'], 'manifest.json')
            batch_size = min(config['batch_size'], len(config['paths2video_files']))

        dl_config = {
            'manifest_filepath': manifest_filepath,
            'labels': self.decoder.vocabulary,
            'batch_size': batch_size,
            'trim_silence': False,
            'shuffle': False,
            'num_workers': config.get('num_workers', min(batch_size, os.cpu_count() - 1)),
            'pin_memory': True,
            'channel_selector': config.get('channel_selector', None),
        }
        if config.get("augmentor"):
            dl_config['augmentor'] = config.get("augmentor")

        temporary_datalayer = self._setup_dataloader_from_config(config=DictConfig(dl_config))
        return temporary_datalayer

    @classmethod
    def list_available_models(cls) -> List[PretrainedModelInfo]:
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.

        Returns:
            List of available pre-trained models.
        """
        results = []

        return results
