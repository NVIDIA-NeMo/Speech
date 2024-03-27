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
from math import ceil
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning import Trainer
from tqdm.auto import tqdm

from nemo.collections.asr.data import audio_to_text_dataset
from nemo.collections.asr.data.audio_to_text_dali import AudioToCharDALIDataset, DALIOutputs
from nemo.collections.asr.data.audio_to_text_lhotse import LhotseSpeechToTextBpeDataset
from nemo.collections.asr.losses.rnnt import RNNTLoss, resolve_rnnt_default_loss_name
from nemo.collections.asr.metrics.wer import WER
from nemo.collections.asr.models.asr_model import ASRModel, ExportableEncDecModel
from nemo.collections.asr.modules.rnnt import RNNTDecoderJoint
from nemo.collections.asr.parts.mixins import (
    ASRModuleMixin,
    ASRTranscriptionMixin,
    TranscribeConfig,
    TranscriptionReturnType,
)
from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTDecoding, RNNTDecodingConfig
from nemo.collections.asr.parts.utils.audio_utils import ChannelSelectorType
from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config
from nemo.collections.common.parts.preprocessing.parsers import make_parser
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.classes.mixins import AccessMixin
from nemo.core.neural_types import AcousticEncodedRepresentation, AudioSignal, LengthsType, NeuralType, SpectrogramType
from nemo.utils import logging


class EncDecRNNTModel(ASRModel, ASRModuleMixin, ExportableEncDecModel, ASRTranscriptionMixin):
    """Base class for encoder decoder RNNT-based models."""

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        # Get global rank and total number of GPU workers for IterableDataset partitioning, if applicable
        # Global_rank and local_rank is set by LightningModule in Lightning 1.2.0
        self.world_size = 1
        if trainer is not None:
            self.world_size = trainer.world_size

        super().__init__(cfg=cfg, trainer=trainer)

        # Initialize components
        self.preprocessor = EncDecRNNTModel.from_config_dict(self.cfg.preprocessor)
        self.encoder = EncDecRNNTModel.from_config_dict(self.cfg.encoder)

        # Update config values required by components dynamically
        with open_dict(self.cfg.decoder):
            self.cfg.decoder.vocab_size = len(self.cfg.labels)

        with open_dict(self.cfg.joint):
            self.cfg.joint.num_classes = len(self.cfg.labels)
            self.cfg.joint.vocabulary = self.cfg.labels
            self.cfg.joint.jointnet.encoder_hidden = self.cfg.model_defaults.enc_hidden
            self.cfg.joint.jointnet.pred_hidden = self.cfg.model_defaults.pred_hidden

        self.decoder = EncDecRNNTModel.from_config_dict(self.cfg.decoder)
        self.joint = EncDecRNNTModel.from_config_dict(self.cfg.joint)

        # Setup RNNT Loss
        loss_name, loss_kwargs = self.extract_rnnt_loss_cfg(self.cfg.get("loss", None))

        num_classes = self.joint.num_classes_with_blank - 1  # for standard RNNT and multi-blank

        if loss_name == 'tdt':
            num_classes = num_classes - self.joint.num_extra_outputs

        self.loss = RNNTLoss(
            num_classes=num_classes,
            loss_name=loss_name,
            loss_kwargs=loss_kwargs,
            reduction=self.cfg.get("rnnt_reduction", "mean_batch"),
        )

        if hasattr(self.cfg, 'spec_augment') and self._cfg.spec_augment is not None:
            self.spec_augmentation = EncDecRNNTModel.from_config_dict(self.cfg.spec_augment)
        else:
            self.spec_augmentation = None

        self.cfg.decoding = self.set_decoding_type_according_to_loss(self.cfg.decoding)
        # Setup decoding objects
        self.decoding = RNNTDecoding(
            decoding_cfg=self.cfg.decoding, decoder=self.decoder, joint=self.joint, vocabulary=self.joint.vocabulary,
        )
        # Setup WER calculation
        self.wer = WER(
            decoding=self.decoding,
            batch_dim_index=0,
            use_cer=self._cfg.get('use_cer', False),
            log_prediction=self._cfg.get('log_prediction', True),
            dist_sync_on_step=True,
        )

        # Whether to compute loss during evaluation
        if 'compute_eval_loss' in self.cfg:
            self.compute_eval_loss = self.cfg.compute_eval_loss
        else:
            self.compute_eval_loss = True

        # Setup fused Joint step if flag is set
        if self.joint.fuse_loss_wer or (
            self.decoding.joint_fused_batch_size is not None and self.decoding.joint_fused_batch_size > 0
        ):
            self.joint.set_loss(self.loss)
            self.joint.set_wer(self.wer)

        # Setup optimization normalization (if provided in config)
        self.setup_optim_normalization()

        # Setup optional Optimization flags
        self.setup_optimization_flags()

        # Setup encoder adapters (from ASRAdapterModelMixin)
        self.setup_adapters()

    def setup_optim_normalization(self):
        """
        Helper method to setup normalization of certain parts of the model prior to the optimization step.

        Supported pre-optimization normalizations are as follows:

        .. code-block:: yaml

            # Variation Noise injection
            model:
                variational_noise:
                    std: 0.0
                    start_step: 0

            # Joint - Length normalization
            model:
                normalize_joint_txu: false

            # Encoder Network - gradient normalization
            model:
                normalize_encoder_norm: false

            # Decoder / Prediction Network - gradient normalization
            model:
                normalize_decoder_norm: false

            # Joint - gradient normalization
            model:
                normalize_joint_norm: false
        """
        # setting up the variational noise for the decoder
        if hasattr(self.cfg, 'variational_noise'):
            self._optim_variational_noise_std = self.cfg['variational_noise'].get('std', 0)
            self._optim_variational_noise_start = self.cfg['variational_noise'].get('start_step', 0)
        else:
            self._optim_variational_noise_std = 0
            self._optim_variational_noise_start = 0

        # Setup normalized gradients for model joint by T x U scaling factor (joint length normalization)
        self._optim_normalize_joint_txu = self.cfg.get('normalize_joint_txu', False)
        self._optim_normalize_txu = None

        # Setup normalized encoder norm for model
        self._optim_normalize_encoder_norm = self.cfg.get('normalize_encoder_norm', False)

        # Setup normalized decoder norm for model
        self._optim_normalize_decoder_norm = self.cfg.get('normalize_decoder_norm', False)

        # Setup normalized joint norm for model
        self._optim_normalize_joint_norm = self.cfg.get('normalize_joint_norm', False)

    def extract_rnnt_loss_cfg(self, cfg: Optional[DictConfig]):
        """
        Helper method to extract the rnnt loss name, and potentially its kwargs
        to be passed.

        Args:
            cfg: Should contain `loss_name` as a string which is resolved to a RNNT loss name.
                If the default should be used, then `default` can be used.
                Optionally, one can pass additional kwargs to the loss function. The subdict
                should have a keyname as follows : `{loss_name}_kwargs`.

                Note that whichever loss_name is selected, that corresponding kwargs will be
                selected. For the "default" case, the "{resolved_default}_kwargs" will be used.

        Examples:
            .. code-block:: yaml

                loss_name: "default"
                warprnnt_numba_kwargs:
                    kwargs2: some_other_val

        Returns:
            A tuple, the resolved loss name as well as its kwargs (if found).
        """
        if cfg is None:
            cfg = DictConfig({})

        loss_name = cfg.get("loss_name", "default")

        if loss_name == "default":
            loss_name = resolve_rnnt_default_loss_name()

        loss_kwargs = cfg.get(f"{loss_name}_kwargs", None)

        logging.info(f"Using RNNT Loss : {loss_name}\n" f"Loss {loss_name}_kwargs: {loss_kwargs}")

        return loss_name, loss_kwargs

    def set_decoding_type_according_to_loss(self, decoding_cfg):
        loss_name, loss_kwargs = self.extract_rnnt_loss_cfg(self.cfg.get("loss", None))

        if loss_name == 'tdt':
            decoding_cfg.durations = loss_kwargs.durations
        elif loss_name == 'multiblank_rnnt':
            decoding_cfg.big_blank_durations = loss_kwargs.big_blank_durations

        return decoding_cfg

    @torch.no_grad()
    def transcribe(
        self,
        audio: List[str],
        batch_size: int = 4,
        return_hypotheses: bool = False,
        partial_hypothesis: Optional[List['Hypothesis']] = None,
        num_workers: int = 0,
        channel_selector: Optional[ChannelSelectorType] = None,
        augmentor: DictConfig = None,
        verbose: bool = True,
        override_config: Optional[TranscribeConfig] = None,
    ) -> TranscriptionReturnType:
        """
        Uses greedy decoding to transcribe audio files. Use this method for debugging and prototyping.

        Args:
            audio: (a list) of paths to audio files. \
                Recommended length per file is between 5 and 25 seconds. \
                But it is possible to pass a few hours long file if enough GPU memory is available.
            batch_size: (int) batch size to use during inference. \
                Bigger will result in better throughput performance but would use more memory.
            return_hypotheses: (bool) Either return hypotheses or text
                With hypotheses can do some postprocessing like getting timestamp or rescoring
            partial_hypothesis: Optional[List['Hypothesis']] - A list of partial hypotheses to be used during rnnt
                decoding. This is useful for streaming rnnt decoding. If this is not None, then the length of this
                list should be equal to the length of the audio list.
            num_workers: (int) number of workers for DataLoader
            channel_selector (int | Iterable[int] | str): select a single channel or a subset of channels from multi-channel audio. If set to `'average'`, it performs averaging across channels. Disabled if set to `None`. Defaults to `None`. Uses zero-based indexing.
            augmentor: (DictConfig): Augment audio samples during transcription if augmentor is applied.
            verbose: (bool) whether to display tqdm progress bar
            override_config: (Optional[TranscribeConfig]) override transcription config pre-defined by the user.
                **Note**: All other arguments in the function will be ignored if override_config is passed.
                You should call this argument as `model.transcribe(audio, override_config=TranscribeConfig(...))`.

        Returns:
            Returns a tuple of 2 items -
            * A list of greedy transcript texts / Hypothesis
            * An optional list of beam search transcript texts / Hypothesis / NBestHypothesis.
        """
        return super().transcribe(
            audio=audio,
            batch_size=batch_size,
            return_hypotheses=return_hypotheses,
            num_workers=num_workers,
            channel_selector=channel_selector,
            augmentor=augmentor,
            verbose=verbose,
            override_config=override_config,
            # Additional arguments
            partial_hypothesis=partial_hypothesis,
        )

    def change_vocabulary(self, new_vocabulary: List[str], decoding_cfg: Optional[DictConfig] = None):
        """
        Changes vocabulary used during RNNT decoding process. Use this method when fine-tuning a pre-trained model.
        This method changes only decoder and leaves encoder and pre-processing modules unchanged. For example, you would
        use it if you want to use pretrained encoder when fine-tuning on data in another language, or when you'd need
        model to learn capitalization, punctuation and/or special characters.

        Args:
            new_vocabulary: list with new vocabulary. Must contain at least 2 elements. Typically, \
                this is target alphabet.
            decoding_cfg: A config for the decoder, which is optional. If the decoding type
                needs to be changed (from say Greedy to Beam decoding etc), the config can be passed here.

        Returns: None

        """
        if self.joint.vocabulary == new_vocabulary:
            logging.warning(f"Old {self.joint.vocabulary} and new {new_vocabulary} match. Not changing anything.")
        else:
            if new_vocabulary is None or len(new_vocabulary) == 0:
                raise ValueError(f'New vocabulary must be non-empty list of chars. But I got: {new_vocabulary}')

            joint_config = self.joint.to_config_dict()
            new_joint_config = copy.deepcopy(joint_config)
            new_joint_config['vocabulary'] = new_vocabulary
            new_joint_config['num_classes'] = len(new_vocabulary)
            del self.joint
            self.joint = EncDecRNNTModel.from_config_dict(new_joint_config)

            decoder_config = self.decoder.to_config_dict()
            new_decoder_config = copy.deepcopy(decoder_config)
            new_decoder_config.vocab_size = len(new_vocabulary)
            del self.decoder
            self.decoder = EncDecRNNTModel.from_config_dict(new_decoder_config)

            del self.loss
            loss_name, loss_kwargs = self.extract_rnnt_loss_cfg(self.cfg.get('loss', None))
            self.loss = RNNTLoss(
                num_classes=self.joint.num_classes_with_blank - 1, loss_name=loss_name, loss_kwargs=loss_kwargs
            )

            if decoding_cfg is None:
                # Assume same decoding config as before
                decoding_cfg = self.cfg.decoding

            # Assert the decoding config with all hyper parameters
            decoding_cls = OmegaConf.structured(RNNTDecodingConfig)
            decoding_cls = OmegaConf.create(OmegaConf.to_container(decoding_cls))
            decoding_cfg = OmegaConf.merge(decoding_cls, decoding_cfg)
            decoding_cfg = self.set_decoding_type_according_to_loss(decoding_cfg)

            self.decoding = RNNTDecoding(
                decoding_cfg=decoding_cfg, decoder=self.decoder, joint=self.joint, vocabulary=self.joint.vocabulary,
            )

            self.wer = WER(
                decoding=self.decoding,
                batch_dim_index=self.wer.batch_dim_index,
                use_cer=self.wer.use_cer,
                log_prediction=self.wer.log_prediction,
                dist_sync_on_step=True,
            )

            # Setup fused Joint step
            if self.joint.fuse_loss_wer or (
                self.decoding.joint_fused_batch_size is not None and self.decoding.joint_fused_batch_size > 0
            ):
                self.joint.set_loss(self.loss)
                self.joint.set_wer(self.wer)

            # Update config
            with open_dict(self.cfg.joint):
                self.cfg.joint = new_joint_config

            with open_dict(self.cfg.decoder):
                self.cfg.decoder = new_decoder_config

            with open_dict(self.cfg.decoding):
                self.cfg.decoding = decoding_cfg

            ds_keys = ['train_ds', 'validation_ds', 'test_ds']
            for key in ds_keys:
                if key in self.cfg:
                    with open_dict(self.cfg[key]):
                        self.cfg[key]['labels'] = OmegaConf.create(new_vocabulary)

            logging.info(f"Changed decoder to output to {self.joint.vocabulary} vocabulary.")

    def change_decoding_strategy(self, decoding_cfg: DictConfig):
        """
        Changes decoding strategy used during RNNT decoding process.

        Args:
            decoding_cfg: A config for the decoder, which is optional. If the decoding type
                needs to be changed (from say Greedy to Beam decoding etc), the config can be passed here.
        """
        if decoding_cfg is None:
            # Assume same decoding config as before
            logging.info("No `decoding_cfg` passed when changing decoding strategy, using internal config")
            decoding_cfg = self.cfg.decoding

        # Assert the decoding config with all hyper parameters
        decoding_cls = OmegaConf.structured(RNNTDecodingConfig)
        decoding_cls = OmegaConf.create(OmegaConf.to_container(decoding_cls))
        decoding_cfg = OmegaConf.merge(decoding_cls, decoding_cfg)
        decoding_cfg = self.set_decoding_type_according_to_loss(decoding_cfg)

        self.decoding = RNNTDecoding(
            decoding_cfg=decoding_cfg, decoder=self.decoder, joint=self.joint, vocabulary=self.joint.vocabulary,
        )

        self.wer = WER(
            decoding=self.decoding,
            batch_dim_index=self.wer.batch_dim_index,
            use_cer=self.wer.use_cer,
            log_prediction=self.wer.log_prediction,
            dist_sync_on_step=True,
        )

        # Setup fused Joint step
        if self.joint.fuse_loss_wer or (
            self.decoding.joint_fused_batch_size is not None and self.decoding.joint_fused_batch_size > 0
        ):
            self.joint.set_loss(self.loss)
            self.joint.set_wer(self.wer)

        self.joint.temperature = decoding_cfg.get('temperature', 1.0)

        # Update config
        with open_dict(self.cfg.decoding):
            self.cfg.decoding = decoding_cfg

        logging.info(f"Changed decoding strategy to \n{OmegaConf.to_yaml(self.cfg.decoding)}")

    def _setup_dataloader_from_config(self, config: Optional[Dict]):
        # Automatically inject args from model config to dataloader config
        audio_to_text_dataset.inject_dataloader_value_from_model_config(self.cfg, config, key='sample_rate')
        audio_to_text_dataset.inject_dataloader_value_from_model_config(self.cfg, config, key='labels')

        if config.get("use_lhotse"):
            return get_lhotse_dataloader_from_config(
                config,
                global_rank=self.global_rank,
                world_size=self.world_size,
                dataset=LhotseSpeechToTextBpeDataset(
                    tokenizer=make_parser(
                        labels=config.get('labels', None),
                        name=config.get('parser', 'en'),
                        unk_id=config.get('unk_index', -1),
                        blank_id=config.get('blank_index', -1),
                        do_normalize=config.get('normalize_transcripts', False),
                    ),
                ),
            )

        dataset = audio_to_text_dataset.get_audio_to_text_char_dataset_from_config(
            config=config,
            local_rank=self.local_rank,
            global_rank=self.global_rank,
            world_size=self.world_size,
            preprocessor_cfg=self._cfg.get("preprocessor", None),
        )

        if dataset is None:
            return None

        if isinstance(dataset, AudioToCharDALIDataset):
            # DALI Dataset implements dataloader interface
            return dataset

        shuffle = config['shuffle']
        if isinstance(dataset, torch.utils.data.IterableDataset):
            shuffle = False

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

    def setup_test_data(self, test_data_config: Optional[Union[DictConfig, Dict]]):
        """
        Sets up the test data loader via a Dict-like object.

        Args:
            test_data_config: A config that contains the information regarding construction
                of an ASR Training dataset.

        Supported Datasets:
            -   :class:`~nemo.collections.asr.data.audio_to_text.AudioToCharDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.AudioToBPEDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.TarredAudioToCharDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.TarredAudioToBPEDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text_dali.AudioToCharDALIDataset`
        """
        if 'shuffle' not in test_data_config:
            test_data_config['shuffle'] = False

        # preserve config
        self._update_dataset_config(dataset_name='test', config=test_data_config)

        self._test_dl = self._setup_dataloader_from_config(config=test_data_config)

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
        }

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "outputs": NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation()),
            "encoded_lengths": NeuralType(tuple('B'), LengthsType()),
        }

    @typecheck()
    def forward(
        self, input_signal=None, input_signal_length=None, processed_signal=None, processed_signal_length=None
    ):
        """
        Forward pass of the model. Note that for RNNT Models, the forward pass of the model is a 3 step process,
        and this method only performs the first step - forward of the acoustic model.

        Please refer to the `training_step` in order to see the full `forward` step for training - which
        performs the forward of the acoustic model, the prediction network and then the joint network.
        Finally, it computes the loss and possibly compute the detokenized text via the `decoding` step.

        Please refer to the `validation_step` in order to see the full `forward` step for inference - which
        performs the forward of the acoustic model, the prediction network and then the joint network.
        Finally, it computes the decoded tokens via the `decoding` step and possibly compute the batch metrics.

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
            A tuple of 2 elements -
            1) The log probabilities tensor of shape [B, T, D].
            2) The lengths of the acoustic sequence after propagation through the encoder, of shape [B].
        """
        has_input_signal = input_signal is not None and input_signal_length is not None
        has_processed_signal = processed_signal is not None and processed_signal_length is not None
        if (has_input_signal ^ has_processed_signal) is False:
            raise ValueError(
                f"{self} Arguments ``input_signal`` and ``input_signal_length`` are mutually exclusive "
                " with ``processed_signal`` and ``processed_signal_len`` arguments."
            )

        if not has_processed_signal:
            processed_signal, processed_signal_length = self.preprocessor(
                input_signal=input_signal, length=input_signal_length,
            )

        # Spec augment is not applied during evaluation/testing
        if self.spec_augmentation is not None and self.training:
            processed_signal = self.spec_augmentation(input_spec=processed_signal, length=processed_signal_length)

        encoded, encoded_len = self.encoder(audio_signal=processed_signal, length=processed_signal_length)
        return encoded, encoded_len

    # PTL-specific methods
    def training_step(self, batch, batch_nb):
        # Reset access registry
        if AccessMixin.is_access_enabled(self.model_guid):
            AccessMixin.reset_registry(self)

        signal, signal_len, transcript, transcript_len = batch

        # forward() only performs encoder forward
        if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
            encoded, encoded_len = self.forward(processed_signal=signal, processed_signal_length=signal_len)
        else:
            encoded, encoded_len = self.forward(input_signal=signal, input_signal_length=signal_len)
        del signal

        # During training, loss must be computed, so decoder forward is necessary
        decoder, target_length, states = self.decoder(targets=transcript, target_length=transcript_len)

        if hasattr(self, '_trainer') and self._trainer is not None:
            log_every_n_steps = self._trainer.log_every_n_steps
            sample_id = self._trainer.global_step
        else:
            log_every_n_steps = 1
            sample_id = batch_nb

        # If experimental fused Joint-Loss-WER is not used
        if not self.joint.fuse_loss_wer:
            # Compute full joint and loss
            joint = self.joint(encoder_outputs=encoded, decoder_outputs=decoder)
            loss_value = self.loss(
                log_probs=joint, targets=transcript, input_lengths=encoded_len, target_lengths=target_length
            )

            # Add auxiliary losses, if registered
            loss_value = self.add_auxiliary_losses(loss_value)

            # Reset access registry
            if AccessMixin.is_access_enabled(self.model_guid):
                AccessMixin.reset_registry(self)

            tensorboard_logs = {
                'train_loss': loss_value,
                'learning_rate': self._optimizer.param_groups[0]['lr'],
                'global_step': torch.tensor(self.trainer.global_step, dtype=torch.float32),
            }

            if (sample_id + 1) % log_every_n_steps == 0:
                self.wer.update(
                    predictions=encoded,
                    predictions_lengths=encoded_len,
                    targets=transcript,
                    targets_lengths=transcript_len,
                )
                _, scores, words = self.wer.compute()
                self.wer.reset()
                tensorboard_logs.update({'training_batch_wer': scores.float() / words})

        else:
            # If experimental fused Joint-Loss-WER is used
            if (sample_id + 1) % log_every_n_steps == 0:
                compute_wer = True
            else:
                compute_wer = False

            # Fused joint step
            loss_value, wer, _, _ = self.joint(
                encoder_outputs=encoded,
                decoder_outputs=decoder,
                encoder_lengths=encoded_len,
                transcripts=transcript,
                transcript_lengths=transcript_len,
                compute_wer=compute_wer,
            )

            # Add auxiliary losses, if registered
            loss_value = self.add_auxiliary_losses(loss_value)

            # Reset access registry
            if AccessMixin.is_access_enabled(self.model_guid):
                AccessMixin.reset_registry(self)

            tensorboard_logs = {
                'train_loss': loss_value,
                'learning_rate': self._optimizer.param_groups[0]['lr'],
                'global_step': torch.tensor(self.trainer.global_step, dtype=torch.float32),
            }

            if compute_wer:
                tensorboard_logs.update({'training_batch_wer': wer})

        # Log items
        self.log_dict(tensorboard_logs)

        # Preserve batch acoustic model T and language model U parameters if normalizing
        if self._optim_normalize_joint_txu:
            self._optim_normalize_txu = [encoded_len.max(), transcript_len.max()]

        return {'loss': loss_value}

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        signal, signal_len, transcript, transcript_len, sample_id = batch

        # forward() only performs encoder forward
        if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
            encoded, encoded_len = self.forward(processed_signal=signal, processed_signal_length=signal_len)
        else:
            encoded, encoded_len = self.forward(input_signal=signal, input_signal_length=signal_len)
        del signal

        best_hyp_text, all_hyp_text = self.decoding.rnnt_decoder_predictions_tensor(
            encoder_output=encoded, encoded_lengths=encoded_len, return_hypotheses=False
        )

        sample_id = sample_id.cpu().detach().numpy()
        return list(zip(sample_id, best_hyp_text))

    def validation_pass(self, batch, batch_idx, dataloader_idx=0):
        signal, signal_len, transcript, transcript_len = batch

        # forward() only performs encoder forward
        if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
            encoded, encoded_len = self.forward(processed_signal=signal, processed_signal_length=signal_len)
        else:
            encoded, encoded_len = self.forward(input_signal=signal, input_signal_length=signal_len)
        del signal

        tensorboard_logs = {}

        # If experimental fused Joint-Loss-WER is not used
        if not self.joint.fuse_loss_wer:
            if self.compute_eval_loss:
                decoder, target_length, states = self.decoder(targets=transcript, target_length=transcript_len)
                joint = self.joint(encoder_outputs=encoded, decoder_outputs=decoder)

                loss_value = self.loss(
                    log_probs=joint, targets=transcript, input_lengths=encoded_len, target_lengths=target_length
                )

                tensorboard_logs['val_loss'] = loss_value

            self.wer.update(
                predictions=encoded,
                predictions_lengths=encoded_len,
                targets=transcript,
                targets_lengths=transcript_len,
            )
            wer, wer_num, wer_denom = self.wer.compute()
            self.wer.reset()

            tensorboard_logs['val_wer_num'] = wer_num
            tensorboard_logs['val_wer_denom'] = wer_denom
            tensorboard_logs['val_wer'] = wer

        else:
            # If experimental fused Joint-Loss-WER is used
            compute_wer = True

            if self.compute_eval_loss:
                decoded, target_len, states = self.decoder(targets=transcript, target_length=transcript_len)
            else:
                decoded = None
                target_len = transcript_len

            # Fused joint step
            loss_value, wer, wer_num, wer_denom = self.joint(
                encoder_outputs=encoded,
                decoder_outputs=decoded,
                encoder_lengths=encoded_len,
                transcripts=transcript,
                transcript_lengths=target_len,
                compute_wer=compute_wer,
            )

            if loss_value is not None:
                tensorboard_logs['val_loss'] = loss_value

            tensorboard_logs['val_wer_num'] = wer_num
            tensorboard_logs['val_wer_denom'] = wer_denom
            tensorboard_logs['val_wer'] = wer

        self.log('global_step', torch.tensor(self.trainer.global_step, dtype=torch.float32))

        return tensorboard_logs

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        metrics = self.validation_pass(batch, batch_idx, dataloader_idx)
        if type(self.trainer.val_dataloaders) == list and len(self.trainer.val_dataloaders) > 1:
            self.validation_step_outputs[dataloader_idx].append(metrics)
        else:
            self.validation_step_outputs.append(metrics)
        return metrics

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        logs = self.validation_pass(batch, batch_idx, dataloader_idx=dataloader_idx)
        test_logs = {name.replace("val_", "test_"): value for name, value in logs.items()}
        if type(self.trainer.test_dataloaders) == list and len(self.trainer.test_dataloaders) > 1:
            self.test_step_outputs[dataloader_idx].append(test_logs)
        else:
            self.test_step_outputs.append(test_logs)
        return test_logs

    def multi_validation_epoch_end(self, outputs, dataloader_idx: int = 0):
        if self.compute_eval_loss:
            val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
            val_loss_log = {'val_loss': val_loss_mean}
        else:
            val_loss_log = {}
        wer_num = torch.stack([x['val_wer_num'] for x in outputs]).sum()
        wer_denom = torch.stack([x['val_wer_denom'] for x in outputs]).sum()
        tensorboard_logs = {**val_loss_log, 'val_wer': wer_num.float() / wer_denom}
        return {**val_loss_log, 'log': tensorboard_logs}

    def multi_test_epoch_end(self, outputs, dataloader_idx: int = 0):
        if self.compute_eval_loss:
            test_loss_mean = torch.stack([x['test_loss'] for x in outputs]).mean()
            test_loss_log = {'test_loss': test_loss_mean}
        else:
            test_loss_log = {}
        wer_num = torch.stack([x['test_wer_num'] for x in outputs]).sum()
        wer_denom = torch.stack([x['test_wer_denom'] for x in outputs]).sum()
        tensorboard_logs = {**test_loss_log, 'test_wer': wer_num.float() / wer_denom}
        return {**test_loss_log, 'log': tensorboard_logs}

    """ Transcription related methods """

    def _transcribe_forward(self, batch: Any, trcfg: TranscribeConfig):
        encoded, encoded_len = self.forward(input_signal=batch[0], input_signal_length=batch[1])
        output = dict(encoded=encoded, encoded_len=encoded_len)
        return output

    def _transcribe_output_processing(
        self, outputs, trcfg: TranscribeConfig
    ) -> Tuple[List['Hypothesis'], List['Hypothesis']]:
        encoded = outputs.pop('encoded')
        encoded_len = outputs.pop('encoded_len')

        best_hyp, all_hyp = self.decoding.rnnt_decoder_predictions_tensor(
            encoded,
            encoded_len,
            return_hypotheses=trcfg.return_hypotheses,
            partial_hypotheses=trcfg.partial_hypothesis,
        )

        # cleanup memory
        del encoded, encoded_len

        hypotheses = []
        all_hypotheses = []

        hypotheses += best_hyp
        if all_hyp is not None:
            all_hypotheses += all_hyp
        else:
            all_hypotheses += best_hyp

        return (hypotheses, all_hypotheses)

    def _setup_transcribe_dataloader(self, config: Dict) -> 'torch.utils.data.DataLoader':
        """
        Setup function for a temporary data loader which wraps the provided audio file.

        Args:
            config: A python dictionary which contains the following keys:
            paths2audio_files: (a list) of paths to audio files. The files should be relatively short fragments. \
                Recommended length per file is between 5 and 25 seconds.
            batch_size: (int) batch size to use during inference. \
                Bigger will result in better throughput performance but would use more memory.
            temp_dir: (str) A temporary directory where the audio manifest is temporarily
                stored.

        Returns:
            A pytorch DataLoader for the given audio file(s).
        """
        if 'manifest_filepath' in config:
            manifest_filepath = config['manifest_filepath']
            batch_size = config['batch_size']
        else:
            manifest_filepath = os.path.join(config['temp_dir'], 'manifest.json')
            batch_size = min(config['batch_size'], len(config['paths2audio_files']))

        dl_config = {
            'manifest_filepath': manifest_filepath,
            'sample_rate': self.preprocessor._sample_rate,
            'labels': self.joint.vocabulary,
            'batch_size': batch_size,
            'trim_silence': False,
            'shuffle': False,
            'num_workers': config.get('num_workers', min(batch_size, os.cpu_count() - 1)),
            'pin_memory': True,
        }

        if config.get("augmentor"):
            dl_config['augmentor'] = config.get("augmentor")

        temporary_datalayer = self._setup_dataloader_from_config(config=DictConfig(dl_config))
        return temporary_datalayer

    def on_after_backward(self):
        super().on_after_backward()
        if self._optim_variational_noise_std > 0 and self.global_step >= self._optim_variational_noise_start:
            for param_name, param in self.decoder.named_parameters():
                if param.grad is not None:
                    noise = torch.normal(
                        mean=0.0,
                        std=self._optim_variational_noise_std,
                        size=param.size(),
                        device=param.device,
                        dtype=param.dtype,
                    )
                    param.grad.data.add_(noise)

        if self._optim_normalize_joint_txu:
            T, U = self._optim_normalize_txu
            if T is not None and U is not None:
                for param_name, param in self.encoder.named_parameters():
                    if param.grad is not None:
                        param.grad.data.div_(U)

                for param_name, param in self.decoder.named_parameters():
                    if param.grad is not None:
                        param.grad.data.div_(T)

        if self._optim_normalize_encoder_norm:
            for param_name, param in self.encoder.named_parameters():
                if param.grad is not None:
                    norm = param.grad.norm()
                    param.grad.data.div_(norm)

        if self._optim_normalize_decoder_norm:
            for param_name, param in self.decoder.named_parameters():
                if param.grad is not None:
                    norm = param.grad.norm()
                    param.grad.data.div_(norm)

        if self._optim_normalize_joint_norm:
            for param_name, param in self.joint.named_parameters():
                if param.grad is not None:
                    norm = param.grad.norm()
                    param.grad.data.div_(norm)

    # EncDecRNNTModel is exported in 2 parts
    def list_export_subnets(self):
        return ['encoder', 'decoder_joint']

    # for export
    @property
    def decoder_joint(self):
        return RNNTDecoderJoint(self.decoder, self.joint)

    def set_export_config(self, args):
        if 'decoder_type' in args:
            if hasattr(self, 'change_decoding_strategy'):
                self.change_decoding_strategy(decoder_type=args['decoder_type'])
            else:
                raise Exception("Model does not have decoder type option")
        super().set_export_config(args)

    @classmethod
    def list_available_models(cls) -> List[PretrainedModelInfo]:
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.

        Returns:
            List of available pre-trained models.
        """
        results = []

        model = PretrainedModelInfo(
            pretrained_model_name="stt_zh_conformer_transducer_large",
            description="For details about this model, please visit https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/stt_zh_conformer_transducer_large",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_zh_conformer_transducer_large/versions/1.8.0/files/stt_zh_conformer_transducer_large.nemo",
        )
        results.append(model)

        return results

    @property
    def wer(self):
        return self._wer

    @wer.setter
    def wer(self, wer):
        self._wer = wer
