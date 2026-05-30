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

import os
from dataclasses import dataclass
from math import ceil
from typing import Dict, List, Optional, Union

import torch
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict
from pytorch_lightning import Trainer

from nemo.collections.asr.data import audio_to_text_dataset
from nemo.collections.asr.data.audio_to_text_dali import AudioToBPEDALIDataset, DALIOutputs
from nemo.collections.asr.data.audio_to_text_lhotse import LhotseSpeechToTextBpeDataset
from nemo.collections.asr.data.audio_to_text_lhotse_prompt_index import LhotseSpeechToTextBpeDatasetWithPromptIndex
from nemo.collections.asr.metrics.wer import WER
from nemo.collections.asr.models.rnnt_bpe_models import EncDecRNNTBPEModel
from nemo.collections.asr.parts.mixins import ASRTranscriptionMixin, PromptStreamingMixin, TranscribeConfig
from nemo.collections.asr.parts.mixins.transcription import TranscriptionReturnType
from nemo.collections.asr.parts.preprocessing.segment import ChannelSelectorType
from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTBPEDecoding
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.classes.mixins import AccessMixin
from nemo.core.neural_types import (
    AcousticEncodedRepresentation,
    AudioSignal,
    LabelsType,
    LengthsType,
    NeuralType,
    SpectrogramType,
)
from nemo.utils import logging, model_utils


@dataclass
class RNNTPromptTranscribeConfig(TranscribeConfig):
    """Transcription configuration for RNNT BPE Model with Prompt conditioning."""

    target_lang: str = "auto"


class EncDecRNNTBPEModelWithPrompt(PromptStreamingMixin, EncDecRNNTBPEModel, ASRTranscriptionMixin):
    """Encoder-decoder RNNT model with subword tokenization and prompt conditioning.

    This is the RNNT-only variant (no auxiliary CTC head) of the prompt-aware
    cache-aware streaming model.  The prompt mechanism concatenates a language-ID
    one-hot vector to the encoder output and projects back to the original
    dimension, allowing the decoder to condition on the target language.
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        cfg = model_utils.convert_model_config_to_dict_config(cfg)
        cfg = model_utils.maybe_update_config_version(cfg)

        if 'tokenizer' not in cfg:
            raise ValueError("`cfg` must have `tokenizer` config to create a tokenizer !")

        if not isinstance(cfg, DictConfig):
            cfg = OmegaConf.create(cfg)

        self._setup_tokenizer(cfg.tokenizer)

        vocabulary = self.tokenizer.tokenizer.get_vocab()

        with open_dict(cfg):
            cfg.labels = ListConfig(list(vocabulary))

        with open_dict(cfg.decoder):
            cfg.decoder.vocab_size = len(vocabulary)

        with open_dict(cfg.joint):
            cfg.joint.num_classes = len(vocabulary)
            cfg.joint.vocabulary = ListConfig(list(vocabulary))
            cfg.joint.jointnet.encoder_hidden = cfg.model_defaults.enc_hidden
            cfg.joint.jointnet.pred_hidden = cfg.model_defaults.pred_hidden

        with open_dict(cfg):
            cfg.num_prompts = cfg.model_defaults.get('num_prompts', 128)

            if 'prompt_dictionary' not in cfg.model_defaults:
                logging.warning(
                    "No prompt_dictionary in config; using empty dict " "(expected during checkpoint restoration)."
                )
                cfg.model_defaults.prompt_dictionary = {}

            self.subsampling_factor = cfg.get('subsampling_factor', 8)

        super().__init__(cfg=cfg, trainer=trainer)

        self.concat = False

        if self.cfg.model_defaults.get('initialize_prompt_feature', False):
            self.initialize_prompt_feature()

    @classmethod
    def restore_from(
        cls,
        restore_path,
        override_config_path=None,
        map_location=None,
        strict=True,
        return_config=False,
        save_restore_connector=None,
        trainer=None,
        validate_access_integrity=True,
    ):
        """Delegate to base EncDecRNNTBPEModel to avoid subclass substitution.

        NeMo's from_config_dict checks issubclass(cls, checkpoint_target_cls)
        and, when True, replaces the checkpoint class with cls.  Because this
        class is a direct subclass of the checkpoint's target class
        (EncDecRNNTBPEModel), the substitution would try to fully instantiate
        EncDecRNNTBPEModelWithPrompt with the checkpoint config — which lacks
        prompt_dictionary and hangs.  Delegating to the parent class keeps
        cls == EncDecRNNTBPEModel so the checkpoint is loaded with its own
        class, matching the behaviour that naturally occurs for hybrid models.
        """
        return EncDecRNNTBPEModel.restore_from(
            restore_path=restore_path,
            override_config_path=override_config_path,
            map_location=map_location,
            strict=strict,
            return_config=return_config,
            save_restore_connector=save_restore_connector,
            trainer=trainer,
            validate_access_integrity=validate_access_integrity,
        )

    def initialize_prompt_feature(self):
        """Initialize model components for prompt feature via concatenation."""
        super().initialize_prompt_feature()
        logging.info("Model with prompt feature has been initialized (RNNT-only)")

        self.decoding = RNNTBPEDecoding(
            decoding_cfg=self.cfg.decoding,
            decoder=self.decoder,
            joint=self.joint,
            tokenizer=self.tokenizer,
        )

        self.wer = WER(
            decoding=self.decoding,
            batch_dim_index=0,
            use_cer=self.cfg.get('use_cer', False),
            log_prediction=self.cfg.get('log_prediction', True),
            dist_sync_on_step=True,
        )

        if self.joint.fuse_loss_wer:
            self.joint.set_loss(self.loss)
            self.joint.set_wer(self.wer)

    # Data loading
    def _setup_dataloader_from_config(self, config: Optional[Dict]):
        if config.get("use_lhotse"):
            if config.get('initialize_prompt_feature', True):
                dataset_config = (
                    OmegaConf.to_container(config, resolve=True) if isinstance(config, DictConfig) else dict(config)
                )
                if hasattr(self, 'cfg') and 'encoder' in self.cfg:
                    dataset_config['encoder'] = (
                        OmegaConf.to_container(self.cfg.encoder, resolve=True)
                        if isinstance(self.cfg.encoder, DictConfig)
                        else dict(self.cfg.encoder)
                    )
                dataset = LhotseSpeechToTextBpeDatasetWithPromptIndex(tokenizer=self.tokenizer, cfg=dataset_config)
                logging.info(
                    "Setting up Lhotse dataset with prompt index support (RNNT-only model creates prompt tensors)"
                )
            else:
                dataset = LhotseSpeechToTextBpeDataset(tokenizer=self.tokenizer)
            return get_lhotse_dataloader_from_config(
                config,
                global_rank=self.global_rank,
                world_size=self.world_size,
                dataset=dataset,
                tokenizer=self.tokenizer,
            )

        dataset = audio_to_text_dataset.get_audio_to_text_bpe_dataset_from_config(
            config=config,
            local_rank=self.local_rank,
            global_rank=self.global_rank,
            world_size=self.world_size,
            tokenizer=self.tokenizer,
            preprocessor_cfg=self.cfg.get("preprocessor", None),
        )

        if dataset is None:
            return None

        if isinstance(dataset, AudioToBPEDALIDataset):
            return dataset

        shuffle = config['shuffle']
        if isinstance(dataset, torch.utils.data.IterableDataset):
            shuffle = False

        if hasattr(dataset, 'collate_fn'):
            collate_fn = dataset.collate_fn
        elif hasattr(dataset.datasets[0], 'collate_fn'):
            collate_fn = dataset.datasets[0].collate_fn
        else:
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

    def _setup_transcribe_dataloader(self, config: Dict) -> 'torch.utils.data.DataLoader':
        if 'manifest_filepath' in config:
            manifest_filepath = config['manifest_filepath']
            batch_size = config['batch_size']
        else:
            manifest_filepath = os.path.join(config['temp_dir'], 'manifest.json')
            batch_size = min(config['batch_size'], len(config['paths2audio_files']))

        target_lang = config.get('target_lang', 'en-US')

        dl_config = {
            'manifest_filepath': manifest_filepath,
            'sample_rate': self.preprocessor._sample_rate,
            'labels': self.joint.vocabulary,
            'batch_size': batch_size,
            'trim_silence': False,
            'shuffle': False,
            'num_workers': config.get('num_workers', min(batch_size, os.cpu_count() - 1)),
            'pin_memory': True,
            'use_lhotse': config.get('use_lhotse', True),
            'use_bucketing': False,
            'drop_last': False,
            'initialize_prompt_feature': True,
            'prompt_dictionary': self.cfg.model_defaults.get('prompt_dictionary'),
            'num_prompts': self.cfg.model_defaults.get('num_prompts', 128),
            'subsampling_factor': self.cfg.get('subsampling_factor', 8),
            'default_lang': target_lang,
            'window_stride': self.cfg.preprocessor.get('window_stride', 0.01),
        }

        if config.get("augmentor"):
            dl_config['augmentor'] = config.get("augmentor")

        return self._setup_dataloader_from_config(config=DictConfig(dl_config))

    def setup_training_data(self, train_data_config: Optional[DictConfig]):
        self._update_dataset_config(dataset_name='train', config=train_data_config)
        self._train_dl = self._setup_dataloader_from_config(config=train_data_config)

        if 'is_tarred' in train_data_config and train_data_config['is_tarred']:
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
        if 'shuffle' not in val_data_config:
            val_data_config['shuffle'] = False
        self._update_dataset_config(dataset_name='validation', config=val_data_config)
        self._validation_dl = self._setup_dataloader_from_config(config=val_data_config)

    def setup_test_data(self, test_data_config: Optional[Union[DictConfig, Dict]]):
        if 'shuffle' not in test_data_config:
            test_data_config['shuffle'] = False
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
            "prompt_indices": NeuralType(tuple('B'), LabelsType()),
        }

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "outputs": NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation()),
            "encoded_lengths": NeuralType(tuple('B'), LengthsType()),
        }

    @typecheck()
    def forward(
        self,
        input_signal=None,
        input_signal_length=None,
        processed_signal=None,
        processed_signal_length=None,
        prompt_indices=None,
    ):
        has_input_signal = input_signal is not None and input_signal_length is not None
        has_processed_signal = processed_signal is not None and processed_signal_length is not None
        if (has_input_signal ^ has_processed_signal) is False:
            raise ValueError(
                f"{self} Arguments ``input_signal`` and ``input_signal_length`` are mutually exclusive "
                " with ``processed_signal`` and ``processed_signal_len`` arguments."
            )

        if not has_processed_signal:
            processed_signal, processed_signal_length = self.preprocessor(
                input_signal=input_signal,
                length=input_signal_length,
            )

        if self.spec_augmentation is not None and self.training:
            processed_signal = self.spec_augmentation(input_spec=processed_signal, length=processed_signal_length)

        encoded, encoded_len = self.encoder(audio_signal=processed_signal, length=processed_signal_length)
        encoded = torch.transpose(encoded, 1, 2)  # B x D x T -> B x T x D

        if self.concat:
            if prompt_indices is None:
                raise ValueError("prompt_indices must be provided when concat mode is enabled.")

            batch_size = encoded.shape[0]
            time_steps = encoded.shape[1]
            num_prompts = self.num_prompts

            prompt = torch.zeros(batch_size, time_steps, num_prompts, dtype=encoded.dtype, device=encoded.device)
            prompt.scatter_(2, prompt_indices.view(batch_size, 1, 1).expand(-1, time_steps, -1), 1.0)

            out_dtype = encoded.dtype
            concat_enc_states = torch.cat([encoded, prompt], dim=-1)
            encoded = self.prompt_kernel(concat_enc_states).to(out_dtype)

        encoded = torch.transpose(encoded, 1, 2)  # B x T x D -> B x D x T
        return encoded, encoded_len

    def training_step(self, batch, batch_nb):
        if AccessMixin.is_access_enabled(self.model_guid):
            AccessMixin.reset_registry(self)

        signal, signal_len, transcript, transcript_len, prompt_indices = batch

        if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
            encoded, encoded_len = self.forward(processed_signal=signal, processed_signal_length=signal_len)
        else:
            encoded, encoded_len = self.forward(
                input_signal=signal, input_signal_length=signal_len, prompt_indices=prompt_indices
            )
        del signal

        decoder, target_length, states = self.decoder(targets=transcript, target_length=transcript_len)

        if hasattr(self, '_trainer') and self._trainer is not None:
            log_every_n_steps = self._trainer.log_every_n_steps
            sample_id = self._trainer.global_step
        else:
            log_every_n_steps = 1
            sample_id = batch_nb

        if not self.joint.fuse_loss_wer:
            joint = self.joint(encoder_outputs=encoded, decoder_outputs=decoder)
            loss_value = self.loss(
                log_probs=joint, targets=transcript, input_lengths=encoded_len, target_lengths=target_length
            )
            loss_value = self.add_auxiliary_losses(loss_value)

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
            if (sample_id + 1) % log_every_n_steps == 0:
                compute_wer = True
            else:
                compute_wer = False

            loss_value, wer, _, _ = self.joint(
                encoder_outputs=encoded,
                decoder_outputs=decoder,
                encoder_lengths=encoded_len,
                transcripts=transcript,
                transcript_lengths=transcript_len,
                compute_wer=compute_wer,
            )
            loss_value = self.add_auxiliary_losses(loss_value)

            if AccessMixin.is_access_enabled(self.model_guid):
                AccessMixin.reset_registry(self)

            tensorboard_logs = {
                'train_loss': loss_value,
                'learning_rate': self._optimizer.param_groups[0]['lr'],
                'global_step': torch.tensor(self.trainer.global_step, dtype=torch.float32),
            }

            if compute_wer:
                tensorboard_logs.update({'training_batch_wer': wer})

        self.log_dict(tensorboard_logs)

        if self._optim_normalize_joint_txu:
            self._optim_normalize_txu = [encoded_len.max(), transcript_len.max()]

        return {'loss': loss_value}

    def validation_pass(self, batch, batch_idx, dataloader_idx=0):
        signal, signal_len, transcript, transcript_len, prompt_indices = batch

        if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
            encoded, encoded_len = self.forward(processed_signal=signal, processed_signal_length=signal_len)
        else:
            encoded, encoded_len = self.forward(
                input_signal=signal, input_signal_length=signal_len, prompt_indices=prompt_indices
            )
        del signal

        tensorboard_logs = {}

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
            compute_wer = True

            if self.compute_eval_loss:
                decoded, target_len, states = self.decoder(targets=transcript, target_length=transcript_len)
            else:
                decoded = None
                target_len = transcript_len

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
        tensorboard_logs = self.validation_pass(batch, batch_idx, dataloader_idx)
        if type(self.trainer.val_dataloaders) == list and len(self.trainer.val_dataloaders) > 1:
            self.validation_step_outputs[dataloader_idx].append(tensorboard_logs)
        else:
            self.validation_step_outputs.append(tensorboard_logs)
        return tensorboard_logs

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        logs = self.validation_pass(batch, batch_idx, dataloader_idx=dataloader_idx)
        test_logs = {name.replace("val_", "test_"): value for name, value in logs.items()}
        if type(self.trainer.test_dataloaders) == list and len(self.trainer.test_dataloaders) > 1:
            self.test_step_outputs[dataloader_idx].append(test_logs)
        else:
            self.test_step_outputs.append(test_logs)
        return test_logs

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        signal, signal_len, transcript, transcript_len, prompt_indices = batch

        if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
            encoded, encoded_len = self.forward(processed_signal=signal, processed_signal_length=signal_len)
        else:
            encoded, encoded_len = self.forward(
                input_signal=signal, input_signal_length=signal_len, prompt_indices=prompt_indices
            )
        del signal

        best_hyp = self.decoding.rnnt_decoder_predictions_tensor(
            encoder_output=encoded, encoded_lengths=encoded_len, return_hypotheses=False
        )

        batch_size = signal_len.shape[0]
        sample_id = torch.arange(batch_idx * batch_size, (batch_idx + 1) * batch_size).cpu().detach().numpy()

        return list(zip(sample_id, best_hyp))

    def _transcribe_forward(self, batch, trcfg: RNNTPromptTranscribeConfig) -> dict:
        audio, audio_lens = batch[0], batch[1]
        if len(batch) >= 5:
            prompt_indices = batch[4]
        else:
            prompt_indices = None

        batch_size = audio.shape[0]

        if prompt_indices is None:
            target_lang = trcfg.target_lang
            prompt_dict = self.cfg.model_defaults.get('prompt_dictionary')

            if not prompt_dict:
                raise ValueError("Prompt dictionary is empty. Cannot create dynamic prompts.")

            if target_lang not in prompt_dict:
                available_keys = list(prompt_dict.keys())
                raise ValueError(
                    f"Unknown target language: '{target_lang}'. "
                    f"Available languages: {available_keys[:10]}{'...' if len(available_keys) > 10 else ''}"
                )

            prompt_id = prompt_dict[target_lang]
            prompt_indices = torch.full((batch_size,), prompt_id, dtype=torch.long, device=audio.device)

        encoded, encoded_len = self.forward(
            input_signal=audio, input_signal_length=audio_lens, prompt_indices=prompt_indices
        )

        return dict(encoded=encoded, encoded_len=encoded_len)

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
        timestamps: Optional[bool] = None,
        override_config: Optional[RNNTPromptTranscribeConfig] = None,
        **prompt,
    ) -> TranscriptionReturnType:
        if timestamps is not None:
            decoding_cfg = self.cfg.decoding
            if timestamps or (override_config is not None and override_config.timestamps):
                return_hypotheses = True
                with open_dict(decoding_cfg):
                    decoding_cfg.compute_timestamps = True
                    decoding_cfg.preserve_alignments = True
            else:
                with open_dict(decoding_cfg):
                    decoding_cfg.compute_timestamps = False
                    decoding_cfg.preserve_alignments = False
            self.change_decoding_strategy(decoding_cfg, verbose=False)

        if override_config is None:
            target_lang = prompt.get('target_lang', 'auto')

            trcfg = RNNTPromptTranscribeConfig(
                batch_size=batch_size,
                return_hypotheses=return_hypotheses,
                num_workers=num_workers,
                channel_selector=channel_selector,
                augmentor=augmentor,
                verbose=verbose,
                timestamps=timestamps,
                target_lang=target_lang,
            )
        else:
            if not isinstance(override_config, RNNTPromptTranscribeConfig):
                raise ValueError(
                    f"override_config must be of type {RNNTPromptTranscribeConfig}, "
                    f"but got {type(override_config)}"
                )
            trcfg = override_config

        return super().transcribe(
            audio=audio,
            batch_size=batch_size,
            return_hypotheses=return_hypotheses,
            partial_hypothesis=partial_hypothesis,
            num_workers=num_workers,
            channel_selector=channel_selector,
            augmentor=augmentor,
            verbose=verbose,
            timestamps=timestamps,
            override_config=trcfg,
        )

    @classmethod
    def get_transcribe_config(cls) -> RNNTPromptTranscribeConfig:
        return RNNTPromptTranscribeConfig()

    @classmethod
    def list_available_models(cls) -> List[PretrainedModelInfo]:
        return None
