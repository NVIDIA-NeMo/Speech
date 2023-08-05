# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import torch
from omegaconf import MISSING, DictConfig, OmegaConf, open_dict
from pytorch_lightning import Trainer
from torch.nn.utils.rnn import pad_sequence

from nemo.collections.asr.data.audio_to_text_dali import DALIOutputs
from nemo.collections.asr.data.audio_to_text_dataset import get_audio_to_text_bpe_dataset_from_config
from nemo.collections.asr.data.text_to_text import (
    TextOrAudioToTextBatch,
    TextToTextBatch,
    TextToTextDataset,
    TextToTextIterableDataset,
)
from nemo.collections.asr.models.asr_model import ASRModel
from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE
from nemo.collections.asr.models.hybrid_rnnt_ctc_bpe_models import EncDecHybridRNNTCTCBPEModel
from nemo.collections.asr.models.rnnt_bpe_models import EncDecRNNTBPEModel
from nemo.collections.asr.modules.conformer_encoder import ConformerEncoder
from nemo.collections.asr.parts.preprocessing.features import clean_spectrogram_batch, normalize_batch
from nemo.collections.asr.parts.submodules.batchnorm import replace_bn_with_fused_bn_all
from nemo.collections.common.data import ConcatDataset, ConcatMapDataset
from nemo.collections.tts.models import FastPitchModel, SpectrogramEnhancerModel
from nemo.core.classes import Dataset, typecheck
from nemo.core.classes.common import PretrainedModelInfo
from nemo.utils import logging
from nemo.utils.enum import PrettyStrEnum
from nemo.utils.exceptions import NeMoBaseException


def _fuse_bn_in_conformer(asr_model: ASRModel):
    """
    Replace BatchNorm with Fused BatchNorm in Conformer and fixes model config inplace
    Expected `encoder` model to exist and be of type ConformerEncoder
    """
    logging.info("Replacing BatchNorm with Fused BatchNorm")
    if not hasattr(asr_model, "encoder"):
        raise NotImplementedError("No encoder found in ASR Model, replacement not supported")
    if not isinstance(asr_model.encoder, ConformerEncoder):
        raise NotImplementedError(f"Unsupported encoder type: {type(asr_model.encoder)}")
    replace_bn_with_fused_bn_all(asr_model.encoder)
    if "conv_norm_type" not in asr_model.cfg.encoder:
        # old CTC models from NGC don't have such param
        logging.warning("conv_norm_type not in encoder config, adding parameter")
        with open_dict(asr_model.cfg):
            asr_model.cfg.encoder.conv_norm_type = "fused_batch_norm"
    else:
        asr_model.cfg.encoder.conv_norm_type = "fused_batch_norm"


@dataclass
class TextDataConfig:
    """
    Text dataset subconfig for text-only dataset
    """

    manifest_filepath: Any = MISSING  # actual Union[str, List[str]], but this type is not supported by OmegaConf
    speakers_filepath: Any = MISSING
    min_words: int = 1
    max_words: int = 45  # 45 - recommended value, ~16.7 sec for LibriSpeech
    tokenizer_workers: int = 1
    asr_tts_sampling_technique: Optional[str] = None
    asr_tts_sampling_temperature: Optional[int] = None
    asr_tts_sampling_probabilities: Optional[List[float]] = None


class ASRWithTTSModel(ASRModel):
    """
    Hybrid ASR-TTS model: a transparent wrapper for ASR model
    with frozen text-to-spectrogram pretrained model, which allows to use text-only data for training/finetuning
    Text-only data can be mixed with audio-text pairs
    """

    asr_model: Union[EncDecRNNTBPEModel, EncDecCTCModelBPE, EncDecHybridRNNTCTCBPEModel]
    tts_model: FastPitchModel
    enhancer_model: Optional[SpectrogramEnhancerModel]

    class ASRModelTypes(PrettyStrEnum):
        """
        Supported ASR types, needed for training from scratch
        """

        RNNT_BPE = "rnnt_bpe"
        CTC_BPE = "ctc_bpe"
        HYBRID_RNNT_CTC_BPE = "hybrid_rnnt_ctc_bpe"

        @classmethod
        def from_asr_model(cls, model: Any):
            if isinstance(model, EncDecRNNTBPEModel):
                return cls.RNNT_BPE
            if isinstance(model, EncDecCTCModelBPE):
                return cls.CTC_BPE
            if isinstance(model, EncDecHybridRNNTCTCBPEModel):
                return cls.HYBRID_RNNT_CTC_BPE
            raise ValueError(f"Unsupported model type: {type(model)}")

        def get_asr_cls(self):
            if self == self.RNNT_BPE:
                return EncDecRNNTBPEModel
            if self == self.CTC_BPE:
                return EncDecCTCModelBPE
            if self == self.HYBRID_RNNT_CTC_BPE:
                return EncDecHybridRNNTCTCBPEModel
            raise NotImplementedError(f"Not implemented for value {self.value}")

    @classmethod
    def list_available_models(cls) -> List[PretrainedModelInfo]:
        return []

    @classmethod
    def _check_config(cls, cfg: DictConfig):
        """
        Check that all required fields are present in config
        Structured configs are not compatible with model serialization, so we check fields manually
        """
        expected_fields = [
            # asr
            "asr_model",
            "asr_model_path",
            "asr_model_fuse_bn",
            "asr_model_type",
            # tts
            "tts_model",
            "tts_model_path",
            # enhancer
            "enhancer_model_path",
            "enhancer_model",
        ]
        for field in expected_fields:
            if field not in cfg:
                raise NeMoBaseException(f"Field {field} is required in config (possibly should be None/null)")

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        self._full_init_guard = False

        self._check_config(cfg)  # check all required keys are in config

        # setup datasets and optimizer after model is fully initialized
        # since it's done automatically, remove options from config
        cfg = copy.deepcopy(cfg)  # copy to avoid modifying original config
        with open_dict(cfg):
            train_ds_cfg = cfg.pop("train_ds", None)
            validation_ds_cfg = cfg.pop("validation_ds", None)
            test_ds_cfg = cfg.pop("test_ds", None)
            optim_cfg = cfg.pop("optim", None)

        super().__init__(cfg, trainer=trainer)

        # tts model
        if cfg.tts_model is not None:
            self.register_nemo_submodule("tts_model", config_field="tts_model", model=FastPitchModel(cfg.tts_model))
        else:
            if cfg.tts_model_path is None:
                raise NeMoBaseException("Either tts_model or tts_model_path should be provided")
            self.register_nemo_submodule(
                "tts_model",
                config_field="tts_model",
                model=FastPitchModel.restore_from(f"{cfg.tts_model_path}", map_location=torch.device("cpu")),
            )
        self.tts_model.freeze()  # tts model should be always frozen

        if cfg.asr_model is not None:
            self.asr_model_type = self.ASRModelTypes(cfg.asr_model_type)  # convert to enum
            self.register_nemo_submodule(
                "asr_model", config_field="asr_model", model=self.asr_model_type.get_asr_cls()(cfg.asr_model)
            )
        else:
            if cfg.asr_model_path is None:
                raise NeMoBaseException("Either asr_model or asr_model_path should be provided")
            self.register_nemo_submodule(
                "asr_model",
                config_field="asr_model",
                model=ASRModel.restore_from(f"{cfg.asr_model_path}", map_location=torch.device("cpu")),
            )
            self.asr_model_type = self.ASRModelTypes.from_asr_model(self.asr_model)
            self.cfg.asr_model_type = f"{self.asr_model_type}"  # save to config

            # replace BatchNorm with FusedBatchNorm
        if cfg.asr_model_fuse_bn:
            _fuse_bn_in_conformer(self.asr_model)
            self.cfg.asr_model_fuse_bn = False  # no need to fuse anymore

        if cfg.enhancer_model is not None:
            self.register_nemo_submodule(
                "enhancer_model", config_field="enhancer_model", model=SpectrogramEnhancerModel(cfg.enhancer_model)
            )
        elif cfg.enhancer_model_path is not None:
            self.register_nemo_submodule(
                "enhancer_model",
                config_field="enhancer_model",
                model=SpectrogramEnhancerModel.restore_from(cfg.enhancer_model_path, map_location=torch.device("cpu")),
            )
        else:
            self.enhancer_model = None

        self._full_init_guard = True

        # initialize optimizer and datasets, asr/tts models are initialized here
        if optim_cfg:
            with open_dict(self.cfg):
                self.cfg.optim = optim_cfg
            self.setup_optimization(optim_config=optim_cfg)
        if train_ds_cfg:
            with open_dict(self.cfg):
                self.cfg.train_ds = train_ds_cfg
            self.setup_training_data(train_data_config=train_ds_cfg)
        if validation_ds_cfg:
            with open_dict(self.cfg):
                self.cfg.validation_ds = validation_ds_cfg
            self.setup_multiple_validation_data(val_data_config=validation_ds_cfg)
        if test_ds_cfg:
            with open_dict(self.cfg):
                self.cfg.test_ds = test_ds_cfg
            self.setup_test_data(test_data_config=test_ds_cfg)

    @classmethod
    def from_asr_config(
        cls,
        asr_cfg: DictConfig,
        asr_model_type: Union[str, ASRModelTypes],
        tts_model_path: Union[str, Path],
        enhancer_model_path: Optional[Union[str, Path]] = None,
        trainer: Trainer = None,
    ):
        """
        Method to construct model from ASR config for training from scratch
        """
        model_type = cls.ASRModelTypes(asr_model_type)
        cfg = DictConfig(
            dict(
                asr_model_path=None,
                asr_model=None,
                asr_model_type=f"{model_type}",
                asr_model_fuse_bn=False,  # for training from scratch always should be False
                tts_model_path=f"{tts_model_path}",
                tts_model=None,
                enhancer_model_path=f"{enhancer_model_path}" if enhancer_model_path is not None else None,
                enhancer_model=None,
                train_ds=None,
                validation_ds=None,
                test_ds=None,
                optim=None,
            )
        )

        asr_cfg = copy.deepcopy(asr_cfg)  # copy not to affect original config
        with open_dict(asr_cfg):
            for subconfig_path in ["train_ds", "validation_ds", "test_ds", "optim"]:
                if subconfig_path in asr_cfg:
                    cfg[subconfig_path] = asr_cfg.pop(subconfig_path)
        cfg.asr_model = asr_cfg
        return cls(cfg=cfg, trainer=trainer)

    @classmethod
    def from_pretrained_models(
        cls,
        asr_model_path: Union[str, Path],
        tts_model_path: Union[str, Path],
        enhancer_model_path: Optional[Union[str, Path]] = None,
        asr_model_fuse_bn: bool = False,
        cfg: Optional[DictConfig] = None,
        trainer: Optional[Trainer] = None,
    ):
        """
        Load model from pretrained ASR and TTS models
        Args:
            asr_model_path: path to .nemo ASR model checkpoint
            tts_model_path: path to .nemo TTS model checkpoint
            enhancer_model_path: path to .nemo enhancer model checkpoint
            asr_model_fuse_bn: automatically fuse batchnorm layers in ASR model
            cfg: optional config for hybrid model
            trainer: Pytorch-Lightning trainer

        Returns:
            ASRWithTTSModel instance
        """
        if cfg is None:
            cfg = DictConfig(
                dict(
                    asr_model_path=f"{asr_model_path}",
                    asr_model=None,
                    tts_model_path=f"{tts_model_path}",
                    tts_model=None,
                    enhancer_model_path=f"{enhancer_model_path}" if enhancer_model_path is not None else None,
                    enhancer_model=None,
                    asr_model_type=None,
                    asr_model_fuse_bn=asr_model_fuse_bn,
                    train_ds=None,
                    validation_ds=None,
                    test_ds=None,
                    optim=None,
                )
            )
        else:
            cfg = copy.deepcopy(cfg)  # copy to avoid modifying original config
            cfg.tts_model_path = f"{tts_model_path}"
            cfg.asr_model_path = f"{asr_model_path}"
            cfg.enhancer_model_path = f"{enhancer_model_path}" if enhancer_model_path is not None else None
        return ASRWithTTSModel(cfg, trainer=trainer)

    def __setattr__(self, name, value):
        # pytorch-lightning magic, allows to call *_step on asr_model
        if name == "_current_fx_name" and self._full_init_guard:
            self.asr_model._current_fx_name = value  # need to make logging inside asr_model work
        return super().__setattr__(name, value)

    def setup_optimization(
        self, optim_config: Optional[Union[DictConfig, Dict]] = None, optim_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Setup optimizer and scheduler. Ensure tts model is frozen.
        Add optimizer and scheduler to asr model, to allow `train_step` on ASR model
        """
        self.tts_model.freeze()
        optimizer, scheduler = super().setup_optimization(optim_config=optim_config, optim_kwargs=optim_kwargs)
        # set ASR model optimizer/scheduler to allow training_step on asr_model
        self.asr_model._optimizer = optimizer
        self.asr_model._scheduler = scheduler
        return optimizer, scheduler

    def setup_validation_data(self, val_data_config: Union[DictConfig, Dict]):
        """Setup validation data for ASR model"""
        return self.asr_model.setup_validation_data(val_data_config)

    def multi_validation_epoch_end(self, outputs, dataloader_idx: int = 0):
        """Validation epoch end hook for ASR model"""
        return self.asr_model.multi_validation_epoch_end(outputs=outputs, dataloader_idx=dataloader_idx)

    def multi_test_epoch_end(self, outputs, dataloader_idx: int = 0):
        """Test epoch end hook for ASR model"""
        return self.asr_model.multi_test_epoch_end(outputs=outputs, dataloader_idx=dataloader_idx)

    def transcribe(self, paths2audio_files: List[str], batch_size: int = 4, verbose: bool = True) -> List[str]:
        """Transcribe audio data using ASR model"""
        return self.asr_model.transcribe(paths2audio_files=paths2audio_files, batch_size=batch_size, verbose=verbose)

    def setup_multiple_validation_data(self, val_data_config: Union[DictConfig, Dict]):
        """Setup multiple validation data for ASR model"""
        self.asr_model.setup_multiple_validation_data(val_data_config)

    def setup_test_data(self, test_data_config: Union[DictConfig, Dict]):
        """Setup test data for ASR model"""
        self.asr_model.setup_test_data(test_data_config)

    def setup_multiple_test_data(self, test_data_config: Union[DictConfig, Dict]):
        """Setup multiple test data for ASR Model"""
        return self.asr_model.setup_multiple_test_data(test_data_config)

    def save_asr_model_to(self, save_path: str):
        """Save ASR model separately"""
        return self.asr_model.save_to(save_path=save_path)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """Validation step, forward to ASR model"""
        loss = self.asr_model.validation_step(batch=batch, batch_idx=batch_idx, dataloader_idx=dataloader_idx)
        if type(self.trainer.val_dataloaders) == list and len(self.trainer.val_dataloaders) > 1:
            self.validation_step_outputs[dataloader_idx].append(loss)
        else:
            self.validation_step_outputs.append(loss)
        return loss

    def on_validation_epoch_end(self) -> Optional[Dict[str, Dict[str, torch.Tensor]]]:
        """Validation epoch end hook, forward to ASR model"""
        return self.asr_model.on_validation_epoch_end()

    def on_test_epoch_end(self) -> Optional[Dict[str, Dict[str, torch.Tensor]]]:
        """Test epoch end hook, forward to ASR model"""
        return self.asr_model.on_test_epoch_end()

    def val_dataloader(self):
        """Get valudation dataloader from ASR model"""
        return self.asr_model.val_dataloader()

    def unfreeze(self) -> None:
        """Unfreeze the ASR model, keep TTS model frozen."""
        super().unfreeze()
        self.tts_model.freeze()  # tts model should be always frozen

    def on_fit_start(self):
        """Call asr_model on_fit_start hook, ensure TTS model is frozen"""
        self.asr_model.on_fit_start()
        self.tts_model.freeze()

    def train(self, mode: bool = True):
        """Train mode, ensure TTS model is frozen"""
        super().train(mode)
        self.tts_model.eval()
        return self

    def _get_tts_spectrogram(
        self, tts_texts: torch.Tensor, speakers: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get TTS spectrogram from text and speaker ids"""
        with torch.no_grad():
            spectrogram, spectrogram_len, *_ = self.tts_model(text=tts_texts, durs=None, pitch=None, speaker=speakers)
            if self.enhancer_model is not None:
                # apply enhancer
                with typecheck.disable_checks():
                    # spectrogram_len are of TokenDurationType, enhancer requires LengthsType
                    # TODO: fix FastPitch model to return LengthsType
                    spectrogram = self.enhancer_model.forward(input_spectrograms=spectrogram, lengths=spectrogram_len)
            spectrogram, *_ = normalize_batch(spectrogram, spectrogram_len, self.asr_model.cfg.preprocessor.normalize)
            return spectrogram, spectrogram_len

    def _get_batch_spect(self, batch: Union[TextToTextBatch, TextOrAudioToTextBatch, tuple]):
        """Get batch with spectrograms from text-only, audio-text or mixed batch data"""
        if isinstance(batch, TextToTextBatch):
            spectrogram, spectrogram_len = self._get_tts_spectrogram(batch.tts_texts, batch.speakers)
            transcript = batch.transcripts
            transcript_len = batch.transcript_lengths
        elif isinstance(batch, TextOrAudioToTextBatch):
            tts_spectrogram, tts_spectrogram_len = self._get_tts_spectrogram(batch.tts_texts, batch.speakers)
            asr_spectrogram, asr_spectrogram_len = self.asr_model.preprocessor(
                input_signal=batch.audio_signals, length=batch.audio_signal_lengths,
            )

            spectrogram = pad_sequence(
                [
                    x.squeeze(0)
                    for x in itertools.chain(
                        torch.tensor_split(tts_spectrogram.transpose(1, 2), tts_spectrogram.size(0)),
                        torch.tensor_split(asr_spectrogram.transpose(1, 2), asr_spectrogram.size(0)),
                    )
                ],
                batch_first=True,
                padding_value=0.0,
            ).transpose(1, 2)
            spectrogram_len = torch.cat([tts_spectrogram_len, asr_spectrogram_len], dim=0)

            transcript = batch.transcripts
            transcript_len = batch.transcript_lengths
        else:
            audio_signal, audio_signal_len, transcript, transcript_len, *_ = batch  # audio batch: 4 or 5 elements
            spectrogram, spectrogram_len = self.asr_model.preprocessor(
                input_signal=audio_signal, length=audio_signal_len
            )
        spectrogram = clean_spectrogram_batch(spectrogram, spectrogram_len)
        return spectrogram.detach(), spectrogram_len.detach(), transcript, transcript_len

    def setup_training_data(self, train_data_config: Optional[Union[DictConfig, Dict]]):
        """
        Setup training data from config: text-only, audio-text or mixed data.
        """
        if train_data_config is None:
            logging.warning("No training data")
            return

        self._update_dataset_config(dataset_name='train', config=train_data_config)
        asr_dataset = get_audio_to_text_bpe_dataset_from_config(
            train_data_config,
            local_rank=self.local_rank,
            global_rank=self.global_rank,
            world_size=self.world_size,
            tokenizer=self.asr_model.tokenizer,
            preprocessor_cfg=self.asr_model.cfg.get("preprocessor", None),
        )

        dataset_iterable = True
        if asr_dataset is not None and isinstance(asr_dataset, Dataset):
            # asr_dataset is map-style, for mixing datasets use map-style text-to-text dataset
            dataset_iterable = False
        if train_data_config.get("text_data") is not None:
            tts_dataset = self._setup_text_dataset_from_config(train_data_config, iterable=dataset_iterable)
        else:
            tts_dataset = None

        if tts_dataset and asr_dataset:
            text_data_config: TextDataConfig = cast(
                TextDataConfig, OmegaConf.merge(OmegaConf.structured(TextDataConfig), train_data_config.text_data)
            )
            concat_kwargs = dict()
            if text_data_config.asr_tts_sampling_technique is not None:
                concat_kwargs["sampling_technique"] = text_data_config.asr_tts_sampling_technique
            if text_data_config.asr_tts_sampling_temperature is not None:
                concat_kwargs["sampling_temperature"] = text_data_config.asr_tts_sampling_temperature
            if text_data_config.asr_tts_sampling_probabilities:
                concat_kwargs["sampling_probabilities"] = text_data_config.asr_tts_sampling_probabilities

            if dataset_iterable:
                dataset = ConcatDataset(datasets=[asr_dataset, tts_dataset], **concat_kwargs)
            else:
                dataset = ConcatMapDataset(datasets=[asr_dataset, tts_dataset], **concat_kwargs)
        else:
            dataset = tts_dataset or asr_dataset

        if dataset is None:
            return

        if tts_dataset:
            collate_fn = tts_dataset.collate_fn
        else:
            if hasattr(asr_dataset, 'collate_fn'):
                collate_fn = asr_dataset.collate_fn
            elif hasattr(asr_dataset.datasets[0], 'collate_fn'):
                # support datasets that are lists of entries
                collate_fn = asr_dataset.datasets[0].collate_fn
            else:
                # support datasets that are lists of lists
                collate_fn = asr_dataset.datasets[0].datasets[0].collate_fn

        shuffle = train_data_config.get("shuffle", True) and not dataset_iterable
        self._train_dl = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=train_data_config['batch_size'],
            collate_fn=collate_fn,
            drop_last=train_data_config.get('drop_last', False),
            shuffle=shuffle,
            num_workers=train_data_config.get('num_workers', 0),
            pin_memory=train_data_config.get('pin_memory', False),
        )

    def _setup_text_dataset_from_config(
        self, train_data_config: DictConfig, iterable=True
    ) -> Union[TextToTextDataset, TextToTextIterableDataset]:
        """
        Construct text-to-text (text-only) dataset from config.

        Args:
            train_data_config: config
            iterable: construct iterable-style datasset if True, otherwise map-style

        Returns:
            text-to-text dataset of TextToTextDataset or TextToTextIterableDataset type
        """
        text_data_config: TextDataConfig = cast(
            TextDataConfig, OmegaConf.merge(OmegaConf.structured(TextDataConfig), train_data_config.text_data)
        )
        if iterable:
            textonly_ds = TextToTextIterableDataset(
                manifest_filepath=text_data_config.manifest_filepath,
                speakers_filepath=text_data_config.speakers_filepath,
                asr_tokenizer=self.asr_model.tokenizer,
                asr_use_start_end_token=train_data_config.get("use_start_end_token", False),
                tts_parser=self.tts_model.parser,
                tts_text_pad_id=self.tts_model.vocab.pad,
                tts_text_normalizer=self.tts_model.normalizer,
                tts_text_normalizer_call_kwargs=self.tts_model.text_normalizer_call_kwargs,
                min_words=text_data_config.min_words,
                max_words=text_data_config.max_words,
                tokenizer_workers=text_data_config.tokenizer_workers,
                num_parts=self.world_size,
                current_part_index=self.global_rank,
            )
        else:
            textonly_ds = TextToTextDataset(
                manifest_filepath=text_data_config.manifest_filepath,
                speakers_filepath=text_data_config.speakers_filepath,
                asr_tokenizer=self.asr_model.tokenizer,
                asr_use_start_end_token=train_data_config.get("use_start_end_token", False),
                tts_parser=self.tts_model.parser,
                tts_text_pad_id=self.tts_model.vocab.pad,
                tts_text_normalizer=self.tts_model.normalizer,
                tts_text_normalizer_call_kwargs=self.tts_model.text_normalizer_call_kwargs,
                min_words=text_data_config.min_words,
                max_words=text_data_config.max_words,
                tokenizer_workers=text_data_config.tokenizer_workers,
            )
        return textonly_ds

    def training_step(self, batch: Union[TextOrAudioToTextBatch, TextToTextBatch, DALIOutputs, tuple], batch_nb: int):
        """
        Training step for ASR-TTS model.
        - construct spectrogram for the batch (from text - using TTS model, from audio - using ASR preprocessor)
        - call training_step on ASR model
        """
        assert not self.tts_model.training
        if isinstance(batch, DALIOutputs):
            return self.asr_model.training_step(batch=batch, batch_nb=batch_nb)
        with torch.no_grad():
            spectrogram, spectrogram_len, transcript, transcript_len = self._get_batch_spect(batch)
        # TODO: maybe support precomputed without DALIOutputs
        return self.asr_model.training_step(
            batch=DALIOutputs(
                dict(
                    processed_signal=spectrogram,
                    processed_signal_len=spectrogram_len,
                    transcript=transcript,
                    transcript_len=transcript_len,
                )
            ),
            batch_nb=batch_nb,
        )
