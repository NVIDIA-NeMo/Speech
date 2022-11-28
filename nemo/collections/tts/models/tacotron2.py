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

import contextlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from hydra.utils import instantiate
from omegaconf import MISSING, DictConfig, OmegaConf, open_dict
from omegaconf.errors import ConfigAttributeError
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from torch import nn

from nemo.collections.common.parts.preprocessing import parsers
from nemo.collections.tts.helpers.helpers import (
    get_mask_from_lengths,
    tacotron2_log_to_tb_func,
    tacotron2_log_to_wandb_func,
)
from nemo.collections.tts.losses.tacotron2loss import Tacotron2Loss
from nemo.collections.tts.models.base import SpectrogramGenerator
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.neural_types.elements import (
    AudioSignal,
    EmbeddedTextType,
    LengthsType,
    LogitsType,
    MelSpectrogramType,
    SequenceToSequenceAlignmentType,
)
from nemo.core.neural_types.neural_type import NeuralType
from nemo.utils import logging, model_utils


@dataclass
class Preprocessor:
    _target_: str = MISSING
    pad_value: float = MISSING


@dataclass
class Tacotron2Config:
    preprocessor: Preprocessor = Preprocessor()
    encoder: Dict[Any, Any] = MISSING
    decoder: Dict[Any, Any] = MISSING
    postnet: Dict[Any, Any] = MISSING
    labels: List = MISSING
    train_ds: Optional[Dict[Any, Any]] = None
    validation_ds: Optional[Dict[Any, Any]] = None


class Tacotron2Model(SpectrogramGenerator):
    """Tacotron 2 Model that is used to generate mel spectrograms from text"""

    def __init__(self, cfg: DictConfig, trainer: 'Trainer' = None):
        # Convert to Hydra 1.0 compatible DictConfig
        cfg = model_utils.convert_model_config_to_dict_config(cfg)
        cfg = model_utils.maybe_update_config_version(cfg)

        # setup normalizer
        self.normalizer = None
        self.text_normalizer_call = None
        self.text_normalizer_call_kwargs = {}
        self._setup_normalizer(cfg)

        # setup tokenizer
        self.tokenizer = None
        if hasattr(cfg, 'text_tokenizer'):
            self._setup_tokenizer(cfg)

            self.num_tokens = len(self.tokenizer.tokens)
            self.tokenizer_pad = self.tokenizer.pad
            self.tokenizer_unk = self.tokenizer.oov
            # assert self.tokenizer is not None
        else:
            self.num_tokens = len(cfg.labels) + 3

        super().__init__(cfg=cfg, trainer=trainer)

        schema = OmegaConf.structured(Tacotron2Config)
        # ModelPT ensures that cfg is a DictConfig, but do this second check in case ModelPT changes
        if isinstance(cfg, dict):
            cfg = OmegaConf.create(cfg)
        elif not isinstance(cfg, DictConfig):
            raise ValueError(f"cfg was type: {type(cfg)}. Expected either a dict or a DictConfig")
        # Ensure passed cfg is compliant with schema
        try:
            OmegaConf.merge(cfg, schema)
            self.pad_value = cfg.preprocessor.pad_value
        except ConfigAttributeError:
            self.pad_value = cfg.preprocessor.params.pad_value
            logging.warning(
                "Your config is using an old NeMo yaml configuration. Please ensure that the yaml matches the "
                "current version in the main branch for future compatibility."
            )

        self._parser = None
        self.audio_to_melspec_precessor = instantiate(cfg.preprocessor)
        self.text_embedding = nn.Embedding(self.num_tokens, 512)
        self.encoder = instantiate(self._cfg.encoder)
        self.decoder = instantiate(self._cfg.decoder)
        self.postnet = instantiate(self._cfg.postnet)
        self.loss = Tacotron2Loss()
        self.calculate_loss = True

    @property
    def parser(self):
        if self._parser is not None:
            return self._parser

        ds_class_name = self._cfg.train_ds.dataset._target_.split(".")[-1]
        if ds_class_name == "TTSDataset":
            self._parser = None
        elif hasattr(self._cfg, "labels"):
            self._parser = parsers.make_parser(
                labels=self._cfg.labels,
                name='en',
                unk_id=-1,
                blank_id=-1,
                do_normalize=True,
                abbreviation_version="fastpitch",
                make_table=False,
            )
        elif ds_class_name == "AudioToCharWithPriorAndPitchDataset":
            self.parser = self.vocab.encode
        else:
            raise ValueError("Wanted to setup parser, but model does not have necessary paramaters")

        return self._parser

    def parse(self, text: str, normalize=True) -> torch.Tensor:
        if self.training:
            logging.warning("parse() is meant to be called in eval mode.")
        if normalize and self.text_normalizer_call is not None:
            text = self.text_normalizer_call(text, **self.text_normalizer_call_kwargs)

        eval_phon_mode = contextlib.nullcontext()
        if hasattr(self.tokenizer, "set_phone_prob"):
            eval_phon_mode = self.tokenizer.set_phone_prob(prob=1.0)

        with eval_phon_mode:
            if self.tokenizer is not None:
                tokens = self.tokenizer.encode(text)
            else:
                tokens = self.parser(text)
                # Old parser doesn't add bos and eos ids, so maunally add it
                tokens = [len(self._cfg.labels)] + tokens + [len(self._cfg.labels) + 1]
        tokens_tensor = torch.tensor(tokens).unsqueeze_(0).to(self.device)
        return tokens_tensor

    @property
    def input_types(self):
        if self.training:
            return {
                "tokens": NeuralType(('B', 'T'), EmbeddedTextType()),
                "token_len": NeuralType(('B'), LengthsType()),
                "audio": NeuralType(('B', 'T'), AudioSignal()),
                "audio_len": NeuralType(('B'), LengthsType()),
            }
        else:
            return {
                "tokens": NeuralType(('B', 'T'), EmbeddedTextType()),
                "token_len": NeuralType(('B'), LengthsType()),
                "audio": NeuralType(('B', 'T'), AudioSignal(), optional=True),
                "audio_len": NeuralType(('B'), LengthsType(), optional=True),
            }

    @property
    def output_types(self):
        if not self.calculate_loss and not self.training:
            return {
                "spec_pred_dec": NeuralType(('B', 'D', 'T'), MelSpectrogramType()),
                "spec_pred_postnet": NeuralType(('B', 'D', 'T'), MelSpectrogramType()),
                "gate_pred": NeuralType(('B', 'T'), LogitsType()),
                "alignments": NeuralType(('B', 'T', 'T'), SequenceToSequenceAlignmentType()),
                "pred_length": NeuralType(('B'), LengthsType()),
            }
        return {
            "spec_pred_dec": NeuralType(('B', 'D', 'T'), MelSpectrogramType()),
            "spec_pred_postnet": NeuralType(('B', 'D', 'T'), MelSpectrogramType()),
            "gate_pred": NeuralType(('B', 'T'), LogitsType()),
            "spec_target": NeuralType(('B', 'D', 'T'), MelSpectrogramType()),
            "spec_target_len": NeuralType(('B'), LengthsType()),
            "alignments": NeuralType(('B', 'T', 'T'), SequenceToSequenceAlignmentType()),
        }

    @typecheck()
    def forward(self, *, tokens, token_len, audio=None, audio_len=None):
        if audio is not None and audio_len is not None:
            spec_target, spec_target_len = self.audio_to_melspec_precessor(audio, audio_len)
        token_embedding = self.text_embedding(tokens).transpose(1, 2)
        encoder_embedding = self.encoder(token_embedding=token_embedding, token_len=token_len)
        if self.training:
            spec_pred_dec, gate_pred, alignments = self.decoder(
                memory=encoder_embedding, decoder_inputs=spec_target, memory_lengths=token_len
            )
        else:
            spec_pred_dec, gate_pred, alignments, pred_length = self.decoder(
                memory=encoder_embedding, memory_lengths=token_len
            )
        spec_pred_postnet = self.postnet(mel_spec=spec_pred_dec)

        if not self.calculate_loss:
            return spec_pred_dec, spec_pred_postnet, gate_pred, alignments, pred_length
        return spec_pred_dec, spec_pred_postnet, gate_pred, spec_target, spec_target_len, alignments

    @typecheck(
        input_types={"tokens": NeuralType(('B', 'T'), EmbeddedTextType())},
        output_types={"spec": NeuralType(('B', 'D', 'T'), MelSpectrogramType())},
    )
    def generate_spectrogram(self, *, tokens):
        self.eval()
        self.calculate_loss = False
        token_len = torch.tensor([len(i) for i in tokens]).to(self.device)
        tensors = self(tokens=tokens, token_len=token_len)
        spectrogram_pred = tensors[1]

        if spectrogram_pred.shape[0] > 1:
            # Silence all frames past the predicted end
            mask = ~get_mask_from_lengths(tensors[-1])
            mask = mask.expand(spectrogram_pred.shape[1], mask.size(0), mask.size(1))
            mask = mask.permute(1, 0, 2)
            spectrogram_pred.data.masked_fill_(mask, self.pad_value)

        return spectrogram_pred

    def training_step(self, batch, batch_idx):
        audio, audio_len, tokens, token_len = batch
        spec_pred_dec, spec_pred_postnet, gate_pred, spec_target, spec_target_len, _ = self.forward(
            audio=audio, audio_len=audio_len, tokens=tokens, token_len=token_len
        )

        loss, _ = self.loss(
            spec_pred_dec=spec_pred_dec,
            spec_pred_postnet=spec_pred_postnet,
            gate_pred=gate_pred,
            spec_target=spec_target,
            spec_target_len=spec_target_len,
            pad_value=self.pad_value,
        )

        output = {
            'loss': loss,
            'progress_bar': {'training_loss': loss},
            'log': {'loss': loss},
        }
        return output

    def validation_step(self, batch, batch_idx):
        audio, audio_len, tokens, token_len = batch
        spec_pred_dec, spec_pred_postnet, gate_pred, spec_target, spec_target_len, alignments = self.forward(
            audio=audio, audio_len=audio_len, tokens=tokens, token_len=token_len
        )

        loss, gate_target = self.loss(
            spec_pred_dec=spec_pred_dec,
            spec_pred_postnet=spec_pred_postnet,
            gate_pred=gate_pred,
            spec_target=spec_target,
            spec_target_len=spec_target_len,
            pad_value=self.pad_value,
        )
        return {
            "val_loss": loss,
            "mel_target": spec_target,
            "mel_postnet": spec_pred_postnet,
            "gate": gate_pred,
            "gate_target": gate_target,
            "alignments": alignments,
        }

    def validation_epoch_end(self, outputs):
        if self.logger is not None and self.logger.experiment is not None:
            logger = self.logger.experiment
            for logger in self.trainer.loggers:
                if isinstance(logger, TensorBoardLogger):
                    logger = logger.experiment
                    break
            if isinstance(logger, TensorBoardLogger):
                tacotron2_log_to_tb_func(
                    logger, outputs[0].values(), self.global_step, tag="val", log_images=True, add_audio=False,
                )
            elif isinstance(logger, WandbLogger):
                tacotron2_log_to_wandb_func(
                    logger, outputs[0].values(), self.global_step, tag="val", log_images=True, add_audio=False,
                )
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()  # This reduces across batches, not workers!
        self.log('val_loss', avg_loss)

    def _setup_normalizer(self, cfg):
        if "text_normalizer" in cfg:
            normalizer_kwargs = {}

            if "whitelist" in cfg.text_normalizer:
                normalizer_kwargs["whitelist"] = self.register_artifact(
                    'text_normalizer.whitelist', cfg.text_normalizer.whitelist
                )

            try:
                self.normalizer = instantiate(cfg.text_normalizer, **normalizer_kwargs)
            except Exception as e:
                logging.error(e)
                raise ImportError(
                    "`pynini` not installed, please install via NeMo/nemo_text_processing/pynini_install.sh"
                )

            self.text_normalizer_call = self.normalizer.normalize
            if "text_normalizer_call_kwargs" in cfg:
                self.text_normalizer_call_kwargs = cfg.text_normalizer_call_kwargs

    def _setup_tokenizer(self, cfg):
        text_tokenizer_kwargs = {}
        if "g2p" in cfg.text_tokenizer and cfg.text_tokenizer.g2p is not None:
            g2p_kwargs = {}

            if "phoneme_dict" in cfg.text_tokenizer.g2p:
                g2p_kwargs["phoneme_dict"] = self.register_artifact(
                    'text_tokenizer.g2p.phoneme_dict', cfg.text_tokenizer.g2p.phoneme_dict,
                )

            if "heteronyms" in cfg.text_tokenizer.g2p:
                g2p_kwargs["heteronyms"] = self.register_artifact(
                    'text_tokenizer.g2p.heteronyms', cfg.text_tokenizer.g2p.heteronyms,
                )

            text_tokenizer_kwargs["g2p"] = instantiate(cfg.text_tokenizer.g2p, **g2p_kwargs)

        self.tokenizer = instantiate(cfg.text_tokenizer, **text_tokenizer_kwargs)

    def __setup_dataloader_from_config(self, cfg, shuffle_should_be: bool = True, name: str = "train"):
        if "dataset" not in cfg or not isinstance(cfg.dataset, DictConfig):
            raise ValueError(f"No dataset for {name}")
        if "dataloader_params" not in cfg or not isinstance(cfg.dataloader_params, DictConfig):
            raise ValueError(f"No dataloder_params for {name}")
        if shuffle_should_be:
            if 'shuffle' not in cfg.dataloader_params:
                logging.warning(
                    f"Shuffle should be set to True for {self}'s {name} dataloader but was not found in its "
                    "config. Manually setting to True"
                )
                with open_dict(cfg.dataloader_params):
                    cfg.dataloader_params.shuffle = True
            elif not cfg.dataloader_params.shuffle:
                logging.error(f"The {name} dataloader for {self} has shuffle set to False!!!")
        elif not shuffle_should_be and cfg.dataloader_params.shuffle:
            logging.error(f"The {name} dataloader for {self} has shuffle set to True!!!")

        dataset = instantiate(
            cfg.dataset,
            text_normalizer=self.normalizer,
            text_normalizer_call_kwargs=self.text_normalizer_call_kwargs,
            text_tokenizer=self.tokenizer,
        )

        return torch.utils.data.DataLoader(dataset, collate_fn=dataset.collate_fn, **cfg.dataloader_params)

    def setup_training_data(self, cfg):
        self._train_dl = self.__setup_dataloader_from_config(cfg)

    def setup_validation_data(self, cfg):
        self._validation_dl = self.__setup_dataloader_from_config(cfg, shuffle_should_be=False, name="validation")

    @classmethod
    def list_available_models(cls) -> 'List[PretrainedModelInfo]':
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.
        Returns:
            List of available pre-trained models.
        """
        list_of_models = []
        model = PretrainedModelInfo(
            pretrained_model_name="tts_en_tacotron2",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/tts_en_tacotron2/versions/1.0.0/files/tts_en_tacotron2.nemo",
            description="This model is trained on LJSpeech sampled at 22050Hz, and can be used to generate female English voices with an American accent.",
            class_=cls,
            aliases=["Tacotron2-22050Hz"],
        )
        list_of_models.append(model)
        return list_of_models
