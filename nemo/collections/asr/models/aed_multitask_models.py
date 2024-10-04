# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import ceil
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from nemo.collections.asr.data.audio_to_text_lhotse_prompted import (
    PromptedAudioToTextLhotseDataset,
    PromptedAudioToTextMiniBatch,
)
from nemo.collections.asr.metrics import BLEU, WER
from nemo.collections.asr.models.asr_model import ASRModel, ExportableEncDecModel
from nemo.collections.asr.parts.mixins import ASRBPEMixin, ASRModuleMixin, ASRTranscriptionMixin
from nemo.collections.asr.parts.mixins.transcription import (
    GenericTranscriptionType,
    InternalTranscribeConfig,
    TranscribeConfig,
)
from nemo.collections.asr.parts.preprocessing.segment import ChannelSelectorType
from nemo.collections.asr.parts.submodules.multitask_decoding import MultiTaskDecoding, MultiTaskDecodingConfig
from nemo.collections.asr.parts.submodules.token_classifier import TokenClassifier
from nemo.collections.asr.parts.utils import manifest_utils
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from nemo.collections.common import tokenizers
from nemo.collections.common.data.lhotse.dataloader import get_lhotse_dataloader_from_config
from nemo.collections.common.metrics import GlobalAverageLossMetric
from nemo.collections.common.parts import transformer_weights_init
from nemo.collections.common.parts.preprocessing.manifest import get_full_path
from nemo.collections.common.prompts.fn import get_prompt_format_fn
from nemo.collections.common.prompts.formatter import PromptFormatter
from nemo.core.classes.common import typecheck
from nemo.core.neural_types import (
    AudioSignal,
    ChannelType,
    LabelsType,
    LengthsType,
    LogprobsType,
    MaskType,
    NeuralType,
    SpectrogramType,
)
from nemo.utils import logging, model_utils

__all__ = ['EncDecMultiTaskModel']


def lens_to_mask(lens, max_length):
    batch_size = lens.shape[0]
    mask = torch.arange(max_length).repeat(batch_size, 1).to(lens.device) < lens[:, None]
    return mask


def _config_check(cfg):
    if 'tokenizer' not in cfg:
        raise ValueError("`cfg` must have `tokenizer` config to create a tokenizer !")
    # Assert config has "prompt_format"
    if "prompt_format" not in cfg:
        raise ValueError("`cfg` must have `prompt_format` config to create a multi task model !")
    # Assert config has `model_defaults`
    if 'model_defaults' not in cfg:
        raise ValueError("`cfg` must have `model_defaults` config to create a model !")
    if "asr_enc_hidden" not in cfg.model_defaults:
        raise ValueError("`cfg.model_defaults` must have `asr_enc_hidden` key !")
    if "lm_enc_hidden" not in cfg.model_defaults:
        raise ValueError("`cfg.model_defaults` must have `lm_enc_hidden` key !")
    if "lm_dec_hidden" not in cfg.model_defaults:
        raise ValueError("`cfg.model_defaults` must have `lm_dec_hidden` key !")


@dataclass
class MultiTaskTranscriptionInternalConfig(InternalTranscribeConfig):
    """
    Configuration for Multi Task Transcription
    """

    manifest_filepath: Optional[str] = None
    primary_language: Optional[str] = None


@dataclass
class MultiTaskTranscriptionConfig(TranscribeConfig):
    """
    Configuration for Multi Task Transcription
    """

    prompt: list[dict[str, dict[str, str]]] | None = None
    text_field: str = "answer"
    lang_field: str = "target_lang"

    _internal: Optional[MultiTaskTranscriptionInternalConfig] = field(
        default_factory=lambda: MultiTaskTranscriptionInternalConfig()
    )

    def __post_init__(self):
        self.prompt = parse_multitask_prompt(self.prompt)


class EncDecMultiTaskModel(ASRModel, ExportableEncDecModel, ASRBPEMixin, ASRModuleMixin, ASRTranscriptionMixin):
    """Base class for AED multi-task models"""

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):

        # Convert to Hydra 1.0 compatible DictConfig
        cfg = model_utils.convert_model_config_to_dict_config(cfg)
        cfg = model_utils.maybe_update_config_version(cfg)
        _config_check(cfg)

        self.prompt_format = cfg.prompt_format
        self.sample_rate = cfg.sample_rate
        self._setup_tokenizer(cfg.tokenizer)

        super().__init__(cfg=cfg, trainer=trainer)

        prompt_cls = PromptFormatter.resolve(self.prompt_format)
        self.prompt = prompt_cls(
            tokenizer=self.tokenizer,
            defaults=OmegaConf.to_container(pd) if (pd := cfg.get("prompt_defaults")) is not None else None,
        )

        # Setup audio preprocessor
        self.preprocessor = EncDecMultiTaskModel.from_config_dict(self.cfg.preprocessor)
        # Setup audio encoder
        self.encoder = EncDecMultiTaskModel.from_config_dict(self.cfg.encoder)

        # Add projection layer if encoder and decoder differ in hidden size
        asr_enc_hidden_size = self.cfg.model_defaults.asr_enc_hidden
        decoder_hidden_size = self.cfg.model_defaults.lm_dec_hidden
        if asr_enc_hidden_size != decoder_hidden_size:
            self.encoder_decoder_proj = torch.nn.Linear(asr_enc_hidden_size, decoder_hidden_size)
        else:
            self.encoder_decoder_proj = torch.nn.Identity()

        transf_encoder_cfg_dict = self.cfg.get('transf_encoder', None)

        # Whether to add Transformer Encoder block between Conformer and Transformer Decoder
        self.use_transf_encoder = False
        if transf_encoder_cfg_dict is not None and transf_encoder_cfg_dict['num_layers'] > 0:
            self.use_transf_encoder = True

            self.transf_encoder = EncDecMultiTaskModel.from_config_dict(transf_encoder_cfg_dict)

            # Initialize weights
            std_init_range = 1 / self.cfg.model_defaults.lm_enc_hidden**0.5
            self.transf_encoder.apply(lambda module: transformer_weights_init(module, std_init_range))

        transf_decoder_cfg_dict = cfg.transf_decoder

        # Transformer decoder
        vocab_size = 8 * ceil(self.tokenizer.vocab_size / 8)

        # Auto inject vocab size for `get_transformer`
        with open_dict(transf_decoder_cfg_dict):
            if 'config_dict' in transf_decoder_cfg_dict:
                transf_decoder_cfg_dict['config_dict']['vocab_size'] = vocab_size

        self.transf_decoder = EncDecMultiTaskModel.from_config_dict(transf_decoder_cfg_dict)

        # Setup token classifier
        with open_dict(self.cfg.head):
            self.cfg.head.num_classes = vocab_size

        self.log_softmax = EncDecMultiTaskModel.from_config_dict(self.cfg.head)

        # Weight tying - if using TokenClassifier only
        if isinstance(self.log_softmax, TokenClassifier):
            self.log_softmax.mlp.layer0.weight = self.transf_decoder.embedding.token_embedding.weight

        # Initialize weights
        std_init_range = 1 / self.cfg.model_defaults.lm_dec_hidden**0.5
        self.transf_decoder.apply(lambda module: transformer_weights_init(module, std_init_range))
        self.log_softmax.apply(lambda module: transformer_weights_init(module, std_init_range))

        # Setup decoding objects
        decoding_cfg = self.cfg.get('decoding', None)

        # In case decoding config not found, use default config
        if decoding_cfg is None:
            decoding_cfg = OmegaConf.structured(MultiTaskDecodingConfig)
            with open_dict(self.cfg):
                self.cfg.decoding = decoding_cfg

        self.decoding = MultiTaskDecoding(
            decoding_cfg=self.cfg.decoding,
            transformer_decoder=self.transf_decoder,
            log_softmax_module=self.log_softmax,
            tokenizer=self.tokenizer,
        )

        # Define autoregressive CE loss
        with open_dict(self.cfg.loss):
            self.cfg.loss.pad_id = self.tokenizer.pad_id

        self.loss = EncDecMultiTaskModel.from_config_dict(self.cfg.loss)

        if hasattr(self.cfg, 'spec_augment') and self.cfg.spec_augment is not None:
            self.spec_augmentation = EncDecMultiTaskModel.from_config_dict(self.cfg.spec_augment)
        else:
            self.spec_augmentation = None

        self.val_loss = GlobalAverageLossMetric(dist_sync_on_step=False, take_avg_loss=True)

        # TODO: PytorchMetrics lets you join two metrics together to save compute. But need to make wer and bleu have same outputs first
        self.wer = WER(self.decoding, log_prediction=self.cfg.get("log_prediction"))
        self.bleu = BLEU(
            self.decoding, tokenize=self.cfg.get('bleu_tokenizer', "13a"), log_prediction=False
        )  # Wer is handling logging

        # Setup encoder adapters (from ASRAdapterModelMixin)
        self.setup_adapters()

    def change_decoding_strategy(self, decoding_cfg: DictConfig):
        """
        Changes decoding strategy used during Multi Task decoding process.

        Args:
            decoding_cfg: A config for the decoder, which is optional. If the decoding type
                needs to be changed (from say Greedy to Beam decoding etc), the config can be passed here.
        """
        if decoding_cfg is None:
            # Assume same decoding config as before
            logging.info("No `decoding_cfg` passed when changing decoding strategy, using internal config")
            decoding_cfg = self.cfg.decoding

        # Assert the decoding config with all hyper parameters
        decoding_cls = OmegaConf.structured(MultiTaskDecodingConfig)
        decoding_cls = OmegaConf.create(OmegaConf.to_container(decoding_cls))
        decoding_cfg = OmegaConf.merge(decoding_cls, decoding_cfg)

        self.decoding = MultiTaskDecoding(
            decoding_cfg=decoding_cfg,
            transformer_decoder=self.transf_decoder,
            log_softmax_module=self.log_softmax,
            tokenizer=self.tokenizer,
        )

        # Update config
        with open_dict(self.cfg.decoding):
            self.cfg.decoding = decoding_cfg

        logging.info(f"Changed decoding strategy to \n{OmegaConf.to_yaml(self.cfg.decoding)}")

    def change_vocabulary(
        self,
        new_tokenizer_dir: Union[str, DictConfig],
        new_tokenizer_type: str,
        decoding_cfg: Optional[DictConfig] = None,
        prompt_format: Optional[str] = None,
    ):
        """
        Changes vocabulary used during AED decoding process. Use this method when fine-tuning on from pre-trained model.
        This method changes only decoder and leaves encoder and pre-processing modules unchanged. For example, you would
        use it if you want to use pretrained encoder when fine-tuning on data in another language, or when you'd need
        model to learn capitalization, punctuation and/or special characters.

        Args:
            new_tokenizer_dir: Directory path to tokenizer or a config for a new tokenizer (if the tokenizer type is `agg`)
            new_tokenizer_type: Type of tokenizer. Can be either `agg`, `bpe` or `wpe`.
            decoding_cfg: A config for the decoding, which is optional. If the decoding type
                needs to be changed (from say Greedy to Beam decoding etc), the config can be passed here.
            prompt_format: A string alias of the object that represents the prompt structure.
                If not None, it will be used to update the prompt format.
        """
        if isinstance(new_tokenizer_dir, (dict, DictConfig)):
            if new_tokenizer_type == 'agg':
                if not isinstance(new_tokenizer_dir, DictConfig):
                    new_tokenizer_dir = OmegaConf.create(new_tokenizer_dir)

                new_tokenizer_cfg = new_tokenizer_dir
            else:
                raise ValueError(
                    f'New tokenizer dir should be a string unless the tokenizer is `agg`, but this tokenizer type is: {new_tokenizer_type}'
                )
        else:
            new_tokenizer_cfg = None

        if new_tokenizer_cfg is not None:
            tokenizer_cfg = new_tokenizer_cfg
        else:
            if not os.path.isdir(new_tokenizer_dir):
                raise NotADirectoryError(
                    f'New tokenizer dir must be non-empty path to a directory. But instead got: {new_tokenizer_dir}'
                )

            if new_tokenizer_type.lower() not in ('bpe', 'wpe'):
                raise ValueError(f'New tokenizer type must be either `bpe` or `wpe`')

            tokenizer_cfg = OmegaConf.create({'dir': new_tokenizer_dir, 'type': new_tokenizer_type})

        if prompt_format is None:
            prompt_format = self.cfg.prompt_format

        # Setup the tokenizer
        self._setup_tokenizer(tokenizer_cfg)

        # Initialize a dummy vocabulary
        vocabulary = self.tokenizer.tokenizer.get_vocab()

        # Setup Decoder
        transf_decoder_cfg_dict = self.transf_decoder.to_config_dict()

        vocab_size = 8 * ceil(self.tokenizer.vocab_size / 8)

        # Auto inject vocab size for `get_transformer`
        with open_dict(transf_decoder_cfg_dict):
            if 'config_dict' in transf_decoder_cfg_dict:
                transf_decoder_cfg_dict['config_dict']['vocab_size'] = vocab_size

        original_decoder_state_dict = self.transf_decoder.state_dict()
        self.transf_decoder = EncDecMultiTaskModel.from_config_dict(transf_decoder_cfg_dict)

        # Partially load the original state dict into the new decoder
        decoder_state_dict = self.transf_decoder.state_dict()
        for og_key, og_value in original_decoder_state_dict.items():
            if og_key in decoder_state_dict and og_value.shape == decoder_state_dict[og_key].shape:
                decoder_state_dict[og_key] = og_value
            else:
                logging.warning(
                    f"Skipping key `{og_key}` in the `transf_decoder` module from original state dict due "
                    f"to shape mismatch after change in vocabulary.\n"
                    f"Original shape: {og_value.shape}, New shape: {decoder_state_dict[og_key].shape}"
                )

        self.transf_decoder.load_state_dict(decoder_state_dict)

        # Setup token classifier
        with open_dict(self.cfg.head):
            self.cfg.head.num_classes = vocab_size

        del self.log_softmax
        self.log_softmax = EncDecMultiTaskModel.from_config_dict(self.cfg.head)

        # Weight tying - if using TokenClassifier only
        if isinstance(self.log_softmax, TokenClassifier):
            self.log_softmax.mlp.layer0.weight = self.transf_decoder.embedding.token_embedding.weight

        # Initialize weights of token classifier
        std_init_range = 1 / self.cfg.model_defaults.lm_dec_hidden**0.5
        self.log_softmax.apply(lambda module: transformer_weights_init(module, std_init_range))

        # Setup Decoding class
        if decoding_cfg is None:
            # Assume same decoding config as before
            decoding_cfg = self.cfg.decoding

        # Assert the decoding config with all hyper parameters
        decoding_cls = OmegaConf.structured(MultiTaskDecodingConfig)
        decoding_cls = OmegaConf.create(OmegaConf.to_container(decoding_cls))
        decoding_cfg = OmegaConf.merge(decoding_cls, decoding_cfg)

        del self.decoding
        self.decoding = MultiTaskDecoding(
            decoding_cfg=decoding_cfg,
            transformer_decoder=self.transf_decoder,
            log_softmax_module=self.log_softmax,
            tokenizer=self.tokenizer,
        )

        with open_dict(self.cfg.decoding):
            self.cfg.decoding = decoding_cfg

        # Setup loss
        with open_dict(self.cfg.loss):
            self.cfg.loss.pad_id = self.tokenizer.pad_id

        del self.loss
        self.loss = EncDecMultiTaskModel.from_config_dict(self.cfg.loss)

        # Update config
        with open_dict(self.cfg):
            self.cfg.prompt_format = prompt_format

        logging.info(f"Changed decoder to output to {vocabulary} vocabulary.")

    def change_prompt(
        self, prompt_format: Optional[str] = None, prompt_defaults: Optional[List[Dict[str, Any]]] = None
    ):
        """
        Changes the prompt format used during Multi Task decoding process.

        Args:
            prompt_format: A string alias of the object that represents the prompt structure.
                If not None, it will be used to update the prompt format.
            prompt_defaults: A dictionary of default values for the prompt format.
        """
        if prompt_format is not None:
            self.prompt_format = prompt_format

        if prompt_defaults is not None:
            # Perform some assertions on the prompt defaults contents
            # Must be a list-like object
            if not isinstance(prompt_defaults, Sequence):
                raise ValueError("`prompt_defaults` must be a list of dictionaries")

            # Must contain dict-like objects
            for item in prompt_defaults:
                if not isinstance(item, Mapping):
                    raise ValueError("`prompt_defaults` must be a list of dictionaries")

                # Each dict item must have a `role` key
                if 'role' not in item:
                    raise ValueError(
                        "`prompt_defaults` must have a `role` key for each item in the list of dictionaries"
                    )

                if 'slots' not in item:
                    raise ValueError(
                        "`prompt_defaults` must have a `slots` key for each item in the list of dictionaries"
                    )

            # Cast to OmegaConf if not already
            if not isinstance(prompt_defaults, ListConfig):
                prompt_defaults = OmegaConf.create(prompt_defaults)

        prompt_cls = PromptFormatter.resolve(self.prompt_format)
        self.prompt = prompt_cls(
            tokenizer=self.tokenizer,
            defaults=OmegaConf.to_container(pd) if (pd := self.cfg.get('prompt_defaults')) is not None else None,
        )

        # Update config
        with open_dict(self.cfg):
            self.cfg.prompt_format = self.prompt_format
            self.cfg.prompt_defaults = prompt_defaults

        logging.info(f"Changed prompt format to `{self.prompt_format}`")

    @torch.no_grad()
    def transcribe(
        self,
        audio: Union[str, List[str], np.ndarray, DataLoader],
        batch_size: int = 4,
        return_hypotheses: bool = False,
        num_workers: int = 0,
        channel_selector: Optional[ChannelSelectorType] = None,
        augmentor: DictConfig = None,
        verbose: bool = True,
        override_config: Optional[MultiTaskTranscriptionConfig] = None,
        **prompt,
    ) -> Union[List[str], List[Hypothesis]]:
        """
        Uses greedy decoding to transcribe audio files. Use this method for debugging and prototyping.
        Args:
            audio: (a single or list) of paths to audio files or a np.ndarray/tensor audio array or path to a manifest file.
                Can also be a dataloader object that provides values that can be consumed by the model.
                Recommended length per file is between 5 and 25 seconds. \
                But it is possible to pass a few hours long file if enough GPU memory is available.
            batch_size: (int) batch size to use during inference.
                Bigger will result in better throughput performance but would use more memory.
            return_hypotheses: (bool) Either return hypotheses or text
                With hypotheses can do some postprocessing like getting timestamp or rescoring
            num_workers: (int) number of workers for DataLoader
            channel_selector (int | Iterable[int] | str): select a single channel or a subset of channels from multi-channel audio. If set to `'average'`, it performs averaging across channels. Disabled if set to `None`. Defaults to `None`.
            augmentor: (DictConfig): Augment audio samples during transcription if augmentor is applied.
            verbose: (bool) whether to display tqdm progress bar
            override_config: (Optional[MultiTaskTranscriptionConfig]) A config to override the default config.
            **prompt: Optional input to construct the prompts for the model. Accepted formats are: 1) legacy Canary-1B API source_lang=<lang>, target_lang=<lang>, etc. 2) explicit single-turn role=<role>, slots={<slot>: <value>, ...} 3) explicit multi-turn: turns=[{"role": <role>, "slots": {<slot>: <value>, ...}}]

        Returns:
            A list of transcriptions (or raw log probabilities if logprobs is True) in the same order as paths2audio_files
        """
        if override_config is None:
            trcfg = MultiTaskTranscriptionConfig(
                batch_size=batch_size,
                return_hypotheses=return_hypotheses,
                num_workers=num_workers,
                channel_selector=channel_selector,
                augmentor=augmentor,
                verbose=verbose,
                prompt=prompt,
            )
        else:
            if not isinstance(override_config, MultiTaskTranscriptionConfig):
                raise ValueError(
                    f"override_config must be of type {MultiTaskTranscriptionConfig}, "
                    f"but got {type(override_config)}"
                )
            trcfg = override_config

        return super().transcribe(audio=audio, override_config=trcfg)

    def _setup_dataloader_from_config(self, config: Optional[Dict]):
        assert config.get("use_lhotse", False), (
            "Multi-task model only supports dataloading with Lhotse. "
            "Please set config.{train,validation,test}_ds.use_lhotse=True"
        )
        global_rank = config.get("global_rank", self.global_rank)
        world_size = config.get("world_size", self.world_size)
        return get_lhotse_dataloader_from_config(
            config,
            global_rank=global_rank,
            world_size=world_size,
            dataset=PromptedAudioToTextLhotseDataset(
                tokenizer=self.tokenizer,
                prompt_format_fn=get_prompt_format_fn(self.prompt_format),
            ),
            tokenizer=self.tokenizer,
        )

    def setup_training_data(self, train_data_config: Optional[DictConfig]):

        # create audio-only data loader
        self._update_dataset_config(dataset_name='train', config=train_data_config)
        self._train_dl = self._setup_dataloader_from_config(config=train_data_config)

        # Need to set this because if using an IterableDataset, the length of the
        # dataloader is the total number of samples rather than the number of batches,
        # and this messes up the tqdm progress bar. So we set the number of steps manually
        # (to the correct number) to fix this.
        if 'is_tarred' in train_data_config and train_data_config['is_tarred']:
            # We also need to check if limit_train_batches is already set.
            # If it's an int, we assume that the user has set it to something sane,
            # i.e. <= # training batches, and don't change it. Otherwise, adjust
            # batches accordingly if it's a float (including 1.0).
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
            -   :class:`~nemo.collections.asr.data.audio_to_text_lhotse_prompted.PromptedAudioToTextLhotseDataset`
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
            -   :class:`~nemo.collections.asr.data.audio_to_text_lhotse_prompted.PromptedAudioToTextLhotseDataset`
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
            "transcript": NeuralType(('B', 'T'), LabelsType(), optional=True),
            "transcript_length": NeuralType(tuple('B'), LengthsType(), optional=True),
            "prompt": NeuralType(('B', 'T'), LabelsType(), optional=True),
            "prompt_length": NeuralType(tuple('B'), LengthsType(), optional=True),
            "sample_id": NeuralType(tuple('B'), LengthsType(), optional=True),
        }

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "transf_log_probs": NeuralType(('B', 'T', 'D'), LogprobsType()),
            "encoded_lengths": NeuralType(tuple('B'), LengthsType()),
            "encoder_states": NeuralType(('B', 'T', 'D'), ChannelType()),
            "encoder_mask": NeuralType(('B', 'T'), MaskType()),
        }

    @typecheck()
    def forward(
        self,
        input_signal=None,
        input_signal_length=None,
        processed_signal=None,
        processed_signal_length=None,
        transcript=None,
        transcript_length=None,
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
                of shape (B, D, T).
            processed_signal_length: Vector of length B, that contains the individual lengths of the
                processed audio sequences.
            # TODO: Add support for `transcript` and `transcript_length` in the docstring

        Returns:
            A tuple of 3 elements -
            1) The log probabilities tensor of shape [B, T, D].
            2) The lengths of the acoustic sequence after propagation through the encoder, of shape [B].
            3) The greedy token predictions of the model of shape [B, T] (via argmax)
        """
        has_input_signal = input_signal is not None and input_signal_length is not None
        has_processed_signal = processed_signal is not None and processed_signal_length is not None
        if (has_input_signal ^ has_processed_signal) == False:
            raise ValueError(
                f"{self} Arguments ``input_signal`` and ``input_signal_length`` are mutually exclusive "
                " with ``processed_signal`` and ``processed_signal_len`` arguments."
            )

        if not has_processed_signal:
            processed_signal, processed_signal_length = self.preprocessor(
                input_signal=input_signal, length=input_signal_length
            )

        if self.spec_augmentation is not None and self.training:
            processed_signal = self.spec_augmentation(input_spec=processed_signal, length=processed_signal_length)

        encoded, encoded_len = self.encoder(audio_signal=processed_signal, length=processed_signal_length)

        enc_states = encoded.permute(0, 2, 1)
        enc_states = self.encoder_decoder_proj(enc_states)
        enc_mask = lens_to_mask(encoded_len, enc_states.shape[1]).to(enc_states.dtype)
        if self.use_transf_encoder:
            enc_states = self.transf_encoder(encoder_states=enc_states, encoder_mask=enc_mask)

        transf_log_probs = None
        if transcript is not None:
            dec_mask = lens_to_mask(transcript_length, transcript.shape[1]).to(transcript.dtype)
            dec_states = self.transf_decoder(
                input_ids=transcript, decoder_mask=dec_mask, encoder_embeddings=enc_states, encoder_mask=enc_mask
            )
            transf_log_probs = self.log_softmax(hidden_states=dec_states)

        return transf_log_probs, encoded_len, enc_states, enc_mask

    # PTL-specific methods
    def training_step(self, batch: PromptedAudioToTextMiniBatch, batch_nb):

        if batch is None:
            return torch.tensor([0.0])

        input_ids, labels = batch.get_decoder_inputs_outputs()

        transf_log_probs, encoded_len, enc_states, enc_mask = self.forward(
            input_signal=batch.audio,
            input_signal_length=batch.audio_lens,
            transcript=input_ids,
            transcript_length=batch.prompted_transcript_lens,
        )

        audio_loss = self.loss(log_probs=transf_log_probs, labels=labels)

        num_frames = batch.audio_lens.sum()
        num_tokens = batch.prompted_transcript_lens.sum()
        tot_frames = batch.audio.numel()
        tot_tokens = batch.prompted_transcript.numel()
        tensorboard_logs = {
            'train_loss': audio_loss,
            'learning_rate': self._optimizer.param_groups[0]['lr'],
            'batch_size': batch.audio.shape[0],
            'num_frames': num_frames,
            'num_tokens': num_tokens,
            'input_to_padding_ratio': num_frames / tot_frames,
            'output_to_padding_ratio': num_tokens / tot_tokens,
        }

        return {'loss': audio_loss, 'log': tensorboard_logs}

    def validation_pass(self, batch: PromptedAudioToTextMiniBatch, batch_idx, dataloader_idx=0, eval_mode="val"):
        input_ids, labels = batch.get_decoder_inputs_outputs()

        transf_log_probs, encoded_len, enc_states, enc_mask = self.forward(
            input_signal=batch.audio,
            input_signal_length=batch.audio_lens,
            transcript=input_ids,
            transcript_length=batch.prompted_transcript_lens,
        )

        transf_loss = self.loss(log_probs=transf_log_probs, labels=labels)
        self.val_loss(loss=transf_loss, num_measurements=transf_log_probs.shape[0] * transf_log_probs.shape[1])
        output_dict = {
            f'{eval_mode}_loss': transf_loss,
        }

        self.wer.update(
            predictions=enc_states,
            predictions_lengths=encoded_len,
            targets=batch.transcript,
            targets_lengths=batch.transcript_lens,
            predictions_mask=enc_mask,
            input_ids=batch.prompt,
        )
        wer, wer_num, wer_denom = self.wer.compute()
        output_dict.update({"val_wer": wer, "val_wer_num": wer_num, "val_wer_denom": wer_denom})
        self.wer.reset()

        self.bleu.update(
            predictions=enc_states,
            predictions_lengths=encoded_len,
            targets=batch.transcript,
            targets_lengths=batch.transcript_lens,
            predictions_mask=enc_mask,
            input_ids=batch.prompt,
        )
        bleu_metrics = self.bleu.compute(prefix=f"{eval_mode}_")
        output_dict.update(bleu_metrics)
        self.bleu.reset()

        return output_dict

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        metrics = self.validation_pass(batch, batch_idx, dataloader_idx, eval_mode="val")
        if type(self.trainer.val_dataloaders) == list and len(self.trainer.val_dataloaders) > 1:
            self.validation_step_outputs[dataloader_idx].append(metrics)
        else:
            self.validation_step_outputs.append(metrics)
        return metrics

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        metrics = self.validation_pass(batch, batch_idx, dataloader_idx, eval_mode="test")
        if type(self.trainer.val_dataloaders) == list and len(self.trainer.val_dataloaders) > 1:
            self.validation_step_outputs[dataloader_idx].append(metrics)
        else:
            self.validation_step_outputs.append(metrics)
        return metrics

    def test_dataloader(self):
        if self._test_dl is not None:
            return self._test_dl

    """ Transcription methods """

    def _transcribe_on_begin(self, audio, trcfg: MultiTaskTranscriptionConfig):
        """
        Transcription setup method.
        Args:
            audio: A list of paths to audio files or a path to a manifest file.
            trcfg: A config for the transcription, which is optional. If the decoding type
                needs to be changed (from say Greedy to Beam decoding etc), the config can be passed here.
        """
        super()._transcribe_on_begin(audio, trcfg)

        # Switch model to evaluation mode
        self.transf_decoder.freeze()

        if isinstance(audio, list):
            logging.debug(f"Found 'audio' to be a list of {len(audio)} items.")
            logging.debug(f"Assuming each item in 'audio' is a path to audio file.")

            if isinstance(self.tokenizer, tokenizers.AggregateTokenizer):
                if hasattr(trcfg, '_internal') and hasattr(trcfg._internal, 'primary_language'):
                    trcfg._internal.primary_language = self.tokenizer.langs[0]
                    logging.debug(f"Transcribing with default setting of {trcfg._internal.primary_language}.")

    def _transcribe_input_manifest_processing(
        self, audio_files: List[str], temp_dir: str, trcfg: MultiTaskTranscriptionConfig
    ) -> Dict[str, Any]:
        """
        Internal function to process the input audio filepaths and return a config dict for the dataloader.
        This implementation adds support for dictionaries as manifest items.

        Args:
            audio_files: A list of string filepaths for audio files, or a single string filepath for a manifest file.
            temp_dir: A temporary directory to store intermediate files.
            trcfg: The transcription config dataclass. Subclasses can change this to a different dataclass if needed.

        Returns:
            A config dict that is used to setup the dataloader for transcription.
        """
        manifest_filepath = trcfg._internal.manifest_filepath

        audio_files = self._may_be_make_dict_and_fix_paths(audio_files, manifest_filepath, trcfg)

        return super()._transcribe_input_manifest_processing(audio_files, temp_dir, trcfg)

    def _transcribe_forward(
        self, batch: PromptedAudioToTextMiniBatch | tuple[torch.Tensor, ...], trcfg: MultiTaskTranscriptionConfig
    ) -> dict:
        """
        Internal function to perform the model's custom forward pass to return outputs that are processed by
        `_transcribe_output_processing()`.
        This function is called by `transcribe()` and `transcribe_generator()` to perform the model's forward pass.

        Args:
            batch: A batch of input data from the data loader that is used to perform the model's forward pass.
            trcfg: The transcription config dataclass. Subclasses can change this to a different dataclass if needed.

        Returns:
            The model's outputs that are processed by `_transcribe_output_processing()`.
        """
        if isinstance(batch, PromptedAudioToTextMiniBatch):
            # Handling regular Canary DataLoader
            audio = batch.audio
            audio_lens = batch.audio_lens
            decoder_input_ids = batch.prompt
        else:
            # Handling TensorDataset / external DataLoader
            audio, audio_lens = batch[0], batch[1]
            if len(batch) == 6:
                # Prompt provided by the user.
                decoder_input_ids = batch[4]
            else:
                # Prompt to be built dynamically.
                decoder_input_ids = None
        batch_size = audio.shape[0]

        log_probs, encoded_len, enc_states, enc_mask = self.forward(input_signal=audio, input_signal_length=audio_lens)

        if decoder_input_ids is None:
            # The dataloader provided only audio + audio_lens, so we
            # are constructing the prompt dynamically using TranscribeConfig.

            # Now ask the prompt formatter about which slots are required.
            # It will return a default prompt structure with default slot values (if available, None otherwise).
            # We iterate over that structure and update slot values based on ``trcfg.prompt``.
            default_turns = self.prompt.get_default_dialog_slots()
            if not trcfg.prompt:
                # No turns were provided, use defaults.
                turns = default_turns
            else:
                # Turns were provided, iterate over them and fill missing slot values using defaults..
                turns = trcfg.prompt.copy()  # shallow copy #1: don't override the config
                for turn in turns:
                    role = turn["role"]
                    # Check if we have defaults for this role.
                    # There shouldn't be more than a single turn for a given role, but if there are,
                    # we'll emit a warning.
                    if default_turns_for_role := [t for t in default_turns if t["role"] == role]:
                        if len(default_turns_for_role) > 1:
                            warnings.warn(
                                f"More than one default turn detected for {role=}. "
                                f"We'll be using default slot values for the first turn of {role=} only."
                            )
                        default_slots = default_turns_for_role[0]["slots"]
                        turn["slots"] = turn["slots"].copy()  # shallow copy #1: don't override the config
                        # fill missing slots using defaults
                        for slot, val in default_slots.items():
                            if turn["slots"].get(slot) is None:
                                turn["slots"][slot] = val

            decoder_input_ids = (
                self.prompt.encode_dialog(turns=turns)["context_ids"]
                .unsqueeze(0)
                .repeat(batch_size, 1)
                .to(trcfg._internal.device)
            )

        return dict(
            log_probs=log_probs,
            encoded_lengths=encoded_len,
            encoder_states=enc_states,
            encoder_mask=enc_mask,
            decoder_input_ids=decoder_input_ids,
        )

    def _transcribe_output_processing(self, outputs, trcfg: MultiTaskTranscriptionConfig) -> GenericTranscriptionType:
        """
        Internal function to process the model's outputs to return the results to the user. This function is called by
        `transcribe()` and `transcribe_generator()` to process the model's outputs.

        Args:
            outputs: The model's outputs that are processed by `_transcribe_forward()`.
            trcfg: The transcription config dataclass. Subclasses can change this to a different dataclass if needed.

        Returns:
            The output can be a list of
            objects, list of list of objects, tuple of objects, tuple of list of objects, or a dict of list of objects.
            Its type is defined in `TranscriptionReturnType`.
        """
        log_probs = outputs.pop('log_probs')
        encoded_len = outputs.pop('encoded_lengths')
        enc_states = outputs.pop('encoder_states')
        enc_mask = outputs.pop('encoder_mask')
        decoder_input_ids = outputs.pop('decoder_input_ids')

        del log_probs, encoded_len

        best_hypotheses, all_hypotheses = self.decoding.decode_predictions_tensor(
            encoder_hidden_states=enc_states,
            encoder_input_mask=enc_mask,
            decoder_input_ids=decoder_input_ids,
            return_hypotheses=trcfg.return_hypotheses,
        )

        del enc_states, enc_mask, decoder_input_ids
        if all_hypotheses is None:
            return best_hypotheses
        return best_hypotheses, all_hypotheses

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
            # when using a list of audio files instead of a manifest (added from TranscrptionMixin)
            manifest_filepath = os.path.join(config['temp_dir'], 'manifest.json')
            batch_size = min(config['batch_size'], len(config['paths2audio_files']))
        dl_config = {
            'manifest_filepath': manifest_filepath,
            'sample_rate': self.preprocessor._sample_rate,
            'batch_size': batch_size,
            'trim_silence': False,
            'shuffle': False,
            'num_workers': min(batch_size, os.cpu_count() - 1),
            'pin_memory': True,
            'use_lhotse': True,
            'use_bucketing': False,
            'drop_last': False,
            'text_field': config.get('text_field', 'answer'),
            'lang_field': config.get('lang_field', 'target_lang'),
            'channel_selector': config.get('channel_selector', None),
        }

        temporary_datalayer = self._setup_dataloader_from_config(config=DictConfig(dl_config))
        return temporary_datalayer

    def _transcribe_on_end(self, trcfg: MultiTaskTranscriptionConfig):
        """
        Internal function to teardown the model after transcription. Perform all teardown and post-checks here.

        Args:
            trcfg: The transcription config dataclass. Subclasses can change this to a different dataclass if needed.
        """
        super()._transcribe_on_end(trcfg)

        self.transf_decoder.unfreeze(partial=True)

    def _may_be_make_dict_and_fix_paths(self, json_items, manifest_path, trcfg: MultiTaskTranscriptionConfig):
        """
        Utility method to convert a list of strings to a list of dictionaries.

        Args:
            json_items: A list of strings or dictionaries.
            manifest_path: A path to a manifest file.
            trcfg: The transcription config dataclass. Subclasses can change this to a different dataclass if needed.

        Returns:
            A list of dictionaries with the audio file paths fixed.
        """
        # This method is a legacy helper for Canary that checks whether prompt slot values were provided
        # in the input manifest and if not, it injects the defaults.
        out_json_items = []
        for item in json_items:
            if isinstance(item, str):
                # assume it is a path to audio file
                entry = {
                    'audio_filepath': item,
                    'duration': 100000,
                }
            elif isinstance(item, dict):
                entry = item
                entry['audio_filepath'] = get_full_path(entry['audio_filepath'], manifest_file=manifest_path)
            else:
                raise ValueError(f"Expected str or dict, got {type(item)}")
            default_turn = [t for t in trcfg.prompt if t["role"] == "user"]
            default_turn = default_turn[0]["slots"] if default_turn else {}
            for k, dv in (("source_lang", "en"), ("target_lang", "en"), ("taskname", "asr"), ("pnc", "yes")):
                if k not in entry:
                    # last-chance fallback injecting legacy Canary defaults if none were provided.
                    entry[k] = default_turn.get(k, dv)
            out_json_items.append(entry)
        return out_json_items

    @classmethod
    def get_transcribe_config(cls) -> MultiTaskTranscriptionConfig:
        """
        Utility method that returns the default config for transcribe() function.

        Returns:
            A dataclass
        """
        return MultiTaskTranscriptionConfig()

    def predict_step(
        self, batch: PromptedAudioToTextMiniBatch, batch_idx=0, dataloader_idx=0, has_processed_signal=False
    ):
        if has_processed_signal:
            processed_signal = batch.audio
            processed_signal_length = batch.audio_lens
            signal = None
            signal_len = None
        else:
            processed_signal = None
            processed_signal_length = None
            signal = batch.audio
            signal_len = batch.audio_lens

        _, _, enc_states, enc_mask = self.forward(
            input_signal=signal,
            input_signal_length=signal_len,
            processed_signal=processed_signal,
            processed_signal_length=processed_signal_length,
        )

        text = self.decoding.decode_predictions_tensor(
            encoder_hidden_states=enc_states,
            encoder_input_mask=enc_mask,
            decoder_input_ids=batch.prompt,
            return_hypotheses=False,
        )[0]
        if batch.cuts:
            return list(zip(batch.cuts, text))
        else:
            return text

    @property
    def adapter_module_names(self) -> List[str]:
        return ['', 'encoder', 'transf_encoder', 'transf_decoder']

    @property
    def oomptimizer_schema(self) -> dict:
        """
        Return a typing schema for optimal batch size calibration for various
        sequence lengths using OOMptimizer.
        """
        return {
            "cls": PromptedAudioToTextMiniBatch,
            "inputs": [
                {"name": "audio", "type": NeuralType(("B", "T"), AudioSignal()), "seq_length": "input"},
                {"name": "audio_lens", "type": NeuralType(("B",), LengthsType()), "seq_length": "input"},
                {
                    "name": "prompted_transcript",
                    "type": NeuralType(("B", "T"), LabelsType()),
                    "seq_length": "output",
                    "vocab_size": self.tokenizer.vocab_size,
                },
                {
                    "name": "prompted_transcript_lens",
                    "type": NeuralType(("B",), LengthsType()),
                    "seq_length": "output",
                },
                {"name": "transcript", "type": "dummy"},
                {"name": "transcript_lens", "type": "dummy"},
                {"name": "prompt", "type": "dummy"},
                {"name": "prompt_lens", "type": "dummy"},
            ],
        }


def parse_multitask_prompt(prompt: dict | None) -> list[dict]:
    if prompt is None or not prompt:
        return []

    # Case 1.
    # Multi-turn prompting format. This format conforms to PromptFormatter API and needs no further modification.
    # This format allows to condition the model on chat history, system+user prompts, etc.
    # Example:
    # model.transcribe(
    #     audio,
    #     turns=[
    #         dict(
    #             role="user",
    #             slots=dict(
    #                 source_lang='en', target_lang='de', task='asr', pnc=True, context='translate this text'
    #             ),
    #         ),
    #         dict(
    #             role="assistant",
    #             slots=dict(message="Calculating the translation of given text. Do you want to proceed?"),
    #         ),
    #         dict(
    #             role="user",
    #             slots=dict(
    #                 source_lang='en', target_lang='de', task='asr', pnc=True, context='Yes, please proceed.'
    #             ),
    #         ),
    #     ],
    # )
    if 'turns' in prompt:
        assert (
            len(prompt) == 1
            and isinstance(prompt["turns"], list)
            and all(isinstance(t, dict) and "role" in t and "slots" in t for t in prompt["turns"])
        ), (
            f"When providing a multi-turn prompt through 'turns', no other keys are allowed "
            f"and the value under prompt['turns'] must be a list of dicts with roles and slot values "
            f"(we received {prompt=})"
        )
        return prompt["turns"]

    values_are_dicts = any(isinstance(v, dict) for k, v in prompt.items() if k != "slots")
    assert not values_are_dicts, (
        f"We don't support dict values for prompt keys other than 'slots'. " f"We received {prompt=}"
    )

    # Case 2.
    # Single-turn prompting format with explicitly provided role and slot names and values.
    # We create a 1-item multi-turn prompt from this input.
    # Example:
    # model.transcribe(
    #     audio,
    #     role="user",
    #     slots=dict(source_lang='en', target_lang='de', task='asr', pnc=True, context='translate this text'),
    # )
    if "role" in prompt and "slots" in prompt:
        assert isinstance(prompt["slots"], dict), (
            f"When providing a single-turn prompt through 'role', 'slots' must also be provided "
            f"(we received {prompt=})."
        )
        return [prompt]

    # Case 3.
    # Legacy prompting format for Canary-1B preserved for backward compatibility.
    # Extra fields are converted to a single-turn prompt with role "user" (unless overridden with 'role').
    # Example:
    # model.transcribe(
    #     audio, pnc=True, source_lang='en', target_lang='de', task='asr', context='translate this text'
    # )
    role = prompt.pop("role", "user")
    return [dict(role=role, slots=prompt)]
