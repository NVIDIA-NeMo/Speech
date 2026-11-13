# SPDX-FileCopyrightText: Copyright (c) 2020, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

import functools
import json
import os
import tarfile
from abc import ABC, abstractmethod
from pathlib import PurePosixPath
from typing import List, Optional

import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from torch import Tensor

import nemo.collections.asr.models as asr_models
from nemo.collections.asr.parts.mixins.asr_adapter_mixins import ASRAdapterModelMixin
from nemo.collections.asr.parts.mixins.streaming import StreamingEncoder
from nemo.collections.asr.parts.utils import asr_module_utils
from nemo.collections.asr.parts.utils.eou_utils import (
    append_hypothesis_segment_,
    find_restart_frame,
    get_restart_token_ids,
    reset_hypothesis_decoder_state_,
)
from nemo.collections.asr.parts.utils.prompt_language_utils import resolve_prompt_index
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from nemo.collections.asr.parts.utils.tokenizer_utils import (
    extract_capitalized_tokens_from_vocab,
    extract_punctuation_from_vocab,
)
from nemo.collections.common import tokenizers
from nemo.collections.common.parts.utils import ModuleBlock
from nemo.core.connectors.save_restore_connector import SaveRestoreConnector
from nemo.utils import app_state, logging
from nemo.utils.file_utils import robust_copy


class ASRBPEMixin(ABC):
    """ASR BPE Mixin class that sets up a Tokenizer via a config

    This mixin class adds the method `_setup_tokenizer(...)`, which can be used by ASR models
    which depend on subword tokenization.

    The setup_tokenizer method adds the following parameters to the class -
        -   tokenizer_cfg: The resolved config supplied to the tokenizer (with `dir` and `type` arguments).
        -   tokenizer_dir: The directory path to the tokenizer vocabulary + additional metadata.
        -   tokenizer_type: The type of the tokenizer. Currently supports `bpe` and `wpe`, as well as `agg`.
        -   vocab_path: Resolved path to the vocabulary text file.

    In addition to these variables, the method will also instantiate and preserve a tokenizer
    (subclass of TokenizerSpec) if successful, and assign it to self.tokenizer.

    The mixin also supports aggregate tokenizers, which consist of ordinary, monolingual tokenizers.
    If a conversion between a monolongual and an aggregate tokenizer (or vice versa) is detected,
    all registered artifacts will be cleaned up.
    """

    # this will be used in configs and nemo artifacts
    AGGREGATE_TOKENIZERS_DICT_PREFIX = 'langs'

    @staticmethod
    def _get_extracted_tokenizer_name(member_name: str) -> str:
        member_file_name = PurePosixPath(member_name).name
        new_name = member_file_name.split("_")[1:]
        if len(new_name) > 1:
            return "_".join(new_name)
        return new_name[0]

    def _setup_tokenizer(self, tokenizer_cfg: DictConfig):
        tokenizer_type = tokenizer_cfg.get('type')
        if tokenizer_type is None:
            raise ValueError("`tokenizer.type` cannot be None")
        elif tokenizer_type.lower() == 'agg':
            self._setup_aggregate_tokenizer(tokenizer_cfg)
        else:
            self._setup_monolingual_tokenizer(tokenizer_cfg)

        self._derive_tokenizer_properties()

    def _setup_monolingual_tokenizer(self, tokenizer_cfg: DictConfig):
        # Prevent tokenizer parallelism (unless user has explicitly set it)
        if 'TOKENIZERS_PARALLELISM' not in os.environ:
            os.environ['TOKENIZERS_PARALLELISM'] = 'false'

        self.tokenizer_cfg = OmegaConf.to_container(tokenizer_cfg, resolve=True)  # type: dict
        self.tokenizer_dir = self.tokenizer_cfg.pop('dir')  # Remove tokenizer directory
        self.tokenizer_type = self.tokenizer_cfg.pop('type').lower()  # Remove tokenizer_type

        self.hf_tokenizer_kwargs = self.tokenizer_cfg.pop("hf_kwargs", {})  # Remove HF tokenizer kwargs

        # just in case the previous tokenizer was an aggregate
        self._cleanup_aggregate_config_and_artifacts_if_needed()

        # Preserve config
        if hasattr(self, 'cfg') and 'tokenizer' in self.cfg:
            self.cfg.tokenizer.dir = self.tokenizer_dir
            self.cfg.tokenizer.type = self.tokenizer_type

            if 'hf_kwargs' in tokenizer_cfg:
                with open_dict(self.cfg.tokenizer):
                    self.cfg.tokenizer.hf_kwargs = tokenizer_cfg.get('hf_kwargs')

        if self.tokenizer_type not in ['bpe', 'wpe']:
            raise ValueError(
                "`tokenizer.type` must be either `bpe` for SentencePiece tokenizer or "
                "`wpe` for BERT based tokenizer"
            )

        if self.tokenizer_type == 'bpe':
            # This is a BPE Tokenizer
            if 'model_path' in self.tokenizer_cfg:
                model_path = self.tokenizer_cfg.get('model_path')
            else:
                model_path = os.path.join(self.tokenizer_dir, 'tokenizer.model')
            model_path = self.register_artifact('tokenizer.model_path', model_path)
            self.model_path = model_path

            if 'special_tokens' in self.tokenizer_cfg:
                special_tokens = self.tokenizer_cfg['special_tokens']

                if special_tokens is not None:
                    raise ValueError("`special_tokens` are no longer supported for SentencePiece based tokenizers.")

            if "custom_tokenizer" in self.tokenizer_cfg:
                self.tokenizer = self.from_config_dict(
                    {"_target_": tokenizer_cfg["custom_tokenizer"]["_target_"], "model_path": model_path}
                )
            else:
                self.tokenizer = tokenizers.SentencePieceTokenizer(model_path=model_path)

            if 'vocab_path' in self.tokenizer_cfg:
                vocab_path = self.tokenizer_cfg.get('vocab_path')
            else:
                vocab_path = os.path.join(self.tokenizer_dir, 'vocab.txt')
            vocab_path = self.register_artifact('tokenizer.vocab_path', vocab_path)
            self.vocab_path = vocab_path

            try:
                if 'spe_tokenizer_vocab' in self.tokenizer_cfg:
                    spe_vocab_path = self.tokenizer_cfg.get('spe_tokenizer_vocab')
                else:
                    spe_vocab_path = os.path.join(self.tokenizer_dir, 'tokenizer.vocab')
                spe_vocab_path = self.register_artifact('tokenizer.spe_tokenizer_vocab', spe_vocab_path)
                self.spe_vocab_path = spe_vocab_path
            except FileNotFoundError:
                # fallback case for older checkpoints that did not preserve the tokenizer.vocab
                self.spe_vocab_path = None

            vocabulary = {}
            for i in range(self.tokenizer.vocab_size):
                piece = self.tokenizer.ids_to_tokens([i])
                piece = piece[0]
                vocabulary[piece] = i + 1

            # wrapper method to get vocabulary conveniently
            def get_vocab():
                return vocabulary

            # attach utility values to the tokenizer wrapper
            self.tokenizer.tokenizer.vocab_size = len(vocabulary)
            self.tokenizer.tokenizer.get_vocab = get_vocab
            self.tokenizer.tokenizer.all_special_tokens = self.tokenizer.special_token_to_id

        else:
            # This is a WPE Tokenizer
            # If path from previous registration exists, remove it
            if 'vocab_path' in self.tokenizer_cfg:
                vocab_path = self.tokenizer_cfg.get('vocab_path')
            else:
                vocab_path = os.path.join(self.tokenizer_dir, 'vocab.txt')
            vocab_path = self.register_artifact('tokenizer.vocab_path', vocab_path)
            self.vocab_path = vocab_path

            # If path from previous registration exists, remove it
            if 'vocab_path' in self.tokenizer_cfg:
                self.tokenizer_cfg.pop('vocab_path')

            self.tokenizer = tokenizers.AutoTokenizer(
                pretrained_model_name='bert-base-cased',
                vocab_file=self.vocab_path,
                mask_token=self.hf_tokenizer_kwargs.get('mask_token', None),
                bos_token=self.hf_tokenizer_kwargs.get('bos_token', None),
                eos_token=self.hf_tokenizer_kwargs.get('eos_token', None),
                pad_token=self.hf_tokenizer_kwargs.get('pad_token', None),
                sep_token=self.hf_tokenizer_kwargs.get('sep_token', None),
                cls_token=self.hf_tokenizer_kwargs.get('cls_token', None),
                unk_token=self.hf_tokenizer_kwargs.get('unk_token', None),
                use_fast=self.hf_tokenizer_kwargs.get('use_fast', False),
            )

        logging.info(
            "Tokenizer {} initialized with {} tokens".format(
                self.tokenizer.__class__.__name__, self.tokenizer.vocab_size
            )
        )

    def _setup_aggregate_tokenizer(self, tokenizer_cfg: DictConfig):
        # Prevent tokenizer parallelism (unless user has explicitly set it)
        if 'TOKENIZERS_PARALLELISM' not in os.environ:
            os.environ['TOKENIZERS_PARALLELISM'] = 'false'

        self.tokenizer_cfg = OmegaConf.to_container(tokenizer_cfg, resolve=True)  # type: dict

        # the aggregate tokenizer does not have one tokenizer_dir but multiple ones
        self.tokenizer_dir = None

        self.tokenizer_cfg.pop('dir', None)  # Remove tokenizer directory, if any
        # Remove tokenizer_type -- obviously if we are here, the type is 'agg'
        self.tokenizer_type = self.tokenizer_cfg.pop('type').lower()

        # the aggregate tokenizer should not have these
        self.hf_tokenizer_kwargs = {}
        self.tokenizer_cfg.pop("hf_kwargs", {})  # Remove HF tokenizer kwargs, if any

        logging.info('_setup_tokenizer: detected an aggregate tokenizer')
        # need to de-register any monolingual config items if they exist
        self._cleanup_monolingual_and_aggregate_config_and_artifacts_if_needed()

        # overwrite tokenizer type
        if hasattr(self, 'cfg') and 'tokenizer' in self.cfg:
            self.cfg.tokenizer.type = self.tokenizer_type

        tokenizers_dict = {}
        # init each of the monolingual tokenizers found in the config and assemble into  AggregateTokenizer
        for lang, tokenizer_config in self.tokenizer_cfg[self.AGGREGATE_TOKENIZERS_DICT_PREFIX].items():
            (
                tokenizer,
                model_path,
                vocab_path,
                spe_vocab_path,
            ) = self._make_tokenizer(tokenizer_config, lang)

            tokenizers_dict[lang] = tokenizer
            if hasattr(self, 'cfg'):
                with open_dict(self.cfg.tokenizer):
                    self.cfg.tokenizer[self.AGGREGATE_TOKENIZERS_DICT_PREFIX][lang]['dir'] = self.tokenizer_cfg[
                        self.AGGREGATE_TOKENIZERS_DICT_PREFIX
                    ][lang]['dir']
                    self.cfg.tokenizer[self.AGGREGATE_TOKENIZERS_DICT_PREFIX][lang]['type'] = self.tokenizer_cfg[
                        self.AGGREGATE_TOKENIZERS_DICT_PREFIX
                    ][lang]['type']

        if "custom_tokenizer" in tokenizer_cfg:
            # Class which implements this is usually a ModelPT, has access to Serializable mixin by extension
            self.tokenizer = self.from_config_dict(
                {"_target_": tokenizer_cfg["custom_tokenizer"]["_target_"], "tokenizers": tokenizers_dict}
            )
        else:
            self.tokenizer = tokenizers.AggregateTokenizer(tokenizers_dict)

    def _make_tokenizer(self, tokenizer_cfg: DictConfig, lang=None):

        tokenizer_type = tokenizer_cfg.get('type').lower()
        tokenizer_dir = tokenizer_cfg.get('dir')

        if tokenizer_type not in ['bpe', 'wpe']:
            raise ValueError(
                '`tokenizer.type` must be either `bpe` for SentencePiece tokenizer or' '`wpe` for BERT based tokenizer'
            )

        # defaults
        model_path = None
        vocab_path = None
        spe_vocab_path = None

        if tokenizer_type == 'bpe':
            # This is a BPE Tokenizer
            if 'model_path' in tokenizer_cfg:
                model_path = tokenizer_cfg.get('model_path')
            else:
                model_path = os.path.join(tokenizer_dir, 'tokenizer.model')

            model_path = self.register_artifact(
                'tokenizer.' + self.AGGREGATE_TOKENIZERS_DICT_PREFIX + '.' + lang + '.model_path', model_path
            )

            if 'special_tokens' in tokenizer_cfg:
                special_tokens = tokenizer_cfg['special_tokens']
                if special_tokens is not None:
                    raise ValueError('`special_tokens` are no longer supported for SentencePiece based tokenizers.')

            # Update special tokens
            tokenizer = tokenizers.SentencePieceTokenizer(model_path=model_path)

            if 'vocab_path' in tokenizer_cfg:
                vocab_path = tokenizer_cfg.get('vocab_path')
            else:
                vocab_path = os.path.join(tokenizer_dir, 'vocab.txt')

            vocab_path = self.register_artifact(
                'tokenizer.' + self.AGGREGATE_TOKENIZERS_DICT_PREFIX + '.' + lang + '.vocab_path', vocab_path
            )

            try:
                if 'spe_tokenizer_vocab' in tokenizer_cfg:
                    spe_vocab_path = tokenizer_cfg.get('spe_tokenizer_vocab')
                else:
                    spe_vocab_path = os.path.join(tokenizer_dir, 'tokenizer.vocab')

                spe_vocab_path = self.register_artifact(
                    'tokenizer.' + self.AGGREGATE_TOKENIZERS_DICT_PREFIX + '.' + lang + '.spe_tokenizer_vocab',
                    spe_vocab_path,
                )

            except FileNotFoundError:
                # fallback case for older checkpoints that did not preserve the tokenizer.vocab
                spe_vocab_path = None

            vocabulary = {}
            for i in range(tokenizer.vocab_size):
                piece = tokenizer.ids_to_tokens([i])
                piece = piece[0]
                vocabulary[piece] = i + 1

            # wrapper method to get vocabulary conveniently
            def get_vocab():
                return vocabulary

            # attach utility values to the tokenizer wrapper
            tokenizer.tokenizer.vocab_size = len(vocabulary)
            tokenizer.tokenizer.get_vocab = get_vocab
            tokenizer.tokenizer.all_special_tokens = tokenizer.special_token_to_id

        else:
            # This is a WPE Tokenizer
            # If path from previous registration exists, remove it
            if 'vocab_path' in tokenizer_cfg:
                vocab_path = tokenizer_cfg.get('vocab_path')
            else:
                vocab_path = os.path.join(tokenizer_dir, 'vocab.txt')

            vocab_path = self.register_artifact(
                'tokenizer.' + self.AGGREGATE_TOKENIZERS_DICT_PREFIX + '.' + lang + '.vocab_path', vocab_path
            )

            # If path from previous registration exists, remove it
            if 'vocab_path' in tokenizer_cfg:
                tokenizer_cfg.pop('vocab_path')

            hf_tokenizer_kwargs = tokenizer_cfg.get('hf_kwargs', {})
            tokenizer = tokenizers.AutoTokenizer(
                pretrained_model_name=hf_tokenizer_kwargs.get('pretrained_model_name', 'bert-base-cased'),
                vocab_file=vocab_path,
                mask_token=hf_tokenizer_kwargs.get('mask_token', None),
                bos_token=hf_tokenizer_kwargs.get('bos_token', None),
                eos_token=hf_tokenizer_kwargs.get('eos_token', None),
                pad_token=hf_tokenizer_kwargs.get('pad_token', None),
                sep_token=hf_tokenizer_kwargs.get('sep_token', None),
                cls_token=hf_tokenizer_kwargs.get('cls_token', None),
                unk_token=hf_tokenizer_kwargs.get('unk_token', None),
                use_fast=hf_tokenizer_kwargs.get('use_fast', False),
            )

        logging.info(
            'Tokenizer {} initialized with {} tokens'.format(tokenizer.__class__.__name__, tokenizer.vocab_size)
        )

        return tokenizer, model_path, vocab_path, spe_vocab_path

    def _cleanup_monolingual_and_aggregate_config_and_artifacts_if_needed(self):
        """
        Clean ups any monolingual and some aggregate config items and artifacts.
        We need to do this when we switch from a monolingual tokenizer to an aggregate one
        or go between aggregate tokenizers which could have a different number of languages
        """
        if hasattr(self, 'cfg'):
            with open_dict(self.cfg.tokenizer):
                self.cfg.tokenizer.pop('dir', None)
                self.cfg.tokenizer.pop('model_path', None)
                self.cfg.tokenizer.pop('vocab_path', None)
                self.cfg.tokenizer.pop('spe_tokenizer_vocab', None)
                self.cfg.tokenizer.pop('hf_kwargs', None)

        # need to de-register any monolingual artifacts if they exist
        if hasattr(self, 'artifacts'):
            self.artifacts.pop('tokenizer.model_path', None)
            self.artifacts.pop('tokenizer.vocab_path', None)
            self.artifacts.pop('tokenizer.spe_tokenizer_vocab', None)

            # just in case we are replacing one aggregate tokenizer with another one, we better
            # clean up the old aggregate artifacts as well
            for akey in list(self.artifacts.keys()):
                if akey.startswith('tokenizer.' + self.AGGREGATE_TOKENIZERS_DICT_PREFIX + '.'):
                    self.artifacts.pop(akey)

    def _cleanup_aggregate_config_and_artifacts_if_needed(self):
        """
        Clean ups any aggregate config items and artifacts.
        We need to do this when we switch from an aggregate tokenizer to a monolingual one
        """
        if hasattr(self, 'cfg'):
            with open_dict(self.cfg.tokenizer):
                self.cfg.tokenizer.pop(self.AGGREGATE_TOKENIZERS_DICT_PREFIX, None)

        # clean up the old aggregate artifacts as well
        if hasattr(self, 'artifacts'):
            for akey in list(self.artifacts.keys()):
                if akey.startswith('tokenizer.' + self.AGGREGATE_TOKENIZERS_DICT_PREFIX + '.'):
                    self.artifacts.pop(akey)

    def save_tokenizers(self, directory: str):
        """
        Save the model tokenizer(s) to the specified directory.

        Args:
            directory: The directory to save the tokenizer(s) to.
        """
        if not hasattr(self, 'cfg'):
            raise RuntimeError(
                "The model has not been initialized with a tokenizer yet. Please call the model's "
                "__init__ and _setup_tokenizer methods first."
            )

        if self.tokenizer_type == 'agg':
            for lang in self.tokenizer.langs:
                subconfig = self.cfg.tokenizer.langs.get(lang)
                new_dir = os.path.join(directory, lang)
                self._extract_tokenizer_from_config(subconfig, new_dir)
        else:
            self._extract_tokenizer_from_config(self.cfg.tokenizer, directory)

    def _extract_tokenizer_from_config(self, tokenizer_cfg: DictConfig, dir: str):
        """
        Extracts the tokenizer from the config and write the objects to dir.
        The file may be from a local path (new model init) or from a .nemo file (restored model).
        If its from a newly initialized model, the file is copied to dir.
        If its from a restored model, the file is extracted from the .nemo file and copied to dir.

        Args:
            tokenizer_cfg: The tokenizer config to extract the tokenizer from.
            dir: The directory to write the tokenizer objects to.
        """
        if not os.path.exists(dir):
            os.makedirs(dir, exist_ok=True)

        nemo_file_objects = []

        for k, v in tokenizer_cfg.items():
            # Check if the value is a filepath (new model init) or has `nemo:` in it (restored model)
            if isinstance(v, str) and os.path.exists(v):
                # local file from first instantiation
                loc = robust_copy(v, dir)
                logging.info(f"Saved {k} at {loc}")

            if isinstance(v, str) and v.startswith('nemo:'):
                nemo_object_name = v[5:]
                nemo_file_objects.append(nemo_object_name)

        if len(nemo_file_objects) > 0:
            logging.debug(f"Copying the following nemo file objects to {dir}: {nemo_file_objects}")

            if not hasattr(self, 'model_guid'):
                raise ValueError(
                    "The model does not have a model_guid attribute. "
                    "Please ensure that the model has been restored from a .nemo file."
                )

            appstate = app_state.AppState()
            restore_path = appstate.get_model_metadata_from_guid(self.model_guid).restoration_path
            if restore_path is None:
                raise ValueError(
                    "The model has not been restored from a .nemo file. Cannot extract the tokenizer "
                    "as the nemo file cannot be located."
                )

            # Read the nemo file without fully extracting all contents
            # we start with an assumption of uncompressed tar,
            # which should be true for versions 1.7.0 and above
            tar_header = "r:"
            try:
                tar_test = tarfile.open(restore_path, tar_header)
                tar_test.close()
            except tarfile.ReadError:
                # can be older checkpoint => try compressed tar
                tar_header = "r:gz"
            with tarfile.open(restore_path, tar_header) as tar:
                for nemo_object_name in nemo_file_objects:
                    members = [x for x in tar.getmembers() if nemo_object_name in x.name]
                    extracted_members = SaveRestoreConnector._safe_extract(tar, dir, members=members)
                    for member in extracted_members:
                        new_name = self._get_extracted_tokenizer_name(member.name)
                        os.rename(os.path.join(dir, member.name), os.path.join(dir, new_name))

                        logging.info(f"Saved {nemo_object_name} at {os.path.join(dir, new_name)}")

    def _derive_tokenizer_properties(self):
        vocab = self.tokenizer.tokenizer.get_vocab()

        self.tokenizer.supports_capitalization = bool(extract_capitalized_tokens_from_vocab(vocab))
        self.tokenizer.supported_punctuation = extract_punctuation_from_vocab(vocab)


class ASRModuleMixin(ASRAdapterModelMixin):
    """
    ASRModuleMixin is a mixin class added to ASR models in order to add methods that are specific
    to a particular instantiation of a module inside of an ASRModel.

    Each method should first check that the module is present within the subclass, and support additional
    functionality if the corresponding module is present.
    """

    def change_conv_asr_se_context_window(self, context_window: int, update_config: bool = True):
        """
        Update the context window of the SqueezeExcitation module if the provided model contains an
        `encoder` which is an instance of `ConvASREncoder`.

        Args:
            context_window:  An integer representing the number of input timeframes that will be used
                to compute the context. Each timeframe corresponds to a single window stride of the
                STFT features.

                Say the window_stride = 0.01s, then a context window of 128 represents 128 * 0.01 s
                of context to compute the Squeeze step.
            update_config: Whether to update the config or not with the new context window.
        """
        asr_module_utils.change_conv_asr_se_context_window(
            self, context_window=context_window, update_config=update_config
        )

    def change_attention_model(
        self,
        self_attention_model: str = None,
        att_context_size: List[int] = None,
        update_config: bool = True,
        rope_base: float = None,
        rotary_fraction: float = None,
        self_attn: ModuleBlock | None = None,
    ):
        """
        Update the self_attention_model if function is available in encoder.

        Args:
            self_attention_model (str): type of the attention layer and positional encoding

                'rel_pos':
                    relative positional embedding and Transformer-XL

                'rel_pos_local_attn':
                    relative positional embedding and Transformer-XL with local attention using
                    overlapping windows. Attention context is determined by att_context_size parameter.

                'abs_pos':
                    absolute positional embedding and Transformer

                'rope':
                    rotary position embedding

                If None is provided, the self_attention_model isn't changed. Defauts to None.
            att_context_size (List[int]): List of 2 ints corresponding to left and right attention context sizes,
                or None to keep as it is. Defauts to None.
            update_config (bool): Whether to update the config or not with the new attention model.
                Defaults to True.
            rope_base (float): Theta base for rotary position embedding. Only used when
                ``self_attention_model='rope'``. If None, the existing value is kept.
            rotary_fraction (float): Fraction of the per-head dim to rotate. Only used when
                ``self_attention_model='rope'``. If None, the existing value is kept.
            self_attn: the block the new attention layers are built from, see `ConformerEncoder.change_attention_model`;
                an encoder built from a `self_attn` block needs a new block here to change its attention model,
                and with `update_config` the block must be a Hydra section.
        """
        if (
            self_attention_model is None
            and att_context_size is None
            and rope_base is None
            and rotary_fraction is None
            and self_attn is None
        ):
            return

        if not hasattr(self, 'encoder'):
            logging.info(
                "Could not change the self_attention_model in encoder "
                "since the model provided does not contain an `encoder` module in its config."
            )
            return

        if not hasattr(self.encoder, "change_attention_model"):
            logging.info("Model encoder doesn't have a change_attention_model method ")
            return

        self.encoder.change_attention_model(
            self_attention_model=self_attention_model,
            att_context_size=att_context_size,
            update_config=update_config,
            device=self.device,
            rope_base=rope_base,
            rotary_fraction=rotary_fraction,
            self_attn=self_attn,
        )
        if update_config:
            with open_dict(self.cfg):
                self.cfg.encoder.self_attention_model = self.encoder.self_attention_model
                self.cfg.encoder.att_context_size = self.encoder.att_context_size
                if self.encoder.self_attention_model == 'rope':
                    self.cfg.encoder.rope_base = self.encoder._cfg.rope_base
                    self.cfg.encoder.rotary_fraction = self.encoder._cfg.rotary_fraction
                if self_attn is not None:
                    self.cfg.encoder.self_attn = self.encoder._cfg.self_attn

    def change_subsampling_conv_chunking_factor(
        self, subsampling_conv_chunking_factor: int, update_config: bool = True
    ):
        """
        Update the conv_chunking_factor (int) if function is available in encoder.
        Default is 1 (auto)
        Set it to -1 (disabled) or to a specific value (power of 2) if you OOM in the conv subsampling layers

        Args:
            conv_chunking_factor (int)
        """

        if not hasattr(self, 'encoder'):
            logging.info(
                "Could not call the change_subsampling_conv_chunking_factor method in encoder "
                "since the model provided does not contain an `encoder` module in its config."
            )
            return

        if not hasattr(self.encoder, "change_subsampling_conv_chunking_factor"):
            logging.info("Model encoder doesn't have a change_subsampling_conv_chunking_factor method ")
            return

        self.encoder.change_subsampling_conv_chunking_factor(subsampling_conv_chunking_factor)
        if update_config:
            with open_dict(self.cfg):
                self.cfg.encoder.subsampling_conv_chunking_factor = subsampling_conv_chunking_factor

    def _apply_prompt_to_encoded(self, encoded: Tensor) -> Tensor:
        """Hook for prompt-conditioned subclasses to inject a language prompt
        into the encoder output. Default: no-op. See ``PromptStreamingMixin``
        for the prompt-aware override."""
        return encoded

    def conformer_stream_step(
        self,
        processed_signal: Tensor,
        processed_signal_length: Tensor = None,
        cache_last_channel: Tensor = None,
        cache_last_time: Tensor = None,
        cache_last_channel_len: Tensor = None,
        keep_all_outputs: bool = True,
        previous_hypotheses: List[Hypothesis] = None,
        previous_pred_out: Tensor = None,
        drop_extra_pre_encoded: int = None,
        return_transcription: bool = True,
        return_log_probs: bool = False,
        bypass_pre_encode: bool = False,
    ):
        """
        It simulates a forward step with caching for streaming purposes.
        It supports the ASR models where their encoder supports streaming like Conformer.
        Args:
            processed_signal: the input audio signals
            processed_signal_length: the length of the audios
            cache_last_channel: the cache tensor for last channel layers like MHA
            cache_last_channel_len: lengths for cache_last_channel
            cache_last_time: the cache tensor for last time layers like convolutions
            keep_all_outputs: if set to True, would not drop the extra outputs specified by encoder.streaming_cfg.valid_out_len
            previous_hypotheses: the hypotheses from the previous step for RNNT models
            previous_pred_out: the predicted outputs from the previous step for CTC models
            drop_extra_pre_encoded: number of steps to drop from the beginning of the outputs after the downsampling module. This can be used if extra paddings are added on the left side of the input.
            return_transcription: whether to decode and return the transcriptions. It can not get disabled for Transducer models.
            return_log_probs: whether to return the log probs, only valid for ctc model

        Returns:
            greedy_predictions: the greedy predictions from the decoder
            all_hyp_or_transcribed_texts: the decoder hypotheses for Transducer models and the transcriptions for CTC models
            cache_last_channel_next: the updated tensor cache for last channel layers to be used for next streaming step
            cache_last_time_next: the updated tensor cache for last time layers to be used for next streaming step
            cache_last_channel_next_len: the updated lengths for cache_last_channel
            best_hyp: the best hypotheses for the Transducer models
            log_probs: the logits tensor of current streaming chunk, only returned when return_log_probs=True
            encoded_len: the length of the output log_probs + history chunk log_probs, only returned when return_log_probs=True
        """
        if not isinstance(self, asr_models.EncDecRNNTModel) and not isinstance(self, asr_models.EncDecCTCModel):
            raise NotImplementedError(f"stream_step does not support {type(self)}!")

        if not isinstance(self.encoder, StreamingEncoder):
            raise NotImplementedError("Encoder of this model does not support streaming!")

        if isinstance(self, asr_models.EncDecRNNTModel) and return_transcription is False:
            logging.info(
                "return_transcription can not be False for Transducer models as decoder returns the transcriptions too."
            )

        if not isinstance(self, asr_models.EncDecCTCModel) and return_log_probs is True:
            logging.info("return_log_probs can only be True for CTC models.")

        (
            encoded,
            encoded_len,
            cache_last_channel_next,
            cache_last_time_next,
            cache_last_channel_next_len,
        ) = self.encoder.cache_aware_stream_step(
            processed_signal=processed_signal,
            processed_signal_length=processed_signal_length,
            cache_last_channel=cache_last_channel,
            cache_last_time=cache_last_time,
            cache_last_channel_len=cache_last_channel_len,
            keep_all_outputs=keep_all_outputs,
            drop_extra_pre_encoded=drop_extra_pre_encoded,
            bypass_pre_encode=bypass_pre_encode,
        )

        encoded = self._apply_prompt_to_encoded(encoded)

        if isinstance(self, asr_models.EncDecCTCModel) or (
            isinstance(self, asr_models.EncDecHybridRNNTCTCModel) and self.cur_decoder == "ctc"
        ):
            if hasattr(self, "ctc_decoder"):
                decoding = self.ctc_decoding
                decoder = self.ctc_decoder
            else:
                decoding = self.decoding
                decoder = self.decoder

            log_probs = decoder(encoder_output=encoded)
            predictions_tensor = log_probs.argmax(dim=-1, keepdim=False)

            # Concatenate the previous predictions with the current one to have the full predictions.
            # We drop the extra predictions for each sample by using the lengths returned by the encoder (encoded_len)
            # Then create a list of the predictions for the batch. The predictions can have different lengths because of the paddings.
            greedy_predictions = []
            if return_transcription:
                all_hyp_or_transcribed_texts = []
            else:
                all_hyp_or_transcribed_texts = None
            for preds_idx, preds in enumerate(predictions_tensor):
                if encoded_len is None:
                    preds_cur = predictions_tensor[preds_idx]
                else:
                    preds_cur = predictions_tensor[preds_idx, : encoded_len[preds_idx]]
                if previous_pred_out is not None:
                    greedy_predictions_concat = torch.cat((previous_pred_out[preds_idx], preds_cur), dim=-1)
                    encoded_len[preds_idx] += len(previous_pred_out[preds_idx])
                else:
                    greedy_predictions_concat = preds_cur
                greedy_predictions.append(greedy_predictions_concat)

                # TODO: make decoding more efficient by avoiding the decoding process from the beginning
                if return_transcription:
                    decoded_out = decoding.ctc_decoder_predictions_tensor(
                        decoder_outputs=greedy_predictions_concat.unsqueeze(0),
                        decoder_lengths=encoded_len[preds_idx : preds_idx + 1],
                        return_hypotheses=False,
                    )
                    all_hyp_or_transcribed_texts.append(decoded_out[0])
            best_hyp = None
        else:
            restart_token_ids = get_restart_token_ids(self)
            if restart_token_ids:
                if bypass_pre_encode:
                    raise NotImplementedError("`restart_after_tokens` is not supported with `bypass_pre_encode`.")
                hyps, (cache_last_channel_next, cache_last_time_next, cache_last_channel_next_len) = (
                    self.decode_chunk_with_restart(
                        encoded=encoded,
                        encoded_len=encoded_len,
                        processed_signal=processed_signal,
                        processed_signal_length=processed_signal_length,
                        hypotheses=previous_hypotheses,
                        caches=(cache_last_channel_next, cache_last_time_next, cache_last_channel_next_len),
                        encode_fresh=functools.partial(self._encode_fresh, keep_all_outputs=keep_all_outputs),
                        restart_token_ids=restart_token_ids,
                        drop_extra_pre_encoded=drop_extra_pre_encoded or 0,
                    )
                )
                best_hyp = self.decoding.postprocess_hypotheses(hyps, return_hypotheses=True)
            else:
                best_hyp = self.decoding.rnnt_decoder_predictions_tensor(
                    encoder_output=encoded,
                    encoded_lengths=encoded_len,
                    return_hypotheses=True,
                    partial_hypotheses=previous_hypotheses,
                )
            greedy_predictions = [hyp.y_sequence for hyp in best_hyp]

            all_hyp_or_transcribed_texts = best_hyp

        result = [
            greedy_predictions,
            all_hyp_or_transcribed_texts,
            cache_last_channel_next,
            cache_last_time_next,
            cache_last_channel_next_len,
            best_hyp,
        ]
        if return_log_probs:
            result.append(log_probs)
            result.append(encoded_len)

        return tuple(result)

    def decode_chunk_with_restart(
        self,
        encoded: Tensor,
        encoded_len: Tensor,
        processed_signal: Tensor,
        processed_signal_length: Tensor,
        hypotheses: Optional[List[Optional[Hypothesis]]],
        caches: tuple,
        encode_fresh,
        restart_token_ids: List[int],
        drop_extra_pre_encoded: int = 0,
    ) -> tuple:
        """
        Greedy-decode one cache-aware streaming chunk, restarting from a fresh state after each token from
        ``decoding.restart_after_tokens`` (e.g. ``<EOU>``). End-of-utterance models are trained on single utterances
        and stop emitting tokens after ``<EOU>`` otherwise.

        For a stream that emits a restart token at chunk frame ``t``, the frames up to ``t`` are re-decoded from the
        previous decoder state, then the decoder state is reset and the rest of the chunk is re-encoded with a fresh
        encoder cache (repeated as needed). A restart token on the last frame resets the state for the next chunk.

        Args:
            encoded: encoder output of the chunk [B, D, T].
            encoded_len: number of valid encoder frames per stream [B].
            processed_signal: chunk features [B, F, T_feat] (including the pre-encode cache).
            processed_signal_length: number of valid features per stream [B].
            hypotheses: raw hypotheses carried from the previous chunk (updated in place), or None.
            caches: (cache_last_channel, cache_last_time, cache_last_channel_len) after encoding the chunk.
            encode_fresh: function ``(features [1, F, T], lengths [1], stream_idx) -> (encoded, encoded_len, caches)``
                encoding the features of one stream from the initial encoder cache.
            restart_token_ids: ids of the restart tokens.
            drop_extra_pre_encoded: number of encoder frames dropped from the beginning of the chunk.

        Returns:
            Tuple of (raw hypotheses, updated caches).
        """
        decoding_computer = getattr(self.decoding.decoding, 'decoding_computer', None)
        if decoding_computer is None:
            raise NotImplementedError("`restart_after_tokens` in streaming requires `greedy_batch` decoding.")
        hypotheses = list(hypotheses) if hypotheses is not None else [None] * encoded.shape[0]
        continuations = self._decode_continuation(encoded, encoded_len, hypotheses)
        caches = list(caches)
        for b in range(encoded.shape[0]):
            enc, num_frames, cont = encoded[b : b + 1], int(encoded_len[b]), continuations[b]
            signal, signal_len = processed_signal[b : b + 1], int(processed_signal_length[b])
            while True:
                prev = hypotheses[b]
                start_frame = int(prev.dec_state.decoded_length) if prev is not None and prev.dec_state else 0
                restart_frame = find_restart_frame(cont, restart_token_ids, start_frame + num_frames)
                if restart_frame is not None:  # re-decode only up to the restart frame
                    restart_frame -= start_frame
                    cont = self._decode_continuation(enc[:, :, : restart_frame + 1], [restart_frame + 1], [prev])[0]
                if prev is None:
                    hypotheses[b] = cont
                else:
                    append_hypothesis_segment_(prev, cont, frame_offset=0)
                if restart_frame is None:
                    if any(int(token) in restart_token_ids for token in cont.y_sequence):  # on the last frame
                        reset_hypothesis_decoder_state_(hypotheses[b], decoding_computer)
                        caches = _replace_cache_row(caches, b)
                    break
                reset_hypothesis_decoder_state_(hypotheses[b], decoding_computer)
                # causal subsampling: (pre-drop) output frame k ends at feature k * subsampling_factor
                tail_start = (restart_frame + 1 + drop_extra_pre_encoded) * self.encoder.subsampling_factor
                signal, signal_len = signal[:, :, tail_start:signal_len], signal_len - tail_start
                drop_extra_pre_encoded = 0  # the rest of the chunk is encoded as the beginning of a new stream
                enc, enc_len, new_cache = encode_fresh(signal, torch.tensor([signal_len], device=enc.device), b)
                caches = _replace_cache_row(caches, b, new_cache)
                num_frames = int(enc_len[0])
                cont = self._decode_continuation(enc, enc_len, [hypotheses[b]])[0]
        return hypotheses, tuple(caches)

    def _decode_continuation(
        self, encoded: Tensor, encoded_len, hypotheses: List[Optional[Hypothesis]]
    ) -> List[Hypothesis]:
        """Raw greedy decoding of new frames, continuing from the decoder states of ``hypotheses`` (not modified)."""
        encoded_len = torch.as_tensor(encoded_len, device=encoded.device)
        states = [
            (
                None
                if hyp is None
                else Hypothesis(
                    score=0.0,
                    y_sequence=torch.empty(0, dtype=torch.long),
                    timestamp=torch.empty(0, dtype=torch.long),
                    dec_state=hyp.dec_state,
                    last_token=hyp.last_token,
                    biasing_cfg=hyp.biasing_cfg,
                )
            )
            for hyp in hypotheses
        ]
        return self.decoding.decoding(encoder_output=encoded, encoded_lengths=encoded_len, partial_hypotheses=states)[
            0
        ]

    def _encode_fresh(self, signal: Tensor, signal_length: Tensor, stream_idx: int, keep_all_outputs: bool):
        """Cache-aware encoding of one stream's features starting from the initial encoder cache."""
        cache = self.encoder.get_initial_cache_state(batch_size=1, dtype=signal.dtype, device=signal.device)
        encoded, encoded_len, *new_cache = self.encoder.cache_aware_stream_step(
            processed_signal=signal,
            processed_signal_length=signal_length,
            cache_last_channel=cache[0],
            cache_last_time=cache[1],
            cache_last_channel_len=cache[2],
            keep_all_outputs=keep_all_outputs,
            drop_extra_pre_encoded=0,
        )
        return self._apply_prompt_to_encoded(encoded), encoded_len, new_cache

    def cache_aware_transcribe_with_restart(
        self, processed_signal: Tensor, processed_signal_length: Tensor, restart_token_ids: List[int]
    ) -> List[Hypothesis]:
        """
        Offline decoding with restarts after ``decoding.restart_after_tokens``, run as cache-aware streaming over
        the features of each utterance, so that the cost stays linear in the audio length.

        Returns:
            Raw (not post-processed) hypotheses.
        """
        if not isinstance(self.encoder, StreamingEncoder):
            raise NotImplementedError("`restart_after_tokens` requires a cache-aware streaming encoder.")
        if self.encoder.streaming_cfg is None:
            self.encoder.setup_streaming_params()
        cfg = self.encoder.streaming_cfg

        def size(value, step):
            return (value[0] if step == 0 else value[1]) if isinstance(value, list) else value

        hypotheses = []
        for b in range(processed_signal.shape[0]):
            signal = processed_signal[b : b + 1, :, : int(processed_signal_length[b])]
            caches = self.encoder.get_initial_cache_state(batch_size=1, dtype=signal.dtype, device=signal.device)
            hyps, position, step = None, 0, 0
            while position < signal.shape[-1]:
                chunk_end = position + size(cfg.chunk_size, step)
                pre_encode_size = size(cfg.pre_encode_cache_size, step)
                chunk = signal[:, :, max(0, position - pre_encode_size) : chunk_end]
                if position < pre_encode_size:  # zero pre-encode cache at the beginning, as in streaming
                    chunk = torch.nn.functional.pad(chunk, (pre_encode_size - position, 0))
                chunk_len = torch.tensor([chunk.shape[-1]], device=chunk.device)
                is_last = chunk_end >= signal.shape[-1]
                encoded, encoded_len, *caches = self.encoder.cache_aware_stream_step(
                    processed_signal=chunk,
                    processed_signal_length=chunk_len,
                    cache_last_channel=caches[0],
                    cache_last_time=caches[1],
                    cache_last_channel_len=caches[2],
                    keep_all_outputs=is_last,
                    drop_extra_pre_encoded=0 if step == 0 else cfg.drop_extra_pre_encoded,
                )
                hyps, caches = self.decode_chunk_with_restart(
                    encoded=self._apply_prompt_to_encoded(encoded),
                    encoded_len=encoded_len,
                    processed_signal=chunk,
                    processed_signal_length=chunk_len,
                    hypotheses=hyps,
                    caches=caches,
                    encode_fresh=functools.partial(self._encode_fresh, keep_all_outputs=is_last),
                    restart_token_ids=restart_token_ids,
                    drop_extra_pre_encoded=0 if step == 0 else cfg.drop_extra_pre_encoded,
                )
                position += size(cfg.shift_size, step)
                step += 1
            hypotheses.extend(hyps)
        return hypotheses

    @torch.no_grad()
    def transcribe_simulate_cache_aware_streaming(
        self,
        paths2audio_files: List[str],
        batch_size: int = 4,
        logprobs: bool = False,
        return_hypotheses: bool = False,
        online_normalization: bool = False,
    ):
        """
        Args:
            paths2audio_files: (a list) of paths to audio files.
            batch_size: (int) batch size to use during inference.
                Bigger will result in better throughput performance but would use more memory.
            logprobs: (bool) pass True to get log probabilities instead of transcripts.
            return_hypotheses: (bool) Either return hypotheses or text
                With hypotheses can do some postprocessing like getting timestamp or rescoring
            online_normalization: (bool) Perform normalization on the run per chunk.
        Returns:
            A list of transcriptions (or raw log probabilities if logprobs is True) in the same order as paths2audio_files
        """
        if paths2audio_files is None or len(paths2audio_files) == 0:
            return {}

        if return_hypotheses and logprobs:
            raise ValueError(
                "Either `return_hypotheses` or `logprobs` can be True at any given time."
                "Returned hypotheses will contain the logprobs."
            )

        if not isinstance(self, asr_models.EncDecCTCModel):
            raise NotImplementedError(f"simulate streaming does not support {type(self)}!")

        if not isinstance(self.encoder, StreamingEncoder):
            raise NotImplementedError("Encoder of this model does not support streaming!")

        data_loader = self._setup_streaming_transcribe_dataloader(paths2audio_files, batch_size, online_normalization)

        total_log_probs = []
        total_texts = []

        for streaming_buffer in data_loader:
            streaming_buffer_iter = iter(streaming_buffer)
            batch_size = len(streaming_buffer.streams_length)
            cache_last_channel, cache_last_time, cache_last_channel_len = self.encoder.get_initial_cache_state(
                batch_size=batch_size
            )
            previous_hypotheses = None
            pred_out_stream = None
            encoded_len = None
            transcribed_texts = None
            batch_log_probs = []

            for step_num, (chunk_audio, chunk_lengths) in enumerate(streaming_buffer_iter):
                drop_extra_pre_encoded = self.encoder.streaming_cfg.drop_extra_pre_encoded if step_num != 0 else 0
                with torch.inference_mode():
                    result = self.conformer_stream_step(
                        processed_signal=chunk_audio,
                        processed_signal_length=chunk_lengths,
                        cache_last_channel=cache_last_channel,
                        cache_last_time=cache_last_time,
                        cache_last_channel_len=cache_last_channel_len,
                        keep_all_outputs=streaming_buffer.is_buffer_empty(),
                        previous_hypotheses=previous_hypotheses,
                        previous_pred_out=pred_out_stream,
                        drop_extra_pre_encoded=drop_extra_pre_encoded,
                        return_transcription=True,
                        return_log_probs=logprobs or return_hypotheses,
                    )
                    if logprobs or return_hypotheses:
                        (
                            pred_out_stream,
                            transcribed_texts,
                            cache_last_channel,
                            cache_last_time,
                            cache_last_channel_len,
                            previous_hypotheses,
                            cur_chunk_log_probs,
                            encoded_len,
                        ) = result
                        batch_log_probs.append(cur_chunk_log_probs.cpu())
                    else:
                        (
                            pred_out_stream,
                            transcribed_texts,
                            cache_last_channel,
                            cache_last_time,
                            cache_last_channel_len,
                            previous_hypotheses,
                        ) = result

            if logprobs or return_hypotheses:
                # concatenate chunk log probs on T dim
                batch_log_probs = torch.cat(batch_log_probs, axis=1)
                for log_probs, log_prob_len in zip(batch_log_probs, encoded_len):
                    total_log_probs.append(log_probs[0:log_prob_len])

            if transcribed_texts is None:
                total_texts += [''] * batch_size
            else:
                total_texts += transcribed_texts

        if logprobs:
            return total_log_probs

        if not return_hypotheses:
            return total_texts

        hyps = []
        for log_probs, text in zip(total_log_probs, total_texts):
            hyps.append(Hypothesis(y_sequence=log_probs, text=text, score=0.0, dec_state=None))
        return hyps

    def _setup_streaming_transcribe_dataloader(
        self, paths2audio_files: List[str], batch_size: int, online_normalization=False
    ):
        """
        Setup function for a temporary data loader which wraps the provided audio file.

        Args:
            paths2audio_files: (a list) of paths to audio files.
            batch_size: (int) batch size to use during inference.
                Bigger will result in better throughput performance but would use more memory.
            online_normalization: whether to do online normalization
        Returns:
            a new batch streaming buffer
        """
        from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer

        streaming_buffer = CacheAwareStreamingAudioBuffer(model=self, online_normalization=online_normalization)
        for sample_idx, sample in enumerate(paths2audio_files):
            processed_signal, processed_signal_length, stream_id = streaming_buffer.append_audio_file(
                sample, stream_id=-1
            )
            logging.info(f'Added this sample to the buffer: {sample}')
            if (sample_idx + 1) % batch_size == 0 or sample_idx == len(paths2audio_files) - 1:
                logging.info(f"Starting to stream samples {sample_idx - len(streaming_buffer) + 1} to {sample_idx}...")
                yield streaming_buffer
                streaming_buffer.reset_buffer()


class VerificationMixin(ABC):
    @staticmethod
    def path2audio_files_to_manifest(paths2audio_files, manifest_filepath):
        """
        Takes paths to audio files and manifest filepath and creates manifest file with the audios
        Args:
            paths2audio_files: paths to audio fragment to be verified
            manifest_filepath: path to manifest file to bre created
        """
        with open(manifest_filepath, 'w', encoding='utf-8') as fp:
            for audio_file in paths2audio_files:
                audio_file = audio_file.strip()
                entry = {'audio_filepath': audio_file, 'offset': 0.0, 'duration': None, 'text': '-', 'label': 'infer'}
                fp.write(json.dumps(entry) + '\n')


class DiarizationMixin(VerificationMixin):
    @abstractmethod
    def diarize(self, paths2audio_files: List[str], batch_size: int = 1) -> List[str]:
        """
        Takes paths to audio files and returns speaker labels
        Args:
            paths2audio_files: paths to audio fragment to be transcribed

        Returns:
            Speaker labels
        """
        pass


class PromptStreamingMixin:
    """Adds language-ID prompt conditioning to a cache-aware streaming ASR model.

    Overrides ``ASRModuleMixin._apply_prompt_to_encoded`` so that
    ``conformer_stream_step`` injects a one-hot language prompt into the
    encoder output. Subclasses must call ``super().initialize_prompt_feature()``
    to populate ``self.concat``, ``self.num_prompts``, and ``self.prompt_kernel``;
    they may then attach their own decoding / WER objects.
    """

    # Plain class-level defaults document the mixin contract. ``prompt_kernel``
    # is intentionally NOT declared here — it's an ``nn.Module`` and a class-level
    # default would shadow ``nn.Module.__getattr__``'s lookup into ``_modules``
    # after the real Sequential is registered in ``initialize_prompt_feature``.
    concat: bool = False
    num_prompts: int = None

    def initialize_prompt_feature(self):
        """Populate the attributes ``_apply_prompt_to_encoded`` depends on.

        Subclasses should call ``super().initialize_prompt_feature()`` first,
        then attach their decoding / WER / joint objects. The mixin sets
        ``self.concat``, ``self.num_prompts``, and ``self.prompt_kernel``.
        """
        self.concat = True
        self.num_prompts = self.cfg.get('num_prompts', 128)

        proj_in_size = self.num_prompts + self._cfg.model_defaults.enc_hidden
        proj_out_size = self._cfg.model_defaults.enc_hidden
        self.prompt_kernel = torch.nn.Sequential(
            torch.nn.Linear(proj_in_size, proj_out_size * 2),
            torch.nn.ReLU(),
            torch.nn.Linear(proj_out_size * 2, proj_out_size),
        )

    def resolve_prompt_language(self, target_lang: Optional[str]) -> int:
        """Map a language name to its prompt index.

        Accepts an exact ``prompt_dictionary`` key (``"de-DE"``, ``"de"``) or a bare language code
        that matches exactly one locale key. None selects the language-agnostic ``"auto"`` prompt.

        Args:
            target_lang: Language name, or None for the language-agnostic prompt.

        Returns:
            (int) Index into the prompt one-hot vector.

        Raises:
            ValueError: If the language is not supported by the model, or is ambiguous.
        """
        return resolve_prompt_index(target_lang, self.cfg.model_defaults.get('prompt_dictionary') or {})

    def set_inference_prompt(self, target_lang: Optional[str]):
        """
        Set the language prompt for streaming inference.

        Call this before ``conformer_stream_step`` to condition decoding on
        a specific language, following the same pattern as
        ``change_decoding_strategy``.

        Args:
            target_lang: Language name, resolved by :meth:`resolve_prompt_language`
                         (e.g. ``"en-US"``, ``"auto"``).
        """
        self._inference_prompt_index = self.resolve_prompt_language(target_lang)
        logging.info(f"Inference prompt set to '{target_lang}' (index {self._inference_prompt_index})")

    def _apply_prompt_to_encoded(self, encoded: Tensor) -> Tensor:
        """
        Inject the language-ID prompt into encoder output during streaming.

        ``encoded`` arrives as (B, D, T) from the encoder cache-aware step.
        Returns the same shape after prompt concatenation + projection.
        """
        if not self.concat or not hasattr(self, '_inference_prompt_index'):
            return encoded

        encoded = encoded.transpose(1, 2)  # (B, D, T) -> (B, T, D)

        batch_size, time_steps, _ = encoded.shape
        prompt = torch.zeros(
            batch_size,
            time_steps,
            self.num_prompts,
            dtype=encoded.dtype,
            device=encoded.device,
        )
        idx = torch.full(
            (batch_size,),
            self._inference_prompt_index,
            dtype=torch.long,
            device=encoded.device,
        )
        prompt.scatter_(2, idx.view(batch_size, 1, 1).expand(-1, time_steps, -1), 1.0)

        out_dtype = encoded.dtype
        encoded = self.prompt_kernel(torch.cat([encoded, prompt], dim=-1)).to(out_dtype)
        return encoded.transpose(1, 2)  # (B, T, D) -> (B, D, T)


def _replace_cache_row(caches: list, idx: int, row: Optional[list] = None) -> list:
    """Replace the encoder cache of stream ``idx`` with ``row`` (1-stream caches), or with the initial (zero) cache."""
    if row is None:
        row = [torch.zeros_like(caches[0][:, :1]), torch.zeros_like(caches[1][:, :1]), torch.zeros_like(caches[2][:1])]
    return [
        torch.cat((caches[0][:, :idx], row[0].to(caches[0].dtype), caches[0][:, idx + 1 :]), dim=1),
        torch.cat((caches[1][:, :idx], row[1].to(caches[1].dtype), caches[1][:, idx + 1 :]), dim=1),
        torch.cat((caches[2][:idx], row[2].to(caches[2].dtype), caches[2][idx + 1 :]), dim=0),
    ]
