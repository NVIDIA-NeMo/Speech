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

import json
import os
import shutil
import tarfile
from abc import ABC, abstractmethod
from typing import List

import torch
from omegaconf import DictConfig, OmegaConf, open_dict

import nemo.collections.asr.models as asr_models
from nemo.collections.asr.parts.mixins.asr_adapter_mixins import ASRAdapterModelMixin
from nemo.collections.asr.parts.mixins.streaming import StreamingEncoder
from nemo.collections.asr.parts.utils import asr_module_utils
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from nemo.collections.common import tokenizers
from nemo.utils import app_state, logging


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

    def _setup_tokenizer(self, tokenizer_cfg: DictConfig):
        tokenizer_type = tokenizer_cfg.get('type')
        if tokenizer_type is None:
            raise ValueError("`tokenizer.type` cannot be None")
        elif tokenizer_type.lower() == 'agg':
            self._setup_aggregate_tokenizer(tokenizer_cfg)
        else:
            self._setup_monolingual_tokenizer(tokenizer_cfg)

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

            # Update special tokens
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
                loc = shutil.copy2(v, dir)
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
            tar = tarfile.open(restore_path, tar_header)

            for nemo_object_name in nemo_file_objects:
                members = [x for x in tar.getmembers() if nemo_object_name in x.name]
                for member in members:
                    tar.extract(member, dir)

                    new_name = member.name.split("_")[1:]
                    if len(new_name) > 1:
                        new_name = "_".join(new_name)
                    else:
                        new_name = new_name[0]
                    os.rename(os.path.join(dir, member.name), os.path.join(dir, new_name))

                    logging.info(f"Saved {nemo_object_name} at {os.path.join(dir, new_name)}")


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
        self, self_attention_model: str = None, att_context_size: List[int] = None, update_config: bool = True
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

                If None is provided, the self_attention_model isn't changed. Defauts to None.
            att_context_size (List[int]): List of 2 ints corresponding to left and right attention context sizes,
                or None to keep as it is. Defauts to None.
            update_config (bool): Whether to update the config or not with the new attention model.
                Defaults to True.
        """
        if self_attention_model is None and att_context_size is None:
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

        self.encoder.change_attention_model(self_attention_model, att_context_size, update_config, self.device)
        if update_config:
            with open_dict(self.cfg):
                self.cfg.encoder.self_attention_model = self_attention_model
                self.cfg.encoder.att_context_size = att_context_size

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

    def conformer_stream_step(
        self,
        processed_signal: torch.Tensor,
        processed_signal_length: torch.Tensor = None,
        cache_last_channel: torch.Tensor = None,
        cache_last_time: torch.Tensor = None,
        cache_last_channel_len: torch.Tensor = None,
        keep_all_outputs: bool = True,
        previous_hypotheses: List[Hypothesis] = None,
        previous_pred_out: torch.Tensor = None,
        drop_extra_pre_encoded: int = None,
        return_transcription: bool = True,
        return_log_probs: bool = False,
    ):
        """
        It simulates a forward step with caching for streaming purposes.
        It supports the ASR models where their encoder supports streaming like Conformer.
        Args:
            processed_signal: the input audio signals
            processed_signal_length: the length of the audios
            cache_last_channel: the cache tensor for last channel layers like MHA
            cache_last_channel_len: engths for cache_last_channel
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
            raise NotImplementedError(f"Encoder of this model does not support streaming!")

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
        )

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
                    all_hyp_or_transcribed_texts.append(decoded_out[0][0])
            best_hyp = None
        else:
            best_hyp, all_hyp_or_transcribed_texts = self.decoding.rnnt_decoder_predictions_tensor(
                encoder_output=encoded,
                encoded_lengths=encoded_len,
                return_hypotheses=True,
                partial_hypotheses=previous_hypotheses,
            )
            greedy_predictions = [hyp.y_sequence for hyp in best_hyp]

            if all_hyp_or_transcribed_texts is None:
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
            raise NotImplementedError(f"Encoder of this model does not support streaming!")

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
