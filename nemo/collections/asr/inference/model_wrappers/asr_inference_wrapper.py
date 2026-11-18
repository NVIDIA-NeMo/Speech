# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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


import copy
from functools import cached_property
from typing import Callable

import torch
from omegaconf import DictConfig, open_dict

from nemo.collections.asr.inference.utils.constants import SENTENCEPIECE_UNDERSCORE
from nemo.collections.asr.inference.utils.device_utils import setup_device
from nemo.collections.asr.inference.utils.pipeline_utils import make_preprocessor_deterministic, set_compile_cache_dir
from nemo.collections.asr.models import ASRModel, EncDecHybridRNNTCTCModel
from nemo.collections.asr.parts.submodules.ctc_decoding import CTCDecodingConfig
from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTDecodingConfig
from nemo.collections.asr.parts.utils.asr_confidence_utils import get_confidence_aggregation_bank
from nemo.collections.asr.parts.utils.eou_utils import EOB_STRING, EOU_STRING

SUPPORTED_CONFIDENCE_AGGREGATORS = get_confidence_aggregation_bank()


class ASRInferenceWrapper:
    """
    Base class for ASR inference wrappers.
    It provides a common interface for ASR inference wrappers.
    Derived classes MUST implement the following methods:
        - __post_init__: Additional post initialization steps that must be implemented in the derived classes.
        - get_blank_id: Returns the blank id for the model.
        - get_vocabulary: Returns the vocabulary for the model.
        - get_subsampling_factor: Returns the subsampling factor for the model.
    """

    # decoder used by the derived classes: 'ctc' or 'rnnt' (selects the decoding config of hybrid models)
    decoder_type: str | None = None

    def __init__(
        self,
        model_name: str,
        decoding_cfg: CTCDecodingConfig | RNNTDecodingConfig,
        device: str = 'cuda',
        device_id: int = 0,
        compute_dtype: str = 'bfloat16',
        use_amp: bool = True,
        compile_encoder: bool = False,
        compile_cache_dir: str | None = None,
    ):
        """
        Initialize the ASR inference wrapper.
        Args:
            model_name: (str) path to the model checkpoint or a model name from the NGC cloud.
            decoding_cfg: (CTCDecodingConfig | RNNTDecodingConfig) decoding configuration.
            device: (str) device to run the model on.
            device_id: (int) device ID to run the model on.
            compute_dtype: (str) compute dtype to run the model on.
            use_amp: (bool) Use Automatic Mixed Precision
            compile_encoder: (bool) Compile the encoder with torch.compile. The compilation happens on the first
                calls, so it should fall into warmup. Fusion changes the order of low-precision arithmetic, so the
                decoded text can differ from eager in a few transcripts.
            compile_cache_dir: (str | None) Directory where the compiled artifacts are stored and reused by later
                runs, so the encoder is not compiled twice. Requires compile_encoder.
        """

        self.decoding_cfg = decoding_cfg
        self.device_str, self.device_id, self.compute_dtype = setup_device(device.strip(), device_id, compute_dtype)
        self.device = torch.device(self.device_str)
        self.use_amp = use_amp
        if compile_cache_dir is not None and not compile_encoder:
            raise ValueError("compile_cache_dir is set but compile_encoder is false, so nothing would be cached")
        self.compile_encoder = compile_encoder
        self.compile_cache_dir = compile_cache_dir
        self.asr_model = self.load_model(model_name, self.device)
        self.asr_model_cfg = self.asr_model._cfg
        self.set_dither_to_zero()
        self.tokenizer = self.asr_model.tokenizer
        self._check_no_restart_after_tokens()

        # post initialization steps that must be implemented in the derived classes
        self.__post_init__()

        # after `__post_init__`, which checks that the model type matches the wrapper
        self._check_not_end_of_utterance_model()

        # after the post initialization, which may still cast and wrap the model
        if self.compile_encoder:
            if self.compile_cache_dir is not None:
                set_compile_cache_dir(self.compile_cache_dir)
            self.compile_encoder_module()

    def compile_encoder_module(self) -> None:
        """
        Compile the encoder with torch.compile. Derived classes whose encoder is called differently override this.
        """
        self.asr_model.encoder = torch.compile(self.asr_model.encoder)

    @staticmethod
    def load_model(model_name: str, map_location: torch.device) -> ASRModel:
        """
        Load the ASR model.
        Args:
            model_name: (str) path to the model checkpoint or a model name from the NGC cloud.
            map_location: (torch.device) device to load the model on.
        Returns:
            (ASRModel) loaded ASR model.
        """
        try:
            if model_name.endswith('.nemo'):
                asr_model = ASRModel.restore_from(model_name, map_location=map_location)
            else:
                asr_model = ASRModel.from_pretrained(model_name, map_location=map_location)
            asr_model.eval()
            return asr_model
        except Exception as e:
            raise RuntimeError(f"Failed to load model {model_name}: {str(e)}") from e

    @property
    def word_separator(self) -> str:
        """
        Returns word separator.
        Returns:
            (str) word separator.
        """
        return self.decoding_cfg.word_seperator

    @property
    def confidence_aggregator(self) -> Callable:
        """
        Returns confidence aggregator function.
        Returns:
            (Callable) confidence aggregator function.
        """
        return SUPPORTED_CONFIDENCE_AGGREGATORS[self.decoding_cfg.confidence_cfg.aggregation]

    def copy_asr_config(self) -> DictConfig:
        """
        Copies the ASR model config.
        Returns:
            (DictConfig) copy of the ASR model configuration.
        """
        return copy.deepcopy(self.asr_model_cfg)

    def create_preprocessor(self) -> tuple[Callable, DictConfig]:
        """
        Creates a deterministic preprocessor from the ASR model configuration.
        Disables normalization, dither and padding.
        Returns:
            (Callable, DictConfig) deterministic preprocessor and its configuration.
        """
        new_asr_config = self.copy_asr_config()
        new_asr_config = make_preprocessor_deterministic(new_asr_config)
        preprocessor_config = copy.deepcopy(new_asr_config.preprocessor)
        preprocessor = ASRModel.from_config_dict(preprocessor_config)
        preprocessor.to(self.device)
        return preprocessor, preprocessor_config

    def supports_capitalization(self) -> bool:
        """
        Checks if the ASR model supports capitalization.
        Returns:
            (bool) True if the ASR model supports capitalization, False otherwise.
        """
        return self.tokenizer.supports_capitalization

    def supports_punctuation(self) -> bool:
        """
        Checks if the ASR model supports punctuation.
        Returns:
            (bool) True if the ASR model supports punctuation, False otherwise.
        """
        return self.supported_punctuation() != set()

    def supported_punctuation(self) -> set:
        """
        Returns supported punctuation symbol set without single quote.
        Returns:
            (set) Set of supported punctuation symbols.
        """
        return self.tokenizer.supported_punctuation - set("'")

    @cached_property
    def punctuation_ids(self) -> set:
        """
        Returns ids of supported punctuation symbols.
        Returns:
            (set) Set of punctuation ids.
        """
        punctuation_ids = set()
        if self.supports_punctuation():
            for punctuation in self.supported_punctuation():
                punctuation_ids.add(self.tokenizer.tokens_to_ids(punctuation)[0])
        return punctuation_ids

    @cached_property
    def underscore_id(self) -> int:
        """
        Returns id of the underscore token.
        Returns:
            (int) underscore id for the model.
        """
        if getattr(self.asr_model.tokenizer, "spm_separator_id", None) is not None:
            return self.asr_model.tokenizer.spm_separator_id
        else:
            return self.asr_model.tokenizer.tokens_to_ids(SENTENCEPIECE_UNDERSCORE)

    @cached_property
    def language_token_ids(self) -> set:
        """
        This property is used for some Riva models that have language tokens included in the vocabulary.
        Returns:
            (set) Set of language token ids.
        """
        vocab = self.get_vocabulary()
        language_token_ids = set()
        for token in vocab:
            if token.startswith("<") and token.endswith(">") and token != "<unk>":
                language_token_ids.add(self.asr_model.tokenizer.tokens_to_ids(token)[0])
        return language_token_ids

    def reset_decoding_strategy(self, decoder_type: str) -> None:
        """
        Reset the decoding strategy for the model.
        Args:
            decoder_type: (str) decoding type either 'ctc', 'rnnt'.
        """
        if isinstance(self.asr_model, EncDecHybridRNNTCTCModel):
            self.asr_model.change_decoding_strategy(decoding_cfg=None, decoder_type=decoder_type)
        else:
            self.asr_model.change_decoding_strategy(None)

    def set_decoding_strategy(self, decoder_type: str) -> None:
        """
        Set the decoding strategy for the model.
        Args:
            decoder_type: (str) decoding type either 'ctc', 'rnnt'.
        """
        if isinstance(self.asr_model, EncDecHybridRNNTCTCModel):
            self.asr_model.change_decoding_strategy(decoding_cfg=self.decoding_cfg, decoder_type=decoder_type)
        else:
            self.asr_model.change_decoding_strategy(self.decoding_cfg)

    def set_dither_to_zero(self) -> None:
        """
        To remove randomness from preprocessor set the dither value to zero.
        """
        self.asr_model.preprocessor.featurizer.dither = 0.0
        with open_dict(self.asr_model_cfg):
            self.asr_model_cfg.preprocessor.dither = 0.0

    def get_window_stride(self) -> float:
        """
        Get the window stride for the model.
        Returns:
            (float) window stride for the model.
        """
        return self.asr_model_cfg.preprocessor.window_stride

    def get_model_stride(self, in_secs: bool = False, in_milliseconds: bool = False) -> float:
        """
        Get the model stride in seconds for the model.
        Args:
            in_secs: (bool) Whether to return the model stride in seconds.
            in_milliseconds: (bool) Whether to return the model stride in milliseconds.
        Returns:
            (float) model stride in seconds or milliseconds.
        """
        if in_secs and in_milliseconds:
            raise ValueError("Cannot return both seconds and milliseconds at the same time.")
        if in_secs:
            return self.get_window_stride() * self.get_subsampling_factor()
        if in_milliseconds:
            return self.get_window_stride() * self.get_subsampling_factor() * 1000

        return self.get_window_stride() * self.get_subsampling_factor()

    # Methods that must be implemented in the derived classes.
    def __post_init__(self):
        """
        Additional post initialization steps that must be implemented in the derived classes.
        """
        raise NotImplementedError()

    def get_blank_id(self) -> int:
        """
        Returns id of the blank token.
        Returns:
            (int) blank id for the model.
        """
        raise NotImplementedError()

    def get_vocabulary(self) -> list[str]:
        """
        Returns the list of vocabulary tokens.
        Returns:
            (list[str]) list of vocabulary tokens.
        """
        raise NotImplementedError()

    def get_subsampling_factor(self) -> int:
        """
        Returns the subsampling factor for the model.
        Returns:
            (int) subsampling factor for the model.
        """
        raise NotImplementedError()

    def _check_no_restart_after_tokens(self) -> None:
        """
        Raise if the pipeline decoding config or the model decoding config of an RNNT decoder sets
        ``decoding.restart_after_tokens``, which the streaming pipelines do not support. For hybrid models used with
        CTC decoding, the model ``decoding`` config belongs to the inactive RNNT decoder and is not checked.
        """
        model_decoding_cfg = self.asr_model_cfg.get('decoding', None) if self.decoder_type == 'rnnt' else None
        for cfg in (self.decoding_cfg, model_decoding_cfg):
            if getattr(cfg, 'restart_after_tokens', None):  # DictConfig or decoding config dataclass
                raise NotImplementedError(
                    "`decoding.restart_after_tokens` is not supported by the streaming inference pipelines. "
                    + _RESTART_AFTER_TOKENS_ALTERNATIVES
                )

    def _check_not_end_of_utterance_model(self) -> None:
        """
        Raise for end-of-utterance models (with ``<EOU>``/``<EOB>`` tokens). They stop emitting tokens after the first
        ``<EOU>`` unless decoding restarts after it, so the pipelines would silently drop the rest of the audio.
        """
        eou_tokens = [token for token in (EOU_STRING, EOB_STRING) if token in self.get_vocabulary()]
        if eou_tokens:
            raise NotImplementedError(
                f"End-of-utterance models (with {', '.join(eou_tokens)} in the vocabulary) are not supported by the "
                "streaming inference pipelines: they stop emitting tokens after the first end of utterance, so the "
                "rest of the audio would not be transcribed. " + _RESTART_AFTER_TOKENS_ALTERNATIVES
            )


_RESTART_AFTER_TOKENS_ALTERNATIVES = (
    "Use `transcribe()` with `decoding.restart_after_tokens` or `speech_to_text_cache_aware_streaming_infer.py` "
    "with `rnnt_decoding.restart_after_tokens` instead (see `examples/asr/asr_eou/README.md`)."
)
