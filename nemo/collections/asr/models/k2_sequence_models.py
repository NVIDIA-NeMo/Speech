# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

from typing import List, Optional

from omegaconf import DictConfig
from pytorch_lightning import Trainer

from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE
from nemo.collections.asr.models.ctc_models import EncDecCTCModel
from nemo.collections.asr.models.rnnt_bpe_models import EncDecRNNTBPEModel
from nemo.collections.asr.models.rnnt_models import EncDecRNNTModel
from nemo.collections.asr.parts.k2.classes import ASRK2Mixin
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.utils import logging


class EncDecK2SeqModel(EncDecCTCModel, ASRK2Mixin):
    """Encoder decoder models with various lattice losses."""

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        loss_type = cfg.graph_module_cfg.get("loss_type", "ctc")
        if loss_type != "ctc" and loss_type != "mmi":
            raise ValueError(f"Class {self.__class__.__name__} does not support `loss_type`={loss_type}")
        super().__init__(cfg=cfg, trainer=trainer)
        self._init_k2()

    @classmethod
    def list_available_models(cls) -> Optional[List[PretrainedModelInfo]]:
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.

        Returns:
            List of available pre-trained models.
        """
        pass

    def change_vocabulary(self, new_vocabulary: List[str]):
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
        super().change_vocabulary(new_vocabulary)

        if self.use_graph_lm:
            self.token_lm = None
            logging.warning(
                f"""With .change_vocabulary() call for a model with criterion_type=`{self.loss.criterion_type}`, 
                a new token_lm has to be set manually: call .update_k2_modules(new_cfg) 
                or update .graph_module_cfg.backend_cfg.token_lm before calling this method."""
            )

        self.update_k2_modules(self.graph_module_cfg)

    @typecheck()
    def forward(
        self, input_signal=None, input_signal_length=None, processed_signal=None, processed_signal_length=None,
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
                of shape (B, D, T) that has undergone processing via some DALI preprocessor.
            processed_signal_length: Vector of length B, that contains the individual lengths of the
                processed audio sequences.

        Returns:
            A tuple of 3 elements -
            1) The log probabilities tensor of shape [B, T, D].
            2) The lengths of the acoustic sequence after propagation through the encoder, of shape [B].
            3) The greedy token predictions of the model of shape [B, T] (via argmax)
        """
        log_probs, encoded_len, greedy_predictions = super().forward(
            input_signal=input_signal,
            input_signal_length=input_signal_length,
            processed_signal=processed_signal,
            processed_signal_length=processed_signal_length,
        )
        return self._forward_k2_post_processing(
            log_probs=log_probs, encoded_length=encoded_len, greedy_predictions=greedy_predictions
        )


class EncDecK2SeqModelBPE(EncDecCTCModelBPE, ASRK2Mixin):
    """Encoder decoder models with Byte Pair Encoding and various lattice losses."""

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        loss_type = cfg.graph_module_cfg.get("loss_type", "ctc")
        if loss_type != "ctc" and loss_type != "mmi":
            raise ValueError(f"Class {self.__class__.__name__} does not support `loss_type`={loss_type}")
        super().__init__(cfg=cfg, trainer=trainer)
        self._init_k2()

    @classmethod
    def list_available_models(cls) -> Optional[List[PretrainedModelInfo]]:
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.

        Returns:
            List of available pre-trained models.
        """
        pass

    def change_vocabulary(self, new_tokenizer_dir: str, new_tokenizer_type: str):
        """
        Changes vocabulary of the tokenizer used during CTC decoding process.
        Use this method when fine-tuning on from pre-trained model.
        This method changes only decoder and leaves encoder and pre-processing modules unchanged. For example, you would
        use it if you want to use pretrained encoder when fine-tuning on a data in another language, or when you'd need
        model to learn capitalization, punctuation and/or special characters.

        Args:
            new_tokenizer_dir: Path to the new tokenizer directory.
            new_tokenizer_type: Either `bpe` or `wpe`. `bpe` is used for SentencePiece tokenizers,
                whereas `wpe` is used for `BertTokenizer`.

        Returns: None

        """
        super().change_vocabulary(new_tokenizer_dir, new_tokenizer_type)

        if self.use_graph_lm:
            self.token_lm = None
            logging.warning(
                f"""With .change_vocabulary() call for a model with criterion_type=`{self.loss.criterion_type}`, 
                a new token_lm has to be set manually: call .update_k2_modules(new_cfg) 
                or update .graph_module_cfg.backend_cfg.token_lm before calling this method."""
            )

        self.update_k2_modules(self.graph_module_cfg)

    @typecheck()
    def forward(
        self, input_signal=None, input_signal_length=None, processed_signal=None, processed_signal_length=None,
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
                of shape (B, D, T) that has undergone processing via some DALI preprocessor.
            processed_signal_length: Vector of length B, that contains the individual lengths of the
                processed audio sequences.

        Returns:
            A tuple of 3 elements -
            1) The log probabilities tensor of shape [B, T, D].
            2) The lengths of the acoustic sequence after propagation through the encoder, of shape [B].
            3) The greedy token predictions of the model of shape [B, T] (via argmax)
        """
        log_probs, encoded_len, greedy_predictions = super().forward(
            input_signal=input_signal,
            input_signal_length=input_signal_length,
            processed_signal=processed_signal,
            processed_signal_length=processed_signal_length,
        )
        return self._forward_k2_post_processing(
            log_probs=log_probs, encoded_length=encoded_len, greedy_predictions=greedy_predictions
        )


class EncDecK2RnntSeqModel(EncDecRNNTModel, ASRK2Mixin):
    """Encoder decoder models with various lattice losses."""

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        loss_type = cfg.graph_module_cfg.get("loss_type", "rnnt")
        criterion_type = cfg.graph_module_cfg.get("criterion_type", "ml")
        if loss_type != "rnnt" or criterion_type != "ml":
            raise ValueError(
                f"""Class {self.__class__.__name__} does not support 
            `criterion_type`={criterion_type} with `loss_type`={loss_type}"""
            )
        super().__init__(cfg=cfg, trainer=trainer)
        self._init_k2()

    @classmethod
    def list_available_models(cls) -> Optional[PretrainedModelInfo]:
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.

        Returns:
            List of available pre-trained models.
        """
        pass

    def change_vocabulary(self, new_vocabulary: List[str]):
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
        super().change_vocabulary(new_vocabulary)

        if self.use_graph_lm:
            self.token_lm = None
            logging.warning(
                f"""With .change_vocabulary() call for a model with criterion_type=`{self.loss.criterion_type}`, 
                a new token_lm has to be set manually: call .update_k2_modules(new_cfg) 
                or update .graph_module_cfg.backend_cfg.token_lm before calling this method."""
            )

        self.update_k2_modules(self.graph_module_cfg)


class EncDecK2RnntSeqModelBPE(EncDecRNNTBPEModel, ASRK2Mixin):
    """Encoder decoder models with Byte Pair Encoding and various lattice losses."""

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        loss_type = cfg.graph_module_cfg.get("loss_type", "rnnt")
        criterion_type = cfg.graph_module_cfg.get("criterion_type", "ml")
        if loss_type != "rnnt" or criterion_type != "ml":
            raise ValueError(
                f"""Class {self.__class__.__name__} does not support 
            `criterion_type`={criterion_type} with `loss_type`={loss_type}"""
            )
        super().__init__(cfg=cfg, trainer=trainer)
        self._init_k2()

    @classmethod
    def list_available_models(cls) -> Optional[PretrainedModelInfo]:
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.

        Returns:
            List of available pre-trained models.
        """
        pass

    def change_vocabulary(self, new_tokenizer_dir: str, new_tokenizer_type: str):
        """
        Changes vocabulary of the tokenizer used during CTC decoding process.
        Use this method when fine-tuning on from pre-trained model.
        This method changes only decoder and leaves encoder and pre-processing modules unchanged. For example, you would
        use it if you want to use pretrained encoder when fine-tuning on a data in another language, or when you'd need
        model to learn capitalization, punctuation and/or special characters.

        Args:
            new_tokenizer_dir: Path to the new tokenizer directory.
            new_tokenizer_type: Either `bpe` or `wpe`. `bpe` is used for SentencePiece tokenizers,
                whereas `wpe` is used for `BertTokenizer`.

        Returns: None

        """
        super().change_vocabulary(new_tokenizer_dir, new_tokenizer_type)

        if self.use_graph_lm:
            self.token_lm = None
            logging.warning(
                f"""With .change_vocabulary() call for a model with criterion_type=`{self.loss.criterion_type}`, 
                a new token_lm has to be set manually: call .update_k2_modules(new_cfg) 
                or update .graph_module_cfg.backend_cfg.token_lm before calling this method."""
            )

        self.update_k2_modules(self.graph_module_cfg)
