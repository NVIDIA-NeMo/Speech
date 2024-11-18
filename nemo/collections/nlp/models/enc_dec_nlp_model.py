# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

from dataclasses import dataclass
from typing import Any

from lightning.pytorch.trainer.trainer import Trainer
from omegaconf.omegaconf import MISSING

from nemo.collections.nlp.models.nlp_model import NLPModel
from nemo.collections.nlp.modules.common.decoder_module import DecoderModule
from nemo.collections.nlp.modules.common.encoder_module import EncoderModule
from nemo.collections.nlp.modules.common.tokenizer_utils import TokenizerConfig
from nemo.core.config.modelPT import ModelConfig


@dataclass
class EncDecNLPModelConfig(ModelConfig):
    encoder_tokenizer: TokenizerConfig = MISSING
    decoder_tokenizer: TokenizerConfig = MISSING
    encoder: Any = MISSING
    decoder: Any = MISSING
    head: Any = MISSING


class EncDecNLPModel(NLPModel):
    """Base class for encoder-decoder NLP models."""

    def __init__(self, cfg: EncDecNLPModelConfig, trainer: Trainer = None):
        super().__init__(cfg=cfg, trainer=trainer)

    @property
    def encoder_vocab_size(self):
        return self.encoder_tokenizer.vocab_size

    @property
    def decoder_vocab_size(self):
        return self.decoder_tokenizer.vocab_size

    @property
    def encoder_tokenizer(self):
        return self._encoder_tokenizer

    @encoder_tokenizer.setter
    def encoder_tokenizer(self, tokenizer):
        self._encoder_tokenizer = tokenizer

    @property
    def decoder_tokenizer(self):
        return self._decoder_tokenizer

    @decoder_tokenizer.setter
    def decoder_tokenizer(self, tokenizer):
        self._decoder_tokenizer = tokenizer

    @property
    def encoder(self) -> EncoderModule:
        return self._encoder

    @encoder.setter
    def encoder(self, encoder):
        self._encoder = encoder

    @property
    def decoder(self) -> DecoderModule:
        return self._decoder

    @decoder.setter
    def decoder(self, decoder):
        self._decoder = decoder
