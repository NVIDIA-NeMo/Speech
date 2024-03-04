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

from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

from omegaconf.omegaconf import MISSING

from nemo.collections.nlp.data.machine_translation.machine_translation_dataset import TranslationDataConfig
from nemo.collections.nlp.models.enc_dec_nlp_model import EncDecNLPModelConfig
from nemo.collections.nlp.modules.common.token_classifier import TokenClassifierConfig
from nemo.collections.nlp.modules.common.tokenizer_utils import TokenizerConfig
from nemo.collections.nlp.modules.common.transformer.transformer import (
    NeMoTransformerConfig,
    NeMoTransformerEncoderConfig,
)
from nemo.collections.nlp.modules.common.transformer.transformer_bottleneck import (
    NeMoTransformerBottleneckDecoderConfig,
    NeMoTransformerBottleneckEncoderConfig,
)
from nemo.core.config.modelPT import OptimConfig, SchedConfig


@dataclass
class MTSchedConfig(SchedConfig):
    name: str = 'InverseSquareRootAnnealing'
    warmup_ratio: Optional[float] = None
    last_epoch: int = -1


# TODO: Refactor this dataclass to to support more optimizers (it pins the optimizer to Adam-like optimizers).
@dataclass
class MTOptimConfig(OptimConfig):
    name: str = 'adam'
    lr: float = 1e-3
    betas: Tuple[float, float] = (0.9, 0.98)
    weight_decay: float = 0.0
    sched: Optional[MTSchedConfig] = field(default_factory=lambda: MTSchedConfig())


@dataclass
class MTEncDecModelConfig(EncDecNLPModelConfig):
    # machine translation configurations
    num_val_examples: int = 3
    num_test_examples: int = 3
    max_generation_delta: int = 10
    label_smoothing: Optional[float] = 0.0
    beam_size: int = 4
    len_pen: float = 0.0
    src_language: Any = 'en'  # Any = str or List[str]
    tgt_language: Any = 'en'  # Any = str or List[str]
    find_unused_parameters: Optional[bool] = True
    shared_tokenizer: Optional[bool] = True
    multilingual: Optional[bool] = False
    preproc_out_dir: Optional[str] = None
    validate_input_ids: Optional[bool] = True
    shared_embeddings: bool = False

    # network architecture configuration
    encoder_tokenizer: Any = MISSING
    encoder: Any = MISSING

    decoder_tokenizer: Any = MISSING
    decoder: Any = MISSING

    head: TokenClassifierConfig = field(default_factory=lambda: TokenClassifierConfig(log_softmax=True))

    # dataset configurations
    train_ds: Optional[TranslationDataConfig] = field(
        default_factory=lambda: TranslationDataConfig(
            src_file_name=MISSING,
            tgt_file_name=MISSING,
            tokens_in_batch=512,
            clean=True,
            shuffle=True,
            cache_ids=False,
            use_cache=False,
        )
    )
    validation_ds: Optional[TranslationDataConfig] = field(
        default_factory=lambda: TranslationDataConfig(
            src_file_name=MISSING,
            tgt_file_name=MISSING,
            tokens_in_batch=512,
            clean=False,
            shuffle=False,
            cache_ids=False,
            use_cache=False,
        )
    )
    test_ds: Optional[TranslationDataConfig] = field(
        default_factory=lambda: TranslationDataConfig(
            src_file_name=MISSING,
            tgt_file_name=MISSING,
            tokens_in_batch=512,
            clean=False,
            shuffle=False,
            cache_ids=False,
            use_cache=False,
        )
    )
    optim: Optional[OptimConfig] = field(default_factory=lambda: MTOptimConfig())


@dataclass
class AAYNBaseConfig(MTEncDecModelConfig):

    # Attention is All You Need Base Configuration
    encoder_tokenizer: TokenizerConfig = field(default_factory=lambda: TokenizerConfig(library='sentencepiece'))
    decoder_tokenizer: TokenizerConfig = field(default_factory=lambda: TokenizerConfig(library='sentencepiece'))

    encoder: NeMoTransformerEncoderConfig = field(
        default_factory=lambda: NeMoTransformerEncoderConfig(
            library='nemo',
            model_name=None,
            pretrained=False,
            hidden_size=512,
            inner_size=2048,
            num_layers=6,
            num_attention_heads=8,
            ffn_dropout=0.1,
            attn_score_dropout=0.1,
            attn_layer_dropout=0.1,
        )
    )

    decoder: NeMoTransformerConfig = field(
        default_factory=lambda: NeMoTransformerConfig(
            library='nemo',
            model_name=None,
            pretrained=False,
            hidden_size=512,
            inner_size=2048,
            num_layers=6,
            num_attention_heads=8,
            ffn_dropout=0.1,
            attn_score_dropout=0.1,
            attn_layer_dropout=0.1,
        )
    )


@dataclass
class MTBottleneckModelConfig(AAYNBaseConfig):
    model_type: str = 'nll'
    min_logv: float = -6
    latent_size: int = -1  # -1 will take value of encoder hidden
    non_recon_warmup_batches: int = 200000
    recon_per_token: bool = True
    log_timing: bool = True

    encoder: NeMoTransformerBottleneckEncoderConfig = field(
        default_factory=lambda: NeMoTransformerBottleneckEncoderConfig(
            library='nemo',
            model_name=None,
            pretrained=False,
            hidden_size=512,
            inner_size=2048,
            num_layers=6,
            num_attention_heads=8,
            ffn_dropout=0.1,
            attn_score_dropout=0.1,
            attn_layer_dropout=0.1,
            arch='seq2seq',
            hidden_steps=32,
            hidden_blocks=1,
            hidden_init_method='params',
        )
    )

    decoder: NeMoTransformerBottleneckDecoderConfig = field(
        default_factory=lambda: NeMoTransformerBottleneckDecoderConfig(
            library='nemo',
            model_name=None,
            pretrained=False,
            inner_size=2048,
            num_layers=6,
            num_attention_heads=8,
            ffn_dropout=0.1,
            attn_score_dropout=0.1,
            attn_layer_dropout=0.1,
            arch='seq2seq',
        )
    )
