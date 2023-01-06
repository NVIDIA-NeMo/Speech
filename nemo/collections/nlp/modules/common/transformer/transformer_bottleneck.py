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

from dataclasses import dataclass
from typing import Dict, Optional

from nemo.collections.nlp.modules.common.transformer.bridge_encoders import BridgeEncoder
from nemo.collections.nlp.modules.common.transformer.perceiver_encoders import PerceiverEncoder
from nemo.collections.nlp.modules.common.transformer.reduction_encoders import PoolingEncoder
from nemo.collections.nlp.modules.common.transformer.transformer import (
    NeMoTransformerConfig,
    TransformerDecoderNM,
    TransformerEncoderNM,
)
from nemo.core.classes.common import typecheck
from nemo.core.neural_types import MaskType, NeuralType
from nemo.core.neural_types.elements import BoolType

__all__ = [
    "NeMoTransformerBottleneckConfig",
    "NeMoTransformerBottleneckEncoderConfig",
    "NeMoTransformerBottleneckDecoderConfig",
    "TransformerBottleneckEncoderNM",
]


@dataclass
class NeMoTransformerBottleneckConfig(NeMoTransformerConfig):
    # architecture details (default is no bottleneck)
    arch: str = ''
    hidden_steps: int = -1
    hidden_blocks: int = 1
    hidden_init_method: str = "params"


@dataclass
class NeMoTransformerBottleneckEncoderConfig(NeMoTransformerBottleneckConfig):
    mask_future: bool = False
    # change return_mask to False to return hidden states only (default for non-bottleneck encoder)
    return_mask: bool = True


@dataclass
class NeMoTransformerBottleneckDecoderConfig(NeMoTransformerBottleneckConfig):
    r2l: bool = False


class TransformerBottleneckEncoderNM(TransformerEncoderNM):

    _SUPPORTED_ARCH = ["seq2seq", "bridge", "perceiver", "max_pool", "avg_pool"]

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        inner_size: int,
        num_attention_heads: int,
        max_sequence_length: int = 512,
        num_token_types: int = 2,
        embedding_dropout: float = 0.0,
        learn_positional_encodings: bool = False,
        ffn_dropout: float = 0.0,
        attn_score_dropout: float = 0.0,
        attn_layer_dropout: float = 0.0,
        hidden_act: str = 'relu',
        mask_future: bool = False,
        pre_ln: bool = False,
        pre_ln_final_layer_norm: bool = True,
        arch: str = '',
        hidden_steps: int = -1,
        hidden_blocks: int = 1,
        hidden_init_method: str = "default",
        padding_idx: int = 0,
        # default whether forward() method returns hidden or (hidden, mask)
        return_mask=True,
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            inner_size=inner_size,
            num_attention_heads=num_attention_heads,
            max_sequence_length=max_sequence_length,
            num_token_types=num_token_types,
            embedding_dropout=embedding_dropout,
            learn_positional_encodings=learn_positional_encodings,
            ffn_dropout=ffn_dropout,
            attn_score_dropout=attn_score_dropout,
            attn_layer_dropout=attn_layer_dropout,
            hidden_act=hidden_act,
            mask_future=mask_future,
            pre_ln=pre_ln,
            pre_ln_final_layer_norm=pre_ln_final_layer_norm,
            padding_idx=padding_idx,
        )

        self._arch = arch
        self._return_mask = return_mask

        # replace encoder
        self._encoder = self._build_encoder(
            arch=arch,
            hidden_steps=hidden_steps,
            hidden_blocks=hidden_blocks,
            hidden_init_method=hidden_init_method,
            hidden_size=hidden_size,
            num_layers=num_layers,
            inner_size=inner_size,
            num_attention_heads=num_attention_heads,
            ffn_dropout=ffn_dropout,
            attn_score_dropout=attn_score_dropout,
            attn_layer_dropout=attn_layer_dropout,
            hidden_act=hidden_act,
            mask_future=mask_future,
            pre_ln=pre_ln,
            pre_ln_final_layer_norm=pre_ln_final_layer_norm,
        )

    def _build_encoder(self, arch, **kwargs):
        """
        Returns a decoder based on architecture arch and kwargs
        """
        # default non-bottleneck transformer encoder
        if (not arch) or (arch == "seq2seq"):
            encoder = self.encoder
        elif arch == "bridge":
            encoder = BridgeEncoder(
                num_layers=kwargs["num_layers"],
                hidden_size=kwargs["hidden_size"],
                inner_size=kwargs["inner_size"],
                num_attention_heads=kwargs["num_attention_heads"],
                attn_score_dropout=kwargs["attn_score_dropout"],
                attn_layer_dropout=kwargs["attn_layer_dropout"],
                ffn_dropout=kwargs["ffn_dropout"],
                hidden_act=kwargs["hidden_act"],
                mask_future=kwargs["mask_future"],
                pre_ln=kwargs["pre_ln"],
                pre_ln_final_layer_norm=kwargs["pre_ln_final_layer_norm"],
                hidden_steps=kwargs["hidden_steps"],
                hidden_blocks=kwargs["hidden_blocks"],
                hidden_init_method=kwargs["hidden_init_method"],
            )
        elif arch == "perceiver":
            encoder = PerceiverEncoder(
                num_layers=kwargs["num_layers"],
                hidden_size=kwargs["hidden_size"],
                inner_size=kwargs["inner_size"],
                num_attention_heads=kwargs["num_attention_heads"],
                attn_score_dropout=kwargs["attn_score_dropout"],
                attn_layer_dropout=kwargs["attn_layer_dropout"],
                ffn_dropout=kwargs["ffn_dropout"],
                hidden_act=kwargs["hidden_act"],
                mask_future=kwargs["mask_future"],
                pre_ln=kwargs["pre_ln"],
                pre_ln_final_layer_norm=kwargs["pre_ln_final_layer_norm"],
                hidden_steps=kwargs["hidden_steps"],
                hidden_blocks=kwargs["hidden_blocks"],
                hidden_init_method=kwargs["hidden_init_method"],
            )
        elif arch == "max_pool":
            encoder = PoolingEncoder(
                num_layers=kwargs["num_layers"],
                hidden_size=kwargs["hidden_size"],
                inner_size=kwargs["inner_size"],
                num_attention_heads=kwargs["num_attention_heads"],
                attn_score_dropout=kwargs["attn_score_dropout"],
                attn_layer_dropout=kwargs["attn_layer_dropout"],
                ffn_dropout=kwargs["ffn_dropout"],
                hidden_act=kwargs["hidden_act"],
                mask_future=kwargs["mask_future"],
                pre_ln=kwargs["pre_ln"],
                pre_ln_final_layer_norm=kwargs["pre_ln_final_layer_norm"],
                hidden_steps=kwargs["hidden_steps"],
                hidden_blocks=kwargs["hidden_blocks"],
                hidden_init_method=kwargs["hidden_init_method"],
                pooling_type="max",
            )
        elif arch == "avg_pool":
            encoder = PoolingEncoder(
                num_layers=kwargs["num_layers"],
                hidden_size=kwargs["hidden_size"],
                inner_size=kwargs["inner_size"],
                num_attention_heads=kwargs["num_attention_heads"],
                attn_score_dropout=kwargs["attn_score_dropout"],
                attn_layer_dropout=kwargs["attn_layer_dropout"],
                ffn_dropout=kwargs["ffn_dropout"],
                hidden_act=kwargs["hidden_act"],
                mask_future=kwargs["mask_future"],
                pre_ln=kwargs["pre_ln"],
                pre_ln_final_layer_norm=kwargs["pre_ln_final_layer_norm"],
                hidden_steps=kwargs["hidden_steps"],
                hidden_blocks=kwargs["hidden_blocks"],
                hidden_init_method=kwargs["hidden_init_method"],
                pooling_type="avg",
            )
        else:
            raise ValueError(f"Unknown arch = {self.arch}, supported arch = {self.supported_arch}")

        return encoder

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        input_types = super().input_types
        input_types.update(
            {"return_mask": NeuralType((), BoolType(), True),}
        )

        return input_types

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        output_types = super().output_types
        output_types.update(
            {"hidden_mask": NeuralType(('B', 'T'), MaskType(), True),}
        )
        return output_types

    @property
    def supported_arch(self):
        return self._SUPPORTED_ARCH

    @property
    def arch(self):
        return self._arch

    @typecheck()
    def forward(self, input_ids, encoder_mask, return_mask=None):
        if return_mask is None:
            return_mask = self._return_mask

        embeddings = self._embedding(input_ids=input_ids)

        if (not self.arch) or (self.arch == "seq2seq"):
            encoder_hidden_states = self._encoder(encoder_states=embeddings, encoder_mask=encoder_mask)
            encoder_hidden_mask = encoder_mask
        else:
            encoder_hidden_states, encoder_hidden_mask = self._encoder(
                encoder_states=embeddings, encoder_mask=encoder_mask,
            )

        if return_mask:
            return encoder_hidden_states, encoder_hidden_mask
        else:
            return encoder_hidden_states


class TransformerBottleneckDecoderNM(TransformerDecoderNM):
    _SUPPORTED_ARCH = ["seq2seq"]

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        inner_size: int,
        num_attention_heads: int,
        max_sequence_length: int = 512,
        num_token_types: int = 2,
        embedding_dropout: float = 0.0,
        learn_positional_encodings: bool = False,
        ffn_dropout: float = 0.0,
        attn_score_dropout: float = 0.0,
        attn_layer_dropout: float = 0.0,
        hidden_act: str = 'relu',
        pre_ln: bool = False,
        pre_ln_final_layer_norm: bool = True,
        arch='',
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            inner_size=inner_size,
            num_attention_heads=num_attention_heads,
            max_sequence_length=max_sequence_length,
            num_token_types=num_token_types,
            embedding_dropout=embedding_dropout,
            learn_positional_encodings=learn_positional_encodings,
            ffn_dropout=ffn_dropout,
            attn_score_dropout=attn_score_dropout,
            attn_layer_dropout=attn_layer_dropout,
            hidden_act=hidden_act,
            pre_ln=pre_ln,
            pre_ln_final_layer_norm=pre_ln_final_layer_norm,
        )

        self._arch = arch

        # replace decoder
        self._decoder = self._build_decoder(
            arch=arch,
            hidden_size=hidden_size,
            num_layers=num_layers,
            inner_size=inner_size,
            num_attention_heads=num_attention_heads,
            max_sequence_length=max_sequence_length,
            num_token_types=num_token_types,
            embedding_dropout=embedding_dropout,
            learn_positional_encodings=learn_positional_encodings,
            ffn_dropout=ffn_dropout,
            attn_score_dropout=attn_score_dropout,
            attn_layer_dropout=attn_layer_dropout,
            hidden_act=hidden_act,
            pre_ln=pre_ln,
            pre_ln_final_layer_norm=pre_ln_final_layer_norm,
        )

    def _build_decoder(self, arch, **kwargs):
        """
        Returns a decoder based on architecture arch and kwargs
        """
        # usual non-bottleneck transformer decoder
        if (not arch) or (arch == "seq2seq"):
            decoder = self.decoder
        else:
            raise ValueError(f"Unknown arch = {self.arch}, supported arch = {self.supported_arch}")

        return decoder

    @property
    def supported_arch(self):
        return self._SUPPORTED_ARCH

    @property
    def arch(self):
        return self._arch
