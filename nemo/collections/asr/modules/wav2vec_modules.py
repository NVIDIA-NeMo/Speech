# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import random
from typing import Dict, List, Tuple

import torch
from omegaconf import DictConfig
from omegaconf.dictconfig import DictConfig
from torch import nn
from torch.nn import functional as F

from nemo.collections.common.parts import form_attention_mask, transformer_weights_init
from nemo.collections.nlp.modules.common.transformer import TransformerEncoder
from nemo.core.classes.module import NeuralModule
from nemo.core.neural_types import AcousticEncodedRepresentation, AudioSignal, LengthsType, NeuralType, SpectrogramType


class TransposeLast(torch.nn.Module):
    """
    Transposes last dimension. Useful for adding to a sequential block.
    """

    def forward(self, x):
        return x.transpose(-2, -1)


class SamePad(torch.nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.remove = kernel_size % 2 == 0

    def forward(self, x):
        if self.remove:
            x = x[:, :, :-1]
        return x


class ConvFeatureEncoder(NeuralModule):
    """
		Encoder used to isolate features in raw audio for Wav2Vec style training.
		Treated as preprocessor module in NeMo ASR training. Defaults values are
		for base model found in Baeski et al (https://arxiv.org/abs/2006.11477),
		save for use of layer normalization as default schema. (Chosen for stability.) 
    """

    @property
    def input_types(self):
        """Returns definitions of module input ports.
        input_signal:
            0: AxisType(BatchTag)
            1: AxisType(TimeTag)
        input_signal_length:
            0: AxisType(BatchTag)
        Note: length is in number of samples, not seconds
        """
        return {
            "input_signal": NeuralType(('B', 'T'), AudioSignal(freq=self._sample_rate)),
            "length": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        """Returns definitions of module output ports. 
        For compatibility, processed features are treated as Spectrogram types
        processed_signal:
            0: AxisType(BatchTag)
            1: AxisType(ChannelTag)
            2: AxisType(ProcessedTimeTag)
        processed_signal_length:
            0: AxisType(BatchTag)
        """
        return {
            "processed_signal": NeuralType(('B', 'C', 'T'), SpectrogramType()),
            "processed_signal_length": NeuralType(tuple('B'), LengthsType()),
        }

    def __init__(
        self,
        conv_layers: List[Dict[str, int]],
        extractor_mode: str = "layer_norm",
        conv_bias: bool = False,
        feature_grad_mult=1.0,
        normalize_audio=True,
        embedding_dim=768,
    ):
        super().__init__()

        self.grad_mult = feature_grad_mult
        self.normalize_input = normalize_audio

        def block(
            n_in, n_out, k, stride, is_layer_norm=False, is_group_norm=False, conv_bias=False,
        ):
            def make_conv():
                conv = nn.Conv1d(n_in, n_out, k, stride=stride, bias=conv_bias)
                nn.init.kaiming_normal_(conv.weight)
                return conv

            assert (is_layer_norm and is_group_norm) is False, "layer norm and group norm are exclusive"

            if is_layer_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Sequential(TransposeLast(), nn.LayerNorm(dim, elementwise_affine=True), TransposeLast()),
                    nn.GELU(),
                )
            elif is_group_norm:
                return nn.Sequential(make_conv(), nn.GroupNorm(dim, dim, affine=True), nn.GELU(),)
            else:
                return nn.Sequential(make_conv(), nn.GELU())

        in_d = 1
        self.layer_cfg = conv_layers
        self.conv_layers = nn.ModuleList()
        self.mode = extractor_mode
        for i, cl in enumerate(conv_layers):
            assert len(cl) == 3, "invalid conv definition: " + str(cl)
            dim, k, stride = cl["emb_dim"], cl["kernel_size"], cl["stride"]

            self.conv_layers.append(
                block(
                    in_d,
                    dim,
                    k,
                    stride,
                    is_layer_norm=self.mode == "layer_norm",
                    is_group_norm=self.mode == "group_norm" and i == 0,  # applied to first layer only
                    conv_bias=conv_bias,
                )
            )
            in_d = dim

        # Model Layers
        final_conv_dim = self.layer_cfg[-1]["emb_dim"]  # Select last conv output layer dimension
        self.post_extract_proj = (  # To project feature encodings to transformer
            nn.Linear(final_conv_dim, embedding_dim) if final_conv_dim != embedding_dim else None
        )
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def apply_layers(self, x):
        for conv in self.conv_layers:
            x = conv(x)
        return x

    def normalize(self, source, lengths):
        with torch.no_grad():  # Normalizes audio source
            for i in range(lengths.size(0)):
                orig = source[i, : lengths[i]]
                norm = F.layer_norm(orig, orig.shape)
                source[i, : lengths[i]] = norm
        return source

    def forward(self, input_signal, length):
        if self.normalize_input:
            input_signal = self.normalize(input_signal, length)

        # BxT -> BxCxT
        processed_signal = input_signal.unsqueeze(1)

        # Applies grad mult scaling
        if self.grad_mult > 0:
            processed_signal = self.apply_layers(processed_signal)
            if self.grad_mult != 1.0:
                processed_signal = GradMultiply.apply(processed_signal, self.grad_mult)
        else:
            with torch.no_grad():  # 0 indicates frozen feature encoder
                processed_signal = self.apply_layers(processed_signal)

        processed_signal = processed_signal.transpose(1, 2)  # B,T,C
        # Project to embedding
        if self.post_extract_proj is not None:
            processed_signal = self.post_extract_proj(processed_signal)

        # Adding normalization for output
        if self.mode == "layer_norm":
            processed_signal = self.layer_norm(processed_signal)

        processed_signal = processed_signal.transpose(1, 2)  # B,C,T

        # Feature lengths will have been changed through convolutions
        processed_signal_length = self.get_lengths(audio_lengths=length)

        return processed_signal, processed_signal_length

    def get_lengths(self, audio_lengths):
        # converts audio lengths to timestep lengths
        for conv in self.layer_cfg:
            kernel = conv["kernel_size"]
            stride = conv["stride"]
            audio_lengths = (
                torch.div(audio_lengths - kernel, stride, rounding_mode='floor') + 1
            )  # from pytorch documentation
        return audio_lengths


class Wav2VecTransformerEncoder(TransformerEncoder):
    """
		Encoder module following Transformer encoder paradigm 
		as described in Vaswani et al. (https://arxiv.org/abs/1706.03762). Used for Wav2Vec
		style encoding of context vectors as described by in Baeski et al (https://arxiv.org/abs/2006.11477).
		Takes convolutional encodings of all time steps and adds to features before applying series
		of self-attention layers. 
		
		Example configs may be found at: https://github.com/NVIDIA/NeMo/tree/main/examples/asr/conf/wav2vec

		Args:
			layer_drop: Floating point value specifying proportion of module for layer dropout (See Fan et al. https://arxiv.org/pdf/1909.11556.pdf).
				If non-zero, each layer will draw from uniform probability to determine if applied in current forward call.
				Occurs only during training step
			pos_embed: Config specifying parameters for contextual embedding convolutions. Module configures convolutional padding
				to maintain number of time steps
				Must contain following:
					embedding_dim: Depth/number of channels of each time step from feature encoding 
					conv_pos: Kernel size for convolution
					conv_pos_groups: Number of groups for convolution
			transformer: Config for transformer encoder. Uses self-attention layers found in: nemo.collections.nlp.modules.common.transformer
				Must contain followign:
					num_layers: Number of attention layers 
					hidden_size: Expected input depth (embedding size between model layers)
					inner_size: Depth of embeddings within feed-forward sections of encoder layers
					num_attention_heads: Number of attention heads
					attn_score_dropout: Probability of dropout applied to attention scores
					attn_layer_dropout: Probability of dropout applied to the output of the attention layers (prior to normalization)
					ffn_dropout: Probability of dropout applied to feed-forward modules
					hidden_act: Activation function for hidden layers
    """

    def __init__(self, pos_embed: DictConfig, transformer: DictConfig, layer_drop: float = 0.0):
        super().__init__(**transformer)  # see nlp.collections

        # positional convolutional embeddings
        emb_dim = pos_embed.embedding_dim
        self.pos_conv = nn.Conv1d(
            emb_dim,
            emb_dim,
            kernel_size=pos_embed.conv_pos,
            padding=pos_embed.conv_pos // 2,  # Padding size preserves time step length
            groups=pos_embed.conv_pos_groups,
        )

        self.layer_drop = layer_drop

        self.dropout = transformer.attn_layer_dropout  # He initialization
        std = math.sqrt((4 * (1.0 - self.dropout)) / (pos_embed.conv_pos * pos_embed.embedding_dim))
        nn.init.normal_(self.pos_conv.weight, mean=0, std=std)
        nn.init.constant_(self.pos_conv.bias, 0)

        self.pos_conv = nn.utils.weight_norm(self.pos_conv, name="weight", dim=2)
        self.pos_conv = nn.Sequential(self.pos_conv, SamePad(pos_embed.conv_pos), nn.GELU())

        self.layer_norm = nn.LayerNorm(emb_dim)
        self.apply(lambda x: transformer_weights_init(x, xavier=False))

    @property
    def input_types(self):
        """Returns definitions of module output ports. 
        We treat features as SpectrogramType for Nemo compatibility
        audio_signal:
            0: AxisType(BatchTag)
            1: AxisType(ChannelTag)
            2: AxisType(ProcessedTimeTag)
        length:
            0: AxisType(BatchTag)
        """
        return {
            "audio_signal": NeuralType(('B', 'C', 'T'), SpectrogramType()),
            "length": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        """Returns definitions of module output ports. 
        We're using SpectrogramType for now to keep things Nemo safe
        processed_signal:
            0: AxisType(BatchTag)
            1: AxisType(ChannelTag)
            2: AxisType(ProcessedTimeTag)
        processed_length:
            0: AxisType(BatchTag)
        """
        return {
            "processed_signal": NeuralType(('B', 'C', 'T'), AcousticEncodedRepresentation()),
            "processed_length": NeuralType(tuple('B'), LengthsType()),
        }

    def forward(self, audio_signal, length):

        # Padding mask needed for transformer
        padding_mask = self.create_padding_mask(length)

        # Applying padding before convolution
        for idx, len in enumerate(length):
            audio_signal[idx, :, len:] = 0.0

        signal_conv = self.pos_conv(audio_signal)  # B, C, T
        audio_signal = audio_signal + signal_conv

        audio_signal = audio_signal.transpose(1, 2)  # B, C, T -> B, T, C
        audio_signal = self.layer_norm(audio_signal)

        context_emb = self.apply_transformer(audio_signal, padding_mask=padding_mask)

        context_emb = context_emb.transpose(1, 2)  # B, T, C -> B, C, T

        return context_emb, length  # Returning length for NeMo compatibility

    def apply_transformer(self, x, padding_mask=None):
        encoder_attn_mask = form_attention_mask(padding_mask)
        if (
            self.layer_drop and self.training
        ):  # Stochastic layer drop as in: Huang et al. https://arxiv.org/pdf/1603.09382.pdf
            for _, layer in enumerate(self.layers):
                p = random.random()
                if p > self.layer_drop:
                    x = layer(x, encoder_attn_mask, x)
        else:
            for _, layer in enumerate(self.layers):
                x = layer(x, encoder_attn_mask, x)
        return x

    def create_padding_mask(self, length):
        # Broadcast to vectorize creating the padding mask
        max_len = max(length)
        padding_mask = torch.arange(max_len, device=length.device)

        # Switch to binary for transformer, 1 for valid tokens, 0 for padding
        padding_mask = (padding_mask.expand(len(length), max_len) < length.unsqueeze(1)).type(torch.uint8)

        return padding_mask


class GradMultiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        res = x.new(x)
        return res

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None
