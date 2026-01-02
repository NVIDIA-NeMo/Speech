# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import random
from typing import Dict

import einops
import torch
import torch.nn as nn

from nemo.collections.asr.modules.conformer_encoder import ConformerEncoder
from nemo.core.classes.common import typecheck

from nemo.core.classes.module import NeuralModule
from nemo.core.neural_types import ChannelType, LengthsType, NeuralType, SpectrogramType
from nemo.utils import logging

__all__ = ['SpectrogramConformerUNet', 'ConformerEncoderUNet']


class ConformerEncoderUNet(ConformerEncoder):
    """
    ConformerEncoder with U-Net-style skip connections for enhanced audio processing.

    Inherits all functionality from ConformerEncoder and adds U-Net skip connections
    where the first encoder layer connects to the last layer, the second to the
    second-last, and so on — but without any time-domain subsampling.

    Based on:
        'Conformer: Convolution-augmented Transformer for Speech Recognition' by Anmol Gulati et al.
        https://arxiv.org/abs/2005.08100

        U-Net skip connections inspired by:
        Le et al., Voicebox: Text-Guided Multilingual Universal Speech Generation at Scale, 2023

    Args:
        n_layers (int): Number of ConformerBlock layers. Must be even (divisible by 2) when
            use_unet_skip_connection=True to enable symmetric skip connections between
            the first and second halves of the encoder.
        use_unet_skip_connection (bool): Enable U-Net style skip connections between encoder layers.
            When True, creates skip connections from the first half of layers to the second half.
            Defaults to True.
        skip_connect_scale (float, optional): Scaling factor applied to skip connections before
            concatenation with the main signal. If None, defaults to 2^(-0.5) ≈ 0.707.
            Defaults to None.
        **kwargs: All other arguments are passed to the parent ConformerEncoder class.
            See :class:`~nemo.collections.asr.modules.conformer_encoder.ConformerEncoder`
            for complete documentation of all available parameters including:
            - Model architecture (feat_in, n_layers, d_model, etc.)
            - Attention settings (self_attention_model, att_context_size, etc.)
            - Subsampling and reduction options
            - Dropout and regularization parameters
            - Streaming and caching configurations

    """

    def __init__(
        self,
        *args,
        use_unet_skip_connection: bool = True,
        skip_connect_scale: float | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.use_unet_skip_connection = use_unet_skip_connection
        self.skip_connect_scale = 2**-0.5 if skip_connect_scale is None else skip_connect_scale

        if not use_unet_skip_connection:
            logging.warning('Skip connections are disabled in the ConformerEncoderUNet.')
            return

        # Validate that n_layers is even for symmetric U-Net skip connections
        if self.n_layers % 2 != 0:
            raise ValueError(
                f"For U-Net skip connections, n_layers must be even (divisible by 2), "
                f"but got n_layers={self.n_layers}. This ensures symmetric skip connections "
                f"between the first and second halves of the encoder."
            )

        new_layers = nn.ModuleList()
        mid = len(self.layers) // 2
        for idx, layer in enumerate(self.layers):
            has_skip = idx >= mid
            combiner = nn.Linear(self.d_model * 2, self.d_model) if has_skip else None
            new_layers.append(nn.ModuleList([combiner, layer]))
        self.layers = new_layers

    def forward_internal(
        self,
        audio_signal,
        length,
        cache_last_channel=None,
        cache_last_time=None,
        cache_last_channel_len=None,
        bypass_pre_encode=None,
    ):
        """
        Forward pass for the ConformerEncoderUNet model with U-Net-style skip connections.

        This method processes the input audio signal through the Conformer layers with optional
        caching and U-Net-style skip connections.

        Main Changes Compared to Original Conformer:
        - Incorporates U-Net-style skip connections between encoder layers.
        """
        if length is None:
            length = audio_signal.new_full(
                (audio_signal.size(0),), audio_signal.size(-1), dtype=torch.int64, device=audio_signal.device
            )

        # select a random att_context_size with the distribution specified by att_context_probs during training
        # for non-validation cases like test, validation or inference, it uses the first mode in self.att_context_size
        if self.training and len(self.att_context_size_all) > 1:
            cur_att_context_size = random.choices(self.att_context_size_all, weights=self.att_context_probs)[0]
        else:
            cur_att_context_size = self.att_context_size

        audio_signal = torch.transpose(audio_signal, 1, 2)

        if isinstance(self.pre_encode, nn.Linear):
            audio_signal = self.pre_encode(audio_signal)
        else:
            audio_signal, length = self.pre_encode(x=audio_signal, lengths=length)
            length = length.to(torch.int64)
            # self.streaming_cfg is set by setup_streaming_cfg(), called in the init
            if self.streaming_cfg.drop_extra_pre_encoded > 0 and cache_last_channel is not None:
                audio_signal = audio_signal[:, self.streaming_cfg.drop_extra_pre_encoded :, :]
                length = (length - self.streaming_cfg.drop_extra_pre_encoded).clamp(min=0)

        if self.reduction_position is not None and cache_last_channel is not None:
            raise ValueError("Caching with reduction feature is not supported yet!")

        max_audio_length = audio_signal.size(1)
        if cache_last_channel is not None:
            cache_len = self.streaming_cfg.last_channel_cache_size
            cache_keep_size = max_audio_length - self.streaming_cfg.cache_drop_size
            max_audio_length = max_audio_length + cache_len
            padding_length = length + cache_len
            offset = torch.neg(cache_last_channel_len) + cache_len
        else:
            padding_length = length
            cache_last_channel_next = None
            cache_len = 0
            offset = None

        audio_signal, pos_emb = self.pos_enc(x=audio_signal, cache_len=cache_len)

        # Create the self-attention and padding masks
        pad_mask, att_mask = self._create_masks(
            att_context_size=cur_att_context_size,
            padding_length=padding_length,
            max_audio_length=max_audio_length,
            offset=offset,
            device=audio_signal.device,
        )

        if cache_last_channel is not None:
            pad_mask = pad_mask[:, cache_len:]
            if att_mask is not None:
                att_mask = att_mask[:, cache_len:]
            # Convert caches from the tensor to list
            cache_last_time_next = []
            cache_last_channel_next = []

        skip_connects = []

        for lth, (drop_prob, (skip_combiner, layer)) in enumerate(zip(self.layer_drop_probs, self.layers)):

            if skip_combiner is None:
                skip_connects.append(audio_signal)
            else:
                skip_connect = skip_connects.pop() * self.skip_connect_scale
                audio_signal = torch.cat((audio_signal, skip_connect), dim=-1)
                audio_signal = skip_combiner(audio_signal)

            original_signal = audio_signal
            if cache_last_channel is not None:
                cache_last_channel_cur = cache_last_channel[lth]
                cache_last_time_cur = cache_last_time[lth]
            else:
                cache_last_channel_cur = None
                cache_last_time_cur = None

            audio_signal = layer(
                x=audio_signal,
                att_mask=att_mask,
                pos_emb=pos_emb,
                pad_mask=pad_mask,
                cache_last_channel=cache_last_channel_cur,
                cache_last_time=cache_last_time_cur,
            )

            if cache_last_channel_cur is not None:
                (audio_signal, cache_last_channel_cur, cache_last_time_cur) = audio_signal
                cache_last_channel_next.append(cache_last_channel_cur)
                cache_last_time_next.append(cache_last_time_cur)

            # applying stochastic depth logic from https://arxiv.org/abs/2102.03216
            if self.training and drop_prob > 0.0:
                should_drop = torch.rand(1) < drop_prob
                # adjusting to match expectation
                if should_drop:
                    # that's not efficient, but it's hard to implement distributed
                    # version of dropping layers without deadlock or random seed meddling
                    # so multiplying the signal by 0 to ensure all weights get gradients
                    audio_signal = audio_signal * 0.0 + original_signal
                else:
                    # not doing this operation if drop prob is 0 as it's identity in that case
                    audio_signal = (audio_signal - original_signal) / (1.0 - drop_prob) + original_signal

            if self.reduction_position == lth:
                audio_signal, length = self.reduction_subsampling(x=audio_signal, lengths=length)
                max_audio_length = audio_signal.size(1)
                # Don't update the audio_signal here because then it will again scale the audio_signal
                # and cause an increase in the WER
                _, pos_emb = self.pos_enc(x=audio_signal, cache_len=cache_len)
                pad_mask, att_mask = self._create_masks(
                    att_context_size=cur_att_context_size,
                    padding_length=length,
                    max_audio_length=max_audio_length,
                    offset=offset,
                    device=audio_signal.device,
                )

            # saving tensors if required for interctc loss
            if self.is_access_enabled(getattr(self, "model_guid", None)):
                if self.interctc_capture_at_layers is None:
                    self.interctc_capture_at_layers = self.access_cfg.get('interctc', {}).get('capture_layers', [])
                if lth in self.interctc_capture_at_layers:
                    lth_audio_signal = audio_signal
                    if self.out_proj is not None:
                        lth_audio_signal = self.out_proj(audio_signal)
                    # shape is the same as the shape of audio_signal output, i.e. [B, D, T]
                    self.register_accessible_tensor(
                        name=f'interctc/layer_output_{lth}', tensor=torch.transpose(lth_audio_signal, 1, 2)
                    )
                    self.register_accessible_tensor(name=f'interctc/layer_length_{lth}', tensor=length)

        if self.out_proj is not None:
            audio_signal = self.out_proj(audio_signal)

        # Reduction
        if self.reduction_position == -1:
            audio_signal, length = self.reduction_subsampling(x=audio_signal, lengths=length)

        audio_signal = torch.transpose(audio_signal, 1, 2)
        length = length.to(dtype=torch.int64)

        if cache_last_channel is not None:
            cache_last_channel_next = torch.stack(cache_last_channel_next, dim=0)
            cache_last_time_next = torch.stack(cache_last_time_next, dim=0)
            return (
                audio_signal,
                length,
                cache_last_channel_next,
                cache_last_time_next,
                torch.clamp(cache_last_channel_len + cache_keep_size, max=cache_len),
            )
        else:
            return audio_signal, length


class SpectrogramConformerUNet(NeuralModule):
    """A Conformer-based model for processing complex-valued spectrograms.

    This model processes complex-valued inputs by stacking real and imaginary components
    along the channel dimension. The stacked tensor is processed using Conformer layers,
    and the output is projected back to generate real and imaginary components of the
    output channels.

    Args:
        in_channels: number of input complex-valued channels
        out_channels: number of output complex-valued channels
        kwargs: additional arguments for ConformerEncoderUNet
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1, **kwargs):
        super().__init__()

        # Number of input channels for this estimator
        if in_channels < 1:
            raise ValueError(
                f'Number of input channels needs to be larger or equal to one, current value {in_channels}'
            )

        self.in_channels = in_channels

        # Number of output channels for this estimator
        if out_channels < 1:
            raise ValueError(
                f'Number of output channels needs to be larger or equal to one, current value {out_channels}'
            )

        self.out_channels = out_channels

        # Conformer-based estimator
        conformer_params = kwargs.copy()
        conformer_params['feat_in'] = conformer_params['feat_out'] = (
            2 * self.in_channels * kwargs['feat_in']
        )  # stack real and imag
        logging.info('Conformer params: %s', conformer_params)
        self.conformer = ConformerEncoderUNet(**conformer_params)
        # logging.info(self.conformer)

        # Output projection to generate real and imaginary components of the output channels
        self.output_projection = torch.nn.Conv2d(
            in_channels=2 * self.in_channels, out_channels=2 * self.out_channels, kernel_size=1
        )

        logging.debug('Initialized %s with', self.__class__.__name__)
        logging.debug('\tin_channels:  %s', self.in_channels)
        logging.debug('\tout_channels: %s', self.out_channels)

    @property
    def context_size(self):
        """Returns the attention context size used by the conformer encoder.

        The context size is a list of two integers [left_context, right_context] that defines
        how many frames to the left and right each frame can attend to in the self-attention
        layers.

        Returns:
            List[int]: The attention context size [left_context, right_context]
        """
        return self.conformer.att_context_size

    @context_size.setter
    def context_size(self, value):
        """Sets the attention context size used by the conformer encoder.

        The context size is a list of two integers [left_context, right_context] that defines
        how many frames to the left and right each frame can attend to in the self-attention
        layers.

        Args:
            value (List[int]): The attention context size [left_context, right_context]
        """
        self.conformer.set_default_att_context_size(value)

    @property
    def input_types(self) -> Dict[str, NeuralType]:
        """Returns definitions of module output ports."""
        return {
            "input": NeuralType(('B', 'C', 'D', 'T'), SpectrogramType()),
            "input_length": NeuralType(('B',), LengthsType(), optional=True),
            # convolutional context
            "cache_last_channel": NeuralType(('D', 'B', 'T', 'D'), ChannelType(), optional=True),
            "cache_last_time": NeuralType(('D', 'B', 'D', 'T'), ChannelType(), optional=True),
            "cache_last_channel_len": NeuralType(tuple('B'), LengthsType(), optional=True),
        }

    @property
    def output_types(self) -> Dict[str, NeuralType]:
        """Returns definitions of module output ports."""
        return {
            "output": NeuralType(('B', 'C', 'D', 'T'), SpectrogramType()),
            "output_length": NeuralType(('B',), LengthsType(), optional=True),
            # convolutional context
            "cache_last_channel_next": NeuralType(('D', 'B', 'T', 'D'), ChannelType(), optional=True),
            "cache_last_time_next": NeuralType(('D', 'B', 'D', 'T'), ChannelType(), optional=True),
            "cache_last_channel_next_len": NeuralType(tuple('B'), LengthsType(), optional=True),
        }

    @typecheck()
    def forward(
        self, input, input_length=None, cache_last_channel=None, cache_last_time=None, cache_last_channel_len=None
    ):

        # Stack real and imaginary components
        B, C_in, D, T = input.shape
        if C_in != self.in_channels:
            raise RuntimeError(f'Unexpected input channel size {C_in}, expected {self.in_channels}')

        input_real_imag = torch.stack([input.real, input.imag], dim=2)
        input = einops.rearrange(input_real_imag, 'B C RI D T -> B (C RI D) T')

        # Conformer
        if cache_last_channel is None:
            # Not using caching mode
            output, output_length = self.conformer(audio_signal=input, length=input_length)
        else:
            # Using caching mode
            output, output_length, cache_last_channel, cache_last_time, cache_last_channel_len = self.conformer(
                audio_signal=input,
                length=input_length,
                cache_last_channel=cache_last_channel,
                cache_last_time=cache_last_time,
                cache_last_channel_len=cache_last_channel_len,
            )

        # Output projection
        output = einops.rearrange(output, 'B (C RI D) T -> B (C RI) D T', C=self.in_channels, RI=2, D=D)
        output = self.output_projection(output)

        # Convert to complex-valued signal
        output = einops.rearrange(output, 'B (C RI) D T -> B C D T RI', C=self.out_channels, RI=2, D=D)

        # torch.view_as_complex doesn't support BFloat16, convert to float32 if needed
        if output.dtype == torch.bfloat16:
            output = output.float()

        output = torch.view_as_complex(output.contiguous())

        if cache_last_channel is None:
            return output, output_length
        else:
            return output, output_length, cache_last_channel, cache_last_time, cache_last_channel_len
