# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import math
import os
from abc import ABC, abstractmethod
from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers import AutoModel

from nemo.collections.asr.modules import AudioToMelSpectrogramPreprocessor
from nemo.collections.audio.parts.utils.transforms import MelSpectrogram, Resample
from nemo.collections.common.parts.utils import ClampActivation, HalfSnake, Snake, mask_sequence_tensor
from nemo.core.classes.common import typecheck
from nemo.core.classes.module import NeuralModule
from nemo.core.neural_types.elements import (
    AudioSignal,
    EncodedRepresentation,
    LengthsType,
    MelSpectrogramType,
    TokenIndex,
    VoidType,
)
from nemo.core.neural_types.neural_type import NeuralType
from nemo.utils import logging

try:
    import fsspec

    HAVE_FSSPEC = True
except ModuleNotFoundError:
    HAVE_FSSPEC = False


from contextlib import contextmanager


@contextmanager
def default_precision(dtype=torch.float32):
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(default_dtype)


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return (kernel_size * dilation - dilation) // 2


def get_padding_2d(kernel_size: Tuple[int, int], dilation: Tuple[int, int]) -> Tuple[int, int]:
    paddings = (get_padding(kernel_size[0], dilation[0]), get_padding(kernel_size[1], dilation[1]))
    return paddings


def get_down_sample_padding(kernel_size: int, stride: int) -> int:
    return (kernel_size - stride + 1) // 2


def get_up_sample_padding(kernel_size: int, stride: int) -> Tuple[int, int]:
    output_padding = (kernel_size - stride) % 2
    padding = (kernel_size - stride + 1) // 2
    return padding, output_padding


class SSLModel(NeuralModule):
    def __init__(self, slm_model_name):
        super().__init__()
        self.ssl_model = AutoModel.from_pretrained(slm_model_name)

    def forward(self, *args, **kwargs):
        return self.ssl_model(*args, **kwargs)


class SLMDiscriminator(NeuralModule):
    """SLM Discriminator, as described in both the StyleTTS2 and Low Frame-Rate Speech Codec papers.

    Args:
        slm_model_name: Hugging Face Speech Language Models name.
        slm_sr: Speech Language Models input sampling rate.
        input_sr: Audio input sampling rate.
        slm_hidden: Speech Language Model hidden dim.
        slm_layers: Speech Language Model number of layers.
        initial_channel: discriminative head number of channels.
        use_spectral_norm: If True uses spectral normalization otherwise uses weight norm.

    """

    def __init__(
        self,
        slm_model_name="microsoft/wavlm-base-plus",
        slm_sr=16000,
        input_sr=22050,
        slm_hidden=768,
        slm_layers=13,
        initial_channel=64,
        use_spectral_norm=False,
    ):
        super().__init__()

        self.resample = Resample(orig_freq=input_sr, new_freq=slm_sr)
        self.slm_model = SSLModel(slm_model_name)

        # Freeze slm model
        self.slm_model.freeze()

        norm_f = (
            torch.nn.utils.parametrizations.weight_norm if use_spectral_norm == False else torch.nn.utils.spectral_norm
        )
        self.pre = norm_f(nn.Conv1d(slm_hidden * slm_layers, initial_channel, 1, 1, padding=0))

        self.convs = nn.ModuleList(
            [
                norm_f(nn.Conv1d(initial_channel, initial_channel * 2, kernel_size=5, padding=2)),
                norm_f(nn.Conv1d(initial_channel * 2, initial_channel * 4, kernel_size=5, padding=2)),
                norm_f(nn.Conv1d(initial_channel * 4, initial_channel * 4, 5, 1, padding=2)),
            ]
        )

        self.conv_post = norm_f(nn.Conv1d(initial_channel * 4, 1, 3, 1, padding=1))

    def _forward(self, x):
        x = self.slm_model(input_values=self.resample(x), output_hidden_states=True).hidden_states
        x = torch.stack(x, dim=1).transpose(-1, -2).flatten(start_dim=1, end_dim=2)

        x = self.pre(x)
        fmap = []
        for layer in self.convs:
            x = layer(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x.unsqueeze(-1))

        x = self.conv_post(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap

    @property
    def input_types(self):
        return {
            "audio_real": NeuralType(('B', 'T_audio'), AudioSignal()),
            "audio_gen": NeuralType(('B', 'T_audio'), AudioSignal()),
        }

    @property
    def output_types(self):
        return {
            "scores_real": [NeuralType(('B', 'C', 'T_out'), VoidType())],
            "scores_gen": [NeuralType(('B', 'C', 'T_out'), VoidType())],
            "fmaps_real": [[NeuralType(('B', 'D', 'T_layer', 'C'), VoidType())]],
            "fmaps_gen": [[NeuralType(('B', 'D', 'T_layer', 'C'), VoidType())]],
        }

    @typecheck()
    def forward(self, audio_real, audio_gen):

        y_d_r, fmap_r = self._forward(audio_real)
        y_d_g, fmap_g = self._forward(audio_gen)

        return [y_d_r.unsqueeze(1)], [y_d_g.unsqueeze(1)], [fmap_r], [fmap_g]


# Torch version of transformers.models.wav2vec2.feature_extraction_wav2vec2.Wav2Vec2FeatureExtractor.zero_mean_unit_var_norm
def zero_mean_unit_var_norm(input_values):
    """
    Normalized to have zero mean and unit variance
    """
    normed_input_values = (input_values - input_values.mean(dim=1).unsqueeze(-1)) / torch.sqrt(
        input_values.var(dim=1).unsqueeze(-1) + 1e-7
    )
    return normed_input_values


##############
# Speaker encoder #
##############
def load_fsspec(path: str, map_location: str = None, **kwargs):
    """Like torch.load but can load from other locations (e.g. s3:// , gs://).

    Args:
        path: Any path or url supported by fsspec.
        map_location: torch.device or str.
        cache: If True, cache a remote file locally for subsequent calls. It is cached under `get_user_data_dir()/tts_cache`. Defaults to True.
        **kwargs: Keyword arguments forwarded to torch.load.

    Returns:
        Object stored in path.
    """
    is_local = os.path.isdir(path) or os.path.isfile(path)
    if is_local:
        return torch.load(path, map_location=map_location, **kwargs)
    else:
        if HAVE_FSSPEC:
            with fsspec.open(path, "rb") as f:
                return torch.load(f, map_location=map_location, **kwargs)
        else:
            logging.error('Could not import fsspec. Loading a checkpoint link is not supported!')
            raise ModuleNotFoundError("fsspec is not installed but is necessary to download remote checkpoints !!")


class PreEmphasis(NeuralModule):
    def __init__(self, coefficient=0.97):
        super().__init__()
        self.coefficient = coefficient
        self.register_buffer("filter", torch.FloatTensor([-self.coefficient, 1.0]).unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        assert len(x.size()) == 2

        x = torch.nn.functional.pad(x.unsqueeze(1), (1, 0), "reflect")
        return torch.nn.functional.conv1d(x, self.filter).squeeze(1)


class SELayer(NeuralModule):
    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class SEBasicBlock(NeuralModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=8):
        super(SEBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class ResNetSpeakerEncoder(NeuralModule):
    """Implementation of the model H/ASP without batch normalization in speaker embedding. This model was proposed in: https://arxiv.org/abs/2009.14153
    Adapted from: https://github.com/clovaai/voxceleb_trainer
    """

    def __init__(
        self,
        input_dim=64,
        proj_dim=512,
        layers=[3, 4, 6, 3],
        num_filters=[32, 64, 128, 256],
        encoder_type="ASP",
        log_input=True,
        use_torch_spec=True,
        audio_config={
            "fft_size": 512,
            "win_length": 400,
            "hop_length": 160,
            "frame_shift_ms": None,
            "frame_length_ms": None,
            "stft_pad_mode": "reflect",
            "sample_rate": 16000,
            "resample": False,
            "preemphasis": 0.97,
            "ref_level_db": 20,
            "do_sound_norm": False,
            "do_trim_silence": False,
            "trim_db": 60,
            "power": 1.5,
            "griffin_lim_iters": 60,
            "num_mels": 64,
            "mel_fmin": 0.0,
            "mel_fmax": 8000.0,
            "spec_gain": 20,
            "signal_norm": False,
            "min_level_db": -100,
            "symmetric_norm": False,
            "max_norm": 4.0,
            "clip_norm": False,
            "stats_path": None,
            "do_rms_norm": True,
            "db_level": -27.0,
        },
    ):
        super(ResNetSpeakerEncoder, self).__init__()

        self.encoder_type = encoder_type
        self.input_dim = input_dim
        self.log_input = log_input
        self.use_torch_spec = use_torch_spec
        self.audio_config = audio_config
        self.proj_dim = proj_dim

        self.conv1 = nn.Conv2d(1, num_filters[0], kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(num_filters[0])

        self.inplanes = num_filters[0]
        self.layer1 = self.create_layer(SEBasicBlock, num_filters[0], layers[0])
        self.layer2 = self.create_layer(SEBasicBlock, num_filters[1], layers[1], stride=(2, 2))
        self.layer3 = self.create_layer(SEBasicBlock, num_filters[2], layers[2], stride=(2, 2))
        self.layer4 = self.create_layer(SEBasicBlock, num_filters[3], layers[3], stride=(2, 2))

        self.instancenorm = nn.InstanceNorm1d(input_dim)
        self.torch_spec = self.get_torch_mel_spectrogram_class(audio_config) if self.use_torch_spec else None

        outmap_size = int(self.input_dim / 8)

        self.attention = nn.Sequential(
            nn.Conv1d(num_filters[3] * outmap_size, 128, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, num_filters[3] * outmap_size, kernel_size=1),
            nn.Softmax(dim=2),
        )

        if self.encoder_type == "SAP":
            out_dim = num_filters[3] * outmap_size
        elif self.encoder_type == "ASP":
            out_dim = num_filters[3] * outmap_size * 2
        else:
            raise ValueError("Undefined encoder")

        self.fc = nn.Linear(out_dim, proj_dim)

        self._init_layers()

    def _init_layers(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def create_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def new_parameter(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out

    def forward(self, x, l2_norm=False):
        """Forward pass of the model.

        Args:
            x (Tensor): Raw waveform signal or spectrogram frames. If input is a waveform, `torch_spec` must be `True`
                to compute the spectrogram on-the-fly.
            l2_norm (bool): Whether to L2-normalize the outputs.

        Shapes:
            - x: :math:`(N, 1, T_{in})` or :math:`(N, D_{spec}, T_{in})`
        """
        with default_precision(torch.float32):
            x.squeeze_(1)
            # if you torch spec compute it otherwise use the mel spec computed by the AP
            if self.use_torch_spec:
                x = self.torch_spec(x)

            if self.log_input:
                x = (x + 1e-6).log()
            x = self.instancenorm(x).unsqueeze(1)

            x = self.conv1(x)
            x = self.relu(x)
            x = self.bn1(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = x.reshape(x.size()[0], -1, x.size()[-1])

            w = self.attention(x)

            if self.encoder_type == "SAP":
                x = torch.sum(x * w, dim=2)
            elif self.encoder_type == "ASP":
                mu = torch.sum(x * w, dim=2)
                sg = torch.sqrt((torch.sum((x**2) * w, dim=2) - mu**2).clamp(min=1e-5))
                x = torch.cat((mu, sg), 1)

            x = x.view(x.size()[0], -1)
            x = self.fc(x)

            if l2_norm:
                x = torch.nn.functional.normalize(x, p=2, dim=1)
        return x

    def get_torch_mel_spectrogram_class(self, audio_config):
        return torch.nn.Sequential(
            PreEmphasis(audio_config["preemphasis"]),
            MelSpectrogram(
                sample_rate=audio_config["sample_rate"],
                n_fft=audio_config["fft_size"],
                win_length=audio_config["win_length"],
                hop_length=audio_config["hop_length"],
                window_fn=torch.hamming_window,
                n_mels=audio_config["num_mels"],
            ),
        )

    def load_checkpoint(self, checkpoint_path: str, strict=True):
        state = load_fsspec(checkpoint_path, map_location=torch.device("cpu"))
        self.load_state_dict(state["model"], strict=strict)


class CodecActivation(nn.Module):
    """
    Choose between activation based on the input parameter.

    Args:
        activation: Name of activation to use. Valid options are "elu" (default), "lrelu", and "snake".
        channels: Input dimension.
    """

    def __init__(self, activation: str = "elu", channels: int = 1):
        super().__init__()
        activation = activation.lower()
        if activation == "elu":
            self.activation = nn.ELU()
        elif activation == "lrelu":
            self.activation = torch.nn.LeakyReLU()
        elif activation == "snake":
            self.activation = Snake(channels)
        elif activation == "half_snake":
            self.activation = HalfSnake(channels)
        else:
            raise ValueError(f"Unknown activation {activation}")

    def forward(self, x):
        return self.activation(x)


class CausalConvTranspose1dNorm(NeuralModule):
    """ConvTranspose1d causal padding and normalization."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = None,
        activation: Optional[str] = None,
        trim_right_ratio: int = 1,
        bias=True,
    ):
        super().__init__()

        self.trim_right_ratio = trim_right_ratio

        # if groups are None, create a group for each out channel as done in Mini Codec
        groups = out_channels if groups is None else groups

        self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, groups=groups, bias=bias)

        if activation is not None:
            self.activation = CodecActivation(activation=activation, channels=out_channels)
        else:
            self.activation = nn.Identity()

        kernel_size = self.conv.kernel_size[0]
        stride = self.conv.stride[0]
        padding_total = kernel_size - stride

        # Trim the padding on the right according to the specified ratio
        # if trim_right_ratio = 1.0, trim everything from right
        self.padding_right = math.ceil(padding_total * self.trim_right_ratio)
        self.padding_left = padding_total - self.padding_right

        # add weight norm
        self.conv = nn.utils.parametrizations.weight_norm(self.conv)

    def apply_weight_norm(self):
        weight_norm = nn.utils.parametrizations.weight_norm
        if hasattr(nn.utils.parametrizations, "weight_norm"):
            weight_norm = nn.utils.parametrizations.weight_norm

        weight_norm(self.conv)

    def remove_weight_norm(self):
        nn.utils.remove_weight_norm(self.conv)

    def forward(self, inputs, input_len):
        hidden_states = self.conv(inputs)

        # unpad
        end = hidden_states.shape[-1] - self.padding_right
        hidden_states = hidden_states[..., self.padding_left : end]
        hidden_states = self.activation(hidden_states)
        # mask
        hidden_states = mask_sequence_tensor(hidden_states, input_len)
        return hidden_states


class CausalConv1dNorm(NeuralModule):
    """Conv1d with causal padding and normalization."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        activation: Optional[str] = None,
        pad_mode: str = "zeros",
        extra_pad_mode: str = "constant",
        bias: bool = True,
    ):
        super().__init__()
        self.extra_pad_mode = extra_pad_mode

        # warn user on unusual setup between dilation and stride
        if stride > 1 and dilation > 1:
            print(
                "CausalConv1dNorm has been initialized with stride > 1 and dilation > 1"
                f" (kernel_size={kernel_size} stride={stride}, dilation={dilation})."
            )

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=pad_mode,
        )
        if activation is not None:
            self.activation = CodecActivation(activation=activation, channels=out_channels)
        else:
            self.activation = nn.Identity()

        kernel_size = self.conv.kernel_size[0]
        stride = torch.tensor(self.conv.stride[0], dtype=torch.int64)
        dilation = self.conv.dilation[0]

        # Effective kernel size with dilations.
        kernel_size = torch.tensor((kernel_size - 1) * dilation + 1, dtype=torch.int64)

        self.register_buffer("stride", stride, persistent=False)
        self.register_buffer("kernel_size", kernel_size, persistent=False)
        self.register_buffer("padding_total", torch.tensor(kernel_size - stride, dtype=torch.int64), persistent=False)

        # add weight norm
        self.conv = nn.utils.parametrizations.weight_norm(self.conv)

    def remove_weight_norm(self):
        nn.utils.remove_weight_norm(self.conv)

    # Copied from transformers.models.encodec.modeling_encodec.EncodecConv1d._get_extra_padding_for_conv1d
    def _get_extra_padding_for_conv1d(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """See `pad_for_conv1d`."""
        with default_precision(torch.float32):
            length = hidden_states.shape[-1]
            n_frames = (length - self.kernel_size + self.padding_total) / self.stride + 1
            n_frames = torch.ceil(n_frames).to(torch.int64) - 1
            ideal_length = (n_frames * self.stride).long() + self.kernel_size - self.padding_total
        return (ideal_length - length).long()

    @staticmethod
    # Copied from transformers.models.encodec.modeling_encodec.EncodecConv1d._pad1d
    def _pad1d(hidden_states: torch.Tensor, paddings: Tuple[int, int], mode: str = "zero", value: float = 0.0):
        """Tiny wrapper around torch.nn.functional.pad, just to allow for reflect padding on small input.
        If this is the case, we insert extra 0 padding to the right before the reflection happens.
        """
        length = hidden_states.shape[-1]
        padding_left, padding_right = paddings
        if not mode == "reflect":
            return nn.functional.pad(hidden_states, paddings, mode, value)

        max_pad = max(padding_left, padding_right)
        extra_pad = 0
        if length <= max_pad:
            extra_pad = max_pad - length + 1
            hidden_states = nn.functional.pad(hidden_states, (0, extra_pad))
        padded = nn.functional.pad(hidden_states, paddings, mode, value)
        end = padded.shape[-1] - extra_pad
        return padded[..., :end]

    def forward(self, inputs, input_len):
        extra_padding = self._get_extra_padding_for_conv1d(inputs)

        # Left padding for causal
        hidden_states = self._pad1d(inputs, (self.padding_total, extra_padding), mode=self.extra_pad_mode)
        hidden_states = self.conv(hidden_states)
        hidden_states = self.activation(hidden_states)

        # mask output
        hidden_states = mask_sequence_tensor(hidden_states, input_len)

        return hidden_states


class Conv1dNorm(NeuralModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        padding: Optional[int] = None,
        pad_mode: str = "reflect",
        activation: Optional[str] = None,
    ):
        super().__init__()
        if not padding:
            padding = get_padding(kernel_size=kernel_size, dilation=dilation)
        conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            padding_mode=pad_mode,
        )
        self.conv = nn.utils.parametrizations.weight_norm(conv)
        if activation is not None:
            self.activation = CodecActivation(activation=activation, channels=out_channels)
        else:
            self.activation = torch.nn.Identity()

    @property
    def input_types(self):
        return {
            "inputs": NeuralType(('B', 'C', 'T'), VoidType()),
            "input_len": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        return {
            "out": NeuralType(('B', 'C', 'T'), VoidType()),
        }

    def remove_weight_norm(self):
        nn.utils.remove_weight_norm(self.conv)

    @typecheck()
    def forward(self, inputs, input_len):
        out = self.conv(inputs)
        out = self.activation(out)
        out = mask_sequence_tensor(out, input_len)
        return out


class ConvTranspose1dNorm(NeuralModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        activation: Optional[str] = None,
    ):
        super().__init__()
        padding, output_padding = get_up_sample_padding(kernel_size, stride)
        conv = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            padding_mode="zeros",
            groups=groups,
        )
        self.conv = nn.utils.parametrizations.weight_norm(conv)

        if activation is not None:
            self.activation = CodecActivation(activation=activation, channels=out_channels)
        else:
            self.activation = nn.Identity()

    @property
    def input_types(self):
        return {
            "inputs": NeuralType(('B', 'C', 'T'), VoidType()),
            "input_len": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        return {
            "out": NeuralType(('B', 'C', 'T'), VoidType()),
        }

    def remove_weight_norm(self):
        nn.utils.remove_weight_norm(self.conv)

    @typecheck()
    def forward(self, inputs, input_len):
        out = self.conv(inputs)
        out = self.activation(out)
        out = mask_sequence_tensor(out, input_len)
        return out


class Conv2dNorm(NeuralModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int],
        stride: Tuple[int, int] = (1, 1),
        dilation: Tuple[int, int] = (1, 1),
    ):
        super().__init__()
        assert len(kernel_size) == len(dilation)
        padding = get_padding_2d(kernel_size, dilation)
        conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            padding=padding,
            padding_mode="reflect",
        )
        self.conv = nn.utils.parametrizations.weight_norm(conv)

    @property
    def input_types(self):
        return {
            "inputs": NeuralType(('B', 'C', 'H', 'T'), VoidType()),
        }

    @property
    def output_types(self):
        return {
            "out": NeuralType(('B', 'C', 'H', 'T'), VoidType()),
        }

    def remove_weight_norm(self):
        nn.utils.remove_weight_norm(self.conv)

    @typecheck()
    def forward(self, inputs):
        return self.conv(inputs)


class PeriodDiscriminator(NeuralModule):
    """
    Period discriminator introduced in HiFi-GAN https://arxiv.org/abs/2010.05646 which attempts to
    discriminate phase information by looking at equally spaced audio samples.

    Args:
        period: Spacing between audio sample inputs.
        lrelu_slope: Slope to use for activation. Leaky relu with slope of 0.1 or 0.2 is recommended for the
           stability of the feature matching loss.
    """

    def __init__(self, period, lrelu_slope=0.1):
        super().__init__()
        self.period = period
        self.activation = nn.LeakyReLU(lrelu_slope)
        self.conv_layers = nn.ModuleList(
            [
                Conv2dNorm(1, 32, kernel_size=(5, 1), stride=(3, 1)),
                Conv2dNorm(32, 128, kernel_size=(5, 1), stride=(3, 1)),
                Conv2dNorm(128, 512, kernel_size=(5, 1), stride=(3, 1)),
                Conv2dNorm(512, 1024, kernel_size=(5, 1), stride=(3, 1)),
                Conv2dNorm(1024, 1024, kernel_size=(5, 1), stride=(1, 1)),
            ]
        )
        self.conv_post = Conv2dNorm(1024, 1, kernel_size=(3, 1))

    @property
    def input_types(self):
        return {
            "audio": NeuralType(('B', 'T_audio'), AudioSignal()),
        }

    @property
    def output_types(self):
        return {
            "score": NeuralType(('B', 'C', 'T_out'), VoidType()),
            "fmap": [NeuralType(('B', 'D', 'T_layer', 'C'), VoidType())],
        }

    @typecheck()
    def forward(self, audio):

        batch_size, time = audio.shape
        out = rearrange(audio, 'B T -> B 1 T')
        # Pad audio so that it is divisible by the period
        if time % self.period != 0:
            n_pad = self.period - (time % self.period)
            out = F.pad(out, (0, n_pad), "reflect")
            time = time + n_pad
        # [batch, 1, (time / period), period]
        out = out.view(batch_size, 1, time // self.period, self.period)

        fmap = []
        for conv in self.conv_layers:
            # [batch, filters, (time / period / stride), period]
            out = conv(inputs=out)
            out = self.activation(out)
            fmap.append(out)
        # [batch, 1, (time / period / strides), period]
        score = self.conv_post(inputs=out)
        fmap.append(score)
        score = rearrange(score, "B 1 T C -> B C T")

        return score, fmap


class MultiPeriodDiscriminator(NeuralModule):
    """
    Wrapper class to aggregate results of multiple period discriminators.

    The periods are expected to be increasing prime numbers in order to maximize coverage and minimize overlap
    """

    def __init__(self, periods: Iterable[int] = (2, 3, 5, 7, 11), lrelu_slope=0.1):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [PeriodDiscriminator(period=period, lrelu_slope=lrelu_slope) for period in periods]
        )

    @property
    def input_types(self):
        return {
            "audio_real": NeuralType(('B', 'T_audio'), AudioSignal()),
            "audio_gen": NeuralType(('B', 'T_audio'), AudioSignal()),
        }

    @property
    def output_types(self):
        return {
            "scores_real": [NeuralType(('B', 'C', 'T_out'), VoidType())],
            "scores_gen": [NeuralType(('B', 'C', 'T_out'), VoidType())],
            "fmaps_real": [[NeuralType(('B', 'D', 'T_layer', 'C'), VoidType())]],
            "fmaps_gen": [[NeuralType(('B', 'D', 'T_layer', 'C'), VoidType())]],
        }

    @typecheck()
    def forward(self, audio_real, audio_gen):
        scores_real = []
        scores_gen = []
        fmaps_real = []
        fmaps_gen = []
        for discriminator in self.discriminators:
            score_real, fmap_real = discriminator(audio=audio_real)
            score_gen, fmap_gen = discriminator(audio=audio_gen)
            scores_real.append(score_real)
            fmaps_real.append(fmap_real)
            scores_gen.append(score_gen)
            fmaps_gen.append(fmap_gen)

        return scores_real, scores_gen, fmaps_real, fmaps_gen


class DiscriminatorSTFT(NeuralModule):
    """
    Discriminator network from EnCodec for Complex STFT input, but without dilations.

    Args:
        filters: number of filters to use in Conv2d layers
        lrelu_slope: Slope to use for activations. Leaky relu with slope of 0.1 or 0.2 is recommended for the
           stability of the feature matching loss
    """

    def __init__(self, filters: int = 32, lrelu_slope: float = 0.1):
        super().__init__()

        self.activation = nn.LeakyReLU(lrelu_slope)
        self.conv_layers = nn.ModuleList(
            [
                Conv2dNorm(2, filters, kernel_size=(3, 9)),
                Conv2dNorm(filters, filters, kernel_size=(3, 9), stride=(1, 2)),
                Conv2dNorm(filters, filters, kernel_size=(3, 9), stride=(1, 2)),
                Conv2dNorm(filters, filters, kernel_size=(3, 9), stride=(1, 2)),
                Conv2dNorm(filters, filters, kernel_size=(3, 3)),
            ]
        )
        self.conv_post = Conv2dNorm(filters, 1, kernel_size=(3, 3))

    @property
    def input_types(self):
        return {
            "spec": NeuralType(('B', 'C', 'T_spec', 'D'), VoidType()),
        }

    @property
    def output_types(self):
        return {
            "scores": NeuralType(('B', 'C', 'T_spec'), VoidType()),
            "fmap": [NeuralType(('B', 'D', 'T_spec', 'C'), VoidType())],
        }

    @typecheck()
    def forward(self, spec):
        fmap = []

        # [batch, 2, T_spec, fft]
        out = spec
        for conv in self.conv_layers:
            # [batch, filters, T_spec, fft // strides]
            out = conv(inputs=out)
            out = self.activation(out)
            fmap.append(out)
        # [batch, 1, T_spec, fft // 8]
        scores = self.conv_post(inputs=out)
        fmap.append(scores)
        scores = rearrange(scores, "B 1 T C -> B C T")

        return scores, fmap


class MultiBandDiscriminatorSTFT(NeuralModule):
    """
    Multi-band STFT discriminator proposed in DAC (https://arxiv.org/abs/2306.06546).

    Computes the complex STFT for a given resolution and splits it into sub-bands,
    which are given to separate discriminator networks.

    Args:
        resolution: STFT resolution, provided as a tuple of 3 integers ordered (num_fft, hop_length, window_length)
        stft_bands: List of tuples, with each tuple having 2 float values (band_start, band_end).
            The floats are in the range [0, 1] representing the fraction of all stft bands.
            For example for n_fft=1024, the stft output has 513 dimensions.
            For band input [(0, 0.25), (0.25, 1.0)] it would use stft dimensions [0 through 127] and [128 through 512].
    """

    def __init__(self, resolution: Tuple[int], stft_bands: Iterable[Tuple[int]]):
        super().__init__()

        self.n_fft, self.hop_length, self.win_length = resolution
        self.register_buffer("window", torch.hann_window(self.win_length, periodic=False))
        self.discriminators = nn.ModuleList([DiscriminatorSTFT() for _ in stft_bands])
        n_stft = self.n_fft // 2 + 1
        self.stft_bands = [(int(band[0] * n_stft), int(band[1] * n_stft)) for band in stft_bands]

    def compute_stft(self, audio):
        # [B, fft, T_spec]
        fft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            normalized=True,
            center=True,
            return_complex=True,
        )
        fft = rearrange(fft, "B fft T -> B T fft")
        # [batch, 2, T_spec, fft]
        out = torch.stack([fft.real, fft.imag], dim=1)
        return out

    @property
    def input_types(self):
        return {
            "audio": NeuralType(('B', 'T_audio'), AudioSignal()),
        }

    @property
    def output_types(self):
        return {
            "scores_list": [NeuralType(('B', 'C', 'T_spec'), VoidType())],
            "fmaps_list": [[NeuralType(('B', 'D', 'T_spec', 'C'), VoidType())]],
        }

    @typecheck()
    def forward(self, audio):
        scores_list = []
        fmap_list = []
        # run spec compute on fp32 and convert out to the model training type
        spec = self.compute_stft(audio.float()).to(audio.dtype)
        for band, disc in zip(self.stft_bands, self.discriminators):
            spec_band = spec[:, :, :, band[0] : band[1]]
            score, fmap = disc(spec=spec_band)
            scores_list.append(score)
            fmap_list.append(fmap)

        return scores_list, fmap_list


class MultiResolutionDiscriminatorSTFT(NeuralModule):
    """
    Multi-resolution discriminator which creates a multi-band discriminator for each input resolution.

    Args:
        resolutions: List of STFT resolutions, each resolution provided as a tuple of 3 integers ordered
            (num_fft, hop_length, window_length)
        stft_bands: List of tuples, with each tuple having 2 float values (band_start, band_end).
            The floats are in the range [0, 1] representing the fraction of all stft bands.
            For example for n_fft=1024, the stft output has 513 dimensions.
            For band input [(0, 0.25), (0.25, 1.0)] it would use stft dimensions [0 through 127] and [128 through 512].
    """

    def __init__(self, resolutions: Iterable[Tuple[int]], stft_bands: Iterable[Tuple[int]]):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [MultiBandDiscriminatorSTFT(resolution=resolution, stft_bands=stft_bands) for resolution in resolutions]
        )

    @property
    def input_types(self):
        return {
            "audio_real": NeuralType(('B', 'T_audio'), AudioSignal()),
            "audio_gen": NeuralType(('B', 'T_audio'), AudioSignal()),
        }

    @property
    def output_types(self):
        return {
            "scores_real": [NeuralType(('B', 'C', 'T_spec'), VoidType())],
            "scores_gen": [NeuralType(('B', 'C', 'T_spec'), VoidType())],
            "fmaps_real": [[NeuralType(('B', 'D', 'T_spec', 'C'), VoidType())]],
            "fmaps_gen": [[NeuralType(('B', 'D', 'T_spec', 'C'), VoidType())]],
        }

    @typecheck()
    def forward(self, audio_real, audio_gen):
        scores_real = []
        scores_gen = []
        fmaps_real = []
        fmaps_gen = []

        for disc in self.discriminators:
            score_real_i, fmap_real_i = disc(audio=audio_real)
            scores_real = scores_real + score_real_i
            fmaps_real = fmaps_real + fmap_real_i

            score_gen_i, fmap_gen_i = disc(audio=audio_gen)
            scores_gen = scores_gen + score_gen_i
            fmaps_gen = fmaps_gen + fmap_gen_i

        return scores_real, scores_gen, fmaps_real, fmaps_gen


class Discriminator(NeuralModule):
    """
    Wrapper class which takes a list of discriminators and aggregates the results across them.
    """

    def __init__(self, discriminators: Iterable[NeuralModule]):
        super().__init__()
        self.discriminators = nn.ModuleList(discriminators)

    @property
    def input_types(self):
        return {
            "audio_real": NeuralType(('B', 'T_audio'), AudioSignal()),
            "audio_gen": NeuralType(('B', 'T_audio'), AudioSignal()),
        }

    @property
    def output_types(self):
        return {
            "scores_real": [NeuralType(('B', 'C', 'T_out'), VoidType())],
            "scores_gen": [NeuralType(('B', 'C', 'T_out'), VoidType())],
            "fmaps_real": [[NeuralType(('B', 'D', 'T_layer', 'C'), VoidType())]],
            "fmaps_gen": [[NeuralType(('B', 'D', 'T_layer', 'C'), VoidType())]],
        }

    @typecheck()
    def forward(self, audio_real, audio_gen):
        scores_real = []
        scores_gen = []
        fmaps_real = []
        fmaps_gen = []
        for discriminator in self.discriminators:
            score_real, score_gen, fmap_real, fmap_gen = discriminator(audio_real=audio_real, audio_gen=audio_gen)
            scores_real += score_real
            fmaps_real += fmap_real
            scores_gen += score_gen
            fmaps_gen += fmap_gen

        return scores_real, scores_gen, fmaps_real, fmaps_gen


class VectorQuantizerBase(NeuralModule, ABC):
    @property
    @abstractmethod
    def num_codebooks(self) -> int:
        pass

    @property
    @abstractmethod
    def codebook_size(self) -> int:
        pass

    @property
    def input_types(self):
        return {
            "inputs": NeuralType(('B', 'D', 'T'), EncodedRepresentation()),
            "input_len": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        return {
            "dequantized": NeuralType(('B', 'D', 'T'), EncodedRepresentation()),
            "indices": NeuralType(('D', 'B', 'T'), TokenIndex()),
        }

    @typecheck()
    @abstractmethod
    def forward(self, inputs: torch.Tensor, input_len: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    @typecheck(
        input_types={
            "inputs": NeuralType(('B', 'D', 'T'), EncodedRepresentation()),
            "input_len": NeuralType(tuple('B'), LengthsType()),
        },
        output_types={"indices": NeuralType(('D', 'B', 'T'), TokenIndex())},
    )
    @abstractmethod
    def encode(self, inputs: torch.Tensor, input_len: torch.Tensor) -> torch.Tensor:
        pass

    @typecheck(
        input_types={
            "indices": NeuralType(('D', 'B', 'T'), TokenIndex()),
            "input_len": NeuralType(tuple('B'), LengthsType()),
        },
        output_types={
            "dequantized": NeuralType(('B', 'D', 'T'), EncodedRepresentation()),
        },
    )
    @abstractmethod
    def decode(self, indices: torch.Tensor, input_len: torch.Tensor) -> torch.Tensor:
        pass


class FiniteScalarQuantizer(VectorQuantizerBase):
    """This quantizer is based on the Finite Scalar Quantization (FSQ) method.
    It quantizes each element of the input vector independently into a number of levels.

    Args:
        num_levels: number of levels for each dimension/element of the input vector
        eps: small regularization constant for scaling

    References:
        Mentzer et al., Finite Scalar Quantization: VQ-VAE Made Simple (https://arxiv.org/abs/2309.15505v1)
    """

    def __init__(self, num_levels: List[int], eps: float = 1e-3):
        super().__init__()

        # index base per dimension of the input vector
        # this is used to convert between per-dimension indices and a codebook token index
        dim_base_index = torch.cumprod(torch.tensor([1] + num_levels[:-1]), dim=0, dtype=torch.int32)
        dim_base_index = rearrange(dim_base_index, 'D -> 1 D 1')
        self.register_buffer('dim_base_index', dim_base_index)

        # Register the number of levels for each dimension
        num_levels = torch.tensor(num_levels, dtype=torch.int32)
        num_levels = rearrange(num_levels, 'D -> 1 D 1')
        self.register_buffer('num_levels', num_levels)

        # Regularization
        self.eps = eps

        logging.debug('Initializing %s with', self.__class__.__name__)
        logging.debug('\tdim:           %s', self.dim)
        logging.debug('\tnum_levels:    %s', self.num_levels)
        logging.debug('\tcodebook_size: %s', self.codebook_size)
        logging.debug('\teps:           %s', self.eps)

    @property
    def num_codebooks(self):
        """Returns the number of codebooks."""
        return 1

    @property
    def codebook_size(self):
        """Returns the size of the corresponding codebook."""
        return self.num_levels.prod().item()

    @property
    def dim(self):
        """Returns the dimension of the input vector."""
        return self.num_levels.numel()

    @property
    def codebook_dim(self):
        """Returns the dimension of the input vector.
        Keeping for compatiblitiy with the original RVQ implementation.
        """
        return self.dim

    @property
    def codes(self):
        """Returns the codebooks entries.

        Note that the codebook entries are implicitly defined by the number of levels.
        """
        indices = torch.arange(self.codebook_size, device=self.dim_base_index.device)
        # [D, B, T]
        indices = rearrange(indices, 'B -> 1 B 1')
        # [B, D, T]
        codes = self.decode(indices=indices, input_len=None)
        # Remove the time dimension
        codes = codes.squeeze(-1)
        return codes

    @property
    def codebook(self):
        """Returns the codebooks entries.
        See self.codes for more details.
        """
        return self.codes

    @staticmethod
    def round(inputs: torch.Tensor, input_len: torch.Tensor) -> torch.Tensor:
        """Round the input tensor to nearest integer
        and use a straight-through estimator for the gradient.
        """
        inputs_rounded = torch.round(inputs)
        return inputs + (inputs_rounded - inputs).detach()

    def compress(self, inputs: torch.Tensor, input_len: torch.Tensor) -> torch.Tensor:
        """Apply compression to the input, to limit to values."""
        output_scale = (self.num_levels - 1) / 2
        # scale down a bit to avoid rounding issues
        output_scale = output_scale * (1 - self.eps)
        # offset for even number of levels
        output_offset = torch.where(self.num_levels % 2 == 0, 0.5, 0)
        # shift for even number of levels
        input_shift = (output_offset / output_scale).tan()
        # compressed output
        output = output_scale * (inputs + input_shift).tanh() - output_offset
        return output

    @typecheck(
        input_types={
            "inputs": NeuralType(('B', 'D', 'T'), EncodedRepresentation()),
            "input_len": NeuralType(tuple('B'), LengthsType()),
        },
        output_types={"codes": NeuralType(('B', 'D', 'T'), TokenIndex())},
    )
    def inputs_to_codes(self, inputs: torch.Tensor, input_len: torch.Tensor) -> torch.Tensor:
        # apply compression
        compressed = self.compress(inputs=inputs, input_len=input_len)
        # apply rounding to nearest integer
        codes = self.round(inputs=compressed, input_len=input_len)
        # normalize to [-1, 1]
        scale = self.num_levels // 2
        codes = codes / scale
        return codes

    def codes_to_nonnegative(self, codes: torch.Tensor) -> torch.Tensor:
        """Convert values centered arouund zero to nonnegative values."""
        scale = offset = self.num_levels // 2
        return scale * codes + offset

    def nonnegative_to_codes(self, codes_nonnegative: torch.Tensor) -> torch.Tensor:
        """Convert nonnegative values to values centered arouund zero."""
        scale = offset = self.num_levels // 2
        return (codes_nonnegative - offset) / scale

    def codes_to_indices(self, codes: torch.Tensor) -> torch.Tensor:
        """Converts a code vector to a single index."""
        if codes.size(1) != self.dim:
            raise RuntimeError(
                f'Input code dimension {codes.size(1)} not matching the expected dimension {self.dim}, input codes shape {codes.shape}'
            )
        # convert code vectors to nonnegative values
        indices = self.codes_to_nonnegative(codes)
        # convert one nonnegative index per dimension to a single index per code vector
        indices = torch.sum(indices * self.dim_base_index, dim=1)
        return indices.to(torch.int32)

    # Implementation of VectorQuantiserBase API
    @typecheck()
    def forward(
        self, inputs: torch.Tensor, input_len: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if inputs.size(1) != self.dim:
            raise RuntimeError(
                f'Input dimension {inputs.size(1)} not matching the expected dimension {self.dim}, inputs shape {inputs.shape}'
            )

        dequantized = self.inputs_to_codes(inputs=inputs, input_len=input_len)
        indices = self.codes_to_indices(codes=dequantized)

        if input_len is not None:
            # apply masking
            dequantized = mask_sequence_tensor(dequantized, input_len)
            indices = mask_sequence_tensor(indices, input_len)

        # only 1 codebook, but return in [D, B, T] format to match RVQ API
        indices = indices.unsqueeze(0)
        return dequantized, indices

    @typecheck(
        input_types={
            "inputs": NeuralType(('B', 'D', 'T'), EncodedRepresentation()),
            "input_len": NeuralType(tuple('B'), LengthsType(), optional=True),
        },
        output_types={"indices": NeuralType(('D', 'B', 'T'), TokenIndex())},
    )
    def encode(self, inputs: torch.Tensor, input_len: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Convert a continuous code vector to a single index."""
        _, indices = self(inputs=inputs, input_len=input_len)
        return indices

    @typecheck(
        input_types={
            "indices": NeuralType(('D', 'B', 'T'), TokenIndex()),
            "input_len": NeuralType(tuple('B'), LengthsType(), optional=True),
        },
        output_types={
            "dequantized": NeuralType(('B', 'D', 'T'), EncodedRepresentation()),
        },
    )
    def decode(self, indices: torch.Tensor, input_len: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Convert a single index to a continuous code vector."""
        if indices.size(0) > 1:
            # codebook dimension used for compatibility with RVQ
            raise ValueError(
                f'Expected a single codebook, got {indices.size(0)} codebooks for indices with shape {indices.shape}.'
            )

        indices = rearrange(indices, 'D B T -> B D T')
        # convert a single index to nonnegative index per-dimension
        codes_nonnegative = (indices // self.dim_base_index) % self.num_levels
        # convert nonnegative codes to codes (centered around zero)
        dequantized = self.nonnegative_to_codes(codes_nonnegative)

        if input_len is not None:
            # apply masking
            dequantized = mask_sequence_tensor(dequantized, input_len)
        return dequantized


class GroupFiniteScalarQuantizer(VectorQuantizerBase):
    """Split the input vector into groups and apply FSQ on each group separately.
    This class is for convenience. Since FSQ is applied on each group separately,
    groups can be defined arbitrarily by splitting the input vector. However, this
    class makes it easy to construct several groups with the same quantization num_levels.

    Args:
        num_groups: number of groups to split the input into, each group will be quantized separately using num_codebooks//num_groups codebooks
        codebook_dim: embedding dimension, will be split into num_groups
        **kwargs: parameters of FiniteScalarQuantizer

    References:
        Yang et al, HiFi-Codec: Group-residual Vector quantization for High Fidelity Audio Codec, 2023 (http://arxiv.org/abs/2305.02765).
    """

    def __init__(self, num_groups: int, num_levels_per_group: List[int], **kwargs):
        super().__init__()

        self.num_groups = num_groups
        self.codebook_dim_per_group = len(num_levels_per_group)

        # Initialize FSQ for each group
        self.fsqs = torch.nn.ModuleList(
            [FiniteScalarQuantizer(num_levels=num_levels_per_group, **kwargs) for _ in range(self.num_groups)]
        )

        logging.debug('Initialized %s with', self.__class__.__name__)
        logging.debug('\tnum_groups:              %d', self.num_groups)
        logging.debug('\tcodebook_dim:            %d', self.codebook_dim)
        logging.debug('\tnum_levels_per_group:    %s', num_levels_per_group)
        logging.debug('\tcodebook_dim_per_group:  %d', self.codebook_dim_per_group)

    @property
    def num_codebooks(self):
        """Returns the number of codebooks."""
        return self.num_groups

    @property
    def codebook_size(self):
        """Returns the size of the codebook for each group."""
        return self.fsqs[0].codebook_size

    @property
    def codebook_dim(self):
        """Input vector dimension."""
        return self.codebook_dim_per_group * self.num_groups

    @typecheck()
    def forward(self, inputs, input_len):
        """Quantize each group separately, then concatenate the results."""
        inputs_grouped = inputs.chunk(self.num_groups, dim=1)

        dequantized, indices = [], []

        for in_group, fsq_group in zip(inputs_grouped, self.fsqs):
            dequantized_group, indices_group = fsq_group(inputs=in_group, input_len=input_len)
            dequantized.append(dequantized_group)
            indices.append(indices_group)

        # concatenate along the feature dimension
        dequantized = torch.cat(dequantized, dim=1)

        # concatente along the codebook dimension
        indices = torch.cat(indices, dim=0)

        return dequantized, indices

    @typecheck(
        input_types={
            "inputs": NeuralType(('B', 'D', 'T'), EncodedRepresentation()),
            "input_len": NeuralType(tuple('B'), LengthsType()),
        },
        output_types={"indices": NeuralType(('D', 'B', 'T'), TokenIndex())},
    )
    def encode(self, inputs: torch.Tensor, input_len: torch.Tensor) -> torch.Tensor:
        """Input is split into groups, each group is encoded separately, then the results are concatenated."""
        inputs_grouped = inputs.chunk(self.num_groups, dim=1)
        indices = []

        for in_group, fsq_group in zip(inputs_grouped, self.fsqs):
            indices_group = fsq_group.encode(inputs=in_group, input_len=input_len)
            indices.append(indices_group)

        # concatenate along the codebook dimension
        indices = torch.cat(indices, dim=0)

        return indices

    @typecheck(
        input_types={
            "indices": NeuralType(('D', 'B', 'T'), TokenIndex()),
            "input_len": NeuralType(tuple('B'), LengthsType()),
        },
        output_types={
            "dequantized": NeuralType(('B', 'D', 'T'), EncodedRepresentation()),
        },
    )
    def decode(self, indices: torch.Tensor, input_len: torch.Tensor) -> torch.Tensor:
        """Input indices are split into groups, each group is decoded separately, then the results are concatenated."""
        indices_grouped = indices.chunk(self.num_groups, dim=0)
        dequantized = []

        for indices_group, fsq_group in zip(indices_grouped, self.fsqs):
            dequantized_group = fsq_group.decode(indices=indices_group, input_len=input_len)
            dequantized.append(dequantized_group)

        # concatenate along the feature dimension
        dequantized = torch.cat(dequantized, dim=1)

        return dequantized

    @typecheck(
        input_types={
            "codes": NeuralType(('B', 'D', 'T'), EncodedRepresentation()),
            "input_len": NeuralType(tuple('B'), LengthsType()),
        },
        output_types={
            "indices": NeuralType(('B', 'D', 'T'), TokenIndex()),
        },
    )
    def codes_to_indices(self, codes: torch.Tensor, input_len: torch.Tensor) -> torch.Tensor:
        """Converts a code vector to indices."""
        codes_rearrange = rearrange(codes, 'B D T -> D B T')
        codes_grouped = codes_rearrange.chunk(self.num_groups, dim=0)
        indices = []

        for codes_group, fsq_group in zip(codes_grouped, self.fsqs):
            codes_group_rearrange = rearrange(codes_group, 'D B T -> B D T')
            # [B, T]
            indices_group = fsq_group.codes_to_indices(codes=codes_group_rearrange)
            indices_group = mask_sequence_tensor(indices_group, input_len)
            indices.append(indices_group)

        # concatenate along the feature dimension
        indices = torch.stack(indices, dim=1)

        return indices


class ResidualBlock(NeuralModule):
    """
    The residual block structure defined by the HiFi-GAN V1 and V2 configurations.

    Args:
        channels: Input dimension.
        filters: Number of channels in the residual convolutions.
        kernel_size: Kernel size of the residual convolutions.
        dilation: Dilation of the residual convolutions.
        dropout_rate: Dropout to apply to residuals.
        activation: Activation to apply in between residual convolutions.
    """

    def __init__(
        self,
        channels: int,
        filters: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout_rate: float = 0.0,
        activation: str = "lrelu",
        is_causal: bool = False,
        pad_mode: str = "reflect",
    ):
        super(ResidualBlock, self).__init__()

        self.input_activation = CodecActivation(activation=activation, channels=channels)
        self.skip_activation = CodecActivation(activation=activation, channels=filters)
        self.dropout = torch.nn.Dropout(dropout_rate)
        if not is_causal:
            self.input_conv = Conv1dNorm(
                in_channels=channels,
                out_channels=filters,
                kernel_size=kernel_size,
                dilation=dilation,
                pad_mode=pad_mode,
            )
            self.skip_conv = Conv1dNorm(
                in_channels=filters, out_channels=channels, kernel_size=kernel_size, pad_mode=pad_mode
            )
        else:
            self.input_conv = CausalConv1dNorm(
                in_channels=channels,
                out_channels=filters,
                kernel_size=kernel_size,
                dilation=dilation,
                pad_mode=pad_mode,
            )
            self.skip_conv = CausalConv1dNorm(
                in_channels=filters, out_channels=channels, kernel_size=kernel_size, pad_mode=pad_mode
            )

    def remove_weight_norm(self):
        self.input_conv.remove_weight_norm()
        self.skip_conv.remove_weight_norm()

    @property
    def input_types(self):
        return {"inputs": NeuralType(('B', 'C', 'T'), VoidType()), "input_len": NeuralType(tuple('B'), LengthsType())}

    @property
    def output_types(self):
        return {"out": NeuralType(('B', 'C', 'T'), EncodedRepresentation())}

    @typecheck()
    def forward(self, inputs, input_len):
        conv_input = self.input_activation(inputs)
        skip_input = self.input_conv(inputs=conv_input, input_len=input_len)
        skip_input = self.skip_activation(skip_input)
        res = self.skip_conv(inputs=skip_input, input_len=input_len)
        res = self.dropout(res)
        out = inputs + res
        return out


class ResidualBlockV2(NeuralModule):
    """
    Residual block which applies activation to output instead of input.

    Args:
        channels: Input dimension.
        filters: Number of channels in the residual convolutions.
        kernel_size: Kernel size of the residual convolutions.
        activation: Activation to apply in between residual convolutions.
        is_causal:  Whether to use causal convolutions.
        pad_mode: Type of padding to use for conv1d layers.
            See https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
    """

    def __init__(
        self,
        channels: int,
        filters: int,
        kernel_size: int = 3,
        activation: str = "lrelu",
        is_causal: bool = False,
        pad_mode: str = "reflect",
    ):
        super(ResidualBlockV2, self).__init__()

        if not is_causal:
            self.input_conv = Conv1dNorm(
                in_channels=channels,
                out_channels=filters,
                kernel_size=kernel_size,
                activation=activation,
                pad_mode=pad_mode,
            )
            self.skip_conv = Conv1dNorm(
                in_channels=filters, out_channels=channels, kernel_size=kernel_size, pad_mode=pad_mode
            )
        else:
            self.input_conv = CausalConv1dNorm(
                in_channels=channels,
                out_channels=filters,
                kernel_size=kernel_size,
                activation=activation,
                pad_mode=pad_mode,
            )
            self.skip_conv = CausalConv1dNorm(
                in_channels=filters, out_channels=channels, kernel_size=kernel_size, pad_mode=pad_mode
            )

        self.output_activation = CodecActivation(activation=activation, channels=channels)

    def remove_weight_norm(self):
        self.input_conv.remove_weight_norm()
        self.skip_conv.remove_weight_norm()

    @property
    def input_types(self):
        return {"inputs": NeuralType(('B', 'C', 'T'), VoidType()), "input_len": NeuralType(tuple('B'), LengthsType())}

    @property
    def output_types(self):
        return {"out": NeuralType(('B', 'C', 'T'), EncodedRepresentation())}

    @typecheck()
    def forward(self, inputs, input_len):
        res = self.input_conv(inputs=inputs, input_len=input_len)
        res = self.skip_conv(inputs=res, input_len=input_len)
        out = inputs + res
        out = self.output_activation(out)
        out = mask_sequence_tensor(out, lengths=input_len)
        return out


class HiFiGANResBlock(NeuralModule):
    """
    Residual block wrapper for HiFi-GAN which creates a block for multiple dilations.

    Args:
        channels: Input dimension.
        kernel_size: Kernel size of the residual blocks.
        dilations: List of dilations. One residual block will be created for each dilation in the list.
        activation: Activation for the residual blocks.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilations: Iterable[int],
        activation: str,
        is_causal: bool = False,
        pad_mode: str = "reflect",
    ):
        super().__init__()

        self.res_blocks = nn.ModuleList(
            [
                ResidualBlock(
                    channels=channels,
                    filters=channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    activation=activation,
                    is_causal=is_causal,
                    pad_mode=pad_mode,
                )
                for dilation in dilations
            ]
        )

    def remove_weight_norm(self):
        for res_block in self.res_blocks:
            res_block.remove_weight_norm()

    @property
    def input_types(self):
        return {
            "inputs": NeuralType(('B', 'C', 'T'), VoidType()),
            "input_len": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        return {"out": NeuralType(('B', 'C', 'T'), VoidType())}

    @typecheck()
    def forward(self, inputs, input_len):
        out = inputs
        for res_block in self.res_blocks:
            out = res_block(inputs=out, input_len=input_len)
        return out


class HiFiGANResLayer(NeuralModule):
    """
    Residual block wrapper for HiFi-GAN which creates a block for multiple kernel sizes and dilations.
    One residual block is created for each combination of kernel size and dilation.

    Args:
        channels: Input dimension.
        kernel_sizes: List of kernel sizes.
        dilations: List of dilations.
        activation: Activation for the residual layers.

    """

    def __init__(
        self,
        channels: int,
        kernel_sizes: Iterable[int],
        dilations: Iterable[int],
        activation: str,
        is_causal: bool = False,
        pad_mode: str = "reflect",
    ):
        super().__init__()

        self.res_blocks = nn.ModuleList(
            [
                HiFiGANResBlock(
                    channels=channels,
                    kernel_size=kernel_size,
                    dilations=dilations,
                    activation=activation,
                    is_causal=is_causal,
                    pad_mode=pad_mode,
                )
                for kernel_size in kernel_sizes
            ]
        )

    def remove_weight_norm(self):
        for res_block in self.res_blocks:
            res_block.remove_weight_norm()

    @property
    def input_types(self):
        return {
            "inputs": NeuralType(('B', 'D', 'T'), VoidType()),
            "input_len": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        return {"out": NeuralType(('B', 'D', 'T'), VoidType())}

    @typecheck()
    def forward(self, inputs, input_len):
        residuals = [res_block(inputs=inputs, input_len=input_len) for res_block in self.res_blocks]
        out = sum(residuals) / len(residuals)
        return out


class CausalHiFiGANEncoder(NeuralModule):
    """
    Causal Audio encoder created by inverting the HiFi-GAN decoder and replacing Conv1D by CausalConv1D.

    Args:
        encoded_dim: Dimension of encoder output.
        down_sample_rates: Rate to upsample for each decoder block. The product of the downsample rates will
            determine the output token rate. For example 2 * 2 * 8 * 8 = 256 samples per token.
        base_channels: Number of filters in the first convolution. The number of channels will be doubled after each
            downsample layer.
        in_kernel_size: Kernel size of the input convolution.
        out_kernel_size: Kernel size of the output convolution.
        resblock_kernel_sizes: List of kernel sizes to use in each residual block.
        resblock_dilation_sizes: List of dilations to use in each residual block.
        activation: Activation to use in residual and downsample layers, defaults to leaky relu.
    """

    def __init__(
        self,
        encoded_dim: int,
        down_sample_rates: Iterable[int] = (2, 2, 8, 8),
        base_channels: int = 32,
        in_kernel_size: int = 7,
        out_kernel_size: int = 7,
        resblock_kernel_sizes: Iterable[int] = (3, 7, 11),
        resblock_dilation_sizes: Iterable[int] = (1, 3, 5),
        activation: str = "lrelu",
        pad_mode: str = "zeros",
    ):
        assert in_kernel_size > 0
        assert out_kernel_size > 0

        super().__init__()

        self.down_sample_rates = down_sample_rates
        self.pre_conv = CausalConv1dNorm(
            in_channels=1, out_channels=base_channels, kernel_size=in_kernel_size, pad_mode=pad_mode
        )

        in_channels = base_channels
        self.activations = nn.ModuleList([])
        self.down_sample_conv_layers = nn.ModuleList([])
        self.res_layers = nn.ModuleList([])
        for i, down_sample_rate in enumerate(self.down_sample_rates):
            res_layer = HiFiGANResLayer(
                channels=in_channels,
                kernel_sizes=resblock_kernel_sizes,
                dilations=resblock_dilation_sizes,
                activation=activation,
                is_causal=True,
                pad_mode=pad_mode,
            )
            self.res_layers.append(res_layer)

            act = CodecActivation(activation, channels=in_channels)
            self.activations.append(act)

            out_channels = 2 * in_channels
            kernel_size = 2 * down_sample_rate

            # padding = get_down_sample_padding(kernel_size=kernel_size, stride=down_sample_rate)
            down_sample_conv = CausalConv1dNorm(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=down_sample_rate,
                pad_mode=pad_mode,
            )
            in_channels = out_channels
            self.down_sample_conv_layers.append(down_sample_conv)

        self.post_activation = CodecActivation(activation, channels=in_channels)
        self.post_conv = CausalConv1dNorm(
            in_channels=in_channels, out_channels=encoded_dim, kernel_size=out_kernel_size, pad_mode=pad_mode
        )

    @property
    def input_types(self):
        return {
            "audio": NeuralType(('B', 'T_audio'), AudioSignal()),
            "audio_len": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        return {
            "encoded": NeuralType(('B', 'D', 'T_encoded'), EncodedRepresentation()),
            "encoded_len": NeuralType(tuple('B'), LengthsType()),
        }

    def remove_weight_norm(self):
        self.pre_conv.remove_weight_norm()
        self.post_conv.remove_weight_norm()
        for res_layer in self.res_layers:
            res_layer.remove_weight_norm()
        for down_sample_conv in self.down_sample_conv_layers:
            down_sample_conv.remove_weight_norm()

    @typecheck()
    def forward(self, audio, audio_len):
        encoded_len = audio_len
        audio = rearrange(audio, "B T -> B 1 T")
        # [B, C, T_audio]
        out = self.pre_conv(inputs=audio, input_len=encoded_len)
        for act, res_layer, down_sample_conv, down_sample_rate in zip(
            self.activations, self.res_layers, self.down_sample_conv_layers, self.down_sample_rates
        ):
            # [B, C, T]
            out = res_layer(inputs=out, input_len=encoded_len)
            out = act(out)

            with default_precision(torch.float32):
                encoded_len = (encoded_len // down_sample_rate).long()
            # [B, 2 * C, T / down_sample_rate]
            out = down_sample_conv(inputs=out, input_len=encoded_len)

        out = self.post_activation(out)
        # [B, encoded_dim, T_encoded]
        encoded = self.post_conv(inputs=out, input_len=encoded_len)
        return encoded, encoded_len


class HiFiGANEncoder(NeuralModule):
    """
    Audio encoder created by inverting the HiFi-GAN decoder.

    Args:
        encoded_dim: Dimension of encoder output.
        down_sample_rates: Rate to upsample for each decoder block. The product of the downsample rates will
            determine the output token rate. For example 2 * 2 * 8 * 8 = 256 samples per token.
        base_channels: Number of filters in the first convolution. The number of channels will be doubled after each
            downsample layer.
        in_kernel_size: Kernel size of the input convolution.
        out_kernel_size: Kernel size of the output convolution.
        resblock_kernel_sizes: List of kernel sizes to use in each residual block.
        resblock_dilation_sizes: List of dilations to use in each residual block.
        activation: Activation to use in residual and downsample layers, defaults to leaky relu.
    """

    def __init__(
        self,
        encoded_dim: int,
        down_sample_rates: Iterable[int] = (2, 2, 8, 8),
        base_channels: int = 32,
        in_kernel_size: int = 7,
        out_kernel_size: int = 7,
        resblock_kernel_sizes: Iterable[int] = (3, 7, 11),
        resblock_dilation_sizes: Iterable[int] = (1, 3, 5),
        activation: str = "lrelu",
        pad_mode: str = "reflect",
    ):
        assert in_kernel_size > 0
        assert out_kernel_size > 0

        super().__init__()

        self.down_sample_rates = down_sample_rates
        self.pre_conv = Conv1dNorm(
            in_channels=1, out_channels=base_channels, kernel_size=in_kernel_size, pad_mode=pad_mode
        )

        in_channels = base_channels
        self.activations = nn.ModuleList([])
        self.down_sample_conv_layers = nn.ModuleList([])
        self.res_layers = nn.ModuleList([])
        for i, down_sample_rate in enumerate(self.down_sample_rates):
            res_layer = HiFiGANResLayer(
                channels=in_channels,
                kernel_sizes=resblock_kernel_sizes,
                dilations=resblock_dilation_sizes,
                activation=activation,
                pad_mode=pad_mode,
            )
            self.res_layers.append(res_layer)

            act = CodecActivation(activation, channels=in_channels)
            self.activations.append(act)

            out_channels = 2 * in_channels
            kernel_size = 2 * down_sample_rate

            padding = get_down_sample_padding(kernel_size=kernel_size, stride=down_sample_rate)
            down_sample_conv = Conv1dNorm(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=down_sample_rate,
                padding=padding,
                pad_mode=pad_mode,
            )
            in_channels = out_channels
            self.down_sample_conv_layers.append(down_sample_conv)

        self.post_activation = CodecActivation(activation, channels=in_channels)
        self.post_conv = Conv1dNorm(
            in_channels=in_channels, out_channels=encoded_dim, kernel_size=out_kernel_size, pad_mode=pad_mode
        )

    @property
    def input_types(self):
        return {
            "audio": NeuralType(('B', 'T_audio'), AudioSignal()),
            "audio_len": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        return {
            "encoded": NeuralType(('B', 'D', 'T_encoded'), EncodedRepresentation()),
            "encoded_len": NeuralType(tuple('B'), LengthsType()),
        }

    def remove_weight_norm(self):
        self.pre_conv.remove_weight_norm()
        self.post_conv.remove_weight_norm()
        for res_layer in self.res_layers:
            res_layer.remove_weight_norm()
        for down_sample_conv in self.down_sample_conv_layers:
            down_sample_conv.remove_weight_norm()

    @typecheck()
    def forward(self, audio, audio_len):
        encoded_len = audio_len
        audio = rearrange(audio, "B T -> B 1 T")
        # [B, C, T_audio]
        out = self.pre_conv(inputs=audio, input_len=encoded_len)
        for act, res_layer, down_sample_conv, down_sample_rate in zip(
            self.activations, self.res_layers, self.down_sample_conv_layers, self.down_sample_rates
        ):
            # [B, C, T]
            out = res_layer(inputs=out, input_len=encoded_len)
            out = act(out)

            with default_precision(torch.float32):
                encoded_len = (encoded_len // down_sample_rate).long()
            # [B, 2 * C, T / down_sample_rate]
            out = down_sample_conv(inputs=out, input_len=encoded_len)

        out = self.post_activation(out)
        # [B, encoded_dim, T_encoded]
        encoded = self.post_conv(inputs=out, input_len=encoded_len)
        return encoded, encoded_len


class CausalHiFiGANDecoder(NeuralModule):
    """
    Codec decoder using the HiFi-GAN generator architecture with Causal Convolutions.

    Args:
        input_dim: Input dimension.
        up_sample_rates: Rate to upsample for each decoder block. The product of the upsample rates should be the same
            as the overall downsample rate for your encoder. For example, a symmetric encoder/decoder can be created
            with encoder downsample rates [2, 2, 8, 8] and decoder upsample rates [8, 8, 2, 2].
        base_channels: Number of filters in the first convolution. The number of channels will be cut in
            half after each upsample layer.
        in_kernel_size: Kernel size of the input convolution.
        out_kernel_size: Kernel size of the output convolution.
        resblock_kernel_sizes: List of kernel sizes to use in each residual block.
        resblock_dilation_sizes: List of dilations to use in each residual block.
        activation: Activation to use in residual and upsample layers, defaults to leaky relu.
        output_activation: Activation to apply to output. To produce a valid audio signal, it should output values in
         the range [-1.0, 1.0]. Supports "tanh" and "clamp".
    """

    def __init__(
        self,
        input_dim: int,
        up_sample_rates: Iterable[int] = (8, 8, 2, 2),
        base_channels: int = 512,
        in_kernel_size: int = 7,
        out_kernel_size: int = 3,
        resblock_kernel_sizes: Iterable[int] = (3, 7, 11),
        resblock_dilation_sizes: Iterable[int] = (1, 3, 5),
        activation: str = "lrelu",
        output_activation: str = "tanh",
        pad_mode: str = "zeros",
        n_groups_equal_to_out_channels: bool = True,
    ):
        assert in_kernel_size > 0
        assert out_kernel_size > 0

        super().__init__()

        self.up_sample_rates = up_sample_rates

        self.pre_conv = CausalConv1dNorm(
            in_channels=input_dim, out_channels=base_channels, kernel_size=in_kernel_size, pad_mode=pad_mode
        )

        in_channels = base_channels
        self.activations = nn.ModuleList([])
        self.up_sample_conv_layers = nn.ModuleList([])
        self.res_layers = nn.ModuleList([])
        for i, up_sample_rate in enumerate(self.up_sample_rates):
            out_channels = in_channels // 2
            kernel_size = 2 * up_sample_rate

            act = CodecActivation(activation, channels=in_channels)
            self.activations.append(act)

            up_sample_conv = CausalConvTranspose1dNorm(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=up_sample_rate,
                groups=out_channels if n_groups_equal_to_out_channels else 1,
            )
            in_channels = out_channels
            self.up_sample_conv_layers.append(up_sample_conv)

            res_layer = HiFiGANResLayer(
                channels=in_channels,
                kernel_sizes=resblock_kernel_sizes,
                dilations=resblock_dilation_sizes,
                activation=activation,
                is_causal=True,
                pad_mode=pad_mode,
            )
            self.res_layers.append(res_layer)

        self.post_activation = CodecActivation(activation, channels=in_channels)
        self.post_conv = CausalConv1dNorm(
            in_channels=in_channels, out_channels=1, kernel_size=out_kernel_size, pad_mode=pad_mode
        )
        if output_activation == "tanh":
            self.out_activation = nn.Tanh()
        elif output_activation == "clamp":
            self.out_activation = ClampActivation()
        else:
            raise ValueError(f"Invalid audio output activation {output_activation}")

    @property
    def input_types(self):
        return {
            "inputs": NeuralType(('B', 'D', 'T_encoded'), VoidType()),
            "input_len": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        return {
            "audio": NeuralType(('B', 'T_audio'), AudioSignal()),
            "audio_len": NeuralType(tuple('B'), LengthsType()),
        }

    def remove_weight_norm(self):
        self.pre_conv.remove_weight_norm()
        for up_sample_conv in self.up_sample_conv_layers:
            up_sample_conv.remove_weight_norm()
        for res_layer in self.res_layers:
            res_layer.remove_weight_norm()

    @typecheck()
    def forward(self, inputs, input_len):
        audio_len = input_len
        # [B, C, T_encoded]
        out = self.pre_conv(inputs=inputs, input_len=audio_len)
        for act, res_layer, up_sample_conv, up_sample_rate in zip(
            self.activations, self.res_layers, self.up_sample_conv_layers, self.up_sample_rates
        ):
            with default_precision(torch.float32):
                audio_len = (audio_len * up_sample_rate).long()
            out = act(out)
            # [B, C / 2, T * up_sample_rate]
            out = up_sample_conv(inputs=out, input_len=audio_len)
            out = res_layer(inputs=out, input_len=audio_len)

        out = self.post_activation(out)
        # [B, 1, T_audio]
        out = self.post_conv(inputs=out, input_len=audio_len)
        audio = self.out_activation(out)
        audio = rearrange(audio, "B 1 T -> B T")
        return audio, audio_len


class HiFiGANDecoder(NeuralModule):
    """
    Codec decoder using the HiFi-GAN generator architecture.

    Default parameters match the HiFi-GAN V1 configuration for 22.05khz.

    Args:
        input_dim: Input dimension.
        up_sample_rates: Rate to upsample for each decoder block. The product of the upsample rates should be the same
            as the overall downsample rate for your encoder. For example, a symmetric encoder/decoder can be created
            with encoder downsample rates [2, 2, 8, 8] and decoder upsample rates [8, 8, 2, 2].
        base_channels: Number of filters in the first convolution. The number of channels will be cut in
            half after each upsample layer.
        in_kernel_size: Kernel size of the input convolution.
        out_kernel_size: Kernel size of the output convolution.
        resblock_kernel_sizes: List of kernel sizes to use in each residual block.
        resblock_dilation_sizes: List of dilations to use in each residual block.
        activation: Activation to use in residual and upsample layers, defaults to leaky relu.
        output_activation: Activation to apply to output. To produce a valid audio signal, it should output values in
         the range [-1.0, 1.0]. Supports "tanh" and "clamp".
    """

    def __init__(
        self,
        input_dim: int,
        up_sample_rates: Iterable[int] = (8, 8, 2, 2),
        base_channels: int = 512,
        in_kernel_size: int = 7,
        out_kernel_size: int = 3,
        resblock_kernel_sizes: Iterable[int] = (3, 7, 11),
        resblock_dilation_sizes: Iterable[int] = (1, 3, 5),
        activation: str = "lrelu",
        output_activation: str = "tanh",
        pad_mode: str = "reflect",
        n_groups_equal_to_out_channels: bool = False,
    ):
        assert in_kernel_size > 0
        assert out_kernel_size > 0

        super().__init__()

        self.up_sample_rates = up_sample_rates

        self.pre_conv = Conv1dNorm(
            in_channels=input_dim, out_channels=base_channels, kernel_size=in_kernel_size, pad_mode=pad_mode
        )

        in_channels = base_channels
        self.activations = nn.ModuleList([])
        self.up_sample_conv_layers = nn.ModuleList([])
        self.res_layers = nn.ModuleList([])
        for i, up_sample_rate in enumerate(self.up_sample_rates):
            out_channels = in_channels // 2
            kernel_size = 2 * up_sample_rate

            act = CodecActivation(activation, channels=in_channels)
            self.activations.append(act)

            up_sample_conv = ConvTranspose1dNorm(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=up_sample_rate,
                groups=out_channels if n_groups_equal_to_out_channels else 1,
            )
            in_channels = out_channels
            self.up_sample_conv_layers.append(up_sample_conv)

            res_layer = HiFiGANResLayer(
                channels=in_channels,
                kernel_sizes=resblock_kernel_sizes,
                dilations=resblock_dilation_sizes,
                activation=activation,
                pad_mode=pad_mode,
            )
            self.res_layers.append(res_layer)

        self.post_activation = CodecActivation(activation, channels=in_channels)
        self.post_conv = Conv1dNorm(
            in_channels=in_channels, out_channels=1, kernel_size=out_kernel_size, pad_mode=pad_mode
        )
        if output_activation == "tanh":
            self.out_activation = nn.Tanh()
        elif output_activation == "clamp":
            self.out_activation = ClampActivation()
        else:
            raise ValueError(f"Invalid audio output activation {output_activation}")

    @property
    def input_types(self):
        return {
            "inputs": NeuralType(('B', 'D', 'T_encoded'), VoidType()),
            "input_len": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        return {
            "audio": NeuralType(('B', 'T_audio'), AudioSignal()),
            "audio_len": NeuralType(tuple('B'), LengthsType()),
        }

    def remove_weight_norm(self):
        self.pre_conv.remove_weight_norm()
        for up_sample_conv in self.up_sample_conv_layers:
            up_sample_conv.remove_weight_norm()
        for res_layer in self.res_layers:
            res_layer.remove_weight_norm()

    @typecheck()
    def forward(self, inputs, input_len):
        audio_len = input_len
        # [B, C, T_encoded]
        out = self.pre_conv(inputs=inputs, input_len=audio_len)
        for act, res_layer, up_sample_conv, up_sample_rate in zip(
            self.activations, self.res_layers, self.up_sample_conv_layers, self.up_sample_rates
        ):
            audio_len = audio_len * up_sample_rate
            out = act(out)
            # [B, C / 2, T * up_sample_rate]
            out = up_sample_conv(inputs=out, input_len=audio_len)
            out = res_layer(inputs=out, input_len=audio_len)

        out = self.post_activation(out)
        # [B, 1, T_audio]
        out = self.post_conv(inputs=out, input_len=audio_len)
        audio = self.out_activation(out)
        audio = rearrange(audio, "B 1 T -> B T")
        return audio, audio_len


class MelSpectrogramProcessor(NeuralModule):
    """
    Wrapper interface for computing mel spectrogram for codec training.
    """

    def __init__(self, sample_rate: int, win_length: int, hop_length: int, mel_dim: int = 80, log_guard: float = 1.0):
        super(MelSpectrogramProcessor, self).__init__()
        self.mel_dim = mel_dim
        self.hop_length = hop_length
        self.preprocessor = AudioToMelSpectrogramPreprocessor(
            sample_rate=sample_rate,
            highfreq=None,
            features=mel_dim,
            pad_to=1,
            exact_pad=True,
            n_window_size=win_length,
            n_window_stride=hop_length,
            window_size=False,
            window_stride=False,
            n_fft=win_length,
            mag_power=1.0,
            log=True,
            log_zero_guard_type="add",
            log_zero_guard_value=log_guard,
            mel_norm=None,
            normalize=None,
            preemph=None,
            dither=0.0,
        )

    @property
    def input_types(self):
        return {
            "audio": NeuralType(('B', 'T_audio'), AudioSignal()),
            "audio_len": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        return {
            "spec": NeuralType(('B', 'D', 'T_spec'), MelSpectrogramType()),
            "spec_len": NeuralType(tuple('B'), LengthsType()),
        }

    @typecheck()
    def forward(self, audio, audio_len):
        spec, spec_len = self.preprocessor(input_signal=audio, length=audio_len)
        return spec, spec_len


class STFTProcessor(NeuralModule):
    """
    Interface for computing log magnitude STFT features.

    Args:
        n_fft: Size of Fourier transform
        win_length: The size of the sliding window frames for windowing and STFT.
        hop_length: The distance between neighboring sliding window frames
        log_guard: Value to add to magnitude STFT before taking log.
    """

    def __init__(self, n_fft, win_length, hop_length, log_guard=1.0, pad_mode="reflect"):
        super(STFTProcessor, self).__init__()

        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.register_buffer("window", torch.hann_window(self.win_length, periodic=False))
        self.log_guard = log_guard
        self.stft_pad_amount = (self.n_fft - self.hop_length) // 2
        self.pad_mode = pad_mode

    @property
    def input_types(self):
        return {
            "audio": NeuralType(('B', 'T_audio'), AudioSignal()),
            "audio_len": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        return {
            "spec": NeuralType(('B', 'D', 'T_spec'), MelSpectrogramType()),
            "spec_len": NeuralType(tuple('B'), LengthsType()),
        }

    @typecheck()
    def forward(self, audio, audio_len):
        spec_len = audio_len // self.hop_length
        audio_padded = torch.nn.functional.pad(audio, (self.stft_pad_amount, self.stft_pad_amount), self.pad_mode)
        # [B, n_fft, T_spec]
        fft = torch.stft(
            audio_padded,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
            center=False,
        )
        fft_mag = torch.abs(fft)
        fft_mag_log = torch.log(fft_mag + self.log_guard)
        fft_mag_log = mask_sequence_tensor(fft_mag_log, spec_len)
        return fft_mag_log, spec_len


class ResNetEncoder(NeuralModule):
    """
    Residual network which uses HiFi-GAN residual blocks to encode spectrogram features without changing
    the time dimension.

    Args:
        in_channels: input dimension
        out_channels: output dimension
        num_layers: number of residual blocks to use
        hidden_channels: encoder hidden dimension
        filters: number of filters in residual block layers
        kernel_size: kernel size in residual block convolutions
        dropout_rate: Optional dropout rate to apply to residuals.
        activation: Activation to use, defaults to leaky relu.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 6,
        hidden_channels: int = 256,
        filters: int = 768,
        kernel_size: int = 3,
        dropout_rate: float = 0.1,
        activation: str = "lrelu",
        pad_mode: str = "reflect",
    ):
        super(ResNetEncoder, self).__init__()

        self.pre_conv = Conv1dNorm(
            in_channels=in_channels, out_channels=hidden_channels, kernel_size=kernel_size, pad_mode=pad_mode
        )
        self.res_layers = nn.ModuleList(
            [
                ResidualBlock(
                    channels=hidden_channels,
                    filters=filters,
                    kernel_size=kernel_size,
                    dropout_rate=dropout_rate,
                    activation=activation,
                    pad_mode=pad_mode,
                )
                for _ in range(num_layers)
            ]
        )
        self.post_activation = CodecActivation(activation, channels=hidden_channels)
        self.post_conv = Conv1dNorm(
            in_channels=hidden_channels, out_channels=out_channels, kernel_size=kernel_size, pad_mode=pad_mode
        )

    def remove_weight_norm(self):
        self.pre_conv.remove_weight_norm()
        self.post_conv.remove_weight_norm()
        for res_layer in self.res_layers:
            res_layer.remove_weight_norm()

    @property
    def input_types(self):
        return {
            "inputs": NeuralType(('B', 'D', 'T'), VoidType()),
            "input_len": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        return {"encoded": NeuralType(('B', 'C', 'T'), EncodedRepresentation())}

    @typecheck()
    def forward(self, inputs, input_len):
        encoded = self.pre_conv(inputs=inputs, input_len=input_len)
        for res_layer in self.res_layers:
            encoded = res_layer(inputs=encoded, input_len=input_len)
        encoded = self.post_activation(encoded)
        encoded = self.post_conv(inputs=encoded, input_len=input_len)
        return encoded


class FullBandMelEncoder(NeuralModule):
    """
    Encoder which encodes the entire mel spectrogram with a single encoder network.

    Args:
        mel_processor: MelSpectrogramProcessor or equivalent class instance for computing the mel spectrogram from
            input audio.
        encoder: ResNetEncoder or equivalent class for encoding the mel spectrogram.
    """

    def __init__(self, mel_processor: NeuralModule, encoder: NeuralModule):
        super(FullBandMelEncoder, self).__init__()
        self.mel_processor = mel_processor
        self.encoder = encoder

    def remove_weight_norm(self):
        self.encoder.remove_weight_norm()

    @property
    def input_types(self):
        return {
            "audio": NeuralType(('B', 'T_audio'), AudioSignal()),
            "audio_len": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        return {
            "encoded": NeuralType(('B', 'C', 'T_encoded'), EncodedRepresentation()),
            "encoded_len": NeuralType(tuple('B'), LengthsType()),
        }

    @typecheck()
    def forward(self, audio, audio_len):
        out, spec_len = self.mel_processor(audio=audio, audio_len=audio_len)
        encoded = self.encoder(inputs=out, input_len=spec_len)
        return encoded, spec_len


class MultiBandMelEncoder(NeuralModule):
    """
    Encoder which splits mel spectrogram into bands and encodes each using separate residual networks.

    Args:
        mel_bands: List of mel spectrogram bands to encode.
            Each list element is tuple of 2 elements with the start and end index of the mel features to use.
        mel_processor: MelSpectrogramProcessor or equivalent class instance for computing the mel spectrogram from
            input audio.
        encoder_kwargs: Arguments for constructing encoder for each mel band.
    """

    def __init__(self, mel_bands: Iterable[Tuple[int, int]], mel_processor: NeuralModule, **encoder_kwargs):
        super(MultiBandMelEncoder, self).__init__()
        self.validate_mel_bands(mel_dim=mel_processor.mel_dim, mel_bands=mel_bands)
        self.mel_bands = mel_bands
        self.mel_processor = mel_processor
        band_dims = [band[1] - band[0] for band in self.mel_bands]
        self.encoders = nn.ModuleList(
            [ResNetEncoder(in_channels=band_dim, **encoder_kwargs) for band_dim in band_dims]
        )

    @staticmethod
    def validate_mel_bands(mel_dim: int, mel_bands: Iterable[Tuple[int, int]]):
        mel_dims_used = np.zeros([mel_dim], dtype=bool)
        for band in mel_bands:
            mel_dims_used[band[0] : band[1]] = True

        if not all(mel_dims_used):
            missing_dims = np.where(~mel_dims_used)
            raise ValueError(f"Mel bands must cover all {mel_dim} dimensions. Missing {missing_dims}.")

        return

    def remove_weight_norm(self):
        for encoder in self.encoders:
            encoder.remove_weight_norm()

    @property
    def input_types(self):
        return {
            "audio": NeuralType(('B', 'T_audio'), AudioSignal()),
            "audio_len": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        return {
            "encoded": NeuralType(('B', 'C', 'T_encoded'), EncodedRepresentation()),
            "encoded_len": NeuralType(tuple('B'), LengthsType()),
        }

    @typecheck()
    def forward(self, audio, audio_len):
        spec, spec_len = self.mel_processor(audio=audio, audio_len=audio_len)
        outputs = []
        for (band_start, band_end), encoder in zip(self.mel_bands, self.encoders):
            # [B, D_band, T]
            spec_band = spec[:, band_start:band_end, :]
            band_out = encoder(inputs=spec_band, input_len=spec_len)
            outputs.append(band_out)
        # [B, C, T]
        encoded = torch.cat(outputs, dim=1)
        return encoded, spec_len


class STFTResidualBlock(NeuralModule):
    """
    Block in multi-resolution STFT encoder which adds an STFT resolution to the encoder latent space, after down
    sampling the input to match the time resoluton of the STFT features.

    Args:
        resolution: STFT resolution, formatted as a 3-tuple (n_fft, hop_length, window_size)
        input_dim: Dimension if input latenct features.
        filters: Number of channels in the residual convolutions.
        kernel_size: Kernel size of the residual convolutions.
        activation: Name of activation function.
        down_sample_rate: Down sample factor to reduce input by before adding STFT encoding.
    """

    def __init__(
        self,
        resolution: Tuple[int],
        input_dim: int,
        filters: int,
        kernel_size: int,
        activation: str,
        down_sample_rate: int,
        pad_mode: str,
    ):
        super(STFTResidualBlock, self).__init__()
        down_sample_kernel_size = down_sample_rate * 2 + 1

        self.down_sample_rate = down_sample_rate
        self.down_sample_conv = Conv1dNorm(
            in_channels=input_dim,
            out_channels=filters,
            kernel_size=down_sample_kernel_size,
            stride=self.down_sample_rate,
            activation=activation,
            pad_mode=pad_mode,
        )

        n_fft, hop_length, win_length = resolution
        stft_dim = n_fft // 2 + 1
        self.spec_processor = STFTProcessor(
            n_fft=n_fft, win_length=win_length, hop_length=hop_length, pad_mode=pad_mode
        )
        self.spec_conv = Conv1dNorm(
            in_channels=stft_dim, out_channels=filters, kernel_size=kernel_size, pad_mode=pad_mode
        )
        self.spec_act = CodecActivation(activation=activation, channels=filters)

        self.res_block = ResidualBlockV2(
            channels=filters, filters=filters, kernel_size=kernel_size, activation=activation, pad_mode=pad_mode
        )

    def remove_weight_norm(self):
        self.input_conv.remove_weight_norm()
        self.skip_conv.remove_weight_norm()

    @property
    def input_types(self):
        return {
            "inputs": NeuralType(('B', 'C', 'T'), VoidType()),
            "input_len": NeuralType(tuple('B'), LengthsType()),
            "audio": NeuralType(('B', 'T_audio'), AudioSignal()),
            "audio_len": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        return {
            "out": NeuralType(('B', 'C', 'T'), EncodedRepresentation()),
            "out_len": NeuralType(tuple('B'), LengthsType()),
        }

    @typecheck()
    def forward(self, inputs, input_len, audio, audio_len):
        out_len = input_len // self.down_sample_rate
        out = self.down_sample_conv(inputs=inputs, input_len=out_len)

        spec, _ = self.spec_processor(audio=audio, audio_len=audio_len)
        spec_res = self.spec_conv(inputs=spec, input_len=out_len)
        out = out + spec_res
        out = self.spec_act(out)

        out = self.res_block(inputs=out, input_len=out_len)
        return out, out_len


class DownSampleResidualBlock(NeuralModule):
    """
    Layer which combines a down sampling layer with a residual block.

    Args:
        channels: Input dimension.
        filters: Number of channels in the residual convolutions.
        kernel_size: Kernel size of the residual convolutions.
        activation: Activation to apply in between residual convolutions.
        down_sample_rate: Factor to down sample time dimension by.
    """

    def __init__(
        self,
        channels: int,
        filters: int,
        kernel_size: int,
        activation: str,
        down_sample_rate: int,
        pad_mode: str,
    ):
        super(DownSampleResidualBlock, self).__init__()
        down_sample_kernel_size = down_sample_rate * 2 + 1

        self.down_sample_rate = down_sample_rate
        self.down_sample_conv = Conv1dNorm(
            in_channels=channels,
            out_channels=filters,
            kernel_size=down_sample_kernel_size,
            stride=self.down_sample_rate,
            activation=activation,
            pad_mode=pad_mode,
        )
        self.res_block = ResidualBlockV2(
            channels=filters, filters=filters, kernel_size=kernel_size, activation=activation, pad_mode=pad_mode
        )

    def remove_weight_norm(self):
        self.input_conv.remove_weight_norm()
        self.skip_conv.remove_weight_norm()

    @property
    def input_types(self):
        return {"inputs": NeuralType(('B', 'C', 'T'), VoidType()), "input_len": NeuralType(tuple('B'), LengthsType())}

    @property
    def output_types(self):
        return {
            "out": NeuralType(('B', 'C', 'T'), EncodedRepresentation()),
            "out_len": NeuralType(tuple('B'), LengthsType()),
        }

    @typecheck()
    def forward(self, inputs, input_len):
        output_len = input_len // self.down_sample_rate
        out = self.down_sample_conv(inputs=inputs, input_len=output_len)
        out = self.res_block(inputs=out, input_len=output_len)
        return out, output_len


class MultiResolutionSTFTEncoder(NeuralModule):
    """
    Encoder which computes log magnitude STFT features at several time resolutions and encodes them into a low
    frame-rate representation.

    Args:
        out_dim: Dimension of encoder output embedding.
        resolutions: List of STFT resolutions, formatted as 3-tuples (n_fft, hop_length, window_size)
        resolution_filter_list: List the same size as 'resolutions', specifying the number of filters in the residual
            block for each STFT resolution.
        down_sample_filter_list: List of filters to use for each down sampling block after initial STFT encoding.
        down_sample_rate_list: List of rates to use for each down sampling block after initial STFT encoding.
            The total down sample rate of the encoder will be 2**(len(resolutions)) * product(down_sample_rate_list)
        kernel_size: Kernel size to use in all convolutions.
        activation: Name of activation function.
        pad_mode: Type of padding to use for conv1d layers.
            See https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
    """

    def __init__(
        self,
        out_dim: int,
        resolutions: List[Tuple[int]],
        resolution_filter_list: List[int],
        down_sample_filter_list: Tuple[int] = (),
        down_sample_rate_list: Tuple[int] = (),
        kernel_size: int = 3,
        activation: str = "lrelu",
        pad_mode: str = "replicate",
    ):
        super(MultiResolutionSTFTEncoder, self).__init__()
        assert len(resolutions) >= 1
        assert len(resolutions) == len(resolution_filter_list)

        n_fft, hop_length, win_length = resolutions[0]
        input_filters = resolution_filter_list[0]
        input_dim = n_fft // 2 + 1
        self.pre_spec_processor = STFTProcessor(
            n_fft=n_fft, win_length=win_length, hop_length=hop_length, pad_mode=pad_mode
        )
        self.pre_conv = Conv1dNorm(
            in_channels=input_dim,
            out_channels=input_filters,
            kernel_size=kernel_size,
            activation=activation,
            pad_mode=pad_mode,
        )
        self.pre_res_block = ResidualBlockV2(
            channels=input_filters,
            filters=input_filters,
            kernel_size=kernel_size,
            activation=activation,
            pad_mode=pad_mode,
        )
        input_dim = input_filters
        self.stft_blocks = nn.ModuleList([])
        for resolution, filters in zip(resolutions[1:], resolution_filter_list[1:]):
            stft_block = STFTResidualBlock(
                resolution=resolution,
                input_dim=input_dim,
                down_sample_rate=2,
                filters=filters,
                kernel_size=kernel_size,
                activation=activation,
                pad_mode=pad_mode,
            )
            self.stft_blocks.append(stft_block)
            input_dim = filters

        if down_sample_filter_list and not down_sample_rate_list:
            down_sample_rate_list = len(down_sample_filter_list) * [2]

        self.down_sample_blocks = nn.ModuleList([])
        for filters, down_sample_rate in zip(down_sample_filter_list, down_sample_rate_list):
            down_sample_block = DownSampleResidualBlock(
                channels=input_dim,
                filters=filters,
                down_sample_rate=down_sample_rate,
                kernel_size=kernel_size,
                activation=activation,
                pad_mode=pad_mode,
            )
            self.down_sample_blocks.append(down_sample_block)
            input_dim = filters

        self.post_conv = Conv1dNorm(
            in_channels=input_dim,
            out_channels=out_dim,
            kernel_size=kernel_size,
            pad_mode=pad_mode,
        )

    def remove_weight_norm(self):
        self.encoder.remove_weight_norm()

    @property
    def input_types(self):
        return {
            "audio": NeuralType(('B', 'T_audio'), AudioSignal()),
            "audio_len": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        return {
            "encoded": NeuralType(('B', 'D', 'T_encoded'), EncodedRepresentation()),
            "encoded_len": NeuralType(tuple('B'), LengthsType()),
        }

    @typecheck()
    def forward(self, audio, audio_len):
        encoded, encoded_len = self.pre_spec_processor(audio=audio, audio_len=audio_len)
        encoded = self.pre_conv(inputs=encoded, input_len=encoded_len)
        encoded = self.pre_res_block(inputs=encoded, input_len=encoded_len)

        for stft_block in self.stft_blocks:
            encoded, encoded_len = stft_block(inputs=encoded, input_len=encoded_len, audio=audio, audio_len=audio_len)

        for down_sample_block in self.down_sample_blocks:
            encoded, encoded_len = down_sample_block(inputs=encoded, input_len=encoded_len)

        encoded = self.post_conv(inputs=encoded, input_len=encoded_len)

        return encoded, encoded_len


class VectorQuantizerIndexConverter(NeuralModule):
    """
    Utility for converting indices between two FSQ definitions.

    Example:

        from nemo.collections.tts.models import AudioCodecModel
        from nemo.collections.tts.modules.audio_codec_modules import GroupFiniteScalarQuantizer, VectorQuantizerIndexConverter

        audio_file = "/home/audio.wav"
        codec_path = "/home/SpectralCodecFps43.nemo"

        device = "cuda:0"

        audio, _ = librosa.load(audio_file, sr=sample_rate)

        audio_tensor = torch.tensor([audio]).to(device)
        audio_len_tensor = torch.tensor([audio.shape[0]]).to(device)

        codec_model = AudioCodecModel.restore_from(codec_path, map_location=device)
        tokens, token_len = codec_model.encode(audio=audio_tensor, audio_len=audio_len_tensor)

        fsq_new = GroupFiniteScalarQuantizer(num_groups=6, num_levels_per_group=[5, 5, 5, 5]).to(device)

        # vector_quantizer_original has 4 codebooks with 6 levels [5, 5, 5, 5, 5, 5]
        # vector_quantizer_new has 6 codebooks with 4 levels [5, 5, 5, 5]
        fsq_converter = VectorQuantizerIndexConverter(
            vector_quantizer_original=codec_model.vector_quantizer,
            vector_quantizer_new=fsq_new
        )

        tokens_new = fsq_converter.convert_original_to_new(audio_tokens=tokens, audio_lens=token_len)
        tokens_original = fsq_converter.convert_new_to_original(audio_tokens=tokens_new, audio_lens=token_len)

    """

    def __init__(self, vector_quantizer_original, vector_quantizer_new):
        super().__init__()
        self.vector_quantizer_original = vector_quantizer_original
        self.vector_quantizer_new = vector_quantizer_new

    # Input [batch, num_codebooks_original, time]
    # Output [batch, num_codebooks_new, time]
    def convert_original_to_new(self, audio_tokens, audio_lens):
        audio_tokens_rearrange = rearrange(audio_tokens, 'B C T -> C B T')
        audio_codes = self.vector_quantizer_original.decode(indices=audio_tokens_rearrange, input_len=audio_lens)
        audio_tokens_new = self.vector_quantizer_new.codes_to_indices(codes=audio_codes, input_len=audio_lens)
        return audio_tokens_new

    # Input [batch, num_codebooks_new, time]
    # Output [batch, num_codebooks_original, time]
    def convert_new_to_original(self, audio_tokens, audio_lens):
        audio_tokens_rearrange = rearrange(audio_tokens, 'B C T -> C B T')
        audio_codes = self.vector_quantizer_new.decode(indices=audio_tokens_rearrange, input_len=audio_lens)
        audio_tokens_original = self.vector_quantizer_original.codes_to_indices(
            codes=audio_codes, input_len=audio_lens
        )
        return audio_tokens_original


class ResNetDecoder(NeuralModule):
    """
    A residual decoder designed for low-latency. Most processing is done at a low frame-rate (e.g. 50 FPS), while
    minimizing the size of the network which upsamples to the final waveform.

    Args:
        input_dim: Dimension of decoder input.
        input_filters: Size of the first CNN layer applied to the decoder input.
        pre_up_sample_rates: Up sample rates to apply prior to main decoder network.
        pre_up_sample_filters: Size of residual blocks in first up sampling blocks.
        n_hidden_layers: Number of residual blocks in the main decoder network, which processes the latent space at
            low frame-rate.
        hidden_filters: Size of each rsidual block in the main decoder network.
        resblock_up_sample_rates: Up sample rates to apply after main decoder network.
        resblock_up_sample_filters: Size of residual blocks in final up sampling blocks.
        resblock_up_sample_kernel_size: Kernel size to use in final up sampling blocks.
        kernel_size: Kernel size to use in all other CNN layers.
        activation: Name of activation to use in residual blocks.
        is_causal: Whether to make the decoder causal.
        pad_mode: Type of padding to use for conv1d layers.
            See https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
    """

    def __init__(
        self,
        input_dim: int,
        input_filters: int,
        pre_up_sample_rates: List[int],
        pre_up_sample_filters: List[int],
        n_hidden_layers: int,
        hidden_filters: int,
        resblock_up_sample_rates: List[int],
        resblock_up_sample_filters: List[int],
        resblock_up_sample_kernel_size: int = 7,
        kernel_size: int = 3,
        activation: str = "half_snake",
        is_causal: bool = False,
        pad_mode: str = "replicate",
    ):
        super().__init__()

        assert len(pre_up_sample_rates) == len(pre_up_sample_filters)
        assert len(resblock_up_sample_rates) == len(resblock_up_sample_filters)

        if not is_causal:
            conv_class = Conv1dNorm
        else:
            conv_class = CausalConv1dNorm

        if not is_causal:
            conv_transpose_class = ConvTranspose1dNorm
        else:
            conv_transpose_class = CausalConvTranspose1dNorm

        self.pre_conv = conv_class(
            in_channels=input_dim,
            out_channels=input_filters,
            kernel_size=kernel_size,
        )

        in_channels = input_filters
        self.pre_up_sample_rates = pre_up_sample_rates
        self.pre_resblocks = nn.ModuleList([])
        self.pre_up_sample_layers = nn.ModuleList([])
        for up_sample_rate, filters in zip(self.pre_up_sample_rates, pre_up_sample_filters):
            res_block = ResidualBlockV2(
                channels=in_channels,
                filters=(2 * in_channels),
                kernel_size=kernel_size,
                activation=activation,
                is_causal=is_causal,
                pad_mode=pad_mode,
            )
            self.pre_resblocks.append(res_block)
            conv = conv_transpose_class(
                in_channels=in_channels,
                out_channels=filters,
                kernel_size=(2 * up_sample_rate),
                stride=up_sample_rate,
                activation=activation,
            )
            self.pre_up_sample_layers.append(conv)

            in_channels = filters

        self.conv_layers = nn.ModuleList(
            [
                ResidualBlockV2(
                    channels=in_channels,
                    filters=hidden_filters,
                    kernel_size=kernel_size,
                    activation=activation,
                    is_causal=is_causal,
                    pad_mode=pad_mode,
                )
                for _ in range(n_hidden_layers)
            ]
        )

        self.resblock_up_sample_rates = resblock_up_sample_rates
        self.resblock_up_sample_layers = nn.ModuleList([])
        self.resblocks = nn.ModuleList([])
        for up_sample_rate, filters in zip(self.resblock_up_sample_rates, resblock_up_sample_filters):
            conv = conv_transpose_class(
                in_channels=in_channels,
                out_channels=filters,
                kernel_size=(2 * up_sample_rate),
                stride=up_sample_rate,
                activation=activation,
            )
            self.resblock_up_sample_layers.append(conv)
            res_block = ResidualBlockV2(
                channels=filters,
                filters=(2 * filters),
                kernel_size=resblock_up_sample_kernel_size,
                activation=activation,
                is_causal=is_causal,
                pad_mode=pad_mode,
            )
            self.resblocks.append(res_block)
            in_channels = filters

        self.post_conv = conv_class(
            in_channels=in_channels, out_channels=1, kernel_size=resblock_up_sample_kernel_size, pad_mode=pad_mode
        )

        self.out_activation = ClampActivation(clamp_training=False)

    @property
    def input_types(self):
        return {
            "inputs": NeuralType(('B', 'D', 'T_encoded'), VoidType()),
            "input_len": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        return {
            "audio": NeuralType(('B', 'T_audio'), AudioSignal()),
            "audio_len": NeuralType(tuple('B'), LengthsType()),
        }

    @typecheck()
    def forward(self, inputs, input_len):

        out = self.pre_conv(inputs=inputs, input_len=input_len)

        audio_len = input_len
        for pre_up_sample_rate, pre_up_sample_layer, pre_resblock in zip(
            self.pre_up_sample_rates, self.pre_up_sample_layers, self.pre_resblocks
        ):
            out = pre_resblock(inputs=out, input_len=audio_len)
            audio_len = pre_up_sample_rate * audio_len
            out = pre_up_sample_layer(inputs=out, input_len=audio_len)

        for conv in self.conv_layers:
            out = conv(inputs=out, input_len=audio_len)

        for resblock_up_sample_rate, resblock_up_sample_layer, resblock in zip(
            self.resblock_up_sample_rates, self.resblock_up_sample_layers, self.resblocks
        ):
            audio_len = resblock_up_sample_rate * audio_len
            out = resblock_up_sample_layer(inputs=out, input_len=audio_len)
            out = resblock(inputs=out, input_len=audio_len)

        out = self.post_conv(inputs=out, input_len=audio_len)
        out = rearrange(out, 'B 1 T -> B T')
        audio = self.out_activation(out)
        audio = mask_sequence_tensor(audio, audio_len)

        return audio, audio_len
