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

import numpy as np
import torch
from torch import Tensor, nn

from nemo.collections.diffusion.vae.blocks import Downsample, Normalize, ResnetBlock, Upsample, make_attn


@dataclass
class AutoEncoderParams:
    """Dataclass for storing autoencoder hyperparameters.

    Attributes
    ----------
    ch_mult : list[int]
        Channel multipliers at each resolution level.
    attn_resolutions : list[int]
        List of resolutions at which attention layers are applied.
    resolution : int, optional
        Input image resolution. Default is 256.
    in_channels : int, optional
        Number of input channels. Default is 3.
    ch : int, optional
        Base channel dimension. Default is 128.
    out_ch : int, optional
        Number of output channels. Default is 3.
    num_res_blocks : int, optional
        Number of residual blocks at each resolution. Default is 2.
    z_channels : int, optional
        Number of latent channels in the compressed representation. Default is 16.
    scale_factor : float, optional
        Scaling factor for latent representations. Default is 0.3611.
    shift_factor : float, optional
        Shift factor for latent representations. Default is 0.1159.
    attn_type : str, optional
        Type of attention to use ('vanilla', 'linear'). Default is 'vanilla'.
    double_z : bool, optional
        If True, produce both mean and log-variance for latent space. Default is True.
    dropout : float, optional
        Dropout rate. Default is 0.0.
    ckpt : str or None, optional
        Path to checkpoint file for loading pretrained weights. Default is None.
    """

    ch_mult: list[int]
    attn_resolutions: list[int]
    resolution: int = 256
    in_channels: int = 3
    ch: int = 128
    out_ch: int = 3
    num_res_blocks: int = 2
    z_channels: int = 16
    scale_factor: float = 0.3611
    shift_factor: float = 0.1159
    attn_type: str = 'vanilla'
    double_z: bool = True
    dropout: float = 0.0
    ckpt: str = None


def nonlinearity(x: Tensor) -> Tensor:
    """Applies the SiLU (Swish) nonlinearity.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        Transformed tensor after applying SiLU activation.
    """
    return torch.nn.functional.silu(x)


class Encoder(nn.Module):
    """Encoder module that downsamples and encodes input images into a latent representation.

    Parameters
    ----------
    ch : int
        Base channel dimension.
    out_ch : int
        Number of output channels.
    ch_mult : list[int]
        Channel multipliers at each resolution level.
    num_res_blocks : int
        Number of residual blocks at each resolution level.
    attn_resolutions : list[int]
        List of resolutions at which attention layers are applied.
    in_channels : int
        Number of input image channels.
    resolution : int
        Input image resolution.
    z_channels : int
        Number of latent channels.
    dropout : float, optional
        Dropout rate. Default is 0.0.
    resamp_with_conv : bool, optional
        Whether to use convolutional resampling. Default is True.
    double_z : bool, optional
        If True, produce mean and log-variance channels for latent space. Default is True.
    use_linear_attn : bool, optional
        If True, use linear attention. Default is False.
    attn_type : str, optional
        Type of attention to use ('vanilla', 'linear'). Default is 'vanilla'.
    """

    def __init__(
        self,
        *,
        ch: int,
        out_ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        attn_resolutions: list[int],
        in_channels: int,
        resolution: int,
        z_channels: int,
        dropout=0.0,
        resamp_with_conv=True,
        double_z=True,
        use_linear_attn=False,
        attn_type="vanilla",
    ):
        super().__init__()
        if use_linear_attn:
            attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(
                    ResnetBlock(
                        in_channels=block_in, out_channels=block_out, temb_channels=self.temb_ch, dropout=dropout
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout
        )
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout
        )

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(
            block_in, 2 * z_channels if double_z else z_channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the Encoder.

        Parameters
        ----------
        x : torch.Tensor
            Input image tensor of shape (B, C, H, W).

        Returns
        -------
        torch.Tensor
            Latent representation before sampling, with shape (B, 2*z_channels, H', W') if double_z=True.
        """
        # timestep embedding
        temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    """Decoder module that upscales and decodes latent representations back into images.

    Parameters
    ----------
    ch : int
        Base channel dimension.
    out_ch : int
        Number of output channels (e.g. 3 for RGB).
    ch_mult : list[int]
        Channel multipliers at each resolution level.
    num_res_blocks : int
        Number of residual blocks at each resolution level.
    attn_resolutions : list[int]
        List of resolutions at which attention layers are applied.
    in_channels : int
        Number of input image channels.
    resolution : int
        Input image resolution.
    z_channels : int
        Number of latent channels.
    dropout : float, optional
        Dropout rate. Default is 0.0.
    resamp_with_conv : bool, optional
        Whether to use convolutional resampling. Default is True.
    give_pre_end : bool, optional
        If True, returns the tensor before the final normalization and convolution. Default is False.
    tanh_out : bool, optional
        If True, applies a tanh activation to the output. Default is False.
    use_linear_attn : bool, optional
        If True, use linear attention. Default is False.
    attn_type : str, optional
        Type of attention to use ('vanilla', 'linear'). Default is 'vanilla'.
    """

    def __init__(
        self,
        *,
        ch: int,
        out_ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        attn_resolutions: list[int],
        in_channels: int,
        resolution: int,
        z_channels: int,
        dropout=0.0,
        resamp_with_conv=True,
        give_pre_end=False,
        tanh_out=False,
        use_linear_attn=False,
        attn_type="vanilla",
        **ignorekwargs,
    ):
        super().__init__()
        if use_linear_attn:
            attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)
        print("Working with z of shape {} = {} dimensions.".format(self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout
        )
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout
        )

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    ResnetBlock(
                        in_channels=block_in, out_channels=block_out, temb_channels=self.temb_ch, dropout=dropout
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z: Tensor) -> Tensor:
        """Forward pass of the Decoder.

        Parameters
        ----------
        z : torch.Tensor
            Latent representation of shape (B, z_channels, H', W').

        Returns
        -------
        torch.Tensor
            Decoded image of shape (B, out_ch, H, W).
        """
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        if self.tanh_out:
            h = torch.tanh(h)
        return h


class DiagonalGaussian(nn.Module):
    """Module that splits an input tensor into mean and log-variance and optionally samples from the Gaussian.

    Parameters
    ----------
    sample : bool, optional
        If True, return a sample from the Gaussian. Otherwise, return the mean. Default is True.
    chunk_dim : int, optional
        Dimension along which to chunk the tensor into mean and log-variance. Default is 1.
    """

    def __init__(self, sample: bool = True, chunk_dim: int = 1):
        super().__init__()
        self.sample = sample
        self.chunk_dim = chunk_dim

    def forward(self, z: Tensor) -> Tensor:
        """Forward pass of the DiagonalGaussian module.

        Parameters
        ----------
        z : torch.Tensor
            Input tensor of shape (..., 2*z_channels, ...).

        Returns
        -------
        torch.Tensor
            If sample=True, returns a sampled tensor from N(mean, var).
            If sample=False, returns the mean.
        """
        mean, logvar = torch.chunk(z, 2, dim=self.chunk_dim)
        if self.sample:
            std = torch.exp(0.5 * logvar)
            return mean + std * torch.randn_like(mean)
        else:
            return mean


class AutoEncoder(nn.Module):
    """Full AutoEncoder model combining an Encoder, Decoder, and latent Gaussian sampling.

    Parameters
    ----------
    params : AutoEncoderParams
        Configuration parameters for the AutoEncoder model.
    """

    def __init__(self, params: AutoEncoderParams):
        super().__init__()
        self.encoder = Encoder(
            resolution=params.resolution,
            in_channels=params.in_channels,
            ch=params.ch,
            ch_mult=params.ch_mult,
            num_res_blocks=params.num_res_blocks,
            z_channels=params.z_channels,
            double_z=params.double_z,
            attn_type=params.attn_type,
            dropout=params.dropout,
            out_ch=params.out_ch,
            attn_resolutions=params.attn_resolutions,
        )
        self.decoder = Decoder(
            resolution=params.resolution,
            in_channels=params.in_channels,
            ch=params.ch,
            out_ch=params.out_ch,
            ch_mult=params.ch_mult,
            num_res_blocks=params.num_res_blocks,
            z_channels=params.z_channels,
            double_z=params.double_z,
            attn_type=params.attn_type,
            dropout=params.dropout,
            attn_resolutions=params.attn_resolutions,
        )
        self.reg = DiagonalGaussian()

        self.scale_factor = params.scale_factor
        self.shift_factor = params.shift_factor
        self.params = params

        if params.ckpt is not None:
            self.load_from_checkpoint(params.ckpt)

    def encode(self, x: Tensor) -> Tensor:
        """Encode an input image to its latent representation.

        Parameters
        ----------
        x : torch.Tensor
            Input image of shape (B, C, H, W).

        Returns
        -------
        torch.Tensor
            Latent representation of the input image.
        """
        z = self.reg(self.encoder(x))
        z = self.scale_factor * (z - self.shift_factor)
        return z

    def decode(self, z: Tensor) -> Tensor:
        """Decode a latent representation back into an image.

        Parameters
        ----------
        z : torch.Tensor
            Latent representation of shape (B, z_channels, H', W').

        Returns
        -------
        torch.Tensor
            Reconstructed image of shape (B, out_ch, H, W).
        """
        z = z / self.scale_factor + self.shift_factor
        return self.decoder(z)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass that encodes and decodes the input image.

        Parameters
        ----------
        x : torch.Tensor
            Input image tensor.

        Returns
        -------
        torch.Tensor
            Reconstructed image.
        """
        return self.decode(self.encode(x))

    def load_from_checkpoint(self, ckpt_path: str):
        """Load the autoencoder weights from a checkpoint file.

        Parameters
        ----------
        ckpt_path : str
            Path to the checkpoint file.
        """
        from safetensors.torch import load_file as load_sft

        state_dict = load_sft(ckpt_path)
        missing, unexpected = self.load_state_dict(state_dict)
        if len(missing) > 0:
            # If logger is not defined, you may replace this with print or similar.
            print(f"Warning: Following keys are missing from checkpoint loaded: {missing}")
