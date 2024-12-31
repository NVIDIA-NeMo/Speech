# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import torch
from einops import rearrange
from torch import Tensor, nn

try:
    from apex.contrib.group_norm import GroupNorm

    OPT_GROUP_NORM = True
except Exception:
    print('Fused optimized group norm has not been installed.')
    OPT_GROUP_NORM = False


def Normalize(in_channels, num_groups=32, act=""):
    """Creates a group normalization layer with specified activation.

    Args:
        in_channels (int): Number of channels in the input.
        num_groups (int, optional): Number of groups for GroupNorm. Defaults to 32.
        act (str, optional): Activation function name. Defaults to "".

    Returns:
        GroupNorm: A normalization layer with optional activation.
    """
    return GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True, act=act)


def nonlinearity(x):
    """Nonlinearity function used in temporal embedding projection.

    Currently implemented as a SiLU (Swish) function.

    Args:
        x (Tensor): Input tensor.

    Returns:
        Tensor: Output after applying SiLU activation.
    """
    return x * torch.sigmoid(x)


class ResnetBlock(nn.Module):
    """A ResNet-style block that can optionally apply a temporal embedding and shortcut projections.

    This block consists of two convolutional layers, normalization, and optional temporal embedding.
    It can adjust channel dimensions between input and output via shortcuts.
    """

    def __init__(self, in_channels, out_channels=None, conv_shortcut=False, dropout=0.0, temb_channels=0):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int, optional): Number of output channels. Defaults to in_channels.
            conv_shortcut (bool, optional): Whether to use a convolutional shortcut. Defaults to False.
            dropout (float, optional): Dropout probability. Defaults to 0.0.
            temb_channels (int, optional): Number of channels in temporal embedding. Defaults to 0.
        """
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels, act="silu")
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels, act="silu")
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, temb):
        """Forward pass of the ResnetBlock.

        Args:
            x (Tensor): Input feature map of shape (B, C, H, W).
            temb (Tensor): Temporal embedding tensor of shape (B, temb_channels).

        Returns:
            Tensor: Output feature map of shape (B, out_channels, H, W).
        """
        h = x
        h = self.norm1(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class Upsample(nn.Module):
    """Upsampling block that increases spatial resolution by a factor of 2.

    Can optionally include a convolution after upsampling.
    """

    def __init__(self, in_channels, with_conv):
        """
        Args:
            in_channels (int): Number of input channels.
            with_conv (bool): If True, apply a convolution after upsampling.
        """
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        """Forward pass of the Upsample block.

        Args:
            x (Tensor): Input feature map (B, C, H, W).

        Returns:
            Tensor: Upsampled feature map (B, C, 2H, 2W).
        """
        # Cast to float32 to as 'upsample_nearest2d_out_frame' op does not support bfloat16
        dtype = x.dtype
        if dtype == torch.bfloat16:
            x = x.to(torch.float32)
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if dtype == torch.bfloat16:
            x = x.to(dtype)
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """Downsampling block that reduces spatial resolution by a factor of 2.

    Can optionally include a convolution before downsampling.
    """

    def __init__(self, in_channels, with_conv):
        """
        Args:
            in_channels (int): Number of input channels.
            with_conv (bool): If True, apply a convolution before downsampling.
        """
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        """Forward pass of the Downsample block.

        Args:
            x (Tensor): Input feature map (B, C, H, W).

        Returns:
            Tensor: Downsampled feature map (B, C, H/2, W/2).
        """
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class AttnBlock(nn.Module):
    """Self-attention block that applies scaled dot-product attention to feature maps.

    Normalizes input, computes queries, keys, and values, then applies attention and a projection.
    """

    def __init__(self, in_channels: int):
        """
        Args:
            in_channels (int): Number of input/output channels.
        """
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels, act="silu")

        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def attention(self, h_: Tensor) -> Tensor:
        """Compute the attention over the input feature maps.

        Args:
            h_ (Tensor): Normalized input feature map (B, C, H, W).

        Returns:
            Tensor: Output after applying scaled dot-product attention (B, C, H, W).
        """
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape
        q = rearrange(q, "b c h w -> b 1 (h w) c").contiguous()
        k = rearrange(k, "b c h w -> b 1 (h w) c").contiguous()
        v = rearrange(v, "b c h w -> b 1 (h w) c").contiguous()
        h_ = nn.functional.scaled_dot_product_attention(q, k, v)

        return rearrange(h_, "b 1 (h w) c -> b c h w", h=h, w=w, c=c, b=b)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the AttnBlock.

        Args:
            x (Tensor): Input feature map (B, C, H, W).

        Returns:
            Tensor: Output feature map after self-attention (B, C, H, W).
        """
        return x + self.proj_out(self.attention(x))


class LinearAttention(nn.Module):
    """Linear Attention block for efficient attention computations.

    Uses linear attention mechanisms to reduce complexity and memory usage.
    """

    def __init__(self, dim, heads=4, dim_head=32):
        """
        Args:
            dim (int): Input channel dimension.
            heads (int, optional): Number of attention heads. Defaults to 4.
            dim_head (int, optional): Dimension per attention head. Defaults to 32.
        """
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        """Forward pass of the LinearAttention block.

        Args:
            x (Tensor): Input feature map (B, C, H, W).

        Returns:
            Tensor: Output feature map after linear attention (B, C, H, W).
        """
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads=self.heads, qkv=3)
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)


class LinAttnBlock(LinearAttention):
    """Wrapper class to provide a linear attention block in a form compatible with other attention blocks."""

    def __init__(self, in_channels):
        """
        Args:
            in_channels (int): Number of input/output channels.
        """
        super().__init__(dim=in_channels, heads=1, dim_head=in_channels)


def make_attn(in_channels, attn_type="vanilla"):
    """Factory function to create an attention block.

    Args:
        in_channels (int): Number of input/output channels.
        attn_type (str, optional): Type of attention block to create. Options: "vanilla", "linear", "none".
                                   Defaults to "vanilla".

    Returns:
        nn.Module: An instance of the requested attention block.
    """
    assert attn_type in ["vanilla", "linear", "none"], f'attn_type {attn_type} unknown'
    print(f"making attention of type '{attn_type}' with {in_channels} in_channels")
    if attn_type == "vanilla":
        return AttnBlock(in_channels)
    elif attn_type == "none":
        return nn.Identity(in_channels)
    else:
        return LinAttnBlock(in_channels)
