# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import torch
from torch import Tensor, nn


def rope(pos: torch.Tensor, dim: int, theta: int) -> torch.Tensor:
    """
    Different from the original ROPE used for flux.
    Megatron attention takes the out product and calculate sin/cos inside, so we only need to get the freqs here
    in the shape of [seq, ..., dim]
    """
    assert dim % 2 == 0, "The dimension must be even."

    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta**scale)

    out = torch.einsum("...n,d->...nd", pos, omega)

    return out.float()


class EmbedND(nn.Module):
    '''
    Generate Rope matrix with preset axes dimensions.
    '''

    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        # pylint: disable=C0116
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        # pylint: disable=C0116
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-1,
        )
        emb = emb.unsqueeze(1).permute(2, 0, 1, 3)
        return torch.stack([emb, emb], dim=-1).reshape(*emb.shape[:-1], -1)


class MLPEmbedder(nn.Module):
    '''
    MLP embedder with two projection layers and Silu in between.
    '''

    def __init__(self, in_dim: int, hidden_dim: int):
        # pylint: disable=C0116
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        # pylint: disable=C0116
        return self.out_layer(self.silu(self.in_layer(x)))


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = True,
    downscale_freq_shift: float = 0,
    scale: float = 1,
    max_period: int = 10000,
):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.

    Args
        timesteps (torch.Tensor):
            a 1-D Tensor of N indices, one per batch element. These may be fractional.
        embedding_dim (int):
            the dimension of the output.
        flip_sin_to_cos (bool):
            Whether the embedding order should be `cos, sin` (if True) or `sin, cos` (if False)
        downscale_freq_shift (float):
            Controls the delta between frequencies between dimensions
        scale (float):
            Scaling factor applied to the embeddings.
        max_period (int):
            Controls the maximum frequency of the embeddings
    Returns
        torch.Tensor: an [N x dim] Tensor of positional embeddings.
    """
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]

    # scale embeddings
    emb = scale * emb

    # concat sine and cosine embeddings
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    # flip sine and cosine embeddings
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


# pylint: disable=C0116
class Timesteps(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        flip_sin_to_cos: bool = True,
        downscale_freq_shift: float = 0,
        scale: float = 1,
        max_period: int = 10000,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale
        self.max_period = max_period

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        t_emb = get_timestep_embedding(
            timesteps,
            self.embedding_dim,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
            scale=self.scale,
            max_period=self.max_period,
        )
        return t_emb


# pylint: disable=C0116


class TimeStepEmbedder(nn.Module):
    """
    A neural network module that embeds timesteps for use in models such as diffusion models.
    It projects the input timesteps to a higher-dimensional space and then embeds them using
    an MLP (Multilayer Perceptron). The projection and embedding provide a learned representation
    of the timestep that can be used in further computations.

    Args:
        embedding_dim (int):
            The dimensionality of the timestep embedding space.
        hidden_dim (int):
            The dimensionality of the hidden layer in the MLPEmbedder.
        flip_sin_to_cos (bool, optional):
            Whether to flip the sine and cosine components during the projection (default is True).
        downscale_freq_shift (float, optional):
            A scaling factor for the frequency shift during the projection (default is 0).
        scale (float, optional):
            A scaling factor applied to the timestep projections (default is 1).
        max_period (int, optional):
            The maximum period for the sine and cosine functions used in projection (default is 10000).

    Methods:
        forward: Takes a tensor of timesteps and returns their embedded representation.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        flip_sin_to_cos: bool = True,
        downscale_freq_shift: float = 0,
        scale: float = 1,
        max_period: int = 10000,
    ):

        super().__init__()

        self.time_proj = Timesteps(
            embedding_dim=embedding_dim,
            flip_sin_to_cos=flip_sin_to_cos,
            downscale_freq_shift=downscale_freq_shift,
            scale=scale,
            max_period=max_period,
        )
        self.time_embedder = MLPEmbedder(in_dim=embedding_dim, hidden_dim=hidden_dim)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        # pylint: disable=C0116
        timesteps_proj = self.time_proj(timesteps)
        timesteps_emb = self.time_embedder(timesteps_proj)

        return timesteps_emb
