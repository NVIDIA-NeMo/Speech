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

import torch
from diffusers import AutoencoderKL
from einops import rearrange


class AutoencoderKLVAE(torch.nn.Module):
    """
    A class that wraps the AutoencoderKL model and provides a decode method.

    Attributes:
        vae (AutoencoderKL): The underlying AutoencoderKL model loaded from a pretrained path.
    """

    def __init__(self, path):
        """
        Initialize the AutoencoderKLVAE instance.

        Args:
            path (str): The path to the pretrained AutoencoderKL model.
        """
        super().__init__()
        self.vae = AutoencoderKL.from_pretrained(path, torch_dtype=torch.bfloat16)

    @torch.no_grad()
    def decode(self, x):
        """
        Decode a latent representation using the underlying VAE model.

        This method takes a latent tensor `x` and decodes it into an image.
        If `x` has a temporal dimension `T` of 1, it
        rearranges the tensor before and after decoding.

        Args:
            x (torch.Tensor): A tensor of shape (B, C, T, H, W), where:
                              B = batch size
                              C = number of channels
                              T = temporal dimension
                              H = height
                              W = width

        Returns:
            torch.Tensor: Decoded image tensor with the same shape as the input (B, C, T, H, W).
        """

        B, C, T, H, W = x.shape
        if T == 1:
            x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = x / self.vae.config.scaling_factor
        out = self.vae.decode(x, return_dict=False)[0]
        if T == 1:
            return rearrange(out, '(b t) c h w -> b c t h w', t=1)
        return out
