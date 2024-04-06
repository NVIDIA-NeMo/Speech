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
"""SAMPLING ONLY."""

import numpy as np
import torch
from tqdm import tqdm

from nemo.collections.multimodal.models.text_to_image.stable_diffusion.samplers import Sampler
from nemo.collections.multimodal.models.text_to_image.stable_diffusion.samplers.base_sampler import AbstractBaseSampler
from nemo.collections.multimodal.modules.stable_diffusion.diffusionmodules.util import extract_into_tensor
from nemo.collections.multimodal.parts.utils import randn_like


class DDIMSampler(AbstractBaseSampler):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__(model, sampler=Sampler.DDIM, schedule="linear", **kwargs)

    @torch.no_grad()
    def p_sampling_fn(
        self,
        x,
        c,
        t,
        index,
        repeat_noise=False,
        use_original_steps=False,
        quantize_denoised=False,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        old_eps=None,
        t_next=None,
    ):
        b, *_, device = *x.shape, x.device
        e_t, model_output = self._get_model_output(
            x, t, unconditional_conditioning, unconditional_guidance_scale, score_corrector, c, corrector_kwargs
        )
        x_prev, pred_x0 = self._get_x_prev_and_pred_x0(
            use_original_steps,
            b,
            index,
            device,
            x,
            t,
            model_output,
            e_t,
            quantize_denoised,
            repeat_noise,
            temperature,
            noise_dropout,
        )
        return x_prev, pred_x0

    def grad_p_sampling_fn(
        self,
        x,
        c,
        t,
        index,
        repeat_noise=False,
        use_original_steps=False,
        quantize_denoised=False,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        old_eps=None,
        t_next=None,
    ):
        b, *_, device = *x.shape, x.device
        e_t, model_output = self._get_model_output(
            x, t, unconditional_conditioning, unconditional_guidance_scale, score_corrector, c, corrector_kwargs
        )
        outs = self._get_x_prev_and_pred_x0(
            use_original_steps,
            b,
            index,
            device,
            x,
            t,
            model_output,
            e_t,
            quantize_denoised,
            repeat_noise,
            temperature,
            noise_dropout,
        )
        return outs, e_t

    @torch.no_grad()
    def stochastic_encode(self, x0, t, use_original_steps=False, noise=None):
        # fast, but does not allow for exact reconstruction
        # t serves as an index to gather the correct alphas
        if use_original_steps:
            sqrt_alphas_cumprod = self.sqrt_alphas_cumprod
            sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod
        else:
            sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)
            sqrt_one_minus_alphas_cumprod = self.ddim_sqrt_one_minus_alphas

        if noise is None:
            noise = randn_like(x0, generator=self.model.rng)
        return (
            extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0
            + extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise
        )

    @torch.no_grad()
    def decode(
        self,
        x_latent,
        cond,
        t_start,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        use_original_steps=False,
    ):

        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        x_dec = x_latent
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long)
            x_dec, _ = self.p_sample_ddim(
                x_dec,
                cond,
                ts,
                index=index,
                use_original_steps=use_original_steps,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=unconditional_conditioning,
            )
        return x_dec
