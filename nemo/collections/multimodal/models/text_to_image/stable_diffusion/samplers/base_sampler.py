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
from abc import ABC, abstractmethod

import numpy as np
import torch
from tqdm import tqdm

from nemo.collections.multimodal.models.text_to_image.stable_diffusion.samplers import Sampler
from nemo.collections.multimodal.modules.stable_diffusion.diffusionmodules.util import (
    make_ddim_sampling_parameters,
    make_ddim_timesteps,
    noise_like,
)


class AbstractBaseSampler(ABC):
    def __init__(self, model, sampler, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule
        assert isinstance(sampler, Sampler), "Sampler should be of ENUM type Sampler"
        self.sampler = sampler

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0.0, verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(
            ddim_discr_method=ddim_discretize,
            num_ddim_timesteps=ddim_num_steps,
            num_ddpm_timesteps=self.ddpm_num_timesteps,
            verbose=verbose,
        )
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, "alphas have to be defined for each timestep"
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(torch.cuda.current_device())
        self.register_buffer("betas", to_torch(self.model.betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", to_torch(self.model.alphas_cumprod_prev))
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1.0 - alphas_cumprod.cpu())),
        )
        self.register_buffer("log_one_minus_alphas_cumprod", to_torch(np.log(1.0 - alphas_cumprod.cpu())))
        self.register_buffer("sqrt_recip_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod.cpu())))
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod.cpu() - 1)),
        )
        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev, ddim_variance = make_ddim_sampling_parameters(
            alphacums=alphas_cumprod.cpu(), ddim_timesteps=self.ddim_timesteps, eta=ddim_eta, verbose=verbose,
        )
        self.register_buffer("ddim_sigmas", ddim_sigmas)
        self.register_buffer("ddim_alphas", ddim_alphas)
        self.register_buffer("ddim_alphas_prev", ddim_alphas_prev)
        self.register_buffer("ddim_variance", ddim_variance)
        self.register_buffer("ddim_sqrt_one_minus_alphas", np.sqrt(1.0 - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev)
            / (1 - self.alphas_cumprod)
            * (1 - self.alphas_cumprod / self.alphas_cumprod_prev)
        )
        self.register_buffer("ddim_sigmas_for_original_num_steps", sigmas_for_original_sampling_steps)

    @abstractmethod
    def p_sampling_fn(self):
        pass

    def dpm_sampling_fn(self):
        pass

    def para_ddim_sampling_fn(self):
        pass

    @torch.no_grad()
    def sample(
        self,
        S,
        batch_size,
        shape,
        conditioning=None,
        callback=None,
        normals_sequence=None,
        img_callback=None,
        quantize_x0=False,
        eta=0.0,
        mask=None,
        x0=None,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
        verbose=True,
        x_T=None,
        log_every_t=100,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        parallelism=8,
        tolerance=0.1,
        # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
        **kwargs,
    ):
        self.verbose = verbose

        if conditioning is not None:
            if isinstance(conditioning, dict):
                ctmp = conditioning[list(conditioning.keys())[0]]
                while isinstance(ctmp, list):
                    ctmp = ctmp[0]
                cbs = ctmp.shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")
        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        if self.verbose:
            print(f"Data shape for sampling is {size}, eta {eta}")

        if self.sampler is Sampler.DPM:
            return self.dpm_sampling_fn(
                shape=shape,
                steps=S,
                conditioning=conditioning,
                unconditional_conditioning=unconditional_conditioning,
                unconditional_guidance_scale=unconditional_guidance_scale,
                x_T=x_T,
            )

        if self.sampler is Sampler.PARA_DDIM:
            return self.para_ddim_sampling_fn(
                cond=conditioning,
                batch_size=batch_size,
                per_latent_shape=shape,
                x_T=x_T,
                steps=S,
                parallelism=parallelism,
                tolerance=tolerance,
                temperature=temperature,
                noise_dropout=noise_dropout,
                quantize_denoised=quantize_x0,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=unconditional_conditioning,
                score_corrector=score_corrector,
                corrector_kwargs=corrector_kwargs,
            )

        samples, intermediates = self.sampling_fn(
            conditioning,
            size,
            callback=callback,
            img_callback=img_callback,
            quantize_denoised=quantize_x0,
            mask=mask,
            x0=x0,
            ddim_use_original_steps=False,
            noise_dropout=noise_dropout,
            temperature=temperature,
            score_corrector=score_corrector,
            corrector_kwargs=corrector_kwargs,
            x_T=x_T,
            log_every_t=log_every_t,
            unconditional_guidance_scale=unconditional_guidance_scale,
            unconditional_conditioning=unconditional_conditioning,
        )
        return samples, intermediates

    @torch.no_grad()
    def sampling_fn(
        self,
        cond,
        shape,
        x_T=None,
        ddim_use_original_steps=False,
        callback=None,
        timesteps=None,
        quantize_denoised=False,
        mask=None,
        x0=None,
        img_callback=None,
        log_every_t=100,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
    ):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, generator=self.model.rng, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]
        intermediates = {"x_inter": [img], "pred_x0": [img]}

        # TODO: Is this needed
        if self.sampler is Sampler.PLMS:
            time_range = list(reversed(range(0, timesteps))) if ddim_use_original_steps else np.flip(timesteps)
        else:
            time_range = reversed(range(0, timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]

        if self.verbose:
            print(f"Running {self.sampler.name} Sampling with {total_steps} timesteps")
            iterator = tqdm(time_range, desc=f"{self.sampler.name} Sampler", total=total_steps)
        else:
            iterator = time_range

        old_eps = []
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)
            if self.sampler is Sampler.PLMS:
                ts_next = torch.full(
                    (b,), time_range[min(i + 1, len(time_range) - 1)], device=device, dtype=torch.long,
                )
            else:
                old_eps = None
                ts_next = None
            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1.0 - mask) * img
            outs = self.p_sampling_fn(
                img,
                cond,
                ts,
                index=index,
                use_original_steps=ddim_use_original_steps,
                quantize_denoised=quantize_denoised,
                temperature=temperature,
                noise_dropout=noise_dropout,
                score_corrector=score_corrector,
                corrector_kwargs=corrector_kwargs,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=unconditional_conditioning,
                old_eps=old_eps,
                t_next=ts_next,
            )
            img, pred_x0 = outs[0], outs[1]
            if self.sampler is Sampler.PLMS:
                e_t = outs[2]
                old_eps.append(e_t)
                if len(old_eps) >= 4:
                    old_eps.pop(0)
            if callback:
                callback(i)
            if img_callback:
                img_callback(pred_x0, i)
            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates["x_inter"].append(img)
                intermediates["pred_x0"].append(pred_x0)
        return img, intermediates

    def single_ddim_denoise_step(
        self,
        img,
        total_steps,
        i,
        b,
        device,
        step,
        cond,
        ddim_use_original_steps=None,
        quantize_denoised=None,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
    ):

        index = total_steps - i - 1
        ts = torch.full((b,), step, device=device, dtype=torch.long)

        outs, eps_t = self.grad_p_sampling_fn(
            img,
            cond,
            ts,
            index=index,
            use_original_steps=ddim_use_original_steps,
            quantize_denoised=quantize_denoised,
            temperature=temperature,
            noise_dropout=noise_dropout,
            score_corrector=score_corrector,
            corrector_kwargs=corrector_kwargs,
            unconditional_guidance_scale=unconditional_guidance_scale,
            unconditional_conditioning=unconditional_conditioning,
            old_eps=None,
            t_next=None,
        )

        img, pred_x0 = outs[0], outs[1]
        return img, pred_x0, eps_t

    def _get_model_output(
        self, x, t, unconditional_conditioning, unconditional_guidance_scale, score_corrector, c, corrector_kwargs,
    ):
        if unconditional_conditioning is None or unconditional_guidance_scale == 1.0:
            model_output = self.model.apply_model(x, t, c)
        elif isinstance(c, dict):
            ### Contolnet conditioning is dict format
            model_t = self.model.apply_model(x, t, c)
            model_uncond = self.model.apply_model(x, t, unconditional_conditioning)
            model_output = model_uncond + unconditional_guidance_scale * (model_t - model_uncond)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            c_in = torch.cat([unconditional_conditioning, c])
            e_t_uncond, model_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
            model_output = e_t_uncond + unconditional_guidance_scale * (model_t - e_t_uncond)
        if self.model.parameterization == "v":
            e_t = self.model.predict_eps_from_z_and_v(x, t, model_output)
        else:
            e_t = model_output
        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)
        return e_t, model_output

    def _get_x_prev_and_pred_x0(
        self,
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
    ):
        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = (
            self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        )
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas

        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index], device=device)
        # current prediction for x_0
        if self.model.parameterization != "v":
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        else:
            pred_x0 = self.model.predict_start_from_z_and_v(x, t, model_output)
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        dir_xt = (1.0 - a_prev - sigma_t ** 2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.0:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0
