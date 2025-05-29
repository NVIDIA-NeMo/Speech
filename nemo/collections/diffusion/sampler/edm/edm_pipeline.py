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
#
# flake8: noqa
# pylint: skip-file

from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.distributed
from einops import rearrange
from megatron.core import parallel_state
from torch import Tensor

from nemo.collections.diffusion.sampler.batch_ops import batch_mul
from nemo.collections.diffusion.sampler.context_parallel import cat_outputs_cp
from nemo.collections.diffusion.sampler.edm.edm import EDMSDE, EDMSampler, EDMScaling


class EDMPipeline:
    """
    EDMPipeline is a class that implements a diffusion model pipeline for video generation. It includes methods for
    initializing the pipeline, encoding and decoding video data, performing training steps, denoising, and generating
    samples.
    Attributes:
        p_mean: Mean for SDE process.
        p_std: Standard deviation for SDE process.
        sigma_max: Maximum noise level.
        sigma_min: Minimum noise level.
        _noise_generator: Generator for noise.
        _noise_level_generator: Generator for noise levels.
        sde: SDE process.
        sampler: Sampler for the diffusion model.
        scaling: Scaling for EDM.
        input_data_key: Key for input video data.
        input_image_key: Key for input image data.
        tensor_kwargs: Tensor keyword arguments.
        loss_reduce: Method for reducing loss.
        loss_scale: Scale factor for loss.
        aesthetic_finetuning: Aesthetic finetuning parameter.
        camera_sample_weight: Camera sample weight parameter.
        loss_mask_enabled: Flag for enabling loss mask.
    Methods:
        noise_level_generator: Returns the noise level generator.
        _initialize_generators: Initializes noise and noise-level generators.
        encode: Encodes input tensor using the video tokenizer.
        decode: Decodes latent tensor using video tokenizer.
        training_step: Performs a single training step for the diffusion model.
        denoise: Performs denoising on the input noise data, noise level, and condition.
        compute_loss_with_epsilon_and_sigma: Computes the loss for training.
        get_per_sigma_loss_weights: Returns loss weights per sigma noise level.
        get_condition_uncondition: Returns conditioning and unconditioning for classifier-free guidance.
        get_x0_fn_from_batch: Creates a function to generate denoised predictions with the sampler.
        generate_samples_from_batch: Generates samples based on input data batch.
        _normalize_video_databatch_inplace: Normalizes video data in-place on a CUDA device to [-1, 1].
        draw_training_sigma_and_epsilon: Draws training noise (epsilon) and noise levels (sigma).
        random_dropout_input: Applies random dropout to the input tensor.
        get_data_and_condition: Retrieves data and conditioning for model input.
    """

    def __init__(
        self,
        net,
        vae=None,
        p_mean=0.0,
        p_std=1.0,
        sigma_max=80,
        sigma_min=0.0002,
        sigma_data=0.5,
        seed=1234,
    ):
        """
        Initializes the EDM pipeline with the given parameters.

        Args:
            net: The DiT model.
            vae: The Video Tokenizer (optional).
            p_mean (float): Mean for the SDE.
            p_std (float): Standard deviation for the SDE.
            sigma_max (float): Maximum sigma value for the SDE.
            sigma_min (float): Minimum sigma value for the SDE.
            sigma_data (float): Sigma value for EDM scaling.
            seed (int): Random seed for reproducibility.

        Attributes:
            vae: The Video Tokenizer.
            net: The DiT model.
            p_mean (float): Mean for the SDE.
            p_std (float): Standard deviation for the SDE.
            sigma_max (float): Maximum sigma value for the SDE.
            sigma_min (float): Minimum sigma value for the SDE.
            sigma_data (float): Sigma value for EDM scaling.
            seed (int): Random seed for reproducibility.
            _noise_generator: Placeholder for noise generator.
            _noise_level_generator: Placeholder for noise level generator.
            sde: Instance of EDMSDE initialized with p_mean, p_std, sigma_max, and sigma_min.
            sampler: Instance of EDMSampler.
            scaling: Instance of EDMScaling initialized with sigma_data.
            input_data_key (str): Key for input data.
            input_image_key (str): Key for input images.
            tensor_kwargs (dict): Tensor keyword arguments for device and dtype.
            loss_reduce (str): Method to reduce loss ('mean' or other).
            loss_scale (float): Scale factor for loss.
        """
        self.vae = vae
        self.net = net

        self.p_mean = p_mean
        self.p_std = p_std
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.sigma_data = sigma_data

        self.seed = seed
        self._noise_generator = None
        self._noise_level_generator = None

        self.sde = EDMSDE(p_mean, p_std, sigma_max, sigma_min)
        self.sampler = EDMSampler()
        self.scaling = EDMScaling(sigma_data)

        self.input_data_key = 'video'
        self.input_image_key = 'images_1024'
        self.tensor_kwargs = {"device": "cuda", "dtype": torch.bfloat16}
        self.loss_reduce = 'mean'
        self.loss_scale = 1.0

    @property
    def noise_level_generator(self):
        """
        Generates noise levels for the EDM pipeline.

        Returns:
            Callable: A function or generator that produces noise levels.
        """
        return self._noise_level_generator

    def _initialize_generators(self):
        """
        Initializes the random number generators for noise and noise level.

        This method sets up two generators:
        1. A PyTorch generator for noise, seeded with a combination of the base seed and the data parallel rank.
        2. A NumPy generator for noise levels, seeded similarly but without considering context parallel rank.

        Returns:
            None
        """
        noise_seed = self.seed + 100 * parallel_state.get_data_parallel_rank(with_context_parallel=True)
        noise_level_seed = self.seed + 100 * parallel_state.get_data_parallel_rank(with_context_parallel=False)
        self._noise_generator = torch.Generator(device='cuda')
        self._noise_generator.manual_seed(noise_seed)
        self._noise_level_generator = np.random.default_rng(noise_level_seed)
        self.sde._generator = self._noise_level_generator

    def training_step(
        self, data_batch: dict[str, torch.Tensor], iteration: int
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """
        Performs a single training step for the diffusion model.

        This method is responsible for executing one iteration of the model's training. It involves:
        1. Adding noise to the input data using the SDE process.
        2. Passing the noisy data through the network to generate predictions.
        3. Computing the loss based on the difference between the predictions and the original data.

        Args:
            data_batch (dict): raw data batch draw from the training data loader.
            iteration (int): Current iteration number.

        Returns:
            A tuple with the output batch and the computed loss.
        """
        # Get the input data to noise and denoise~(image, video) and the corresponding conditioner.
        x0_from_data_batch, x0, condition = self.get_data_and_condition(data_batch)

        # Sample pertubation noise levels and N(0, 1) noises
        sigma, epsilon = self.draw_training_sigma_and_epsilon(x0.size(), condition)

        if parallel_state.is_pipeline_last_stage():
            output_batch, pred_mse, edm_loss = self.compute_loss_with_epsilon_and_sigma(
                data_batch, x0_from_data_batch, x0, condition, epsilon, sigma
            )

            return output_batch, edm_loss
        else:
            net_output = self.compute_loss_with_epsilon_and_sigma(
                data_batch, x0_from_data_batch, x0, condition, epsilon, sigma
            )
            return net_output

    def denoise(self, xt: torch.Tensor, sigma: torch.Tensor, condition: dict[str, torch.Tensor]):
        """
        Performs denoising on the input noise data, noise level, and condition

        Args:
            xt (torch.Tensor): The input noise data.
            sigma (torch.Tensor): The noise level.
            condition (dict[str, torch.Tensor]): conditional information

        Returns:
            Predicted clean data (x0) and noise (eps_pred).
        """

        xt = xt.to(**self.tensor_kwargs)
        sigma = sigma.to(**self.tensor_kwargs)
        # get precondition for the network
        c_skip, c_out, c_in, c_noise = self.scaling(sigma=sigma)

        net_output = self.net(
            x=batch_mul(c_in, xt),  # Eq. 7 of https://arxiv.org/pdf/2206.00364.pdf
            timesteps=c_noise,  # Eq. 7 of https://arxiv.org/pdf/2206.00364.pdf
            **condition,
        )

        if not parallel_state.is_pipeline_last_stage():
            return net_output

        x0_pred = batch_mul(c_skip, xt) + batch_mul(c_out, net_output)

        # get noise prediction based on sde
        eps_pred = batch_mul(xt - x0_pred, 1.0 / sigma)

        return x0_pred, eps_pred

    def compute_loss_with_epsilon_and_sigma(
        self,
        data_batch: dict[str, torch.Tensor],
        x0_from_data_batch: torch.Tensor,
        x0: torch.Tensor,
        condition: dict[str, torch.Tensor],
        epsilon: torch.Tensor,
        sigma: torch.Tensor,
    ):
        """
        Computes the loss for training.

        Args:
            data_batch: Batch of input data.
            x0_from_data_batch: Raw input tensor.
            x0: Latent tensor.
            condition: Conditional input data.
            epsilon: Noise tensor.
            sigma: Noise level tensor.

        Returns:
            The computed loss.
        """
        # Get the mean and stand deviation of the marginal probability distribution.
        mean, std = self.sde.marginal_prob(x0, sigma)
        # Generate noisy observations
        xt = mean + batch_mul(std, epsilon)  # corrupted data

        if parallel_state.is_pipeline_last_stage():
            # make prediction
            x0_pred, eps_pred = self.denoise(xt, sigma, condition)
            # loss weights for different noise levels
            weights_per_sigma = self.get_per_sigma_loss_weights(sigma=sigma)
            pred_mse = (x0 - x0_pred) ** 2
            edm_loss = batch_mul(pred_mse, weights_per_sigma)

            output_batch = {
                "x0": x0,
                "xt": xt,
                "sigma": sigma,
                "weights_per_sigma": weights_per_sigma,
                "condition": condition,
                "model_pred": {"x0_pred": x0_pred, "eps_pred": eps_pred},
                "mse_loss": pred_mse.mean(),
                "edm_loss": edm_loss.mean(),
            }
            return output_batch, pred_mse, edm_loss
        else:
            # make prediction
            x0_pred = self.denoise(xt, sigma, condition)
            return x0_pred.contiguous()

    def get_per_sigma_loss_weights(self, sigma: torch.Tensor):
        """
        Args:
            sigma (tensor): noise level

        Returns:
            loss weights per sigma noise level
        """
        return (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2

    def get_condition_uncondition(self, data_batch: Dict):
        """Returns conditioning and unconditioning for classifier-free guidance."""
        _, _, condition = self.get_data_and_condition(data_batch, dropout_rate=0.0)

        if 'neg_t5_text_embeddings' in data_batch:
            data_batch['t5_text_embeddings'] = data_batch['neg_t5_text_embeddings']
            data_batch["t5_text_mask"] = data_batch["neg_t5_text_mask"]
            _, _, uncondition = self.get_data_and_condition(data_batch, dropout_rate=1.0)
        else:
            _, _, uncondition = self.get_data_and_condition(data_batch, dropout_rate=1.0)

        return condition, uncondition

    def get_x0_fn_from_batch(
        self,
        data_batch: Dict,
        guidance: float = 1.5,
        is_negative_prompt: bool = False,
    ) -> Callable:
        """
        Creates a function to generate denoised predictions with the sampler.

        Args:
            data_batch: Batch of input data.
            guidance: Guidance scale factor.
            is_negative_prompt: Whether to use negative prompts.

        Returns:
            A callable to predict clean data (x0).
        """
        condition, uncondition = self.get_condition_uncondition(data_batch)

        def x0_fn(noise_x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
            cond_x0, _ = self.denoise(noise_x, sigma, condition)
            uncond_x0, _ = self.denoise(noise_x, sigma, uncondition)
            return cond_x0 + guidance * (cond_x0 - uncond_x0)

        return x0_fn

    def generate_samples_from_batch(
        self,
        data_batch: Dict,
        guidance: float = 1.5,
        state_shape: Tuple | None = None,
        is_negative_prompt: bool = False,
        num_steps: int = 35,
    ) -> Tensor:
        """
        Generates samples based on input data batch.

        Args:
            data_batch: Batch of input data.
            guidance: Guidance scale factor.
            state_shape: Shape of the state.
            is_negative_prompt: Whether to use negative prompts.
            num_steps: Number of steps for sampling.
            solver_option: SDE Solver option.

        Returns:
            Generated samples from diffusion model.
        """
        cp_enabled = parallel_state.get_context_parallel_world_size() > 1

        if self._noise_generator is None:
            self._initialize_generators()
        x0_fn = self.get_x0_fn_from_batch(data_batch, guidance, is_negative_prompt=is_negative_prompt)

        state_shape = list(state_shape)
        state_shape[1] //= parallel_state.get_context_parallel_world_size()
        x_sigma_max = (
            torch.randn(state_shape, **self.tensor_kwargs, generator=self._noise_generator) * self.sde.sigma_max
        )

        samples = self.sampler(x0_fn, x_sigma_max, num_steps=num_steps, sigma_max=self.sde.sigma_max)

        if cp_enabled:
            cp_group = parallel_state.get_context_parallel_group()
            samples = cat_outputs_cp(samples, seq_dim=2, cp_group=cp_group)

        return samples

    def draw_training_sigma_and_epsilon(self, x0_size: int, condition: Any) -> torch.Tensor:
        """
        Draws training noise (epsilon) and noise levels (sigma).

        Args:
            x0_size: Shape of the input tensor.
            condition: Conditional input (unused).

        Returns:
            Noise level (sigma) and noise (epsilon).
        """
        del condition
        batch_size = x0_size[0]
        if self._noise_generator is None:
            self._initialize_generators()
        epsilon = torch.randn(x0_size, **self.tensor_kwargs, generator=self._noise_generator)
        return self.sde.sample_t(batch_size).to(**self.tensor_kwargs), epsilon

    def random_dropout_input(self, in_tensor: torch.Tensor, dropout_rate: Optional[float] = None) -> torch.Tensor:
        """
        Applies random dropout to the input tensor.

        Args:
            in_tensor: Input tensor.
            dropout_rate: Dropout probability (optional).

        Returns:
            Conditioning with random dropout applied.
        """
        dropout_rate = dropout_rate if dropout_rate is not None else self.dropout_rate
        return batch_mul(
            torch.bernoulli((1.0 - dropout_rate) * torch.ones(in_tensor.shape[0])).type_as(in_tensor),
            in_tensor,
        )

    def get_data_and_condition(self, data_batch: dict[str, Tensor], dropout_rate=0.2) -> Tuple[Tensor]:
        """
        Retrieves data and conditioning for model input.

        Args:
            data_batch: Batch of input data.
            dropout_rate: Dropout probability for conditioning.

        Returns:
            Raw data, latent data, and conditioning information.
        """
        # Latent state
        raw_state = data_batch["video"] * self.sigma_data
        # assume data is already encoded
        latent_state = raw_state

        # Condition
        condition = {}  # Create a new dictionary for condition
        # Copy all keys from data_batch except 'video'
        for key, value in data_batch.items():
            if key not in ['video', 't5_text_embeddings']:
                condition[key] = value
        condition['crossattn_emb'] = self.random_dropout_input(
            data_batch['t5_text_embeddings'], dropout_rate=dropout_rate
        )

        return raw_state, latent_state, condition
