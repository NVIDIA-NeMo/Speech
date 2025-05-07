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

import os
from typing import List, Optional, Union

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file as load_safetensors
from safetensors.torch import save_file as save_safetensors
from torch import nn
from tqdm import tqdm

from nemo.collections.diffusion.encoders.conditioner import FrozenCLIPEmbedder, FrozenT5Embedder
from nemo.collections.diffusion.models.flux.model import Flux
from nemo.collections.diffusion.models.flux_controlnet.model import FluxControlNet, FluxControlNetConfig
from nemo.collections.diffusion.sampler.flow_matching.flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from nemo.collections.diffusion.utils.flux_ckpt_converter import flux_transformer_converter
from nemo.collections.diffusion.utils.flux_pipeline_utils import FluxModelParams
from nemo.collections.diffusion.vae.autoencoder import AutoEncoder
from nemo.utils import logging


class FluxInferencePipeline(nn.Module):
    """
    A pipeline for performing image generation with flux.

    Args:
        params (FluxModelParams, optional):
            Configuration parameters for the model pipeline, including device settings and model configurations.
        flux (Flux, optional):
            A pre-initialized Flux model used for the transformation process.
            If None, a new Flux model is created using the configuration in `params`.
        vae (AutoEncoder, optional):
            A pre-initialized VAE (Variational Autoencoder) model.
            If None, a new VAE model is created using the configuration in `params.vae_config`.
        t5 (FrozenT5Embedder, optional):
            A pre-initialized FrozenT5Embedder model.
            If None, a new T5 model is created using the configuration in `params.t5_params`.
        clip (FrozenCLIPEmbedder, optional):
            A pre-initialized FrozenCLIPEmbedder model.
            If None, a new CLIP model is created using the configuration in `params.clip_params`.
        scheduler_steps (int, optional):
            The number of scheduler steps to use for inference. Default is 1000.

    Attributes:
        device (torch.device): The device (CPU or GPU) where the models will be placed.
        vae (AutoEncoder): The VAE model used for image reconstruction or generation.
        clip_encoder (FrozenCLIPEmbedder): The CLIP encoder for processing image-text inputs.
        t5_encoder (FrozenT5Embedder): The T5 encoder for processing text inputs.
        transformer (Flux): The Flux model used for image-text joint processing.
        vae_scale_factor (float): A scale factor for the VAE, based on the number of channels in the VAE.
        scheduler (FlowMatchEulerDiscreteScheduler): Scheduler used for controlling the flow of inference steps.
        params (FluxModelParams): Configuration parameters used for model setup.

    Methods:
        load_from_pretrained:
            Loads model weights from a checkpoint.
        encoder_prompt:
            Encodes text prompts and retrieves embeddings.
        _prepare_latent_image_ids:
            Prepares latent image ids for the generation process.
        _pack_latents:
            Packs latents into the desired format for input to the model.
        _unpack_latents:
            Unpacks latents from the model into image format.
        _calculate_shift:
            Calculates the shift parameter used for controlling sequence lengths in the model.
        prepare_latents:
            Prepares the latent tensors and latent image ids for generation.
        _generate_rand_latents:
            Generates random latents using a specified generator.
        numpy_to_pil:
            Converts a numpy array or a batch of images to PIL images.
        torch_to_numpy:
            Converts a tensor of images to a numpy array.
        denormalize:
            Denormalizes the image to the range [0, 1].
        __call__:
            Runs the entire image generation process based on the input prompt, including encoding,
            latent preparation, inference, and output generation.

    Example:
        pipeline = FluxInferencePipeline(params)
        images = pipeline(
            prompt=["A beautiful sunset over a mountain range"],
            height=512,
            width=512,
            num_inference_steps=50,
            guidance_scale=7.5
        )
    """

    def __init__(
        self,
        params: FluxModelParams = None,
        flux: Optional[Flux] = None,
        vae: Optional[AutoEncoder] = None,
        t5: Optional[FrozenT5Embedder] = None,
        clip: Optional[FrozenCLIPEmbedder] = None,
        scheduler_steps: int = 1000,
    ):
        """
        Initializes the FluxInferencePipeline with the provided models and configurations.

        Args:
            params (FluxModelParams, optional):
                Configuration parameters for the model pipeline, including device settings and model configurations.
            flux (Flux, optional):
                A pre-initialized Flux model used for the transformation process.
                If None, a new Flux model is created using the configuration in `params`.
            vae (AutoEncoder, optional):
                A pre-initialized VAE (Variational Autoencoder) model.
                If None, a new VAE model is created using the configuration in `params.vae_config`.
            t5 (FrozenT5Embedder, optional):
                A pre-initialized FrozenT5Embedder model.
                If None, a new T5 model is created using the configuration in `params.t5_params`.
            clip (FrozenCLIPEmbedder, optional):
                A pre-initialized FrozenCLIPEmbedder model.
                If None, a new CLIP model is created using the configuration in `params.clip_params`.
            scheduler_steps (int, optional): The number of scheduler steps to use for inference. Default is 1000.
        """
        super().__init__()
        self.device = params.device
        params.clip_params.device = self.device
        params.t5_params.device = self.device

        self.vae = AutoEncoder(params.vae_config).to(self.device).eval() if vae is None else vae
        self.clip_encoder = (
            FrozenCLIPEmbedder(
                version=params.clip_params.version,
                max_length=params.clip_params.max_length,
                always_return_pooled=params.clip_params.always_return_pooled,
                device=params.clip_params.device,
            )
            if clip is None
            else clip
        )
        self.t5_encoder = (
            FrozenT5Embedder(
                params.t5_params.version,
                max_length=params.t5_params.max_length,
                device=params.t5_params.device,
                load_config_only=params.t5_params.load_config_only,
            )
            if t5 is None
            else t5
        )
        self.transformer = Flux(params.flux_config).to(self.device).eval() if flux is None else flux
        self.vae_scale_factor = 2 ** (len(self.vae.params.ch_mult))
        self.scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=scheduler_steps)
        self.params = params

    def load_from_pretrained(self, ckpt_path, do_convert_from_hf=True, save_converted_model_to=None):
        """
        Loads the model's weights from a checkpoint. If HF ckpt is provided, it will be converted to NeMo
        format and save it to local folder.

        Args:
            ckpt_path (str):
                Path to the checkpoint file.
            do_convert_from_hf (bool, optional):
                Whether to convert the checkpoint from Hugging Face format before loading. Default is True.
            save_converted_model_to (str, optional):
                Path to save the converted checkpoint if `do_convert_from_hf` is True. Default is None.

        Logs:
            The function logs information about missing or unexpected keys during checkpoint loading.
        """
        if do_convert_from_hf:
            ckpt = flux_transformer_converter(ckpt_path, self.transformer.config)
            if save_converted_model_to is not None:
                save_path = os.path.join(save_converted_model_to, 'nemo_flux_transformer.safetensors')
                save_safetensors(ckpt, save_path)
                logging.info(f'saving converted transformer checkpoint to {save_path}')
        else:
            ckpt = load_safetensors(ckpt_path)
        missing, unexpected = self.transformer.load_state_dict(ckpt, strict=False)
        missing = [k for k in missing if not k.endswith('_extra_state')]
        # These keys are mcore specific and should not affect the model performance
        if len(missing) > 0:
            logging.info(
                f"The following keys are missing during checkpoint loading, "
                f"please check the ckpt provided or the image quality may be compromised.\n {missing}"
            )
            logging.info(f"Found unexepected keys: \n {unexpected}")

    def encoder_prompt(
        self,
        prompt: Union[str, List[str]],
        num_images_per_prompt: int = 1,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = 'cuda',
        dtype: Optional[torch.dtype] = torch.float,
    ):
        """
        Encodes a text prompt (or a batch of prompts) into embeddings using both T5 and CLIP models.

        Args:
            prompt (Union[str, List[str]]):
                The text prompt(s) to be encoded. Can be a string or a list of strings.
            num_images_per_prompt (int, optional):
                The number of images to generate per prompt. Default is 1.
            prompt_embeds (torch.FloatTensor, optional):
                Precomputed prompt embeddings, if available. Default is None.
            pooled_prompt_embeds (torch.FloatTensor, optional):
                Precomputed pooled prompt embeddings, if available. Default is None.
            max_sequence_length (int, optional):
                The maximum sequence length for the text model. Default is 512.
            device (torch.device, optional):
                The device (CPU or CUDA) on which the models are placed. Default is 'cuda'.
            dtype (torch.dtype, optional):
                The data type for tensor operations. Default is `torch.float`.

        Returns:
            Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
                - The prompt embeddings.
                - The pooled prompt embeddings.
                - The text IDs for the prompt.

        Raises:
            ValueError: If neither `prompt` nor `prompt_embeds` are provided.
        """
        if prompt is not None:
            batch_size = len(prompt)
        elif prompt_embeds is not None:
            batch_size = prompt_embeds.shape[0]
        else:
            raise ValueError("Either prompt or prompt_embeds must be provided.")
        if device == 'cuda' and self.t5_encoder.device != device:
            self.t5_encoder.to(device)
        if prompt_embeds is None:
            prompt_embeds = self.t5_encoder(prompt, max_sequence_length=max_sequence_length)
        seq_len = prompt_embeds.shape[1]
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1).to(dtype=dtype)

        if device == 'cuda' and self.clip_encoder.device != device:
            self.clip_encoder.to(device)
        if pooled_prompt_embeds is None:
            _, pooled_prompt_embeds = self.clip_encoder(prompt)

        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt, 1)
        pooled_prompt_embeds = pooled_prompt_embeds.view(batch_size * num_images_per_prompt, -1).to(dtype=dtype)

        dtype = dtype if dtype is not None else self.t5_encoder.dtype
        text_ids = torch.zeros(batch_size, prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)
        text_ids = text_ids.repeat(num_images_per_prompt, 1, 1)

        return prompt_embeds.transpose(0, 1), pooled_prompt_embeds, text_ids

    @staticmethod
    def _prepare_latent_image_ids(batch_size: int, height: int, width: int, device: torch.device, dtype: torch.dtype):
        """
        Prepares latent image IDs for input into the model. These IDs represent the image grid.

        Args:
            batch_size (int): The number of samples in the batch.
            height (int): The height of the image.
            width (int): The width of the image.
            device (torch.device): The device to place the tensor.
            dtype (torch.dtype): The data type for the tensor.

        Returns:
            torch.FloatTensor: A tensor representing the latent image IDs.
        """
        latent_image_ids = torch.zeros(height // 2, width // 2, 3)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height // 2)[:, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width // 2)[None, :]

        latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

        latent_image_ids = latent_image_ids[None, :].repeat(batch_size, 1, 1, 1)
        latent_image_ids = latent_image_ids.reshape(
            batch_size, latent_image_id_height * latent_image_id_width, latent_image_id_channels
        )

        return latent_image_ids.to(device=device, dtype=dtype)

    @staticmethod
    def _pack_latents(latents, batch_size, num_channels_latents, height, width):
        """
        Packs latents into desired shape, e.g. (B, C, H, W) --> (B, (H//2)*(W//2), C * 4).

        Args:
            latents (torch.Tensor): The latents to be packed.
            batch_size (int): The number of samples in the batch.
            num_channels_latents (int): The number of channels in the latents.
            height (int): The height of the image.
            width (int): The width of the image.

        Returns:
            torch.Tensor: The packed latents.
        """
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

        return latents

    @staticmethod
    def _unpack_latents(latents, height, width, vae_scale_factor):
        """
        Unpacks the latents from the model output into an image format suitable for further processing.

        The method reshapes and permutes the latents, adjusting their dimensions according to the
        specified `vae_scale_factor` to match the expected resolution of the image.

        Args:
            latents (torch.Tensor): The latents output from the model, typically in a compact, compressed format.
            height (int): The original height of the image before scaling, used to adjust the latent dimensions.
            width (int): The original width of the image before scaling, used to adjust the latent dimensions.
            vae_scale_factor (int): A scale factor used to adjust the resolution of the image when unpacking.
                This factor istypically the inverse of the VAE downsampling factor.

        Returns:
            torch.Tensor: The unpacked latents reshaped to match the expected dimensions for image reconstruction.
                The output tensor will have shape `(batch_size, channels, height * 2, width * 2)`.

        Notes:
            - This function is intended to convert latents back into a format
            that can be decoded into images by the VAE.
        """
        batch_size, num_patches, channels = latents.shape

        height = height // vae_scale_factor
        width = width // vae_scale_factor

        latents = latents.view(batch_size, height, width, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)

        latents = latents.reshape(batch_size, channels // (2 * 2), height * 2, width * 2)

        return latents

    @staticmethod
    def _calculate_shift(
        image_seq_len,
        base_seq_len: int = 256,
        max_seq_len: int = 4096,
        base_shift: float = 0.5,
        max_shift: float = 1.16,
    ):
        # pylint: disable=C0116
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        mu = image_seq_len * m + b
        return mu

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        """
        Prepares and optionally generates image latents for use in the image generation pipeline.

        This method can either use the provided latents (if already available) or generate new random latents
        using a random generator. The generated latents are then packed and prepared for the model to process.

        Args:
            batch_size (int): The number of samples in the batch.
            num_channels_latents (int): The number of channels in the latents (e.g., depth of the latent tensor).
            height (int): The height of the image to be generated (before scaling).
            width (int): The width of the image to be generated (before scaling).
            dtype (torch.dtype): The data type to use for the latents (e.g., `torch.float32`).
            device (torch.device): The device on which the latents will reside (e.g., 'cuda').
            generator (Union[torch.Generator, List[torch.Generator]]): A random number generator or a list of
                generatorsfor generating random latents. If a list is provided, its length must match the batch size.
            latents (Optional[torch.FloatTensor]): An optional pre-existing latent tensor. If provided, it is used
                instead of generating new latents.

        Returns:
            tuple: A tuple containing:
                - latents (torch.Tensor):
                    The prepared latents, with shape `(batch_size, num_channels_latents, height, width)`.
                - latent_image_ids (torch.Tensor):
                    A tensor containing latent image IDs for each batch sample, used for indexing
                    in the model.

        Raises:
            ValueError: If a list of generators is provided but its length does not match the batch size.

        """
        height = 2 * int(height) // self.vae_scale_factor
        width = 2 * int(width) // self.vae_scale_factor

        shape = (batch_size, num_channels_latents, height, width)

        if latents is not None:
            latent_image_ids = self._prepare_latent_image_ids(batch_size, height, width, device, dtype)
            return latents.to(device=device, dtype=dtype), latent_image_ids

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = FluxInferencePipeline._generate_rand_latents(
            shape, generator=generator, device=device, dtype=dtype, batch_size=batch_size
        )
        latents = self._pack_latents(latents, batch_size, num_channels_latents, height, width)

        latent_image_ids = self._prepare_latent_image_ids(batch_size, height, width, device, dtype)

        return latents.transpose(0, 1), latent_image_ids

    @staticmethod
    def _generate_rand_latents(
        shape,
        generator,
        device,
        dtype,
        batch_size,
    ):
        '''
        Create random latents using a random generator or a list of generators.
        '''
        if isinstance(generator, list):
            shape = (1,) + shape[1:]
            latents = [
                torch.randn(shape, generator=generator[i], device=device, dtype=dtype) for i in range(batch_size)
            ]
            latents = torch.cat(latents, dim=0).to(device=device)
        else:
            latents = torch.randn(shape, generator=generator, device=device, dtype=dtype)

        return latents

    @staticmethod
    def numpy_to_pil(images):
        """
        Convert a numpy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]

        return pil_images

    @staticmethod
    def torch_to_numpy(images):
        '''
        Convert a torch image or a batch of images to a numpy image.
        '''
        numpy_images = images.float().cpu().permute(0, 2, 3, 1).numpy()
        return numpy_images

    @staticmethod
    def denormalize(image):
        # pylint: disable=C0116
        return (image / 2 + 0.5).clamp(0, 1)

    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        num_inference_steps: int = 28,
        timesteps: Optional[List[int]] = None,
        guidance_scale: float = 7.0,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        max_sequence_length: int = 512,
        device: torch.device = 'cuda',
        dtype: torch.dtype = torch.float32,
        save_to_disk: bool = True,
        offload: bool = False,
        output_path: str = None,
    ):
        """
        Generates images based on a given text prompt and various model parameters.
        Optionally saves the images to disk.

        This method orchestrates the process of generating images by embedding the prompt, preparing the latent
        vectors, iterating through timesteps in the diffusion process, and then decoding the latent representation
        back into an image. It supports both the generation of latent representations or final images in a desired
        output format (e.g., PIL image). The images are optionally saved to disk with a unique filename based
        on the prompt.

        Args:
            prompt (Union[str, List[str]]):
                A text prompt or a list of text prompts to guide image generation. Each prompt
                generates one or more images based on the `num_images_per_prompt`.
            height (Optional[int]):
                The height of the output image. Default is 512.
            width (Optional[int]):
                The width of the output image. Default is 512.
            num_inference_steps (int):
                The number of steps for the diffusion process. Default is 28.
            timesteps (Optional[List[int]]):
                A list of specific timesteps for the diffusion process. If not provided,
                they are automatically calculated.
            guidance_scale (float):
                The scale of the guidance signal, typically used to control the strength of prompt conditioning.
            num_images_per_prompt (Optional[int]):
                The number of images to generate per prompt. Default is 1.
            generator (Optional[Union[torch.Generator, List[torch.Generator]]]):
                A random number generator or a list of generators
                for generating latents. If a list is provided, it should match the batch size.
            latents (Optional[torch.FloatTensor]):
                Pre-existing latents to use instead of generating new ones.
            prompt_embeds (Optional[torch.FloatTensor]):
                Optionally pre-computed prompt embeddings to skip the prompt encoding step.
            pooled_prompt_embeds (Optional[torch.FloatTensor]):
                Optionally pre-computed pooled prompt embeddings.
            output_type (Optional[str]):
                The format of the output. Can be "latent" or "pil" (PIL image). Default is "pil".
            max_sequence_length (int):
                The maximum sequence length for tokenizing the prompt. Default is 512.
            device (torch.device):
                The device on which the computation should take place (e.g., 'cuda'). Default is 'cuda'.
            dtype (torch.dtype):
                The data type of the latents and model weights. Default is `torch.float32`.
            save_to_disk (bool):
                Whether or not to save the generated images to disk. Default is True.
            offload (bool):
                Whether or not to offload model components to CPU to free up GPU memory during the process.
                Default is False.

        Returns:
            Union[List[Image.Image], torch.Tensor]:
                The generated images or latents, depending on the `output_type` argument.
                If `output_type` is "pil", a list of PIL images is returned. If "latent", the latents are returned.

        Raises:
            ValueError: If neither a `prompt` nor `prompt_embeds` is provided.

        Notes:
            - The model expects a device of 'cuda'.
              The method will raise an assertion error if a different device is provided.
            - The method handles both prompt-based and pre-embedded prompt input,
              providing flexibility for different usage scenarios.
            - If `save_to_disk` is enabled, images will be saved with a filename derived from the prompt text.
        """
        assert device == 'cuda', 'Transformer blocks in Mcore must run on cuda devices'

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
            prompt = [prompt]
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        elif prompt_embeds is not None and isinstance(prompt_embeds, torch.FloatTensor):
            batch_size = prompt_embeds.shape[0]
        else:
            raise ValueError("Either prompt or prompt_embeds must be provided.")

        # get text prompt embeddings
        prompt_embeds, pooled_prompt_embeds, text_ids = self.encoder_prompt(
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=dtype,
        )
        if offload:
            self.t5_encoder.to('cpu')
            self.clip_encoder.to('cpu')
            torch.cuda.empty_cache()

        # prepare image latents
        num_channels_latents = self.transformer.in_channels // 4
        latents, latent_image_ids = self.prepare_latents(
            batch_size * num_images_per_prompt, num_channels_latents, height, width, dtype, device, generator, latents
        )
        # prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        image_seq_len = latents.shape[0]

        mu = FluxInferencePipeline._calculate_shift(
            image_seq_len,
            self.scheduler.base_image_seq_len,
            self.scheduler.max_image_seq_len,
            self.scheduler.base_shift,
            self.scheduler.max_shift,
        )

        self.scheduler.set_timesteps(sigmas=sigmas, device=device, mu=mu)
        timesteps = self.scheduler.timesteps

        if device == 'cuda' and device != self.device:
            self.transformer.to(device)
        with torch.no_grad():
            for i, t in tqdm(enumerate(timesteps)):
                timestep = t.expand(latents.shape[1]).to(device=latents.device, dtype=latents.dtype)
                if self.transformer.guidance_embed:
                    guidance = torch.tensor([guidance_scale], device=device).expand(latents.shape[1])
                else:
                    guidance = None
                with torch.autocast(device_type='cuda', dtype=latents.dtype):
                    pred = self.transformer(
                        img=latents,
                        txt=prompt_embeds,
                        y=pooled_prompt_embeds,
                        timesteps=timestep / 1000,
                        img_ids=latent_image_ids,
                        txt_ids=text_ids,
                        guidance=guidance,
                    )
                    latents = self.scheduler.step(pred, t, latents)[0]
            if offload:
                self.transformer.to('cpu')
                torch.cuda.empty_cache()

            if output_type == "latent":
                return latents.transpose(0, 1)
            elif output_type == "pil":
                latents = self._unpack_latents(latents.transpose(0, 1), height, width, self.vae_scale_factor)
                if device == 'cuda' and device != self.device:
                    self.vae.to(device)
                with torch.autocast(device_type='cuda', dtype=latents.dtype):
                    image = self.vae.decode(latents)
                if offload:
                    self.vae.to('cpu')
                    torch.cuda.empty_cache()
                image = FluxInferencePipeline.denormalize(image)
                image = FluxInferencePipeline.torch_to_numpy(image)
                image = FluxInferencePipeline.numpy_to_pil(image)
        if save_to_disk:
            print('Saving to disk')
            os.makedirs(output_path, exist_ok=True)
            assert len(image) == int(len(prompt) * num_images_per_prompt)
            prompt = [p[:40] + f'_{idx}' for p in prompt for idx in range(num_images_per_prompt)]
            for file_name, image in zip(prompt, image):
                image.save(os.path.join(output_path, f'{file_name}.png'))

        return image


class FluxControlNetInferencePipeline(FluxInferencePipeline):
    '''
    Flux Contronlnet inference pipeline initializes controlnet component in addition to a normal flux pipeline.
    '''

    def __init__(
        self,
        params: Optional[FluxModelParams] = None,
        contorlnet_config: Optional[FluxControlNetConfig] = None,
        flux: Flux = None,
        vae: AutoEncoder = None,
        t5: FrozenT5Embedder = None,
        clip: FrozenCLIPEmbedder = None,
        scheduler_steps: int = 1000,
        flux_controlnet: FluxControlNet = None,
    ):
        '''
        Same as Flux Inference Pipeline with controlnet object.
        '''
        super().__init__(
            params,
            flux,
            vae,
            t5,
            clip,
            scheduler_steps,
        )
        self.flux_controlnet = FluxControlNet(contorlnet_config) if flux_controlnet is None else flux_controlnet

    def load_from_pretrained(
        self, flux_ckpt_path, controlnet_ckpt_path, do_convert_from_hf=True, save_converted_model_to=None
    ):
        '''
        Converts both flux base model and flux controlnet ckpt into NeMo format.
        '''
        if do_convert_from_hf:
            flux_ckpt = flux_transformer_converter(flux_ckpt_path, self.transformer.config)
            flux_controlnet_ckpt = flux_transformer_converter(controlnet_ckpt_path, self.flux_controlnet.config)

            if save_converted_model_to is not None:
                save_path = os.path.join(save_converted_model_to, 'nemo_flux_transformer.safetensors')
                save_safetensors(flux_ckpt, save_path)
                logging.info(f'saving converted transformer checkpoint to {save_path}')
                save_path = os.path.join(save_converted_model_to, 'nemo_flux_controlnet_transformer.safetensors')
                save_safetensors(flux_controlnet_ckpt, save_path)
                logging.info(f'saving converted transformer checkpoint to {save_path}')
        else:
            flux_ckpt = load_safetensors(flux_ckpt_path)
            flux_controlnet_ckpt = load_safetensors(controlnet_ckpt_path)
        missing, unexpected = self.transformer.load_state_dict(flux_ckpt, strict=False)
        missing = [k for k in missing if not k.endswith('_extra_state')]
        # These keys are mcore specific and should not affect the model performance
        if len(missing) > 0:
            logging.info(
                f"The following keys are missing during flux checkpoint loading, "
                f"please check the ckpt provided or the image quality may be compromised.\n {missing}"
            )
            logging.info(f"Found unexepected keys: \n {unexpected}")

        missing, unexpected = self.flux_controlnet.load_state_dict(flux_controlnet_ckpt, strict=False)
        missing = [k for k in missing if not k.endswith('_extra_state')]
        # These keys are mcore specific and should not affect the model performance
        if len(missing) > 0:
            logging.info(
                f"The following keys are missing during controlnet checkpoint loading, "
                f"please check the ckpt provided or the image quality may be compromised.\n {missing}"
            )
            logging.info(f"Found unexepected keys: \n {unexpected}")

    def pil_to_numpy(self, images):
        '''
        PIL image to numpy array
        '''
        if not isinstance(images, list):
            images = [images]
        images = [np.array(image).astype(np.float32) / 255.0 for image in images]
        images = np.stack(images, axis=0)

        return images

    def numpy_to_pt(self, images: np.ndarray) -> torch.Tensor:
        '''
        Convert numpy image into torch tensors
        '''
        if images.ndim == 3:
            images = images[..., None]

        images = torch.from_numpy(images.transpose(0, 3, 1, 2))
        return images

    def prepare_image(
        self,
        images,
        height,
        width,
        batch_size,
        num_images_per_prompt,
        device,
        dtype,
    ):
        '''
        Preprocess image into torch tensor, also duplicate by batch size.
        '''
        if isinstance(images, torch.Tensor):
            pass
        else:
            orig_height, orig_width = images[0].height, images[0].width
            if height != orig_height or width != orig_width:
                images = [image.resize((width, height), resample=3) for image in images]

            images = self.pil_to_numpy(images)
            images = self.numpy_to_pt(images)
        image_batch_size = images.shape[0]
        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            repeat_by = num_images_per_prompt
        images = images.repeat_interleave(repeat_by, dim=0)

        images = images.to(device=device, dtype=dtype)

        return images

    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        num_inference_steps: int = 28,
        timesteps: Optional[List[int]] = None,
        guidance_scale: float = 7.0,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        max_sequence_length: int = 512,
        device: torch.device = 'cuda',
        dtype: torch.dtype = torch.float32,
        save_to_disk: bool = True,
        offload: bool = False,
        control_guidance_start: float = 0.0,
        control_guidance_end: float = 1.0,
        control_image: Union[Image.Image, torch.FloatTensor] = None,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        output_path: str = None,
    ):
        """
        Generates images based on a given text prompt and optionally incorporates control images and ControlNet for
        guidance.

        This method generates images by embedding the prompt, preparing the latent vectors, iterating through timesteps
        in the diffusion process, and then decoding the latent representation back into an image. The method supports
        control images through ControlNet, where the `control_image` is used to condition the image generation.
        It also allows you to specify custom guidance scales and other parameters. Generated images can be saved
        to disk if requested.

        Args:
            prompt (Union[str, List[str]]):
                A text prompt or a list of text prompts to guide image generation. Each prompt generates one or more
                images based on the `num_images_per_prompt`.
            height (Optional[int]):
                The height of the output image. Default is 512.
            width (Optional[int]):
                The width of the output image. Default is 512.
            num_inference_steps (int):
                The number of steps for the diffusion process. Default is 28.
            timesteps (Optional[List[int]]):
                A list of specific timesteps for the diffusion process. If not provided, they are automatically
                calculated.
            guidance_scale (float):
                The scale of the guidance signal, typically used to control the strength of prompt conditioning.
            num_images_per_prompt (Optional[int]):
                The number of images to generate per prompt. Default is 1.
            generator (Optional[Union[torch.Generator, List[torch.Generator]]]):
                A random number generator or a list of generators for generating latents. If a list is provided,
                it should match the batch size.
            latents (Optional[torch.FloatTensor]):
                Pre-existing latents to use instead of generating new ones.
            prompt_embeds (Optional[torch.FloatTensor]):
                Optionally pre-computed prompt embeddings to skip the prompt encoding step.
            pooled_prompt_embeds (Optional[torch.FloatTensor]):
                Optionally pre-computed pooled prompt embeddings.
            output_type (Optional[str]):
                The format of the output. Can be "latent" or "pil" (PIL image). Default is "pil".
            max_sequence_length (int):
                The maximum sequence length for tokenizing the prompt. Default is 512.
            device (torch.device):
                The device on which the computation should take place (e.g., 'cuda'). Default is 'cuda'.
            dtype (torch.dtype):
                The data type of the latents and model weights. Default is `torch.float32`.
            save_to_disk (bool):
                Whether or not to save the generated images to disk. Default is True.
            offload (bool):
                Whether or not to offload model components to CPU to free up GPU memory during the process.
                Default is False.
            control_guidance_start (float):
                The start point for control guidance to apply during the diffusion process.
            control_guidance_end (float):
                The end point for control guidance to apply during the diffusion process.
            control_image (Union[Image.Image, torch.FloatTensor]):
                The image used for conditioning the generation process via ControlNet.
            controlnet_conditioning_scale (Union[float, List[float]]):
                Scaling factors to control the impact of the control image in the generation process.
                Can be a single value or a list for multiple images.

        Returns:
            Union[List[Image.Image], torch.Tensor]:
                The generated images or latents, depending on the `output_type` argument.
                If `output_type` is "pil", a list of PIL images is returned. If "latent", the latents are returned.

        Raises:
            ValueError: If neither a `prompt` nor `prompt_embeds` is provided.

        Notes:
            - The model expects a device of 'cuda'.
              The method will raise an assertion error if a different device is provided.
            - The method supports conditional image generation using ControlNet, where a `control_image` can guide the
              generation process.
            - If `save_to_disk` is enabled, images will be saved with a filename derived from the prompt text.
        """
        assert device == 'cuda', 'Transformer blocks in Mcore must run on cuda devices'

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
            prompt = [prompt]
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        elif prompt_embeds is not None and isinstance(prompt_embeds, torch.FloatTensor):
            batch_size = prompt_embeds.shape[0]
        else:
            raise ValueError("Either prompt or prompt_embeds must be provided.")

        # get text prompt embeddings
        prompt_embeds, pooled_prompt_embeds, text_ids = self.encoder_prompt(
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=dtype,
        )
        if offload:
            self.t5_encoder.to('cpu')
            self.clip_encoder.to('cpu')
            torch.cuda.empty_cache()

        # prepare image latents
        num_channels_latents = self.transformer.in_channels // 4
        latents, latent_image_ids = self.prepare_latents(
            batch_size * num_images_per_prompt, num_channels_latents, height, width, dtype, device, generator, latents
        )

        # prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        image_seq_len = latents.shape[0]

        mu = FluxInferencePipeline._calculate_shift(
            image_seq_len,
            self.scheduler.base_image_seq_len,
            self.scheduler.max_image_seq_len,
            self.scheduler.base_shift,
            self.scheduler.max_shift,
        )

        self.scheduler.set_timesteps(sigmas=sigmas, device=device, mu=mu)
        timesteps = self.scheduler.timesteps

        control_image = self.prepare_image(
            images=control_image,
            height=height,
            width=width,
            batch_size=batch_size * num_images_per_prompt,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
            dtype=torch.float32,
        )

        height, width = control_image.shape[-2:]
        if self.flux_controlnet.input_hint_block is None:
            if device == 'cuda' and self.device != device:
                self.vae.to(device)
            with torch.no_grad():
                control_image = self.vae.encode(control_image).to(dtype=dtype)

            height_control_image, width_control_image = control_image.shape[2:]
            control_image = self._pack_latents(
                control_image,
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height_control_image,
                width_control_image,
            ).transpose(0, 1)

        controlnet_keep = []
        for i in range(len(timesteps)):
            controlnet_keep.append(
                1.0
                - float(i / len(timesteps) < control_guidance_start or (i + 1) / len(timesteps) > control_guidance_end)
            )
        if device == 'cuda' and device != self.device:
            self.transformer.to(device)
            self.flux_controlnet.to(device)
        with torch.no_grad():
            for i, t in tqdm(enumerate(timesteps)):
                timestep = t.expand(latents.shape[1]).to(device=latents.device, dtype=latents.dtype)
                if self.transformer.guidance_embed:
                    guidance = torch.tensor([guidance_scale], device=device).expand(latents.shape[1])
                else:
                    guidance = None

                conditioning_scale = controlnet_keep[i] * controlnet_conditioning_scale

                with torch.autocast(device_type='cuda', dtype=latents.dtype):
                    controlnet_double_block_samples, controlnet_single_block_samples = self.flux_controlnet(
                        img=latents,
                        controlnet_cond=control_image,
                        txt=prompt_embeds,
                        y=pooled_prompt_embeds,
                        timesteps=timestep / 1000,
                        img_ids=latent_image_ids,
                        txt_ids=text_ids,
                        guidance=guidance,
                        conditioning_scale=conditioning_scale,
                    )
                    pred = self.transformer(
                        img=latents,
                        txt=prompt_embeds,
                        y=pooled_prompt_embeds,
                        timesteps=timestep / 1000,
                        img_ids=latent_image_ids,
                        txt_ids=text_ids,
                        guidance=guidance,
                        controlnet_double_block_samples=controlnet_double_block_samples,
                        controlnet_single_block_samples=controlnet_single_block_samples,
                    )
                    latents = self.scheduler.step(pred, t, latents)[0]
            if offload:
                self.transformer.to('cpu')
                torch.cuda.empty_cache()

            if output_type == "latent":
                return latents.transpose(0, 1)
            elif output_type == "pil":
                latents = self._unpack_latents(latents.transpose(0, 1), height, width, self.vae_scale_factor)
                if device == 'cuda' and device != self.device:
                    self.vae.to(device)
                with torch.autocast(device_type='cuda', dtype=latents.dtype):
                    image = self.vae.decode(latents)
                if offload:
                    self.vae.to('cpu')
                    torch.cuda.empty_cache()
                image = FluxInferencePipeline.denormalize(image)
                image = FluxInferencePipeline.torch_to_numpy(image)
                image = FluxInferencePipeline.numpy_to_pil(image)
        if save_to_disk:
            print('Saving to disk')
            os.makedirs(output_path, exist_ok=True)
            assert len(image) == int(len(prompt) * num_images_per_prompt)
            prompt = [p[:40] + f'_{idx}' for p in prompt for idx in range(num_images_per_prompt)]
            for file_name, image in zip(prompt, image):
                image.save(os.path.join(output_path, f'{file_name}.png'))

        return image
