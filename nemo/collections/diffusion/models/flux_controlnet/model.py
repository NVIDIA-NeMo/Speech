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

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
from megatron.core import parallel_state
from megatron.core.models.common.vision_module.vision_module import VisionModule
from megatron.core.tensor_parallel.layers import ColumnParallelLinear
from megatron.core.transformer.transformer_config import TransformerConfig
from torch.nn import functional as F

from nemo.collections.diffusion.models.dit.dit_layer_spec import (
    FluxSingleTransformerBlock,
    MMDiTLayer,
    get_flux_double_transformer_engine_spec,
    get_flux_single_transformer_engine_spec,
)
from nemo.collections.diffusion.models.flux.layers import EmbedND, MLPEmbedder, TimeStepEmbedder
from nemo.collections.diffusion.models.flux.model import FluxConfig, FluxModelParams, MegatronFluxModel
from nemo.collections.diffusion.models.flux_controlnet.layers import ControlNetConditioningEmbedding
from nemo.lightning import io
from nemo.utils import logging


def zero_module(module):
    """
    Initializes all parameters of the given module to zero.

    Args:
        module (nn.Module): The module whose parameters will be initialized to zero.

    Returns:
        nn.Module: The same module with zero-initialized parameters.
    """
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


def flux_controlnet_data_step(dataloader_iter):
    """
    Processes a single step of data from a dataloader iterator for the Flux ControlNet.

    Args:
        dataloader_iter (Iterator): An iterator over the dataloader that provides batches of data.

    Returns:
        dict: A processed batch dictionary with an added 'loss_mask' key.
    """
    batch = next(dataloader_iter)
    if isinstance(batch, tuple) and len(batch) == 3:
        _batch = batch[0]
    else:
        _batch = batch

    _batch['loss_mask'] = torch.Tensor([1.0]).cuda(non_blocking=True)
    return _batch


@dataclass
class FluxControlNetConfig(TransformerConfig, io.IOMixin):
    '''
    Flux config inherits from TransformerConfig class.
    '''

    num_layers: int = 1  # dummy setting
    patch_size: int = 1
    in_channels: int = 64
    num_joint_layers: int = 4
    num_single_layers: int = 10
    hidden_size: int = 3072
    num_attention_heads: int = 24
    vec_in_dim: int = 768
    context_dim: int = 4096
    guidance_embed: bool = True
    num_mode: int = None
    model_channels: int = 256
    conditioning_embedding_channels: int = None
    rotary_interleaved: bool = True
    layernorm_epsilon: float = 1e-06
    hidden_dropout: float = 0
    attention_dropout: float = 0
    add_qkv_bias: bool = True
    use_cpu_initialization: bool = True

    load_from_flux_transformer: bool = True
    guidance_scale: float = 3.5

    data_step_fn: Callable = flux_controlnet_data_step


class FluxControlNet(VisionModule):
    """
    A VisionModule-based neural network designed for Flux ControlNet tasks.


    Args:
        config (FluxControlNetConfig):
        Configuration object containing model parameters such as input channels, hidden size, patch size,
            and number of transformer layers.
    """

    def __init__(self, config: FluxControlNetConfig):
        """
        Initializes the FluxControlNet model with embeddings, transformer layers, and optional conditioning blocks.

        Args:
            config (FluxControlNetConfig): Configuration object with model parameters.
        """
        super().__init__(config)
        self.out_channels = config.in_channels
        self.hidden_size = config.hidden_size
        self.patch_size = config.patch_size

        self.pos_embed = EmbedND(dim=self.hidden_size, theta=10000, axes_dim=[16, 56, 56])
        self.img_embed = nn.Linear(config.in_channels, self.hidden_size)
        self.txt_embed = nn.Linear(config.context_dim, self.hidden_size)
        self.timestep_embedding = TimeStepEmbedder(config.model_channels, self.hidden_size)
        self.vector_embedding = MLPEmbedder(in_dim=config.vec_in_dim, hidden_dim=self.hidden_size)
        if config.guidance_embed:
            self.guidance_embedding = (
                MLPEmbedder(in_dim=config.model_channels, hidden_dim=self.hidden_size)
                if config.guidance_embed
                else nn.Identity()
            )
        self.double_blocks = nn.ModuleList(
            [
                MMDiTLayer(
                    config=config,
                    submodules=get_flux_double_transformer_engine_spec().submodules,
                    layer_number=i,
                    context_pre_only=False,
                )
                for i in range(config.num_joint_layers)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                FluxSingleTransformerBlock(
                    config=config,
                    submodules=get_flux_single_transformer_engine_spec().submodules,
                    layer_number=i,
                )
                for i in range(config.num_single_layers)
            ]
        )

        # ContolNet Blocks
        self.controlnet_double_blocks = nn.ModuleList()
        for _ in range(config.num_joint_layers):
            self.controlnet_double_blocks.append(
                zero_module(
                    ColumnParallelLinear(
                        self.hidden_size,
                        self.hidden_size,
                        config=config,
                        init_method=nn.init.normal_,
                        gather_output=True,
                    )
                )
            )

        self.controlnet_single_blocks = nn.ModuleList()
        for _ in range(config.num_single_layers):
            self.controlnet_single_blocks.append(
                zero_module(
                    ColumnParallelLinear(
                        self.hidden_size,
                        self.hidden_size,
                        config=config,
                        init_method=nn.init.normal_,
                        gather_output=True,
                    )
                )
            )

        if config.conditioning_embedding_channels is not None:
            self.input_hint_block = ControlNetConditioningEmbedding(
                conditioning_embedding_channels=config.conditioning_embedding_channels,
                block_out_channels=(16, 16, 16, 16),
            )
            self.controlnet_x_embedder = torch.nn.Linear(config.in_channels, self.hidden_size)
        else:
            self.input_hint_block = None
            self.controlnet_x_embedder = zero_module(torch.nn.Linear(config.in_channels, self.hidden_size))

    def load_from_flux_transformer(self, flux):
        """
        Loads pre-trained weights from a Flux Transformer model into the FluxControlNet.

        Args:
            flux (FluxTransformer): A pre-trained Flux Transformer model.
        """
        logging.info("Loading ControlNet layer weights from Flux...")
        self.pos_embed.load_state_dict(flux.pos_embed.state_dict())
        self.img_embed.load_state_dict(flux.img_embed.state_dict())
        self.txt_embed.load_state_dict(flux.txt_embed.state_dict())
        self.timestep_embedding.load_state_dict(flux.timestep_embedding.state_dict())
        self.vector_embedding.load_state_dict(flux.vector_embedding.state_dict())
        self.double_blocks.load_state_dict(flux.double_blocks.state_dict(), strict=False)
        self.single_blocks.load_state_dict(flux.single_blocks.state_dict(), strict=False)

    def get_fp8_context(self):
        "context manager for fp8 recipe"
        # This is first and last 2 for mamba
        if not self.config.fp8:
            fp8_context = nullcontext()
        else:
            import transformer_engine  # To keep out TE dependency when not training in fp8

            if self.config.fp8 == "e4m3":
                fp8_format = transformer_engine.common.recipe.Format.E4M3
            elif self.config.fp8 == "hybrid":
                fp8_format = transformer_engine.common.recipe.Format.HYBRID
            else:
                raise ValueError("E4M3 and HYBRID are the only supported FP8 formats.")

            fp8_recipe = transformer_engine.common.recipe.DelayedScaling(
                margin=self.config.fp8_margin,
                interval=self.config.fp8_interval,
                fp8_format=fp8_format,
                amax_compute_algo=self.config.fp8_amax_compute_algo,
                amax_history_len=self.config.fp8_amax_history_len,
                override_linear_precision=(False, False, not self.config.fp8_wgrad),
            )
            fp8_group = None
            if parallel_state.model_parallel_is_initialized():
                fp8_group = parallel_state.get_amax_reduction_group(with_context_parallel=True)
            fp8_context = transformer_engine.pytorch.fp8_autocast(
                enabled=True, fp8_recipe=fp8_recipe, fp8_group=fp8_group
            )
        return fp8_context

    def forward(
        self,
        img: torch.Tensor,
        controlnet_cond: torch.Tensor,
        txt: torch.Tensor = None,
        y: torch.Tensor = None,
        timesteps: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        conditioning_scale: float = 1.0,
    ):
        """
        Forward pass for the FluxControlNet model.

        Args:
            img (torch.Tensor): Input image tensor.
            controlnet_cond (torch.Tensor): Conditioning tensor for ControlNet.
            txt (torch.Tensor, optional): Text embedding tensor. Default is None.
            y (torch.Tensor, optional): Vector embedding tensor. Default is None.
            timesteps (torch.LongTensor, optional): Time step tensor. Default is None.
            img_ids (torch.Tensor, optional): Image IDs. Default is None.
            txt_ids (torch.Tensor, optional): Text IDs. Default is None.
            guidance (torch.Tensor, optional): Guidance tensor. Default is None.
            conditioning_scale (float, optional): Scaling factor for conditioning. Default is 1.0.

        Returns:
            torch.Tensor: The output of the forward pass.
        """
        hidden_states = self.img_embed(img)
        encoder_hidden_states = self.txt_embed(txt)
        if self.input_hint_block is not None:
            controlnet_cond = self.input_hint_block(controlnet_cond)
            batch_size, channels, height_pw, width_pw = controlnet_cond.shape
            height = height_pw // self.config.patch_size
            width = width_pw // self.config.patch_size
            controlnet_cond = controlnet_cond.reshape(
                batch_size, channels, height, self.patch_size, width, self.patch_size
            )
            controlnet_cond = controlnet_cond.permute(0, 2, 4, 1, 3, 5)
            controlnet_cond = controlnet_cond.reshape(batch_size, height * width, -1)

        hidden_states = hidden_states + self.controlnet_x_embedder(controlnet_cond)

        timesteps = timesteps.to(img.dtype) * 1000
        vec_emb = self.timestep_embedding(timesteps)
        if guidance is not None:
            vec_emb = vec_emb + self.guidance_embedding(self.timestep_embedding.time_proj(guidance * 1000))
        vec_emb = vec_emb + self.vector_embedding(y)

        ids = torch.cat((txt_ids, img_ids), dim=1)
        rotary_pos_emb = self.pos_embed(ids)

        double_block_samples = ()
        for id_block, block in enumerate(self.double_blocks):
            with self.get_fp8_context():
                hidden_states, encoder_hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    rotary_pos_emb=rotary_pos_emb,
                    emb=vec_emb,
                )
                double_block_samples = double_block_samples + (hidden_states,)

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=0)

        single_block_samples = ()
        for id_block, block in enumerate(self.single_blocks):
            with self.get_fp8_context():
                hidden_states, _ = block(
                    hidden_states=hidden_states,
                    rotary_pos_emb=rotary_pos_emb,
                    emb=vec_emb,
                )
                single_block_samples = single_block_samples + (hidden_states[encoder_hidden_states.shape[0] :, ...],)

        controlnet_double_block_samples = ()
        for double_block_sample, control_block in zip(double_block_samples, self.controlnet_double_blocks):
            double_block_sample, bias = control_block(double_block_sample)
            double_block_sample = double_block_sample + bias if bias else double_block_sample
            controlnet_double_block_samples += (double_block_sample,)

        controlnet_single_block_samples = ()
        for single_block_sample, control_block in zip(single_block_samples, self.controlnet_single_blocks):
            single_block_sample, bias = control_block(single_block_sample)
            single_block_sample = single_block_sample + bias if bias else single_block_sample
            controlnet_single_block_samples += (single_block_sample,)

        controlnet_double_block_samples = [sample * conditioning_scale for sample in controlnet_double_block_samples]
        controlnet_single_block_samples = [sample * conditioning_scale for sample in controlnet_single_block_samples]

        controlnet_double_block_samples = (
            None if len(controlnet_double_block_samples) == 0 else controlnet_double_block_samples
        )
        controlnet_single_block_samples = (
            None if len(controlnet_single_block_samples) == 0 else controlnet_single_block_samples
        )

        return controlnet_double_block_samples, controlnet_single_block_samples


class FluxControlnetForwardWrapper(VisionModule):
    '''
    A wrapper combines flux and flux controlnet forward pass for easier initialization.
    '''

    def __init__(self, flux_config: FluxConfig, flux_controlnet_config: FluxControlNetConfig):
        '''
        Create flux and flux controlnet instances by their config.
        '''
        super().__init__(flux_config)

        self.flux = self.config.configure_model()
        for param in self.flux.parameters():
            param.requires_grad = False

        self.flux_controlnet = FluxControlNet(flux_controlnet_config)
        if flux_controlnet_config.load_from_flux_transformer:
            self.flux_controlnet.load_from_flux_transformer(self.flux)

    def forward(
        self,
        packed_noisy_model_input,
        control_image,
        prompt_embeds,
        pooled_prompt_embeds,
        timesteps,
        latent_image_ids,
        text_ids,
        guidance_vec,
    ):
        '''
        Forward pass for the FluxControlnetForwardWrapper model.
        '''
        # NOTE: This module is wrapped with Fully Sharded Data Parallel (FSDP).
        # To ensure that FSDP can accurately capture the forward and backward
        # processes, it is crucial that the output tensor from this forward
        # function can be used to construct a complete autograd graph.
        controlnet_double_block_samples, controlnet_single_block_samples = self.flux_controlnet(
            img=packed_noisy_model_input,
            controlnet_cond=control_image,
            txt=prompt_embeds,
            y=pooled_prompt_embeds,
            timesteps=timesteps / 1000,
            img_ids=latent_image_ids,
            txt_ids=text_ids,
            guidance=guidance_vec,
        )
        noise_pred = self.flux(
            img=packed_noisy_model_input,
            txt=prompt_embeds,
            y=pooled_prompt_embeds,
            timesteps=timesteps / 1000,
            img_ids=latent_image_ids,
            txt_ids=text_ids,
            guidance=guidance_vec,
            controlnet_double_block_samples=controlnet_double_block_samples,
            controlnet_single_block_samples=controlnet_single_block_samples,
        )

        return noise_pred


class MegatronFluxControlNetModel(MegatronFluxModel):
    """
    Megatron wrapper for flux controlnet model.

    Args:
        flux_params (FluxModelParams): Parameters to configure the Flux model.
        flux_controlnet_config (FluxControlNetConfig): Configuration specific to the FluxControlNet.

    Methods:
        configure_model:
            Configures the model by wrapping the FluxControlNet with the appropriate layers and settings,
            configuring the VAE, scheduler, and text encoders.
        data_step:
            A wrapper around the data-step function specific to FluxControlNet, controlling how data is processed.
        forward:
            Executes a forward pass through FluxControlNet.
        training_step:
            A wrapper step method that calls forward_step with a data batch from data loader.
        forward_step:
            Handles the forward pass specific to training, computing the model's output.
        validation_step:
            Calls inference pipeline with current model weights and save inference result together with the control
            image.
    """

    def __init__(self, flux_params: FluxModelParams, flux_controlnet_config: FluxControlNetConfig):
        super().__init__(flux_params)
        self.flux_controlnet_config = flux_controlnet_config
        self.optim.connect(self)

    def configure_model(self):
        '''
        Initialize flux and controlnet modules, vae, scheduler, and text encoders with given configs.
        '''
        if not hasattr(self, "module"):
            self.module = FluxControlnetForwardWrapper(self.config, self.flux_controlnet_config)

            self.configure_vae(self.vae_config)
            self.configure_scheduler()
            self.configure_text_encoders(self.clip_params, self.t5_params)
            # Have to disable requiring grads for those params not getting one, otherwise custom fsdp fails at assert
            # when there is no single layer, encoder_hidden_states related params are not included in computation graph
            for name, param in self.module.named_parameters():
                if self.flux_controlnet_config.num_single_layers == 0:
                    if 'context' in name or 'added' in name:
                        param.requires_grad = False
                # When getting rid of concat, the projection bias in attention and mlp bias are identical
                # So this bias is skipped and not included in the computation graph
                if 'single_blocks' in name and 'self_attention.linear_proj.bias' in name:
                    param.requires_grad = False

    def data_step(self, dataloader_iter):
        '''
        Retrive data batch from dataloader iterator and do necessary processing before feeding into train steps.
        '''
        return self.flux_controlnet_config.data_step_fn(dataloader_iter)

    def forward(self, *args, **kwargs):
        '''
        Calling the controlnet forward pass.
        '''
        # FSDP module -> Bfloat16 module -> ForwardWrapper
        forward_wrapper = self.module.module.module
        return forward_wrapper(*args, **kwargs)

    def training_step(self, batch, batch_idx=None) -> torch.Tensor:
        '''
        A wrapper method takes data batch and returns the results of forward_step.
        '''
        # In mcore the loss-function is part of the forward-pass (when labels are provided)
        return self.forward_step(batch)

    def forward_step(self, batch) -> torch.Tensor:
        '''
        The main forward step function.
        '''
        if self.optim.config.bf16:
            self.autocast_dtype = torch.bfloat16
        elif self.optim.config.fp16:
            self.autocast_dtype = torch.float
        else:
            self.autocast_dtype = torch.float32

        if self.image_precached:
            latents = batch['latents'].cuda(non_blocking=True)
            control_latents = batch['control_latents'].cuda(non_blocking=True)
        else:
            img = batch['images'].cuda(non_blocking=True)
            latents = self.vae.encode(img).to(dtype=self.autocast_dtype)
            hint = batch['hint'].cuda(non_blocking=True)
            control_latents = self.vae.encode(hint).to(dtype=self.autocast_dtype)

        latent_image_ids = self._prepare_latent_image_ids(
            batch_size=latents.shape[0],
            height=latents.shape[2],
            width=latents.shape[3],
            device=latents.device,
            dtype=self.autocast_dtype,
        )

        latents = self._pack_latents(
            latents,
            batch_size=latents.shape[0],
            num_channels_latents=latents.shape[1],
            height=latents.shape[2],
            width=latents.shape[3],
        )

        control_image = self._pack_latents(
            control_latents,
            batch_size=control_latents.shape[0],
            num_channels_latents=control_latents.shape[1],
            height=control_latents.shape[2],
            width=control_latents.shape[3],
        ).transpose(0, 1)

        batch_size = latents.shape[0]
        noise = torch.randn_like(latents, device=latents.device, dtype=latents.dtype)
        u = self.compute_density_for_timestep_sampling(
            "logit_normal",
            batch_size,
        )
        indices = (u * self.scheduler.num_train_timesteps).long()
        timesteps = self.scheduler.timesteps[indices].to(device=latents.device)

        sigmas = self.scheduler.sigmas.to(device=latents.device, dtype=latents.dtype)
        schduler_timesteps = self.scheduler.timesteps.to(device=latents.device)
        step_indices = [(schduler_timesteps == t).nonzero().item() for t in timesteps]
        timesteps = timesteps.to(dtype=latents.dtype)
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < latents.ndim:
            sigma = sigma.unsqueeze(-1)
        packed_noisy_model_input = (1.0 - sigma) * latents + sigma * noise
        packed_noisy_model_input = packed_noisy_model_input.transpose(0, 1)

        if self.config.guidance_embed:
            guidance_vec = torch.full(
                (packed_noisy_model_input.shape[1],),
                self.config.guidance_scale,
                device=latents.device,
                dtype=latents.dtype,
            )
        else:
            guidance_vec = None
        if self.text_precached:
            prompt_embeds = batch['prompt_embeds'].cuda(non_blocking=True).transpose(0, 1)
            pooled_prompt_embeds = batch['pooled_prompt_embeds'].cuda(non_blocking=True)
            text_ids = batch['text_ids'].cuda(non_blocking=True)
        else:
            txt = batch['txt']
            prompt_embeds, pooled_prompt_embeds, text_ids = self.encode_prompt(
                txt, device=latents.device, dtype=latents.dtype
            )

        with torch.cuda.amp.autocast(
            self.autocast_dtype in (torch.half, torch.bfloat16),
            dtype=self.autocast_dtype,
        ):
            noise_pred = self.forward(
                packed_noisy_model_input=packed_noisy_model_input,
                control_image=control_image,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                timesteps=timesteps,
                latent_image_ids=latent_image_ids,
                text_ids=text_ids,
                guidance_vec=guidance_vec,
            )

            target = (noise - latents).transpose(0, 1)
            loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")
            return loss

    def validation_step(self, batch, batch_idx=None):
        '''
        Initialize flux controlnet pipeline with current model components.

        Saves the inference results together with the hint image to log folder.
        '''
        logging.info("Start validation step")
        from nemo.collections.diffusion.models.flux.pipeline import FluxControlNetInferencePipeline

        pipe = FluxControlNetInferencePipeline(
            params=self.params,
            contorlnet_config=self.flux_controlnet_config,
            flux=self.module.module.module.flux,
            vae=self.vae,
            t5=self.t5,
            clip=self.clip,
            scheduler_steps=self.params.scheduler_steps,
            flux_controlnet=self.module.module.module.flux_controlnet,
        )

        if self.image_precached and self.text_precached:
            latents = batch['latents'].cuda(non_blocking=True)
            control_latents = batch['control_latents'].cuda(non_blocking=True)
            prompt_embeds = batch['prompt_embeds'].cuda(non_blocking=True).transpose(0, 1)
            pooled_prompt_embeds = batch['pooled_prompt_embeds'].cuda(non_blocking=True)
            log_images = pipe(
                latents=latents,
                control_image=control_latents,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                height=latents.shape[2] * self.vae_scale_factor,
                width=latents.shape[3] * self.vae_scale_factor,
                num_inference_steps=30,
                num_images_per_prompt=1,
                guidance_scale=7.0,
                dtype=self.autocast_dtype,
                save_to_disk=False,
            )
            log_images[0].save(f"{self.logger.log_dir}/step={self.global_step}_rank{self.local_rank}.png")
        else:
            img = batch['images'].cuda(non_blocking=True)
            hint = batch['hint'].cuda(non_blocking=True)
            text = batch['txt']
            log_images = pipe(
                text,
                control_image=hint,
                num_inference_steps=30,
                num_images_per_prompt=1,
                height=img.shape[2],
                width=img.shape[3],
                guidance_scale=7.0,
                dtype=self.autocast_dtype,
                save_to_disk=False,
            )
            log_images[0].save(f"{self.logger.log_dir}/step={self.global_step}_rank{self.local_rank}_{text}.png")
            hint = pipe.torch_to_numpy(hint)
            hint = pipe.numpy_to_pil(hint)
            hint[0].save(f"{self.logger.log_dir}/step={self.global_step}_rank{self.local_rank}_control.png")
        return torch.tensor([0.0], device=torch.cuda.current_device())
