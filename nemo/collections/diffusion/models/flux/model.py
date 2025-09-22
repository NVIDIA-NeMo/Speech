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

import math
import os
from contextlib import nullcontext
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Callable, Optional

import lightning.pytorch as L
import numpy as np
import torch
from megatron.core import parallel_state
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.dist_checkpointing.utils import replace_prefix_for_sharding
from megatron.core.models.common.vision_module.vision_module import VisionModule
from megatron.core.optimizer import OptimizerConfig
from megatron.core.transformer.enums import ModelType
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import openai_gelu, sharded_state_dict_default
from safetensors.torch import load_file as load_safetensors
from safetensors.torch import save_file as save_safetensors
from torch import nn
from torch.nn import functional as F

from nemo.collections.diffusion.encoders.conditioner import FrozenCLIPEmbedder, FrozenT5Embedder
from nemo.collections.diffusion.models.dit.dit_layer_spec import (
    AdaLNContinuous,
    FluxSingleTransformerBlock,
    MMDiTLayer,
    get_flux_double_transformer_engine_spec,
    get_flux_single_transformer_engine_spec,
)
from nemo.collections.diffusion.models.flux.layers import EmbedND, MLPEmbedder, TimeStepEmbedder
from nemo.collections.diffusion.sampler.flow_matching.flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from nemo.collections.diffusion.utils.flux_ckpt_converter import (
    _import_qkv,
    _import_qkv_bias,
    flux_transformer_converter,
)
from nemo.collections.diffusion.vae.autoencoder import AutoEncoder, AutoEncoderConfig
from nemo.collections.llm import fn
from nemo.lightning import io, teardown
from nemo.lightning.megatron_parallel import MaskedTokenLossReduction
from nemo.lightning.pytorch.optim import MegatronOptimizerModule, OptimizerModule
from nemo.utils import logging


# pylint: disable=C0116
def flux_data_step(dataloader_iter):
    batch = next(dataloader_iter)
    if isinstance(batch, tuple) and len(batch) == 3:
        _batch = batch[0]
    else:
        _batch = batch

    _batch['loss_mask'] = torch.ones(1, device="cuda")
    return _batch


@dataclass
class FluxConfig(TransformerConfig, io.IOMixin):
    """
    transformer related Flux Config
    """

    num_layers: int = 1  # dummy setting
    num_joint_layers: int = 19
    num_single_layers: int = 38
    hidden_size: int = 3072
    num_attention_heads: int = 24
    activation_func: Callable = openai_gelu
    add_qkv_bias: bool = True
    in_channels: int = 64
    context_dim: int = 4096
    model_channels: int = 256
    patch_size: int = 1
    guidance_embed: bool = False
    vec_in_dim: int = 768
    rotary_interleaved: bool = True
    apply_rope_fusion: bool = False
    layernorm_epsilon: float = 1e-06
    hidden_dropout: float = 0
    attention_dropout: float = 0
    use_cpu_initialization: bool = True
    gradient_accumulation_fusion: bool = False
    enable_cuda_graph: bool = False
    cuda_graph_scope: Optional[str] = None  # full, full_iteration
    use_te_rng_tracker: bool = False
    cuda_graph_warmup_steps: int = 2

    guidance_scale: float = 3.5
    data_step_fn: Callable = flux_data_step
    ckpt_path: Optional[str] = None
    load_dist_ckpt: bool = False
    do_convert_from_hf: bool = False
    save_converted_model_to = None

    def configure_model(self):
        model = Flux(config=self)
        return model


@dataclass
class T5Config:
    """
    T5 Config
    """

    version: Optional[str] = field(default_factory=lambda: "google/t5-v1_1-xxl")
    max_length: Optional[int] = field(default_factory=lambda: 512)
    load_config_only: bool = False


@dataclass
class ClipConfig:
    """
    Clip Config
    """

    version: Optional[str] = field(default_factory=lambda: "openai/clip-vit-large-patch14")
    max_length: Optional[int] = field(default_factory=lambda: 77)
    always_return_pooled: Optional[bool] = field(default_factory=lambda: True)


@dataclass
class FluxModelParams:
    """
    Flux Model Params
    """

    flux_config: FluxConfig = field(default_factory=FluxConfig)
    vae_config: AutoEncoderConfig = field(
        default_factory=lambda: AutoEncoderConfig(ch_mult=[1, 2, 4, 4], attn_resolutions=[])
    )
    clip_params: ClipConfig = field(default_factory=ClipConfig)
    t5_params: T5Config = field(default_factory=T5Config)

    scheduler_steps: int = 1000
    device: str = 'cuda'


# pylint: disable=C0116
class Flux(VisionModule):
    """
    NeMo implementation of Flux model, with flux transformer and single flux transformer blocks implemented with
    Megatron Core.

    Args:
        config (FluxConfig): Configuration object containing the necessary parameters for setting up the model,
                              such as the number of channels, hidden size, attention heads, and more.

    Attributes:
        out_channels (int): The number of output channels for the model.
        hidden_size (int): The size of the hidden layers.
        num_attention_heads (int): The number of attention heads for the transformer.
        patch_size (int): The size of the image patches.
        in_channels (int): The number of input channels for the image.
        guidance_embed (bool): A flag to indicate if guidance embedding should be used.
        pos_embed (EmbedND): Position embedding layer for the model.
        img_embed (nn.Linear): Linear layer to embed image input into the hidden space.
        txt_embed (nn.Linear): Linear layer to embed text input into the hidden space.
        timestep_embedding (TimeStepEmbedder): Embedding layer for timesteps, used in generative models.
        vector_embedding (MLPEmbedder): MLP embedding for vector inputs.
        guidance_embedding (nn.Module or nn.Identity): Optional MLP embedding for guidance, or identity if not used.
        double_blocks (nn.ModuleList): A list of transformer blocks for the double block layers.
        single_blocks (nn.ModuleList): A list of transformer blocks for the single block layers.
        norm_out (AdaLNContinuous): Normalization layer for the output.
        proj_out (nn.Linear): Final linear layer for output projection.

    Methods:
        forward: Performs a forward pass through the network, processing images, text, timesteps, and guidance.
        load_from_pretrained: Loads model weights from a pretrained checkpoint, with optional support for distribution
                              and conversion from Hugging Face format.

    """

    def __init__(self, config: FluxConfig):
        # pylint: disable=C0116
        super().__init__(config)
        self.out_channels = config.in_channels
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.patch_size = config.patch_size
        self.in_channels = config.in_channels
        self.guidance_embed = config.guidance_embed

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

        self.norm_out = AdaLNContinuous(config=config, conditioning_embedding_dim=self.hidden_size)
        self.proj_out = nn.Linear(self.hidden_size, self.patch_size * self.patch_size * self.out_channels, bias=True)
        if self.config.ckpt_path is not None:
            self.load_from_pretrained(
                self.config.ckpt_path,
                do_convert_from_hf=self.config.do_convert_from_hf,
                load_dist_ckpt=self.config.load_dist_ckpt,
                save_converted_model_to=self.config.save_converted_model_to,
            )

    def get_fp8_context(self):
        # This is first and last 2 for mamba
        if not self.config.fp8:
            fp8_context = nullcontext()
        else:
            # To keep out TE dependency when not training in fp8
            from transformer_engine.common.recipe import (
                DelayedScaling,
                Float8BlockScaling,
                Float8CurrentScaling,
                Format,
                MXFP8BlockScaling,
            )
            from transformer_engine.pytorch import fp8_autocast

            if self.config.fp8 == "e4m3":
                fp8_format = Format.E4M3
            elif self.config.fp8 == "hybrid":
                fp8_format = Format.HYBRID
            else:
                raise ValueError("E4M3 and HYBRID are the only supported FP8 formats.")

            # Defaults to delayed scaling for backward compatibility
            if not self.config.fp8_recipe:
                self.config.fp8_recipe = "delayed"

            if self.config.fp8_recipe == "delayed":
                fp8_recipe = DelayedScaling(
                    margin=self.config.fp8_margin,
                    interval=self.config.fp8_interval,
                    fp8_format=fp8_format,
                    amax_compute_algo=self.config.fp8_amax_compute_algo,
                    amax_history_len=self.config.fp8_amax_history_len,
                    override_linear_precision=(False, False, not self.config.fp8_wgrad),
                )
            elif self.config.fp8_recipe == "current":
                fp8_recipe = Float8CurrentScaling(fp8_format=fp8_format)
            elif self.config.fp8_recipe == "block":
                fp8_recipe = Float8BlockScaling(fp8_format=fp8_format)
            elif self.config.fp8_recipe == "mxfp8":
                fp8_recipe = MXFP8BlockScaling(fp8_format=fp8_format)
            else:
                raise ValueError(f"Unsupported FP8 recipe: {self.config.fp8_recipe}")

            fp8_group = None
            if parallel_state.model_parallel_is_initialized():
                fp8_group = parallel_state.get_amax_reduction_group(with_context_parallel=True)
            fp8_context = fp8_autocast(enabled=True, fp8_recipe=fp8_recipe, fp8_group=fp8_group)
        return fp8_context

    def forward(
        self,
        img: torch.Tensor,
        txt: torch.Tensor = None,
        y: torch.Tensor = None,
        timesteps: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        controlnet_double_block_samples: torch.Tensor = None,
        controlnet_single_block_samples: torch.Tensor = None,
    ):
        """
        Forward pass through the model, processing image, text, and additional inputs like guidance and timesteps.

        Args:
            img (torch.Tensor):
                The image input tensor.
            txt (torch.Tensor, optional):
                The text input tensor (default is None).
            y (torch.Tensor, optional):
                The vector input for embedding (default is None).
            timesteps (torch.LongTensor, optional):
                The timestep input, typically used in generative models (default is None).
            img_ids (torch.Tensor, optional):
                Image IDs for positional encoding (default is None).
            txt_ids (torch.Tensor, optional):
                Text IDs for positional encoding (default is None).
            guidance (torch.Tensor, optional):
                Guidance input for conditioning (default is None).
            controlnet_double_block_samples (torch.Tensor, optional):
                Optional controlnet samples for double blocks (default is None).
            controlnet_single_block_samples (torch.Tensor, optional):
                Optional controlnet samples for single blocks (default is None).

        Returns:
            torch.Tensor: The final output tensor from the model after processing all inputs.
        """
        hidden_states = self.img_embed(img)
        encoder_hidden_states = self.txt_embed(txt)

        timesteps = timesteps.to(img.dtype) * 1000
        vec_emb = self.timestep_embedding(timesteps)

        if guidance is not None:
            vec_emb = vec_emb + self.guidance_embedding(self.timestep_embedding.time_proj(guidance * 1000))
        vec_emb = vec_emb + self.vector_embedding(y)

        ids = torch.cat((txt_ids, img_ids), dim=1)
        rotary_pos_emb = self.pos_embed(ids)
        for id_block, block in enumerate(self.double_blocks):
            with self.get_fp8_context():
                hidden_states, encoder_hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    rotary_pos_emb=rotary_pos_emb,
                    emb=vec_emb,
                )

                if controlnet_double_block_samples is not None:
                    interval_control = len(self.double_blocks) / len(controlnet_double_block_samples)
                    interval_control = int(np.ceil(interval_control))
                    hidden_states = hidden_states + controlnet_double_block_samples[id_block // interval_control]

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=0)

        for id_block, block in enumerate(self.single_blocks):
            with self.get_fp8_context():
                hidden_states, _ = block(
                    hidden_states=hidden_states,
                    rotary_pos_emb=rotary_pos_emb,
                    emb=vec_emb,
                )

                if controlnet_single_block_samples is not None:
                    interval_control = len(self.single_blocks) / len(controlnet_single_block_samples)
                    interval_control = int(np.ceil(interval_control))
                    hidden_states = torch.cat(
                        [
                            hidden_states[: encoder_hidden_states.shape[0]],
                            hidden_states[encoder_hidden_states.shape[0] :]
                            + controlnet_single_block_samples[id_block // interval_control],
                        ]
                    )

        hidden_states = hidden_states[encoder_hidden_states.shape[0] :, ...]

        hidden_states = self.norm_out(hidden_states, vec_emb)
        output = self.proj_out(hidden_states)

        return output

    def load_from_pretrained(
        self, ckpt_path, do_convert_from_hf=False, save_converted_model_to=None, load_dist_ckpt=False
    ):
        # pylint: disable=C0116
        if load_dist_ckpt:
            from megatron.core import dist_checkpointing

            sharded_sd_metadata = dist_checkpointing.load_content_metadata(ckpt_path)
            sharded_state_dict = dict(
                state_dict=self.sharded_state_dict(prefix="module.", metadata=sharded_sd_metadata)
            )
            loaded_state_dict = dist_checkpointing.load(
                sharded_state_dict=sharded_state_dict, checkpoint_dir=ckpt_path
            )
            ckpt = {k.removeprefix("module."): v for k, v in loaded_state_dict["state_dict"].items()}
        else:
            if do_convert_from_hf:
                ckpt = flux_transformer_converter(ckpt_path, self.config)
                if save_converted_model_to is not None:
                    os.makedirs(save_converted_model_to, exist_ok=True)
                    save_path = os.path.join(save_converted_model_to, 'nemo_flux_transformer.safetensors')
                    save_safetensors(ckpt, save_path)
                    logging.info(f'saving converted transformer checkpoint to {save_path}')
            else:
                ckpt = load_safetensors(ckpt_path)
        missing, unexpected = self.load_state_dict(ckpt, strict=False)
        missing = [k for k in missing if not k.endswith('_extra_state')]
        # These keys are mcore specific and should not affect the model performance
        if len(missing) > 0:
            logging.info(
                f"The following keys are missing during checkpoint loading, "
                f"please check the ckpt provided or the image quality may be compromised.\n {missing}"
            )
            logging.info(f"Found unexepected keys: \n {unexpected}")
        logging.info(f"Restored flux model weights from {ckpt_path}")

    def sharded_state_dict(self, prefix='', sharded_offsets: tuple = (), metadata: dict = None) -> ShardedStateDict:
        sharded_state_dict = {}
        layer_prefix = f'{prefix}double_blocks.'
        for layer in self.double_blocks:
            offset = layer._get_layer_offset(self.config)

            global_layer_offset = layer.layer_number
            state_dict_prefix = f'{layer_prefix}{global_layer_offset - offset}.'
            sharded_prefix = f'{layer_prefix}{global_layer_offset}.'
            sharded_pp_offset = []

            layer_sharded_state_dict = layer.sharded_state_dict(state_dict_prefix, sharded_pp_offset, metadata)
            replace_prefix_for_sharding(layer_sharded_state_dict, state_dict_prefix, sharded_prefix)

            sharded_state_dict.update(layer_sharded_state_dict)

        layer_prefix = f'{prefix}single_blocks.'
        for layer in self.single_blocks:
            offset = layer._get_layer_offset(self.config)

            global_layer_offset = layer.layer_number
            state_dict_prefix = f'{layer_prefix}{global_layer_offset - offset}.'
            sharded_prefix = f'{layer_prefix}{global_layer_offset}.'
            sharded_pp_offset = []

            layer_sharded_state_dict = layer.sharded_state_dict(state_dict_prefix, sharded_pp_offset, metadata)
            replace_prefix_for_sharding(layer_sharded_state_dict, state_dict_prefix, sharded_prefix)

            sharded_state_dict.update(layer_sharded_state_dict)

        for name, module in self.named_children():
            if not (module is self.single_blocks or module is self.double_blocks):
                sharded_state_dict.update(
                    sharded_state_dict_default(module, f'{prefix}{name}.', sharded_offsets, metadata)
                )
        return sharded_state_dict


class MegatronFluxModel(L.LightningModule, io.IOMixin, io.ConnectorMixin, fn.FNMixin):
    '''
    Megatron wrapper for flux.

    Args:
        flux_params (FluxModelParams): Parameters to configure the Flux model.
    '''

    def __init__(
        self,
        flux_params: FluxModelParams,
        optim: Optional[OptimizerModule] = None,
        seed: int = 42,
    ):
        # pylint: disable=C0116
        self.params = flux_params
        self.config = flux_params.flux_config
        super().__init__()
        self._training_loss_reduction = None
        self._validation_loss_reduction = None

        self.vae_config = self.params.vae_config
        self.clip_params = self.params.clip_params
        self.t5_params = self.params.t5_params
        self.optim = optim or MegatronOptimizerModule(config=OptimizerConfig(lr=1e-4, use_distributed_optimizer=False))
        self.optim.connect(self)
        self.model_type = ModelType.encoder_or_decoder
        self.text_precached = self.t5_params is None or self.clip_params is None
        self.image_precached = self.vae_config is None
        self.seed = seed

    def setup(self, stage: str):
        super().setup(stage)
        torch.manual_seed(self.seed + 100 * parallel_state.get_data_parallel_rank())

    def configure_model(self):
        # pylint: disable=C0116
        if not hasattr(self, "module"):
            self.module = self.config.configure_model()
        self.configure_vae(self.vae_config)
        self.configure_scheduler()
        self.configure_text_encoders(self.clip_params, self.t5_params)
        for name, param in self.module.named_parameters():
            if self.config.num_single_layers == 0:
                if 'context' in name or 'added' in name:
                    param.requires_grad = False
            # When getting rid of concat, the projection bias in attention and mlp bias are identical
            # So this bias is skipped and not included in the computation graph
            if 'single_blocks' in name and 'self_attention.linear_proj.bias' in name:
                param.requires_grad = False

    def configure_scheduler(self):
        # pylint: disable=C0116
        self.scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=self.params.scheduler_steps,
        )

    def configure_vae(self, vae):
        # pylint: disable=C0116
        if isinstance(vae, nn.Module):
            self.vae = vae.eval().cuda()
            self.vae_scale_factor = 2 ** (len(self.vae.params.ch_mult) - 1)
            self.vae_channels = self.vae.params.z_channels
            for param in self.vae.parameters():
                param.requires_grad = False
        elif isinstance(vae, AutoEncoderConfig):
            self.vae = AutoEncoder(vae).eval().cuda()
            self.vae_scale_factor = 2 ** (len(vae.ch_mult) - 1)
            self.vae_channels = vae.z_channels
            for param in self.vae.parameters():
                param.requires_grad = False
        else:
            logging.info("Vae not provided, assuming the image input is precached...")
            self.vae = None
            self.vae_scale_factor = 8
            self.vae_channels = 16

    def configure_text_encoders(self, clip, t5):
        # pylint: disable=C0116
        if isinstance(clip, nn.Module):
            self.clip = clip
        elif isinstance(clip, ClipConfig):
            self.clip = FrozenCLIPEmbedder(
                version=self.clip_params.version,
                max_length=self.clip_params.max_length,
                always_return_pooled=self.clip_params.always_return_pooled,
                device=torch.cuda.current_device(),
            )
        else:
            logging.info("CLIP encoder not provided, assuming the text embeddings is precached...")
            self.clip = None

        if isinstance(t5, nn.Module):
            self.t5 = t5
        elif isinstance(t5, T5Config):
            self.t5 = FrozenT5Embedder(
                self.t5_params.version,
                max_length=self.t5_params.max_length,
                device=torch.cuda.current_device(),
                load_config_only=self.t5_params.load_config_only,
            )
        else:
            logging.info("T5 encoder not provided, assuming the text embeddings is precached...")
            self.t5 = None

    # pylint: disable=C0116
    def data_step(self, dataloader_iter):
        return self.config.data_step_fn(dataloader_iter)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def training_step(self, batch, batch_idx=None) -> torch.Tensor:
        # In mcore the loss-function is part of the forward-pass (when labels are provided)
        return self.forward_step(batch)

    def validation_step(self, batch, batch_idx=None) -> torch.Tensor:
        # In mcore the loss-function is part of the forward-pass (when labels are provided)

        return self.forward_step(batch)

    # pylint: disable=C0116
    def forward_step(self, batch) -> torch.Tensor:
        # pylint: disable=C0116
        if self.optim.config.bf16:
            self.autocast_dtype = torch.bfloat16
        elif self.optim.config.fp16:
            self.autocast_dtype = torch.float
        else:
            self.autocast_dtype = torch.float32

        if self.image_precached:
            latents = batch['latents'].cuda(non_blocking=True)
        else:
            img = batch['images'].cuda(non_blocking=True)
            latents = self.vae.encode(img).to(dtype=self.autocast_dtype)
        latents, noise, packed_noisy_model_input, latent_image_ids, guidance_vec, timesteps = (
            self.prepare_image_latent(latents)
        )
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
                img=packed_noisy_model_input,
                txt=prompt_embeds,
                y=pooled_prompt_embeds,
                timesteps=timesteps / 1000,
                img_ids=latent_image_ids,
                txt_ids=text_ids,
                guidance=guidance_vec,
            )

            noise_pred = self._unpack_latents(
                noise_pred.transpose(0, 1),
                latents.shape[2],
                latents.shape[3],
            ).transpose(0, 1)

            target = noise - latents
            loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")
        return loss

    def encode_prompt(self, prompt, device='cuda', dtype=torch.float32):
        # pylint: disable=C0116
        prompt_embeds = self.t5(prompt).transpose(0, 1)
        _, pooled_prompt_embeds = self.clip(prompt)
        text_ids = torch.zeros(prompt_embeds.shape[1], prompt_embeds.shape[0], 3).to(device=device, dtype=dtype)

        return prompt_embeds, pooled_prompt_embeds.to(dtype=dtype), text_ids

    def compute_density_for_timestep_sampling(
        self,
        weighting_scheme: str,
        batch_size: int,
        logit_mean: float = 0.0,
        logit_std: float = 1.0,
        mode_scale: float = 1.29,
    ):
        """
        Compute the density for sampling the timesteps when doing SD3 training.

        Courtesy: This was contributed by Rafie Walker in https://github.com/huggingface/diffusers/pull/8528.

        SD3 paper reference: https://arxiv.org/abs/2403.03206v1.
        """
        if weighting_scheme == "logit_normal":
            # See 3.1 in the SD3 paper ($rf/lognorm(0.00,1.00)$).
            u = torch.normal(mean=logit_mean, std=logit_std, size=(batch_size,), device="cpu")
            u = torch.nn.functional.sigmoid(u)
        elif weighting_scheme == "mode":
            u = torch.rand(size=(batch_size,), device="cpu")
            u = 1 - u - mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
        else:
            u = torch.rand(size=(batch_size,), device="cpu")
        return u

    def prepare_image_latent(self, latents):
        # pylint: disable=C0116
        latent_image_ids = self._prepare_latent_image_ids(
            latents.shape[0],
            latents.shape[2],
            latents.shape[3],
            latents.device,
            latents.dtype,
        )

        noise = torch.randn_like(latents, device=latents.device, dtype=latents.dtype)
        batch_size = latents.shape[0]
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

        noisy_model_input = (1.0 - sigma) * latents + sigma * noise
        packed_noisy_model_input = self._pack_latents(
            noisy_model_input,
            batch_size=latents.shape[0],
            num_channels_latents=latents.shape[1],
            height=latents.shape[2],
            width=latents.shape[3],
        )

        if self.config.guidance_embed:
            guidance_vec = torch.full(
                (noisy_model_input.shape[0],),
                self.config.guidance_scale,
                device=latents.device,
                dtype=latents.dtype,
            )
        else:
            guidance_vec = None

        return (
            latents.transpose(0, 1),
            noise.transpose(0, 1),
            packed_noisy_model_input.transpose(0, 1),
            latent_image_ids,
            guidance_vec,
            timesteps,
        )

    def _unpack_latents(self, latents, height, width):
        # pylint: disable=C0116
        batch_size, num_patches, channels = latents.shape

        # adjust h and w for patching
        latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)

        latents = latents.reshape(batch_size, channels // 4, height, width)

        return latents

    @lru_cache
    def _prepare_latent_image_ids(
        self, batch_size: int, height: int, width: int, device: torch.device, dtype: torch.dtype
    ):
        # pylint: disable=C0116
        latent_image_ids = torch.zeros(height // 2, width // 2, 3)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height // 2)[:, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width // 2)[None, :]

        latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

        latent_image_ids = latent_image_ids[None, :].repeat(batch_size, 1, 1, 1)
        latent_image_ids = latent_image_ids.reshape(
            batch_size, latent_image_id_height * latent_image_id_width, latent_image_id_channels
        )

        return latent_image_ids.to(device=device, dtype=dtype, non_blocking=True)

    def _pack_latents(self, latents, batch_size, num_channels_latents, height, width):
        # pylint: disable=C0116
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

        return latents

    def set_input_tensor(self, tensor):
        # pylint: disable=C0116
        pass

    @property
    def training_loss_reduction(self) -> MaskedTokenLossReduction:
        # pylint: disable=C0116
        if not self._training_loss_reduction:
            self._training_loss_reduction = MaskedTokenLossReduction()

        return self._training_loss_reduction

    @property
    def validation_loss_reduction(self) -> MaskedTokenLossReduction:
        # pylint: disable=C0116
        # pylint: disable=C0116
        if not self._validation_loss_reduction:
            self._validation_loss_reduction = MaskedTokenLossReduction(validation_step=True)

        return self._validation_loss_reduction


@io.model_importer(MegatronFluxModel, "hf")
class HFFluxImporter(io.ModelConnector["FluxTransformer2DModel", MegatronFluxModel]):
    '''
    Convert a HF ckpt into NeMo dist-ckpt compatible format.
    '''

    # pylint: disable=C0116
    def init(self) -> MegatronFluxModel:
        return MegatronFluxModel(self.config)

    def apply(self, output_path: Path) -> Path:
        from diffusers import FluxTransformer2DModel

        source = FluxTransformer2DModel.from_pretrained(str(self), subfolder="transformer")
        target = self.init()
        trainer = self.nemo_setup(target)
        self.convert_state(source, target)
        print(f"Converted flux transformer to Nemo, saving to {output_path}")

        self.nemo_save(output_path, trainer)

        print(f"Converted flux transformer saved to {output_path}")

        teardown(trainer, target)

        return output_path

    @property
    def config(self) -> FluxConfig:
        from diffusers import FluxTransformer2DModel

        source = FluxTransformer2DModel.from_pretrained(str(self), subfolder="transformer")
        source_config = source.config
        flux_config = FluxConfig(
            num_layers=1,  # dummy setting
            num_joint_layers=source_config.num_layers,
            num_single_layers=source_config.num_single_layers,
            hidden_size=source_config.num_attention_heads * source_config.attention_head_dim,
            num_attention_heads=source_config.num_attention_heads,
            activation_func=openai_gelu,
            add_qkv_bias=True,
            in_channels=source_config.in_channels,
            context_dim=source_config.joint_attention_dim,
            model_channels=256,
            patch_size=source_config.patch_size,
            guidance_embed=source_config.guidance_embeds,
            vec_in_dim=source_config.pooled_projection_dim,
            rotary_interleaved=True,
            layernorm_epsilon=1e-06,
            hidden_dropout=0,
            attention_dropout=0,
            use_cpu_initialization=True,
        )

        output = FluxModelParams(
            flux_config=flux_config,
            vae_config=None,
            clip_params=None,
            t5_params=None,
            scheduler_steps=1000,
            device='cuda',
        )
        return output

    def convert_state(self, source, target):
        # pylint: disable=C0301
        mapping = {
            'transformer_blocks.*.norm1.linear.weight': 'double_blocks.*.adaln.adaLN_modulation.1.weight',
            'transformer_blocks.*.norm1.linear.bias': 'double_blocks.*.adaln.adaLN_modulation.1.bias',
            'transformer_blocks.*.norm1_context.linear.weight': 'double_blocks.*.adaln_context.adaLN_modulation.1.weight',
            'transformer_blocks.*.norm1_context.linear.bias': 'double_blocks.*.adaln_context.adaLN_modulation.1.bias',
            'transformer_blocks.*.attn.norm_q.weight': 'double_blocks.*.self_attention.q_layernorm.weight',
            'transformer_blocks.*.attn.norm_k.weight': 'double_blocks.*.self_attention.k_layernorm.weight',
            'transformer_blocks.*.attn.norm_added_q.weight': 'double_blocks.*.self_attention.added_q_layernorm.weight',
            'transformer_blocks.*.attn.norm_added_k.weight': 'double_blocks.*.self_attention.added_k_layernorm.weight',
            'transformer_blocks.*.attn.to_out.0.weight': 'double_blocks.*.self_attention.linear_proj.weight',
            'transformer_blocks.*.attn.to_out.0.bias': 'double_blocks.*.self_attention.linear_proj.bias',
            'transformer_blocks.*.attn.to_add_out.weight': 'double_blocks.*.self_attention.added_linear_proj.weight',
            'transformer_blocks.*.attn.to_add_out.bias': 'double_blocks.*.self_attention.added_linear_proj.bias',
            'transformer_blocks.*.ff.net.0.proj.weight': 'double_blocks.*.mlp.linear_fc1.weight',
            'transformer_blocks.*.ff.net.0.proj.bias': 'double_blocks.*.mlp.linear_fc1.bias',
            'transformer_blocks.*.ff.net.2.weight': 'double_blocks.*.mlp.linear_fc2.weight',
            'transformer_blocks.*.ff.net.2.bias': 'double_blocks.*.mlp.linear_fc2.bias',
            'transformer_blocks.*.ff_context.net.0.proj.weight': 'double_blocks.*.context_mlp.linear_fc1.weight',
            'transformer_blocks.*.ff_context.net.0.proj.bias': 'double_blocks.*.context_mlp.linear_fc1.bias',
            'transformer_blocks.*.ff_context.net.2.weight': 'double_blocks.*.context_mlp.linear_fc2.weight',
            'transformer_blocks.*.ff_context.net.2.bias': 'double_blocks.*.context_mlp.linear_fc2.bias',
            'single_transformer_blocks.*.norm.linear.weight': 'single_blocks.*.adaln.adaLN_modulation.1.weight',
            'single_transformer_blocks.*.norm.linear.bias': 'single_blocks.*.adaln.adaLN_modulation.1.bias',
            'single_transformer_blocks.*.proj_mlp.weight': 'single_blocks.*.mlp.linear_fc1.weight',
            'single_transformer_blocks.*.proj_mlp.bias': 'single_blocks.*.mlp.linear_fc1.bias',
            'single_transformer_blocks.*.attn.norm_q.weight': 'single_blocks.*.self_attention.q_layernorm.weight',
            'single_transformer_blocks.*.attn.norm_k.weight': 'single_blocks.*.self_attention.k_layernorm.weight',
            'single_transformer_blocks.*.proj_out.bias': 'single_blocks.*.mlp.linear_fc2.bias',
            'norm_out.linear.bias': 'norm_out.adaLN_modulation.1.bias',
            'norm_out.linear.weight': 'norm_out.adaLN_modulation.1.weight',
            'proj_out.bias': 'proj_out.bias',
            'proj_out.weight': 'proj_out.weight',
            'time_text_embed.guidance_embedder.linear_1.bias': 'guidance_embedding.in_layer.bias',
            'time_text_embed.guidance_embedder.linear_1.weight': 'guidance_embedding.in_layer.weight',
            'time_text_embed.guidance_embedder.linear_2.bias': 'guidance_embedding.out_layer.bias',
            'time_text_embed.guidance_embedder.linear_2.weight': 'guidance_embedding.out_layer.weight',
            'x_embedder.bias': 'img_embed.bias',
            'x_embedder.weight': 'img_embed.weight',
            'time_text_embed.timestep_embedder.linear_1.bias': 'timestep_embedding.time_embedder.in_layer.bias',
            'time_text_embed.timestep_embedder.linear_1.weight': 'timestep_embedding.time_embedder.in_layer.weight',
            'time_text_embed.timestep_embedder.linear_2.bias': 'timestep_embedding.time_embedder.out_layer.bias',
            'time_text_embed.timestep_embedder.linear_2.weight': 'timestep_embedding.time_embedder.out_layer.weight',
            'context_embedder.bias': 'txt_embed.bias',
            'context_embedder.weight': 'txt_embed.weight',
            'time_text_embed.text_embedder.linear_1.bias': 'vector_embedding.in_layer.bias',
            'time_text_embed.text_embedder.linear_1.weight': 'vector_embedding.in_layer.weight',
            'time_text_embed.text_embedder.linear_2.bias': 'vector_embedding.out_layer.bias',
            'time_text_embed.text_embedder.linear_2.weight': 'vector_embedding.out_layer.weight',
        }
        return io.apply_transforms(
            source,
            target,
            mapping=mapping,
            transforms=[
                import_double_block_qkv,
                import_double_block_qkv_bias,
                import_added_qkv,
                import_added_qkv_bias,
                import_single_block_qkv,
                import_single_block_qkv_bias,
                transform_single_proj_out,
            ],
        )


@io.state_transform(
    source_key=(
        "transformer_blocks.*.attn.to_q.weight",
        "transformer_blocks.*.attn.to_k.weight",
        "transformer_blocks.*.attn.to_v.weight",
    ),
    target_key=("double_blocks.*.self_attention.linear_qkv.weight"),
)
def import_double_block_qkv(ctx: io.TransformCTX, q, k, v):
    transformer_config = ctx.target.config
    return _import_qkv(transformer_config, q, k, v)


@io.state_transform(
    source_key=(
        "transformer_blocks.*.attn.to_q.bias",
        "transformer_blocks.*.attn.to_k.bias",
        "transformer_blocks.*.attn.to_v.bias",
    ),
    target_key=("double_blocks.*.self_attention.linear_qkv.bias"),
)
def import_double_block_qkv_bias(ctx: io.TransformCTX, qb, kb, vb):
    transformer_config = ctx.target.config
    return _import_qkv_bias(transformer_config, qb, kb, vb)


@io.state_transform(
    source_key=(
        "transformer_blocks.*.attn.add_q_proj.weight",
        "transformer_blocks.*.attn.add_k_proj.weight",
        "transformer_blocks.*.attn.add_v_proj.weight",
    ),
    target_key=("double_blocks.*.self_attention.added_linear_qkv.weight"),
)
def import_added_qkv(ctx: io.TransformCTX, q, k, v):
    transformer_config = ctx.target.config
    return _import_qkv(transformer_config, q, k, v)


@io.state_transform(
    source_key=(
        "transformer_blocks.*.attn.add_q_proj.bias",
        "transformer_blocks.*.attn.add_k_proj.bias",
        "transformer_blocks.*.attn.add_v_proj.bias",
    ),
    target_key=("double_blocks.*.self_attention.added_linear_qkv.bias"),
)
def import_added_qkv_bias(ctx: io.TransformCTX, qb, kb, vb):
    transformer_config = ctx.target.config
    return _import_qkv_bias(transformer_config, qb, kb, vb)


@io.state_transform(
    source_key=(
        "single_transformer_blocks.*.attn.to_q.weight",
        "single_transformer_blocks.*.attn.to_k.weight",
        "single_transformer_blocks.*.attn.to_v.weight",
    ),
    target_key=("single_blocks.*.self_attention.linear_qkv.weight"),
)
def import_single_block_qkv(ctx: io.TransformCTX, q, k, v):
    transformer_config = ctx.target.config
    return _import_qkv(transformer_config, q, k, v)


@io.state_transform(
    source_key=(
        "single_transformer_blocks.*.attn.to_q.bias",
        "single_transformer_blocks.*.attn.to_k.bias",
        "single_transformer_blocks.*.attn.to_v.bias",
    ),
    target_key=("single_blocks.*.self_attention.linear_qkv.bias"),
)
def import_single_block_qkv_bias(ctx: io.TransformCTX, qb, kb, vb):
    transformer_config = ctx.target.config
    return _import_qkv_bias(transformer_config, qb, kb, vb)


@io.state_transform(
    source_key=('single_transformer_blocks.*.proj_out.weight'),
    target_key=('single_blocks.*.mlp.linear_fc2.weight', 'single_blocks.*.self_attention.linear_proj.weight'),
)
def transform_single_proj_out(proj_weight):
    linear_fc2 = proj_weight.detach()[:, 3072:].clone()
    linear_proj = proj_weight.detach()[:, :3072].clone()
    return linear_fc2, linear_proj


# pylint: disable=C0116
