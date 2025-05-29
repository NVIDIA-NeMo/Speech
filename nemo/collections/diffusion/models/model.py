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
# pylint: skip-file

import importlib
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from einops import rearrange
from megatron.core import parallel_state
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_config import TransformerConfig
from torch import nn
from typing_extensions import override

from nemo.collections.diffusion.models.dit_llama.dit_llama_model import DiTLlamaModel
from nemo.collections.diffusion.sampler.edm.edm_pipeline import EDMPipeline
from nemo.collections.llm.gpt.model.base import GPTModel
from nemo.lightning import io
from nemo.lightning.megatron_parallel import MaskedTokenLossReduction, MegatronLossReduction
from nemo.lightning.pytorch.optim import OptimizerModule

from .dit.dit_model import DiTCrossAttentionModel


def dit_forward_step(model, batch) -> torch.Tensor:
    """Forward pass of DiT."""
    return model(**batch)


def dit_data_step(module, dataloader_iter):
    """DiT data batch preparation."""
    batch = next(dataloader_iter)[0]
    batch = get_batch_on_this_cp_rank(batch)
    batch = {k: v.to(device='cuda', non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}

    cu_seqlens = batch['seq_len_q'].cumsum(dim=0).to(torch.int32)
    zero = torch.zeros(1, dtype=torch.int32, device="cuda")
    cu_seqlens = torch.cat((zero, cu_seqlens))

    cu_seqlens_kv = batch['seq_len_kv'].cumsum(dim=0).to(torch.int32)
    cu_seqlens_kv = torch.cat((zero, cu_seqlens_kv))

    batch['packed_seq_params'] = {
        'self_attention': PackedSeqParams(
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            qkv_format=module.qkv_format,
        ),
        'cross_attention': PackedSeqParams(
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens_kv,
            qkv_format=module.qkv_format,
        ),
    }

    return batch


def get_batch_on_this_cp_rank(data: Dict):
    """Split the data for context parallelism."""
    from megatron.core import mpu

    cp_size = mpu.get_context_parallel_world_size()
    cp_rank = mpu.get_context_parallel_rank()

    if cp_size > 1:
        num_valid_tokens_in_ub = None
        if 'loss_mask' in data and data['loss_mask'] is not None:
            num_valid_tokens_in_ub = data['loss_mask'].sum()

        for key, value in data.items():
            if (value is not None) and (key in ['video', 'video_latent', 'noise_latent', 'pos_ids']):
                if len(value.shape) > 5:
                    value = value.squeeze(0)
                if len(value.shape) == 5:
                    B, C, T, H, W = value.shape
                    data[key] = value.view(B, C, cp_size, T // cp_size, H, W)[:, :, cp_rank, ...].contiguous()
                else:
                    B, S, D = value.shape
                    data[key] = value.view(B, cp_size, S // cp_size, D)[:, cp_rank, ...].contiguous()
                # TODO: sequence packing
        loss_mask = data["loss_mask"]
        data["loss_mask"] = loss_mask.view(loss_mask.shape[0], cp_size, loss_mask.shape[1] // cp_size)[
            :, cp_rank, ...
        ].contiguous()
        data['num_valid_tokens_in_ub'] = num_valid_tokens_in_ub
    return data


@dataclass
class DiTConfig(TransformerConfig, io.IOMixin):
    """
    Config for DiT-S model
    """

    crossattn_emb_size: int = 1024
    add_bias_linear: bool = False
    gated_linear_unit: bool = False

    num_layers: int = 12
    hidden_size: int = 384
    max_img_h: int = 80
    max_img_w: int = 80
    max_frames: int = 34
    patch_spatial: int = 2
    num_attention_heads: int = 6
    layernorm_epsilon = 1e-6
    normalization = "RMSNorm"
    add_bias_linear = False
    qk_layernorm_per_head = True
    layernorm_zero_centered_gamma = False

    fp16_lm_cross_entropy: bool = False
    parallel_output: bool = True
    share_embeddings_and_output_weights: bool = True

    # max_position_embeddings: int = 5400
    hidden_dropout: float = 0
    attention_dropout: float = 0

    bf16: bool = True
    params_dtype: torch.dtype = torch.bfloat16

    vae_module: str = 'nemo.collections.diffusion.vae.diffusers_vae.AutoencoderKLVAE'
    vae_path: str = None
    sigma_data: float = 0.5

    in_channels: int = 16

    data_step_fn = dit_data_step
    forward_step_fn = dit_forward_step

    replicated_t_embedder = True

    seq_length: int = 2048

    qkv_format: str = 'sbhd'
    attn_mask_type: AttnMaskType = AttnMaskType.no_mask

    @override
    def configure_model(self, tokenizer=None, vp_stage: Optional[int] = None) -> DiTCrossAttentionModel:
        """Configure DiT Model from MCore."""
        vp_size = self.virtual_pipeline_model_parallel_size
        if vp_size:
            p_size = self.pipeline_model_parallel_size
            assert (
                self.num_layers // p_size
            ) % vp_size == 0, "Make sure the number of model chunks is the same across all pipeline stages."

        if isinstance(self, DiTLlama30BConfig):
            model = DiTLlamaModel
        else:
            model = DiTCrossAttentionModel

        # During fake lightning initialization, pass 0 to bypass the assertion that vp_stage must be
        # non-None when using virtual pipeline model parallelism
        vp_stage = vp_stage or 0
        return model(
            self,
            fp16_lm_cross_entropy=self.fp16_lm_cross_entropy,
            parallel_output=self.parallel_output,
            pre_process=parallel_state.is_pipeline_first_stage(ignore_virtual=False, vp_stage=vp_stage),
            post_process=parallel_state.is_pipeline_last_stage(ignore_virtual=False, vp_stage=vp_stage),
            max_img_h=self.max_img_h,
            max_img_w=self.max_img_w,
            max_frames=self.max_frames,
            patch_spatial=self.patch_spatial,
            vp_stage=vp_stage,
        )

    def configure_vae(self):
        """Dynamically import video tokenizer."""
        return dynamic_import(self.vae_module)(self.vae_path)


@dataclass
class DiTBConfig(DiTConfig):
    """DiT-B"""

    num_layers: int = 12
    hidden_size: int = 768
    num_attention_heads: int = 12


@dataclass
class DiTLConfig(DiTConfig):
    """DiT-L"""

    num_layers: int = 24
    hidden_size: int = 1024
    num_attention_heads: int = 16


@dataclass
class DiTXLConfig(DiTConfig):
    """DiT-XL"""

    num_layers: int = 28
    hidden_size: int = 1152
    num_attention_heads: int = 16


@dataclass
class DiT7BConfig(DiTConfig):
    """DiT-7B"""

    num_layers: int = 32
    hidden_size: int = 3072
    num_attention_heads: int = 24


@dataclass
class DiTLlama30BConfig(DiTConfig):
    """MovieGen 30B"""

    num_layers: int = 48
    hidden_size: int = 6144
    ffn_hidden_size: int = 16384
    num_attention_heads: int = 48
    num_query_groups: int = 8
    gated_linear_unit: int = True
    bias_activation_fusion: int = True
    activation_func: Callable = F.silu
    normalization: str = "RMSNorm"
    layernorm_epsilon: float = 1e-5
    max_frames: int = 128
    max_img_h: int = 240
    max_img_w: int = 240
    patch_spatial: int = 2

    init_method_std: float = 0.01
    add_bias_linear: bool = False
    seq_length: int = 256

    bias_activation_fusion: bool = True
    masked_softmax_fusion: bool = True
    persist_layer_norm: bool = True
    bias_dropout_fusion: bool = True


@dataclass
class DiTLlama5BConfig(DiTLlama30BConfig):
    """MovieGen 5B"""

    num_layers: int = 32
    hidden_size: int = 3072
    ffn_hidden_size: int = 8192
    num_attention_heads: int = 24


@dataclass
class DiTLlama1BConfig(DiTLlama30BConfig):
    """MovieGen 1B"""

    num_layers: int = 16
    hidden_size: int = 2048
    ffn_hidden_size: int = 8192
    num_attention_heads: int = 32


@dataclass
class ECDiTLlama1BConfig(DiTLlama1BConfig):
    "EC-DiT 1B"
    moe_router_load_balancing_type: str = 'expert_choice'
    moe_token_dispatcher_type: str = 'alltoall'
    moe_grouped_gemm: bool = True
    moe_expert_capacity_factor: float = 8
    moe_pad_expert_input_to_capacity: bool = True
    moe_router_topk: int = 1
    num_moe_experts: int = 64
    ffn_hidden_size: int = 1024


class DiTModel(GPTModel):
    """
    Diffusion Transformer Model
    """

    def __init__(
        self,
        config: Optional[DiTConfig] = None,
        optim: Optional[OptimizerModule] = None,
        model_transform: Optional[Callable[[nn.Module], nn.Module]] = None,
        tokenizer: Optional[Any] = None,
    ):
        super().__init__(config or DiTConfig(), optim=optim, model_transform=model_transform)

        self.vae = None

        self._training_loss_reduction = None
        self._validation_loss_reduction = None

        self.diffusion_pipeline = EDMPipeline(net=self, sigma_data=self.config.sigma_data)

        self._noise_generator = None
        self.seed = 42

        self.vae = None

    def load_state_dict(self, state_dict, strict=False):
        self.module.load_state_dict(state_dict, strict=False)

    def data_step(self, dataloader_iter) -> Dict[str, Any]:
        return self.config.data_step_fn(dataloader_iter)

    def forward(self, *args, **kwargs):
        return self.module.forward(*args, **kwargs)

    def forward_step(self, batch) -> torch.Tensor:
        if parallel_state.is_pipeline_last_stage(ignore_virtual=False, vp_stage=self.vp_stage):
            output_batch, loss = self.diffusion_pipeline.training_step(batch, 0)
            loss = torch.mean(loss, dim=-1)
            return loss
        else:
            output_tensor = self.diffusion_pipeline.training_step(batch, 0)
            return output_tensor

    def training_step(self, batch, batch_idx=None) -> torch.Tensor:
        # In mcore the loss-function is part of the forward-pass (when labels are provided)
        return self.forward_step(batch)

    def on_validation_start(self):
        if self.vae is None:
            if self.config.vae_path is None:
                warnings.warn('vae_path not specified skipping validation')
                return None
            self.vae = self.config.configure_vae()
        self.vae.to('cuda')

    def on_validation_end(self):
        """Move video tokenizer to CPU after validation."""
        if self.vae is not None:
            self.vae.to('cpu')

    def validation_step(self, batch, batch_idx=None) -> torch.Tensor:
        """Generated validation video sample and logs to wandb."""
        # In mcore the loss-function is part of the forward-pass (when labels are provided)
        state_shape = batch['video'].shape
        sample = self.diffusion_pipeline.generate_samples_from_batch(
            batch,
            guidance=7,
            state_shape=state_shape,
            num_steps=35,
            is_negative_prompt=True if 'neg_t5_text_embeddings' in batch else False,
        )

        # TODO visualize more than 1 sample
        sample = sample[0, None]
        C, T, H, W = batch['latent_shape'][0]
        seq_len_q = batch['seq_len_q'][0]

        sample = rearrange(
            sample[0, None, :seq_len_q],
            'B (T H W) (ph pw pt C) -> B C (T pt) (H ph) (W pw)',
            ph=self.config.patch_spatial,
            pw=self.config.patch_spatial,
            C=C,
            T=T,
            H=H // self.config.patch_spatial,
            W=W // self.config.patch_spatial,
        )

        video = (1.0 + self.vae.decode(sample / self.config.sigma_data)).clamp(0, 2) / 2  # [B, 3, T, H, W]

        video = (video * 255).to(torch.uint8).cpu().numpy().astype(np.uint8)

        result = rearrange(video, 'b c t h w -> (b t) c h w')

        # wandb is on the last rank for megatron, first rank for nemo
        wandb_rank = 0

        if parallel_state.get_data_parallel_src_rank() == wandb_rank:
            if torch.distributed.get_rank() == wandb_rank:
                gather_list = [None for _ in range(parallel_state.get_data_parallel_world_size())]
            else:
                gather_list = None
            torch.distributed.gather_object(
                result, gather_list, wandb_rank, group=parallel_state.get_data_parallel_group()
            )
            if gather_list is not None:
                videos = []
                for video in gather_list:
                    try:
                        videos.append(wandb.Video(video, fps=24, format='mp4'))
                    except Exception as e:
                        warnings.warn(f'Error saving video as mp4: {e}')
                        videos.append(wandb.Video(video, fps=24))
                wandb.log({'prediction': videos})

        return None

    @property
    def training_loss_reduction(self) -> MaskedTokenLossReduction:
        if not self._training_loss_reduction:
            self._training_loss_reduction = MaskedTokenLossReduction()

        return self._training_loss_reduction

    @property
    def validation_loss_reduction(self) -> MaskedTokenLossReduction:
        if not self._validation_loss_reduction:
            self._validation_loss_reduction = DummyLossReduction()

        return self._validation_loss_reduction

    def on_validation_model_zero_grad(self) -> None:
        '''
        Small hack to avoid first validation on resume.
        This will NOT work if the gradient accumulation step should be performed at this point.
        https://github.com/Lightning-AI/pytorch-lightning/discussions/18110
        '''
        super().on_validation_model_zero_grad()
        if self.trainer.ckpt_path is not None and getattr(self, '_restarting_skip_val_flag', True):
            self.trainer.sanity_checking = True
            self._restarting_skip_val_flag = False


class DummyLossReduction(MegatronLossReduction):
    """
    Diffusion Loss Reduction
    """

    def __init__(self, validation_step: bool = False, val_drop_last: bool = True) -> None:
        super().__init__()
        self.validation_step = validation_step
        self.val_drop_last = val_drop_last

    def forward(
        self, batch: Dict[str, torch.Tensor], forward_out: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        return torch.tensor(0.0, device=torch.cuda.current_device()), {
            "avg": torch.tensor(0.0, device=torch.cuda.current_device())
        }

    def reduce(self, losses_reduced_per_micro_batch) -> torch.Tensor:
        return torch.tensor(0.0, device=torch.cuda.current_device())


def dynamic_import(full_path):
    """
    Dynamically import a class or function from a given full path.

    :param full_path: The full path to the class or function (e.g., "package.module.ClassName")
    :return: The imported class or function
    :raises ImportError: If the module or attribute cannot be imported
    :raises AttributeError: If the attribute does not exist in the module
    """
    try:
        # Split the full path into module path and attribute name
        module_path, attribute_name = full_path.rsplit('.', 1)
    except ValueError as e:
        raise ImportError(
            f"Invalid full path '{full_path}'. It should contain both module and attribute names."
        ) from e

    # Import the module
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(f"Cannot import module '{module_path}'.") from e

    # Retrieve the attribute from the module
    try:
        attribute = getattr(module, attribute_name)
    except AttributeError as e:
        raise AttributeError(f"Module '{module_path}' does not have an attribute '{attribute_name}'.") from e

    return attribute
