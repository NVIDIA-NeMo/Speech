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


from typing import Dict, Literal, Optional

import torch
import torch.nn as nn
from diffusers.models.embeddings import Timesteps
from einops import rearrange, repeat
from megatron.core import parallel_state, tensor_parallel
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.models.common.vision_module.vision_module import VisionModule
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.enums import ModelType
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import make_sharded_tensor_for_checkpoint
from torch import Tensor

from nemo.collections.diffusion.models.dit import dit_embeddings
from nemo.collections.diffusion.models.dit.dit_embeddings import ParallelTimestepEmbedding
from nemo.collections.diffusion.models.dit.dit_layer_spec import (
    get_dit_adaln_block_with_transformer_engine_spec as DiTLayerWithAdaLNspec,
)


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class RMSNorm(nn.Module):
    def __init__(self, channel: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(channel))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, spatial_patch_size, temporal_patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size, spatial_patch_size * spatial_patch_size * temporal_patch_size * out_channels, bias=False
        )
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=False))

    def forward(self, x_BT_HW_D, emb_B_D):
        shift_B_D, scale_B_D = self.adaLN_modulation(emb_B_D).chunk(2, dim=1)
        T = x_BT_HW_D.shape[0] // emb_B_D.shape[0]
        shift_BT_D, scale_BT_D = repeat(shift_B_D, "b d -> (b t) d", t=T), repeat(scale_B_D, "b d -> (b t) d", t=T)
        x_BT_HW_D = modulate(self.norm_final(x_BT_HW_D), shift_BT_D, scale_BT_D)
        x_BT_HW_D = self.linear(x_BT_HW_D)
        return x_BT_HW_D


class DiTCrossAttentionModel(VisionModule):
    """
    DiTCrossAttentionModel is a VisionModule that implements a DiT model with a cross-attention block.
    Attributes:
        config (TransformerConfig): Configuration for the transformer.
        pre_process (bool): Whether to apply pre-processing steps.
        post_process (bool): Whether to apply post-processing steps.
        fp16_lm_cross_entropy (bool): Whether to use fp16 for cross-entropy loss.
        parallel_output (bool): Whether to use parallel output.
        position_embedding_type (Literal["learned_absolute", "rope"]): Type of position embedding.
        max_img_h (int): Maximum image height.
        max_img_w (int): Maximum image width.
        max_frames (int): Maximum number of frames.
        patch_spatial (int): Spatial patch size.
        patch_temporal (int): Temporal patch size.
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        transformer_decoder_layer_spec (DiTLayerWithAdaLNspec): Specification for the transformer decoder layer.
        add_encoder (bool): Whether to add an encoder.
        add_decoder (bool): Whether to add a decoder.
        share_embeddings_and_output_weights (bool): Whether to share embeddings and output weights.
        concat_padding_mask (bool): Whether to concatenate padding mask.
        pos_emb_cls (str): Class of position embedding.
        model_type (ModelType): Type of the model.
        decoder (TransformerBlock): Transformer decoder block.
        t_embedder (torch.nn.Sequential): Time embedding layer.
        x_embedder (nn.Conv3d): Convolutional layer for input embedding.
        pos_embedder (dit_embeddings.SinCosPosEmb3D): Position embedding layer.
        final_layer_linear (torch.nn.Linear): Final linear layer.
        affline_norm (RMSNorm): Affine normalization layer.
    Methods:
        forward(x: Tensor, timesteps: Tensor, crossattn_emb: Tensor, packed_seq_params: PackedSeqParams = None, pos_ids: Tensor = None, **kwargs) -> Tensor:
            Forward pass of the model.
        set_input_tensor(input_tensor: Tensor) -> None:
            Sets input tensor to the model.
        sharded_state_dict(prefix: str = 'module.', sharded_offsets: tuple = (), metadata: Optional[Dict] = None) -> ShardedStateDict:
            Sharded state dict implementation for backward-compatibility.
        tie_embeddings_weights_state_dict(tensor, sharded_state_dict: ShardedStateDict, output_layer_weight_key: str, first_stage_word_emb_key: str) -> None:
            Ties the embedding and output weights in a given sharded state dict.
    """

    def __init__(
        self,
        config: TransformerConfig,
        pre_process: bool = True,
        post_process: bool = True,
        fp16_lm_cross_entropy: bool = False,
        parallel_output: bool = True,
        position_embedding_type: Literal["learned_absolute", "rope"] = "rope",
        max_img_h: int = 80,
        max_img_w: int = 80,
        max_frames: int = 34,
        patch_spatial: int = 1,
        patch_temporal: int = 1,
        in_channels: int = 16,
        out_channels: int = 16,
        transformer_decoder_layer_spec=DiTLayerWithAdaLNspec,
        pos_embedder=dit_embeddings.SinCosPosEmb3D,
        vp_stage: Optional[int] = None,
        **kwargs,
    ):
        super(DiTCrossAttentionModel, self).__init__(config=config)

        self.config: TransformerConfig = config

        self.transformer_decoder_layer_spec = transformer_decoder_layer_spec(attn_mask_type=config.attn_mask_type)
        self.pre_process = pre_process
        self.post_process = post_process
        self.add_encoder = True
        self.add_decoder = True
        self.fp16_lm_cross_entropy = fp16_lm_cross_entropy
        self.parallel_output = parallel_output
        self.position_embedding_type = position_embedding_type
        self.share_embeddings_and_output_weights = False
        self.concat_padding_mask = True
        self.pos_emb_cls = 'sincos'
        self.patch_spatial = patch_spatial
        self.patch_temporal = patch_temporal
        self.vp_stage = vp_stage

        # megatron core pipelining currently depends on model type
        # TODO: remove this dependency ?
        self.model_type = ModelType.encoder_or_decoder

        # Transformer decoder
        self.decoder = TransformerBlock(
            config=self.config,
            spec=self.transformer_decoder_layer_spec,
            pre_process=self.pre_process,
            post_process=False,
            post_layer_norm=False,
            vp_stage=vp_stage,
        )

        self.t_embedder = torch.nn.Sequential(
            Timesteps(self.config.hidden_size, flip_sin_to_cos=False, downscale_freq_shift=0),
            dit_embeddings.ParallelTimestepEmbedding(self.config.hidden_size, self.config.hidden_size, seed=1234),
        )

        self.fps_embedder = nn.Sequential(
            Timesteps(num_channels=256, flip_sin_to_cos=False, downscale_freq_shift=1),
            ParallelTimestepEmbedding(256, 256, seed=1234),
        )

        if self.pre_process:
            self.x_embedder = torch.nn.Linear(in_channels * patch_spatial**2, self.config.hidden_size)

        if pos_embedder is dit_embeddings.SinCosPosEmb3D:
            if self.pre_process:
                self.pos_embedder = pos_embedder(
                    config,
                    t=max_frames // patch_temporal,
                    h=max_img_h // patch_spatial,
                    w=max_img_w // patch_spatial,
                )
        else:
            self.pos_embedder = pos_embedder(
                config,
                t=max_frames // patch_temporal,
                h=max_img_h // patch_spatial,
                w=max_img_w // patch_spatial,
                seed=1234,
            )
            if parallel_state.get_pipeline_model_parallel_world_size() > 1:
                for p in self.pos_embedder.parameters():
                    setattr(p, "pipeline_parallel", True)

        if self.post_process:
            self.final_layer_linear = torch.nn.Linear(
                self.config.hidden_size,
                patch_spatial**2 * patch_temporal * out_channels,
            )

        self.affline_norm = RMSNorm(self.config.hidden_size)
        if parallel_state.get_pipeline_model_parallel_world_size() > 1:
            setattr(self.affline_norm.weight, "pipeline_parallel", True)

    def forward(
        self,
        x: Tensor,
        timesteps: Tensor,
        crossattn_emb: Tensor,
        packed_seq_params: PackedSeqParams = None,
        pos_ids: Tensor = None,
        **kwargs,
    ) -> Tensor:
        """Forward pass.

        Args:
            x (Tensor): vae encoded data (b s c)
            encoder_decoder_attn_mask (Tensor): cross-attention mask between encoder and decoder
            inference_params (InferenceParams): relevant arguments for inferencing

        Returns:
            Tensor: loss tensor
        """
        B = x.shape[0]
        fps = kwargs.get(
            'fps',
            torch.tensor(
                [
                    30,
                ]
                * B,
                dtype=torch.bfloat16,
                device=x.device,
            ),
        ).view(-1)
        if self.pre_process:
            # transpose to match
            x_B_S_D = self.x_embedder(x)
            if isinstance(self.pos_embedder, dit_embeddings.SinCosPosEmb3D):
                pos_emb = None
                x_B_S_D += self.pos_embedder(pos_ids)
            else:
                pos_emb = self.pos_embedder(pos_ids)
                pos_emb = rearrange(pos_emb, "B S D -> S B D")
            x_S_B_D = rearrange(x_B_S_D, "B S D -> S B D").contiguous()
        else:
            # intermediate stage of pipeline
            x_S_B_D = None  ### should it take encoder_hidden_states
            if (not hasattr(self, "pos_embedder")) or isinstance(self.pos_embedder, dit_embeddings.SinCosPosEmb3D):
                pos_emb = None
            else:
                # if transformer blocks need pos_emb, then pos_embedder should
                # be replicated across pp ranks.
                pos_emb = rearrange(self.pos_embedder(pos_ids), "B S D -> S B D").contiguous()

        timesteps_B_D = self.t_embedder(timesteps.flatten()).to(torch.bfloat16)  # (b d_text_embedding)

        affline_emb_B_D = timesteps_B_D
        fps_B_D = self.fps_embedder(fps)
        fps_B_D = nn.functional.pad(fps_B_D, (0, self.config.hidden_size - fps_B_D.shape[1]))
        affline_emb_B_D += fps_B_D
        affline_emb_B_D = self.affline_norm(affline_emb_B_D)

        crossattn_emb = rearrange(crossattn_emb, 'B S D -> S B D').contiguous()

        if self.config.sequence_parallel:
            if self.pre_process:
                x_S_B_D = tensor_parallel.scatter_to_sequence_parallel_region(x_S_B_D)
            if hasattr(self, "pos_embedder") and isinstance(
                self.pos_embedder, dit_embeddings.FactorizedLearnable3DEmbedding
            ):
                pos_emb = tensor_parallel.scatter_to_sequence_parallel_region(pos_emb)
            crossattn_emb = tensor_parallel.scatter_to_sequence_parallel_region(crossattn_emb)
            # `scatter_to_sequence_parallel_region` returns a view, which prevents
            # the original tensor from being garbage collected. Clone to facilitate GC.
            # Has a small runtime cost (~0.5%).
            if self.config.clone_scatter_output_in_embedding:
                if self.pre_process:
                    x_S_B_D = x_S_B_D.clone()
                crossattn_emb = crossattn_emb.clone()

        x_S_B_D = self.decoder(
            hidden_states=x_S_B_D,
            attention_mask=affline_emb_B_D,
            context=crossattn_emb,
            context_mask=None,
            rotary_pos_emb=pos_emb,
            packed_seq_params=packed_seq_params,
        )

        if not self.post_process:
            return x_S_B_D

        if self.config.sequence_parallel:
            x_S_B_D = tensor_parallel.gather_from_sequence_parallel_region(x_S_B_D)

        x_S_B_D = self.final_layer_linear(x_S_B_D)
        return rearrange(x_S_B_D, "S B D -> B S D")

    def set_input_tensor(self, input_tensor: Tensor) -> None:
        """Sets input tensor to the model.

        See megatron.model.transformer.set_input_tensor()

        Args:
            input_tensor (Tensor): Sets the input tensor for the model.
        """
        # This is usually handled in schedules.py but some inference code still
        # gives us non-lists or None
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]

        assert len(input_tensor) == 1, 'input_tensor should only be length 1 for gpt/bert'
        self.decoder.set_input_tensor(input_tensor[0])

    def sharded_state_dict(
        self, prefix: str = 'module.', sharded_offsets: tuple = (), metadata: Optional[Dict] = None
    ) -> ShardedStateDict:
        """Sharded state dict implementation for GPTModel backward-compatibility (removing extra state).

        Args:
            prefix (str): Module name prefix.
            sharded_offsets (tuple): PP related offsets, expected to be empty at this module level.
            metadata (Optional[Dict]): metadata controlling sharded state dict creation.

        Returns:
            ShardedStateDict: sharded state dict for the GPTModel
        """
        sharded_state_dict = super().sharded_state_dict(prefix, sharded_offsets, metadata)

        for module in ['t_embedder']:
            for param_name, param in getattr(self, module).named_parameters():
                weight_key = f'{prefix}{module}.{param_name}'
                self._set_embedder_weights_replica_id(param, sharded_state_dict, weight_key)
        return sharded_state_dict

    def _set_embedder_weights_replica_id(
        self, tensor: Tensor, sharded_state_dict: ShardedStateDict, embedder_weight_key: str
    ) -> None:
        """set replica ids of the weights in t_embedder for sharded state dict.

        Args:
            sharded_state_dict (ShardedStateDict): state dict with the weight to tie
            weight_key (str): key of the weight in the state dict.
                This entry will be replaced with a tied version

        Returns: None, acts in-place
        """
        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        vp_stage = self.vp_stage if self.vp_stage is not None else 0
        vp_world = self.config.get("virtual_pipeline_model_parallel_size", 1)
        pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        if embedder_weight_key in sharded_state_dict:
            del sharded_state_dict[embedder_weight_key]
        replica_id = (
            tp_rank,
            (vp_stage + pp_rank * vp_world),
            parallel_state.get_data_parallel_rank(with_context_parallel=True),
        )

        sharded_state_dict[embedder_weight_key] = make_sharded_tensor_for_checkpoint(
            tensor=tensor,
            key=embedder_weight_key,
            replica_id=replica_id,
            allow_shape_mismatch=False,
        )
