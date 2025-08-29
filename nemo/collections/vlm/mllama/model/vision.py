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

import collections
import copy
import types
from contextlib import nullcontext
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from megatron.core import InferenceParams, parallel_state, tensor_parallel
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules

try:
    from megatron.core.transformer.custom_layers.transformer_engine import (
        TEColumnParallelLinear,
        TEDotProductAttention,
        TERowParallelLinear,
    )
except ImportError:
    from nemo.utils import logging

    # These Defaults are needed to make sure the code compiles
    TEColumnParallelLinear = None
    TENorm = None
    TERowParallelLinear = None
    logging.warning(
        "Failed to import Transformer Engine dependencies. "
        "`from megatron.core.transformer.custom_layers.transformer_engine import *`"
        "If using NeMo Run, this is expected. Otherwise, please verify the Transformer Engine installation."
    )

from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from megatron.core.utils import make_viewless_tensor
from torch import Tensor, nn

if TYPE_CHECKING:
    from nemo.collections.vlm import CrossAttentionVisionConfig

try:
    from megatron.core.transformer.custom_layers.transformer_engine import TEDelayedScaling, TENorm

    HAVE_TE = True
    LayerNormImpl = TENorm
except ImportError:
    from megatron.core.transformer.torch_layer_norm import WrappedTorchLayerNorm

    HAVE_TE = False
    LayerNormImpl = WrappedTorchLayerNorm


def to_2tuple(x):
    """
    Convert an input to a 2-tuple.
    """
    if isinstance(x, collections.abc.Iterable):
        return x
    return (x, x)


def build_encoder_attention_mask(
    x: torch.Tensor, ar_ids: torch.Tensor, ntok: int, num_chunks: int, supported_aspect_ratios: List[List[int]]
):
    """
    Build attention masks for a vision encoder to handle padding and token alignment.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, sequence_length).
        ar_ids (torch.Tensor): Aspect ratio IDs for masking.
        ntok (int): Number of tokens.
        num_chunks (int): Number of chunks in the data.
        supported_aspect_ratios (List[List[int]]): List of supported aspect ratios.

    Returns:
        torch.Tensor: Tensor containing the attention mask.
    """
    masks = []
    dtype = x.dtype
    for ar_id in ar_ids:
        arx = supported_aspect_ratios[ar_id - 1]
        mask_i = torch.ones((num_chunks, x.shape[1] // num_chunks), device=x.device)
        mask_i[: arx[0] * arx[1], :ntok] = 0
        mask_i = mask_i.view(num_chunks * x.shape[1] // num_chunks, -1)
        mask_i = mask_i @ mask_i.T
        mask_i = mask_i.unsqueeze(0)
        masks.append(mask_i)
    masks = torch.stack(masks).to(dtype) * torch.finfo(dtype).min
    return masks


# Use this spec for an implementation using modules in TE
def get_image_transformer_layer_spec() -> ModuleSpec:
    """
    Create a specification for an image transformer layer.
    """
    image_transformer_submodules = TransformerLayerSubmodules(
        input_layernorm=TENorm,
        self_attention=ModuleSpec(
            module=SelfAttentionNoBias,
            params={"attn_mask_type": AttnMaskType.no_mask},
            submodules=SelfAttentionSubmodules(
                linear_qkv=TEColumnParallelLinear,
                core_attention=TEDotProductAttention,
                linear_proj=TERowParallelLinear,
                q_layernorm=IdentityOp,
                k_layernorm=IdentityOp,
            ),
        ),
        self_attn_bda=get_bias_dropout_add,
        pre_mlp_layernorm=TENorm,
        mlp=ModuleSpec(
            module=MLP,
            submodules=MLPSubmodules(
                linear_fc1=TEColumnParallelLinear,
                linear_fc2=TERowParallelLinear,
            ),
        ),
        mlp_bda=get_bias_dropout_add,
    )
    return ModuleSpec(module=ImageTransformerLayer, submodules=image_transformer_submodules)


def forward_with_return_intermediate(
    self,
    hidden_states: Tensor,
    attention_mask: Tensor,
    context: Tensor = None,
    context_mask: Tensor = None,
    rotary_pos_emb: Tensor = None,
    attention_bias: Tensor = None,
    inference_params: InferenceParams = None,
    packed_seq_params: PackedSeqParams = None,
    return_intermediate: List[int] = None,
):
    """
    Perform a forward pass through the transformer layers with optional intermediate outputs.
    Override regular MCore transformer layer forward pass.
    """
    # hidden_states (float): [s, b, h]
    # attention_mask (bool): [1, 1, s, s]

    if not self.pre_process:
        # See set_input_tensor()
        hidden_states = self.input_tensor

    hidden_states = make_viewless_tensor(inp=hidden_states, requires_grad=True, keep_graph=True)

    if self.config.sequence_parallel:
        rng_context = tensor_parallel.get_cuda_rng_tracker().fork()
    else:
        rng_context = nullcontext()

    if self.config.fp8:
        import transformer_engine  # To keep out TE dependency when not training in fp8

        if self.config.fp8 == "e4m3":
            fp8_format = transformer_engine.common.recipe.Format.E4M3
        elif self.config.fp8 == "hybrid":
            fp8_format = transformer_engine.common.recipe.Format.HYBRID
        else:
            raise ValueError("E4M3 and HYBRID are the only supported FP8 formats.")

        fp8_recipe = TEDelayedScaling(
            config=self.config,
            fp8_format=fp8_format,
            override_linear_precision=(False, False, not self.config.fp8_wgrad),
        )
        fp8_group = None
        if parallel_state.model_parallel_is_initialized():
            fp8_group = parallel_state.get_amax_reduction_group(with_context_parallel=True)
        fp8_context = transformer_engine.pytorch.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe, fp8_group=fp8_group)
    else:
        fp8_context = nullcontext()

    with rng_context and fp8_context:
        # Forward pass.
        if self.config.recompute_granularity == 'full' and self.training:
            assert return_intermediate is None, (
                "Config `return_intermediate` cannot be used with " "`recompute_granularity='full'`. "
            )
            hidden_states = self._checkpointed_forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                context=context,
                context_mask=context_mask,
                rotary_pos_emb=rotary_pos_emb,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
            )
        else:
            intermediate_hidden_states = []
            for l_no, layer in enumerate(self.layers):
                if return_intermediate is not None and l_no in return_intermediate:
                    intermediate_hidden_states.append(hidden_states)

                with self.offload_context:
                    hidden_states, context = layer(
                        hidden_states=hidden_states,
                        attention_mask=attention_mask,
                        context=context,
                        context_mask=context_mask,
                        rotary_pos_emb=rotary_pos_emb,
                        attention_bias=attention_bias,
                        inference_params=inference_params,
                        packed_seq_params=packed_seq_params,
                    )
                    # CUDA graph doesn't output context and is expected to be None
                    assert (
                        (context is None)
                        or (not self.config.enable_cuda_graph)
                        or (self.config.cuda_graph_scope == "full_iteration")
                        or (not self.training)
                    )

                if (
                    torch.is_grad_enabled()
                    and self.config.cpu_offloading
                    and self.group_prefetch_offload_commit_async is not None
                ):
                    hidden_states = self.group_prefetch_offload_commit_async(hidden_states)

        # Final layer norm.
        if self.final_layernorm is not None:
            hidden_states = self.final_layernorm(hidden_states)
            # TENorm produces a "viewed" tensor. This will result in schedule.py's
            # deallocate_output_tensor() throwing an error, so a viewless tensor is
            # created to prevent this.
            hidden_states = make_viewless_tensor(inp=hidden_states, requires_grad=True, keep_graph=True)

        if return_intermediate is not None:
            return hidden_states, torch.stack(intermediate_hidden_states, dim=-1)

        return hidden_states


class ColumnParallelConv2dPatch(MegatronModule):
    """
    Conv2D Patching layer with model parallelism. Applies convolution in a column-parallel fashion.

    Args:
        config (TransformerConfig): Configuration object for the layer.
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (Union[int, Tuple[int, int]]): Size of the convolution kernel.
        stride (Union[int, Tuple[int, int]]): Stride of the convolution.
        bias (Optional[bool], default=False): Whether to include a bias term.

    Input:
        torch.Tensor: Input tensor of shape (batch_size, in_channels, width, height).

    Output:
        torch.Tensor: Output tensor of shape (batch_size, num_tokens, out_channels).
    """

    def __init__(
        self,
        config: TransformerConfig,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]],
        bias: Optional[bool] = False,
    ) -> None:
        super().__init__(config=config)
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self._unfold = torch.nn.Unfold(kernel_size=kernel_size, stride=stride)
        self._linear = TEColumnParallelLinear(
            in_channels * kernel_size[0] * kernel_size[1],
            out_channels,
            bias=bias,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name='conv1',
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        x = self._unfold(x)
        x = x.permute(0, 2, 1)
        x = F.linear(x, self._linear.weight)
        x = tensor_parallel.gather_from_tensor_model_parallel_region(x)
        return x


class PrecomputedTilePositionEmbedding(torch.nn.Module):
    """
    Module to compute positional embeddings for tiles with optional gating.

    Args:
        config (TransformerConfig): Configuration object.
        gated (bool, default=False): Whether to apply gating to the embeddings.
    """

    def __init__(
        self,
        config: TransformerConfig,
        gated: bool = False,
    ):
        super().__init__()
        self.max_num_tiles = config.max_num_tiles
        self.hidden_size = config.hidden_size
        self.max_aspect_ratio_id = config.max_aspect_ratio_id

        self.embedding = nn.Embedding(self.max_aspect_ratio_id + 1, self.max_num_tiles * self.hidden_size)
        self.gated = gated
        if gated:
            self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, hidden_states: torch.Tensor, aspect_ratio_ids: torch.Tensor) -> torch.Tensor:
        """Forward."""
        embeddings = self.embedding(aspect_ratio_ids)
        embeddings = embeddings.reshape(-1, self.max_num_tiles, 1, self.hidden_size)

        if self.gated:
            embeddings = embeddings * self.gate.tanh()

        hidden_states = hidden_states + embeddings
        return hidden_states


class SelfAttentionNoBias(SelfAttention):
    """
    Self-attention layer implementation without bias.

    Args:
        config (TransformerConfig): Configuration for the transformer.
        submodules (SelfAttentionSubmodules): Submodules required for self-attention.
        layer_number (int): The layer number in the transformer stack.
        attn_mask_type (AttnMaskType): Type of attention mask to apply.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: SelfAttentionSubmodules,
        layer_number: int,
        attn_mask_type=AttnMaskType.padding,
        **kwargs,
    ):
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            **kwargs,
        )

        # Override to remove bias since we don't have a good config for this.
        self.linear_qkv = build_module(
            submodules.linear_qkv,
            self.config.hidden_size,
            self.query_projection_size + 2 * self.kv_projection_size,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name='qkv',
        )

        self.linear_proj = build_module(
            submodules.linear_proj,
            self.query_projection_size,
            self.config.hidden_size,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=False,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name='proj',
        )


class ImageTransformerLayer(TransformerLayer):
    """
    Transformer layer adapted for processing image data with optional gating.

    Args:
        config (TransformerConfig): Transformer configuration object.
        submodules (TransformerLayerSubmodules): Submodules to use in the layer.
        layer_number (int, default=1): Layer number in the transformer.
        hidden_dropout (float, optional): Dropout rate for hidden layers.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: TransformerLayerSubmodules,
        layer_number: int = 1,
        hidden_dropout: float = None,
        **kwargs,
    ):
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            hidden_dropout=hidden_dropout,
            **kwargs,
        )
        self.gated = self.config.gated
        if self.gated:
            self.gate_attn = nn.Parameter(torch.zeros(1, dtype=self.config.params_dtype))
            self.gate_ffn = nn.Parameter(torch.zeros(1, dtype=self.config.params_dtype))

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        context=None,
        rotary_pos_emb=None,
        attention_bias=None,
        inference_params=None,
        packed_seq_params=None,
        **kwargs,
    ):
        """Forward."""
        # hidden_states: [s, b, h]

        # Residual connection.
        residual = hidden_states

        # Optional Input Layer norm
        input_layernorm_output = self.input_layernorm(hidden_states)

        # Self attention.
        attention_output_with_bias = self.self_attention(
            input_layernorm_output,
            attention_mask=attention_mask,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
        )

        _gate_attn = 1 if not self.gated else self.gate_attn.tanh()
        assert isinstance(
            attention_output_with_bias, tuple
        ), "`attention_output_with_bias` needs to be tuple for gating."
        attention_output_with_bias = tuple(
            _gate_attn * output if output is not None else None for output in attention_output_with_bias
        )

        with self.bias_dropout_add_exec_handler():
            hidden_states = self.self_attn_bda(self.training, self.config.bias_dropout_fusion)(
                attention_output_with_bias, residual, self.hidden_dropout
            )

        # Residual connection.
        residual = hidden_states

        # Optional Layer norm post the cross-attention.
        pre_mlp_layernorm_output = self.pre_mlp_layernorm(hidden_states)

        # MLP.
        mlp_output_with_bias = self.mlp(pre_mlp_layernorm_output)

        _gate_ffn = 1 if not self.gated else self.gate_ffn.tanh()
        assert isinstance(mlp_output_with_bias, tuple), "`mlp_output_with_bias` needs to be tuple for gating."
        mlp_output_with_bias = tuple(
            _gate_ffn * output if output is not None else None for output in mlp_output_with_bias
        )

        with self.bias_dropout_add_exec_handler():
            hidden_states = self.mlp_bda(self.training, self.config.bias_dropout_fusion)(
                mlp_output_with_bias, residual, self.hidden_dropout
            )

        output = make_viewless_tensor(inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True)

        # CUDA graph requires returned values to be Tensors
        if self.config.external_cuda_graph and self.training:
            return output
        return output, context


class VisionEncoder(MegatronModule):
    """
    Vision encoder module for processing image inputs with patch-based embeddings.

    Args:
        config ('CrossAttentionVisionConfig'): Configuration object for the encoder.
        image_size (int, default=560): Input image size.
        patch_size (int, default=14): Size of patches extracted from the image.
        in_channels (int, default=3): Number of input channels.
        pre_process (bool, default=True): Whether to preprocess input.
        post_process (bool, default=True): Whether to postprocess output.
        return_intermediate (Optional[bool]): Whether to return intermediate layers.
    """

    def __init__(
        self,
        config: 'CrossAttentionVisionConfig',
        image_size: int = 560,
        patch_size: int = 14,
        in_channels: int = 3,
        pre_process: bool = True,
        post_process: bool = True,
        return_intermediate=None,
    ):
        super().__init__(config=config)
        self.return_intermediate = return_intermediate
        self.image_size = to_2tuple(image_size)
        self.patch_size = to_2tuple(patch_size)
        self.grid_size = (
            self.image_size[0] // self.patch_size[0],
            self.image_size[1] // self.patch_size[1],
        )
        self.pre_process = pre_process
        self.post_process = post_process

        self.max_aspect_ratio_id = self.config.max_aspect_ratio_id
        self.max_num_tiles = config.max_num_tiles
        width = config.hidden_size
        self.conv1 = ColumnParallelConv2dPatch(
            config=config,
            in_channels=in_channels,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )
        scale = width**-0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(self.grid_size[0] * self.grid_size[1] + 1, width))
        self.ln_post = LayerNormImpl(config=config, hidden_size=width)
        self.ln_pre = LayerNormImpl(config=config, hidden_size=width)
        self.transformer = TransformerBlock(
            config=self.config,
            spec=get_image_transformer_layer_spec(),
            post_layer_norm=False,
            pre_process=self.pre_process,
            post_process=self.post_process,
        )
        self.transformer.forward = types.MethodType(forward_with_return_intermediate, self.transformer)
        # pre and post tile position embedding
        global_config = copy.deepcopy(self.config)
        global_config.num_layers = self.config.num_global_layers
        global_config.gated = True
        self.global_transformer = TransformerBlock(
            config=global_config,
            spec=get_image_transformer_layer_spec(),
            post_layer_norm=False,
            pre_process=self.pre_process,
            post_process=self.post_process,
        )
        # pre and post tile position embedding
        self.pre_tile_pos_embed = PrecomputedTilePositionEmbedding(
            config=config,
            gated=True,
        )
        self.post_tile_pos_embed = PrecomputedTilePositionEmbedding(
            config=config,
            gated=True,
        )
        self.gated_tile_positional_embedding = nn.Embedding(
            self.max_aspect_ratio_id + 1, self.max_num_tiles * (self.grid_size[0] * self.grid_size[1] + 1) * width
        )
        self.gated_positional_embedding_gate = nn.Parameter(torch.zeros(1))

    def apply_positional_embedding(self, x, aspect_ratio_ids):
        """Apply regular position embedding and tile positonal embedding."""
        bsz, num_chunks, num_tokens, dim = x.shape
        x = x.view(bsz * num_chunks, num_tokens, dim)
        x = x + self.positional_embedding * (1 - self.gated_positional_embedding_gate.tanh())
        x = x.view(bsz, num_chunks, num_tokens, dim)
        tile_position_embedding = self.gated_tile_positional_embedding(aspect_ratio_ids)
        tile_position_embedding = tile_position_embedding.reshape(bsz, num_chunks, num_tokens, dim)
        x = x + self.gated_positional_embedding_gate.tanh() * tile_position_embedding
        return x

    def apply_class_embedding(self, x):
        """Concat class embedding tokens."""
        x = torch.cat(
            [
                self.class_embedding.to(x.dtype)
                + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                x,
            ],
            dim=1,
        )  # shape = [*, grid ** 2 + 1, width]
        return x

    def forward(self, images: torch.Tensor, ar_ids: torch.Tensor) -> torch.Tensor:
        """Forward."""
        if images.ndim == 5:
            num_concurrent_media = 1
            bsz, num_chunks, nch, w, h = images.shape
        else:
            bsz, num_concurrent_media, num_chunks, nch, w, h = images.shape

        images = images.reshape(bsz * num_concurrent_media * num_chunks, nch, w, h)
        ar_ids = ar_ids.reshape(bsz * num_concurrent_media, 1)

        # patch embedding
        x = images.reshape(bsz * num_concurrent_media * num_chunks, nch, w, h)
        x = self.conv1(x)  # shape = [*, width, grid ** 2]
        _, ntok, dim = x.shape
        x = x.reshape(bsz * num_concurrent_media, num_chunks, ntok, dim)

        # tile embeddings
        x = self.pre_tile_pos_embed(x, ar_ids)
        x = x.reshape(bsz * num_concurrent_media * num_chunks, ntok, dim)

        # apply cls token
        x = self.apply_class_embedding(x)
        ntok += 1

        # apply position embeddings
        x = x.reshape(bsz * num_concurrent_media, num_chunks, ntok, dim)
        x = self.apply_positional_embedding(x, ar_ids)

        x = self.ln_pre(x)

        # Compute the number of tokens to pad (to be consistent with HF)
        npad = (8 - (x.shape[-2] % 8)) % 8
        # Compute padding tuple for pad function
        padding = (0, 0, 0, npad)  # (pad_left, pad_right, pad_left for dim -2, pad_right for dim -2)
        # Pad the tensor
        x = F.pad(x, padding, mode="constant", value=0)

        x = x.view(bsz * num_concurrent_media, -1, dim)

        attn_bias = build_encoder_attention_mask(x, ar_ids, ntok, num_chunks, self.config.supported_aspect_ratios)
        x = x.transpose(0, 1).contiguous()
        x, int_x = self.transformer(
            hidden_states=x,
            attention_mask=None,
            attention_bias=attn_bias,
            return_intermediate=self.return_intermediate,
        )

        # [ntok * num_concurrent_media * num_chunks, bsz, hidden_size]
        # -> [bsz, ntok * num_concurrent_media * num_chunks, hidden_size]
        x, int_x = x.transpose(0, 1).contiguous(), int_x.transpose(0, 1).contiguous()
        x = self.ln_post(x)
        x = x.reshape(bsz * num_concurrent_media, num_chunks, ntok + npad, dim)
        x = self.post_tile_pos_embed(x, ar_ids)
        x = x.reshape(bsz * num_concurrent_media, num_chunks * (ntok + npad), dim)
        x = x.transpose(0, 1).contiguous()
        x = self.global_transformer(
            hidden_states=x,
            attention_mask=None,
            attention_bias=attn_bias,
        )
        x = x.transpose(0, 1)
        x = x.reshape(bsz * num_concurrent_media, num_chunks, ntok + npad, dim)
        x = x[:, :, :ntok]

        # adding back intermediate layer outputs
        x = x.reshape(bsz, num_concurrent_media, num_chunks, ntok, dim)
        int_x = int_x.reshape(bsz * num_concurrent_media, num_chunks, ntok + npad, -1)
        int_x = int_x[:, :, :ntok]
        # int_x = contract_num_tokens_from_mult8(int_x, npad)
        int_x = int_x.reshape(bsz, num_concurrent_media, num_chunks, ntok, -1)
        x = torch.cat([x, int_x], dim=-1)
        return x
