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

import math
from contextlib import nullcontext
from dataclasses import dataclass
from typing import List, Literal, Optional, Union

import torch
from megatron.core import InferenceParams, parallel_state, tensor_parallel
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.dist_checkpointing.utils import replace_prefix_for_sharding
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add

from megatron.core.models.gpt.gpt_model import GPTModel as MCoreGPTModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.attention import Attention
from megatron.core.transformer.custom_layers.transformer_engine import (
    TEColumnParallelLinear,
    TEDotProductAttention,
    TELayerNormColumnParallelLinear,
    TERowParallelLinear,
)
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from megatron.core.transformer.utils import sharded_state_dict_default
from megatron.core.utils import make_viewless_tensor
from torch import Tensor, nn

from nemo.utils import logging

try:
    from megatron.core.transformer.custom_layers.transformer_engine import TEDelayedScaling, TENorm

    HAVE_TE = True
    LayerNormImpl = TENorm
except ImportError:
    from megatron.core.transformer.torch_layer_norm import WrappedTorchLayerNorm

    HAVE_TE = False
    LayerNormImpl = WrappedTorchLayerNorm


@dataclass
class MLlamaCrossAttentionSubmodules:
    """
    Defines the submodules required for cross-attention layers in the Llama architecture.
    """

    linear_q: Union[ModuleSpec, type] = None
    linear_kv: Union[ModuleSpec, type] = None
    core_attention: Union[ModuleSpec, type] = None
    linear_proj: Union[ModuleSpec, type] = None
    q_layernorm: Union[ModuleSpec, type] = None
    k_layernorm: Union[ModuleSpec, type] = None


class CrossAttentionTextModel(MCoreGPTModel):
    """
    GPT-based model with integrated cross-attention layers for multimodal tasks.
    """

    def __init__(
        self,
        config: TransformerConfig,
        transformer_layer_spec: ModuleSpec,
        vocab_size: int,
        max_sequence_length: int,
        pre_process: bool = True,
        post_process: bool = True,
        fp16_lm_cross_entropy: bool = False,
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = False,
        position_embedding_type: Literal['learned_absolute', 'rope', 'none'] = 'learned_absolute',
        rotary_percent: float = 1.0,
        rotary_base: int = 10000,
        seq_len_interpolation_factor: Optional[float] = None,
    ):
        super().__init__(
            config,
            transformer_layer_spec,
            vocab_size,
            max_sequence_length,
            pre_process,
            post_process,
            fp16_lm_cross_entropy,
            parallel_output,
            share_embeddings_and_output_weights,
            position_embedding_type,
            rotary_percent,
            rotary_base,
            seq_len_interpolation_factor,
        )

        # Overwrite the self.decoder
        self.decoder = CrossAttentionTransformerBlock(
            config=self.config,
            spec=transformer_layer_spec,
            pre_process=self.pre_process,
            post_process=self.post_process,
        )

        if self.pre_process:
            self.learnable_embedding = tensor_parallel.VocabParallelEmbedding(
                num_embeddings=8,
                embedding_dim=self.config.hidden_size,
                init_method=self.config.init_method,
                reduce_scatter_embeddings=False,  # TODO double check this
                config=self.config,
            )

            self.num_frozen_embeddings = self.embedding.word_embeddings.num_embeddings
            self._thresh = self.num_frozen_embeddings - 1

    def get_partially_trainable_embedding(self, x):
        """Get word embedding w/ few extra learnable tokens."""
        xz = torch.zeros_like(x, device=x.device)
        oz = torch.ones_like(x, device=x.device)
        x_orig = torch.minimum(x, torch.tensor(self._thresh, device=x.device))
        x_new = torch.maximum(x, torch.tensor(self._thresh + 1, device=x.device)) - self.num_frozen_embeddings

        mask_orig = torch.where(x >= self.num_frozen_embeddings, xz, oz).unsqueeze(-1)
        mask_new = torch.where(x < self.num_frozen_embeddings, xz, oz).unsqueeze(-1)

        x_orig = self.embedding(x_orig, None).transpose(0, 1)
        x_new = self.learnable_embedding(x_new).type_as(x_orig)
        return x_orig * mask_orig.type_as(x_orig) + x_new * mask_new.type_as(x_new)

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        decoder_input: Tensor = None,
        cross_attention_masks: Tensor = None,
        full_text_row_masked_out_mask: Tensor = None,
        xattn_caches: Optional[List] = None,
        labels: Tensor = None,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict = None,
    ) -> Tensor:
        """Forward."""
        # Decoder embedding.
        if decoder_input is not None:
            pass
        elif self.pre_process:
            raise ValueError("Require: decoder_input is not None or self.pre_process is False")
        else:
            # intermediate stage of pipeline
            # decoder will get hidden_states from encoder.input_tensor
            decoder_input = None

        # Rotary positional embeddings (embedding is None for PP intermediate devices)
        rotary_pos_emb = None
        if self.position_embedding_type == 'rope':
            rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
                inference_params,
                self.decoder,
                decoder_input,
                self.config,
                packed_seq_params=None,
            )
            rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len)

        dtype = decoder_input.dtype
        cross_attention_bias = cross_attention_masks.to(dtype) * torch.finfo(dtype).min

        # Run decoder.
        hidden_states = self.decoder(
            hidden_states=decoder_input,
            attention_mask=attention_mask,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb,
            packed_seq_params=packed_seq_params,
            cross_attention_masks=None,
            full_text_row_masked_out_mask=full_text_row_masked_out_mask,
            xattn_caches=xattn_caches,
            cross_attention_bias=cross_attention_bias,
            **(extra_block_kwargs or {}),
        )

        if not self.post_process:
            return hidden_states

        # logits and loss
        output_weight = None
        if self.share_embeddings_and_output_weights:
            output_weight = self.shared_embedding_or_output_weight()
        logits, _ = self.output_layer(hidden_states, weight=output_weight)

        if labels is None:
            # [s b h] => [b s h]
            return logits.transpose(0, 1).contiguous()

        loss = self.compute_language_model_loss(labels, logits)

        return loss


class CrossAttentionTransformerBlock(TransformerBlock):
    """
    Transformer block with integrated cross-attention layers for multimodal tasks.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.fusion_schedule = [
            x - self._get_layer_offset()
            for x in self.config.fusion_schedule
            if 0 <= (x - self._get_layer_offset()) < self.num_layers_per_pipeline_rank
        ]
        self.xattn_layers = []

        for i in range(self.num_layers_per_pipeline_rank):
            if i in self.fusion_schedule:
                layer_spec = ModuleSpec(
                    module=CrossAttentionTransformerLayer,
                    submodules=TransformerLayerSubmodules(
                        cross_attention=ModuleSpec(
                            module=MLlamaCrossAttention,
                            params={"attn_mask_type": AttnMaskType.no_mask},
                            submodules=MLlamaCrossAttentionSubmodules(
                                linear_q=TELayerNormColumnParallelLinear,  # This wraps attention_norm before attention
                                linear_kv=TEColumnParallelLinear,
                                core_attention=TEDotProductAttention,
                                linear_proj=TERowParallelLinear,
                                q_layernorm=TENorm,
                                k_layernorm=TENorm,
                            ),
                        ),
                        cross_attn_bda=get_bias_dropout_add,
                        pre_mlp_layernorm=IdentityOp,
                        mlp=ModuleSpec(
                            module=MLP,
                            submodules=MLPSubmodules(
                                linear_fc1=TELayerNormColumnParallelLinear,  # This wraps ffn_norm before feed_forward
                                linear_fc2=TERowParallelLinear,
                            ),
                        ),
                        mlp_bda=get_bias_dropout_add,
                    ),
                )
                self.xattn_layers.append(build_module(layer_spec, config=self.config, layer_number=i + 1))
            else:
                self.xattn_layers.append(DummyCrossAttentionTransformerLayer(config=self.config))
        self.xattn_layers = torch.nn.ModuleList(self.xattn_layers)

        assert len(self.xattn_layers) == len(self.layers), 'Check PP implementation for cross attention layers!'

    def _get_layer_offset(self):
        """Get correct layer offset when encoder pipeline parallel size > 0."""
        encoder_pipeline_model_parallel_size = getattr(self.config, "encoder_pipeline_model_parallel_size", 0)
        decoder_pipeline_model_parallel_rank = (
            parallel_state.get_pipeline_model_parallel_rank() - encoder_pipeline_model_parallel_size
        )
        return decoder_pipeline_model_parallel_rank * self.num_layers_per_pipeline_rank

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        xattn_caches: Optional[List] = None,
        cross_attention_masks: Tensor = None,
        full_text_row_masked_out_mask: Tensor = None,
        rotary_pos_emb: Tensor = None,
        attention_bias: Tensor = None,
        cross_attention_bias: Tensor = None,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
    ):
        """Forward."""
        # hidden_states (float): [s, b, h]
        # attention_mask (bool): [1, 1, s, s]

        if not self.pre_process:
            hidden_states = self.input_tensor

        hidden_states = make_viewless_tensor(
            inp=hidden_states,
            requires_grad=True,
            keep_graph=True,
        )

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
            fp8_context = transformer_engine.pytorch.fp8_autocast(
                enabled=True, fp8_recipe=fp8_recipe, fp8_group=fp8_group
            )
        else:
            fp8_context = nullcontext()

        with rng_context and fp8_context:
            # Forward pass.
            if self.config.recompute_granularity == 'full' and self.training:
                raise NotImplementedError
            else:
                for l_no, (layer, xattn_layer) in enumerate(zip(self.layers, self.xattn_layers)):
                    layer: TransformerLayer
                    xattn_layer: Union[DummyCrossAttentionTransformerLayer, CrossAttentionTransformerLayer]
                    with self.offload_context:
                        if (len(self.cuda_graphs) == 0) or (not self.training):
                            hidden_states, context = xattn_layer(
                                hidden_states=hidden_states,
                                cross_attention_masks=cross_attention_masks,
                                xattn_cache=xattn_caches[l_no],
                                full_text_row_masked_out_mask=full_text_row_masked_out_mask,
                                rotary_pos_emb=rotary_pos_emb,
                                cross_attention_bias=cross_attention_bias,
                                inference_params=None,  # Skip inference_params for xattn
                                packed_seq_params=packed_seq_params,
                            )
                            hidden_states, context = layer(
                                hidden_states=hidden_states,
                                attention_mask=attention_mask,
                                rotary_pos_emb=rotary_pos_emb,
                                attention_bias=attention_bias,
                                inference_params=inference_params,
                                packed_seq_params=packed_seq_params,
                            )
                            # CUDA graph doesn't output context and is expected to be None
                            assert (context is None) or (not self.config.enable_cuda_graph) or (not self.training)
                        else:
                            assert (len(self.cuda_graphs) > l_no) and (
                                self.current_microbatch < len(self.cuda_graphs[l_no])
                            )
                            hidden_states = self.cuda_graphs[l_no][self.current_microbatch](
                                hidden_states, is_first_microbatch=(self.current_microbatch == 0)
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
            hidden_states = make_viewless_tensor(inp=hidden_states, requires_grad=True, keep_graph=True)

        return hidden_states

    def sharded_state_dict(
        self, prefix: str = '', sharded_offsets: tuple = (), metadata: dict = None
    ) -> ShardedStateDict:
        """Update shareded state dict for cross-attention layers"""
        sharded_state_dict = {}

        layer_prefix = f'{prefix}layers.'
        num_layers = self.config.num_layers
        for layer in self.layers:
            offset = layer._get_layer_offset(layer.config)
            global_layer_offset = layer.layer_number - 1  # self.layer_number starts at 1
            state_dict_prefix = f'{layer_prefix}{global_layer_offset - offset}.'  # module list index in TransformerBlock # pylint: disable=line-too-long
            sharded_prefix = layer_prefix
            sharded_pp_offset = [(0, global_layer_offset, num_layers)]  # PP sharding offset for ShardedTensors
            layer_sharded_state_dict = layer.sharded_state_dict(state_dict_prefix, sharded_pp_offset, metadata)
            replace_prefix_for_sharding(layer_sharded_state_dict, state_dict_prefix, sharded_prefix)
            sharded_state_dict.update(layer_sharded_state_dict)

        xlayer_prefix = f'{prefix}xattn_layers.'
        for xlayer in self.xattn_layers:
            if isinstance(xlayer, DummyCrossAttentionTransformerLayer):
                continue
            offset = xlayer._get_layer_offset(xlayer.config)
            global_layer_offset = xlayer.layer_number - 1
            state_dict_prefix = f'{xlayer_prefix}{global_layer_offset - offset}.'  # module list index in TransformerBlock # pylint: disable=line-too-long
            sharded_prefix = f'{xlayer_prefix}{global_layer_offset}.'
            sharded_pp_offset = []
            xlayer_sharded_state_dict = xlayer.sharded_state_dict(state_dict_prefix, sharded_pp_offset, metadata)
            replace_prefix_for_sharding(xlayer_sharded_state_dict, state_dict_prefix, sharded_prefix)
            sharded_state_dict.update(xlayer_sharded_state_dict)

        # Add modules other than self.layers
        for name, module in self.named_children():
            if not module is self.layers and not module is self.xattn_layers:
                sharded_state_dict.update(
                    sharded_state_dict_default(module, f'{prefix}{name}.', sharded_offsets, metadata)
                )

        return sharded_state_dict


class CrossAttentionTransformerLayer(TransformerLayer):
    """
    Transformer layer with cross-attention for integration.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: TransformerLayerSubmodules,
        layer_number: int = 1,
        hidden_dropout: float = None,
    ):
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            hidden_dropout=hidden_dropout,
        )

        self.gate_attn = nn.Parameter(torch.zeros(1, dtype=self.config.params_dtype))
        self.gate_ffn = nn.Parameter(torch.zeros(1, dtype=self.config.params_dtype))

    def compute_xattn_kv_cache(self, xattn_tokens: Tensor) -> Tensor:
        """Compute cross-attention kv cahce."""
        return self.cross_attention._compute_xattn_kv_cache(xattn_tokens)

    def forward(
        self,
        hidden_states,
        cross_attention_masks,
        xattn_cache=None,
        full_text_row_masked_out_mask=None,
        rotary_pos_emb=None,
        cross_attention_bias=None,
        inference_params=None,
        packed_seq_params=None,
    ):
        """Forward."""
        # hidden_states: [s, b, h]

        # Residual connection.
        residual = hidden_states

        # Optional Layer norm after self-attention
        pre_cross_attn_layernorm_output = self.pre_cross_attn_layernorm(hidden_states)

        # Cross attention.
        attention_output_with_bias = self.cross_attention(
            pre_cross_attn_layernorm_output,
            cross_attention_masks=cross_attention_masks,
            xattn_cache=xattn_cache,
            full_text_row_masked_out_mask=full_text_row_masked_out_mask,
            rotary_pos_emb=rotary_pos_emb,
            cross_attention_bias=cross_attention_bias,
            inference_params=inference_params,
        )

        _gate_attn = self.gate_attn.tanh()
        assert isinstance(
            attention_output_with_bias, tuple
        ), "`attention_output_with_bias` needs to be tuple for gating."
        attention_output_with_bias = tuple(
            _gate_attn * output if output is not None else None for output in attention_output_with_bias
        )

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.cross_attn_bda(self.training, self.config.bias_dropout_fusion)(
                attention_output_with_bias, residual, self.hidden_dropout
            )

        # Residual connection.
        residual = hidden_states

        # Optional Layer norm post the cross-attention.
        pre_mlp_layernorm_output = self.pre_mlp_layernorm(hidden_states)

        # MLP.
        mlp_output_with_bias = self.mlp(pre_mlp_layernorm_output)

        _gate_ffn = self.gate_ffn.tanh() * full_text_row_masked_out_mask
        assert isinstance(mlp_output_with_bias, tuple), "`mlp_output_with_bias` needs to be tuple for gating."
        mlp_output_with_bias = tuple(
            _gate_ffn * output if output is not None else None for output in mlp_output_with_bias
        )

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.mlp_bda(self.training, self.config.bias_dropout_fusion)(
                mlp_output_with_bias, residual, self.hidden_dropout
            )

        # Jit compiled function creates 'view' tensor. This tensor
        # potentially gets saved in the MPU checkpoint function context,
        # which rejects view tensors. While making a viewless tensor here
        # won't result in memory savings (like the data loader, or
        # p2p_communication), it serves to document the origin of this
        # 'view' tensor.
        output = make_viewless_tensor(inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True)

        return output, None  # context


class DummyCrossAttentionTransformerLayer(MegatronModule):
    """Dummy cross-attention transformer block with tanh-gated attention and feedforward."""

    def __call__(
        self,
        hidden_states: Tensor,
        *args,
        **kwargs,
    ):
        return hidden_states, None

    def compute_xattn_kv_cache(self, xattn_tokens: Tensor) -> Optional[Tensor]:
        # pylint: disable=C0115,C0116
        return None


class MLlamaCrossAttention(Attention):
    """
    Cross-attention layer for Llama multimodal tasks.

    Cross-attention layer takes input with size [s, b, h] and context with size
    [s, b, h] and returns output of the same size.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: MLlamaCrossAttentionSubmodules,
        layer_number: int,
        attn_mask_type=AttnMaskType.padding,
    ):
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            attention_type="cross",
        )

        # TODO might need special care when TP>8
        assert self.query_projection_size % self.kv_projection_size == 0

        self.linear_q = build_module(
            submodules.linear_q,
            self.config.hidden_size,
            self.query_projection_size,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=self.config.add_bias_linear,
            skip_bias_add=False,
            is_expert=False,
        )

        self.linear_kv = build_module(
            submodules.linear_kv,
            self.config.hidden_size,
            2 * self.kv_projection_size,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=self.config.add_bias_linear,
            skip_bias_add=False,
            is_expert=False,
        )

        self.q_layernorm = build_module(
            submodules.q_layernorm,
            hidden_size=self.hidden_size_per_attention_head,
            config=self.config,
            eps=self.config.layernorm_epsilon,
        )

        self.k_layernorm = build_module(
            submodules.k_layernorm,
            hidden_size=self.hidden_size_per_attention_head,
            config=self.config,
            eps=self.config.layernorm_epsilon,
        )

    def get_key_value_tensors(self, key_value_states):
        """Get key value tensors."""
        mixed_kv, _ = self.linear_kv(key_value_states)

        # [sk, b, (np * 2 * hn)] --> [sk, b, np, 2 * hn]
        new_tensor_shape = mixed_kv.size()[:-1] + (
            self.num_query_groups_per_partition,
            2 * self.hidden_size_per_attention_head,
        )
        mixed_kv = mixed_kv.view(*new_tensor_shape)

        # [sk, b, np, 2 * hn] --> 2 [sk, b, np, hn]
        (key, value) = tensor_parallel.split_tensor_along_last_dim(mixed_kv, 2)
        # Apply LayerNorm
        key = self.k_layernorm(key.contiguous())
        return key, value

    def get_query_tensor(self, hidden_states):
        """ "Get query tensor."""
        # Attention head [sq, b, h] --> [sq, b, hp]
        query, _ = self.linear_q(hidden_states)

        # [sq, b, hp] --> [sq, b, np, hn]
        new_tensor_shape = query.size()[:-1] + (
            self.num_attention_heads_per_partition,
            self.hidden_size_per_attention_head,
        )
        query = query.view(*new_tensor_shape)

        # Apply LayerNorm
        query = self.q_layernorm(query)

        return query

    def get_query_key_value_tensors(self, hidden_states, key_value_states):
        """Get query key value tensors."""
        query = self.get_query_tensor(hidden_states)
        key, value = self.get_key_value_tensors(key_value_states)
        return query, key, value

    def forward(
        self,
        hidden_states,
        cross_attention_masks,
        xattn_cache=None,
        full_text_row_masked_out_mask=None,
        inference_params=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        cross_attention_bias=None,
        packed_seq_params=None,
    ):
        """Forward."""
        # hidden_states: [sq, b, h]
        if self.config.flash_decode:
            rotary_pos_emb = None
        else:
            assert rotary_pos_cos is None and rotary_pos_sin is None

        # For self attention we just duplicate the rotary_pos_emb if it isn't already
        if rotary_pos_emb is not None and not isinstance(rotary_pos_emb, tuple):
            rotary_pos_emb = (rotary_pos_emb,) * 2

        # =====================
        # Query, Key, and Value
        # =====================
        # Get the query, key and value tensors based on the type of attention -
        # self or cross attn.
        query = self.get_query_tensor(hidden_states)
        key, value = xattn_cache

        # ===================================================
        # Adjust key, value, and rotary_pos_emb for inference
        # ===================================================
        query, key, value, rotary_pos_emb, attn_mask_type = self._adjust_key_value_for_inference(
            inference_params, query, key, value, rotary_pos_emb, rotary_pos_cos, rotary_pos_sin
        )

        if packed_seq_params is not None:
            query = query.squeeze(1)
            key = key.squeeze(1)
            value = value.squeeze(1)

        # ==================================
        # core attention computation
        # ==================================

        if self.checkpoint_core_attention and self.training:
            core_attn_out = self._checkpointed_attention_forward(
                query,
                key,
                value,
                cross_attention_masks,
                attn_mask_type=attn_mask_type,
                attention_bias=cross_attention_bias,
                packed_seq_params=packed_seq_params,
            )
        else:
            core_attn_out = self.core_attention(
                query,
                key,
                value,
                cross_attention_masks,
                attn_mask_type=attn_mask_type,
                attention_bias=cross_attention_bias,
                packed_seq_params=packed_seq_params,
            )

        if packed_seq_params is not None:
            # reshape to same output shape as unpacked case
            # (t, np, hn) -> (t, b=1, h=np*hn)
            # t is the pack size = sum (sq_i)
            # note that batch is a dummy dimension in the packed case
            core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)

        # [b, head, s, dim]
        core_attn_out = core_attn_out * full_text_row_masked_out_mask

        # =================
        # Output. [sq, b, h]
        # =================

        output, bias = self.linear_proj(core_attn_out)

        return output, bias

    def _compute_xattn_kv_cache(self, xattn_tokens: Tensor) -> Tensor:
        key, value = self.get_key_value_tensors(xattn_tokens)
        return torch.stack([key, value])


def apply_rope_scaling(
    inv_freq,
    factor: int = 8,
    low_freq_factor: int = 1,
    high_freq_factor: int = 4,
    old_context_len: int = 8192,
):
    """
    Apply scaling to rotary embeddings for positional encoding.

    Args:
        inv_freq (Tensor): Tensor of inverse frequencies.
        factor (int): Scaling factor for medium-to-high frequencies.
        low_freq_factor (int): Factor for identifying low frequencies.
        high_freq_factor (int): Factor for identifying high frequencies.
        old_context_len (int): Original context length for scaling computation.

    Returns:
        Tensor: Scaled inverse frequencies.
    """
    logging.info(
        f"Apply rope scaling with factor={factor}, low_freq_factor={low_freq_factor}, "
        f"high_freq_factor={high_freq_factor}, old_context_len={old_context_len}."
    )

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor

    wavelen = 2 * math.pi / inv_freq
    # wavelen < high_freq_wavelen: do nothing
    # wavelen > low_freq_wavelen: divide by factor
    inv_freq_llama = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
    # otherwise: interpolate between the two, using a smooth factor
    smooth_factor = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    smoothed_inv_freq = (1 - smooth_factor) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
    is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
    inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)

    return inv_freq_llama
