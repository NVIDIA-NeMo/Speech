# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
# coding=utf-8


"""Transformer."""

import torch

from nemo.collections.nlp.modules.common.megatron.layer_type import LayerType
from nemo.collections.nlp.modules.common.megatron.module import MegatronModule
from nemo.collections.nlp.modules.common.megatron.transformer import ParallelTransformer, ParallelTransformerLayer_
from nemo.collections.nlp.modules.common.megatron.utils import ApexGuardDefaults

try:
    from apex.transformer.enums import AttnMaskType, ModelType

    HAVE_APEX = True

except (ImportError, ModuleNotFoundError):

    HAVE_APEX = False

    # fake missing classes with None attributes
    ModelType = AttnMaskType = AttnType = LayerType = ApexGuardDefaults()

try:
    from megatron.core import parallel_state

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    HAVE_MEGATRON_CORE = False

""" We use the following notation throughout this file:
     h: hidden size
     n: number of attention heads
     p: number of model parallel partitions
     np: n/p
     hp: h/p
     hn: h/n
     b: batch size
     s: sequence length
     l: number of layers
    Transformer takes input of size [s, b, h] and returns a
    tensor of the same size. We use the following arguments:
        hyperparameters: transformer hyperparameters
"""


class DropPath(MegatronModule):
    """Drop paths (Stochastic Depth) per sample
    (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_state):
        if self.drop_prob == 0.0 or not self.training:
            return hidden_state
        keep_prob = 1 - self.drop_prob
        # work with diff dim tensors, not just 2D ConvNets
        # hidden_state: [s, b, h]
        shape = (1,) + (hidden_state.shape[1],) + (1,) * (hidden_state.ndim - 2)
        random_tensor = keep_prob + torch.rand(shape, dtype=hidden_state.dtype, device=hidden_state.device)
        random_tensor.floor_()  # binarize
        output = hidden_state.div(keep_prob) * random_tensor
        return output


class ParallelVisionTransformerLayer_(ParallelTransformerLayer_):
    """A single transformer layer.

    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.
    """

    def __init__(
        self,
        config,
        init_method,
        output_layer_init_method,
        layer_number,
        hidden_size,
        ffn_hidden_size,
        num_attention_heads,
        layer_type=LayerType.encoder,
        self_attn_mask_type=AttnMaskType.padding,
        fp32_residual_connection=False,
        precision=16,
        apply_query_key_layer_scaling=True,
        kv_channels=None,
        layernorm_epsilon=1e-5,
        hidden_dropout=0.1,
        bias_dropout_add_fusion=True,
        persist_layer_norm=False,
        bias_activation_fusion=True,
        openai_gelu=False,
        onnx_safe=False,
        masked_softmax_fusion=True,
        attention_dropout=0.1,
        ffn_dropout=0.0,
        drop_path_rate=0.0,
        activation='gelu',
        megatron_legacy=False,
        bias=True,
        chunk_size=64,
        normalization='layernorm',
        transformer_block_type='pre_ln',
        headscale=False,
        activations_checkpoint_granularity=None,
        normalize_attention_scores=True,
        use_flash_attention=False,
    ):
        kwargs = locals()
        for key in ["self", "__class__"]:
            kwargs.pop(key)
        drop_path_rate = kwargs.pop("drop_path_rate")
        super(ParallelVisionTransformerLayer_, self).__init__(**kwargs)

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else None

    def forward(
        self,
        hidden_states,
        attention_mask,
        encoder_output=None,
        enc_dec_attn_mask=None,
        layer_past=None,
        get_key_value=False,
        set_inference_key_value_memory=False,
        inference_max_sequence_len=None,
        rotary_pos_emb=None,
        # list of positional embedding tensors, first one self attention, second one and third one are for cross attention (q, k)
        self_attention_relative_position_bias=None,
        cross_attention_relative_position_bias=None,
        checkpoint_core_attention=False,
    ):
        # Self attention.
        if rotary_pos_emb is not None:
            # self attention pos_emb is (q, q)
            self_attention_pos_emb = (rotary_pos_emb[0], rotary_pos_emb[0])
            cross_attention_pos_emb = (rotary_pos_emb[1], rotary_pos_emb[2])
        else:
            self_attention_pos_emb = None
            cross_attention_pos_emb = None

        if self.layer_type != LayerType.retrieval_decoder_after_self_attn:
            # hidden_states: [b, s, h]

            # Pre-LN: x -> LN -> MHA -> Residual -> LN -> MLP -> Residual
            # Post-LN: x -> MHA -> Residual -> LN -> MLP -> Residual -> LN
            # Normformer: x -> LN -> MHA -> LN -> Residual -> MLP (w/LN) -> Residual

            residual = hidden_states
            # Layer norm at the beginning of the transformer layer.
            if self.transformer_block_type in ['pre_ln', 'normformer']:
                hidden_states = self.input_layernorm(hidden_states)

            attention_output, attention_bias = self.self_attention(
                hidden_states,
                attention_mask,
                layer_past=layer_past,
                get_key_value=get_key_value,
                set_inference_key_value_memory=set_inference_key_value_memory,
                inference_max_sequence_len=inference_max_sequence_len,
                rotary_pos_emb=self_attention_pos_emb,
                relative_position_bias=self_attention_relative_position_bias,
                checkpoint_core_attention=checkpoint_core_attention,
            )

            if get_key_value:
                attention_output, presents = attention_output

            # If normformer, apply norm on the output of the self attention.
            if self.transformer_block_type == 'normformer':
                # Normformer normalization
                attention_output = (
                    attention_output + attention_bias if attention_bias is not None else attention_output
                )
                attention_output = self.post_attention_normformer_norm(attention_output)
                attention_bias = None

            # jit scripting for a nn.module (with dropout) is not
            # trigerring the fusion kernel. For now, we use two
            # different nn.functional routines to account for varying
            # dropout semantics during training and inference phases.

            if self.drop_path is None:
                bias_dropout_add_func = self._get_bias_droput_add_func(
                    transformer_block_type=self.transformer_block_type, position_after='attention'
                )
                if attention_bias is not None:
                    attention_bias = attention_bias.expand_as(residual)

                layernorm_input = bias_dropout_add_func(
                    attention_output, attention_bias, residual, self.hidden_dropout
                )
            else:
                assert self.transformer_block_type != 'normformer', "Normfomer doesn't support drop_path"
                out = torch.nn.functional.dropout(
                    attention_output + attention_bias, p=self.hidden_dropout, training=self.training
                )
                layernorm_input = residual + self.drop_path(out)
            # print(f"Layer: {self.layer_number} Attention checksum {layernorm_input.sum()}")

            # Post-LN normalization after residual
            if self.transformer_block_type == 'post_ln':
                normalization_output = self.input_layernorm(layernorm_input)
                layernorm_input = normalization_output
            elif self.transformer_block_type in ['pre_ln', 'normformer']:
                # Layer norm post the self attention.
                normalization_output = self.post_attention_layernorm(layernorm_input)
        else:
            layernorm_input, normalization_output = hidden_states

        if self.layer_type == LayerType.decoder_pre_mlp:
            return layernorm_input, normalization_output

        if (
            self.layer_type == LayerType.decoder
            or self.layer_type == LayerType.retrieval_decoder
            or self.layer_type == LayerType.retrieval_encoder
            or self.layer_type == LayerType.retrieval_decoder_after_self_attn
        ):
            if (
                self.layer_type == LayerType.retrieval_decoder
                or self.layer_type == LayerType.retrieval_decoder_after_self_attn
            ):
                attention_output, attention_bias = self.inter_attention(
                    normalization_output,
                    enc_dec_attn_mask,
                    encoder_output=encoder_output,
                    rotary_pos_emb=cross_attention_pos_emb,
                    set_inference_key_value_memory=set_inference_key_value_memory,
                    inference_max_sequence_len=inference_max_sequence_len,
                    checkpoint_core_attention=checkpoint_core_attention,
                )
            else:
                attention_output, attention_bias = self.inter_attention(
                    normalization_output,
                    enc_dec_attn_mask,
                    encoder_output=encoder_output,
                    rotary_pos_emb=cross_attention_pos_emb,
                    relative_position_bias=cross_attention_relative_position_bias,
                    checkpoint_core_attention=checkpoint_core_attention,
                )

            # If normformer, apply norm on the output of the self attention.
            if self.transformer_block_type == 'normformer':
                # Normformer normalization
                attention_output = (
                    attention_output + attention_bias if attention_bias is not None else attention_output
                )
                attention_output = self.post_inter_attention_normformer_norm(attention_output)
                attention_bias = None

            residual = layernorm_input

            bias_dropout_add_func = self._get_bias_droput_add_func(
                transformer_block_type=self.transformer_block_type, position_after='attention'
            )

            layernorm_input = bias_dropout_add_func(attention_output, attention_bias, residual, self.hidden_dropout)
            # print(f"Layer: {self.layer_number} Cross-Attention checksum {layernorm_input.sum()}")
            normalization_output = self.post_inter_attention_layernorm(layernorm_input)
            # Post-LN normalization after residual
            if self.transformer_block_type == 'post_ln':
                layernorm_input = normalization_output
        # MLP.
        mlp_output, mlp_bias = self.mlp(normalization_output)

        residual = layernorm_input

        if self.drop_path is None:
            bias_dropout_add_func = self._get_bias_droput_add_func(
                transformer_block_type=self.transformer_block_type, position_after='mlp'
            )

            output = bias_dropout_add_func(mlp_output, mlp_bias, residual, self.hidden_dropout)

        else:
            out = torch.nn.functional.dropout(mlp_output + mlp_bias, p=self.hidden_dropout, training=self.training)
            output = residual + self.drop_path(out)
        # print(f"Layer: {self.layer_number} MLP + Dropout + Residual checksum {output.sum()}")

        if self.transformer_block_type == 'post_ln':
            output = self.post_attention_layernorm(output)

        if get_key_value:
            output = [output, presents]

        return output


class ParallelVisionTransformerLayer(ParallelVisionTransformerLayer_):
    def __init__(self, **kwargs):
        super(ParallelVisionTransformerLayer, self).__init__(**kwargs)
        precision = kwargs['precision']
        if precision in ['bf16', 'bf16-mixed']:
            self.dtype = torch.bfloat16
        elif precision in [16, '16', '16-mixed']:
            self.dtype = torch.float16
        elif precision in [32, '32', '32-true']:
            self.dtype = torch.float32
        else:
            raise ValueError(f"Cannot recognize precision {precision}")

    def forward(
        self,
        hidden_states,
        attention_mask,
        encoder_output=None,
        enc_dec_attn_mask=None,
        rotary_pos_emb=None,
        layer_past=None,
        get_key_value=False,
        set_inference_key_value_memory=False,
        inference_max_sequence_len=None,
        self_attention_relative_position_bias=None,
        cross_attention_relative_position_bias=None,
        checkpoint_core_attention=False,
    ):
        kwargs = locals()
        for key in ["self", "__class__"]:
            kwargs.pop(key)
        if self.dtype == torch.float32:
            return super().forward(**kwargs)
        with torch.autocast(device_type="cuda", dtype=self.dtype):
            return super().forward(**kwargs)


class ParallelVisionTransformer(ParallelTransformer):
    """Transformer class."""

    def __init__(
        self,
        config,
        init_method,
        output_layer_init_method,
        num_layers,
        hidden_size,
        ffn_hidden_size,
        num_attention_heads,
        apply_query_key_layer_scaling=True,
        kv_channels=None,
        layer_type=LayerType.encoder,  # it can be a list of types or single type
        self_attn_mask_type=AttnMaskType.padding,
        pre_process=True,
        post_process=True,
        precision=16,
        fp32_residual_connection=False,
        activations_checkpoint_method=None,
        activations_checkpoint_num_layers=None,
        layernorm_epsilon=1e-5,
        hidden_dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.0,
        drop_path_rate=0.0,
        bias_activation_fusion=True,
        bias_dropout_add_fusion=True,
        masked_softmax_fusion=True,
        persist_layer_norm=False,
        openai_gelu=False,
        onnx_safe=False,
        activation='gelu',
        model_type=ModelType.encoder_or_decoder,
        megatron_legacy=False,
        bias=True,
        chunk_size=64,
        normalization='layernorm',
        transformer_block_type='pre_ln',
        headscale=False,
        layer_number_offset=0,  # this is use only for attention norm_factor scaling
        activations_checkpoint_granularity=None,
        normalize_attention_scores=True,
        ub_tp_comm_overlap=False,
        use_flash_attention=False,
    ):
        kwargs = locals()
        for key in ["self", "__class__"]:
            kwargs.pop(key)
        self.drop_path_rate = kwargs.pop("drop_path_rate")
        super(ParallelVisionTransformer, self).__init__(**kwargs)

        self.num_layers = self.get_num_layers(num_layers)

        self.drop_path_rates = [
            rate.item()
            for rate in torch.linspace(
                0, self.drop_path_rate, self.num_layers * parallel_state.get_pipeline_model_parallel_world_size()
            )
        ]

        # Rebuild with vision transformer layers.
        def build_layer(layer_number):
            if isinstance(layer_type, list):
                lt = layer_type[layer_number - 1]
            else:
                lt = layer_type
            return ParallelVisionTransformerLayer(
                config=config,
                init_method=init_method,
                output_layer_init_method=output_layer_init_method,
                layer_number=layer_number + layer_number_offset,
                hidden_size=hidden_size,
                ffn_hidden_size=ffn_hidden_size,
                num_attention_heads=num_attention_heads,
                apply_query_key_layer_scaling=apply_query_key_layer_scaling,
                kv_channels=kv_channels,
                layer_type=lt,
                self_attn_mask_type=self_attn_mask_type,
                precision=precision,
                fp32_residual_connection=fp32_residual_connection,
                layernorm_epsilon=layernorm_epsilon,
                hidden_dropout=hidden_dropout,
                attention_dropout=attention_dropout,
                ffn_dropout=ffn_dropout,
                drop_path_rate=self.drop_path_rates[layer_number - 1],
                bias_activation_fusion=bias_activation_fusion,
                bias_dropout_add_fusion=bias_dropout_add_fusion,
                masked_softmax_fusion=masked_softmax_fusion,
                persist_layer_norm=persist_layer_norm,
                openai_gelu=openai_gelu,
                onnx_safe=onnx_safe,
                activation=activation,
                megatron_legacy=megatron_legacy,
                bias=bias,
                chunk_size=chunk_size,
                normalization=normalization,
                transformer_block_type=transformer_block_type,
                headscale=headscale,
                activations_checkpoint_granularity=activations_checkpoint_granularity,
                normalize_attention_scores=normalize_attention_scores,
                use_flash_attention=use_flash_attention,
            )

        if parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:
            assert num_layers % parallel_state.get_virtual_pipeline_model_parallel_world_size() == 0, (
                'num_layers_per_stage must be divisible by ' 'virtual_pipeline_model_parallel_size'
            )

            # self.model_type != ModelType.encoder_and_decoder
            assert self.model_type.value != 2, f'virtual pipeline parallel currently only supported for GPT'

            # Number of layers in each model chunk is the number of layers in the stage,
            # divided by the number of model chunks in a stage.
            self.num_layers = self.num_layers // parallel_state.get_virtual_pipeline_model_parallel_world_size()
            # With 8 layers, 2 stages, and 4 model chunks, we want an assignment of
            # layers to stages like (each list is a model chunk):
            # Stage 0: [0]  [2]  [4]  [6]
            # Stage 1: [1]  [3]  [5]  [7]
            # With 8 layers, 2 stages, and 2 virtual stages, we want an assignment of
            # layers to stages like (each list is a model chunk):
            # Stage 0: [0, 1]  [4, 5]
            # Stage 1: [2, 3]  [6, 7]
            offset = parallel_state.get_virtual_pipeline_model_parallel_rank() * (
                num_layers // parallel_state.get_virtual_pipeline_model_parallel_world_size()
            ) + (parallel_state.get_pipeline_model_parallel_rank() * self.num_layers)
        else:
            # Each stage gets a contiguous set of layers.
            if (
                self.model_type == ModelType.encoder_and_decoder
                and parallel_state.get_pipeline_model_parallel_world_size() > 1
            ):
                pipeline_rank = parallel_state.get_pipeline_model_parallel_rank()
                if layer_type == LayerType.encoder:
                    offset = pipeline_rank * self.num_layers
                else:
                    num_ranks_in_enc = parallel_state.get_pipeline_model_parallel_split_rank()
                    offset = (pipeline_rank - num_ranks_in_enc) * self.num_layers
            else:
                offset = parallel_state.get_pipeline_model_parallel_rank() * self.num_layers

        self.layers = torch.nn.ModuleList([build_layer(i + 1 + offset) for i in range(self.num_layers)])
