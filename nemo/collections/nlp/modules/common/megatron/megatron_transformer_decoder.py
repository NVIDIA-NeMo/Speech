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

"""Transformer based language model."""

from nemo.collections.nlp.modules.common.megatron.layer_type import LayerType
from nemo.collections.nlp.modules.common.megatron.megatron_decoder_module import MegatronDecoderModule
from nemo.collections.nlp.modules.common.megatron.module import MegatronModule
from nemo.collections.nlp.modules.common.megatron.transformer import ParallelTransformer
from nemo.collections.nlp.modules.common.megatron.utils import (
    ApexGuardDefaults,
    attn_mask_postprocess,
    build_attention_mask_3d,
)
from nemo.core.classes.exportable import Exportable

try:
    from apex.transformer.enums import AttnMaskType, ModelType

    HAVE_APEX = True
except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False
    # fake missing classes with None attributes
    AttnMaskType = ApexGuardDefaults()
    LayerType = ApexGuardDefaults()
    ModelType = ApexGuardDefaults()


__all__ = ["MegatronTransformerDecoderModule"]


class MegatronTransformerDecoderModule(MegatronModule, Exportable, MegatronDecoderModule):
    """Transformer decoder model.
    """

    def __init__(
        self,
        init_method,
        output_layer_init_method,
        hidden_size,
        ffn_hidden_size,
        num_layers,
        num_attention_heads,
        apply_query_key_layer_scaling=True,
        kv_channels=None,
        pre_process=True,
        post_process=True,
        use_cpu_initialization=False,
        decoder_attn_mask_type=AttnMaskType.causal,
        hidden_dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.0,
        precision=16,
        fp32_residual_connection=False,
        activations_checkpoint_method=None,
        activations_checkpoint_num_layers=1,
        activations_checkpoint_granularity=None,
        layernorm_epsilon=1e-5,
        bias_activation_fusion=True,
        bias_dropout_add_fusion=True,
        masked_softmax_fusion=True,
        persist_layer_norm=False,
        openai_gelu=False,
        onnx_safe=False,
        activation='gelu',
        bias=True,
        normalization='layernorm',
        transformer_block_type='pre_ln',
        headscale=False,
        parent_model_type=ModelType.encoder_or_decoder,
        megatron_legacy=False,
        normalize_attention_scores=True,
        num_moe_experts=1,
        moe_frequency=1,
        moe_dropout=0.0,
    ):
        super(MegatronTransformerDecoderModule, self).__init__()

        self.pre_process = pre_process
        self.post_process = post_process
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.init_method = init_method
        self.model_attn_mask_type = decoder_attn_mask_type
        self.hidden_dropout = hidden_dropout
        self.output_layer_init_method = output_layer_init_method
        self.parent_model_type = parent_model_type
        self.normalization = normalization

        if kv_channels is None:

            assert (
                hidden_size % num_attention_heads == 0
            ), 'hidden_size must be divisible by num_attention_heads if kv_channels is None'
            kv_channels = hidden_size // num_attention_heads

        # Transformer.
        self.model = ParallelTransformer(
            layer_type=LayerType.decoder,
            init_method=self.init_method,
            output_layer_init_method=self.output_layer_init_method,
            num_layers=self.num_layers,
            hidden_size=self.hidden_size,
            num_attention_heads=num_attention_heads,
            apply_query_key_layer_scaling=apply_query_key_layer_scaling,
            kv_channels=kv_channels,
            ffn_hidden_size=ffn_hidden_size,
            self_attn_mask_type=self.model_attn_mask_type,
            pre_process=self.pre_process,
            post_process=self.post_process,
            precision=precision,
            fp32_residual_connection=fp32_residual_connection,
            activations_checkpoint_method=activations_checkpoint_method,
            activations_checkpoint_num_layers=activations_checkpoint_num_layers,
            activations_checkpoint_granularity=activations_checkpoint_granularity,
            layernorm_epsilon=layernorm_epsilon,
            hidden_dropout=hidden_dropout,
            attention_dropout=attention_dropout,
            ffn_dropout=ffn_dropout,
            use_cpu_initialization=use_cpu_initialization,
            bias_activation_fusion=bias_activation_fusion,
            bias_dropout_add_fusion=bias_dropout_add_fusion,
            masked_softmax_fusion=masked_softmax_fusion,
            persist_layer_norm=persist_layer_norm,
            openai_gelu=openai_gelu,
            onnx_safe=onnx_safe,
            activation=activation,
            bias=bias,
            normalization=normalization,
            model_type=parent_model_type,
            transformer_block_type=transformer_block_type,
            headscale=headscale,
            gradient_accumulation_fusion=False,  # TODO: This has to be False for enc-dec models for now.
            megatron_legacy=megatron_legacy,
            normalize_attention_scores=normalize_attention_scores,
            num_moe_experts=num_moe_experts,
            moe_frequency=moe_frequency,
            moe_dropout=moe_dropout,
        )
        self._model_key = 'model'

    def set_input_tensor(self, input_tensor):
        """ See megatron.model.transformer.set_input_tensor()"""
        self.model.set_input_tensor(input_tensor)

    def forward(
        self,
        dec_input,
        dec_attn_mask,
        enc_output,
        enc_attn_mask,
        layer_past=None,
        get_key_value=False,
        dec_self_attention_relative_position_bias=None,
        dec_cross_attention_relative_position_bias=None,
    ):
        # convert to Megatron mask
        dec_attn_mask_3d = build_attention_mask_3d(
            source_mask=dec_attn_mask, target_mask=dec_attn_mask, attn_mask_type=self.model_attn_mask_type,
        )
        enc_dec_attn_mask_3d = build_attention_mask_3d(
            source_mask=dec_attn_mask, target_mask=enc_attn_mask, attn_mask_type=AttnMaskType.padding,
        )

        # transformer decoder
        dec_output = self.model(
            dec_input,
            attn_mask_postprocess(dec_attn_mask_3d),
            layer_past=layer_past,
            get_key_value=get_key_value,
            encoder_output=enc_output,
            enc_dec_attn_mask=attn_mask_postprocess(enc_dec_attn_mask_3d),
            self_attention_relative_position_bias=dec_self_attention_relative_position_bias,
            cross_attention_relative_position_bias=dec_cross_attention_relative_position_bias,
        )

        return dec_output

    def state_dict_for_save_checkpoint(self, destination=None, prefix='', keep_vars=False):
        """For easy load."""

        state_dict_ = {}

        state_dict_[self._model_key] = self.model.state_dict_for_save_checkpoint(destination, prefix, keep_vars)

        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""

        # Encoder.
        if self._model_key in state_dict:
            state_dict_ = state_dict[self._model_key]
        # for backward compatibility.
        elif 'transformer' in state_dict:
            state_dict_ = state_dict['transformer']
        else:
            # for backward compatibility.
            state_dict_ = {}
            for key in state_dict.keys():
                if 'transformer.' in key:
                    state_dict_[key.split('transformer.')[1]] = state_dict[key]

        # for backward compatibility.
        state_dict_self_attention = {}
        for key in state_dict_.keys():
            if '.attention.' in key:
                state_dict_self_attention[key.replace(".attention.", ".self_attention.")] = state_dict_[key]
            else:
                state_dict_self_attention[key] = state_dict_[key]
        state_dict_ = state_dict_self_attention

        self.model.load_state_dict(state_dict_, strict=strict)
