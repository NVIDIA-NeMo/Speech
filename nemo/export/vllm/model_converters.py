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

from abc import ABC, abstractmethod
from typing import Generator, Optional, Tuple

import torch


class ModelConverter(ABC):
    """
    Abstract class that defines the interface for a converter that implements model-specific conversion functions
    for deploying NeMo checkpoints on vLLM.
    """

    def __init__(self, model_type: str):
        self.model_type = model_type

    @abstractmethod
    def get_architecture(self) -> Optional[str]:
        """
        Returns the HF architecture name for the current model, such as 'LlamaForCausalLM'.
        """
        pass

    def convert_config(self, nemo_model_config: dict, hf_config: dict) -> None:
        """
        Implements any custom HF configuration adjustments in the 'hf_config' dict that are necessary
        for this model after the common translation takes place in NemoModelConfig's constructor.
        """
        pass

    @abstractmethod
    def convert_weights(
        self, nemo_model_config: dict, state_dict: dict
    ) -> Generator[Tuple[str, torch.tensor], None, None]:
        """
        Returns or yields a sequence of (name, tensor) tuples that contain model weights in the HF format.
        """
        pass

    def requires_bos_token(self) -> bool:
        """
        Returns True if the model requires a 'bos' token to be used at the beginning of the input sequence.
        NeMo checkpoints do not store this information.
        """
        return False


class LlamaConverter(ModelConverter):

    def get_architecture(self):
        if self.model_type == 'llama':
            return 'LlamaForCausalLM'
        if self.model_type == 'mistral':
            return 'MistralForCausalLM'
        return None

    def convert_weights(self, nemo_model_config, state_dict):
        hidden_size = nemo_model_config["hidden_size"]
        head_num = nemo_model_config["num_attention_heads"]
        num_query_groups = nemo_model_config["num_query_groups"]
        num_layers = nemo_model_config["num_layers"]
        head_size = hidden_size // head_num
        heads_per_group = head_num // num_query_groups
        qkv_total_dim = head_num + 2 * num_query_groups

        yield ('model.embed_tokens.weight', state_dict['model.embedding.word_embeddings.weight'])
        yield ('model.norm.weight', state_dict['model.decoder.final_layernorm.weight'])
        if not nemo_model_config.get("share_embeddings_and_output_weights", False):
            yield ('lm_head.weight', state_dict['model.output_layer.weight'])

        for layer in range(int(num_layers)):
            qkv_weights = state_dict['model.decoder.layers.self_attention.linear_qkv.weight'][layer]
            qkv_weights = qkv_weights.reshape([qkv_total_dim, head_size, hidden_size])

            q_slice = torch.cat(
                [
                    torch.arange((heads_per_group + 2) * i, (heads_per_group + 2) * i + heads_per_group)
                    for i in range(num_query_groups)
                ]
            )
            k_slice = torch.arange(heads_per_group, qkv_total_dim, (heads_per_group + 2))
            v_slice = torch.arange(heads_per_group + 1, qkv_total_dim, (heads_per_group + 2))

            for name, slice in [('q_proj', q_slice), ('k_proj', k_slice), ('v_proj', v_slice)]:
                weight_name = f'model.layers.{layer}.self_attn.{name}.weight'
                yield (weight_name, qkv_weights[slice].reshape(-1, hidden_size))

            linear_proj_weight = state_dict['model.decoder.layers.self_attention.linear_proj.weight'][layer]
            yield (f'model.layers.{layer}.self_attn.o_proj.weight', linear_proj_weight)

            gate_proj_weight, up_proj_weight = torch.chunk(
                state_dict['model.decoder.layers.mlp.linear_fc1.weight'][layer], 2, dim=0
            )
            yield (f'model.layers.{layer}.mlp.gate_proj.weight', gate_proj_weight)
            yield (f'model.layers.{layer}.mlp.up_proj.weight', up_proj_weight)

            mlp_up_weight = state_dict['model.decoder.layers.mlp.linear_fc2.weight'][layer]
            yield (f'model.layers.{layer}.mlp.down_proj.weight', mlp_up_weight)

            input_layernorm_weight = state_dict['model.decoder.layers.self_attention.linear_qkv.layer_norm_weight'][
                layer
            ]
            yield (f'model.layers.{layer}.input_layernorm.weight', input_layernorm_weight)

            post_attn_layernorm_weight = state_dict['model.decoder.layers.mlp.linear_fc1.layer_norm_weight'][layer]
            yield (f'model.layers.{layer}.post_attention_layernorm.weight', post_attn_layernorm_weight)

    def requires_bos_token(self):
        return True


class MixtralConverter(ModelConverter):

    def get_architecture(self):
        if self.model_type == 'mixtral':
            return 'MixtralForCausalLM'
        return None

    def convert_weights(self, nemo_model_config, state_dict):
        hidden_size = nemo_model_config["hidden_size"]
        head_num = nemo_model_config["num_attention_heads"]
        num_query_groups = nemo_model_config["num_query_groups"]
        num_layers = nemo_model_config["num_layers"]
        num_moe_experts = nemo_model_config["num_moe_experts"]
        head_size = hidden_size // head_num
        heads_per_group = head_num // num_query_groups
        qkv_total_dim = head_num + 2 * num_query_groups

        yield ('model.embed_tokens.weight', state_dict['model.embedding.word_embeddings.weight'])
        yield ('model.norm.weight', state_dict['model.decoder.final_layernorm.weight'])
        yield ('lm_head.weight', state_dict['model.output_layer.weight'])

        for layer in range(int(num_layers)):
            qkv_weights = state_dict['model.decoder.layers.self_attention.linear_qkv.weight'][layer]
            qkv_weights = qkv_weights.reshape([qkv_total_dim, head_size, hidden_size])

            q_slice = torch.cat(
                [
                    torch.arange((heads_per_group + 2) * i, (heads_per_group + 2) * i + heads_per_group)
                    for i in range(num_query_groups)
                ]
            )
            k_slice = torch.arange(heads_per_group, qkv_total_dim, (heads_per_group + 2))
            v_slice = torch.arange(heads_per_group + 1, qkv_total_dim, (heads_per_group + 2))

            for name, slice in [('q_proj', q_slice), ('k_proj', k_slice), ('v_proj', v_slice)]:
                weight_name = f'model.layers.{layer}.self_attn.{name}.weight'
                yield (weight_name, qkv_weights[slice].reshape(-1, hidden_size))

            linear_proj_weight = state_dict['model.decoder.layers.self_attention.linear_proj.weight'][layer]
            yield (f'model.layers.{layer}.self_attn.o_proj.weight', linear_proj_weight)

            mlp_router_weight = state_dict['model.decoder.layers.mlp.router.weight'][layer]
            yield (f'model.layers.{layer}.block_sparse_moe.gate.weight', mlp_router_weight)

            for expert in range(num_moe_experts):
                linear_fc1_weight = state_dict['model.decoder.layers.mlp.experts.experts.linear_fc1.weight'][layer][
                    expert
                ]
                gate_proj_weight, up_proj_weight = torch.chunk(linear_fc1_weight, 2, dim=0)
                yield (f'model.layers.{layer}.block_sparse_moe.experts.{expert}.w1.weight', gate_proj_weight)
                yield (f'model.layers.{layer}.block_sparse_moe.experts.{expert}.w3.weight', up_proj_weight)

                linear_fc2_weight = state_dict['model.decoder.layers.mlp.experts.experts.linear_fc2.weight'][layer][
                    expert
                ]
                yield (f'model.layers.{layer}.block_sparse_moe.experts.{expert}.w2.weight', linear_fc2_weight)

            input_layernorm_weight = state_dict['model.decoder.layers.self_attention.linear_qkv.layer_norm_weight'][
                layer
            ]
            yield (f'model.layers.{layer}.input_layernorm.weight', input_layernorm_weight)

            post_attn_layernorm_weight = state_dict['model.decoder.layers.pre_mlp_layernorm.weight'][layer]
            yield (f'model.layers.{layer}.post_attention_layernorm.weight', post_attn_layernorm_weight)

    def requires_bos_token(self):
        return True


class GemmaConverter(ModelConverter):

    def get_architecture(self):
        if self.model_type == 'gemma':
            return 'GemmaForCausalLM'
        return None

    def convert_weights(self, nemo_model_config, state_dict):
        num_layers = nemo_model_config["num_layers"]
        num_query_groups = nemo_model_config["num_query_groups"]
        head_num = nemo_model_config["num_attention_heads"]
        head_size = nemo_model_config["kv_channels"]
        hidden_size = nemo_model_config["hidden_size"]
        zero_centered_gamma = nemo_model_config.get("layernorm_zero_centered_gamma", False)
        heads_per_group = head_num // num_query_groups

        yield ('model.embed_tokens.weight', state_dict['model.embedding.word_embeddings.weight'])

        final_layernorm_weight = state_dict['model.decoder.final_layernorm.weight']
        if not zero_centered_gamma:
            final_layernorm_weight -= 1.0
        yield ('model.norm.weight', final_layernorm_weight)

        for layer in range(int(num_layers)):
            input_layernorm_weight = state_dict['model.decoder.layers.self_attention.linear_qkv.layer_norm_weight'][
                layer
            ]
            if not zero_centered_gamma:
                input_layernorm_weight -= 1.0
            yield (f'model.layers.{layer}.input_layernorm.weight', input_layernorm_weight)

            post_attention_layernorm_weight = state_dict['model.decoder.layers.mlp.linear_fc1.layer_norm_weight'][
                layer
            ]
            if not zero_centered_gamma:
                post_attention_layernorm_weight -= 1.0
            yield (f'model.layers.{layer}.post_attention_layernorm.weight', post_attention_layernorm_weight)

            gate_up_combined_weight = state_dict['model.decoder.layers.mlp.linear_fc1.weight'][layer]
            gate_size = gate_up_combined_weight.shape[0] // 2
            yield (f'model.layers.{layer}.mlp.gate_proj.weight', gate_up_combined_weight[:gate_size, :])
            yield (f'model.layers.{layer}.mlp.up_proj.weight', gate_up_combined_weight[gate_size:, :])

            down_proj_weight = state_dict['model.decoder.layers.mlp.linear_fc2.weight'][layer]
            yield (f'model.layers.{layer}.mlp.down_proj.weight', down_proj_weight)

            self_attn_o_proj_weight = state_dict['model.decoder.layers.self_attention.linear_proj.weight'][layer]
            yield (f'model.layers.{layer}.self_attn.o_proj.weight', self_attn_o_proj_weight)

            qkv_weight = state_dict['model.decoder.layers.self_attention.linear_qkv.weight'][layer]
            qkv_intermediate_size = head_num + 2 * num_query_groups
            qkv_weight = qkv_weight.reshape(qkv_intermediate_size, head_size, hidden_size)

            q_weight = torch.empty((head_num, head_size, hidden_size), dtype=qkv_weight.dtype)
            k_weight = torch.empty((num_query_groups, head_size, hidden_size), dtype=qkv_weight.dtype)
            v_weight = torch.empty((num_query_groups, head_size, hidden_size), dtype=qkv_weight.dtype)

            ptr = 0
            for i in range(num_query_groups):
                q_weight[i * heads_per_group : (i + 1) * heads_per_group, :, :] = qkv_weight[
                    ptr : ptr + heads_per_group, ::
                ]
                ptr += heads_per_group
                k_weight[i : i + 1, :, :] = qkv_weight[ptr : ptr + 1, :, :]
                ptr += 1
                v_weight[i : i + 1, :, :] = qkv_weight[ptr : ptr + 1, :, :]
                ptr += 1
            assert ptr == qkv_intermediate_size

            q_weight = q_weight.reshape(head_num * head_size, hidden_size)
            k_weight = k_weight.reshape(num_query_groups * head_size, hidden_size)
            v_weight = v_weight.reshape(num_query_groups * head_size, hidden_size)

            yield (f'model.layers.{layer}.self_attn.q_proj.weight', q_weight)
            yield (f'model.layers.{layer}.self_attn.k_proj.weight', k_weight)
            yield (f'model.layers.{layer}.self_attn.v_proj.weight', v_weight)

    def requires_bos_token(self):
        return True


class Starcoder2Converter(ModelConverter):

    def get_architecture(self):
        if self.model_type == 'starcoder2':
            return 'Starcoder2ForCausalLM'
        return None

    def convert_config(self, nemo_model_config, hf_config):
        window_sizes = nemo_model_config.get('window_size')
        if window_sizes is not None:
            hf_config['sliding_window'] = window_sizes[0]

        # 'tie_word_embeddings = False' means that there is a 'lm_head.weight' tensor.
        # This converter assumes that it's always there.
        # If there is a version of starcoder2 where it's not there, we'll need to copy
        # 'model.embed_tokens.weight' into 'lm_head.weight' and still set 'tie_word_embeddings = False'
        # because at this point we don't know if the weight is there or not, and this configuration
        # is not stored in NeMo checkpoints.
        hf_config['tie_word_embeddings'] = False

    def convert_weights(self, nemo_model_config, state_dict):
        num_layers = nemo_model_config["num_layers"]
        num_query_groups = nemo_model_config["num_query_groups"]
        head_num = nemo_model_config["num_attention_heads"]
        hidden_size = nemo_model_config["hidden_size"]
        head_size = hidden_size // head_num
        heads_per_group = head_num // num_query_groups
        qkv_total_dim = head_num + 2 * num_query_groups

        if 'bias' in nemo_model_config:
            has_bias = nemo_model_config["bias"]
        else:
            has_bias = nemo_model_config["add_bias_linear"]

        yield ('model.embed_tokens.weight', state_dict['model.embedding.word_embeddings.weight'])

        yield ('model.norm.weight', state_dict['model.decoder.final_layernorm.weight'])
        if has_bias:
            yield ('model.norm.bias', state_dict['model.decoder.final_layernorm.bias'])

        yield ('lm_head.weight', state_dict['model.output_layer.weight'])

        for layer in range(int(num_layers)):
            # q,k,v
            qkv_weights = state_dict['model.decoder.layers.self_attention.linear_qkv.weight'][layer]
            qkv_weights = qkv_weights.reshape([qkv_total_dim, head_size, hidden_size])
            if has_bias:
                qkv_bias = state_dict['model.decoder.layers.self_attention.linear_qkv.bias'][layer]
                qkv_bias = qkv_bias.reshape([qkv_total_dim, head_size])

            q_slice = torch.cat(
                [
                    torch.arange((heads_per_group + 2) * i, (heads_per_group + 2) * i + heads_per_group)
                    for i in range(num_query_groups)
                ]
            )
            k_slice = torch.arange(heads_per_group, qkv_total_dim, (heads_per_group + 2))
            v_slice = torch.arange(heads_per_group + 1, qkv_total_dim, (heads_per_group + 2))

            for name, slice in [('q_proj', q_slice), ('k_proj', k_slice), ('v_proj', v_slice)]:
                qkv_weights_slice = qkv_weights[slice].reshape(-1, hidden_size)
                yield (f'model.layers.{layer}.self_attn.{name}.weight', qkv_weights_slice)
                if has_bias:
                    qkv_bias_slice = qkv_bias[slice].reshape(-1)
                    yield (f'model.layers.{layer}.self_attn.{name}.bias', qkv_bias_slice)

            # Attention dense
            yield (
                f'model.layers.{layer}.self_attn.o_proj.weight',
                state_dict['model.decoder.layers.self_attention.linear_proj.weight'][layer],
            )
            if has_bias:
                yield (
                    f'model.layers.{layer}.self_attn.o_proj.bias',
                    state_dict['model.decoder.layers.self_attention.linear_proj.bias'][layer],
                )

            # MLP FC1
            yield (
                f'model.layers.{layer}.mlp.c_fc.weight',
                state_dict['model.decoder.layers.mlp.linear_fc1.weight'][layer],
            )
            if has_bias:
                yield (
                    f'model.layers.{layer}.mlp.c_fc.bias',
                    state_dict['model.decoder.layers.mlp.linear_fc1.bias'][layer],
                )

            # MLP FC2
            yield (
                f'model.layers.{layer}.mlp.c_proj.weight',
                state_dict['model.decoder.layers.mlp.linear_fc2.weight'][layer],
            )
            if has_bias:
                yield (
                    f'model.layers.{layer}.mlp.c_proj.bias',
                    state_dict['model.decoder.layers.mlp.linear_fc2.bias'][layer],
                )

            # Input LayerNorm
            yield (
                f'model.layers.{layer}.input_layernorm.weight',
                state_dict['model.decoder.layers.self_attention.linear_qkv.layer_norm_weight'][layer],
            )
            if has_bias:
                yield (
                    f'model.layers.{layer}.input_layernorm.bias',
                    state_dict['model.decoder.layers.self_attention.linear_qkv.layer_norm_bias'][layer],
                )

            # Post-attention LayerNorm
            yield (
                f'model.layers.{layer}.post_attention_layernorm.weight',
                state_dict['model.decoder.layers.mlp.linear_fc1.layer_norm_weight'][layer],
            )
            if has_bias:
                yield (
                    f'model.layers.{layer}.post_attention_layernorm.bias',
                    state_dict['model.decoder.layers.mlp.linear_fc1.layer_norm_bias'][layer],
                )


_MODEL_CONVERTERS = {
    'llama': LlamaConverter,
    'mistral': LlamaConverter,
    'mixtral': MixtralConverter,
    'gemma': GemmaConverter,
    'starcoder2': Starcoder2Converter,
}


def register_model_converter(model_type, cls):
    """
    Establishes a mapping from short model type to a class that converts the model from Nemo format
    to a vLLM compatible format.
    """
    _MODEL_CONVERTERS[model_type] = cls


def get_model_converter(model_type) -> Optional[ModelConverter]:
    """
    Returns an instance of the the model conversion class for the given model type, or None.
    """
    cls = _MODEL_CONVERTERS.get(model_type, None)
    if cls is None:
        return None
    return cls(model_type)
