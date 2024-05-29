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

import dataclasses
from dataclasses import dataclass, field
from typing import Dict, List, get_args, get_origin

import numpy as np
import tensorrt as trt
import torch.nn as nn
from tensorrt_llm._utils import pad_vocab_size
from tensorrt_llm.functional import is_gated_activation
from transformers import LlamaConfig, PretrainedConfig
from transformers.models.llama.modeling_llama import LlamaRMSNorm

from nemo.export.trt_llm.tensor_utils import get_tensor_from_dict, split, torch_to_numpy_with_dtype


DECODER_GPT2 = "gpt2"
DECODER_GPTJ = "gptj"
DECODER_LLAMA = "llama"
DECODER_GPTNEXT = "gptnext"
DECODER_FALCON = "falcon"
DECODER_GEMMA = "gemma"

QUANTIZATION_NONE = ""
QUANTIZATION_FP8 = "fp8"
QUANTIZATION_INT8_SQ = "int8_sq"

LINEAR_COLUMN = "column"
LINEAR_ROW = "row"

LAYERNORM_DEFAULT = ""
LAYERNORM_RMS = "rms"

LAYER_DEFAULT = ""
LAYER_QKV = "qkv"


@dataclass
class EmbeddingConfig:
    """The embedding layer config."""

    weight: np.array = None
    # Whether the embedding weights are local
    is_local: bool = False

    @staticmethod
    def from_nn_module(module: nn.Module, dtype=trt.float16):
        """Converts an nn.Module to an EmbeddingConfig."""
        return EmbeddingConfig(weight=torch_to_numpy_with_dtype(module.weight, dtype))

    @property
    def local_vocab_size(self):
        """Infers the vocab_size from the embedding layer weights shape."""
        return self.weight.shape[0]

    @property
    def hidden_size(self):
        """Infers the hidden_size from the embedding layer weights shape."""
        return self.weight.shape[1]


@dataclass
class LayernormConfig:
    """The layernorm layer config."""

    weight: np.array = None
    bias: np.array = None
    layernorm_type: str = LAYERNORM_DEFAULT

    @staticmethod
    def from_nn_module(module: nn.Module, dtype=trt.float16):
        """Converts an nn.Module to an LayernormConfig."""
        layernorm_type = LAYERNORM_RMS if type(module) is LlamaRMSNorm else LAYERNORM_DEFAULT

        config = LayernormConfig(weight=torch_to_numpy_with_dtype(module.weight, dtype), layernorm_type=layernorm_type)
        if layernorm_type == LAYERNORM_DEFAULT:
            config.bias = torch_to_numpy_with_dtype(module.bias, dtype)

        return config


@dataclass
class LinearConfig:
    """The linear layer config."""

    linear_type: str = ""
    weight: np.array = None
    bias: np.array = None
    activation_scaling_factor: np.array = None
    weights_scaling_factor: np.array = None
    prequant_scaling_factor: np.array = None
    layer_type: str = LAYER_DEFAULT

    @staticmethod
    def from_nn_module(module: nn.Module, linear_type: str, rank=0, tensor_parallel=1, dtype=trt.float16):
        """Converts an nn.Module to an LinearConfig."""
        weight = torch_to_numpy_with_dtype(module.weight, dtype)
        if "Conv1D" in type(module).__name__:
            weight = weight.transpose()
        else:
            assert type(module) is nn.Linear

        config = LinearConfig()
        config.linear_type = linear_type
        config.weight = np.ascontiguousarray(
            split(weight, tensor_parallel, rank, dim=0 if linear_type == LINEAR_COLUMN else 1)
        )

        if hasattr(module, "bias") and module.bias is not None:
            if linear_type == LINEAR_COLUMN:
                config.bias = np.ascontiguousarray(
                    split(
                        torch_to_numpy_with_dtype(module.bias, dtype),
                        tensor_parallel,
                        rank,
                    )
                )
            else:
                config.bias = torch_to_numpy_with_dtype(module.bias, dtype)

        return config

    @staticmethod
    def from_qkv_nn_modules(qkv_modules: List[nn.Module], rank=0, tensor_parallel=1, dtype=trt.float16):
        """Converts the qkv modules to an LinearConfig."""
        config = LinearConfig()
        config.linear_type = LINEAR_COLUMN
        config.layer_type = LAYER_QKV
        if len(qkv_modules) == 1:
            # QKV layers combined as a single module, e.g. GPT2
            qkv_module = qkv_modules[0]
            assert "Conv1D" in type(qkv_module).__name__

            qkv_shape = qkv_module.weight.shape
            # Decode the concat QKV weights and split them to different GPU rank.
            config.weight = np.ascontiguousarray(
                split(
                    torch_to_numpy_with_dtype(qkv_module.weight, dtype=dtype).reshape(
                        qkv_shape[0], 3, qkv_shape[-1] // 3
                    ),
                    tensor_parallel,
                    rank,
                    dim=-1,
                )
                .reshape(qkv_shape[0], -1)
                .transpose()
            )
            config.bias = np.ascontiguousarray(
                split(
                    torch_to_numpy_with_dtype(qkv_module.bias, dtype=dtype).reshape(3, qkv_shape[-1] // 3),
                    tensor_parallel,
                    rank,
                    dim=-1,
                ).reshape(-1)
            )

        elif len(qkv_modules) == 3:
            # Separate QKV layers
            for m in qkv_modules:
                assert type(m) is nn.Linear
                assert not (hasattr(m, "bias") and m.bias is not None)

            q_weight = split(torch_to_numpy_with_dtype(qkv_modules[0].weight), tensor_parallel, rank)
            k_weight = split(torch_to_numpy_with_dtype(qkv_modules[1].weight), tensor_parallel, rank)
            v_weight = split(torch_to_numpy_with_dtype(qkv_modules[2].weight), tensor_parallel, rank)
            split_v = np.concatenate((q_weight, k_weight, v_weight))
            config.weight = np.ascontiguousarray(split_v)

        else:
            assert False, f"QKV modules format {qkv_modules} not supported"

        return config


@dataclass
class MoEMLPConfig:
    """The MLP layer config."""

    fc1: LinearConfig = None
    fc2: LinearConfig = None
    router: LinearConfig = None
    hidden_act: str = ""

    @staticmethod
    def from_nemo(
        weights_dict: Dict[str, np.ndarray],
        llm_config: PretrainedConfig,
        layer_id: int,
        rank: int = 0,
        is_mcore: bool = False,
    ):
        """Converts the nemo weights and config to `MLPConfig`."""
        mlp = MoEMLPConfig(hidden_act=llm_config.activation_function)
        mlp.fc1 = LinearConfig(linear_type=LINEAR_COLUMN)

        mlp.fc1.weight = get_tensor_from_dict(
            weights_dict, f"layers.{layer_id}.mlp.experts.experts.linear_fc1.weight.{rank}"
        )

        mlp.fc1.bias = get_tensor_from_dict(
            weights_dict, f"layers.{layer_id}.mlp.experts.experts.linear_fc1.bias.{rank}"
        )

        mlp.fc2 = LinearConfig(linear_type=LINEAR_ROW)
        mlp.fc2.weight = get_tensor_from_dict(
            weights_dict, f"layers.{layer_id}.mlp.experts.experts.linear_fc2.weight.{rank}"
        )
        mlp.fc2.bias = get_tensor_from_dict(
            weights_dict, f"layers.{layer_id}.mlp.experts.experts.linear_fc2.bias.{rank}"
        )

        mlp.router = LinearConfig(linear_type=LINEAR_ROW)
        mlp.router.weight = get_tensor_from_dict(weights_dict, f"layers.{layer_id}.mlp.router.weight.{rank}")
        return mlp


@dataclass
class AttentionConfig:
    """The attention layer config."""

    qkv: LinearConfig = None
    dense: LinearConfig = None

    rotary_dim: int = -np.inf

    @staticmethod
    def from_nemo(
        weights_dict: Dict[str, np.ndarray],
        layer_id: int,
        rank: int = 0,
    ):
        """Converts the nemo weights and config to `AttentionConfig`."""
        attention = AttentionConfig()
        attention.qkv = LinearConfig(linear_type=LINEAR_COLUMN, layer_type=LAYER_QKV)
        attention.qkv.weight = get_tensor_from_dict(
            weights_dict, f"layers.{layer_id}.attention.query_key_value.weight.{rank}"
        )
        attention.qkv.bias = get_tensor_from_dict(
            weights_dict,
            f"layers.{layer_id}.attention.query_key_value.bias.{rank}",
        )

        attention.dense = LinearConfig(linear_type=LINEAR_ROW)
        attention.dense.weight = get_tensor_from_dict(weights_dict, f"layers.{layer_id}.attention.dense.weight.{rank}")
        attention.dense.bias = get_tensor_from_dict(
            weights_dict,
            f"layers.{layer_id}.attention.dense.bias",
        )
        return attention


@dataclass
class MLPConfig:
    """The MLP layer config."""

    fc: LinearConfig = None
    gate: LinearConfig = None
    proj: LinearConfig = None
    hidden_act: str = ""

    @staticmethod
    def from_nemo(
        weights_dict: Dict[str, np.ndarray],
        llm_config: PretrainedConfig,
        layer_id: int,
        rank: int = 0,
        is_mcore: bool = False,
    ):
        """Converts the nemo weights and config to `MLPConfig`."""
        mlp = MLPConfig(hidden_act=llm_config.activation_function)
        mlp.fc = LinearConfig(linear_type=LINEAR_COLUMN)
        mlp.fc.weight = get_tensor_from_dict(weights_dict, f"layers.{layer_id}.mlp.dense_h_to_4h.weight.{rank}")

        # print("********** mlp.fc.weight : ", mlp.fc.weight )

        mlp.fc.bias = get_tensor_from_dict(
            weights_dict,
            f"layers.{layer_id}.mlp.dense_h_to_4h.bias.{rank}",
        )

        gated = is_gated_activation(mlp.hidden_act)
        is_fast_glu = mlp.hidden_act in ['fast-geglu', 'fast-swiglu', 'fast-reglu']
        if gated:
            mlp.gate = LinearConfig(linear_type=LINEAR_COLUMN)
            layer_name = (
                f"layers.{layer_id}.mlp.dense_h_to_4h_2.weight.{rank}"
                if isinstance(llm_config, LlamaConfig) and not is_mcore and not is_fast_glu
                else f"layers.{layer_id}.mlp.dense_h_to_4h.gate.weight.{rank}"
            )
            mlp.gate.weight = get_tensor_from_dict(
                weights_dict,
                layer_name,
            )
            mlp.gate.bias = get_tensor_from_dict(
                weights_dict,
                f"layers.{layer_id}.mlp.dense_h_to_4h.gate.bias.{rank}",
            )

        mlp.proj = LinearConfig(linear_type=LINEAR_ROW)
        mlp.proj.weight = get_tensor_from_dict(weights_dict, f"layers.{layer_id}.mlp.dense_4h_to_h.weight.{rank}")
        mlp.proj.bias = get_tensor_from_dict(weights_dict, f"layers.{layer_id}.mlp.dense_4h_to_h.bias")
        return mlp


@dataclass
class DecoderLayerConfig:
    """The decoder layer config."""

    decoder_type: str = ""
    input_layernorm: LayernormConfig = None
    mlp_layernorm: LayernormConfig = None  # Falcon 40B/180B has mlp_layernorm
    attention: AttentionConfig = None
    post_layernorm: LayernormConfig = None
    mlp: MLPConfig = None

    num_attention_heads: int = 0

    num_kv_heads: int = 0
    kv_channels: int = None
    max_position_embeddings: int = 0
    rotary_pct: float = 0
    rotary_base: int = 10000
    rotary_scaling: float = None
    position_embedding_type: str = None

    moe_num_experts: int = None
    moe_top_k: int = None
    moe_tp_mode: int = None
    moe_renorm_mode: int = None

    vocab_size: int = 0
    norm_epsilon: float = 0.0
    max_lora_rank: int = 64

    @property
    def is_moe(self):
        return self.moe_num_experts is not None and self.moe_num_experts > 1

    @property
    def hidden_size(self):
        """Returns the hidden size of the transformer model."""
        if self.is_moe:
            return self.mlp.fc2.weight.shape[1]
        else:
            return self.mlp.fc.weight.shape[1]

    @property
    def ffn_hidden_size_local(self):
        """Returns the ffn hidden size of the transformer model."""
        if self.is_moe:
            return self.mlp.fc2.weight.shape[-1]
        else:
            return self.mlp.fc.weight.shape[0]

    @staticmethod
    def from_nemo(
        weights_dict: Dict[str, np.ndarray],
        llm_config: PretrainedConfig,
        decoder_type: str,
        layer_id: int,
        rank: int = 0,
        is_mcore: bool = False,
    ):
        """Converts the nemo weights and config to `DecoderLayerConfig`."""
        layer_config = DecoderLayerConfig(
            decoder_type=decoder_type,
            num_attention_heads=llm_config.n_head,
            max_position_embeddings=llm_config.n_positions,
            rotary_pct=llm_config.rotary_pct if hasattr(llm_config, "rotary_pct") else 1.0,
            rotary_base=(llm_config.rotary_base if hasattr(llm_config, "rotary_base") else 10000),
            rotary_scaling=(llm_config.rotary_scaling if hasattr(llm_config, "rotary_scaling") else None),
            position_embedding_type=(
                llm_config.position_embedding_type if hasattr(llm_config, "position_embedding_type") else None
            ),
            num_kv_heads=(llm_config.num_kv_heads if hasattr(llm_config, "num_kv_heads") else 0),
            kv_channels=(llm_config.kv_channels if hasattr(llm_config, "kv_channels") else None),
            moe_num_experts=(llm_config.moe_num_experts if hasattr(llm_config, "moe_num_experts") else None),
            moe_top_k=(llm_config.moe_top_k if hasattr(llm_config, "moe_top_k") else None),
            moe_tp_mode=(llm_config.moe_tp_mode if hasattr(llm_config, "moe_tp_mode") else None),
            moe_renorm_mode=(llm_config.moe_renorm_mode if hasattr(llm_config, "moe_renorm_mode") else None),
            vocab_size=llm_config.vocab_size,
            norm_epsilon=llm_config.norm_epsilon,
        )
        layer_config.input_layernorm = LayernormConfig()
        layer_config.input_layernorm.layernorm_type = (
            LAYERNORM_RMS if isinstance(llm_config, LlamaConfig) else LAYERNORM_DEFAULT
        )
        layer_config.input_layernorm.weight = get_tensor_from_dict(
            weights_dict,
            f"layers.{layer_id}.input_layernorm.weight",
        )
        layer_config.input_layernorm.bias = get_tensor_from_dict(
            weights_dict,
            f"layers.{layer_id}.input_layernorm.bias",
        )

        layer_config.mlp_layernorm = LayernormConfig()
        layer_config.mlp_layernorm.layernorm_type = LAYERNORM_DEFAULT  # Falcon uses default layernorm
        layer_config.mlp_layernorm.weight = get_tensor_from_dict(
            weights_dict,
            f"layers.{layer_id}.pre_mlp_layernorm.weight",
        )
        layer_config.mlp_layernorm.bias = get_tensor_from_dict(
            weights_dict,
            f"layers.{layer_id}.pre_mlp_layernorm.bias",
        )

        layer_config.post_layernorm = LayernormConfig()
        layer_config.post_layernorm.layernorm_type = (
            LAYERNORM_RMS if isinstance(llm_config, LlamaConfig) else LAYERNORM_DEFAULT
        )

        layer_config.post_layernorm.weight = get_tensor_from_dict(
            weights_dict,
            f"layers.{layer_id}.post_attention_layernorm.weight",
        )
        layer_config.post_layernorm.bias = get_tensor_from_dict(
            weights_dict,
            f"layers.{layer_id}.post_attention_layernorm.bias",
        )

        if layer_config.post_layernorm.weight is None:  # Falcon doesn't have post layernorm
            layer_config.post_layernorm = None

        if layer_config.mlp_layernorm.weight is None:
            layer_config.mlp_layernorm = None

        layer_config.attention = AttentionConfig.from_nemo(
            weights_dict,
            layer_id,
            rank,
        )

        moe = False
        if llm_config.moe_num_experts is not None:
            if llm_config.moe_num_experts > 1:
                moe = True

        if moe:
            layer_config.mlp = MoEMLPConfig.from_nemo(weights_dict, llm_config, layer_id, rank, is_mcore)
        else:
            layer_config.mlp = MLPConfig.from_nemo(weights_dict, llm_config, layer_id, rank, is_mcore)

        return layer_config


def _from_dict(class_type, data):
    """Helper function to load the data as a class_type. class_type must be a dataclass."""
    if data is None:
        return None

    if dataclasses.is_dataclass(class_type):
        fieldtypes = {f.name: f.type for f in dataclasses.fields(class_type)}
        return class_type(**{f: _from_dict(fieldtypes[f], data[f]) for f in data})
    elif get_origin(class_type) == list and dataclasses.is_dataclass(get_args(class_type)[0]):
        list_value = []
        for child in data:
            child_class_type = get_args(class_type)[0]
            list_value.append(_from_dict(child_class_type, child))
        return list_value
    else:
        return data


@dataclass
class ModelConfig:
    """The full LLM model config that includes the full information needed for tensorrt_llm engine building.

    This class includes all the fields that tensorrt_llm supports, but not all of the fields are required.
    """

    # Global metadata
    quantization: str = QUANTIZATION_NONE
    dtype: str = "float16"

    # Model structure and weights
    vocab_embedding: EmbeddingConfig = None
    positional_embedding: EmbeddingConfig = None
    layers: List[DecoderLayerConfig] = field(default_factory=list)
    final_layernorm: LayernormConfig = None
    lm_head: LinearConfig = None

    # Ptuning metadata
    use_prompt_tuning: bool = False
    use_parallel_embedding: bool = False
    max_lora_rank: int = 64

    # Parallel metadata
    mapping = None

    def to_dict(self) -> dict:
        """Converts the instance to a python dict."""
        return dataclasses.asdict(self)

    @staticmethod
    def from_dict(d: dict):
        """Load a dict to a `ModelConfig` instance."""
        return _from_dict(ModelConfig, d)

    @property
    def vocab_size(self):
        """Returns the vocab_size of the model."""
        return (
            self.vocab_embedding.local_vocab_size * self.mapping.tp_size
            if self.vocab_embedding.is_local
            else self.vocab_embedding.local_vocab_size
        )

    @property
    def vocab_size_padded(self):
        """Returns the padded vocab_size of the model rounds to the tensor_parallel."""
        return pad_vocab_size(self.vocab_size, self.mapping.tp_size)

    @property
    def hidden_size(self):
        """Returns the hidden_size of the model."""
        return self.vocab_embedding.hidden_size

    @property
    def max_position_embeddings(self):
        """Returns the max_position_embedding of the model."""
        return self.layers[0].max_position_embeddings

    @property
    def num_attention_heads(self):
        """Returns the num_attention_heads of the model."""
        return self.layers[0].num_attention_heads

    @property
    def num_kv_heads(self):
        """Returns the num_key_value_heads of the model."""
        return self.layers[0].num_kv_heads if self.layers[0].num_kv_heads > 0 else self.num_attention_heads

    @property
    def head_size(self):
        """Returns the head_size of the model."""
        return self.layers[0].kv_channels

    @property
    def hidden_act(self):
        """Returns the hidden_act of the model."""
        return self.layers[0].mlp.hidden_act
