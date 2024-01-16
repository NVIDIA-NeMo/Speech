# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from typing import Dict

from omegaconf import DictConfig

try:
    from nemo.collections.nlp.modules.common.megatron.adapters.mcore_mixins import (
        MCoreGPTEmbeddingMixin,
        MCoreMLPMixin,
        MCoreSelfAttentionMixin,
        MCoreTransformerLayerMixin,
    )
except (ImportError, ModuleNotFoundError):
    MCoreGPTEmbeddingMixin = MCoreSelfAttentionMixin = MCoreTransformerLayerMixin = MCoreMLPMixin = None

from nemo.collections.nlp.modules.common.megatron.adapters.parallel_adapters import (
    AdapterName,
    InfusedAdapterConfig,
    LoraKQVAdapterConfig,
    LoraKQVAdapterWeightTyingConfig,
    MLPInfusedAdapterConfig,
    ParallelLinearAdapterConfig,
    ParallelLinearAdapterWeightTyingConfig,
    PromptEncoderAdapterConfig,
)


class PEFTConfig:
    # superclass for adapter name and config
    def __init__(self, peft_cfg: DictConfig, name_key_to_cfg: Dict):
        self.name_key_to_cfg = name_key_to_cfg

        self.layer_selection = peft_cfg.get("layer_selection", None)
        self.weight_tying = peft_cfg.get(
            "weight_tying", False
        )  # TODO: move this attr to LoraPEFTConfig and AdapterPEFTConfig classes

    def get_config_dict(self):
        return self.name_key_to_cfg


class SelectivePEFTConfig(PEFTConfig):
    def __init__(self, cfg):
        selective_cfg = cfg.peft.selective_tuning
        super().__init__(selective_cfg, name_key_to_cfg={})
        self.tunable_base_param_names = selective_cfg.get("tunable_base_param_names", [])


class LoraPEFTConfig(PEFTConfig):
    def __init__(self, cfg):
        lora_cfg = cfg.peft.lora_tuning
        if cfg.get("kv_channels", None) is None:
            assert (
                cfg.hidden_size % cfg.num_attention_heads == 0
            ), 'hidden_size must be divisible by num_attention_heads if kv_channels is None'
            kv_channels = cfg.hidden_size // cfg.num_attention_heads
        else:
            kv_channels = cfg.kv_channels
        projection_size = kv_channels * cfg.num_attention_heads
        num_query_groups = cfg.get("num_query_groups", None)
        if num_query_groups is None:
            num_query_groups = cfg.num_attention_heads
        qkv_projection_size = projection_size + (2 * kv_channels * num_query_groups)

        config_args = {
            "in_features": cfg.hidden_size,
            "out_features": qkv_projection_size,
            "dim": lora_cfg.adapter_dim,
            "norm_position": None,
            "norm_type": None,
            "activation": "identity",
            "column_init_method": lora_cfg.get("column_init_method", "normal"),
            "row_init_method": lora_cfg.get("row_init_method", "zero"),
            "gather_output": False,
            "dropout": lora_cfg.adapter_dropout,
        }

        if lora_cfg.weight_tying:
            position_embedding_strategy = lora_cfg.get("position_embedding_strategy", None)
            if position_embedding_strategy is None:
                dim_position_embeddings = 0
            elif position_embedding_strategy == "add":
                dim_position_embeddings = cfg.hidden_size
            elif position_embedding_strategy == "biasadd":
                dim_position_embeddings = 3 * projection_size
            elif position_embedding_strategy == "concat":
                dim_position_embeddings = lora_cfg.adapter_dim
            elif position_embedding_strategy == "mlpconcat":
                dim_position_embeddings = lora_cfg.adapter_dim
            else:
                raise RuntimeError(
                    f"Unknown position embedding strategy {position_embedding_strategy} for tied weights"
                )
            config_args.update(
                {
                    "num_position_embeddings": cfg.num_layers,
                    "dim_position_embeddings": dim_position_embeddings,
                    "position_embedding_strategy": position_embedding_strategy,
                }
            )
            adapter_cfg = LoraKQVAdapterWeightTyingConfig(**config_args)
        else:
            adapter_cfg = LoraKQVAdapterConfig(**config_args)

        name_key_to_cfg = {
            AdapterName.LORA_KQV_ADAPTER: adapter_cfg,
        }
        self.name_key_to_mcore_mixins = {AdapterName.LORA_KQV_ADAPTER: [("self_attention", MCoreSelfAttentionMixin)]}

        super().__init__(lora_cfg, name_key_to_cfg)


class IA3PEFTConfig(PEFTConfig):
    def __init__(self, cfg):
        mlp_infused_adapter_cfg = MLPInfusedAdapterConfig(
            in_features=cfg.ffn_hidden_size // cfg.tensor_model_parallel_size
        )
        infused_adapter_cfg = InfusedAdapterConfig(in_features=cfg.hidden_size // cfg.tensor_model_parallel_size)

        name_key_to_cfg = {
            AdapterName.KEY_INFUSED: infused_adapter_cfg,
            AdapterName.VALUE_INFUSED: infused_adapter_cfg,
            AdapterName.MLP_INFUSED: mlp_infused_adapter_cfg,
        }
        self.name_key_to_mcore_mixins = {
            AdapterName.KEY_INFUSED: [("self_attention", MCoreSelfAttentionMixin)],
            AdapterName.VALUE_INFUSED: [("self_attention", MCoreSelfAttentionMixin)],
            AdapterName.MLP_INFUSED: [("mlp", MCoreMLPMixin)],
        }

        super().__init__(cfg.peft.ia3_tuning, name_key_to_cfg)


class PtuningPEFTConfig(PEFTConfig):
    def __init__(self, cfg):
        adapter_cfg = PromptEncoderAdapterConfig(
            cfg.peft.p_tuning.virtual_tokens,
            cfg.peft.p_tuning.bottleneck_dim,
            cfg.peft.p_tuning.embedding_dim,
            cfg.peft.p_tuning.init_std,
            cfg.hidden_size,
        )
        name_key_to_cfg = {AdapterName.PTUNING_ADAPTER: adapter_cfg}
        self.name_key_to_mcore_mixins = {AdapterName.PTUNING_ADAPTER: [('embedding', MCoreGPTEmbeddingMixin)]}
        self.virtual_tokens = cfg.peft.p_tuning.virtual_tokens

        super().__init__(cfg.peft.p_tuning, name_key_to_cfg)


class CanonicalAdaptersPEFTConfig(PEFTConfig):
    def __init__(self, cfg):
        adapter_tuning_cfg = cfg.peft.adapter_tuning

        config_args = {
            "in_features": cfg.hidden_size,
            "out_features": cfg.hidden_size,
            "dim": adapter_tuning_cfg.adapter_dim,
            "norm_position": adapter_tuning_cfg.get("norm_position", "pre"),
            "norm_type": adapter_tuning_cfg.get("norm_type", "mixedfusedlayernorm"),
            "column_init_method": adapter_tuning_cfg.get("column_init_method", "xavier"),
            "row_init_method": adapter_tuning_cfg.get("row_init_method", "zero"),
            "dropout": adapter_tuning_cfg.adapter_dropout,
        }

        if adapter_tuning_cfg.weight_tying:
            config_args.update(
                {
                    "num_position_embeddings": cfg.num_layers * 2,
                    "dim_position_embeddings": cfg.hidden_size,
                    "position_embedding_strategy": adapter_tuning_cfg.get("position_embedding_strategy", None),
                }
            )
            adapter_cfg = ParallelLinearAdapterWeightTyingConfig(**config_args)
        else:
            adapter_cfg = ParallelLinearAdapterConfig(**config_args)

        name_key_to_cfg = {
            AdapterName.PRE_ATTN_ADAPTER: adapter_cfg,
            AdapterName.POST_ATTN_ADAPTER: adapter_cfg,
        }
        self.name_key_to_mcore_mixins = {
            AdapterName.PRE_ATTN_ADAPTER: [("", MCoreTransformerLayerMixin)],
            AdapterName.POST_ATTN_ADAPTER: [("", MCoreTransformerLayerMixin)],
        }

        super().__init__(adapter_tuning_cfg, name_key_to_cfg)


PEFT_CONFIG_MAP = {
    "adapter": CanonicalAdaptersPEFTConfig,
    "ia3": IA3PEFTConfig,
    "ptuning": PtuningPEFTConfig,
    "lora": LoraPEFTConfig,
    "selective": SelectivePEFTConfig,
    'none': None,
    None: None,
}
