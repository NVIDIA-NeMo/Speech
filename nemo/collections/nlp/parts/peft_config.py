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

from nemo.utils import logging

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
    Lora4HtoHAdapterConfig,
    LoraDenseAttentionAdapterConfig,
    LoraHto4HAdapterConfig,
    LoraKQVAdapterConfig,
    LoraKQVAdapterWeightTyingConfig,
    LoraUnfusedHto4HAdapterConfig,
    LoraUnfusedKQVAdapterConfig,
    MLPInfusedAdapterConfig,
    ParallelLinearAdapterConfig,
    ParallelLinearAdapterWeightTyingConfig,
    PromptEncoderAdapterConfig,
)

PEFT_MODULE_MAP = {
    "qkv_module": "attention_qkv",
    "dense_module": "attention_dense",
    "hto4h_module": "mlp_fc1",
    "4htoh_module": "mlp_fc2",
    "attention": "attention",
    "mlp": "mlp",
    "all": "all",
}


def get_target_modules(lora_cfg):
    original_target_modules = lora_cfg.get("target_modules", ["attention_qkv"])
    target_modules = []

    for module in original_target_modules:
        if module == PEFT_MODULE_MAP["attention"]:
            if PEFT_MODULE_MAP['qkv_module'] not in target_modules:
                target_modules.append(PEFT_MODULE_MAP['qkv_module'])
            if PEFT_MODULE_MAP['dense_module'] not in target_modules:
                target_modules.append(PEFT_MODULE_MAP['dense_module'])
        elif module == PEFT_MODULE_MAP["mlp"]:
            if PEFT_MODULE_MAP['hto4h_module'] not in target_modules:
                target_modules.append(PEFT_MODULE_MAP['hto4h_module'])
            if PEFT_MODULE_MAP['4htoh_module'] not in target_modules:
                target_modules.append(PEFT_MODULE_MAP['4htoh_module'])
        elif module == PEFT_MODULE_MAP["all"]:
            for sub_module in [
                PEFT_MODULE_MAP['qkv_module'],
                PEFT_MODULE_MAP['dense_module'],
                PEFT_MODULE_MAP['hto4h_module'],
                PEFT_MODULE_MAP['4htoh_module'],
            ]:
                if sub_module not in target_modules:
                    target_modules.append(sub_module)
        else:
            if module not in target_modules:
                target_modules.append(module)

    return target_modules


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

    def _calculate_kv_channels(self, cfg):
        if cfg.get("kv_channels", None) is None:
            assert (
                cfg.hidden_size % cfg.num_attention_heads == 0
            ), 'hidden_size must be divisible by num_attention_heads if kv_channels is None'
            kv_channels = cfg.hidden_size // cfg.num_attention_heads
        else:
            kv_channels = cfg.kv_channels
        return kv_channels


class SelectivePEFTConfig(PEFTConfig):
    def __init__(self, cfg):
        selective_cfg = cfg.peft.selective_tuning
        super().__init__(selective_cfg, name_key_to_cfg={})
        self.tunable_base_param_names = selective_cfg.get("tunable_base_param_names", [])


class LoraPEFTConfig(PEFTConfig):
    def __init__(self, cfg):
        lora_cfg = cfg.peft.lora_tuning
        kv_channels = self._calculate_kv_channels(cfg)
        projection_size = kv_channels * cfg.num_attention_heads
        num_query_groups = cfg.get("num_query_groups", cfg.num_attention_heads)
        if num_query_groups is None:
            # Cover the case where num_query_groups is explicitly set to null
            num_query_groups = cfg.num_attention_heads

        qkv_projection_size = projection_size + (2 * kv_channels * num_query_groups)

        fast_glu_activation = cfg.get('activation', 'gelu') in ['fast-geglu', 'fast-swiglu', 'fast-reglu']

        target_modules = get_target_modules(lora_cfg)
        name_key_to_cfg = {}
        name_key_to_mcore_mixins = {}

        for module in target_modules:
            if module == PEFT_MODULE_MAP["qkv_module"]:
                if lora_cfg.get("variant", "nemo") == "canonical":
                    _adapter_name = AdapterName.LORA_UNFUSED_KQV_ADAPTER
                    _adapter_cfg_cls = LoraUnfusedKQVAdapterConfig
                    adapter_cfg = self._create_lora_config(
                        cfg,
                        lora_cfg,
                        cfg.hidden_size,
                        qkv_projection_size,
                        _adapter_cfg_cls,
                        num_query_groups=num_query_groups,
                        kv_channels=kv_channels,
                    )
                else:
                    _adapter_name = AdapterName.LORA_KQV_ADAPTER
                    _adapter_cfg_cls = LoraKQVAdapterConfig
                    adapter_cfg = self._create_lora_config(
                        cfg, lora_cfg, cfg.hidden_size, qkv_projection_size, _adapter_cfg_cls
                    )
                name_key_to_cfg[_adapter_name] = adapter_cfg
                name_key_to_mcore_mixins[_adapter_name] = [("self_attention", MCoreSelfAttentionMixin)]

            elif module == PEFT_MODULE_MAP["dense_module"]:
                adapter_cfg = self._create_lora_config(
                    cfg, lora_cfg, cfg.hidden_size, cfg.hidden_size, LoraDenseAttentionAdapterConfig
                )
                name_key_to_cfg[AdapterName.LORA_DENSE_ATTENTION_ADAPTER] = adapter_cfg
                name_key_to_mcore_mixins[AdapterName.LORA_DENSE_ATTENTION_ADAPTER] = [
                    ("self_attention", MCoreSelfAttentionMixin)
                ]

            elif module == PEFT_MODULE_MAP["hto4h_module"]:
                hto4h_projection_size = cfg.ffn_hidden_size * 2 if fast_glu_activation else cfg.ffn_hidden_size
                if lora_cfg.get("variant", "nemo") == "canonical":
                    _adapter_name = AdapterName.LORA_UNFUSED_Hto4H_ADAPTER
                    _adapter_cfg_cls = LoraUnfusedHto4HAdapterConfig
                else:
                    _adapter_name = AdapterName.LORA_Hto4H_ADAPTER
                    _adapter_cfg_cls = LoraHto4HAdapterConfig

                adapter_cfg = self._create_lora_config(
                    cfg, lora_cfg, cfg.hidden_size, hto4h_projection_size, _adapter_cfg_cls
                )
                name_key_to_cfg[_adapter_name] = adapter_cfg
                name_key_to_mcore_mixins[_adapter_name] = [("mlp", MCoreMLPMixin)]
            elif module == PEFT_MODULE_MAP["4htoh_module"]:
                adapter_cfg = self._create_lora_config(
                    cfg, lora_cfg, cfg.ffn_hidden_size, cfg.hidden_size, Lora4HtoHAdapterConfig
                )
                name_key_to_cfg[AdapterName.LORA_4HtoH_ADAPTER] = adapter_cfg
                name_key_to_mcore_mixins[AdapterName.LORA_4HtoH_ADAPTER] = [("mlp", MCoreMLPMixin)]
            else:
                logging.error(
                    f"Unrecognized target_module string: {module}.\n"
                    f"The possible options are: {list(PEFT_MODULE_MAP.values())}"
                )
                exit(1)

        self.name_key_to_mcore_mixins = name_key_to_mcore_mixins
        super().__init__(lora_cfg, name_key_to_cfg)

    def _create_lora_config(
        self, cfg, lora_cfg, in_features, out_features, adapter_cfg_cls, num_query_groups=None, kv_channels=None
    ):
        config_args = {
            "in_features": in_features,
            "out_features": out_features,
            "dim": lora_cfg.adapter_dim,
            "norm_position": None,
            "norm_type": None,
            "activation": "identity",
            "column_init_method": lora_cfg.get("column_init_method", "normal"),
            "row_init_method": lora_cfg.get("row_init_method", "zero"),
            "gather_output": False,
            "dropout": lora_cfg.adapter_dropout,
            "alpha": lora_cfg.get("alpha", lora_cfg.adapter_dim),
            "dropout_position": lora_cfg.get("dropout_position", "post"),
            "a2a_experimental": lora_cfg.get("a2a_experimental", False),
        }

        if adapter_cfg_cls == LoraUnfusedKQVAdapterConfig:
            assert num_query_groups is not None, "num_query_groups must be provided for canonical Lora"
            assert kv_channels is not None, "kv_channels must be provided for canonical Lora"
            config_args.update({"num_query_groups": num_query_groups, "kv_channels": kv_channels})
            config_args.pop("out_features")

        if lora_cfg.weight_tying:
            position_embedding_strategy = lora_cfg.get("position_embedding_strategy", None)
            if position_embedding_strategy is None:
                dim_position_embeddings = 0
            elif position_embedding_strategy == "add":
                dim_position_embeddings = cfg.hidden_size
            elif position_embedding_strategy == "biasadd":
                dim_position_embeddings = 3 * out_features
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

        adapter_cfg = adapter_cfg_cls(**config_args)

        return adapter_cfg


class IA3PEFTConfig(PEFTConfig):
    def __init__(self, cfg):
        mlp_infused_adapter_cfg = MLPInfusedAdapterConfig(
            in_features=cfg.ffn_hidden_size // cfg.tensor_model_parallel_size
        )

        kv_channels = self._calculate_kv_channels(cfg)
        num_query_groups = cfg.get("num_query_groups", cfg.num_attention_heads)
        kv_projection_size = kv_channels * num_query_groups
        infused_adapter_cfg = InfusedAdapterConfig(in_features=kv_projection_size // cfg.tensor_model_parallel_size)

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


class SDLoraPEFTConfig(PEFTConfig):
    def __init__(self, cfg):
        lora_cfg = cfg.peft.lora_tuning

        # Stable diffusion has different attn dimensions, we pass a dummy config and infer from each module when adding adapter
        config_args = {
            "in_features": None,
            "out_features": None,
            "dim": lora_cfg.adapter_dim,
            "norm_position": None,
            "norm_type": None,
            "activation": "identity",
            "column_init_method": lora_cfg.get("column_init_method", "normal"),
            "row_init_method": lora_cfg.get("row_init_method", "zero"),
            "gather_output": False,
            "dropout": lora_cfg.adapter_dropout,
            "network_alpha": lora_cfg.network_alpha,
        }

        name_key_to_cfg = {AdapterName.PARALLEL_LINEAR_ADAPTER: ParallelLinearAdapterConfig(**config_args)}
        self.name_key_to_mcore_mixins = None
        super().__init__(lora_cfg, name_key_to_cfg)


PEFT_CONFIG_MAP = {
    "adapter": CanonicalAdaptersPEFTConfig,
    "ia3": IA3PEFTConfig,
    "ptuning": PtuningPEFTConfig,
    "lora": LoraPEFTConfig,
    "selective": SelectivePEFTConfig,
    'none': None,
    None: None,
    "sdlora": SDLoraPEFTConfig,
}
