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

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Optional

import torch

from nemo.utils import logging

try:
    from megatron.core import parallel_state
    from megatron.core.models.mamba import MambaModel as MCoreMambaModel
    from megatron.core.models.mamba.mamba_layer_specs import mamba_stack_spec

    HAVE_MEGATRON_CORE_OR_TE = True

except (ImportError, ModuleNotFoundError):
    logging.warning("The package `megatron.core` was not imported in this environment which is needed for SSMs.")
    HAVE_MEGATRON_CORE_OR_TE = False

from megatron.core.transformer.transformer_config import TransformerConfig
from nemo.collections.llm.gpt.model.base import GPTModel, gpt_data_step
from nemo.lightning import get_vocab_size, io, teardown


def ssm_forward_step(model, batch) -> torch.Tensor:

    forward_args = {
        "input_ids": batch["tokens"],
        "position_ids": batch["position_ids"],
        "labels": batch["labels"],
    }
    forward_args["attention_mask"] = None
    return model(**forward_args)


@dataclass
class SSMConfig(TransformerConfig, io.IOMixin):
    # From megatron.core.models.mamba.mamba_model.MambaModel
    fp16_lm_cross_entropy: bool = False
    parallel_output: bool = True
    share_embeddings_and_output_weights: bool = False
    num_layers: int = 2
    mamba_ssm_ngroups: int = 8
    num_attention_heads: int = 1
    hybrid_attention_ratio: float = 0.0
    hybrid_mlp_ratio: float = 0.0
    hybrid_override_pattern: str = None
    post_process: bool = True
    pre_process: bool = True
    seq_length: int = 2048
    # Mamba with no attention has no need for position embeddings, so none is default
    position_embedding_type: Literal['learned_absolute', 'rope', 'none'] = 'none'
    rotary_percent: float = 1.0
    rotary_base: int = 10000
    seq_len_interpolation_factor: Optional[float] = None
    apply_rope_fusion: bool = True
    make_vocab_size_divisible_by: int = 128
    gated_linear_unit: bool = False
    fp32_residual_connections: bool = True
    normalization: str = 'RMSNorm'
    add_bias_linear: bool = False
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0
    layernorm_epsilon: float = 1e-5
    # TODO: Move this to better places?
    get_attention_mask_from_fusion: bool = False

    forward_step_fn: Callable = ssm_forward_step
    data_step_fn: Callable = gpt_data_step

    def configure_model(self, tokenizer) -> "MCoreMambaModel":

        return MCoreMambaModel(
            self,
            mamba_stack_spec=mamba_stack_spec,
            vocab_size=get_vocab_size(self, tokenizer.vocab_size, self.make_vocab_size_divisible_by),
            max_sequence_length=self.seq_length,
            mamba_ssm_ngroups=self.mamba_ssm_ngroups,
            hybrid_attention_ratio=self.hybrid_attention_ratio,
            hybrid_mlp_ratio=self.hybrid_mlp_ratio,
            hybrid_override_pattern=self.hybrid_override_pattern,
            position_embedding_type=self.position_embedding_type,
            rotary_percent=self.rotary_percent,
            rotary_base=self.rotary_base,
            seq_len_interpolation_factor=self.seq_len_interpolation_factor,
            pre_process=parallel_state.is_pipeline_first_stage(),
            post_process=parallel_state.is_pipeline_last_stage(),
        )


@io.model_importer(GPTModel, "pytorch")
class PyTorchSSMImporter(io.ModelConnector["GPTModel", GPTModel]):

    def __new__(cls, path: str, model_config=None):
        instance = super().__new__(cls, path)
        instance.model_config = model_config
        return instance

    def init(self) -> GPTModel:

        return GPTModel(self.config, tokenizer=self.tokenizer)

    def apply(self, output_path: Path) -> Path:

        source = torch.load(str(self), map_location='cpu')
        if 'model' in source:
            source = source['model']

        class ModelState:
            def __init__(self, state_dict):
                self._state_dict = state_dict

            def state_dict(self):
                return self._state_dict

        source = ModelState(source)
        target = self.init()
        trainer = self.nemo_setup(target)
        self.convert_state(source, target)
        self.nemo_save(output_path, trainer)

        logging.info(f"Converted SSM model to Nemo, model saved to {output_path}")

        teardown(trainer, target)
        del trainer, target

        return output_path

    def convert_state(self, source, target):

        if self.model_config.mapping_type == "base":
            mapping = {
                'backbone.embedding.weight': 'embedding.word_embeddings.weight',
                'backbone.layers.*.mixer.A_log': 'decoder.layers.*.mixer.A_log',
                'backbone.layers.*.mixer.D': 'decoder.layers.*.mixer.D',
                'backbone.layers.*.mixer.conv1d.weight': 'decoder.layers.*.mixer.conv1d.weight',
                'backbone.layers.*.mixer.conv1d.bias': 'decoder.layers.*.mixer.conv1d.bias',
                'backbone.layers.*.mixer.in_proj.weight': 'decoder.layers.*.mixer.in_proj.weight',
                'backbone.layers.*.mixer.dt_bias': 'decoder.layers.*.mixer.dt_bias',
                'backbone.layers.*.mixer.out_proj.weight': 'decoder.layers.*.mixer.out_proj.weight',
                'backbone.layers.*.mixer.norm.weight': 'decoder.layers.*.mixer.norm.weight',
                'backbone.layers.*.norm.weight': 'decoder.layers.*.mixer.in_proj.layer_norm_weight',
                'backbone.norm_f.weight': 'decoder.final_norm.weight',
                'lm_head.weight': 'output_layer.weight',
            }
        elif "nvidia" in self.model_config.mapping_type:
            mapping = {
                'embedding.word_embeddings.weight': 'embedding.word_embeddings.weight',
                'decoder.layers.*.mixer.A_log': 'decoder.layers.*.mixer.A_log',
                'decoder.layers.*.mixer.D': 'decoder.layers.*.mixer.D',
                'decoder.layers.*.mixer.conv1d.weight': 'decoder.layers.*.mixer.conv1d.weight',
                'decoder.layers.*.mixer.conv1d.bias': 'decoder.layers.*.mixer.conv1d.bias',
                'decoder.layers.*.mixer.in_proj.weight': 'decoder.layers.*.mixer.in_proj.weight',
                'decoder.layers.*.mixer.dt_bias': 'decoder.layers.*.mixer.dt_bias',
                'decoder.layers.*.mixer.out_proj.weight': 'decoder.layers.*.mixer.out_proj.weight',
                'decoder.layers.*.mixer.norm.weight': 'decoder.layers.*.mixer.norm.weight',
                'decoder.layers.*.norm.weight': 'decoder.layers.*.mixer.in_proj.layer_norm_weight',
                'decoder.final_norm.weight': 'decoder.final_norm.weight',
                'output_layer.weight': 'output_layer.weight',
            }
            if "hybrid" in self.model_config.mapping_type:
                mapping.update(
                    {
                        'decoder.layers.*.mlp.linear_fc1.layer_norm_weight': 'decoder.layers.*.mlp.linear_fc1.layer_norm_weight',
                        'decoder.layers.*.mlp.linear_fc1.weight': 'decoder.layers.*.mlp.linear_fc1.weight',
                        'decoder.layers.*.mlp.linear_fc2.weight': 'decoder.layers.*.mlp.linear_fc2.weight',
                        'decoder.layers.*.self_attention.linear_proj.weight': 'decoder.layers.*.self_attention.linear_proj.weight',
                        'decoder.layers.*.self_attention.linear_qkv.layer_norm_weight': 'decoder.layers.*.self_attention.linear_qkv.layer_norm_weight',
                        'decoder.layers.*.self_attention.linear_qkv.weight': 'decoder.layers.*.self_attention.linear_qkv.weight',
                    }
                )
        else:
            raise AttributeError(f"mapping type [{self.mapping_type}] not found.")
        return io.apply_transforms(source, target, mapping=mapping)

    @property
    def tokenizer(self):
        from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer

        tokenizer = get_nmt_tokenizer(
            library=self.model_config.tokenizer_library,
            model_name=self.model_config.tokenizer_name,
            tokenizer_model=self.model_config.tokenizer_model_path,
            use_fast=True,
        )

        return tokenizer

    @property
    def config(self) -> SSMConfig:
        return self.model_config


@dataclass
class BaseMambaConfig130M(SSMConfig):
    hybrid_override_pattern: str = "M" * 24
    num_layers: int = 24
    seq_length: int = 2048
    hidden_size: int = 768
    mamba_ssm_ngroups: int = 1
    ffn_hidden_size: int = 768
    make_vocab_size_divisible_by: int = 16
    tokenizer_library: str = 'huggingface'
    tokenizer_name: str = "EleutherAI/gpt-neox-20b"
    mapping_type: str = "base"


@dataclass
class BaseMambaConfig370M(SSMConfig):
    hybrid_override_pattern: str = "M" * 48
    num_layers: int = 48
    seq_length: int = 2048
    hidden_size: int = 1024
    mamba_ssm_ngroups: int = 1
    ffn_hidden_size: int = 1024
    make_vocab_size_divisible_by: int = 16
    tokenizer_library: str = 'huggingface'
    tokenizer_name: str = "EleutherAI/gpt-neox-20b"
    mapping_type: str = "base"


@dataclass
class BaseMambaConfig780M(SSMConfig):
    hybrid_override_pattern: str = "M" * 48
    num_layers: int = 48
    seq_length: int = 2048
    hidden_size: int = 1536
    mamba_ssm_ngroups: int = 1
    ffn_hidden_size: int = 1536
    make_vocab_size_divisible_by: int = 16
    tokenizer_library: str = 'huggingface'
    tokenizer_name: str = "EleutherAI/gpt-neox-20b"
    mapping_type: str = "base"


@dataclass
class BaseMambaConfig1_3B(SSMConfig):
    hybrid_override_pattern: str = "M" * 48
    num_layers: int = 48
    seq_length: int = 2048
    hidden_size: int = 2048
    mamba_ssm_ngroups: int = 1
    ffn_hidden_size: int = 2048
    make_vocab_size_divisible_by: int = 16
    tokenizer_library: str = 'huggingface'
    tokenizer_name: str = "EleutherAI/gpt-neox-20b"
    mapping_type: str = "base"


@dataclass
class BaseMambaConfig2_7B(SSMConfig):
    hybrid_override_pattern: str = "M" * 64
    num_layers: int = 64
    seq_length: int = 2048
    hidden_size: int = 2560
    mamba_ssm_ngroups: int = 1
    ffn_hidden_size: int = 2560
    make_vocab_size_divisible_by: int = 16
    tokenizer_library: str = 'huggingface'
    tokenizer_name: str = "EleutherAI/gpt-neox-20b"
    mapping_type: str = "base"


@dataclass
class NVIDIAMambaConfig8B(SSMConfig):
    hybrid_override_pattern: str = "M" * 56
    num_layers: int = 56
    seq_length: int = 4096
    hidden_size: int = 4096
    mamba_ssm_ngroups: int = 8
    ffn_hidden_size: int = 4096
    make_vocab_size_divisible_by: int = 128
    tokenizer_library: str = 'megatron'
    tokenizer_name: str = "GPTSentencePieceTokenizer"
    mapping_type: str = "nvidia-pure"


@dataclass
class NVIDIAMambaHybridConfig8B(SSMConfig):
    hybrid_override_pattern: str = "M-M-M--M-M*-M-M-M-M--M*-M-M-M-M-M*--M-M-M-M-M*-M--M-M-M-"
    num_layers: int = 56
    seq_length: int = 4096
    hidden_size: int = 4096
    mamba_ssm_ngroups: int = 8
    ffn_hidden_size: int = 16384
    num_attention_heads: int = 32
    num_query_groups: int = 8
    make_vocab_size_divisible_by: int = 128
    tokenizer_library: str = 'megatron'
    tokenizer_name: str = "GPTSentencePieceTokenizer"
    mapping_type: str = "nvidia-hybrid"


__all__ = [
    "SSMConfig",
    "BaseMambaConfig130M",
    "BaseMambaConfig370M",
    "BaseMambaConfig780M",
    "BaseMambaConfig1_3B",
    "BaseMambaConfig2_7B",
    "NVIDIAMambaConfig8B",
    "NVIDIAMambaHybridConfig8B",
]
