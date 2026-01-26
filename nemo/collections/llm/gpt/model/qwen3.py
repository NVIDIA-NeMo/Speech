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

from dataclasses import dataclass
from functools import cached_property, partial
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Callable, Optional

import torch
from torch import nn

from nemo.collections.llm.gpt.model.base import GPTModel, torch_dtype_from_mcore_config
from nemo.collections.llm.gpt.model.qwen2 import Qwen2Config
from nemo.collections.llm.utils import Config, is_safe_repo
from nemo.lightning import OptimizerModule, io, teardown
from nemo.lightning.io.state import TransformFns
from nemo.lightning.pytorch.utils import dtype_from_hf

if TYPE_CHECKING:
    from transformers import AutoModelForCausalLM
    from transformers import Qwen3Config as HFQwen3Config

    from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
    from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec


@dataclass
class Qwen3Config(Qwen2Config):
    """
    Base config for Qwen 3 Models
    """

    add_qkv_bias: bool = False
    qk_layernorm: bool = True
    kv_channels: Optional[int] = 128
    num_query_groups: int = 8
    max_position_embeddings: int = 40960
    vocab_size: int = 151936


@dataclass
class Qwen3MoEConfig(Qwen3Config):
    """
    Base config for Qwen 3 MoE Models
    """

    num_moe_experts: int = 128
    moe_router_load_balancing_type: str = "aux_loss"
    moe_aux_loss_coeff: float = 1e-3
    moe_router_topk: int = 8
    moe_router_pre_softmax: bool = False
    moe_grouped_gemm: bool = True
    moe_token_dispatcher_type: str = "alltoall"
    moe_permute_fusion: bool = True


@dataclass
class Qwen3Config600M(Qwen3Config):
    """
    Config for Qwen 3 0.6B: https://huggingface.co/Qwen/Qwen3-0.6B
    """

    num_layers: int = 28
    hidden_size: int = 1024
    num_attention_heads: int = 16
    ffn_hidden_size: int = 3072
    share_embeddings_and_output_weights: bool = True


@dataclass
class Qwen3Config1P7B(Qwen3Config):
    """
    Config for Qwen 3 1.7B: https://huggingface.co/Qwen/Qwen3-1.7B
    """

    num_layers: int = 28
    hidden_size: int = 2048
    num_attention_heads: int = 16
    ffn_hidden_size: int = 6144
    share_embeddings_and_output_weights: bool = True


@dataclass
class Qwen3Config4B(Qwen3Config):
    """
    Config for Qwen 3 4B: https://huggingface.co/Qwen/Qwen3-4B
    """

    num_layers: int = 36
    hidden_size: int = 2560
    num_attention_heads: int = 32
    ffn_hidden_size: int = 9728
    share_embeddings_and_output_weights: bool = True


@dataclass
class Qwen3Config8B(Qwen3Config):
    """
    Config for Qwen 3 8B: https://huggingface.co/Qwen/Qwen3-8B
    """

    num_layers: int = 36
    hidden_size: int = 4096
    num_attention_heads: int = 32
    ffn_hidden_size: int = 12288


@dataclass
class Qwen3Config14B(Qwen3Config):
    """
    Config for Qwen 3 14B: https://huggingface.co/Qwen/Qwen3-14B
    """

    num_layers: int = 40
    hidden_size: int = 5120
    num_attention_heads: int = 40
    ffn_hidden_size: int = 17408


@dataclass
class Qwen3Config32B(Qwen3Config):
    """
    Config for Qwen 3 32B: https://huggingface.co/Qwen/Qwen3-32B
    """

    num_layers: int = 64
    hidden_size: int = 5120
    num_attention_heads: int = 64
    ffn_hidden_size: int = 25600


@dataclass
class Qwen3Config30B_A3B(Qwen3MoEConfig):
    """
    Config for Qwen 3 30B-A3B: https://huggingface.co/Qwen/Qwen3-30B-A3B
    """

    num_layers: int = 48
    hidden_size: int = 2048
    num_attention_heads: int = 32
    num_query_groups: int = 4
    ffn_hidden_size: int = 6144
    moe_ffn_hidden_size: int = 768


@dataclass
class Qwen3Config235B_A22B(Qwen3MoEConfig):
    """
    Config for Qwen 3 235B-A22B: https://huggingface.co/Qwen/Qwen3-235B-A22B
    """

    num_layers: int = 94
    hidden_size: int = 4096
    num_attention_heads: int = 64
    num_query_groups: int = 4
    ffn_hidden_size: int = 12288
    moe_ffn_hidden_size: int = 1536


class Qwen3Model(GPTModel):
    """
    Base model for Qwen 3
    """

    def __init__(
        self,
        config: Annotated[Optional[Qwen3Config], Config[Qwen3Config]] = None,
        optim: Optional[OptimizerModule] = None,
        tokenizer: Optional["TokenizerSpec"] = None,
        model_transform: Optional[Callable[[nn.Module], nn.Module]] = None,
    ):
        super().__init__(config or Qwen3Config(), optim=optim, tokenizer=tokenizer, model_transform=model_transform)


@io.model_importer(Qwen3Model, "hf")
class HFQwen3Importer(io.ModelConnector["AutoModelForCausalLM", Qwen3Model]):
    # pylint: disable=C0115,C0116
    def init(self) -> Qwen3Model:
        return Qwen3Model(self.config, tokenizer=self.tokenizer)

    def apply(self, output_path: Path, trust_remote_code: bool | None = None) -> Path:
        from transformers import AutoModelForCausalLM

        # logging.setLevel(logging.DEBUG)
        self.trust_remote_code = trust_remote_code
        source = AutoModelForCausalLM.from_pretrained(
            str(self),
            torch_dtype='auto',
            trust_remote_code=is_safe_repo(
                trust_remote_code=self.trust_remote_code,
                hf_path=str(self),
            ),
        )
        target = self.init()
        trainer = self.nemo_setup(target)
        self.convert_state(source, target)
        self.nemo_save(output_path, trainer)

        print(f"Converted Qwen 3 model to Nemo, model saved to {output_path}")

        teardown(trainer, target)
        del trainer, target

        return output_path

    def convert_state(self, source, target):
        mapping = {
            "model.embed_tokens.weight": "embedding.word_embeddings.weight",
            "**.self_attn.o_proj.weight": "**.self_attention.linear_proj.weight",
            "**.self_attn.q_norm.weight": "**.self_attention.q_layernorm.weight",
            "**.self_attn.k_norm.weight": "**.self_attention.k_layernorm.weight",
            "**.input_layernorm.weight": "**.self_attention.linear_qkv.layer_norm_weight",
            "model.norm.weight": "decoder.final_layernorm.weight",
            "lm_head.weight": "output_layer.weight",
        }
        is_moe = self.config.num_moe_experts is not None
        if is_moe:
            mapping.update(
                {
                    "**.mlp.experts.*.down_proj.weight": "**.mlp.experts.linear_fc2.weight*",
                    "**.mlp.gate.weight": "**.mlp.router.weight",
                    "**.post_attention_layernorm.weight": "**.pre_mlp_layernorm.weight",
                }
            )
        else:
            mapping.update(
                {
                    "**.mlp.down_proj.weight": "**.mlp.linear_fc2.weight",
                    "**.post_attention_layernorm.weight": "**.mlp.linear_fc1.layer_norm_weight",
                }
            )

        if getattr(source.config, "tie_word_embeddings", False):
            del mapping["lm_head.weight"]

        transforms = [
            io.state_transform(
                source_key=(
                    "**.self_attn.q_proj.weight",
                    "**.self_attn.k_proj.weight",
                    "**.self_attn.v_proj.weight",
                ),
                target_key="**.self_attention.linear_qkv.weight",
                fn=TransformFns.merge_qkv,
            ),
            (
                io.state_transform(
                    source_key=("**.mlp.gate_proj.weight", "**.mlp.up_proj.weight"),
                    target_key="**.mlp.linear_fc1.weight",
                    fn=TransformFns.merge_fc1,
                )
                if not is_moe
                else io.state_transform(
                    source_key=("**.mlp.experts.*.gate_proj.weight", "**.mlp.experts.*.up_proj.weight"),
                    target_key="**.mlp.experts.linear_fc1.weight*",
                    fn=TransformFns.merge_fc1,
                )
            ),
        ]
        return io.apply_transforms(source, target, mapping=mapping, transforms=transforms)

    @cached_property
    def tokenizer(self) -> "AutoTokenizer":
        from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer

        return AutoTokenizer(
            self.save_hf_tokenizer_assets(str(self)),
            trust_remote_code=is_safe_repo(
                trust_remote_code=self.trust_remote_code,
                hf_path=str(self),
            ),
        )

    @cached_property
    def config(self) -> Qwen3Config:
        from transformers import AutoConfig as HFAutoConfig
        from transformers import GenerationConfig

        source = HFAutoConfig.from_pretrained(
            str(self),
            trust_remote_code=is_safe_repo(
                trust_remote_code=self.trust_remote_code,
                hf_path=str(self),
            ),
        )
        generation_config = GenerationConfig.from_pretrained(str(self))

        is_moe = getattr(source, "num_experts", None) is not None
        if is_moe:
            qwen3_config_cls = partial(Qwen3MoEConfig, moe_ffn_hidden_size=source.moe_intermediate_size)
        else:
            qwen3_config_cls = Qwen3Config
        output = qwen3_config_cls(
            num_layers=source.num_hidden_layers,
            hidden_size=source.hidden_size,
            ffn_hidden_size=source.intermediate_size,
            num_attention_heads=source.num_attention_heads,
            num_query_groups=source.num_key_value_heads,
            init_method_std=source.initializer_range,
            layernorm_epsilon=source.rms_norm_eps,
            vocab_size=source.vocab_size,
            make_vocab_size_divisible_by=1187,
            rotary_base=source.rope_theta,
            share_embeddings_and_output_weights=getattr(source, "tie_word_embeddings", False),
            fp16=(dtype_from_hf(source) == torch.float16),
            bf16=(dtype_from_hf(source) == torch.bfloat16),
            params_dtype=dtype_from_hf(source),
            generation_config=generation_config,
        )

        return output


@io.model_exporter(Qwen3Model, "hf")
class HFQwen3Exporter(io.ModelConnector[Qwen3Model, "AutoModelForCausalLM"]):
    # pylint: disable=C0115,C0116
    def init(self, dtype=torch.bfloat16) -> "AutoModelForCausalLM":
        from transformers import AutoModelForCausalLM
        from transformers.modeling_utils import no_init_weights

        with no_init_weights():
            return AutoModelForCausalLM.from_config(
                self.config,
                trust_remote_code=is_safe_repo(
                    trust_remote_code=self.trust_remote_code,
                    hf_path=str(self),
                ),
                torch_dtype=dtype,
            )

    def apply(self, output_path: Path, trust_remote_code: bool | None = None) -> Path:
        self.trust_remote_code = trust_remote_code
        source, _ = self.nemo_load(str(self))
        target = self.init(torch_dtype_from_mcore_config(source.config))
        target = self.convert_state(source, target)

        target = target.cpu()
        target.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        return output_path

    def convert_state(self, source, target):
        mapping = {
            "**.self_attention.linear_proj.weight": "**.self_attn.o_proj.weight",
            "**.self_attention.linear_qkv.layer_norm_weight": "**.input_layernorm.weight",
            "**.self_attention.q_layernorm.weight": "**.self_attn.q_norm.weight",
            "**.self_attention.k_layernorm.weight": "**.self_attn.k_norm.weight",
            "decoder.final_layernorm.weight": "model.norm.weight",
        }
        is_moe = getattr(self.config, "num_experts", 0) > 0
        if is_moe:
            mapping.update(
                {
                    "**.mlp.experts.linear_fc2.weight*": "**.mlp.experts.*.down_proj.weight",
                    "**.mlp.router.weight": "**.mlp.gate.weight",
                    "**.pre_mlp_layernorm.weight": "**.post_attention_layernorm.weight",
                }
            )
        else:
            mapping.update(
                {
                    "**.mlp.linear_fc2.weight": "**.mlp.down_proj.weight",
                    "**.mlp.linear_fc1.layer_norm_weight": "**.post_attention_layernorm.weight",
                }
            )
        transforms = [
            io.state_transform(
                source_key="**.self_attention.linear_qkv.weight",
                target_key=(
                    "**.self_attn.q_proj.weight",
                    "**.self_attn.k_proj.weight",
                    "**.self_attn.v_proj.weight",
                ),
                fn=TransformFns.split_qkv,
            ),
            (
                io.state_transform(
                    source_key="**.mlp.linear_fc1.weight",
                    target_key=("**.mlp.gate_proj.weight", "**.mlp.up_proj.weight"),
                    fn=TransformFns.split_fc1,
                )
                if not is_moe
                else io.state_transform(
                    source_key="**.mlp.experts.linear_fc1.weight*",
                    target_key=("**.mlp.experts.*.gate_proj.weight", "**.mlp.experts.*.up_proj.weight"),
                    fn=TransformFns.split_fc1,
                )
            ),
            io.state_transform(
                source_key="embedding.word_embeddings.weight",
                target_key="model.embed_tokens.weight",
                fn=TransformFns.prune_padding,
            ),
        ]
        if not self.config.tie_word_embeddings:
            transforms.append(
                io.state_transform(
                    source_key="output_layer.weight",
                    target_key="lm_head.weight",
                    fn=TransformFns.prune_padding,
                )
            )

        return io.apply_transforms(
            source,
            target,
            mapping=mapping,
            transforms=transforms,
        )

    @property
    def tokenizer(self):
        return io.load_context(str(self)).model.tokenizer.tokenizer

    @property
    def config(self) -> "HFQwen3Config":
        from transformers import Qwen3Config as HFQwen3Config
        from transformers import Qwen3MoeConfig as HFQwen3MoeConfig

        source: Qwen3Config = io.load_context(str(self), subpath="model.config")
        is_moe = source.num_moe_experts is not None
        hf_config_cls = (
            partial(
                HFQwen3MoeConfig,
                moe_intermediate_size=source.moe_ffn_hidden_size,
                num_experts=source.num_moe_experts,
                num_experts_per_tok=source.moe_router_topk,
                router_aux_loss_coef=source.moe_aux_loss_coeff,
                norm_topk_prob=True,
            )
            if is_moe
            else HFQwen3Config
        )

        return hf_config_cls(
            architectures=["Qwen3ForCausalLM"],
            num_hidden_layers=source.num_layers,
            hidden_size=source.hidden_size,
            intermediate_size=source.ffn_hidden_size,
            num_attention_heads=source.num_attention_heads,
            head_dim=source.kv_channels,
            max_position_embeddings=source.max_position_embeddings,
            initializer_range=source.init_method_std,
            rms_norm_eps=source.layernorm_epsilon,
            num_key_value_heads=source.num_query_groups,
            rope_theta=source.rotary_base,
            vocab_size=getattr(source, 'vocab_size', self.tokenizer.vocab_size),
            sliding_window=None,
            tie_word_embeddings=source.share_embeddings_and_output_weights,
            max_window_layers=source.num_layers,
            bos_token_id=151643,
            eos_token_id=151645,
        )


__all__ = [
    "Qwen3Config",
    "Qwen3Config600M",
    "Qwen3Config1P7B",
    "Qwen3Config4B",
    "Qwen3Config8B",
    "Qwen3Config14B",
    "Qwen3Config32B",
    "Qwen3Config30B_A3B",
    "Qwen3Config235B_A22B",
    "Qwen3Model",
]
