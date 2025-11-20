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
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Callable, Optional

import torch
import torch.nn.functional as F
from torch import nn

from nemo.collections.llm.gpt.model.base import GPTConfig, GPTModel, torch_dtype_from_mcore_config
from nemo.collections.llm.utils import Config, is_safe_repo
from nemo.lightning import OptimizerModule, io, teardown
from nemo.lightning.io.state import TransformFns
from nemo.lightning.pytorch.utils import dtype_from_hf

if TYPE_CHECKING:
    from transformers import AutoModelForCausalLM
    from transformers import Qwen2Config as HFQwen2Config

    from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
    from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec


@dataclass
class Qwen2Config(GPTConfig):
    """
    Base config for Qwen 2 Models
    """

    normalization: str = "RMSNorm"
    activation_func: Callable = F.silu
    gated_linear_unit: bool = True
    add_bias_linear: bool = False
    add_qkv_bias: bool = True
    seq_length: int = 4096
    init_method_std: int = 0.02
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0
    vocab_size: int = 151936
    share_embeddings_and_output_weights: Optional[bool] = False
    layernorm_epsilon: float = 1e-6
    rotary_base: float = 1000000.0
    position_embedding_type: str = "rope"


@dataclass
class Qwen2Config500M(Qwen2Config):
    """
    Config for Qwen 2 0.5B: https://huggingface.co/Qwen/Qwen2-0.5B
    """

    num_layers: int = 24
    hidden_size: int = 896
    num_attention_heads: int = 14
    num_query_groups: int = 2
    ffn_hidden_size: int = 4864


@dataclass
class Qwen25Config500M(Qwen2Config500M):
    """
    Config for Qwen 2.5 0.5B: https://huggingface.co/Qwen/Qwen2.5-0.5B
    """

    seq_length: int = 32768


@dataclass
class Qwen2Config1P5B(Qwen2Config):
    """
    Config for Qwen 2 1.5B: https://huggingface.co/Qwen/Qwen2-1.5B
    """

    num_layers: int = 28
    hidden_size: int = 1536
    num_attention_heads: int = 12
    num_query_groups: int = 2
    ffn_hidden_size: int = 8960


@dataclass
class Qwen25Config3B(Qwen2Config):
    """
    Config for Qwen 2.5 3B: https://huggingface.co/Qwen/Qwen2.5-3B
    """

    num_layers: int = 36
    hidden_size: int = 2048
    num_attention_heads: int = 16
    num_query_groups: int = 2
    ffn_hidden_size: int = 11008
    vocab_size: int = 151936
    share_embeddings_and_output_weights: bool = True


@dataclass
class Qwen25Config1P5B(Qwen2Config1P5B):
    """
    Config for Qwen 2.5 1.5B: https://huggingface.co/Qwen/Qwen2.5-1.5B
    """

    seq_length: int = 131072


@dataclass
class Qwen2Config7B(Qwen2Config):
    """
    Config for Qwen 2 7B: https://huggingface.co/Qwen/Qwen2-7B
    """

    num_layers: int = 28
    hidden_size: int = 3584
    num_attention_heads: int = 28
    num_query_groups: int = 4
    ffn_hidden_size: int = 18944
    vocab_size: int = 152064


@dataclass
class Qwen25Config7B(Qwen2Config7B):
    """
    Config for Qwen 2.5 7B: https://huggingface.co/Qwen/Qwen2.5-7B
    """

    seq_length: int = 131072


@dataclass
class Qwen25Config14B(Qwen2Config):
    """
    Config for Qwen 2.5 14B: https://huggingface.co/Qwen/Qwen2.5-14B
    """

    num_layers: int = 48
    hidden_size: int = 5120
    num_attention_heads: int = 40
    num_query_groups: int = 8
    ffn_hidden_size: int = 13824
    vocab_size: int = 152064
    layernorm_epsilon: float = 1e-5
    seq_length: int = 131072


@dataclass
class Qwen25Config32B(Qwen2Config):
    """
    Config for Qwen 2.5 32B: https://huggingface.co/Qwen/Qwen2.5-32B
    """

    num_layers: int = 64
    hidden_size: int = 5120
    num_attention_heads: int = 40
    num_query_groups: int = 8
    ffn_hidden_size: int = 27648
    vocab_size: int = 152064
    layernorm_epsilon: float = 1e-5
    seq_length: int = 131072


@dataclass
class Qwen2Config72B(Qwen2Config):
    """
    Config for Qwen 2 72B: https://huggingface.co/Qwen/Qwen2-72B
    """

    num_layers: int = 80
    hidden_size: int = 8192
    num_attention_heads: int = 64
    num_query_groups: int = 8
    ffn_hidden_size: int = 29568
    vocab_size: int = 152064
    layernorm_epsilon: float = 1e-5


@dataclass
class Qwen25Config72B(Qwen2Config72B):
    """
    Config for Qwen 2.5 72B: https://huggingface.co/Qwen/Qwen2.5-72B
    """

    seq_length: int = 131072


class Qwen2Model(GPTModel):
    """
    Base model for Qwen 2
    """

    def __init__(
        self,
        config: Annotated[Optional[Qwen2Config], Config[Qwen2Config]] = None,
        optim: Optional[OptimizerModule] = None,
        tokenizer: Optional["TokenizerSpec"] = None,
        model_transform: Optional[Callable[[nn.Module], nn.Module]] = None,
    ):
        super().__init__(config or Qwen2Config(), optim=optim, tokenizer=tokenizer, model_transform=model_transform)


@io.model_importer(Qwen2Model, "hf")
class HFQwen2Importer(io.ModelConnector["AutoModelForCausalLM", Qwen2Model]):
    # pylint: disable=C0115,C0116
    def init(self) -> Qwen2Model:
        return Qwen2Model(self.config, tokenizer=self.tokenizer)

    def apply(self, output_path: Path, trust_remote_code: bool | None = None) -> Path:
        from transformers import AutoModelForCausalLM

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

        print(f"Converted Qwen model to Nemo, model saved to {output_path}")

        teardown(trainer, target)
        del trainer, target

        return output_path

    def convert_state(self, source, target):
        mapping = {
            "model.embed_tokens.weight": "embedding.word_embeddings.weight",
            "model.layers.*.self_attn.o_proj.weight": "decoder.layers.*.self_attention.linear_proj.weight",
            "model.layers.*.mlp.down_proj.weight": "decoder.layers.*.mlp.linear_fc2.weight",
            "model.layers.*.input_layernorm.weight": "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight",
            "model.layers.*.post_attention_layernorm.weight": "decoder.layers.*.mlp.linear_fc1.layer_norm_weight",
            "model.norm.weight": "decoder.final_layernorm.weight",
            "lm_head.weight": "output_layer.weight",
        }

        transforms = [
            io.state_transform(
                source_key=(
                    "model.layers.*.self_attn.q_proj.weight",
                    "model.layers.*.self_attn.k_proj.weight",
                    "model.layers.*.self_attn.v_proj.weight",
                ),
                target_key="decoder.layers.*.self_attention.linear_qkv.weight",
                fn=TransformFns.merge_qkv,
            ),
            io.state_transform(
                source_key=(
                    "model.layers.*.self_attn.q_proj.bias",
                    "model.layers.*.self_attn.k_proj.bias",
                    "model.layers.*.self_attn.v_proj.bias",
                ),
                target_key="decoder.layers.*.self_attention.linear_qkv.bias",
                fn=TransformFns.merge_qkv_bias,
            ),
            io.state_transform(
                source_key=("model.layers.*.mlp.gate_proj.weight", "model.layers.*.mlp.up_proj.weight"),
                target_key="decoder.layers.*.mlp.linear_fc1.weight",
                fn=TransformFns.merge_fc1,
            ),
        ]
        return io.apply_transforms(source, target, mapping=mapping, transforms=transforms)

    @property
    def tokenizer(self) -> "AutoTokenizer":
        from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer

        return AutoTokenizer(
            self.save_hf_tokenizer_assets(str(self)),
            trust_remote_code=is_safe_repo(
                trust_remote_code=self.trust_remote_code,
                hf_path=str(self),
            ),
        )

    @property
    def config(self) -> Qwen2Config:
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

        output = Qwen2Config(
            num_layers=source.num_hidden_layers,
            hidden_size=source.hidden_size,
            ffn_hidden_size=source.intermediate_size,
            num_attention_heads=source.num_attention_heads,
            num_query_groups=source.num_key_value_heads,
            init_method_std=source.initializer_range,
            layernorm_epsilon=source.rms_norm_eps,
            gated_linear_unit=True,
            make_vocab_size_divisible_by=128,
            rotary_base=source.rope_theta,
            share_embeddings_and_output_weights=False,
            vocab_size=source.vocab_size,
            seq_length=source.max_position_embeddings,
            fp16=(dtype_from_hf(source) == torch.float16),
            bf16=(dtype_from_hf(source) == torch.bfloat16),
            params_dtype=dtype_from_hf(source),
            generation_config=generation_config,
        )

        return output


@io.model_exporter(Qwen2Model, "hf")
class HFQwen2Exporter(io.ModelConnector[Qwen2Model, "AutoModelForCausalLM"]):
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
            "decoder.layers.*.self_attention.linear_proj.weight": "model.layers.*.self_attn.o_proj.weight",
            "decoder.layers.*.mlp.linear_fc2.weight": "model.layers.*.mlp.down_proj.weight",
            "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "model.layers.*.input_layernorm.weight",
            "decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "model.layers.*.post_attention_layernorm.weight",
            "decoder.final_layernorm.weight": "model.norm.weight",
        }

        transforms = [
            io.state_transform(
                source_key="decoder.layers.*.self_attention.linear_qkv.weight",
                target_key=(
                    "model.layers.*.self_attn.q_proj.weight",
                    "model.layers.*.self_attn.k_proj.weight",
                    "model.layers.*.self_attn.v_proj.weight",
                ),
                fn=TransformFns.split_qkv,
            ),
            io.state_transform(
                source_key="decoder.layers.*.self_attention.linear_qkv.bias",
                target_key=(
                    "model.layers.*.self_attn.q_proj.bias",
                    "model.layers.*.self_attn.k_proj.bias",
                    "model.layers.*.self_attn.v_proj.bias",
                ),
                fn=TransformFns.split_qkv_bias,
            ),
            io.state_transform(
                source_key="decoder.layers.*.mlp.linear_fc1.weight",
                target_key=("model.layers.*.mlp.gate_proj.weight", "model.layers.*.mlp.up_proj.weight"),
                fn=TransformFns.split_fc1,
            ),
            io.state_transform(
                source_key="embedding.word_embeddings.weight",
                target_key="model.embed_tokens.weight",
                fn=TransformFns.prune_padding,
            ),
            io.state_transform(
                source_key="output_layer.weight",
                target_key="lm_head.weight",
                fn=TransformFns.prune_padding,
            ),
        ]
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
    def config(self) -> "HFQwen2Config":
        from transformers import Qwen2Config as HFQwen2Config

        source: Qwen2Config = io.load_context(str(self), subpath="model.config")

        return HFQwen2Config(
            architectures=["Qwen2ForCausalLM"],
            num_hidden_layers=source.num_layers,
            hidden_size=source.hidden_size,
            intermediate_size=source.ffn_hidden_size,
            num_attention_heads=source.num_attention_heads,
            head_dim=(
                source.kv_channels
                if source.kv_channels is not None
                else source.hidden_size // source.num_attention_heads
            ),
            max_position_embeddings=source.seq_length,
            initializer_range=source.init_method_std,
            rms_norm_eps=source.layernorm_epsilon,
            num_key_value_heads=source.num_query_groups,
            rope_theta=source.rotary_base,
            vocab_size=getattr(source, 'vocab_size', self.tokenizer.vocab_size),
            sliding_window=source.seq_length,
            tie_word_embeddings=False,
        )


__all__ = [
    "Qwen2Config",
    "Qwen2Config500M",
    "Qwen2Config1P5B",
    "Qwen25Config3B",
    "Qwen2Config7B",
    "Qwen2Config72B",
    "Qwen25Config500M",
    "Qwen25Config1P5B",
    "Qwen25Config7B",
    "Qwen25Config14B",
    "Qwen25Config32B",
    "Qwen25Config72B",
    "Qwen2Model",
]
