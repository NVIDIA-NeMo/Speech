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

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, List, Optional

import torch
import torch.nn.functional as F
from torch import nn
from typing_extensions import Annotated

from nemo.collections.llm.gpt.model.base import GPTConfig, GPTModel, torch_dtype_from_mcore_config
from nemo.collections.llm.utils import Config
from nemo.lightning import io, teardown
from nemo.lightning.io.state import TransformFns
from nemo.lightning.pytorch.optim import OptimizerModule
from nemo.lightning.pytorch.utils import dtype_from_hf

if TYPE_CHECKING:
    from transformers import MistralConfig, MistralForCausalLM

    from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
    from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec


@dataclass
class MistralConfig7B(GPTConfig):
    """
    Mistral 7B config.
    """

    normalization: str = "RMSNorm"
    activation_func: Callable = F.silu
    position_embedding_type: str = "rope"
    add_bias_linear: bool = False
    gated_linear_unit: bool = True

    num_layers: int = 32
    hidden_size: int = 4096
    num_attention_heads: int = 32
    num_query_groups: int = 8
    ffn_hidden_size: int = 14336
    seq_length: int = 32768
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    share_embeddings_and_output_weights: bool = False

    init_method_std: float = 0.02
    layernorm_epsilon: float = 1e-5
    window_size: List[int] = field(default_factory=lambda: [4096, 0])
    cp_comm_type: str = "a2a"
    params_dtype: torch.dtype = torch.bfloat16


@dataclass
class MistralNeMoConfig12B(MistralConfig7B):
    """
    https://mistral.ai/news/mistral-nemo/
    """

    num_layers: int = 40
    hidden_size: int = 5120
    kv_channels: int = 128
    seq_length: int = 4096  # but   "max_position_embeddings": 1024000,

    window_size: List[int] = None
    cp_comm_type: str = None
    rotary_percent: float = 1.0
    rotary_base: float = 1000000.0
    params_dtype: torch.dtype = torch.bfloat16


@dataclass
class MistralNeMoConfig123B(MistralConfig7B):
    """
    https://mistral.ai/news/mistral-large-2407/
    """

    num_layers: int = 88
    hidden_size: int = 12288
    ffn_hidden_size: int = 28672
    num_attention_heads: int = 96
    kv_channels: int = 128
    seq_length: int = 4096  # but   "max_position_embeddings": 131072,

    window_size: List[int] = None
    cp_comm_type: str = None
    rotary_percent: float = 1.0
    rotary_base: float = 1000000.0
    params_dtype: torch.dtype = torch.bfloat16


@dataclass
class MistralSmall3Config24B(MistralConfig7B):
    """
    https://mistral.ai/news/mistral-small-3/
    """

    num_layers: int = 40
    hidden_size: int = 5120
    ffn_hidden_size: int = 32768
    num_attention_heads: int = 32
    kv_channels: int = 128
    seq_length: int = 32768

    window_size: List[int] = None
    cp_comm_type: str = None
    rotary_percent: float = 1.0
    rotary_base: float = 100000000.0
    params_dtype: torch.dtype = torch.bfloat16


class MistralModel(GPTModel):
    """ """

    def __init__(
        self,
        config: Annotated[Optional[MistralConfig7B], Config[MistralConfig7B]] = None,
        optim: Optional[OptimizerModule] = None,
        tokenizer: Optional["TokenizerSpec"] = None,
        model_transform: Optional[Callable[[nn.Module], nn.Module]] = None,
    ):
        super().__init__(
            config or MistralConfig7B(), optim=optim, tokenizer=tokenizer, model_transform=model_transform
        )


@io.model_importer(MistralModel, "hf")
class HFMistralImporter(io.ModelConnector["MistralForCausalLM", MistralModel]):
    """ """

    def init(self) -> MistralModel:
        return MistralModel(self.config, tokenizer=self.tokenizer)

    def apply(self, output_path: Path) -> Path:
        from transformers import MistralForCausalLM

        source = MistralForCausalLM.from_pretrained(str(self), torch_dtype='auto')
        target = self.init()
        trainer = self.nemo_setup(target)
        self.convert_state(source, target)
        self.nemo_save(output_path, trainer)

        print(f"Converted Mistral 7B model to Nemo, model saved to {output_path}")

        teardown(trainer, target)
        del trainer, target

        return output_path

    def convert_state(self, source, target):
        """ """
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
                source_key=("model.layers.*.mlp.gate_proj.weight", "model.layers.*.mlp.up_proj.weight"),
                target_key="decoder.layers.*.mlp.linear_fc1.weight",
                fn=TransformFns.merge_fc1,
            ),
        ]
        return io.apply_transforms(source, target, mapping=mapping, transforms=transforms)

    @property
    def tokenizer(self) -> "AutoTokenizer":
        """ """
        from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer

        return AutoTokenizer(self.save_hf_tokenizer_assets(str(self)))

    @property
    def config(self) -> MistralConfig7B:
        """ """
        from transformers import GenerationConfig, MistralConfig

        source = MistralConfig.from_pretrained(str(self))
        generation_config = GenerationConfig.from_pretrained(str(self))

        def make_vocab_size_divisible_by(mistral_vocab_size):
            base = 128
            while mistral_vocab_size % base != 0:
                base //= 2
            return base

        window_size, cp_comm_type = (None, None)
        if getattr(source, 'sliding_window', None) is not None:
            window_size = [source.sliding_window, 0]
            cp_comm_type = 'a2a'
        output = MistralConfig7B(
            seq_length=source.sliding_window,
            num_layers=source.num_hidden_layers,
            hidden_size=source.hidden_size,
            ffn_hidden_size=source.intermediate_size,
            kv_channels=getattr(source, 'head_dim', source.hidden_size // source.num_attention_heads),
            num_attention_heads=source.num_attention_heads,
            # max_position_embeddings=source.max_position_embeddings,
            init_method_std=source.initializer_range,
            layernorm_epsilon=source.rms_norm_eps,
            num_query_groups=source.num_key_value_heads,
            rotary_base=source.rope_theta,
            gated_linear_unit=True,
            make_vocab_size_divisible_by=make_vocab_size_divisible_by(source.vocab_size),
            window_size=window_size,
            cp_comm_type=cp_comm_type,
            share_embeddings_and_output_weights=False,
            fp16=(dtype_from_hf(source) == torch.float16),
            bf16=(dtype_from_hf(source) == torch.bfloat16),
            params_dtype=dtype_from_hf(source),
            generation_config=generation_config,
        )

        return output


@io.model_exporter(MistralModel, "hf")
class HFMistralExporter(io.ModelConnector[MistralModel, "MistralForCausalLM"]):
    """ """

    def init(self, dtype=torch.bfloat16) -> "MistralForCausalLM":
        from transformers import AutoModelForCausalLM
        from transformers.modeling_utils import no_init_weights

        with no_init_weights():
            return AutoModelForCausalLM.from_config(self.config, torch_dtype=dtype)

    def apply(self, output_path: Path) -> Path:
        # TODO: Make it work with lazy init
        # with torch.device("meta"):
        #     target = self.init()
        source, _ = self.nemo_load(str(self))
        target = self.init(torch_dtype_from_mcore_config(source.config))
        target = self.convert_state(source, target)

        # TODO: Make sure we don't need to do this
        target = target.cpu()
        target.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        return output_path

    def convert_state(self, source, target):
        """ """
        mapping = {
            "decoder.layers.*.self_attention.linear_proj.weight": "model.layers.*.self_attn.o_proj.weight",
            "decoder.layers.*.mlp.linear_fc2.weight": "model.layers.*.mlp.down_proj.weight",
            "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "model.layers.*.input_layernorm.weight",
            "decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "model.layers.*.post_attention_layernorm.weight",
            "decoder.final_layernorm.weight": "model.norm.weight",
        }

        transforms = [
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
                source_key="decoder.layers.*.mlp.linear_fc1.weight",
                target_key=("model.layers.*.mlp.gate_proj.weight", "model.layers.*.mlp.up_proj.weight"),
                fn=TransformFns.split_fc1,
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
        """ """
        return io.load_context(str(self)).model.tokenizer.tokenizer

    @property
    def config(self) -> "MistralConfig":
        """ """
        source: MistralConfig7B = io.load_context(str(self), subpath="model.config")

        from transformers import MistralConfig as HfMistralConfig

        return HfMistralConfig(
            architectures=["MistralForCausalLM"],
            sliding_window=source.window_size[0] if source.window_size is not None else None,
            num_hidden_layers=source.num_layers,
            hidden_size=source.hidden_size,
            intermediate_size=source.ffn_hidden_size,
            num_attention_heads=source.num_attention_heads,
            max_position_embeddings=source.seq_length,
            initializer_range=source.init_method_std,
            rms_norm_eps=source.layernorm_epsilon,
            num_key_value_heads=source.num_query_groups,
            rope_theta=source.rotary_base,
            vocab_size=self.tokenizer.vocab_size,
            head_dim=source.kv_channels,
        )


__all__ = [
    "MistralConfig7B",
    "MistralModel",
]
