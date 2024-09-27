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
from typing import TYPE_CHECKING, Annotated, Callable, Optional

import torch.nn.functional as F
from torch import nn

from nemo.collections.llm.gpt.model.base import GPTConfig, GPTModel
from nemo.collections.llm.utils import Config
from nemo.lightning import OptimizerModule, io, teardown
from nemo.lightning.pytorch.utils import dtype_from_hf

if TYPE_CHECKING:
    from transformers import GPTBigCodeConfig as HFStarcoderConfig
    from transformers import GPTBigCodeForCausalLM

    from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
    from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec


@dataclass
class StarcoderConfig(GPTConfig):
    # configs that are common across model sizes
    normalization: str = "LayerNorm"
    activation_func: Callable = F.gelu
    add_bias_linear: bool = True
    seq_length: int = 8192
    position_embedding_type: str = "learned_absolute"
    hidden_dropout: float = 0.2
    attention_dropout: float = 0.2
    init_method_std: float = 0.01
    layernorm_epsilon: float = 1e-5
    share_embeddings_and_output_weights: bool = False
    kv_channels: int = None
    num_query_groups: int = 1
    attention_softmax_in_fp32: bool = True
    bias_activation_fusion: bool = True
    bias_dropout_fusion: bool = True


@dataclass
class StarcoderConfig15B(StarcoderConfig):
    num_layers: int = 40
    hidden_size: int = 6144
    ffn_hidden_size: int = 24576
    num_attention_heads: int = 48
    init_method_std: float = 0.02


class StarcoderModel(GPTModel):
    def __init__(
        self,
        config: Annotated[Optional[StarcoderConfig], Config[StarcoderConfig]] = None,
        optim: Optional[OptimizerModule] = None,
        tokenizer: Optional["TokenizerSpec"] = None,
        model_transform: Optional[Callable[[nn.Module], nn.Module]] = None,
    ):
        super().__init__(
            config or StarcoderConfig(), optim=optim, tokenizer=tokenizer, model_transform=model_transform
        )


@io.model_importer(StarcoderModel, "hf")
class HFStarcoderImporter(io.ModelConnector["GPTBigCodeForCausalLM", StarcoderModel]):
    def init(self) -> StarcoderModel:
        return StarcoderModel(self.config, tokenizer=self.tokenizer)

    def apply(self, output_path: Path) -> Path:
        from transformers import GPTBigCodeForCausalLM

        source = GPTBigCodeForCausalLM.from_pretrained(str(self), torch_dtype='auto')
        target = self.init()
        trainer = self.nemo_setup(target)
        self.convert_state(source, target)
        self.nemo_save(output_path, trainer)

        print(f"Converted Starcoder model to Nemo, model saved to {output_path}")

        teardown(trainer, target)
        del trainer, target

        return output_path

    def convert_state(self, source, target):
        mapping = {
            "transformer.wte.weight": "embedding.word_embeddings.weight",
            "transformer.wpe.weight": "embedding.position_embeddings.weight",
            "transformer.h.*.attn.c_proj.weight": "decoder.layers.*.self_attention.linear_proj.weight",
            "transformer.h.*.attn.c_proj.bias": "decoder.layers.*.self_attention.linear_proj.bias",
            "transformer.h.*.attn.c_attn.weight": "decoder.layers.*.self_attention.linear_qkv.weight",
            "transformer.h.*.attn.c_attn.bias": "decoder.layers.*.self_attention.linear_qkv.bias",
            "transformer.h.*.mlp.c_fc.weight": "decoder.layers.*.mlp.linear_fc1.weight",
            "transformer.h.*.mlp.c_fc.bias": "decoder.layers.*.mlp.linear_fc1.bias",
            "transformer.h.*.mlp.c_proj.weight": "decoder.layers.*.mlp.linear_fc2.weight",
            "transformer.h.*.mlp.c_proj.bias": "decoder.layers.*.mlp.linear_fc2.bias",
            "transformer.h.*.ln_1.weight": "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight",
            "transformer.h.*.ln_1.bias": "decoder.layers.*.self_attention.linear_qkv.layer_norm_bias",
            "transformer.h.*.ln_2.weight": "decoder.layers.*.mlp.linear_fc1.layer_norm_weight",
            "transformer.h.*.ln_2.bias": "decoder.layers.*.mlp.linear_fc1.layer_norm_bias",
            "transformer.ln_f.weight": "decoder.final_layernorm.weight",
            "transformer.ln_f.bias": "decoder.final_layernorm.bias",
            "lm_head.weight": "output_layer.weight",
        }

        return io.apply_transforms(source, target, mapping=mapping)

    @property
    def tokenizer(self) -> "AutoTokenizer":
        from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer

        return AutoTokenizer(self.save_hf_tokenizer_assets(str(self)))

    @property
    def config(self) -> StarcoderConfig:
        from transformers import GPTBigCodeConfig as HFStarcoderConfig

        source = HFStarcoderConfig.from_pretrained(str(self))

        def make_vocab_size_divisible_by(vocab_size):
            base = 128
            while vocab_size % base != 0:
                base //= 2
            return base

        output = StarcoderConfig(
            num_layers=source.n_layer,
            hidden_size=source.n_embd,
            ffn_hidden_size=source.n_inner,
            num_attention_heads=source.n_head,
            init_method_std=source.initializer_range,
            seq_length=source.n_positions,
            layernorm_epsilon=source.layer_norm_epsilon,
            num_query_groups=1,
            make_vocab_size_divisible_by=make_vocab_size_divisible_by(source.vocab_size),
            share_embeddings_and_output_weights=False,
            fp16=(dtype_from_hf(source) == torch.float16),
            bf16=(dtype_from_hf(source) == torch.bfloat16),
            params_dtype=dtype_from_hf(source),
        )

        return output


@io.model_exporter(StarcoderModel, "hf")
class HFStarcoderExporter(io.ModelConnector[StarcoderModel, "GPTBigCodeForCausalLM"]):
    def init(self) -> "GPTBigCodeForCausalLM":
        from transformers import GPTBigCodeForCausalLM
        from transformers.modeling_utils import no_init_weights

        with no_init_weights(True):
            return GPTBigCodeForCausalLM._from_config(self.config)

    def apply(self, output_path: Path) -> Path:
        target = self.init()
        source, _ = self.nemo_load(str(self))
        target = self.convert_state(source, target)

        target = target.cpu()
        target.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        return output_path

    def convert_state(self, source, target):
        mapping = {
            "embedding.word_embeddings.weight": "transformer.wte.weight",
            "embedding.position_embeddings.weight": "transformer.wpe.weight",
            "decoder.layers.*.self_attention.linear_proj.weight": "transformer.h.*.attn.c_proj.weight",
            "decoder.layers.*.self_attention.linear_proj.bias": "transformer.h.*.attn.c_proj.bias",
            "decoder.layers.*.self_attention.linear_qkv.weight": "transformer.h.*.attn.c_attn.weight",
            "decoder.layers.*.self_attention.linear_qkv.bias": "transformer.h.*.attn.c_attn.bias",
            "decoder.layers.*.mlp.linear_fc1.weight": "transformer.h.*.mlp.c_fc.weight",
            "decoder.layers.*.mlp.linear_fc1.bias": "transformer.h.*.mlp.c_fc.bias",
            "decoder.layers.*.mlp.linear_fc2.weight": "transformer.h.*.mlp.c_proj.weight",
            "decoder.layers.*.mlp.linear_fc2.bias": "transformer.h.*.mlp.c_proj.bias",
            "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "transformer.h.*.ln_1.weight",
            "decoder.layers.*.self_attention.linear_qkv.layer_norm_bias": "transformer.h.*.ln_1.bias",
            "decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "transformer.h.*.ln_2.weight",
            "decoder.layers.*.mlp.linear_fc1.layer_norm_bias": "transformer.h.*.ln_2.bias",
            "decoder.final_layernorm.weight": "transformer.ln_f.weight",
            "decoder.final_layernorm.bias": "transformer.ln_f.bias",
            "output_layer.weight": "lm_head.weight",
        }

        return io.apply_transforms(source, target, mapping=mapping)

    @property
    def tokenizer(self):
        return io.load_context(str(self)).model.tokenizer.tokenizer

    @property
    def config(self) -> "HFStarcoderConfig":
        from transformers import sGPTBigCodeConfig as HFStarcoderConfig

        source: StarcoderConfig = io.load_context(str(self)).model.config

        return HFStarcoderConfig(
            num_hidden_layers=source.num_layers,
            hidden_size=source.hidden_size,
            intermediate_size=source.ffn_hidden_size,
            num_attention_heads=source.num_attention_heads,
            head_dim=(
                source.kv_channels
                if source.kv_channels is not None
                else source.hidden_size // source.num_attention_heads
            ),
            tie_word_embeddings=source.share_embeddings_and_output_weights,
            max_position_embeddings=source.seq_length,
            initializer_range=source.init_method_std,
            norm_eps=source.layernorm_epsilon,
            num_key_value_heads=source.num_query_groups,
            vocab_size=self.tokenizer.vocab_size,
        )
