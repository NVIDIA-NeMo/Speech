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

import torch
import torch.nn.functional as F
from torch import nn

from nemo.collections.llm.gpt.model.base import GPTConfig, GPTModel, torch_dtype_from_mcore_config
from nemo.collections.llm.utils import Config
from nemo.lightning import OptimizerModule, io, teardown
from nemo.lightning.io.state import TransformFns
from nemo.lightning.pytorch.utils import dtype_from_hf

if TYPE_CHECKING:
    from transformers import AutoModelForCausalLM

    from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
    from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec


@dataclass
class Baichuan2Config(GPTConfig):
    """
    Configuration class for the Baichuan2 Config, inheriting from GPTConfig.
    """

    normalization: str = "RMSNorm"
    activation_func: Callable = F.silu
    gated_linear_unit: bool = True
    add_bias_linear: bool = False
    seq_length: int = 4096
    init_method_std: int = 0.02
    layernorm_epsilon: float = 1e-6
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0
    share_embeddings_and_output_weights: bool = False


@dataclass
class Baichuan2Config7B(Baichuan2Config):
    """
    Configuration class for the Baichuan2 7B Config, inheriting from Baichuan2Config.
    """

    num_layers: int = 32
    hidden_size: int = 4096
    num_attention_heads: int = 32
    num_query_groups: int = 32
    ffn_hidden_size: int = 11008
    position_embedding_type: str = "rope"


class Baichuan2Model(GPTModel):
    """
    Baichuan2 model implementation based on the GPT model architecture.

    This class provides a high-level interface for Baichuan2 models,
    implementing the specific architecture and settings needed for Baichuan2 models.
    """

    def __init__(
        self,
        config: Annotated[Optional[Baichuan2Config], Config[Baichuan2Config]] = None,
        optim: Optional[OptimizerModule] = None,
        tokenizer: Optional["TokenizerSpec"] = None,
        model_transform: Optional[Callable[[nn.Module], nn.Module]] = None,
    ):
        super().__init__(
            config or Baichuan2Config(), optim=optim, tokenizer=tokenizer, model_transform=model_transform
        )


@io.model_importer(Baichuan2Model, "hf")
class HFBaichuan2Importer(io.ModelConnector["AutoModelForCausalLM", Baichuan2Model]):
    """
    Importer for converting Hugging Face Baichuan2 models to NeMo format.

    This class handles the conversion of Hugging Face's BaichuanForCausalLM models
    to NeMo's Baichuan2 format, including weight mapping and configuration translation.
    """

    def init(self) -> Baichuan2Model:
        """
        Initialize a NeMo Baichuan2Model instance.

        Returns:
            Baichuan2Model: Initialized NeMo Llama model with the appropriate configuration
                        and tokenizer.
        """
        return Baichuan2Model(self.config, tokenizer=self.tokenizer)

    def apply(self, output_path: Path) -> Path:
        """
        Apply the conversion from HF to NeMo format.

        Args:
            output_path: Path where the converted model will be saved

        Returns:
            Path: Path to the saved NeMo model
        """
        from transformers import AutoModelForCausalLM

        source = AutoModelForCausalLM.from_pretrained(str(self), trust_remote_code=True, torch_dtype='auto')
        target = self.init()
        trainer = self.nemo_setup(target)
        self.convert_state(source, target)
        self.nemo_save(output_path, trainer)

        print(f"Converted Baichuan model to Nemo, model saved to {output_path}")

        teardown(trainer, target)
        del trainer, target

        return output_path

    def convert_state(self, source, target):
        """
        Convert state dict from HF format to NeMo format.

        Maps the weights from the HF model to the NeMo model according to
        the appropriate mapping scheme.

        Args:
            source: Source HF model
            target: Target NeMo model

        Returns:
            The result of applying the transforms
        """
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
            _import_qkv,
            io.state_transform(
                source_key=("model.layers.*.mlp.gate_proj.weight", "model.layers.*.mlp.up_proj.weight"),
                target_key="decoder.layers.*.mlp.linear_fc1.weight",
                fn=TransformFns.merge_fc1,
            ),
        ]
        return io.apply_transforms(source, target, mapping=mapping, transforms=transforms)

    @property
    def tokenizer(self) -> "AutoTokenizer":
        """
        Get the tokenizer for the HF model.

        Returns:
            AutoTokenizer: Tokenizer instance initialized from the HF model's tokenizer
        """
        from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer

        return AutoTokenizer(self.save_hf_tokenizer_assets(str(self)), trust_remote_code=True)

    @property
    def config(self) -> Baichuan2Config:
        """
        Create a NeMo Baichuan2Config from the HF model config.

        Translates the HF configuration parameters to the equivalent NeMo
        configuration.

        Returns:
            Baichuan2Config: NeMo configuration for Baichuan2 models
        """
        from transformers import AutoConfig as HFAutoConfig

        source = HFAutoConfig.from_pretrained(str(self), trust_remote_code=True)

        def make_vocab_size_divisible_by(vocab_size):
            base = 128
            while vocab_size % base != 0:
                base //= 2
            return base

        output = Baichuan2Config(
            num_layers=source.num_hidden_layers,
            hidden_size=source.hidden_size,
            ffn_hidden_size=source.intermediate_size,
            num_attention_heads=source.num_attention_heads,
            init_method_std=source.initializer_range,
            layernorm_epsilon=source.rms_norm_eps,
            gated_linear_unit=True,
            make_vocab_size_divisible_by=make_vocab_size_divisible_by(source.vocab_size),
            share_embeddings_and_output_weights=False,
            position_embedding_type="rope" if source.num_hidden_layers == 32 else "alibi",
            fp16=(dtype_from_hf(source) == torch.float16),
            bf16=(dtype_from_hf(source) == torch.bfloat16),
            params_dtype=dtype_from_hf(source),
        )

        return output


@io.model_exporter(Baichuan2Model, "hf")
class HFBaichuan2Exporter(io.ModelConnector[Baichuan2Model, "AutoModelForCausalLM"]):
    """
    Exporter for converting NeMo Baichuan2Model to Hugging Face format.

    This class handles the conversion of NeMo's Baichuan2Model to Hugging Face's
    BaichuanForCausalLM format, including weight mapping and configuration translation.
    """

    def init(self, dtype=torch.bfloat16, model_name=None) -> "AutoModelForCausalLM":
        """
        Initialize a HF BaichuanForCausalLM instance.

        Args:
            dtype: Data type for model parameters

        Returns:
            AutoModelForCausalLM: Initialized HF Baichuan model
        """
        from transformers import AutoConfig, AutoModelForCausalLM
        from transformers.modeling_utils import no_init_weights

        if model_name is None:
            model_name = "baichuan-inc/Baichuan2-7B-Base"
        with no_init_weights():
            # Since Baichuan2 is not importable from transformers, we can only initialize the HF model
            # from a known checkpoint folder containing the config file and modeling files.
            # The model_name will need to be passed in.
            config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            hf_model = AutoModelForCausalLM.from_config(
                config,
                trust_remote_code=True,
                torch_dtype=dtype,
            )
            # Register the AutoModel Hook so that the custom modeling files are saved during save_pretrained()
            type(hf_model).register_for_auto_class("AutoModelForCausalLM")
            return hf_model

    def apply(self, output_path: Path, target_model_name=None) -> Path:
        source, _ = self.nemo_load(str(self))
        target = self.init(torch_dtype_from_mcore_config(source.config), model_name=target_model_name)
        target = self.convert_state(source, target)

        target = target.cpu()
        target.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        return output_path

    def convert_state(self, source, target):
        """
        Convert state dict from NeMo format to HF format.

        Maps the weights from the NeMo model to the HF model according to
        the appropriate mapping scheme.

        Args:
            source: Source NeMo model
            target: Target HF model

        Returns:
            The target model with weights transferred from source
        """
        mapping = {
            "embedding.word_embeddings.weight": "model.embed_tokens.weight",
            "decoder.layers.*.self_attention.linear_proj.weight": "model.layers.*.self_attn.o_proj.weight",
            "decoder.layers.*.mlp.linear_fc2.weight": "model.layers.*.mlp.down_proj.weight",
            "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "model.layers.*.input_layernorm.weight",
            "decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "model.layers.*.post_attention_layernorm.weight",
            "decoder.final_layernorm.weight": "model.norm.weight",
            "output_layer.weight": "lm_head.weight",
        }

        transforms = [
            _export_qkv,
            io.state_transform(
                source_key="decoder.layers.*.mlp.linear_fc1.weight",
                target_key=("model.layers.*.mlp.gate_proj.weight", "model.layers.*.mlp.up_proj.weight"),
                fn=TransformFns.split_fc1,
            ),
        ]
        return io.apply_transforms(source, target, mapping=mapping, transforms=transforms)

    @property
    def tokenizer(self):
        """
        Get the tokenizer from the NeMo model.

        Returns:
            TokenizerSpec: Tokenizer from the NeMo model
        """
        return io.load_context(str(self)).model.tokenizer.tokenizer


@io.state_transform(
    source_key="model.layers.*.self_attn.W_pack.weight",
    target_key="decoder.layers.*.self_attention.linear_qkv.weight",
)
def _import_qkv(ctx: io.TransformCTX, qkv_weights):
    megatron_config = ctx.target.config

    head_num = megatron_config.num_attention_heads
    num_query_groups = megatron_config.num_query_groups
    heads_per_group = head_num // num_query_groups
    hidden_size = megatron_config.hidden_size
    head_size = megatron_config.kv_channels

    qkv_weights = qkv_weights.unflatten(0, (3, hidden_size))
    old_tensor_shape = qkv_weights[0].size()
    new_q_tensor_shape = (head_num, head_size) + old_tensor_shape[1:]
    new_kv_tensor_shape = (num_query_groups, head_size) + old_tensor_shape[1:]
    q = qkv_weights[0].squeeze().view(*new_q_tensor_shape)
    k = qkv_weights[1].squeeze().view(*new_kv_tensor_shape)
    v = qkv_weights[2].squeeze().view(*new_kv_tensor_shape)
    qkv_weights = torch.empty((0, head_size) + old_tensor_shape[1:]).type_as(qkv_weights)
    for i in range(num_query_groups):
        qkv_weights = torch.cat((qkv_weights, q[i * heads_per_group : (i + 1) * heads_per_group, :, :]))
        qkv_weights = torch.cat((qkv_weights, k[i : i + 1, :, :]))
        qkv_weights = torch.cat((qkv_weights, v[i : i + 1, :, :]))

    qkv_weights = qkv_weights.reshape([head_size * (head_num + 2 * num_query_groups), hidden_size])
    return qkv_weights


@io.state_transform(
    source_key="decoder.layers.*.self_attention.linear_qkv.weight",
    target_key="model.layers.*.self_attn.W_pack.weight",
)
def _export_qkv(ctx: io.TransformCTX, qkv_weights):
    megatron_config = ctx.source.config

    head_num = megatron_config.num_attention_heads
    num_query_groups = megatron_config.num_query_groups
    heads_per_group = head_num // num_query_groups
    hidden_size = megatron_config.hidden_size
    head_size = megatron_config.kv_channels
    qkv_total_dim = head_num + 2 * num_query_groups

    qkv_weights = qkv_weights.reshape([qkv_total_dim, head_size, hidden_size])

    q_slice = torch.cat(
        [
            torch.arange((heads_per_group + 2) * i, (heads_per_group + 2) * i + heads_per_group)
            for i in range(num_query_groups)
        ]
    )
    k_slice = torch.arange(heads_per_group, qkv_total_dim, (heads_per_group + 2))
    v_slice = torch.arange(heads_per_group + 1, qkv_total_dim, (heads_per_group + 2))

    return torch.cat(
        [
            qkv_weights[q_slice].reshape(-1, hidden_size),
            qkv_weights[k_slice].reshape(-1, hidden_size),
            qkv_weights[v_slice].reshape(-1, hidden_size),
        ]
    )


__all__ = [
    "Baichuan2Config",
    "Baichuan2Config7B",
    "Baichuan2Model",
]
