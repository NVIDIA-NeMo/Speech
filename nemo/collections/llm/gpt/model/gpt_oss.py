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
import json
import math
import os
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Callable, List, Literal, Optional, Tuple, Union

import torch
from safetensors import safe_open
from torch import nn
from transformers import AutoModelForCausalLM, GenerationConfig

from nemo.collections.common.tokenizers import AutoTokenizer
from nemo.collections.common.tokenizers.tiktoken_tokenizer import TiktokenTokenizer
from nemo.collections.llm.gpt.model.base import GPTConfig, GPTModel, torch_dtype_from_mcore_config
from nemo.collections.llm.utils import Config
from nemo.lightning import OptimizerModule, io, teardown
from nemo.lightning.io.state import TransformFns, _ModelState
from nemo.utils import logging
from nemo.utils.import_utils import safe_import_from

if TYPE_CHECKING:
    from peft import AutoPeftModelForCausalLM, PeftConfig

    from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec

quick_gelu, HAVE_QUICK_GELU = safe_import_from("megatron.core.activations", "quick_gelu", alt=object)


@dataclass
class GPTOSSConfig(GPTConfig):
    """
    Base config for GPT-OSS
    """

    hidden_size: int = 2880
    num_attention_heads: int = 64
    num_query_groups: int = 8
    ffn_hidden_size: int = 2880
    kv_channels: Optional[int] = 64
    normalization: str = "RMSNorm"
    gated_linear_unit: bool = True
    add_bias_linear: bool = True
    share_embeddings_and_output_weights: bool = False
    vocab_size: int = 201088
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0

    position_embedding_type: str = "yarn"
    rotary_base: int = 150000
    yarn_rotary_scaling_factor: float = 32.0
    yarn_original_max_position_embeddings: int = 4096
    yarn_beta_fast: float = 32.0
    yarn_beta_slow: float = 1.0
    yarn_correction_range_round_to_int: bool = False
    yarn_mscale: float = 1.0
    yarn_mscale_all_dim: float = 1.0

    moe_router_topk: int = 4
    moe_router_pre_softmax: bool = False
    moe_grouped_gemm: bool = True
    moe_token_dispatcher_type: str = "alltoall"
    moe_permute_fusion: bool = True
    moe_ffn_hidden_size: int = 2880
    moe_router_load_balancing_type: str = "none"
    seq_length: int = 131072
    window_size: Optional[Tuple[int, int]] = (128, 0)
    softmax_type: Literal['vanilla', 'off-by-one', 'learnable'] = "learnable"
    activation_func: Callable = quick_gelu
    glu_linear_offset: float = 1.0
    bias_activation_fusion: bool = True
    window_attn_skip_freq: Optional[Union[int, List[int]]] = 2  # alternative SWA/full
    activation_func_clamp_value: Optional[float] = 7.0


@dataclass
class GPTOSSConfig120B(GPTOSSConfig):
    """Config for GPT-OSS 120B"""

    num_layers: int = 36
    num_moe_experts: int = 128


@dataclass
class GPTOSSConfig20B(GPTOSSConfig):
    """Config for GPT-OSS 20B"""

    num_layers: int = 24
    num_moe_experts: int = 32


class GPTOSSModel(GPTModel):
    """
    Base model for GPT-OSS
    """

    def __init__(
        self,
        config: Annotated[Optional[GPTOSSConfig], Config[GPTOSSConfig]] = None,
        optim: Optional[OptimizerModule] = None,
        tokenizer: Optional["TokenizerSpec"] = None,
        model_transform: Optional[Callable[[nn.Module], nn.Module]] = None,
    ):
        assert HAVE_QUICK_GELU, "Megatron version does not support gpt-oss model."
        super().__init__(config or GPTOSSConfig(), optim=optim, tokenizer=tokenizer, model_transform=model_transform)


class _BaseGPTOSSImporter(io.ModelConnector["AutoModelForCausalLM", GPTOSSModel]):
    # pylint: disable=C0115,C0116
    def init(self) -> GPTOSSModel:
        return GPTOSSModel(self.config, tokenizer=self.tokenizer)

    def hf_ckpt_load(self):
        loaded_bf16_data = {}
        loaded_mxfp4_data = {}
        folder_path = str(self)
        # Check if the provided folder path exists
        if not os.path.isdir(folder_path):
            logging.error(f"Error: Folder '{folder_path}' not found.")
            return {}

        # Iterate through all files in the specified folder
        for filename in os.listdir(folder_path):
            # Construct the full file path
            file_path = os.path.join(folder_path, filename)

            # Check if the file is a .safetensors file and is actually a file
            if os.path.isfile(file_path) and filename.endswith(".safetensors"):
                logging.debug(f"Attempting to load: {filename}")
                try:
                    with safe_open(file_path, framework="pt", device="cpu") as f:
                        for key in f.keys():
                            if key.endswith("blocks") or key.endswith("scales"):
                                loaded_mxfp4_data[key] = f.get_tensor(key)
                            else:
                                loaded_bf16_data[key] = f.get_tensor(key)
                    logging.debug(f"Successfully loaded '{filename}'")

                except Exception as e:
                    logging.error(f"Error loading '{filename}': {e}")
            else:
                logging.debug(f"Skipping non-safetensors file or directory: {filename}")

        # Convert MXFP4 weights to BF16
        for k, v in loaded_mxfp4_data.items():
            if k.endswith("scales"):
                continue  # process scales in the iteration of blocks
            blocks = v
            scales = loaded_mxfp4_data[k.replace("blocks", "scales")].to(torch.int32) - 127
            new_key = k.replace(".blocks", "").replace("_blocks", "")
            loaded_bf16_data[new_key] = self._dequantize_mxfp4(blocks, scales)
            logging.debug(f"Successfully dequantized {new_key}")

        return loaded_bf16_data

    def _dequantize_mxfp4(
        self,
        blocks: torch.Tensor,
        scales: torch.Tensor,
        *,
        dtype: torch.dtype = torch.bfloat16,
        rows_per_chunk: int = 32768 * 1024,
    ) -> torch.Tensor:
        assert blocks.shape[:-1] == scales.shape, f"{blocks.shape=} does not match {scales.shape=}"
        FP4_VALUES = [
            +0.0,
            +0.5,
            +1.0,
            +1.5,
            +2.0,
            +3.0,
            +4.0,
            +6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ]
        lut = torch.tensor(FP4_VALUES, dtype=dtype, device=blocks.device)

        *prefix_shape, G, B = blocks.shape
        rows_total = math.prod(prefix_shape) * G

        blocks = blocks.reshape(rows_total, B)
        scales = scales.reshape(rows_total, 1)

        out = torch.empty(rows_total, B * 2, dtype=dtype, device=blocks.device)

        for r0 in range(0, rows_total, rows_per_chunk):
            r1 = min(r0 + rows_per_chunk, rows_total)

            blk = blocks[r0:r1]
            exp = scales[r0:r1]

            # nibble indices -> int64
            idx_lo = (blk & 0x0F).to(torch.long)
            idx_hi = (blk >> 4).to(torch.long)

            sub = out[r0:r1]
            sub[:, 0::2] = lut[idx_lo]
            sub[:, 1::2] = lut[idx_hi]

            torch.ldexp(sub, exp, out=sub)
            del idx_lo, idx_hi, blk, exp

        return out.reshape(*prefix_shape, G, B * 2).view(*prefix_shape, G * B * 2)


@io.model_importer(GPTOSSModel, "hf")
class HFGPTOSSImporter(_BaseGPTOSSImporter):
    """Importer for GPT-OSS models from Hugging Face"""

    # pylint: disable=C0115,C0116
    def apply(self, output_path: Path) -> Path:
        logging.setLevel(logging.DEBUG)
        source_state = self.hf_ckpt_load()
        source = _ModelState(source_state)
        target = self.init()
        trainer = self.nemo_setup(target)
        self.convert_state(source, target)
        self.nemo_save(output_path, trainer)

        print(f"Converted GPT-OSS model to Nemo, model saved to {output_path}")

        teardown(trainer, target)
        del trainer, target

        return output_path

    def convert_state(self, source, target):
        mapping = {
            "model.embed_tokens.weight": "embedding.word_embeddings.weight",
            "model.norm.weight": "decoder.final_layernorm.weight",
            "lm_head.weight": "output_layer.weight",
            "**.input_layernorm.weight": "**.self_attention.linear_qkv.layer_norm_weight",
            "**.self_attn.o_proj.bias": "**.self_attention.linear_proj.bias",
            "**.self_attn.o_proj.weight": "**.self_attention.linear_proj.weight",
            "**.self_attn.sinks": "**.self_attention.core_attention.softmax_offset",
            "**.post_attention_layernorm.weight": "**.pre_mlp_layernorm.weight",
            "**.mlp.router.bias": "**.mlp.router.bias",
            "**.mlp.router.weight": "**.mlp.router.weight",
        }

        transforms = [
            io.state_transform(
                source_key=(
                    "**.self_attn.q_proj.bias",
                    "**.self_attn.k_proj.bias",
                    "**.self_attn.v_proj.bias",
                ),
                target_key="**.self_attention.linear_qkv.bias",
                fn=TransformFns.merge_qkv_bias,
            ),
            io.state_transform(
                source_key=(
                    "**.self_attn.q_proj.weight",
                    "**.self_attn.k_proj.weight",
                    "**.self_attn.v_proj.weight",
                ),
                target_key="**.self_attention.linear_qkv.weight",
                fn=TransformFns.merge_qkv,
            ),
            io.state_transform(
                source_key="**.mlp.experts.gate_up_proj_bias",
                target_key="**.mlp.experts.linear_fc1.bias*",
                fn=lambda x: [_interleave(e) for e in [*x]],
            ),
            io.state_transform(
                source_key="**.mlp.experts.gate_up_proj",
                target_key="**.mlp.experts.linear_fc1.weight*",
                fn=lambda x: [_interleave(e) for e in [*x]],
            ),
            io.state_transform(
                source_key="**.mlp.experts.down_proj_bias",
                target_key="**.mlp.experts.linear_fc2.bias*",
                fn=lambda x: [*x],
            ),
            io.state_transform(
                source_key="**.mlp.experts.down_proj",
                target_key="**.mlp.experts.linear_fc2.weight*",
                fn=lambda x: [*x],
            ),
        ]

        io.apply_transforms(source, target, mapping=mapping, transforms=transforms, cast_dtype=torch.bfloat16)
        return target

    @cached_property
    def tokenizer(self) -> "AutoTokenizer":
        from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer

        return AutoTokenizer(self.save_hf_tokenizer_assets(str(self)), trust_remote_code=True)

    @cached_property
    def config(self) -> GPTOSSConfig:
        from transformers import AutoConfig as HFAutoConfig
        from transformers import GenerationConfig

        source = HFAutoConfig.from_pretrained(str(self), trust_remote_code=True)
        generation_config = GenerationConfig.from_pretrained(str(self))
        return GPTOSSConfig(
            num_layers=source.num_hidden_layers,
            num_moe_experts=source.num_local_experts,
            bf16=True,
            params_dtype=torch.bfloat16,
            generation_config=generation_config,
        )


@io.model_importer(GPTOSSModel, "openai")
class OpenAIGPTOSSImporter(_BaseGPTOSSImporter):
    """Importer for GPT-OSS models from OpenAI's checkpoint format"""

    # pylint: disable=C0115,C0116
    def apply(self, output_path: Path) -> Path:
        source_state = self.hf_ckpt_load()

        source = _ModelState(source_state)
        target = self.init()
        trainer = self.nemo_setup(target)
        self.convert_state(source, target)
        self.nemo_save(output_path, trainer)

        print(f"Converted GPT-OSS model to Nemo, model saved to {output_path}")

        teardown(trainer, target)
        del trainer, target

        return output_path

    def convert_state(self, source, target):
        mapping = {
            "embedding.weight": "embedding.word_embeddings.weight",
            "norm.scale": "decoder.final_layernorm.weight",
            "unembedding.weight": "output_layer.weight",
            "**.attn.norm.scale": "**.self_attention.linear_qkv.layer_norm_weight",
            "**.attn.out.bias": "**.self_attention.linear_proj.bias",
            "**.attn.out.weight": "**.self_attention.linear_proj.weight",
            "**.attn.sinks": "**.self_attention.core_attention.softmax_offset",
            "**.mlp.norm.scale": "**.pre_mlp_layernorm.weight",
            "**.mlp.gate.bias": "**.mlp.router.bias",
            "**.mlp.gate.weight": "**.mlp.router.weight",
        }

        transforms = [
            io.state_transform(
                source_key="**.attn.qkv.bias",
                target_key="**.self_attention.linear_qkv.bias",
                fn=TransformFns.merge_qkv_bias_concat,
            ),
            io.state_transform(
                source_key="**.attn.qkv.weight",
                target_key="**.self_attention.linear_qkv.weight",
                fn=TransformFns.merge_qkv_concat,
            ),
        ]

        # moe names for TEGroupedMLP
        for source_key, target_key in (
            ("**.mlp.mlp1_weight", "**.mlp.experts.linear_fc1.weight*"),
            ("**.mlp.mlp1_bias", "**.mlp.experts.linear_fc1.bias*"),
        ):
            transforms.append(io.state_transform(source_key, target_key, lambda x: [_interleave(e) for e in [*x]]))
        for source_key, target_key in (
            ("**.mlp.mlp2_bias", "**.mlp.experts.linear_fc2.bias*"),
            ("**.mlp.mlp2_weight", "**.mlp.experts.linear_fc2.weight*"),
        ):
            transforms.append(io.state_transform(source_key, target_key, lambda x: [*x]))

        io.apply_transforms(source, target, mapping=mapping, transforms=transforms, cast_dtype=torch.bfloat16)
        return target

    @cached_property
    def tokenizer(self) -> "TiktokenTokenizer":
        return TiktokenTokenizer(encoding_name="o200k_harmony")

    @cached_property
    def config(self) -> GPTOSSConfig:
        with open(self / "config.json") as f:
            ckpt_config = json.load(f)
        generation_config = GenerationConfig.from_pretrained(str(self))
        return GPTOSSConfig(
            num_layers=ckpt_config['num_hidden_layers'],
            num_moe_experts=ckpt_config['num_experts'],
            bf16=True,
            params_dtype=torch.bfloat16,
            generation_config=generation_config,
        )


def _interleave(elem):
    return torch.cat((elem[::2, ...], elem[1::2, ...]), dim=0)


def _uninterleave(elem):
    gate, up = torch.chunk(elem, 2, dim=0)
    output = torch.empty_like(elem)
    output[::2, ...] = gate
    output[1::2, ...] = up
    return output


@io.model_exporter(GPTOSSModel, "hf")
class HFGPTOSSExporter(io.ModelConnector[GPTOSSModel, "AutoModelForCausalLM"]):
    # pylint: disable=C0115,C0116
    def init(self, dtype=torch.bfloat16) -> "AutoModelForCausalLM":
        from transformers import AutoModelForCausalLM
        from transformers.modeling_utils import no_init_weights

        with no_init_weights():
            return AutoModelForCausalLM.from_config(self.config, trust_remote_code=True, torch_dtype=dtype)

    def apply(self, output_path: Path) -> Path:
        source, _ = self.nemo_load(str(self))
        target = self.init(torch_dtype_from_mcore_config(source.config))
        target = self.convert_state(source, target)

        target = target.cpu()
        target.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        return output_path

    def convert_state(self, source, target):
        mapping = {
            "embedding.word_embeddings.weight": "model.embed_tokens.weight",
            "decoder.final_layernorm.weight": "model.norm.weight",
            "output_layer.weight": "lm_head.weight",
            "**.self_attention.linear_qkv.layer_norm_weight": "**.input_layernorm.weight",
            "**.self_attention.linear_proj.bias": "**.self_attn.o_proj.bias",
            "**.self_attention.linear_proj.weight": "**.self_attn.o_proj.weight",
            "**.self_attention.core_attention.softmax_offset": "**.self_attn.sinks",
            "**.pre_mlp_layernorm.weight": "**.post_attention_layernorm.weight",
            "**.mlp.router.bias": "**.mlp.router.bias",
            "**.mlp.router.weight": "**.mlp.router.weight",
        }
        transforms = [
            io.state_transform(
                source_key="**.self_attention.linear_qkv.bias",
                target_key=(
                    "**.self_attn.q_proj.bias",
                    "**.self_attn.k_proj.bias",
                    "**.self_attn.v_proj.bias",
                ),
                fn=TransformFns.split_qkv_bias,
            ),
            io.state_transform(
                source_key="**.self_attention.linear_qkv.weight",
                target_key=(
                    "**.self_attn.q_proj.weight",
                    "**.self_attn.k_proj.weight",
                    "**.self_attn.v_proj.weight",
                ),
                fn=TransformFns.split_qkv,
            ),
        ]

        def stack_experts(*args):
            t = torch.stack(args)
            if len(t.shape) == 3:
                t = t.transpose(-1, -2)
            return t

        def stack_uninterleave_experts(*args):
            t = torch.stack([_uninterleave(e) for e in args])
            if len(t.shape) == 3:
                t = t.transpose(-1, -2)
            return t

        for source_key, target_key in (
            ("**.mlp.experts.linear_fc2.bias*", "**.mlp.experts.down_proj_bias"),
            ("**.mlp.experts.linear_fc2.weight*", "**.mlp.experts.down_proj"),
        ):
            transforms.append(io.state_transform(source_key, target_key, stack_experts))
        for source_key, target_key in (
            ("**.mlp.experts.linear_fc1.bias*", "**.mlp.experts.gate_up_proj_bias"),
            ("**.mlp.experts.linear_fc1.weight*", "**.mlp.experts.gate_up_proj"),
        ):
            transforms.append(io.state_transform(source_key, target_key, stack_uninterleave_experts))

        return io.apply_transforms(
            source,
            target,
            mapping=mapping,
            transforms=transforms,
        )

    @cached_property
    def tokenizer(self):
        return io.load_context(str(self), subpath="model.tokenizer")

    @cached_property
    def config(self):
        from transformers import GptOssConfig as HFGptOssConfig

        source: GPTOSSConfig = io.load_context(str(self), subpath="model.config")
        return HFGptOssConfig(
            num_hidden_layers=source.num_layers,
            num_local_experts=source.num_moe_experts,
        )


@io.model_exporter(GPTOSSModel, "hf-peft")
class HFGPTOSSPEFTExporter(HFGPTOSSExporter):
    """Exporter for converting NeMo GPT-OSS models with PEFT adapters to Hugging Face format.

    This class extends HFLlamaExporter to handle Parameter-Efficient Fine-Tuning (PEFT)
    adapters, specifically LoRA and DoRA adapters.
    """

    def init(self, dtype=torch.bfloat16) -> "AutoPeftModelForCausalLM":
        """Initialize a HF PEFT model.

        Args:
            dtype: Data type for model parameters

        Returns:
            AutoPeftModelForCausalLM: Initialized HF PEFT model
        """
        from peft import get_peft_model

        from nemo.lightning.ckpt_utils import ADAPTER_META_FILENAME
        from nemo.lightning.io.pl import ckpt_to_weights_subdir

        model = super().init(dtype=dtype)

        # Infer base model checkpoint from checkpoint metadata file
        adapter_meta_path = ckpt_to_weights_subdir(str(self), is_saving=False) / ADAPTER_META_FILENAME
        with open(adapter_meta_path, "r") as f:
            model_ckpt_path = json.load(f)['model_ckpt_path']
        model.name_or_path = os.path.join(model_ckpt_path.split("/")[-2:])
        return get_peft_model(model, self.peft_config, autocast_adapter_dtype=False)

    def apply(self, output_path: Path) -> Path:
        """Apply the conversion from NeMo PEFT model to HF format.

        Args:
            output_path: Path where the converted model will be saved

        Returns:
            Path: Path to the saved HF PEFT model
        """
        from nemo.collections.llm.peft import CanonicalLoRA, DoRA, LoRA

        self.peft_obj: Union[LoRA, DoRA, CanonicalLoRA] = io.load_context(str(self), subpath="model.model_transform")

        source, _ = self.nemo_load(str(self))
        target = self.init(torch_dtype_from_mcore_config(source.config))
        target = self.convert_state(source, target)
        target = target.cpu()
        target.save_pretrained(output_path, save_embedding_layers=False)

        return output_path

    def convert_state(self, source, target):
        """Convert state dict from NeMo PEFT model to HF PEFT format.

        Maps the weights from the NeMo model to the HF model according to
        the appropriate mapping scheme for PEFT adapters.

        Args:
            source: Source NeMo model with PEFT adapters
            target: Target HF model

        Returns:
            The target model with weights transferred from source
        """
        from nemo.collections.llm.peft import CanonicalLoRA

        # nemo and HF prefixes
        pn = "decoder.layers."
        ph = "base_model.model.model.layers."

        p_proj = "self_attention.linear_proj.adapter"
        p_qkv = "self_attention.linear_qkv.adapter"

        mapping = {
            # linear_proj
            f"{pn}*.{p_proj}.linear_in.weight": f"{ph}*.self_attn.o_proj.lora_A.default.weight",
            f"{pn}*.{p_proj}.linear_out.weight": f"{ph}*.self_attn.o_proj.lora_B.default.weight",
        }
        transforms = []

        if isinstance(self.peft_obj, CanonicalLoRA):
            mapping.update(
                {
                    # linear_qkv for canonical lora
                    f"{pn}*.{p_qkv}.adapter_q.linear_in.weight": f"{ph}*.self_attn.q_proj.lora_A.default.weight",
                    f"{pn}*.{p_qkv}.adapter_q.linear_out.weight": f"{ph}*.self_attn.q_proj.lora_B.default.weight",
                    f"{pn}*.{p_qkv}.adapter_k.linear_in.weight": f"{ph}*.self_attn.k_proj.lora_A.default.weight",
                    f"{pn}*.{p_qkv}.adapter_k.linear_out.weight": f"{ph}*.self_attn.k_proj.lora_B.default.weight",
                    f"{pn}*.{p_qkv}.adapter_v.linear_in.weight": f"{ph}*.self_attn.v_proj.lora_A.default.weight",
                    f"{pn}*.{p_qkv}.adapter_v.linear_out.weight": f"{ph}*.self_attn.v_proj.lora_B.default.weight",
                }
            )
        else:
            transforms.extend(
                [
                    # linear_qkv for performant lora
                    io.state_transform(
                        source_key=f"{pn}*.self_attention.linear_qkv.adapter.linear_in.weight",
                        target_key=(
                            f"{ph}*.self_attn.q_proj.lora_A.default.weight",
                            f"{ph}*.self_attn.k_proj.lora_A.default.weight",
                            f"{ph}*.self_attn.v_proj.lora_A.default.weight",
                        ),
                        fn=TransformFns.duplicate3,
                    ),
                    io.state_transform(
                        source_key=f"{pn}*.self_attention.linear_qkv.adapter.linear_out.weight",
                        target_key=(
                            f"{ph}*.self_attn.q_proj.lora_B.default.weight",
                            f"{ph}*.self_attn.k_proj.lora_B.default.weight",
                            f"{ph}*.self_attn.v_proj.lora_B.default.weight",
                        ),
                        fn=TransformFns.split_qkv,
                    ),
                ]
            )

        return io.apply_transforms(
            source,
            target,
            mapping=mapping,
            transforms=transforms,
        )

    @property
    def peft_config(self) -> "PeftConfig":
        """Create a PEFT config for the HF model.

        Translates the NeMo PEFT configuration to the equivalent HF PEFT
        configuration.

        Returns:
            PeftConfig: HF PEFT configuration
        """
        from peft import LoraConfig

        from nemo.collections.llm.peft import DoRA

        assert (
            not self.peft_obj.dropout or self.peft_obj.dropout_position == 'pre'
        ), "LoRA dropout_position must be 'pre' to convert to HF."

        NEMO2HF = {
            'linear_q': ['q_proj'],
            'linear_k': ['k_proj'],
            'linear_v': ['v_proj'],
            'linear_qkv': ['q_proj', 'k_proj', 'v_proj'],
            'linear_proj': ['o_proj'],
            'linear_fc1': ['gate_up_proj'],  # unlike llama, gpt-oss has gate up proj as one weight
            'linear_fc2': ['down_proj'],
        }

        # Infer HF target modules from NeMo target modules
        hf_target_modules = []
        for tm in self.peft_obj.target_modules:
            if tm in ('linear_fc1', 'linear_fc2'):
                raise ValueError(
                    f"target module {tm} is not supported in LoRA export because the"
                    f"NeMo implementation is not interchangeable with the HF "
                    f"implementation."
                )
            else:
                hf_target_modules.extend(NEMO2HF[tm])

        return LoraConfig(
            r=self.peft_obj.dim,
            target_modules=hf_target_modules,
            lora_alpha=self.peft_obj.alpha,
            lora_dropout=self.peft_obj.dropout,
            use_dora=isinstance(self.peft_obj, DoRA),
        )


__all__ = [
    "GPTOSSConfig",
    "GPTOSSConfig120B",
    "GPTOSSConfig20B",
    "GPTOSSModel",
]
