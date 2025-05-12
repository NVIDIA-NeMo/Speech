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
import re
from dataclasses import dataclass, field
from functools import cached_property, partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import yaml
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.transformer_config import MLATransformerConfig
from safetensors.torch import load_file
from torch import nn
from transformers import AutoConfig

from nemo.collections.llm.gpt.model.base import (
    HAVE_TE,
    GPTConfig,
    GPTModel,
    gpt_data_step,
    torch_dtype_from_dict_config,
)
from nemo.export.trt_llm.nemo_ckpt_loader.nemo_file import load_distributed_model_weights
from nemo.lightning import io, teardown
from nemo.lightning.io.state import TransformFns, _ModelState
from nemo.lightning.pytorch.optim import OptimizerModule
from nemo.lightning.pytorch.utils import dtype_from_hf
from nemo.utils import logging

if TYPE_CHECKING:
    from megatron.core.transformer import ModuleSpec
    from transformers import AutoModelForCausalLM

    from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
    from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec

if HAVE_TE:
    from megatron.core.utils import is_te_min_version


@dataclass
class DeepSeekConfig(MLATransformerConfig, GPTConfig):
    """
    Base config for DeepSeek V2 and V3 models.
    """

    transformer_layer_spec: Union['ModuleSpec', Callable[["GPTConfig"], 'ModuleSpec']] = partial(
        get_gpt_decoder_block_spec, use_transformer_engine=HAVE_TE
    )

    # Model
    normalization: str = "RMSNorm"
    activation_func: Callable = F.silu
    gated_linear_unit: bool = True  # swiglu
    position_embedding_type: str = "rope"
    add_bias_linear: bool = False
    share_embeddings_and_output_weights: bool = False
    num_attention_heads: int = 128
    kv_channels: int = 128
    max_position_embeddings: int = 4096
    seq_length: int = 4096
    rotary_base: float = 10000.0
    make_vocab_size_divisible_by: int = 3200
    mtp_num_layers: Optional[int] = None
    mtp_loss_scaling_factor: Optional[float] = None

    # Regularization
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    qk_layernorm: bool = True

    # MoE
    moe_grouped_gemm: bool = True
    moe_router_pre_softmax: bool = True
    moe_token_dispatcher_type: str = "alltoall"
    moe_router_load_balancing_type: str = 'seq_aux_loss'
    moe_shared_expert_overlap: bool = True
    moe_router_dtype: Optional[str] = 'fp32'

    # MLA
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_head_dim: int = 128
    qk_pos_emb_head_dim: int = 64
    v_head_dim: int = 128
    rotary_scaling_factor: float = 40
    mscale: float = 1.0
    mscale_all_dim: float = 1.0

    # Miscellaneous
    init_method_std: float = 0.006
    layernorm_epsilon: float = 1e-6
    bf16: bool = True
    params_dtype: torch.dtype = torch.bfloat16
    async_tensor_model_parallel_allreduce: bool = True
    attention_softmax_in_fp32: bool = False
    persist_layer_norm: bool = True
    num_layers_in_first_pipeline_stage: Optional[int] = None
    num_layers_in_last_pipeline_stage: Optional[int] = None
    account_for_embedding_in_pipeline_split: bool = False
    account_for_loss_in_pipeline_split: bool = False

    # fusions
    apply_rope_fusion: bool = False
    bias_activation_fusion: bool = True
    bias_dropout_fusion: bool = True
    masked_softmax_fusion: bool = True
    gradient_accumulation_fusion: bool = True
    cross_entropy_loss_fusion: bool = True
    cross_entropy_fusion_impl: str = "te"
    moe_permute_fusion: bool = is_te_min_version("2.1.0") if HAVE_TE else False

    def __post_init__(self):
        super().__post_init__()
        if self.mtp_num_layers is not None:
            self.data_step_fn = partial(gpt_data_step, use_mtp=True)


@dataclass
class DeepSeekV2Config(DeepSeekConfig):
    """
    DeepSeek-V2 Model: https://github.com/deepseek-ai/DeepSeek-V2
    """

    num_layers: int = 60
    hidden_size: int = 5120
    ffn_hidden_size: int = 12288
    num_moe_experts: int = 160
    moe_ffn_hidden_size: int = 1536
    moe_shared_expert_intermediate_size: int = 3072  # 1536 * 2 shared experts
    moe_layer_freq: Union[int, List[int]] = field(default_factory=lambda: [0] + [1] * 59)  # first layer is dense
    moe_router_topk: int = 6
    moe_router_num_groups: int = 8
    moe_router_group_topk: int = 3
    moe_router_topk_scaling_factor: float = 16.0
    moe_aux_loss_coeff: float = 1e-3
    mscale: float = 0.707
    mscale_all_dim: float = 0.707


@dataclass
class DeepSeekV2LiteConfig(DeepSeekV2Config):
    """
    DeepSeek-V2-Lite Model: https://github.com/deepseek-ai/DeepSeek-V2
    HuggingFace: https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite
    """

    num_layers: int = 27
    hidden_size: int = 2048
    ffn_hidden_size: int = 10944
    num_attention_heads: int = 16
    kv_channels: int = 16
    q_lora_rank: int = None
    num_moe_experts: int = 64
    moe_ffn_hidden_size: int = 1408
    moe_shared_expert_intermediate_size: int = 2816  # 1408 * 2 shared experts
    moe_layer_freq: Union[int, List[int]] = field(default_factory=lambda: [0] + [1] * 26)  # first layer is dense
    moe_router_topk: int = 6
    moe_router_num_groups: int = 1
    moe_router_group_topk: int = 1
    moe_router_topk_scaling_factor: float = 1.0


@dataclass
class DeepSeekV3Config(DeepSeekConfig):
    """
    DeepSeek-V3 Model: https://github.com/deepseek-ai/DeepSeek-V3
    """

    num_layers: int = 61
    hidden_size: int = 7168
    ffn_hidden_size: int = 18432
    num_moe_experts: int = 256
    moe_ffn_hidden_size: int = 2048
    moe_shared_expert_intermediate_size: int = 2048  # 2048 * 1 shared expert
    moe_layer_freq: Union[int, List[int]] = field(
        default_factory=lambda: [0] * 3 + [1] * 58
    )  # first three layers are dense
    moe_router_topk: int = 8
    moe_router_num_groups: int = 8
    moe_router_group_topk: int = 4
    moe_router_topk_scaling_factor: float = 2.5
    moe_aux_loss_coeff: float = 1e-4
    make_vocab_size_divisible_by: int = 1280
    moe_router_score_function: str = "sigmoid"
    moe_router_enable_expert_bias: bool = True
    moe_router_bias_update_rate: float = 1e-3
    mscale: float = 1.0
    mscale_all_dim: float = 1.0


class DeepSeekModel(GPTModel):
    # pylint: disable=C0115,C0116
    def __init__(
        self,
        config: Optional[DeepSeekConfig] = None,
        optim: Optional[OptimizerModule] = None,
        tokenizer: Optional["TokenizerSpec"] = None,
        model_transform: Optional[Callable[[nn.Module], nn.Module]] = None,
    ):
        super().__init__(
            config or DeepSeekV2Config(), optim=optim, tokenizer=tokenizer, model_transform=model_transform
        )


@io.model_importer(DeepSeekModel, ext="hf")
class HFDeepSeekImporter(io.ModelConnector["AutoModelForCausalLM", DeepSeekModel]):
    # pylint: disable=C0115,C0116
    def init(self) -> DeepSeekModel:
        return DeepSeekModel(self.config, tokenizer=self.tokenizer)

    def apply(self, output_path: Path, convert_mtp: bool = False) -> Path:
        from transformers import AutoModelForCausalLM

        self.convert_mtp = convert_mtp
        self._verify_source()
        source = AutoModelForCausalLM.from_pretrained(str(self), trust_remote_code=True, torch_dtype='auto')
        target = self.init()
        trainer = self.nemo_setup(target)
        self.convert_state(source, target)
        self.nemo_save(output_path, trainer)

        logging.info(f"Converted DeepSeek model to Nemo, model saved to {output_path}")

        teardown(trainer, target)
        del trainer, target

        return output_path

    def _verify_source(self):
        source_config = AutoConfig.from_pretrained(str(self), trust_remote_code=True)
        assert 'quantization_config' not in source_config, (
            "HuggingFace cannot load DeepSeek V3's FP8 checkpoint directly. You must convert the checkpoint "
            "to BF16. See NeMo documentation for more details: "
            "https://nemo-framework-tme.gitlab-master-pages.nvidia.com/documentation/user-guide/latest/llms/"
            "deepseek_v3.html#nemo-2-0-finetuning-recipes "
        )

    def _modify_source_state(self, source: nn.Module) -> _ModelState:
        """
        In deepseek, HF weight `model.layers.*.post_attention_layernorm.weight` is mapped to mcore weight
        a) `decoder.layers.*.mlp.linear_fc1.layer_norm_weight`, if the layer is dense
        b) `decoder.layers.*.pre_mlp_layernorm.weight`, if the layer is MoE

        We rename model.layers.*.post_attention_layernorm.weight in the first case to prevent a one-to-many mapping
        """

        state_dict = source.state_dict()

        for layer_i, use_moe in enumerate(self.config.moe_layer_freq):
            if use_moe == 0:
                weight = state_dict.pop(f"model.layers.{layer_i}.post_attention_layernorm.weight")
                state_dict[f"model.layers.{layer_i}.dense-post_attention_layernorm.weight"] = weight

        source = _ModelState(state_dict)
        return source

    def _add_mtp_to_source(self, source: nn.Module | _ModelState) -> None:
        # Load MTP weights from disk, since it is not in HF model
        mtp_hf_layer_low = self.config.num_layers  # 61 if DeepSeek V3
        mtp_hf_layer_high = self.config.num_layers + self.config.mtp_num_layers - 1  # 61 if DeepSeek V3
        # Identify which file to load
        with open(self / "model.safetensors.index.json", 'r') as file:
            manifest = json.load(file)

        safetensor_files_to_load = set()
        mtp_hf_keys = set()
        for k, fname in manifest['weight_map'].items():
            if match := re.match(r".*\.layers\.(\d+)\.", k):
                if mtp_hf_layer_low <= int(match.group(1)) <= mtp_hf_layer_high:
                    safetensor_files_to_load.add(fname)
                    mtp_hf_keys.add(k)

        mtp_state_dict = {}
        for safetensor_file in safetensor_files_to_load:
            for k, v in load_file(self / safetensor_file).items():
                if k in mtp_hf_keys:
                    # ensure HF keys "mtp" are alphabetically after "layers",
                    # since mcore keys "mtp" are after "decoder"
                    # This allows us to reuse the mapping and transforms for MTP
                    mtp_state_dict[k.replace(".layers.", ".mtp.")] = v
        source.state_dict().update(mtp_state_dict)

    def convert_state(self, source, target):
        # pylint: disable=C0301
        mapping = {
            # Embed
            "model.embed_tokens.weight": "embedding.word_embeddings.weight",
            # Attention
            "**.input_layernorm.weight": "**.input_layernorm.weight",
            "**.self_attn.o_proj.weight": "**.self_attention.linear_proj.weight",
            "**.self_attn.q_a_proj.weight": "**.self_attention.linear_q_down_proj.weight",
            "**.self_attn.q_b_proj.weight": "**.self_attention.linear_q_up_proj.weight",
            "**.self_attn.kv_a_proj_with_mqa.weight": "**.self_attention.linear_kv_down_proj.weight",
            "**.self_attn.kv_b_proj.weight": "**.self_attention.linear_kv_up_proj.weight",
            "**.self_attn.q_a_layernorm.weight": "**.self_attention.linear_q_up_proj.layer_norm_weight",
            "**.self_attn.kv_a_layernorm.weight": "**.self_attention.linear_kv_up_proj.layer_norm_weight",
            "**.dense-post_attention_layernorm.weight": "**.mlp.linear_fc1.layer_norm_weight",
            "**.post_attention_layernorm.weight": "**.pre_mlp_layernorm.weight",
            # Dense MLP
            # **.mlp.{gate|up}_proj.weight: **.mlp.linear_fc1.weight
            "**.mlp.down_proj.weight": "**.mlp.linear_fc2.weight",
            # MoE
            "**.mlp.gate.weight": "**.mlp.router.weight",
            # **.mlp.experts.*.{gate|up}_proj.weight: **.mlp.experts.linear_fc1.weight*
            "**.mlp.experts.*.down_proj.weight": "**.mlp.experts.linear_fc2.weight*",
            # **.mlp.shared_experts.{gate|up}_proj.weight： **.mlp.shared_experts.linear_fc1.weight
            "**.mlp.shared_experts.down_proj.weight": "**.mlp.shared_experts.linear_fc2.weight",
            # LM Head
            "model.norm.weight": "decoder.final_layernorm.weight",
            "lm_head.weight": "output_layer.weight",
        }
        # For lite model
        if self.config.q_lora_rank is None:
            del mapping["**.self_attn.q_a_proj.weight"]
            del mapping["**.self_attn.q_b_proj.weight"]
            mapping["**.self_attn.q_proj.weight"] = "**.self_attention.linear_q_proj.weight"
        # Account for Mcore local spec
        if self.config.q_lora_rank is not None and not isinstance(
            target.module.decoder.layers[0].self_attention.q_layernorm, IdentityOp
        ):
            mapping["**.self_attn.q_a_layernorm.weight"] = "**.self_attention.q_layernorm.weight"

        if not isinstance(target.module.decoder.layers[0].self_attention.kv_layernorm, IdentityOp):
            mapping["**.self_attn.kv_a_layernorm.weight"] = "**.self_attention.kv_layernorm.weight"

        if not isinstance(target.module.decoder.layers[0].pre_mlp_layernorm, IdentityOp):
            del mapping["**.dense-post_attention_layernorm.weight"]
            source = _ModelState(source.state_dict)
        else:
            source = self._modify_source_state(source)

        if hasattr(self.config, "moe_router_enable_expert_bias") and self.config.moe_router_enable_expert_bias:
            mapping.update(
                {
                    "**.mlp.gate.e_score_correction_bias": "**.mlp.router.expert_bias",
                }
            )

        transforms = [
            io.state_transform(
                source_key=("**.mlp.gate_proj.weight", "**.mlp.up_proj.weight"),
                target_key="**.mlp.linear_fc1.weight",
                fn=TransformFns.merge_fc1,
            ),
            io.state_transform(
                source_key=(
                    "**.mlp.experts.*.gate_proj.weight",
                    "**.mlp.experts.*.up_proj.weight",
                ),
                target_key="**.mlp.experts.linear_fc1.weight*",
                fn=TransformFns.merge_fc1,
            ),
            io.state_transform(
                source_key=(
                    "**.mlp.shared_experts.gate_proj.weight",
                    "**.mlp.shared_experts.up_proj.weight",
                ),
                target_key="**.mlp.shared_experts.linear_fc1.weight",
                fn=TransformFns.merge_fc1,
            ),
        ]

        # Convert MTP weights
        if getattr(self.config, "mtp_num_layers", None) and self.convert_mtp:
            self._add_mtp_to_source(source)
            mapping.update(
                {
                    'model.mtp.*.eh_proj.weight': "mtp.layers.*.eh_proj.weight",
                    'model.mtp.*.enorm.weight': "mtp.layers.*.enorm.weight",
                    'model.mtp.*.hnorm.weight': "mtp.layers.*.hnorm.weight",
                    'model.mtp.*.shared_head.norm.weight': "mtp.layers.*.shared_head_norm.weight",
                }
            )

        return io.apply_transforms(
            source,
            target,
            mapping=mapping,
            transforms=transforms,
        )

    @cached_property
    def tokenizer(self) -> "AutoTokenizer":
        from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer

        return AutoTokenizer(self.save_hf_tokenizer_assets(str(self)), use_fast=True)

    @cached_property
    def config(self) -> DeepSeekConfig:
        from transformers import AutoConfig as HFAutoConfig
        from transformers import GenerationConfig

        source = HFAutoConfig.from_pretrained(str(self), trust_remote_code=True)
        try:
            generation_config = GenerationConfig.from_pretrained(str(self))
        except OSError:
            generation_config = None

        n_moe_layers = source.num_hidden_layers - source.first_k_dense_replace
        is_v3 = source.scoring_func == "sigmoid"
        if is_v3:
            v3_kwargs = {
                "moe_router_score_function": "sigmoid",
                "moe_router_enable_expert_bias": True,
                "mtp_num_layers": source.num_nextn_predict_layers if self.convert_mtp else None,
            }
        else:
            v3_kwargs = {}
        return DeepSeekConfig(
            num_layers=source.num_hidden_layers,
            hidden_size=source.hidden_size,
            ffn_hidden_size=source.intermediate_size,
            num_attention_heads=source.num_attention_heads,
            kv_channels=source.num_key_value_heads,
            q_lora_rank=source.q_lora_rank,
            num_moe_experts=source.n_routed_experts,
            moe_ffn_hidden_size=source.moe_intermediate_size,
            moe_shared_expert_intermediate_size=source.moe_intermediate_size * source.n_shared_experts,
            moe_layer_freq=[0] * source.first_k_dense_replace + [1] * n_moe_layers,
            moe_router_topk=source.num_experts_per_tok,
            moe_router_num_groups=source.n_group,
            moe_router_group_topk=source.topk_group,
            moe_router_topk_scaling_factor=source.routed_scaling_factor,
            moe_aux_loss_coeff=getattr(source, "aux_loss_alpha", 0.001),
            kv_lora_rank=source.kv_lora_rank,
            qk_head_dim=source.qk_nope_head_dim,
            qk_pos_emb_head_dim=source.qk_rope_head_dim,
            v_head_dim=source.v_head_dim,
            make_vocab_size_divisible_by=1280 if is_v3 else 3200,
            fp16=(dtype_from_hf(source) == torch.float16),
            bf16=(dtype_from_hf(source) == torch.bfloat16),
            params_dtype=dtype_from_hf(source),
            generation_config=generation_config,
            **v3_kwargs,
        )


@io.model_exporter(DeepSeekModel, "hf")
class HFDeepSeekExporter(io.ModelConnector[DeepSeekModel, "AutoModelForCausalLM"]):
    # pylint: disable=C0115,C0116
    def init(self, dtype=torch.bfloat16, model_name="deepseek-ai/DeepSeek-V3") -> "AutoModelForCausalLM":
        from transformers import AutoConfig, AutoModelForCausalLM
        from transformers.modeling_utils import no_init_weights

        with no_init_weights():
            # Since DeepSeek is not importable from transformers, we can only initialize the HF model
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

    def ckpt_load(self, path: Path) -> Tuple[Dict, Dict]:
        """
        This function loads the state dict directly from a distributed checkpoint, and modify the state dict
        so that it is consistent with the key names you would get from loading the checkpoint into a model.
        This is a more memory-efficient method to obtain a state dict without initializing the nemo model.

        Args:
            path (Path): The path from which the model will be loaded.

        Returns
        -------
            Tuple[Dict, Dict]: The loaded state dict and the yaml config dict.
        """
        model_yaml = path / "context" / "model.yaml"
        if not model_yaml.exists():
            raise FileNotFoundError("model.yaml is not found in the context folder of the checkpoint.")
        with open(model_yaml, 'r') as stream:
            config = yaml.safe_load(stream)

        dist_ckpt_folder = path / "weights"
        state_dict = {}
        for k, v in load_distributed_model_weights(dist_ckpt_folder, True).items():
            if '_extra_state' in k:
                continue
            new_k = k.replace("module.", "")
            if '.experts.experts.' in k:
                # split experts into multiple tensors
                for i in range(v.size(0)):
                    state_dict[new_k.replace(".experts.experts.", ".experts.") + str(i)] = v[i]
            else:
                state_dict[new_k] = v
        return state_dict, config['config']

    def apply(self, output_path: Path, target_model_name=None) -> Path:
        logging.info("Loading DeepSeek NeMo checkpoint. This may take a while...")
        source, source_config = self.ckpt_load(self)
        logging.info("DeepSeek NeMo checkpoint loaded.")
        if target_model_name is None:
            # Before DeepSeek is fully supported by HF, it is necessary to pass in a local HF checkpoint that
            # is used to initialize the HF model. The following
            logging.warning(
                "Before DeepSeek is officially supported in HF, you should pass in a local HF "
                "checkpoint using llm.export_ckpt(..., target_model_name=<local hf path>)"
            )
            if source_config['moe_router_enable_expert_bias']:
                target_model_name = "deepseek-ai/DeepSeek-V3"
            elif source_config['q_lora_rank'] is not None:
                target_model_name = "deepseek-ai/DeepSeek-V2"
            else:
                target_model_name = "deepseek-ai/DeepSeek-V2-Lite"
            logging.info(
                f"Your model is determined to be {target_model_name} based on the config. If this is not correct, "
                f"please pass in a local HF checkpoint."
            )

        target = self.init(torch_dtype_from_dict_config(source_config), model_name=target_model_name)
        target = self.convert_state(source, target, source_config)

        target = target.cpu()
        logging.info(f"Converted DeepSeek model to HF, saving model to {output_path}...")
        target.save_pretrained(output_path, safe_serialization=False)
        self.tokenizer.save_pretrained(output_path)

        return output_path

    def convert_state(self, source, target, source_config):
        # pylint: disable=C0301
        mapping = {
            # Embed
            "embedding.word_embeddings.weight": "model.embed_tokens.weight",
            # Attention
            "decoder.layers.*.input_layernorm.weight": "model.layers.*.input_layernorm.weight",
            "decoder.layers.*.self_attention.linear_proj.weight": "model.layers.*.self_attn.o_proj.weight",
            "decoder.layers.*.self_attention.linear_q_down_proj.weight": "model.layers.*.self_attn.q_a_proj.weight",
            "decoder.layers.*.self_attention.linear_q_up_proj.weight": "model.layers.*.self_attn.q_b_proj.weight",
            "decoder.layers.*.self_attention.linear_kv_down_proj.weight": "model.layers.*.self_attn.kv_a_proj_with_mqa.weight",
            "decoder.layers.*.self_attention.linear_kv_up_proj.weight": "model.layers.*.self_attn.kv_b_proj.weight",
            "decoder.layers.*.self_attention.linear_q_up_proj.layer_norm_weight": "model.layers.*.self_attn.q_a_layernorm.weight",
            "decoder.layers.*.self_attention.linear_kv_up_proj.layer_norm_weight": "model.layers.*.self_attn.kv_a_layernorm.weight",
            "decoder.layers.*.pre_mlp_layernorm.weight": "model.layers.*.post_attention_layernorm.weight",
            # Dense MLP
            # decoder.layers.*.mlp.linear_fc1.weight: model.layers.*.mlp.{gate|up}_proj.weight
            "decoder.layers.*.mlp.linear_fc2.weight": "model.layers.*.mlp.down_proj.weight",
            # MoE
            "decoder.layers.*.mlp.router.weight": "model.layers.*.mlp.gate.weight",
            # decoder.layers.*.mlp.experts.linear_fc1.weight*: model.layers.*.mlp.experts.*.{gate|up}_proj.weight
            "decoder.layers.*.mlp.experts.linear_fc2.weight*": "model.layers.*.mlp.experts.*.down_proj.weight",
            # decoder.layers.*.mlp.shared_experts.linear_fc1.weight: model.layers.*.mlp.shared_experts.{gate|up}_proj.weight
            "decoder.layers.*.mlp.shared_experts.linear_fc2.weight": "model.layers.*.mlp.shared_experts.down_proj.weight",
            # LM Head
            "decoder.final_layernorm.weight": "model.norm.weight",
            "output_layer.weight": "lm_head.weight",
        }
        # For lite model
        if source_config['q_lora_rank'] is None:
            del mapping["decoder.layers.*.self_attention.linear_q_down_proj.weight"]
            del mapping["decoder.layers.*.self_attention.linear_q_up_proj.weight"]
            mapping["decoder.layers.*.self_attention.linear_q_proj.weight"] = "model.layers.*.self_attn.q_proj.weight"
        # Account for Mcore local spec
        if source_config['q_lora_rank'] is not None and 'decoder.layers.0.self_attention.q_layernorm.weight' in source:
            mapping["decoder.layers.*.self_attention.q_layernorm.weight"] = mapping.pop(
                "decoder.layers.*.self_attention.linear_q_up_proj.layer_norm_weight"
            )

        if 'decoder.layers.0.self_attention.kv_layernorm.weight' in source:
            mapping["decoder.layers.*.self_attention.kv_layernorm.weight"] = mapping.pop(
                "decoder.layers.*.self_attention.linear_kv_up_proj.layer_norm_weight"
            )

        if source_config.get('moe_router_enable_expert_bias', False):
            mapping.update(
                {
                    "decoder.layers.*.mlp.router.expert_bias": "model.layers.*.mlp.gate.e_score_correction_bias",
                }
            )

        transforms = [
            io.state_transform(
                source_key="decoder.layers.*.mlp.linear_fc1.weight",
                target_key=("model.layers.*.mlp.gate_proj.weight", "model.layers.*.mlp.up_proj.weight"),
                fn=TransformFns.split_fc1,
            ),
            io.state_transform(
                source_key="decoder.layers.*.mlp.experts.linear_fc1.weight*",
                target_key=(
                    "model.layers.*.mlp.experts.*.gate_proj.weight",
                    "model.layers.*.mlp.experts.*.up_proj.weight",
                ),
                fn=TransformFns.split_fc1,
            ),
            io.state_transform(
                source_key="decoder.layers.*.mlp.shared_experts.linear_fc1.weight",
                target_key=(
                    "model.layers.*.mlp.shared_experts.gate_proj.weight",
                    "model.layers.*.mlp.shared_experts.up_proj.weight",
                ),
                fn=TransformFns.split_fc1,
            ),
        ]
        source = self._modify_source_state(source, source_config)

        return io.apply_transforms(
            source,
            target,
            mapping=mapping,
            transforms=transforms,
        )

    def _modify_source_state(self, source: Dict[str, Any], source_config: Dict[str, Any]) -> _ModelState:
        """
        In deepseek, HF weight `model.layers.*.post_attention_layernorm.weight` is mapped to mcore weight
        a) `decoder.layers.*.mlp.linear_fc1.layer_norm_weight`, if the layer is dense
        b) `decoder.layers.*.pre_mlp_layernorm.weight`, if the layer is MoE

        We rename decoder.layers.*.mlp.linear_fc1.layer_norm_weight in the first case to unify key names
        """
        for layer_i in range(source_config['num_layers']):
            if f"decoder.layers.{layer_i}.mlp.linear_fc1.layer_norm_weight" in source:
                weight = source.pop(f"decoder.layers.{layer_i}.mlp.linear_fc1.layer_norm_weight")
                source[f"decoder.layers.{layer_i}.pre_mlp_layernorm.weight"] = weight
        modified_source = _ModelState(source)
        return modified_source

    @property
    def tokenizer(self) -> 'AutoTokenizer':
        return io.load_context(self, subpath="model").tokenizer


__all__ = [
    "DeepSeekConfig",
    "DeepSeekV2Config",
    "DeepSeekV2LiteConfig",
    "DeepSeekV3Config",
    "DeepSeekModel",
]
