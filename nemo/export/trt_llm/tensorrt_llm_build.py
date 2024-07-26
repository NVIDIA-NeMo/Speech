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


import logging
import tensorrt_llm
from tensorrt_llm._common import check_max_num_tokens
from tensorrt_llm.builder import BuildConfig, Builder
from tensorrt_llm.commands.build import build as build_trtllm
from tensorrt_llm.logger import logger
from tensorrt_llm.lora_manager import LoraConfig
from tensorrt_llm.models.modeling_utils import add_lora, optimize_model, preprocess_weights
from tensorrt_llm.plugin import PluginConfig

MODEL_NAME = "NeMo"

LOGGER = logging.getLogger("NeMo")


def build_and_save_engine(
    max_input_len=1024,
    max_output_len=1024,
    max_batch_size=4,
    model_dir=None,
    model_weights=None,
    model_config=None,
    model_type='gpt',
    lora_ckpt_list=None,
    use_lora_plugin=None,
    max_lora_rank=64,
    lora_target_modules=None,
    max_prompt_embedding_table_size=0,
    enable_multi_block_mode: bool = False,
    paged_kv_cache: bool = True,
    remove_input_padding: bool = True,
    paged_context_fmha: bool = False,
    use_custom_all_reduce: bool = True,
    use_refit: bool = False,
    max_num_tokens: int = None,
    max_seq_len: int = None,
    opt_num_tokens: int = None,
    max_beam_width: int = 1,
    tokens_per_block: int = 128,
    multiple_profiles: bool = False,
    gpt_attention_plugin: str = "auto",
    gemm_plugin: str = "auto",
):
    architecture = "LLaMAForCausalLM" if model_config.architecture == "LlamaForCausalLM" else model_config.architecture
    try:
        model_cls = getattr(tensorrt_llm.models, architecture)
    except:
        raise AttributeError(f"Could not find TRTLLM model type: {model_type}!")

    logger.set_level("info")
    plugin_config = PluginConfig()
    plugin_config.gpt_attention_plugin = gpt_attention_plugin
    plugin_config.gemm_plugin = gemm_plugin
    plugin_config.set_nccl_plugin(use_custom_all_reduce=use_custom_all_reduce)
    plugin_config.multi_block_mode = enable_multi_block_mode
    if paged_kv_cache:
        plugin_config.enable_paged_kv_cache(tokens_per_block=tokens_per_block)
    else:
        plugin_config.paged_kv_cache = False
    plugin_config.remove_input_padding = remove_input_padding
    plugin_config.use_paged_context_fmha = paged_context_fmha
    plugin_config.multiple_profiles = multiple_profiles

    if max_seq_len is None:
        max_seq_len = max_input_len + max_output_len

    max_num_tokens, opt_num_tokens = check_max_num_tokens(
        max_num_tokens=max_num_tokens,
        opt_num_tokens=opt_num_tokens,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        max_input_len=max_input_len,
        max_beam_width=max_beam_width,
        remove_input_padding=remove_input_padding,
        enable_context_fmha=plugin_config.context_fmha,
        tokens_per_block=tokens_per_block,
        multiple_profiles=multiple_profiles,
    )

    build_dict = {
        'max_input_len': max_input_len,
        'max_output_len': max_output_len,
        'max_batch_size': max_batch_size,
        'max_beam_width': max_beam_width,
        'max_seq_len': max_seq_len,
        'max_num_tokens': max_num_tokens,
        'opt_num_tokens': opt_num_tokens,
        'max_prompt_embedding_table_size': max_prompt_embedding_table_size,
        'gather_context_logits': False,
        'gather_generation_logits': False,
        'strongly_typed': False,
        'builder_opt': None,
        'use_refit': use_refit,
        'multiple_profiles': multiple_profiles,
    }
    build_config = BuildConfig.from_dict(build_dict, plugin_config=plugin_config)

    if use_lora_plugin is not None:
        # build_config.plugin_config.set_lora_plugin(use_lora_plugin)
        # build_config.plugin_config._lora_plugin = use_lora_plugin
        lora_config = LoraConfig(
            lora_dir=lora_ckpt_list,
            lora_ckpt_source='nemo',
            max_lora_rank=max_lora_rank,
            lora_target_modules=lora_target_modules,
        )
        build_config.lora_config = lora_config

    model = model_cls.from_config(model_config)
    model = optimize_model(
        model,
        use_parallel_embedding=model_config.use_parallel_embedding,
        share_embedding_table=model_config.share_embedding_table,
    )
    preprocess_weights(model_weights, model_config)
    model.load(model_weights)
    engine = build_trtllm(model, build_config)
    engine.save(model_dir)

    return engine
