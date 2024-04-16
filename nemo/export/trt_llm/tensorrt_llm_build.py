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


import argparse
import logging
import os
import time
from pathlib import Path
from typing import List

import tensorrt as trt
import tensorrt_llm
import torch
from tensorrt_llm import str_dtype_to_trt
from tensorrt_llm._utils import np_dtype_to_trt
from tensorrt_llm.builder import Builder
from tensorrt_llm.logger import logger
from tensorrt_llm.models.modeling_utils import add_lora
from tensorrt_llm.network import net_guard
from tensorrt_llm.plugin.plugin import ContextFMHAType
from tensorrt_llm.quantization import QuantMode

MODEL_NAME = "NeMo"

LOGGER = logging.getLogger("NeMo")


def get_engine_name(model, dtype, tp_size, pp_size, rank):
    """Returns the engine file name based on the provided info."""
    if pp_size == 1:
        return '{}_{}_tp{}_rank{}.engine'.format(model, dtype, tp_size, rank)
    return '{}_{}_tp{}_pp{}_rank{}.engine'.format(model, dtype, tp_size, pp_size, rank)


def serialize_engine(engine, path):
    """Serializes the engine to path."""
    logger.info(f"Serializing engine to {path}...")
    tik = time.time()
    with open(path, "wb") as f:
        f.write(bytearray(engine))
    tok = time.time()
    t = time.strftime("%H:%M:%S", time.gmtime(tok - tik))
    logger.info(f"Engine serialized. Total time: {t}")


def refit_runtime_engine(params, cuda_engine):
    '''
        @brief: Inplace refit one TensorRT cuda engine using weights from the network,
            user should guarantee that the engine is built with REFIT flag, and the network has the same structure with the engine.
        @param engine_buffer: A serialized TensorRT engine.
        @param network: Network object.
        @return: A serialized TRT engine if refit successfully, None otherwise
    '''
    logger.info(f'Refit runtime engine')
    tik = time.time()

    # Refit engine
    assert params is not None
    refitter = trt.Refitter(cuda_engine, logger.trt_logger)
    for name, param in params:
        trt_param = trt.Weights(np_dtype_to_trt(param._value.dtype), param._value.ctypes.data, param._value.size)

        if trt_param is None or not refitter.set_named_weights(name, trt_param):
            logger.error(f'Failed to refit weight: {name}')
            return None

    if not refitter.refit_cuda_engine():
        logger.error(f'Failed to refit engine.')
        return None

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    logger.info(f'Total time of refitting {cuda_engine.name}: {t}')

    return cuda_engine


def build_rank_engine(
    tensorrt_llm_gpt, builder: Builder, builder_config: tensorrt_llm.builder.BuilderConfig, engine_name, args,
):

    str_dtype_to_trt(args.dtype)
    ootb = os.getenv("OOTB", False)

    network = builder.create_network()
    network.trt_network.name = engine_name

    # We have to use the attention plugin for most of the models.
    if args.use_gpt_attention_plugin:
        network.plugin_config.set_gpt_attention_plugin(dtype=args.use_gpt_attention_plugin)

    if not ootb:
        network.plugin_config.use_custom_all_reduce = False

        if args.use_gemm_plugin:
            network.plugin_config.set_gemm_plugin(dtype=args.use_gemm_plugin)
        if args.use_layernorm_plugin:
            network.plugin_config.set_layernorm_plugin(dtype=args.use_layernorm_plugin)
        assert not (args.enable_context_fmha and args.enable_context_fmha_fp32_acc)
        if args.enable_context_fmha:
            network.plugin_config.set_context_fmha(ContextFMHAType.enabled)
        if args.enable_context_fmha_fp32_acc:
            network.plugin_config.set_context_fmha(ContextFMHAType.enabled_with_fp32_acc)
        if args.remove_input_padding:
            network.plugin_config.enable_remove_input_padding()
        else:
            network.plugin_config.remove_input_padding = False
        if args.paged_kv_cache:
            network.plugin_config.enable_paged_kv_cache()
        else:
            network.plugin_config.paged_kv_cache = False
        if args.use_ib_gpt_attention_plugin:
            network.plugin_config.set_inflight_batching_gpt_attention_plugin(dtype=args.use_ib_gpt_attention_plugin)
        if args.enable_multi_block_mode:
            network.plugin_config.enable_mmha_multi_block_mode()

        if args.use_lora_plugin:
            network.plugin_config.set_lora_plugin(dtype=args.use_lora_plugin)

        if args.use_lookup_plugin:
            # Use the plugin for the embedding parallelism and sharing
            network.plugin_config.set_lookup_plugin(dtype=args.dtype)
    else:
        LOGGER.warning("Build engine in OOTB mode, disable all plugins except nccl.")

    if args.mapping.world_size > 1:
        network.plugin_config.set_nccl_plugin(args.dtype)

    with net_guard(network):
        # Prepare
        network.set_named_parameters(tensorrt_llm_gpt.named_parameters())

        # Forward
        inputs = tensorrt_llm_gpt.prepare_inputs(
            max_batch_size=args.max_batch_size,
            max_input_len=args.max_input_len,
            max_new_tokens=args.max_input_len + args.max_output_len,
            use_cache=True,
            max_beam_width=args.max_beam_width,
            paged_kv_cache=args.paged_kv_cache,
            tokens_per_block=args.tokens_per_block,
            prompt_embedding_table_size=args.max_prompt_embedding_table_size,
            lora_target_modules=args.lora_target_modules,
        )
        tensorrt_llm_gpt(*inputs)

    # Network -> Engine
    engine = builder.build_engine(network, builder_config)
    if args.mapping.rank == 0 or args.use_refit:
        config_path = args.output_dir / "config.json"
        builder.save_config(builder_config, config_path)
    return engine


def _build_impl(tensorrt_llm_model, args):
    torch.cuda.set_device(args.mapping.rank % args.gpus_per_node)
    tensorrt_llm.logger.set_level(args.log_level)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timing_cache_file = args.timing_cache if args.timing_cache else args.output_dir / "model.cache"
    timing_cache = timing_cache_file

    if args.use_lora_plugin is not None:
        add_lora(tensorrt_llm_model, args.max_lora_rank)

    builder = Builder()
    apply_query_key_layer_scaling = False

    builder_config = builder.create_builder_config(
        name=MODEL_NAME,
        precision=args.dtype,
        timing_cache=timing_cache,
        tensor_parallel=args.mapping.tp_size,
        pipeline_parallel=args.mapping.pp_size,
        world_size=args.mapping.tp_size * args.mapping.pp_size,
        parallel_build=args.parallel_build,
        num_layers=tensorrt_llm_model._num_layers,
        num_heads=tensorrt_llm_model._num_heads,
        num_kv_heads=tensorrt_llm_model._num_kv_heads,
        head_size=tensorrt_llm_model._head_size,
        hidden_size=tensorrt_llm_model._hidden_size,
        vocab_size=tensorrt_llm_model._vocab_size,
        hidden_act=tensorrt_llm_model.hidden_act,
        max_position_embeddings=tensorrt_llm_model.max_position_embeddings,
        add_bos=tensorrt_llm_model._add_bos,
        apply_query_key_layer_scaling=apply_query_key_layer_scaling,
        max_batch_size=args.max_batch_size,
        max_input_len=args.max_input_len,
        max_output_len=args.max_output_len,
        max_beam_width=args.max_beam_width,
        max_num_tokens=None,
        max_draft_len=0,
        int8="int8" in args.quantization,
        opt_level=args.builder_opt,
        paged_kv_cache=args.paged_kv_cache,
        tokens_per_block=args.tokens_per_block,
        max_prompt_embedding_table_size=args.max_prompt_embedding_table_size,
        use_parallel_embedding=args.use_parallel_embedding,
        embedding_sharding_dim=args.embedding_sharding_dim,
        fp8="fp8" in args.quantization,
        use_refit=args.use_refit,
        gather_context_logits=False,
        gather_generation_logits=False,
        quant_mode=args.quant_mode,
        lora_target_modules=args.lora_target_modules,
        max_lora_rank=args.max_lora_rank,
    )

    tp_size = args.mapping.tp_size
    pp_size = args.mapping.pp_size
    rank = args.mapping.rank
    engine_name = get_engine_name(MODEL_NAME, args.dtype, tp_size, pp_size, rank)
    engine = build_rank_engine(tensorrt_llm_model, builder, builder_config, engine_name, args)
    assert engine is not None, f"Failed to build engine for rank {rank}"

    serialize_engine(engine, args.output_dir / engine_name)

    if args.mapping.rank == 0:
        ok = builder.save_timing_cache(builder_config, timing_cache_file)
        assert ok, "Failed to save timing cache."


def build(
    tensorrt_llm_model,
    output_dir: Path,
    mapping=None,
    dtype="float16",
    timing_cache="",
    log_level="info",
    max_batch_size=1,
    max_input_len=200,
    max_output_len=200,
    max_beam_width=1,
    max_prompt_embedding_table_size=0,
    parallel_build=False,
    gpus_per_node=1,
    quantization=None,
    use_inflight_batching=False,
    paged_kv_cache=False,
    enable_context_fmha: bool = True,
    enable_multi_block_mode=False,
    use_refit=False,
    use_lora_plugin: str = None,
    lora_target_modules: List[str] = None,
    max_lora_rank: int = 64,
):
    """Builds the tensorrt_llm_model to engine."""
    args = argparse.Namespace()
    args.mapping = mapping
    args.dtype = dtype
    args.timing_cache = timing_cache
    args.log_level = log_level
    args.max_batch_size = max_batch_size
    args.max_input_len = max_input_len
    args.max_output_len = max_output_len
    args.max_beam_width = max_beam_width
    args.use_gpt_attention_plugin = dtype
    args.use_gemm_plugin = dtype
    args.use_layernorm_plugin = False
    args.parallel_build = parallel_build
    args.enable_context_fmha = enable_context_fmha
    args.enable_context_fmha_fp32_acc = False
    args.gpus_per_node = gpus_per_node
    args.builder_opt = None
    args.output_dir = Path(output_dir)
    args.remove_input_padding = True
    args.use_smooth_quant = False
    args.use_weight_only = False
    args.weight_only_precision = "int8"
    args.per_channel = False
    args.per_token = False
    args.int8_kv_cache = False
    args.random_seed = None
    args.paged_kv_cache = paged_kv_cache
    args.max_prompt_embedding_table_size = max_prompt_embedding_table_size
    args.use_inflight_batching = use_inflight_batching
    args.use_ib_gpt_attention_plugin = False
    args.use_parallel_embedding = False
    args.embedding_sharding_dim = 0
    args.use_lookup_plugin = False
    args.tokens_per_block = 64
    args.quantization = quantization
    args.enable_multi_block_mode = enable_multi_block_mode
    args.use_refit = use_refit
    args.use_lora_plugin = use_lora_plugin
    args.lora_target_modules = lora_target_modules
    args.max_lora_rank = max_lora_rank

    logger.set_level(args.log_level)

    assert not (
        args.use_smooth_quant and args.use_weight_only
    ), "You cannot enable both SmoothQuant and INT8 weight-only together."

    assert not (
        args.use_smooth_quant and args.use_weight_only
    ), "You cannot enable both SmoothQuant and INT8 weight-only together."

    if args.use_ib_gpt_attention_plugin:
        logger.warning(
            "use_ib_gpt_attention_plugin is deprecated. Use combination of"
            " --use_gpt_attention_plugin=dtype --use_inflight_batching instead."
        )

    if args.use_inflight_batching:
        assert args.use_gpt_attention_plugin, "You have to use GPT attention plugin for in-flight batching mode"

        if not args.paged_kv_cache:
            logger.warning("Paged kv cache feature will enabled for in-flight batching mode.")
            args.paged_kv_cache = True

        if not args.remove_input_padding:
            logger.warning("Remove input padding feature will enabled for in-flight batching mode.")
            args.remove_input_padding = True

    if args.use_smooth_quant:
        args.quant_mode = QuantMode.use_smooth_quant(args.per_token, args.per_channel)
    elif args.use_weight_only:
        args.quant_mode = QuantMode.use_weight_only(args.weight_only_precision == "int4")
    else:
        args.quant_mode = QuantMode(0)

    if args.int8_kv_cache:
        args.quant_mode = args.quant_mode.set_int8_kv_cache()

    if args.random_seed is not None:
        torch.manual_seed(args.random_seed)

    if args.mapping.is_first_pp_rank():
        if tensorrt_llm_model._modules['vocab_embedding'].tp_size > 1:
            args.use_parallel_embedding = True
            args.embedding_sharding_dim = tensorrt_llm_model._modules['vocab_embedding'].sharding_dim

    tik = time.time()
    _build_impl(tensorrt_llm_model, args)

    tok = time.time()
    t = time.strftime("%H:%M:%S", time.gmtime(tok - tik))
    logger.info(f"Total time of building all {args.mapping.world_size} engines: {t}")
