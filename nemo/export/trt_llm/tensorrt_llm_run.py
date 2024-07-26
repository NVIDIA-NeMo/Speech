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


import csv
import json
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import tensorrt_llm
import torch
from mpi4py.futures import MPIPoolExecutor
from tensorrt_llm.lora_manager import LoraManager
from tensorrt_llm.quantization import QuantMode
from tensorrt_llm.runtime import ModelConfig, ModelRunner, ModelRunnerCpp, SamplingConfig

from transformers import PreTrainedTokenizer

LOGGER = logging.getLogger("NeMo")

use_trtllm_bindings = True
try:
    from tensorrt_llm.bindings import GptJsonConfig, GptSession, GptSessionConfig, KvCacheConfig, WorldConfig
except Exception as e:
    use_trtllm_bindings = False

use_cpp_gpt_session = True
try:
    from tensorrt_llm.runtime.model_runner_cpp import ModelRunnerCppGptSession
except Exception as e:
    use_cpp_gpt_session = False


@dataclass
class TensorrtLLMHostContext:
    """The host side context for TRT LLM inference."""

    executor: MPIPoolExecutor = None
    world_size: int = 1
    tokenizer: PreTrainedTokenizer = None
    max_batch_size: int = 0
    max_input_len: int = 0
    add_bos: bool = False


@dataclass
class TensorrtLLMWorkerContext:
    """The MPI worker side context for TRT LLM inference."""

    decoder: ModelRunner = None
    sampling_config: SamplingConfig = None
    lora_manager: LoraManager = None
    max_batch_size: int = 0
    max_input_len: int = 0


# This is a global context that will be initialized during the model loading process as MPI worker.
tensorrt_llm_worker_context = TensorrtLLMWorkerContext()


def _read_config(config_path: Path):
    with open(config_path, "r") as f:
        config = json.load(f)

    tensor_parallel_size = config["builder_config"]["tensor_parallel"]
    pipeline_parallel_size = config["builder_config"]["pipeline_parallel"]
    world_size = tensor_parallel_size * pipeline_parallel_size

    assert world_size <= torch.cuda.device_count(), f"Not enough GPUs, requesting {world_size}"

    num_heads = config["builder_config"]["num_heads"]
    num_kv_heads = config["builder_config"].get("num_kv_heads", num_heads)
    head_size = config["builder_config"]["head_size"]
    hidden_size = config["builder_config"]["hidden_size"] // tensor_parallel_size

    num_heads = num_heads // tensor_parallel_size
    num_kv_heads = (num_kv_heads + tensor_parallel_size - 1) // tensor_parallel_size

    if "tokens_per_block" in config["plugin_config"]:
        tokens_per_block = config["plugin_config"]["tokens_per_block"]
    else:
        tokens_per_block = config["builder_config"]["tokens_per_block"]

    if quantization := config["builder_config"].get("quantization"):
        # Field "quantization" (dict) is introduced for quantized Nemo checkpoints support.
        # For regular Nemo checkpoints "quant_mode" field should be used (default: 0).
        quant_mode = QuantMode.from_quant_algo(quantization['quant_algo'], quantization['kv_cache_quant_algo'])
    else:
        quant_mode = QuantMode(config["builder_config"]["quant_mode"])

    model_config = ModelConfig(
        model_name=config["builder_config"]["name"],
        max_batch_size=config["builder_config"]["max_batch_size"],
        max_beam_width=config["builder_config"]["max_beam_width"],
        vocab_size=config["builder_config"]["vocab_size"],
        num_layers=config["builder_config"]["num_layers"],
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        hidden_size=hidden_size,
        head_size=head_size,
        gpt_attention_plugin=config["plugin_config"]["gpt_attention_plugin"],
        remove_input_padding=config["plugin_config"]["remove_input_padding"],
        paged_kv_cache=config["plugin_config"]["paged_kv_cache"],
        tokens_per_block=tokens_per_block,
        max_prompt_embedding_table_size=config["builder_config"]["max_prompt_embedding_table_size"],
        dtype=config["builder_config"]["precision"],
        lora_plugin=config["plugin_config"]["lora_plugin"],
        lora_target_modules=config["builder_config"]["lora_target_modules"],
        quant_mode=quant_mode,
        use_custom_all_reduce=config["plugin_config"]["use_custom_all_reduce"],
        use_context_fmha_for_generation=config["plugin_config"]["use_context_fmha_for_generation"],
        gather_context_logits=config["builder_config"]["gather_context_logits"],
        gather_generation_logits=config["builder_config"]["gather_generation_logits"],
    )

    dtype = config["builder_config"]["precision"]
    max_input_len = config["builder_config"]["max_input_len"]
    max_batch_size = config["builder_config"]["max_batch_size"]

    return model_config, world_size, tensor_parallel_size, pipeline_parallel_size, dtype, max_input_len, max_batch_size


def _load(
    tokenizer: PreTrainedTokenizer,
    engine_dir,
    lora_ckpt_list=None,
    num_beams=1,
    use_python_runtime: bool = True,
    enable_chunked_context: bool = False,
    max_tokens_in_paged_kv_cache: int = None,
):
    """The impl of `load` API for on a single GPU worker."""
    try:
        tensorrt_llm.logger.set_level("info")

        engine_dir = Path(engine_dir)
        config_path = engine_dir / "config.json"
        # model_config, world_size, tp_size, pp_size, dtype, max_input_len, max_batch_size = _read_config(config_path)

        with open(config_path, "r") as f:
            config = json.load(f)

        max_batch_size = config["build_config"]["max_batch_size"]
        max_input_len = config["build_config"]["max_input_len"]
        # max_output_len = config["build_config"]["max_output_len"]
        max_beam_width = config["build_config"]["max_beam_width"]

        runtime_rank = tensorrt_llm.mpi_rank()

        if use_python_runtime:
            decoder = ModelRunner.from_dir(
                engine_dir=engine_dir,
                lora_dir=lora_ckpt_list,
                lora_ckpt_source="nemo",
                rank=runtime_rank,
                debug_mode=False,
            )
        else:
            decoder = ModelRunnerCpp.from_dir(
                engine_dir=engine_dir,
                lora_dir=lora_ckpt_list,
                lora_ckpt_source="nemo",
                rank=runtime_rank,
                max_batch_size=max_batch_size,
                max_input_len=max_input_len,
                # max_output_len=max_output_len,
                max_beam_width=max_beam_width,
                enable_chunked_context=enable_chunked_context,
                max_tokens_in_paged_kv_cache=max_tokens_in_paged_kv_cache,
                debug_mode=False,
            )

        sampling_config = SamplingConfig(
            end_id=tokenizer.eos_token_id, pad_id=tokenizer.eos_token_id, num_beams=num_beams
        )

        # Initialize the global context so it can be used during `run` API.
        global tensorrt_llm_worker_context
        tensorrt_llm_worker_context.decoder = decoder
        tensorrt_llm_worker_context.sampling_config = sampling_config
        tensorrt_llm_worker_context.max_batch_size = max_batch_size
        tensorrt_llm_worker_context.max_input_len = max_input_len

    except Exception as e:
        print(e)
        raise e


def _forward(
    input_tensors: List[torch.IntTensor],
    max_output_len: int,
    top_k: int = 1,
    top_p: float = 0.0,
    temperature: float = 1.0,
    prompt_table=None,
    task_vocab_size=None,
    task_ids: List[int] = None,
    lora_uids: List[str] = None,
    stop_words_list=None,
    bad_words_list=None,
    no_repeat_ngram_size=None,
    streaming: bool = False,
    multiprocessed_env=False,
    **sampling_kwargs,
) -> Optional[torch.IntTensor]:
    """The impl of `forward` API for on a single GPU worker with tensor as IO.

    Returns:
        the output tokens tensor with shape [batch_size, num_beams, output_len].
    """
    try:
        # Loading the global context initialized from the `load` API.
        global tensorrt_llm_worker_context
        decoder = tensorrt_llm_worker_context.decoder
        assert decoder is not None, "Invalid worker context, decoder is not loaded."
        sampling_config = tensorrt_llm_worker_context.sampling_config
        max_batch_size = tensorrt_llm_worker_context.max_batch_size
        max_input_len = tensorrt_llm_worker_context.max_input_len

        batch_size = len(input_tensors)
        assert batch_size <= max_batch_size, f"batch size {batch_size} exceedng max batch size {max_batch_size}"
        input_lengths = [t.shape[0] for t in input_tensors]
        max_length = max(input_lengths)
        assert max_length <= max_input_len, f"input length {max_length} exceedng max input length {max_input_len}"
        pad_id = sampling_config.pad_id
        end_id = sampling_config.end_id
        num_beams = sampling_config.num_beams

        with torch.no_grad():
            prompt_tasks = None if task_ids is None else ",".join(str(task) for task in task_ids)

            if prompt_table is not None:
                prompt_table = prompt_table.reshape(1, *prompt_table.shape)
                tmp_dir = tempfile.TemporaryDirectory()
                prompt_table_path = os.path.join(tmp_dir.name, 'prompt_table.npy')
                np.save(prompt_table_path, prompt_table.cpu().float().numpy())
                prompt_table = prompt_table_path

            outputs = decoder.generate(
                input_tensors,
                max_new_tokens=max_output_len,
                end_id=end_id,
                pad_id=pad_id,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                num_beams=num_beams,
                stop_words_list=stop_words_list,
                bad_words_list=bad_words_list,
                lora_uids=lora_uids,
                prompt_table_path=prompt_table,
                prompt_table=prompt_table,
                prompt_tasks=prompt_tasks,
                streaming=streaming,
                output_sequence_lengths=True,
                return_dict=True,
            )

            torch.cuda.synchronize()

            if prompt_table is not None:
                tmp_dir.cleanup()

        runtime_rank = tensorrt_llm.mpi_rank()
        if runtime_rank == 0 or multiprocessed_env:
            return outputs
        else:
            return None

    except Exception as e:
        print(e)
        raise e


def load(
    tokenizer: PreTrainedTokenizer,
    engine_dir: str,
    lora_ckpt_list: List[str] = None,
    num_beams: int = 1,
    use_python_runtime: bool = True,
    enable_chunked_context: bool = False,
    max_tokens_in_paged_kv_cache: int = None,
) -> TensorrtLLMHostContext:
    """Loaded the compiled LLM model and run it.

    It also supports running the TRT LLM model on multi-GPU.
    """
    # the parent dir of the engine_dir
    config_path = os.path.join(engine_dir, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    world_size = config["pretrained_config"]["mapping"]["world_size"]
    if world_size == 1:
        _load(
            tokenizer,
            engine_dir,
            lora_ckpt_list,
            num_beams,
            use_python_runtime,
            enable_chunked_context,
            max_tokens_in_paged_kv_cache,
        )
        executor = None
    elif tensorrt_llm.mpi_world_size() > 1:
        _load(
            tokenizer,
            engine_dir,
            lora_ckpt_list,
            num_beams,
            use_python_runtime,
            enable_chunked_context,
            max_tokens_in_paged_kv_cache,
        )
        executor = None
        tensorrt_llm.mpi_barrier()
    else:
        executor = MPIPoolExecutor(max_workers=world_size)
        futures = []
        for _ in range(world_size):
            future = executor.submit(
                _load,
                tokenizer,
                engine_dir,
                lora_ckpt_list,
                num_beams,
                use_python_runtime,
                enable_chunked_context,
                max_tokens_in_paged_kv_cache,
            )
            futures.append(future)
        for future in futures:
            future.result()

    max_batch_size = config["build_config"]["max_batch_size"]
    max_input_len = config["build_config"]["max_input_len"]
    architectures_that_need_bos_token = [
        "GemmaForCausalLM",
        "LLaMAForCausalLM",
        "MistralForCausalLM",
        "MixtralForCausalLM",
    ]
    add_bos = config["pretrained_config"]["architecture"] in architectures_that_need_bos_token

    return TensorrtLLMHostContext(
        executor=executor,
        world_size=world_size,
        tokenizer=tokenizer,
        max_batch_size=max_batch_size,
        max_input_len=max_input_len,
        add_bos=add_bos,
    )


def forward(
    input_tensors: List[torch.IntTensor],
    max_output_len: int,
    host_context: TensorrtLLMHostContext,
    top_k: int = 1,
    top_p: float = 0.0,
    temperature: float = 1.0,
    prompt_table=None,
    task_vocab_size=None,
    task_ids: List[int] = None,
    lora_uids: List[str] = None,
    stop_words_list=None,
    bad_words_list=None,
    no_repeat_ngram_size=None,
    streaming: bool = False,
    multiprocessed_env=False,
    **sampling_kwargs,
) -> Optional[torch.IntTensor]:
    """Run the loaded model with the host_context provided from the `load` API."""
    batch_size = len(input_tensors)
    max_batch_size = host_context.max_batch_size
    assert batch_size <= max_batch_size, f"batch size {batch_size} exceedng max batch size {max_batch_size}"
    max_length = max([t.shape[0] for t in input_tensors])
    max_input_len = host_context.max_input_len
    assert max_length <= max_input_len, f"input length {max_length} exceedng max input length {max_input_len}"

    world_size = host_context.world_size
    if world_size == 1 or multiprocessed_env:
        return _forward(
            input_tensors=input_tensors,
            max_output_len=max_output_len,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            prompt_table=prompt_table,
            task_vocab_size=task_vocab_size,
            task_ids=task_ids,
            lora_uids=lora_uids,
            stop_words_list=stop_words_list,
            bad_words_list=bad_words_list,
            no_repeat_ngram_size=no_repeat_ngram_size,
            streaming=streaming,
            multiprocessed_env=multiprocessed_env,
            **sampling_kwargs,
        )
    else:
        executor = host_context.executor
        futures = []
        for _ in range(world_size):
            future = executor.submit(
                _forward,
                input_tensors=input_tensors,
                max_output_len=max_output_len,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                prompt_table=prompt_table,
                task_vocab_size=task_vocab_size,
                task_ids=task_ids,
                lora_uids=lora_uids,
                stop_words_list=stop_words_list,
                bad_words_list=bad_words_list,
                no_repeat_ngram_size=no_repeat_ngram_size,
                streaming=streaming,
                **sampling_kwargs,
            )
            futures.append(future)
        for future in futures:
            result = future.result()
            if result is not None:
                return result

        raise RuntimeError("Internal error")


def load_distributed(engine_dir, model_parallel_rank, gpus_per_node):
    """Loads TRTLLM engines in a distributed gpu environment, in particular
    this function creates a custom mapping of device_id to WorldConfig
    """
    global tensorrt_llm_worker_context
    if isinstance(tensorrt_llm_worker_context.decoder, ModelRunnerCppGptSession):
        return

    config_path = Path(engine_dir) / f"config_{torch.distributed.get_rank()}.json"
    json_config = GptJsonConfig.parse_file(config_path)
    model_config = json_config.model_config

    max_beam_width = model_config.max_beam_width
    max_batch_size = model_config.max_batch_size
    max_input_len = model_config.max_input_len
    max_seq_len = model_config.max_seq_len

    tp_size = json_config.tensor_parallelism
    pp_size = json_config.pipeline_parallelism
    assert tp_size <= gpus_per_node, "Multinode TP is not unsupported"

    # TRTLLM asserts that rank equals the device num however this
    # is not true for the megatron mapping of TP->DP->PP.
    # So we manipulate TRTLLM to emulate a TP->PP single node setup
    # TRTLLM is expected to fix this in future releases
    offset = (torch.cuda.current_device() - model_parallel_rank % gpus_per_node + gpus_per_node) % gpus_per_node
    device_ids = [i for i in range(gpus_per_node)]
    for _ in range(offset):
        device_ids.append(device_ids.pop(0))
    world_config = WorldConfig.mpi(
        gpus_per_node=gpus_per_node, tensor_parallelism=tp_size, pipeline_parallelism=pp_size, device_ids=device_ids
    )
    engine_filename = json_config.engine_filename(world_config)
    serialize_path = Path(engine_dir) / engine_filename
    assert torch.cuda.current_device() == world_config.device

    session_config = GptSessionConfig(
        max_batch_size=max_batch_size, max_beam_width=max_beam_width, max_sequence_length=max_seq_len
    )
    session_config.gen_micro_batch_size = max_batch_size
    session_config.ctx_micro_batch_size = max_batch_size
    session_config.kv_cache_config = KvCacheConfig(
        max_tokens=max_seq_len * max_batch_size, max_attention_window=max_seq_len
    )

    with open(serialize_path, "rb") as f:
        engine_data = bytearray(f.read())

    session = GptSession(session_config, model_config, world_config, engine_data)
    decoder = ModelRunnerCppGptSession(
        session,
        lora_manager=None,
        max_batch_size=max_batch_size,
        max_input_len=max_input_len,
        max_seq_len=max_seq_len,
        max_beam_width=max_beam_width,
    )

    tensorrt_llm_worker_context.decoder = decoder
    tensorrt_llm_worker_context.max_batch_size = max_batch_size
    tensorrt_llm_worker_context.max_input_len = max_input_len
    # Save the model config in case for refit
    tensorrt_llm_worker_context.model_config = model_config


def refit(weights_dict):
    global tensorrt_llm_worker_context
    dtype = tensorrt_llm_worker_context.model_config.data_type
    tensorrt_llm_worker_context.decoder.session.refit_engine(weights_dict, dtype)


def prepare_input_tensors(
    input_texts: List[str],
    host_context: TensorrtLLMHostContext,
    prompt_table=None,
    task_vtoken_counts: List[int] = None,
    task_ids: List[int] = None,
):
    tokenizer = host_context.tokenizer

    if host_context.add_bos:
        bos_tokens = [tokenizer.bos_token_id]
    else:
        bos_tokens = []

    input_tokens = [bos_tokens + tokenizer.encode(t) for t in input_texts]

    # If p-tuning is used, we need to prepend vtokens to each input.
    if prompt_table is not None:

        # Go over the tokenized prompts and prepend vtokens.
        # The number of vtokens could be different for each task.
        for prompt_index in range(len(input_texts)):
            # Find out the number of vtokens to generate
            task_id = task_ids[prompt_index]
            num_vtokens = task_vtoken_counts[task_id]

            # Create a tensor with vtokens, e.g. 32000, 32001, 32002... when vocab_size=32000
            # TRT-LLM will convert each vtoken into its corresponding embedding row from the prompt table.
            vocab_size = tokenizer.vocab_size
            vtokens = list(range(vocab_size, vocab_size + num_vtokens))

            # Concatenate the vtokens with the real tokens
            real_tokens = input_tokens[prompt_index]
            input_tokens[prompt_index] = vtokens + real_tokens

    # Convert input token lists to tensors
    input_tensors = [torch.IntTensor(token_list) for token_list in input_tokens]

    return input_tensors


def generate(
    input_texts: List[str],
    max_output_len: int,
    host_context: TensorrtLLMHostContext,
    top_k: int = 1,
    top_p: float = 0.0,
    temperature: float = 1.0,
    prompt_table=None,
    task_vocab_size=None,
    task_vtoken_counts: List[int] = None,
    task_ids: List[int] = None,
    lora_uids: List[str] = None,
    stop_words_list=None,
    bad_words_list=None,
    no_repeat_ngram_size=None,
    streaming: bool = False,
    output_log_probs=False,
    multiprocessed_env=False,
    **sampling_kwargs,
) -> Optional[List[List[str]]]:
    """Generate the output sequence from the input sequence.

    Returns a 2D string list with shape [batch_size, num_beams].
    """
    tokenizer = host_context.tokenizer
    input_tensors = prepare_input_tensors(input_texts, host_context, prompt_table, task_vtoken_counts, task_ids)

    stop_words_list_tensors = None
    if stop_words_list is not None:
        stop_words_arrays = to_word_list_format(stop_words_list, tokenizer)
        stop_words_list_tensors = (
            torch.Tensor(stop_words_arrays).to(torch.int32).to(torch.cuda.current_device()).contiguous()
        )

    bad_words_list_tensors = None
    if bad_words_list is not None:
        bad_words_arrays = to_word_list_format(bad_words_list, tokenizer)
        bad_words_list_tensors = (
            torch.Tensor(bad_words_arrays).to(torch.int32).to(torch.cuda.current_device()).contiguous()
        )

    if no_repeat_ngram_size is not None:
        no_repeat_ngram_size = torch.IntTensor(no_repeat_ngram_size).to(torch.cuda.current_device())

    outputs = forward(
        input_tensors=input_tensors,
        max_output_len=max_output_len,
        host_context=host_context,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        prompt_table=prompt_table,
        task_vocab_size=task_vocab_size,
        task_ids=task_ids,
        lora_uids=lora_uids,
        stop_words_list=stop_words_list_tensors,
        bad_words_list=bad_words_list_tensors,
        no_repeat_ngram_size=no_repeat_ngram_size,
        streaming=False,
        output_log_probs=output_log_probs,
        multiprocessed_env=multiprocessed_env,
        **sampling_kwargs,
    )
    assert outputs is not None
    if tensorrt_llm.mpi_rank() != 0:
        return None

    output_ids = outputs['output_ids']
    sequence_lengths = outputs['sequence_lengths']
    input_lengths = [t.shape[0] for t in input_tensors]

    output_lines_list = [
        tokenizer.batch_decode(output_ids[b, :, input_lengths[b] : sequence_lengths[b][0]])
        for b in range(output_ids.shape[0])
    ]

    if output_log_probs:
        return output_lines_list, log_probs
    return output_lines_list


def generate_streaming(
    input_texts: List[str],
    max_output_len: int,
    host_context: TensorrtLLMHostContext,
    top_k: int = 1,
    top_p: float = 0.0,
    temperature: float = 1.0,
    prompt_table=None,
    task_vocab_size=None,
    task_vtoken_counts: List[int] = None,
    task_ids: List[int] = None,
    lora_uids: List[str] = None,
    stop_words_list=None,
    bad_words_list=None,
    no_repeat_ngram_size=None,
    **sampling_kwargs,
) -> Optional[List[List[str]]]:
    """Generate the output sequence from the input sequence.

    Returns a 2D string list with shape [batch_size, num_beams].
    """
    tokenizer = host_context.tokenizer
    input_tensors = prepare_input_tensors(input_texts, host_context, prompt_table, task_vtoken_counts, task_ids)

    batch_size = len(input_texts)

    stop_words_list_tensors = None
    if stop_words_list is not None:
        stop_words_list_tensors = [tokenizer.encode(t) for t in stop_words_list]
        stop_words_list_tensors = torch.IntTensor(stop_words_list_tensors)
        stop_words_list_tensors = (
            stop_words_list_tensors.unsqueeze(0).repeat(batch_size, 1, 1).to(torch.cuda.current_device())
        )

    bad_words_list_tensors = None
    if bad_words_list is not None:
        bad_words_list_tensors = [tokenizer.encode(t) for t in bad_words_list]
        bad_words_list_tensors = torch.IntTensor(bad_words_list_tensors)
        bad_words_list_tensors = (
            bad_words_list_tensors.unsqueeze(0).repeat(batch_size, 1, 1).to(torch.cuda.current_device())
        )

    if no_repeat_ngram_size is not None:
        no_repeat_ngram_size = torch.IntTensor(no_repeat_ngram_size).to(torch.cuda.current_device())

    outputs = forward(
        input_tensors=input_tensors,
        max_output_len=max_output_len,
        host_context=host_context,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        prompt_table=prompt_table,
        task_vocab_size=task_vocab_size,
        task_ids=task_ids,
        lora_uids=lora_uids,
        stop_words_list=stop_words_list_tensors,
        bad_words_list=bad_words_list_tensors,
        no_repeat_ngram_size=no_repeat_ngram_size,
        streaming=True,
        **sampling_kwargs,
    )
    assert outputs is not None

    input_lengths = [t.shape[0] for t in input_tensors]

    # 'outputs' is a generator that yields one generator, not sure why... Unwrap that.
    for output in outputs:
        output_ids = output['output_ids']
        # Now iterate over the partial outputs, decode and yield each intermediate result.
        generated_tokens = 0
        for partial_outputs in output_ids:
            if partial_outputs is None:
                break
            # partial_outputs is a tensor with shape=(len(input_texts), 1, output_length),
            # where the last dimension contains a progressively increasing number of valid, generated tokens.
            assert partial_outputs.shape[0] == len(input_texts)
            outputs = []
            generated_tokens += 1

            # For each input in the batch...
            for input_index in range(len(input_texts)):
                # Extract the generated part of the output tensor and decode it.
                input_length = input_lengths[input_index]
                decoded_output = tokenizer.batch_decode(
                    partial_outputs[input_index, :, input_length : input_length + generated_tokens]
                )[0]
                outputs.append(decoded_output)

            # Yield the list of decoded partial responses.
            yield outputs
        # See above - 'outputs' yields just one item.
        break


def unload(host_context: TensorrtLLMHostContext):
    """Frees the GPU resource from the TensorrtLLMHostContext and reset the host_context."""
    if host_context.executor is not None:
        host_context.executor.shutdown(wait=True)
        host_context.executor = None
        return

    global tensorrt_llm_worker_context
    tensorrt_llm_worker_context.decoder = None
    tensorrt_llm_worker_context = TensorrtLLMWorkerContext()


def to_word_list_format(
    word_dict: List[List[str]],
    tokenizer=None,
    ref_str="<extra_id_1>",
):
    '''
    format of word_dict
        len(word_dict) should be same to batch_size
        word_dict[i] means the words for batch i
        len(word_dict[i]) must be 1, which means it only contains 1 string
        This string can contains several sentences and split by ",".
        For example, if word_dict[2] = " I am happy, I am sad", then this function will return
        the ids for two short sentences " I am happy" and " I am sad".
    '''
    assert tokenizer is not None, "need to set tokenizer"

    flat_ids = []
    offsets = []
    # The encoding of a single word can't always be trusted. See
    #   https://github.com/NVIDIA/NeMo/blob/bb575b72fd0be51ae10cc77d9f89ddb9e9d3b96d/nemo/collections/nlp/modules/common/text_generation_strategy.py#L229
    ids_ref = tokenizer.encode(ref_str)
    for word_dict_item in word_dict:
        item_flat_ids = []
        item_offsets = []

        if isinstance(word_dict_item[0], bytes):
            word_dict_item = [word_dict_item[0].decode()]

        words = list(csv.reader(word_dict_item))[0]
        for word in words:
            ids = tokenizer.encode(f"{ref_str}{word}")
            if ids[0 : len(ids_ref)] == ids_ref:
                # It worked! We can obtain the token(s) associated to `word` by stripping the prefix tokens.
                ids = ids[len(ids_ref) :]
            else:
                # Unfortunately the prefix was merged with `word`. We could try with a different prefix, but
                # for now we just use the basic encoding since this should be a very rare edge case.
                ids = tokenizer.encode(word)
                logging.warning(f"The encoding of word '{word}' into tokens {ids} might be incorrect")

            if len(ids) == 0:
                continue

            item_flat_ids += ids
            item_offsets.append(len(ids))

        flat_ids.append(np.array(item_flat_ids))
        offsets.append(np.cumsum(np.array(item_offsets)))

    pad_to = max(1, max(len(ids) for ids in flat_ids))

    for i, (ids, offs) in enumerate(zip(flat_ids, offsets)):
        flat_ids[i] = np.pad(ids, (0, pad_to - len(ids)), constant_values=0)
        offsets[i] = np.pad(offs, (0, pad_to - len(offs)), constant_values=-1)

    return np.array([flat_ids, offsets], dtype="int32").transpose((1, 0, 2))
