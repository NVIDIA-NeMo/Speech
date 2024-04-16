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


import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import tensorrt_llm
import torch
from mpi4py.futures import MPIPoolExecutor
from tensorrt_llm.logger import logger
from tensorrt_llm.lora_manager import LoraManager
from tensorrt_llm.quantization import QuantMode
from tensorrt_llm.runtime import ModelConfig, SamplingConfig
from transformers import PreTrainedTokenizer

from nemo.export.trt_llm.tensor_utils import get_tensor_parallel_group
from nemo.export.trt_llm.tensorrt_llm_model import LMHeadModelBuilder

from nemo.export.trt_llm.tensorrt_llm_build import get_engine_name, MODEL_NAME, refit_runtime_engine  # isort:skip
from nemo.export.trt_llm.nemo_utils import to_word_list_format  # isort:skip


LOGGER = logging.getLogger("NeMo")


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

    decoder: tensorrt_llm.runtime.GenerationSession = None
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
        quant_mode=QuantMode(config["builder_config"]["quant_mode"]),
        use_custom_all_reduce=config["plugin_config"]["use_custom_all_reduce"],
        use_context_fmha_for_generation=config["plugin_config"]["use_context_fmha_for_generation"],
        gather_context_logits=config["builder_config"]["gather_context_logits"],
        gather_generation_logits=config["builder_config"]["gather_generation_logits"],
    )

    dtype = config["builder_config"]["precision"]
    max_input_len = config["builder_config"]["max_input_len"]
    max_batch_size = config["builder_config"]["max_batch_size"]

    return model_config, world_size, tensor_parallel_size, pipeline_parallel_size, dtype, max_input_len, max_batch_size


def _load(tokenizer: PreTrainedTokenizer, engine_dir, lora_ckpt_list=None, num_beams=1):
    """The impl of `load` API for on a single GPU worker."""
    try:
        tensorrt_llm.logger.set_level("info")

        engine_dir = Path(engine_dir)
        config_path = engine_dir / "config.json"
        model_config, world_size, tp_size, pp_size, dtype, max_input_len, max_batch_size = _read_config(config_path)

        runtime_rank = tensorrt_llm.mpi_rank()

        assert runtime_rank < torch.cuda.device_count(), f"Rank {runtime_rank} out of bound"
        runtime_mapping = tensorrt_llm.Mapping(world_size, runtime_rank, tp_size=tp_size, pp_size=pp_size)

        torch.cuda.set_device(runtime_rank % runtime_mapping.gpus_per_node)
        engine_name = get_engine_name(MODEL_NAME, dtype, tp_size, pp_size, runtime_rank)
        serialize_path = os.path.join(engine_dir, engine_name)
        logger.info(f"Reading from serialize path {serialize_path}")

        with open(serialize_path, "rb") as f:
            engine_buffer = f.read()
        decoder = tensorrt_llm.runtime.GenerationSession(
            model_config, engine_buffer, runtime_mapping, debug_mode=False
        )

        sampling_config = SamplingConfig(
            end_id=tokenizer.eos_token_id, pad_id=tokenizer.eos_token_id, num_beams=num_beams
        )

        if decoder.use_lora_plugin:
            lora_manager = LoraManager()
            if lora_ckpt_list is not None:
                lora_manager.load_from_nemo(
                    model_files=lora_ckpt_list, model_config=model_config, runtime_mapping=runtime_mapping,
                )
        else:
            lora_manager = None

        # Initialize the global context so it can be used during `run` API.
        global tensorrt_llm_worker_context
        tensorrt_llm_worker_context.decoder = decoder
        tensorrt_llm_worker_context.sampling_config = sampling_config
        tensorrt_llm_worker_context.lora_manager = lora_manager
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
        lora_manager = tensorrt_llm_worker_context.lora_manager
        max_batch_size = tensorrt_llm_worker_context.max_batch_size
        max_input_len = tensorrt_llm_worker_context.max_input_len

        batch_size = len(input_tensors)
        assert batch_size <= max_batch_size, f"batch size {batch_size} exceedng max batch size {max_batch_size}"
        input_lengths = [t.shape[0] for t in input_tensors]
        max_length = max(input_lengths)
        assert max_length <= max_input_len, f"input length {max_length} exceedng max input length {max_input_len}"
        pad_id = sampling_config.pad_id

        if decoder.remove_input_padding:
            line_encoded = torch.concat(input_tensors).cuda()
        else:
            line_encoded = torch.nested.to_padded_tensor(
                torch.nested.nested_tensor(input_tensors, dtype=torch.int32), pad_id
            ).cuda()

        input_lengths = torch.tensor(input_lengths, dtype=torch.int32).cuda()

        if prompt_table is None:
            ptuning_args = []
        else:
            if task_vocab_size is None:
                raise Exception("task_vocab_size cannot be None")

            task_vocab_size = torch.tensor([task_vocab_size], dtype=torch.int32, device="cuda")
            task_ids = torch.tensor(task_ids, dtype=torch.int32, device="cuda")
            prompt_table = prompt_table.cuda()
            ptuning_args = [prompt_table, task_ids, task_vocab_size]

        with torch.no_grad():
            sampling_config.top_k = top_k
            sampling_config.top_p = top_p
            sampling_config.temperature = temperature
            for key, param in sampling_kwargs.items():
                # set any additional SamplingConfig kwargs
                setattr(sampling_config, key, param)

            decoder.setup(
                batch_size,
                max_context_length=max_length,
                max_new_tokens=max_output_len,
                lora_manager=lora_manager,
                lora_uids=lora_uids,
            )

            outputs = decoder.decode(
                line_encoded,
                input_lengths,
                sampling_config,
                *ptuning_args,
                stop_words_list=stop_words_list,
                bad_words_list=bad_words_list,
                no_repeat_ngram_size=no_repeat_ngram_size,
                streaming=streaming,
                output_sequence_lengths=True,
                return_dict=True,
            )
            torch.cuda.synchronize()

        runtime_rank = tensorrt_llm.mpi_rank()
        if runtime_rank == 0 or multiprocessed_env:
            return outputs, decoder.log_probs
        else:
            return None

    except Exception as e:
        print(e)
        raise e


def load(
    tokenizer: PreTrainedTokenizer, engine_dir: str, lora_ckpt_list: List[str] = None, num_beams: int = 1
) -> TensorrtLLMHostContext:
    """Loaded the compiled LLM model and run it.

    It also supports running the TRT LLM model on multi-GPU.
    """
    # the parent dir of the engine_dir
    config_path = os.path.join(engine_dir, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    world_size = config["builder_config"]["world_size"]
    if world_size == 1:
        _load(tokenizer, engine_dir, lora_ckpt_list, num_beams)
        executor = None
    else:
        executor = MPIPoolExecutor(max_workers=world_size)
        futures = []
        for _ in range(world_size):
            future = executor.submit(_load, tokenizer, engine_dir, lora_ckpt_list, num_beams)
            futures.append(future)
        for future in futures:
            future.result()

    max_batch_size = config["builder_config"]["max_batch_size"]
    max_input_len = config["builder_config"]["max_input_len"]
    add_bos = config["builder_config"]["add_bos"]

    return TensorrtLLMHostContext(
        executor=executor,
        world_size=world_size,
        tokenizer=tokenizer,
        max_batch_size=max_batch_size,
        max_input_len=max_input_len,
        add_bos=add_bos,
    )


def load_refit(
    tokenizer,
    engine_dir: str,
    lora_ckpt_list: List[str] = None,
    num_beams: int = 1,
    model_configs: List = None,
    stream=None,
) -> TensorrtLLMHostContext:
    """Loaded the compiled LLM model and run it.

    It also supports running the TRT LLM model on multi-GPU.
    """

    config_path = os.path.join(engine_dir, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    """The impl of `load` API for on a single GPU worker."""
    tensorrt_llm.logger.set_level("error")

    engine_dir = Path(engine_dir)
    config_path = engine_dir / "config.json"

    (
        model_config,
        world_size,
        tensor_parallel_size,
        pipeline_parallel_size,
        dtype,
        max_input_len,
        max_batch_size,
    ) = _read_config(config_path)

    runtime_rank = torch.cuda.current_device()
    assert runtime_rank < torch.cuda.device_count(), f"Rank {runtime_rank} out of bound"

    # Manipulate the tensorrt_llm mapping to make it compatible with the multiprocessed env.
    assert tensorrt_llm.mpi_world_size() == torch.distributed.get_world_size(), "MPI world size mismatch"
    runtime_mapping = tensorrt_llm.Mapping(
        world_size=tensorrt_llm.mpi_world_size(), rank=runtime_rank, tp_size=tensorrt_llm.mpi_world_size(), pp_size=1,
    )

    engine_name = get_engine_name(
        MODEL_NAME, dtype, tensor_parallel_size, pipeline_parallel_size, tensorrt_llm.mpi_rank()
    )

    logger.info(f"Loading engine: Rank ({tensorrt_llm.mpi_rank()} -> {engine_dir}/{engine_name}")

    serialize_path = os.path.join(engine_dir, engine_name)
    with open(serialize_path, "rb") as f:
        engine_buffer = f.read()

    decoder = tensorrt_llm.runtime.GenerationSession(
        model_config, engine_buffer, runtime_mapping, debug_mode=False, stream=stream
    )
    runtime_mapping.rank = runtime_rank
    runtime_mapping.tp_group = get_tensor_parallel_group(
        tensor_parallel_size
    )  # Override the tp_group to support TP+DP
    runtime_mapping.tp_rank = runtime_rank
    runtime_mapping.tp_size = tensor_parallel_size
    runtime_mapping.pp_group = [runtime_rank]
    runtime_mapping.pp_rank = 0

    sampling_config = SamplingConfig(end_id=tokenizer.eos_token_id, pad_id=tokenizer.eos_token_id, num_beams=num_beams)

    if decoder.use_lora_plugin:
        lora_manager = LoraManager()
        if lora_ckpt_list is not None:
            lora_manager.load_from_nemo(
                model_files=lora_ckpt_list, model_config=model_config, runtime_mapping=runtime_mapping,
            )
    else:
        lora_manager = None

    # create a new builder and refit the current engine
    new_builder = LMHeadModelBuilder(model_configs[0])
    engine = decoder.runtime.engine
    refit_runtime_engine(new_builder.named_parameters(), engine)

    # Initialize the global context so it can be used during `run` API.
    global tensorrt_llm_worker_context
    tensorrt_llm_worker_context.decoder = decoder
    tensorrt_llm_worker_context.sampling_config = sampling_config
    tensorrt_llm_worker_context.lora_manager = lora_manager
    tensorrt_llm_worker_context.max_batch_size = max_batch_size
    tensorrt_llm_worker_context.max_input_len = max_input_len

    max_batch_size = config["builder_config"]["max_batch_size"]
    max_input_len = config["builder_config"]["max_input_len"]

    return TensorrtLLMHostContext(
        executor=None,
        world_size=world_size,
        tokenizer=tokenizer,
        max_batch_size=max_batch_size,
        max_input_len=max_input_len,
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


def generate(
    input_texts: List[str],
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
    output_log_probs=False,
    multiprocessed_env=False,
    **sampling_kwargs,
) -> Optional[List[List[str]]]:
    """Generate the output sequence from the input sequence.

    Returns a 2D string list with shape [batch_size, num_beams].
    """
    tokenizer = host_context.tokenizer

    if host_context.add_bos:
        input_tensors = [torch.IntTensor([tokenizer.bos_token_id] + tokenizer.encode(t)) for t in input_texts]
    else:
        input_tensors = [torch.IntTensor(tokenizer.encode(t)) for t in input_texts]

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

    outputs, log_probs = forward(
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

    if host_context.add_bos:
        input_tensors = [torch.IntTensor([tokenizer.bos_token_id] + tokenizer.encode(t)) for t in input_texts]
    else:
        input_tensors = [torch.IntTensor(tokenizer.encode(t)) for t in input_texts]

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

    outputs, log_probs = forward(
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
