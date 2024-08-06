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
import os.path
from typing import Iterable, List, Optional, Union

import numpy
import wrapt
from vllm import RequestOutput, SamplingParams
from vllm.config import CacheConfig, DeviceConfig, LoadConfig, LoadFormat, ParallelConfig, SchedulerConfig
from vllm.executor.ray_utils import initialize_ray_cluster

from nemo.deploy import ITritonDeployable
from nemo.deploy.utils import cast_output
from nemo.export.vllm.engine import NemoLLMEngine
from nemo.export.vllm.model_config import NemoModelConfig
from nemo.export.vllm.model_loader import NemoModelLoader

LOGGER = logging.getLogger("NeMo")


@wrapt.decorator
def noop_decorator(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


use_pytriton = True
try:
    from pytriton.decorators import batch
    from pytriton.model_config import Tensor
except Exception:
    use_pytriton = False


class vLLMExporter(ITritonDeployable):
    """
    The Exporter class implements conversion from a Nemo checkpoint format to something compatible with vLLM,
    loading the model in vLLM, and binding that model to a Triton server.

    Example:
        from nemo.export.vllm import Exporter
        from nemo.deploy import DeployPyTriton

        exporter = Exporter()
        exporter.export(
            nemo_checkpoint='/path/to/checkpoint.nemo',
            model_dir='/path/to/temp_dir',
            model_type='llama')

        server = DeployPyTriton(
            model=exporter,
            triton_model_name='LLAMA')

        server.deploy()
        server.serve()
        server.stop()
    """

    def __init__(self):
        self.request_id = 0

    def export(
        self,
        nemo_checkpoint: str,
        model_dir: str,
        model_type: str,
        device: str = 'auto',
        tensor_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        max_model_len: int = None,
        dtype: str = 'auto',
        seed: int = 0,
        log_stats: bool = True,
        weight_storage: str = 'auto',
        gpu_memory_utilization: float = 0.9,
    ):
        """
        Exports the Nemo checkpoint to vLLM and initializes the engine.

        Args:
            nemo_checkpoint (str): path to the nemo checkpoint.
            model_dir (str): path to a temporary directory to store weights and the tokenizer model.
                The temp dir may persist between subsequent export operations, in which case
                converted weights may be reused to speed up the export.
            model_type (str): type of the model, such as "llama", "mistral", "mixtral".
                Needs to be compatible with transformers.AutoConfig.
            device (str): type of the device to use by the vLLM engine.
                Supported values are "auto", "cuda", "cpu", "neuron".
            tensor_parallel_size (int): tensor parallelism.
            pipeline_parallel_size (int): pipeline parallelism.
                Values over 1 are not currently supported by vLLM.
            max_model_len (int): model context length.
            dtype (str): data type for model weights and activations.
                Possible choices: auto, half, float16, bfloat16, float, float32
                "auto" will use FP16 precision for FP32 and FP16 models,
                and BF16 precision for BF16 models.
            seed (int): random seed value.
            log_stats (bool): enables logging inference performance statistics by vLLM.
            weight_storage (str): controls how converted weights are stored:
                "file" - always write weights into a file inside 'model_dir',
                "memory" - always do an in-memory conversion,
                "cache" - reuse existing files if they are newer than the nemo checkpoint,
                "auto" - use "cache" for multi-GPU runs and "memory" for single-GPU runs.
            gpu_memory_utilization (float): The fraction of GPU memory to be used for the model
                executor, which can range from 0 to 1.
        """

        # Pouplate the basic configuration structures
        device_config = DeviceConfig(device)

        model_config = NemoModelConfig(
            nemo_checkpoint,
            model_dir,
            model_type,
            tokenizer_mode='auto',
            dtype=dtype,
            seed=seed,
            revision=None,
            code_revision=None,
            tokenizer_revision=None,
            max_model_len=max_model_len,
            quantization=None,  # TODO ???
            quantization_param_path=None,
            enforce_eager=False,
            max_seq_len_to_capture=None,
        )

        parallel_config = ParallelConfig(
            pipeline_parallel_size=pipeline_parallel_size, tensor_parallel_size=tensor_parallel_size
        )

        # See if we have an up-to-date safetensors file
        safetensors_file = os.path.join(model_config.model, 'model.safetensors')
        safetensors_file_valid = os.path.exists(safetensors_file) and os.path.getmtime(
            safetensors_file
        ) > os.path.getmtime(nemo_checkpoint)

        # Decide how we're going to convert the weights
        if weight_storage == 'auto':
            if parallel_config.distributed_executor_backend is not None:
                save_weights = not safetensors_file_valid
                inmemory_weight_conversion = False
            else:
                save_weights = False
                inmemory_weight_conversion = True

        elif weight_storage == 'cache':
            save_weights = not safetensors_file_valid
            inmemory_weight_conversion = False

        elif weight_storage == 'file':
            save_weights = True
            inmemory_weight_conversion = False

        elif weight_storage == 'memory':
            save_weights = False
            inmemory_weight_conversion = True

        else:
            raise ValueError(f'Unsupported value for weight_storage: "{weight_storage}"')

        # Convert the weights ahead-of-time, if needed
        if save_weights:
            NemoModelLoader.convert_and_store_nemo_weights(model_config, safetensors_file)
        elif not inmemory_weight_conversion:
            LOGGER.info(f'Using cached weights in {safetensors_file}')

        # TODO: these values are the defaults from vllm.EngineArgs.
        cache_config = CacheConfig(
            block_size=16,
            gpu_memory_utilization=gpu_memory_utilization,
            swap_space=4,
            cache_dtype='auto',
            sliding_window=model_config.get_sliding_window(),
        )

        # TODO: these values are the defaults from vllm.EngineArgs.
        scheduler_config = SchedulerConfig(
            max_num_batched_tokens=None,
            max_num_seqs=256,
            # Note: max_model_len can be derived by model_config if the input value is None
            max_model_len=model_config.max_model_len,
            use_v2_block_manager=False,
            num_lookahead_slots=0,
            delay_factor=0.0,
            enable_chunked_prefill=False,
        )

        load_config = LoadConfig(
            load_format=NemoModelLoader if inmemory_weight_conversion else LoadFormat.SAFETENSORS,
            download_dir=None,
            model_loader_extra_config=None,
        )

        # Initialize the cluster and specify the executor class.
        if device_config.device_type == "neuron":
            from vllm.executor.neuron_executor import NeuronExecutor

            executor_class = NeuronExecutor
        elif device_config.device_type == "cpu":
            from vllm.executor.cpu_executor import CPUExecutor

            executor_class = CPUExecutor
        elif parallel_config.distributed_executor_backend == "ray":
            initialize_ray_cluster(parallel_config)
            from vllm.executor.ray_gpu_executor import RayGPUExecutor

            executor_class = RayGPUExecutor
        elif parallel_config.distributed_executor_backend == "mp":
            from vllm.executor.multiproc_gpu_executor import MultiprocessingGPUExecutor

            executor_class = MultiprocessingGPUExecutor
        else:
            assert parallel_config.world_size == 1, "Ray is required if parallel_config.world_size > 1."
            from vllm.executor.gpu_executor import GPUExecutor

            executor_class = GPUExecutor

        # Initialize the engine
        self.engine = NemoLLMEngine(
            model_config=model_config,
            cache_config=cache_config,
            parallel_config=parallel_config,
            scheduler_config=scheduler_config,
            device_config=device_config,
            load_config=load_config,
            lora_config=None,
            multimodal_config=None,
            speculative_config=None,
            decoding_config=None,
            observability_config=None,
            prompt_adapter_config=None,
            executor_class=executor_class,
            log_stats=log_stats,
        )

    def _add_request_to_engine(
        self, prompt: str, max_output_len: int, temperature: float = 1.0, top_k: int = 1, top_p: float = 0.0
    ) -> str:
        if top_p <= 0.0:
            top_p = 1.0

        sampling_params = SamplingParams(max_tokens=max_output_len, temperature=temperature, top_k=top_k, top_p=top_p)

        request_id = str(self.request_id)
        self.request_id += 1

        self.engine.add_request(request_id, prompt, sampling_params)

        return request_id

    def _forward_regular(self, request_ids: List[str]):
        responses = [None] * len(request_ids)
        finished = [False] * len(request_ids)

        while not all(finished):
            request_outputs: List[RequestOutput] = self.engine.step()

            for request_output in request_outputs:
                if not request_output.finished:
                    continue

                try:
                    request_index = request_ids.index(request_output.request_id)
                except ValueError:
                    continue

                finished[request_index] = request_output.finished
                output_text = request_output.outputs[-1].text
                responses[request_index] = output_text

        return [[response] for response in responses]

    def _forward_streaming(self, request_ids: List[str]):
        responses = [None] * len(request_ids)
        finished = [False] * len(request_ids)

        while not all(finished):
            request_outputs: List[RequestOutput] = self.engine.step()

            for request_output in request_outputs:
                try:
                    request_index = request_ids.index(request_output.request_id)
                except ValueError:
                    continue

                finished[request_index] = request_output.finished
                output_text = request_output.outputs[-1].text
                responses[request_index] = output_text

            yield [[response] for response in responses]

    def _add_triton_request_to_engine(self, inputs: numpy.ndarray, index: int) -> str:
        return self._add_request_to_engine(
            prompt=inputs['prompts'][index][0].decode('UTF-8'),
            max_output_len=inputs['max_output_len'][index][0],
            temperature=inputs['temperature'][index][0],
            top_k=inputs['top_k'][index][0],
            top_p=inputs['top_p'][index][0],
        )

    @property
    def get_triton_input(self):
        inputs = (
            Tensor(name="prompts", shape=(-1,), dtype=bytes),
            Tensor(name="max_output_len", shape=(-1,), dtype=numpy.int_, optional=True),
            Tensor(name="top_k", shape=(-1,), dtype=numpy.int_, optional=True),
            Tensor(name="top_p", shape=(-1,), dtype=numpy.single, optional=True),
            Tensor(name="temperature", shape=(-1,), dtype=numpy.single, optional=True),
        )
        return inputs

    @property
    def get_triton_output(self):
        outputs = (Tensor(name="outputs", shape=(-1,), dtype=bytes),)
        return outputs

    @batch
    def triton_infer_fn(self, **inputs: numpy.ndarray):
        request_ids = []
        num_requests = len(inputs["prompts"])
        for index in range(num_requests):
            request_id = self._add_triton_request_to_engine(inputs, index)
            request_ids.append(request_id)

        responses = self._forward_regular(request_ids)
        responses = [r[0] for r in responses]

        output_tensor = cast_output(responses, numpy.bytes_)
        return {'outputs': output_tensor}

    @batch
    def triton_infer_fn_streaming(self, **inputs: numpy.ndarray):
        request_ids = []
        num_requests = len(inputs["prompts"])
        for index in range(num_requests):
            request_id = self._add_triton_request_to_engine(inputs, index)
            request_ids.append(request_id)

        for responses in self._forward_streaming(request_ids):
            responses = [r[0] for r in responses]
            output_tensor = cast_output(responses, numpy.bytes_)
            yield {'outputs': output_tensor}

    # Mimic the TensorRTLLM exporter's forward function, even though we don't support many of its features.
    def forward(
        self,
        input_texts: List[str],
        max_output_len: int = 64,
        top_k: int = 1,
        top_p: float = 0.0,
        temperature: float = 1.0,
        stop_words_list: Optional[List[str]] = None,
        bad_words_list: Optional[List[str]] = None,
        no_repeat_ngram_size: Optional[int] = None,
        task_ids: Optional[List[str]] = None,
        lora_uids: Optional[List[str]] = None,
        prompt_embeddings_table=None,
        prompt_embeddings_checkpoint_path: Optional[str] = None,
        streaming: bool = False,
        output_log_probs: bool = False,
    ) -> Union[List[List[str]], Iterable[List[List[str]]]]:
        """
        The forward function performs LLM evaluation on the provided array of prompts with other parameters shared,
        and returns the generated texts. If 'streaming' is True, the output texts are returned incrementally
        with a generator: one token appended to each output at a time. If 'streaming' is false, the final output texts
        are returned as a single list of responses.
        """

        if stop_words_list is not None and stop_words_list != []:
            raise NotImplementedError("stop_words_list is not supported")

        if bad_words_list is not None and bad_words_list != []:
            raise NotImplementedError("bad_words_list is not supported")

        if no_repeat_ngram_size is not None:
            raise NotImplementedError("no_repeat_ngram_size is not supported")

        if task_ids is not None and task_ids != []:
            raise NotImplementedError("task_ids is not supported")

        if lora_uids is not None and lora_uids != []:
            raise NotImplementedError("lora_uids is not supported")

        if prompt_embeddings_table is not None:
            raise NotImplementedError("prompt_embeddings_table is not supported")

        if prompt_embeddings_checkpoint_path is not None:
            raise NotImplementedError("prompt_embeddings_checkpoint_path is not supported")

        if output_log_probs:
            raise NotImplementedError("output_log_probs is not supported")

        request_ids = []
        for prompt in input_texts:
            request_id = self._add_request_to_engine(
                prompt=prompt, max_output_len=max_output_len, temperature=temperature, top_k=top_k, top_p=top_p
            )
            request_ids.append(request_id)

        if streaming:
            return self._forward_streaming(request_ids)
        else:
            return self._forward_regular(request_ids)
