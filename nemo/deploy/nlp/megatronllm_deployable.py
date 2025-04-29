# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.distributed
import wrapt
from jinja2 import Template
from megatron.core.dist_checkpointing.validation import StrictHandling
from megatron.core.inference.common_inference_params import CommonInferenceParams
from megatron.core.inference.inference_request import InferenceRequest

import nemo.lightning as nl
from nemo.collections.llm import inference
from nemo.deploy import ITritonDeployable
from nemo.deploy.utils import NEMO2, broadcast_list, cast_output, nemo_checkpoint_version, str_ndarray2list


@wrapt.decorator
def noop_decorator(func):
    """A no-op decorator that returns the original function unchanged.

    Used as a fallback when pytriton's batch decorator is not available.

    Args:
        func: The function to decorate

    Returns:
        The original function without any modifications
    """

    def wrapper(*args, **kwargs):
        """
        Wrapper method returning the func.
        """
        return func(*args, **kwargs)

    return wrapper


use_pytriton = True
batch = noop_decorator
try:
    from pytriton.decorators import batch, first_value
    from pytriton.model_config import Tensor
except Exception:
    use_pytriton = False

LOGGER = logging.getLogger("NeMo")


class MegatronLLMDeploy:
    """
    A factory class for creating deployable instances of Megatron LLM models.
    This class provides a method to get the appropriate deployable instance
    based on the version of the NeMo checkpoint model used.
    """

    @staticmethod
    def get_deployable(
        nemo_checkpoint_filepath: str,
        num_devices: int = 1,
        num_nodes: int = 1,
        tensor_model_parallel_size: int = 1,
        pipeline_model_parallel_size: int = 1,
        context_parallel_size: int = 1,
        legacy_ckpt: bool = False,
    ):
        """
        Returns the appropriate deployable instance for the given NeMo checkpoint.

        Args:
            nemo_checkpoint_filepath (str): Path to the .nemo checkpoint file.
            num_devices (int): Number of devices to use for deployment.
            num_nodes (int): Number of nodes to use for deployment.
            tensor_model_parallel_size (int): Size of the tensor model parallelism.
            pipeline_model_parallel_size (int): Size of the pipeline model parallelism.
            context_parallel_size (int): Size of the context parallelism.

        Returns:
            ITritonDeployable: An instance of a deployable class compatible with Triton inference server.
        """
        if nemo_checkpoint_version(nemo_checkpoint_filepath) == NEMO2:
            return MegatronLLMDeployableNemo2(
                nemo_checkpoint_filepath=nemo_checkpoint_filepath,
                num_devices=num_devices,
                num_nodes=num_nodes,
                tensor_model_parallel_size=tensor_model_parallel_size,
                pipeline_model_parallel_size=pipeline_model_parallel_size,
                context_parallel_size=context_parallel_size,
                legacy_ckpt=legacy_ckpt,
            )
        else:
            raise Exception("Only NeMo 2.0 checkpoint is supported.")


class MegatronLLMDeployableNemo2(ITritonDeployable):
    """
    Triton inference server compatible deploy class for a .nemo model file

    Args:
        nemo_checkpoint_filepath (str): path for the nemo checkpoint.
        num_devices (int): number of GPUs.
        num_nodes (int): number of nodes.
        tensor_model_parallel_size (int): tensor parallelism.
        pipeline_parallelism_size (int): pipeline parallelism.
        context_parallel_size (int): context parallelism.
        params_dtype (torch.dtype): max input length.
        inference_batch_times_seqlen_threshold (int): squence threshold.
        inference_max_seq_length (int): max_seq_length for inference. Required by MCoreEngine (>=0.12). Defaults to
        4096.
    """

    def __init__(
        self,
        nemo_checkpoint_filepath: str = None,
        num_devices: int = 1,
        num_nodes: int = 1,
        tensor_model_parallel_size: int = 1,
        pipeline_model_parallel_size: int = 1,
        context_parallel_size: int = 1,
        expert_model_parallel_size: int = 1,
        expert_tensor_parallel_size: int = 1,
        params_dtype: torch.dtype = torch.bfloat16,
        inference_batch_times_seqlen_threshold: int = 1000,
        inference_max_seq_length: int = 4096,
        legacy_ckpt: bool = False,
    ):
        self.nemo_checkpoint_filepath = nemo_checkpoint_filepath

        strategy = nl.MegatronStrategy(
            tensor_model_parallel_size=tensor_model_parallel_size,
            pipeline_model_parallel_size=pipeline_model_parallel_size,
            context_parallel_size=context_parallel_size,
            expert_model_parallel_size=expert_model_parallel_size,
            expert_tensor_parallel_size=expert_tensor_parallel_size,
            sequence_parallel=False,
            setup_optimizers=False,
            store_optimizer_states=False,
            ckpt_load_strictness=StrictHandling.LOG_ALL if legacy_ckpt else None,
        )

        trainer = nl.Trainer(
            accelerator="gpu",
            devices=num_devices,
            num_nodes=num_nodes,
            strategy=strategy,
            plugins=nl.MegatronMixedPrecision(
                precision="bf16-mixed",
                params_dtype=torch.bfloat16,
                pipeline_dtype=torch.bfloat16,
                autocast_enabled=False,
                grad_reduce_in_fp32=False,
            ),
        )

        self.inference_wrapped_model, self.mcore_tokenizer = inference.setup_model_and_tokenizer(
            path=Path(nemo_checkpoint_filepath),
            trainer=trainer,
            params_dtype=params_dtype,
            inference_batch_times_seqlen_threshold=inference_batch_times_seqlen_threshold,
            inference_max_seq_length=inference_max_seq_length,
        )

    def generate(
        self,
        prompts: List[str],
        max_batch_size: int = 4,
        inference_params: Optional[CommonInferenceParams] = None,
        random_seed: Optional[int] = None,
    ) -> List[InferenceRequest]:
        """
        Generates text based on the provided input prompts.

        Args:
            prompts (List[str]): A list of input strings.
            max_batch_size (int): The maximum batch size used for inference.
            inference_params (Optional[CommonInferenceParams]): Parameters for controlling the inference process.
            random_seed (Optional[int]): A random seed for reproducibility.

        Returns:
            List[InferenceRequest]: A list containing the generated results.
        """

        inference_params = inference_params or CommonInferenceParams()
        results = inference.generate(
            model=self.inference_wrapped_model,
            tokenizer=self.mcore_tokenizer,
            prompts=prompts,
            max_batch_size=max_batch_size,
            random_seed=random_seed,
            inference_params=inference_params,
        )
        return list(results)

    def generate_other_ranks(self):
        """
        Generate function for ranks other than the rank 0.
        """

        while True:
            message = torch.empty(1, dtype=torch.long, device="cuda")
            torch.distributed.broadcast(message, src=0)
            if message == 0:
                prompts = broadcast_list(data=[None], src=0)
                max_batch_size, random_seed, temperature, top_k, top_p, num_tokens_to_generate, log_probs = (
                    broadcast_list(data=[None], src=0)
                )

                inference_params = CommonInferenceParams(
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    num_tokens_to_generate=num_tokens_to_generate,
                    return_log_probs=log_probs,
                )

                self.generate(prompts, max_batch_size, inference_params, random_seed)
            else:
                return

    def apply_chat_template(self, messages, add_generation_prompt=True):
        """
        Load the chat template.
        Works when model's tokenizer has chat template (typically chat models).
        """
        try:
            tokenizer_chat_template = self.mcore_tokenizer.tokenizer.tokenizer.chat_template
            bos_token = self.mcore_tokenizer.tokenizer.tokenizer.bos_token
            template = Template(tokenizer_chat_template)
        except AttributeError:
            # If the tokenizer does not have chat_template
            raise ValueError(
                "The tokenizer does not have chat template, if you would like to evaluate chat model \
                             ensure your model's tokenizer has a chat template"
            )
        # Render the template with the provided messages
        rendered_output = template.render(
            messages=messages, bos_token=bos_token, add_generation_prompt=add_generation_prompt
        )

        return rendered_output

    def remove_eos_token(self, text):
        """
        Removes eos token if it exists in the output, otherwise does nothing
        """
        eos_token = self.mcore_tokenizer.tokenizer.tokenizer.eos_token
        output = []
        for t in text:
            if eos_token in t:
                output.append(t.rsplit(eos_token, 1)[0])
            else:
                output.append(t)
        return output

    def str_to_dict(self, json_str):
        """
        Convert str to dict.
        """
        return json.loads(json_str)

    @property
    def get_triton_input(self):
        inputs = (
            Tensor(name="prompts", shape=(-1,), dtype=bytes),
            Tensor(name="max_length", shape=(-1,), dtype=np.int_, optional=True),
            Tensor(name="max_batch_size", shape=(-1,), dtype=np.int_, optional=True),
            Tensor(name="top_k", shape=(-1,), dtype=np.int_, optional=True),
            Tensor(name="top_p", shape=(-1,), dtype=np.single, optional=True),
            Tensor(name="temperature", shape=(-1,), dtype=np.single, optional=True),
            Tensor(name="random_seed", shape=(-1,), dtype=np.int_, optional=True),
            Tensor(name="compute_logprob", shape=(-1,), dtype=np.bool_, optional=True),
            Tensor(name="apply_chat_template", shape=(-1,), dtype=np.bool_, optional=True),
        )
        return inputs

    @property
    def get_triton_output(self):
        return (
            Tensor(name="sentences", shape=(-1,), dtype=bytes),
            Tensor(name="log_probs", shape=(-1,), dtype=np.single),
        )

    @batch
    @first_value(
        "max_length",
        "max_batch_size",
        "top_k",
        "top_p",
        "temperature",
        "random_seed",
        "compute_logprob",
        "apply_chat_template",
    )
    def triton_infer_fn(self, **inputs: np.ndarray):
        output_infer = {}
        prompts = str_ndarray2list(inputs.pop("prompts"))
        max_batch_size = inputs.pop("max_batch_size", 32)
        random_seed = inputs.pop("random_seed", None)
        temperature = inputs.pop("temperature", 1.0)
        top_k = inputs.pop("top_k", 1)
        top_p = inputs.pop("top_p", 0.0)
        num_tokens_to_generate = inputs.pop("max_length", 256)
        log_probs = inputs.pop("compute_logprob", False)
        apply_chat_template = inputs.pop("apply_chat_template", False)
        text_only = True

        if apply_chat_template:
            # Deserialize the JSON string back to a dictionary
            prompts = [self.str_to_dict(prompt) for prompt in prompts]
            prompts = [self.apply_chat_template(prompt) for prompt in prompts]
            # Input to generate should be list of string, otherwise if its string directly TE raises an error:
            # The provided qkv memory layout is not supported!
        if torch.distributed.is_initialized():
            if torch.distributed.get_world_size() > 1:
                torch.distributed.broadcast(torch.tensor([0], dtype=torch.long, device="cuda"), src=0)
                broadcast_list(prompts, src=0)
                broadcast_list(
                    data=[
                        max_batch_size,
                        random_seed,
                        temperature,
                        top_k,
                        top_p,
                        num_tokens_to_generate,
                        log_probs,
                    ],
                    src=0,
                )

        inference_params = CommonInferenceParams(
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            num_tokens_to_generate=num_tokens_to_generate,
            return_log_probs=log_probs,
        )

        results = self.generate(prompts, max_batch_size, inference_params, random_seed)
        output_texts = [r.generated_text if text_only else r for r in results]
        output_texts = self.remove_eos_token(output_texts)
        output_infer = {"sentences": cast_output(output_texts, np.bytes_)}
        if log_probs:
            output_log_probs = []  ## will have 2 np arrays if 2 prompts are sent
            for r in results:
                # Convert to torch tensor and then move to cpu as generated_log_probs is a list and cant be moved
                # to cpu otherwise
                lp = torch.tensor(r.generated_log_probs).cpu().detach().numpy()
                if len(lp) == 0:
                    output_log_probs.append([0])
                else:
                    output_log_probs.append(lp)
            output_infer["log_probs"] = np.array(output_log_probs)

        return output_infer
