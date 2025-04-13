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


import logging
from typing import Any, List, Optional

import numpy as np
import torch
from peft import PeftModel
from pytriton.decorators import batch
from pytriton.model_config import Tensor
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from nemo.deploy import ITritonDeployable
from nemo.deploy.utils import broadcast_list, cast_output, str_ndarray2list

LOGGER = logging.getLogger("NeMo")

SUPPORTED_TASKS = ["text-generation"]


class HuggingFaceLLMDeploy(ITritonDeployable):
    """A Triton inference server compatible wrapper for HuggingFace models.

    This class provides a standardized interface for deploying HuggingFace models
    in Triton inference server. It supports various NLP tasks and handles model
    loading, inference, and deployment configurations.

    Args:
        hf_model_id_path (Optional[str]): Path to the HuggingFace model or model identifier.
            Can be a local path or a model ID from HuggingFace Hub.
        hf_peft_model_id_path (Optional[str]): Path to the PEFT model or model identifier.
            Can be a local path or a model ID from HuggingFace Hub.
        tokenizer_id_path (Optional[str]): Path to the tokenizer or tokenizer identifier.
            If None, will use the same path as hf_model_id_path.
        model (Optional[AutoModel]): Pre-loaded HuggingFace model.
        tokenizer (Optional[AutoTokenizer]): Pre-loaded HuggingFace tokenizer.
        tokenizer_padding (bool): Whether to enable padding in tokenizer. Defaults to True.
        tokenizer_truncation (bool): Whether to enable truncation in tokenizer. Defaults to True.
        tokenizer_padding_side (str): Which side to pad on ('left' or 'right'). Defaults to 'left'.
        task (str): HuggingFace task type (e.g., "text-generation"). Defaults to "text-generation".
        **hf_kwargs: Additional keyword arguments to pass to HuggingFace model loading.
    """

    def __init__(
        self,
        hf_model_id_path: Optional[str] = None,
        hf_peft_model_id_path: Optional[str] = None,
        tokenizer_id_path: Optional[str] = None,
        model: Optional[AutoModel] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        tokenizer_padding=True,
        tokenizer_truncation=True,
        tokenizer_padding_side="left",
        task: Optional[str] = "text-generation",
        **hf_kwargs,
    ):
        if hf_model_id_path is None and model is None:
            raise ValueError("hf_model_id_path or model parameters has to be passed.")
        elif hf_model_id_path is not None and model is not None:
            LOGGER.warning(
                "hf_model_id_path will be ignored and the HuggingFace model " "set with model parameter will be used."
            )

        assert task in SUPPORTED_TASKS, "Task {0} is not a support task.".format(task)

        self.hf_model_id_path = hf_model_id_path
        self.hf_peft_model_id_path = hf_peft_model_id_path
        self.task = task
        self.model = model
        self.tokenizer = tokenizer
        self.tokenizer_padding = tokenizer_padding
        self.tokenizer_truncation = tokenizer_truncation
        self.tokenizer_padding_side = tokenizer_padding_side

        if tokenizer_id_path is None:
            self.tokenizer_id_path = hf_model_id_path
        else:
            self.tokenizer_id_path = tokenizer_id_path

        if model is None:
            self._load(**hf_kwargs)

    def _load(self, **hf_kwargs) -> None:
        """
        Load the HuggingFace pipeline with the specified model and task.

        This method initializes the HuggingFace AutoModel classes using the provided model
        configuration and task type. It handles the model and tokenizer loading
        process.

        Raises:
            AssertionError: If task is not specified.
        """
        assert self.task is not None, "A task has to be given for the generation task."

        if self.task == "text-generation":
            self.model = AutoModelForCausalLM.from_pretrained(self.hf_model_id_path, **hf_kwargs)

            if self.hf_peft_model_id_path is not None:
                self.model = PeftModel.from_pretrained(self.model, self.hf_peft_model_id_path)
        else:
            raise ValueError("Task {0} is not supported.".format(self.task))

        self.model.cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_id_path,
            trust_remote_code=hf_kwargs.pop("trust_remote_code", False),
            padding=self.tokenizer_padding,
            truncation=self.tokenizer_truncation,
            padding_side=self.tokenizer_padding_side,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(
        self,
        **kwargs: Any,
    ) -> List[str]:
        """Generate text based on the provided input prompts.

        This method processes input prompts through the loaded pipeline and
        generates text according to the specified parameters.

        Args:
            **kwargs: Generation parameters including:
                - text_inputs: List of input prompts
                - max_length: Maximum number of tokens to generate
                - num_return_sequences: Number of sequences to generate per prompt
                - temperature: Sampling temperature
                - top_k: Number of highest probability tokens to consider
                - top_p: Cumulative probability threshold for token sampling
                - do_sample: Whether to use sampling
                - return_full_text: Whether to return full text or only generated part

        Returns:
            If output logits and output scores are False:
            List[str]: A list of generated texts, one for each input prompt.
            If output logits and output scores are True:
            Dict: A dictionary containing:
                - sentences: List of generated texts
                - logits: List of logits
                - scores: List of scores

        Raises:
            RuntimeError: If the pipeline is not initialized.
        """

        if not self.model:
            raise RuntimeError("Model is not initialized")

        inputs = self.tokenizer(
            kwargs["text_inputs"],
            return_tensors="pt",
            padding=self.tokenizer_padding,
            truncation=self.tokenizer_truncation,
        )
        kwargs = {**inputs, **kwargs}
        kwargs.pop("text_inputs")
        for key, val in kwargs.items():
            if torch.is_tensor(val):
                kwargs[key] = val.cuda()

        with torch.no_grad():
            generated_ids = self.model.generate(**kwargs)
        return_dict_in_generate = kwargs.get("return_dict_in_generate", False)
        if return_dict_in_generate:
            output = {"sentences": self.tokenizer.batch_decode(generated_ids["sequences"], skip_special_tokens=True)}
            if kwargs.get("output_logits", False):
                output["logits"] = generated_ids["logits"]
            if kwargs.get("output_scores", False):
                output["scores"] = generated_ids["scores"]
        else:
            output = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return output

    def generate_other_ranks(self):
        """
        Generate function for ranks other than the rank 0.
        """

        while True:
            message = torch.empty(1, dtype=torch.long, device="cuda")
            torch.distributed.broadcast(message, src=0)
            if message == 0:
                prompts = broadcast_list(data=[None], src=0)
                temperature, top_k, top_p, num_tokens_to_generate, output_logits, output_scores = broadcast_list(
                    data=[None], src=0
                )

                return_dict_in_generate = False
                if output_logits or output_scores:
                    return_dict_in_generate = True

                self.generate(
                    text_inputs=prompts,
                    do_sample=True,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                    max_new_tokens=num_tokens_to_generate,
                    output_logits=output_logits,
                    output_scores=output_scores,
                    return_dict_in_generate=return_dict_in_generate,
                )
            else:
                return

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
            Tensor(name="max_length", shape=(-1,), dtype=np.int_, optional=True),
            Tensor(name="output_logits", shape=(-1,), dtype=np.bool_, optional=True),
            Tensor(name="output_scores", shape=(-1,), dtype=np.bool_, optional=True),
        )
        return inputs

    @property
    def get_triton_output(self):
        return (
            Tensor(name="sentences", shape=(-1,), dtype=bytes),
            Tensor(name="logits", shape=(-1,), dtype=np.single),
            Tensor(name="scores", shape=(-1,), dtype=np.single),
        )

    @batch
    def triton_infer_fn(self, **inputs: np.ndarray):
        output_infer = {}

        try:
            prompts = str_ndarray2list(inputs.pop("prompts"))
            temperature = inputs.pop("temperature")[0][0] if "temperature" in inputs else 1.0
            top_k = int(inputs.pop("top_k")[0][0] if "top_k" in inputs else 1)
            top_p = inputs.pop("top_p")[0][0] if "top_k" in inputs else 0.0
            num_tokens_to_generate = inputs.pop("max_length")[0][0] if "max_length" in inputs else 256
            output_logits = inputs.pop("output_logits")[0][0] if "output_logits" in inputs else False
            output_scores = inputs.pop("output_scores")[0][0] if "output_scores" in inputs else False
            return_dict_in_generate = False
            if output_logits or output_scores:
                return_dict_in_generate = True

            if torch.distributed.is_initialized():
                if torch.distributed.get_world_size() > 1:
                    torch.distributed.broadcast(torch.tensor([0], dtype=torch.long, device="cuda"), src=0)
                    broadcast_list(prompts, src=0)
                    broadcast_list(
                        data=[
                            temperature,
                            top_k,
                            top_p,
                            num_tokens_to_generate,
                            output_logits,
                            output_scores,
                        ],
                        src=0,
                    )

            output = self.generate(
                text_inputs=prompts,
                do_sample=True,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                max_new_tokens=num_tokens_to_generate,
                output_logits=output_logits,
                output_scores=output_scores,
                return_dict_in_generate=return_dict_in_generate,
            )

            if isinstance(output, dict):
                output_infer = {"sentences": cast_output(output["sentences"], np.bytes_)}

                if "scores" in output.keys():
                    output_scores = []
                    for r in output["scores"]:
                        lp = torch.tensor(r).cpu().detach().numpy()
                        if len(lp) == 0:
                            output_scores.append([0])
                        else:
                            output_scores.append(lp)
                    output_infer["scores"] = np.array(output_scores).transpose(1, 0, 2)

                if "logits" in output.keys():
                    output_logits = []
                    for r in output["logits"]:
                        lp = torch.tensor(r).cpu().detach().numpy()
                        if len(lp) == 0:
                            output_logits.append([0])
                        else:
                            output_logits.append(lp)
                    output_infer["logits"] = np.array(output_logits).transpose(1, 0, 2)
            else:
                output_infer = {"sentences": cast_output(output, np.bytes_)}

        except Exception as error:
            err_msg = "An error occurred: {0}".format(str(error))
            output_infer["sentences"] = cast_output([err_msg], np.bytes_)

        return output_infer
