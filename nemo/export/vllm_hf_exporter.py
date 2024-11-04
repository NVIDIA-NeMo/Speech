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


from typing import List

import numpy as np
from pytriton.decorators import batch
from pytriton.model_config import Tensor
from vllm import LLM, SamplingParams

from nemo.deploy import ITritonDeployable
from nemo.deploy.utils import cast_output, str_ndarray2list


class vLLMHFExporter(ITritonDeployable):
    """
    The Exporter class uses vLLM APIs to convert a HF model to vLLM and makes the class,
    deployable with Triton server.

    Example:
        from nemo.export import vLLMHFExporter
        from nemo.deploy import DeployPyTriton

        exporter = vLLMHFExporter()
        exporter.export(model="/path/to/model/")

        server = DeployPyTriton(
            model=exporter,
            triton_model_name='model'
        )

        server.deploy()
        server.serve()
        server.stop()
    """

    def __init__(self):
        self.model = None

    def export(self, model):
        """
        Exports the HF checkpoint to vLLM and initializes the engine.
        Args:
            model (str): model name or the path
        """
        self.model = LLM(model=model)

    @property
    def get_triton_input(self):
        inputs = (
            Tensor(name="prompts", shape=(-1,), dtype=bytes),
            Tensor(name="max_output_len", shape=(-1,), dtype=np.int_, optional=True),
            Tensor(name="top_k", shape=(-1,), dtype=np.int_, optional=True),
            Tensor(name="top_p", shape=(-1,), dtype=np.single, optional=True),
            Tensor(name="temperature", shape=(-1,), dtype=np.single, optional=True),
        )
        return inputs

    @property
    def get_triton_output(self):
        outputs = (Tensor(name="outputs", shape=(-1,), dtype=bytes),)
        return outputs

    @batch
    def triton_infer_fn(self, **inputs: np.ndarray):
        try:
            infer_input = {"input_texts": str_ndarray2list(inputs.pop("prompts"))}
            if "max_output_len" in inputs:
                infer_input["max_output_len"] = inputs.pop("max_output_len")[0][0]
            if "top_k" in inputs:
                infer_input["top_k"] = inputs.pop("top_k")[0][0]
            if "top_p" in inputs:
                infer_input["top_p"] = inputs.pop("top_p")[0][0]
            if "temperature" in inputs:
                infer_input["temperature"] = inputs.pop("temperature")[0][0]

            output_texts = self.forward(**infer_input)
            output = cast_output(output_texts, np.bytes_)
        except Exception as error:
            err_msg = "An error occurred: {0}".format(str(error))
            output = cast_output([err_msg], np.bytes_)

        return {"outputs": output}

    def forward(
        self,
        input_texts: List[str],
        max_output_len: int = 64,
        top_k: int = 1,
        top_p: float = 0.0,
        temperature: float = 1.0,
    ):
        assert self.model is not None, "Model is not initialized."

        sampling_params = SamplingParams(
            max_tokens=max_output_len, temperature=temperature, top_k=int(top_k), top_p=top_p
        )
        request_output = self.model.generate(input_texts, sampling_params)
        output = []
        for o in request_output:
            output.append(o.outputs[0].text)

        return output
