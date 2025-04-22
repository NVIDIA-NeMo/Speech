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


import numpy as np
from pytriton.decorators import batch
from pytriton.model_config import Tensor

from nemo.deploy import DeployPyTriton, ITritonDeployable
from nemo.deploy.nlp import NemoQueryLLM
from nemo.deploy.utils import cast_output, str_ndarray2list


class MockModel(ITritonDeployable):

    @property
    def get_triton_input(self):
        inputs = (
            Tensor(name="prompts", shape=(-1,), dtype=bytes),
            Tensor(name="max_output_len", shape=(-1,), dtype=np.int_, optional=True),
            Tensor(name="output_context_logits", shape=(-1,), dtype=np.bool_, optional=False),
            Tensor(name="output_generation_logits", shape=(-1,), dtype=np.bool_, optional=False),
        )
        return inputs

    @property
    def get_triton_output(self):
        outputs = (Tensor(name="outputs", shape=(-1,), dtype=bytes),)
        return outputs

    @batch
    def triton_infer_fn(self, **inputs: np.ndarray):
        infer_input = {"input_texts": str_ndarray2list(inputs.pop("prompts"))}
        if "max_output_len" in inputs:
            infer_input["max_output_len"] = inputs.pop("max_output_len")[0][0]

        output_dict = dict()
        output_dict["outputs"] = cast_output("I am good, how about you?", np.bytes_)
        return output_dict


def test_nemo_deploy_query():
    model_name = "mock_model"
    model = MockModel()
    nm = DeployPyTriton(
        model=model,
        triton_model_name=model_name,
        max_batch_size=32,
        http_port=9002,
        grpc_port=8001,
        address="0.0.0.0",
        allow_grpc=True,
        allow_http=True,
        streaming=False,
    )
    nm.deploy()
    nm.run()

    nq = NemoQueryLLM(url="localhost:9002", model_name=model_name)
    output_deployed = nq.query_llm(
        prompts=["Hey, how is it going?"],
        max_output_len=20,
    )
    nm.stop()

    assert output_deployed is not None, "Output cannot be none."
    assert output_deployed == "I am good, how about you?", "Output cannot be none."
