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

from typing import Optional, Union

from pydantic import BaseModel, Field


class ApiEndpoint(BaseModel):
    """
    Represents evaluation Standard API target.api_endpoint object
    """

    url: str = Field(description="Url of the model", default="http://0.0.0.0:8000")
    model_id: str = Field(description="Name of the model in API", default="triton_model")
    nemo_checkpoint_path: Optional[str] = Field(
        description="Path for nemo 2.0 checkpoint",
        default=None,
    )
    nemo_triton_http_port: Optional[int] = Field(
        description="HTTP port that was used for the PyTriton server in the deploy method. Default: 8000.",
        default=8000,
    )


class EvaluationTarget(BaseModel):
    """
    Represents evaluation Standard API target object
    """

    api_endpoint: ApiEndpoint = Field(description="Api endpoint to be used for evaluation")


class ConfigParams(BaseModel):
    """
    Represents evaluation Standard API config.params object
    """

    top_p: float = Field(
        description="Limits to the top tokens within a certain probability",
        default=0.9999999,
    )
    temperature: float = Field(
        description="Temp of 0 indicates greedy decoding, where the token with highest prob is chosen",
        default=0.0000001,
    )
    limit_samples: Optional[Union[int, float]] = Field(
        description="Limit evaluation to `limit` samples. Default: use all samples", default=None
    )
    num_fewshot: Optional[int] = Field(
        description="Number of examples in few-shot context. Default: None which means no few_shots are used.",
        default=None,
    )
    max_new_tokens: Optional[int] = Field(description="max tokens to generate", default=256)
    batch_size: Optional[int] = Field(description="batch size to use for evaluation", default=1)
    top_k: Optional[int] = Field(
        description="Limits to a certain number (K) of the top tokens to consider",
        default=1,
    )
    add_bos: Optional[bool] = Field(
        description="whether a special bos token should be added when encoding a string",
        default=False,
    )
    bootstrap_iters: int = Field(
        description="Number of iterations for bootstrap statistics",
        default=100000,
    )


class EvaluationConfig(BaseModel):
    """
    Represents evaluation Standard API config object
    """

    type: str = Field(description="Name/type of the task")
    params: ConfigParams = Field(description="Parameters to be used for evaluation", default=ConfigParams())
