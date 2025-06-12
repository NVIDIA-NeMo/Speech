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

from typing import Any, Dict, Optional, Union

from pydantic import BaseModel, Field


class ApiEndpoint(BaseModel):
    """
    Represents evaluation Standard API target.api_endpoint object
    """

    url: str = Field(description="Url of the model", default="http://0.0.0.0:8080/v1/completions/")
    model_id: str = Field(description="Name of the model in API", default="triton_model")
    type: str = Field(description="The type of the target (chat or completions)", default="completions")
    nemo_checkpoint_path: Optional[str] = Field(
        description="Path for nemo 2.0 checkpoint",
        default=None,
        deprecated="This parameter is deprecated and not used for evaluations with NVIDIA Evals Factory.",
    )
    nemo_triton_http_port: Optional[int] = Field(
        description="HTTP port that was used for the PyTriton server in the deploy method. Default: 8000.",
        default=8000,
        deprecated="This parameter is deprecated and not used for evaluations with NVIDIA Evals Factory.",
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
        default=0,
    )
    temperature: float = Field(
        description="Temp of 0 indicates greedy decoding, where the token with highest prob is chosen",
        default=0,
    )
    limit_samples: Optional[Union[int, float]] = Field(
        description="Limit evaluation to `limit` samples. Default: use all samples", default=None
    )
    num_fewshot: Optional[int] = Field(
        description="Number of examples in few-shot context. Default: None.",
        default=None,
        deprecated="This parameter is deprecated and not used for evaluations with NVIDIA Evals Factory. "
        "In order to specify few-shot please use extra.num_fewshot.",
    )
    max_new_tokens: Optional[int] = Field(description="max tokens to generate", default=None)
    max_retries: Optional[int] = Field(description="Number of REST request retries", default=None)
    parallelism: Optional[int] = Field(description="Number of parallel requests to be sent to the server", default=1)
    task: Optional[str] = Field(description="Name of the task", default=None)
    request_timeout: Optional[int] = Field(description="REST response timeout", default=300)
    extra: Optional[Dict[str, Any]] = Field(
        description="Framework specific parameters to be used for evaluation (e.g. num_fewshot)", default_factory=dict
    )
    batch_size: Optional[int] = Field(
        description="batch size to use for evaluation",
        default=1,
        deprecated="This parameter is deprecated and not used for evaluations with NVIDIA Evals Factory.",
    )
    top_k: Optional[int] = Field(
        description="Limits to a certain number (K) of the top tokens to consider",
        default=1,
        deprecated="This parameter is deprecated and not used for evaluations with NVIDIA Evals Factory.",
    )
    add_bos: Optional[bool] = Field(
        description="whether a special bos token should be added when encoding a string",
        default=False,
        deprecated="This parameter is deprecated and not used for evaluations with NVIDIA Evals Factory.",
    )
    bootstrap_iters: int = Field(
        description="Number of iterations for bootstrap statistics",
        default=100000,
        deprecated="This parameter is deprecated and not used for evaluations with NVIDIA Evals Factory.",
    )

    def __init__(self, **data):
        """
        WAR for default tokenizer coming from a gated repo in nvidia-lm-eval==25.03.
        The tokenizer is not used for generation tasks so should be set to None
        """
        super().__init__(**data)
        if "tokenizer" not in self.extra:
            self.extra["tokenizer"] = None


class EvaluationConfig(BaseModel):
    """
    Represents evaluation Standard API config object
    """

    output_dir: str = Field(description="Directory to output the results", default="results")
    supported_endpoint_types: Optional[list[str]] = Field(
        description="Supported endpoint types like chat or completions", default=None
    )
    type: str = Field(description="Name/type of the task")
    params: ConfigParams = Field(description="Parameters to be used for evaluation", default=ConfigParams())


class AdapterConfig(BaseModel):
    """Adapter is a mechanism for hooking into the chain of requests/responses btw benchmark and endpoint."""

    @staticmethod
    def get_validated_config(run_config: dict[str, Any]) -> "AdapterConfig | None":
        """Factory. Shall return `None` if the adapter_config is not passed, or validate the schema.

        Args:
            run_config: is the main dict of a configuration run, see `api_dataclasses`.
        """
        # CAVEAT: adaptor will be bypassed alltogether in a rare case when streaming is requested.
        if run_config.get("target", {}).get("api_endpoint", {}).get("stream", False):
            return None

        adapter_config = run_config.get("target", {}).get("api_endpoint", {}).get("adapter_config")
        if not adapter_config:
            return None

        adapter_config["endpoint_type"] = run_config.get("target", {}).get("api_endpoint", {}).get("type", "")

        return AdapterConfig.model_validate(adapter_config)

    api_url: str = Field(
        description="The URL where the model endpoint is served",
    )

    local_port: Optional[int] = Field(
        description="Local port to use for the adapter server. If `None` (default) will choose any free port",
        default=None,
    )

    use_reasoning: bool = Field(
        description="Whether to use the clean-reasoning-tokens adapter. See `end_reasoning_token`.",
        default=False,
    )

    end_reasoning_token: str = Field(
        description="Token that singifies the end of reasoning output",
        default="</think>",
    )

    custom_system_prompt: Optional[str] = Field(
        description="A custom system prompt to replace original one (if not None).",
        default=None,
    )

    max_logged_responses: int | None = Field(
        description="Maximum number of responses to log. Set to 0 to disable. If None, all will be logged.",
        default=5,
    )

    max_logged_requests: int | None = Field(
        description="Maximum number of requests to log. Set to 0 to disable. If None, all will be logged.",
        default=5,
    )


class MisconfigurationError(Exception):
    """
    Exception raised when evaluation is not correctly configured.
    """

    pass
