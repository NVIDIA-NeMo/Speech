# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class ModelInfo:
    checkpoint: str
    model_type: Optional[str] = None
    min_tps: int = 1
    prompt_template: Tuple[str] = (
        "The capital of France is",
        "Largest animal in the sea is",
        "Fastest animal in the world is",
    )
    expected_keyword: Tuple[str] = (
        "Paris",
        "Whale",
        "Cheetah",
    )
    max_output_len: int = 128
    max_batch_size: int = 10
    p_tuning_checkpoint: Optional[str] = None
    lora_checkpoint: Optional[str] = None
    model_dir: str = "/tmp/trt_llm_model"


def get_infer_test_data():
    test_data = {}

    test_data["NEMOTRON3-22B-base-32k-v1"] = ModelInfo(
        checkpoint="/opt/checkpoints/nemotron-3-22b-chat-32k-sft_v1.0/nemotron-3-22b-chat-32k-sft.nemo",
        model_type="gptnext",
        min_tps=2,
    )

    test_data["NEMOTRON3-22B-base-32k-v2"] = ModelInfo(
        checkpoint="/opt/checkpoints/nemotron-3-22b-chat-32k-sft_v2.0/nemotron-3-22b-chat-32k-sft.nemo",
        model_type="gptnext",
        min_tps=2,
    )

    test_data["NEMOTRON3-22B-base-32k-v3"] = ModelInfo(
        checkpoint="/opt/checkpoints/nemotron-3-22b-chat-32k-steerlm_v1.0/nemotron-3-22b-chat-32k-steerlm.nemo",
        model_type="gptnext",
        min_tps=2,
    )

    test_data["NEMOTRON4-340B-base-fp8"] = ModelInfo(
        checkpoint="/opt/checkpoints/NEMOTRON4-340B-base-fp8/NEMOTRON4-340B-base-fp8-1-qnemo",
        min_tps=8,
    )

    test_data["LLAMA2-7B-base"] = ModelInfo(
        checkpoint="/opt/checkpoints/LLAMA2-7B-base/LLAMA2-7B-base-1.nemo",
        model_type="llama",
        p_tuning_checkpoint="/opt/checkpoints/LLAMA2-7B-PTuning/LLAMA2-7B-PTuning-1.nemo",
        lora_checkpoint="/opt/checkpoints/LLAMA2-7B-Lora/LLAMA2-7B-Lora-1.nemo",
    )

    test_data["LLAMA2-7B-fp8-sft"] = ModelInfo(
        checkpoint="/opt/checkpoints/LLAMA2-7B-base/LLAMA2-7B-fp8-sft.nemo",
        model_type="llama",
    )

    test_data["LLAMA2-13B-base"] = ModelInfo(
        checkpoint="/opt/checkpoints/LLAMA2-13B-base/LLAMA2-13B-base-1.nemo",
        model_type="llama",
        p_tuning_checkpoint="/opt/checkpoints/LLAMA2-13B-PTuning/LLAMA2-13B-PTuning-1.nemo",
    )

    test_data["LLAMA2-70B-base"] = ModelInfo(
        checkpoint="/opt/checkpoints/LLAMA2-70B-base/LLAMA2-70B-base-1.nemo",
        model_type="llama",
        min_tps=2,
    )

    test_data["LLAMA2-7B-code"] = ModelInfo(
        checkpoint="/opt/checkpoints/LLAMA2-7B-code/LLAMA2-7B-code-1.nemo",
        model_type="llama",
    )

    test_data["LLAMA2-7B-base-int4-awq"] = ModelInfo(
        checkpoint="/opt/checkpoints/LLAMA2-7B-base-int4-awq/LLAMA2-7B-base-int4-awq-1-qnemo",
    )

    test_data["LLAMA2-7B-base-int8-sq"] = ModelInfo(
        checkpoint="/opt/checkpoints/LLAMA2-7B-base-int8-sq/LLAMA2-7B-base-int8-sq-1-qnemo",
    )

    test_data["LLAMA2-13B-base-fp8"] = ModelInfo(
        checkpoint="/opt/checkpoints/LLAMA2-13B-base-fp8/LLAMA2-13B-base-fp8-1-qnemo",
    )

    test_data["LLAMA3-8B-base-fp8"] = ModelInfo(
        checkpoint="/opt/checkpoints/LLAMA3-8B-base-fp8/LLAMA3-8B-base-fp8-1-qnemo",  # TODO: compress
    )

    test_data["LLAMA3-70B-base-fp8"] = ModelInfo(
        checkpoint="/opt/checkpoints/LLAMA3-70B-base-fp8/LLAMA3-70B-base-fp8-1-qnemo",
        min_tps=8,
    )

    test_data["LLAMA2-70B-base-int4-awq"] = ModelInfo(
        checkpoint="/opt/checkpoints/LLAMA2-70B-base-int4-awq/LLAMA2-70B-base-int4-awq-1-qnemo",
        min_tps=4,
    )

    test_data["LLAMA2-70B-base-int8-sq"] = ModelInfo(
        checkpoint="/opt/checkpoints/LLAMA2-70B-base-int8-sq/LLAMA2-70B-base-int8-sq-1-qnemo",
        min_tps=2,
    )

    test_data["FALCON-7B-base"] = ModelInfo(
        checkpoint="/opt/checkpoints/FALCON-7B-base/FALCON-7B-base-1.nemo",
        model_type="falcon",
    )

    test_data["FALCON-40B-base"] = ModelInfo(
        checkpoint="/opt/checkpoints/FALCON-40B-base/FALCON-40B-base-1.nemo",
        model_type="falcon",
        min_tps=2,
    )

    test_data["FALCON-180B-base"] = ModelInfo(
        checkpoint="/opt/checkpoints/FALCON-180B-base/FALCON-180B-base-1.nemo",
        model_type="falcon",
        min_tps=8,
    )

    test_data["STARCODER1-15B-python"] = ModelInfo(
        checkpoint="/opt/checkpoints/STARCODER1-15B-python/STARCODER1-15B-python-1.nemo",
        model_type="starcoder",
        prompt_template=("def fibonnaci(n",),
        expected_keyword=("fibonnaci",),
        max_output_len=128,
        max_batch_size=5,
    )

    test_data["STARCODER2-15B-4k-vfinal"] = ModelInfo(
        checkpoint="/opt/checkpoints/STARCODER2-15B-4k-vfinal/STARCODER2-15B-4k-vfinal-1.nemo",
        model_type="starcoder",
        prompt_template=("def fibonnaci(n",),
        expected_keyword=("fibonnaci",),
        max_output_len=128,
        max_batch_size=5,
    )

    test_data["GEMMA-2B-base"] = ModelInfo(
        checkpoint="/opt/checkpoints/GEMMA-2B-base/GEMMA-2B-base-1.nemo",
        model_type="gemma",
    )

    return test_data
