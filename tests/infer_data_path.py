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

import urllib.request as req
from pathlib import Path


def get_infer_test_data():
    test_data = {}

    test_data["NV-GPT-8B-Base-4k"] = {}
    test_data["NV-GPT-8B-Base-4k"]["model_type"] = "gptnext"
    test_data["NV-GPT-8B-Base-4k"]["min_gpus"] = 1
    test_data["NV-GPT-8B-Base-4k"]["location"] = "Local"
    test_data["NV-GPT-8B-Base-4k"]["trt_llm_model_dir"] = "/tmp/NV-GPT-8B-Base-4k/nv-gpt-8b-base-4k_v1.0/"
    test_data["NV-GPT-8B-Base-4k"][
        "checkpoint"
    ] = "/opt/checkpoints/NV-GPT-8B-Base-4k/nv-gpt-8b-base-4k_v1.0/NV-GPT-8B-Base-4k.nemo"
    test_data["NV-GPT-8B-Base-4k"]["p_tuning_checkpoint"] = "/opt/checkpoints/NV-GPT-8B-PTuning/nv-gpt-8B-ptuning.nemo"
    test_data["NV-GPT-8B-Base-4k"]["prompt_template"] = [
        "The capital of France is",
        "Largest animal in the sea is",
        "Fastest animal in the world is",
    ]
    test_data["NV-GPT-8B-Base-4k"]["expected_keyword"] = ["Paris", "Whale", "Cheetah"]
    test_data["NV-GPT-8B-Base-4k"]["max_output_token"] = 128
    test_data["NV-GPT-8B-Base-4k"]["max_batch_size"] = 10

    test_data["NV-GPT-8B-Base-16k"] = {}
    test_data["NV-GPT-8B-Base-16k"]["model_type"] = "gptnext"
    test_data["NV-GPT-8B-Base-16k"]["min_gpus"] = 1
    test_data["NV-GPT-8B-Base-16k"]["location"] = "Local"
    test_data["NV-GPT-8B-Base-16k"]["trt_llm_model_dir"] = "/tmp/NV-GPT-8B-Base-16k/nv-gpt-8b-base-16k_v1.0/"
    test_data["NV-GPT-8B-Base-16k"][
        "checkpoint"
    ] = "/opt/checkpoints/NV-GPT-8B-Base-16k/nv-gpt-8b-base-16k_v1.0/NV-GPT-8B-Base-16k.nemo"
    test_data["NV-GPT-8B-Base-16k"]["prompt_template"] = [
        "The capital of France is",
        "Largest animal in the sea is",
        "Fastest animal in the world is",
    ]
    test_data["NV-GPT-8B-Base-16k"]["expected_keyword"] = ["Paris", "Whale", "Cheetah"]
    test_data["NV-GPT-8B-Base-16k"]["max_output_token"] = 128
    test_data["NV-GPT-8B-Base-16k"]["max_batch_size"] = 20

    test_data["NV-GPT-8B-QA-4k"] = {}
    test_data["NV-GPT-8B-QA-4k"]["model_type"] = "gptnext"
    test_data["NV-GPT-8B-QA-4k"]["min_gpus"] = 1
    test_data["NV-GPT-8B-QA-4k"]["location"] = "Local"
    test_data["NV-GPT-8B-QA-4k"]["trt_llm_model_dir"] = "/tmp/NV-GPT-8B-QA-4k/nv-gpt-8b-qa-4k_v1.0/"
    test_data["NV-GPT-8B-QA-4k"][
        "checkpoint"
    ] = "/opt/checkpoints/NV-GPT-8B-QA-4k/nv-gpt-8b-qa-4k_v1.0/NV-GPT-8B-QA-4k.nemo"
    test_data["NV-GPT-8B-QA-4k"]["prompt_template"] = [
        "What is the capital of France?",
        "What is the largest animal in the sea?",
        "What is the fastest animal in the world?",
    ]
    test_data["NV-GPT-8B-QA-4k"]["expected_keyword"] = ["Paris", "Whale", "Cheetah"]
    test_data["NV-GPT-8B-QA-4k"]["max_output_token"] = 96
    test_data["NV-GPT-8B-QA-4k"]["max_batch_size"] = 20

    test_data["NV-GPT-8B-Chat-4k-SFT"] = {}
    test_data["NV-GPT-8B-Chat-4k-SFT"]["model_type"] = "gptnext"
    test_data["NV-GPT-8B-Chat-4k-SFT"]["min_gpus"] = 1
    test_data["NV-GPT-8B-Chat-4k-SFT"]["location"] = "Local"
    test_data["NV-GPT-8B-Chat-4k-SFT"]["trt_llm_model_dir"] = "/tmp/NV-GPT-8B-Chat-4k-SFT/nv-gpt-8b-chat-4k-sft_v1.0/"
    test_data["NV-GPT-8B-Chat-4k-SFT"][
        "checkpoint"
    ] = "/opt/checkpoints/NV-GPT-8B-Chat-4k-SFT/nv-gpt-8b-chat-4k-sft_v1.0/NV-GPT-8B-Chat-4k-SFT.nemo"
    test_data["NV-GPT-8B-Chat-4k-SFT"]["prompt_template"] = [
        "What is the capital of France?",
        "What is the largest animal in the sea?",
        "What is the fastest animal in the world?",
    ]
    test_data["NV-GPT-8B-Chat-4k-SFT"]["expected_keyword"] = ["Paris", "Whale", "Cheetah"]
    test_data["NV-GPT-8B-Chat-4k-SFT"]["max_output_token"] = 256
    test_data["NV-GPT-8B-Chat-4k-SFT"]["max_batch_size"] = 5

    test_data["NV-GPT-8B-Chat-4k-RLHF"] = {}
    test_data["NV-GPT-8B-Chat-4k-RLHF"]["model_type"] = "gptnext"
    test_data["NV-GPT-8B-Chat-4k-RLHF"]["min_gpus"] = 1
    test_data["NV-GPT-8B-Chat-4k-RLHF"]["location"] = "Local"
    test_data["NV-GPT-8B-Chat-4k-RLHF"][
        "trt_llm_model_dir"
    ] = "/tmp/NV-GPT-8B-Chat-4k-RLHF/nv-gpt-8b-chat-4k-rlhf_v1.0/"
    test_data["NV-GPT-8B-Chat-4k-RLHF"][
        "checkpoint"
    ] = "/opt/checkpoints/NV-GPT-8B-Chat-4k-RLHF/nv-gpt-8b-chat-4k-rlhf_v1.0/NV-GPT-8B-Chat-4k-RLHF.nemo"
    test_data["NV-GPT-8B-Chat-4k-RLHF"]["prompt_template"] = [
        "What is the capital of France?",
        "What is the largest animal in the sea?",
        "What is the fastest animal in the world?",
    ]
    test_data["NV-GPT-8B-Chat-4k-RLHF"]["expected_keyword"] = ["Paris", "Whale", "Cheetah"]
    test_data["NV-GPT-8B-Chat-4k-RLHF"]["max_output_token"] = 128
    test_data["NV-GPT-8B-Chat-4k-RLHF"]["max_batch_size"] = 10

    test_data["NV-GPT-8B-Chat-4k-SteerLM"] = {}
    test_data["NV-GPT-8B-Chat-4k-SteerLM"]["model_type"] = "gptnext"
    test_data["NV-GPT-8B-Chat-4k-SteerLM"]["min_gpus"] = 1
    test_data["NV-GPT-8B-Chat-4k-SteerLM"]["location"] = "Local"
    test_data["NV-GPT-8B-Chat-4k-SteerLM"][
        "trt_llm_model_dir"
    ] = "/tmp/NV-GPT-8B-Chat-4k-SteerLM/nv-gpt-8b-chat-4k-steerlm_v1.0/"
    test_data["NV-GPT-8B-Chat-4k-SteerLM"][
        "checkpoint"
    ] = "/opt/checkpoints/NV-GPT-8B-Chat-4k-SteerLM/nv-gpt-8b-chat-4k-steerlm_v1.0/NV-GPT-8B-Chat-4k-SteerLM.nemo"
    test_data["NV-GPT-8B-Chat-4k-SteerLM"]["prompt_template"] = [
        "What is the capital of France?",
        "What is the largest animal in the sea?",
        "What is the fastest animal in the world?",
    ]
    test_data["NV-GPT-8B-Chat-4k-SteerLM"]["expected_keyword"] = ["Paris", "Whale", "Cheetah"]
    test_data["NV-GPT-8B-Chat-4k-SteerLM"]["max_output_token"] = 128
    test_data["NV-GPT-8B-Chat-4k-SteerLM"]["max_batch_size"] = 10

    test_data["GPT-43B-Base"] = {}
    test_data["GPT-43B-Base"]["model_type"] = "gptnext"
    test_data["GPT-43B-Base"]["min_gpus"] = 2
    test_data["GPT-43B-Base"]["location"] = "Local"
    test_data["GPT-43B-Base"]["trt_llm_model_dir"] = "/tmp/GPT-43B-Base/gpt-43B-base/"
    test_data["GPT-43B-Base"]["checkpoint"] = "/opt/checkpoints/GPT-43B-Base/gpt-43B-base.nemo"
    test_data["GPT-43B-Base"]["prompt_template"] = [
        "The capital of France is",
        "Largest animal in the sea is",
        "Fastest animal in the world is",
    ]
    test_data["GPT-43B-Base"]["expected_keyword"] = ["Paris", "Whale", "Cheetah"]
    test_data["GPT-43B-Base"]["max_output_token"] = 128
    test_data["GPT-43B-Base"]["max_batch_size"] = 10

    test_data["LLAMA2-7B-base"] = {}
    test_data["LLAMA2-7B-base"]["model_type"] = "llama"
    test_data["LLAMA2-7B-base"]["min_gpus"] = 1
    test_data["LLAMA2-7B-base"]["location"] = "Local"
    test_data["LLAMA2-7B-base"]["trt_llm_model_dir"] = "/tmp/LLAMA2-7B-base/trt_llm_model-1/"
    test_data["LLAMA2-7B-base"]["checkpoint"] = "/opt/checkpoints/LLAMA2-7B-base/LLAMA2-7B-base-1.nemo"
    test_data["LLAMA2-7B-base"]["p_tuning_checkpoint"] = "/opt/checkpoints/LLAMA2-7B-PTuning/LLAMA2-7B-PTuning-1.nemo"
    test_data["LLAMA2-7B-base"]["lora_checkpoint"] = "/opt/checkpoints/LLAMA2-7B-Lora/LLAMA2-7B-Lora-1.nemo"
    test_data["LLAMA2-7B-base"]["prompt_template"] = [
        "The capital of France is",
        "Largest animal in the sea",
        "Fastest animal in the world",
    ]
    test_data["LLAMA2-7B-base"]["expected_keyword"] = ["Paris", "Whale", "Cheetah"]
    test_data["LLAMA2-7B-base"]["max_output_token"] = 128
    test_data["LLAMA2-7B-base"]["max_batch_size"] = 10

    test_data["LLAMA2-13B-base"] = {}
    test_data["LLAMA2-13B-base"]["model_type"] = "llama"
    test_data["LLAMA2-13B-base"]["min_gpus"] = 1
    test_data["LLAMA2-13B-base"]["location"] = "Local"
    test_data["LLAMA2-13B-base"]["trt_llm_model_dir"] = "/tmp/LLAMA2-13B-base/trt_llm_model-1/"
    test_data["LLAMA2-13B-base"]["checkpoint"] = "/opt/checkpoints/LLAMA2-13B-base/LLAMA2-13B-base-1.nemo"
    test_data["LLAMA2-13B-base"][
        "p_tuning_checkpoint"
    ] = "/opt/checkpoints/LLAMA2-13B-PTuning/LLAMA2-13B-PTuning-1.nemo"
    test_data["LLAMA2-13B-base"]["prompt_template"] = [
        "The capital of France is",
        "Largest animal in the sea is",
        "Fastest animal in the world is",
    ]
    test_data["LLAMA2-13B-base"]["expected_keyword"] = ["Paris", "Whale", "Cheetah"]
    test_data["LLAMA2-13B-base"]["max_output_token"] = 128
    test_data["LLAMA2-13B-base"]["max_batch_size"] = 10

    test_data["LLAMA2-70B-base"] = {}
    test_data["LLAMA2-70B-base"]["model_type"] = "llama"
    test_data["LLAMA2-70B-base"]["min_gpus"] = 2
    test_data["LLAMA2-70B-base"]["location"] = "Local"
    test_data["LLAMA2-70B-base"]["trt_llm_model_dir"] = "/tmp/LLAMA2-70B-base/trt_llm_model-1/"
    test_data["LLAMA2-70B-base"]["checkpoint"] = "/opt/checkpoints/LLAMA2-70B-base/LLAMA2-70B-base-1.nemo"
    test_data["LLAMA2-70B-base"]["prompt_template"] = [
        "The capital of France is",
        "Largest animal in the sea is",
        "Fastest animal in the world is",
    ]
    test_data["LLAMA2-70B-base"]["expected_keyword"] = ["Paris", "Whale", "Cheetah"]
    test_data["LLAMA2-70B-base"]["max_output_token"] = 128
    test_data["LLAMA2-70B-base"]["max_batch_size"] = 10

    test_data["LLAMA2-7B-code"] = {}
    test_data["LLAMA2-7B-code"]["model_type"] = "llama"
    test_data["LLAMA2-7B-code"]["min_gpus"] = 1
    test_data["LLAMA2-7B-code"]["location"] = "Local"
    test_data["LLAMA2-7B-code"]["trt_llm_model_dir"] = "/tmp/LLAMA2-7B-code/trt_llm_model-1/"
    test_data["LLAMA2-7B-code"]["checkpoint"] = "/opt/checkpoints/LLAMA2-7B-code/LLAMA2-7B-code-1.nemo"
    test_data["LLAMA2-7B-code"]["prompt_template"] = [
        "You are an expert programmer that writes simple, concise code and explanations. Write a python function to generate the nth fibonacci number."
    ]
    test_data["LLAMA2-7B-code"]["expected_keyword"] = ["Here"]
    test_data["LLAMA2-7B-code"]["max_output_token"] = 128
    test_data["LLAMA2-7B-code"]["max_batch_size"] = 10

    test_data["LLAMA2-7B-base-fp8"] = {}
    test_data["LLAMA2-7B-base-fp8"]["model_type"] = "llama"
    test_data["LLAMA2-7B-base-fp8"]["min_gpus"] = 1
    test_data["LLAMA2-7B-base-fp8"]["location"] = "Local"
    test_data["LLAMA2-7B-base-fp8"]["trt_llm_model_dir"] = "/tmp/LLAMA2-7B-base-fp8/trt_llm_model-1/"
    test_data["LLAMA2-7B-base-fp8"]["checkpoint"] = "/opt/checkpoints/LLAMA2-7B-base-fp8/LLAMA2-7B-base-fp8-1.qnemo"
    test_data["LLAMA2-7B-base-fp8"]["prompt_template"] = [
        "The capital of France is",
        "Largest animal in the sea is",
        "Fastest animal in the world is",
    ]
    test_data["LLAMA2-7B-base-fp8"]["expected_keyword"] = ["Paris", "Whale", "Cheetah"]
    test_data["LLAMA2-7B-base-fp8"]["max_output_token"] = 128
    test_data["LLAMA2-7B-base-fp8"]["max_batch_size"] = 10

    test_data["LLAMA2-7B-base-int4"] = {}
    test_data["LLAMA2-7B-base-int4"]["model_type"] = "llama"
    test_data["LLAMA2-7B-base-int4"]["min_gpus"] = 1
    test_data["LLAMA2-7B-base-int4"]["location"] = "Local"
    test_data["LLAMA2-7B-base-int4"]["trt_llm_model_dir"] = "/tmp/LLAMA2-7B-base-int4/trt_llm_model-1/"
    test_data["LLAMA2-7B-base-int4"]["checkpoint"] = "/opt/checkpoints/LLAMA2-7B-base-int4/LLAMA2-7B-base-int4-1.qnemo"
    test_data["LLAMA2-7B-base-int4"]["prompt_template"] = [
        "The capital of France is",
        "Largest animal in the sea is",
        "Fastest animal in the world is",
    ]
    test_data["LLAMA2-7B-base-int4"]["expected_keyword"] = ["Paris", "Whale", "Cheetah"]
    test_data["LLAMA2-7B-base-int4"]["max_output_token"] = 128
    test_data["LLAMA2-7B-base-int4"]["max_batch_size"] = 10

    test_data["LLAMA2-7B-base-int8"] = {}
    test_data["LLAMA2-7B-base-int8"]["model_type"] = "llama"
    test_data["LLAMA2-7B-base-int8"]["min_gpus"] = 1
    test_data["LLAMA2-7B-base-int8"]["location"] = "Local"
    test_data["LLAMA2-7B-base-int8"]["trt_llm_model_dir"] = "/tmp/LLAMA2-7B-base-int8/trt_llm_model-1/"
    test_data["LLAMA2-7B-base-int8"]["checkpoint"] = "/opt/checkpoints/LLAMA2-7B-base-int8/LLAMA2-7B-base-int8-1.qnemo"
    test_data["LLAMA2-7B-base-int8"]["prompt_template"] = [
        "The capital of France is",
        "Largest animal in the sea is",
        "Fastest animal in the world is",
    ]
    test_data["LLAMA2-7B-base-int8"]["expected_keyword"] = ["Paris", "Whale", "Cheetah"]
    test_data["LLAMA2-7B-base-int8"]["max_output_token"] = 128
    test_data["LLAMA2-7B-base-int8"]["max_batch_size"] = 10

    test_data["LLAMA2-13B-base-fp8"] = {}
    test_data["LLAMA2-13B-base-fp8"]["model_type"] = "llama"
    test_data["LLAMA2-13B-base-fp8"]["min_gpus"] = 2
    test_data["LLAMA2-13B-base-fp8"]["location"] = "Local"
    test_data["LLAMA2-13B-base-fp8"]["trt_llm_model_dir"] = "/tmp/LLAMA2-13B-base-fp8/trt_llm_model-1/"
    test_data["LLAMA2-13B-base-fp8"]["checkpoint"] = "/opt/checkpoints/LLAMA2-13B-base-fp8/LLAMA2-13B-base-fp8-1-qnemo"
    test_data["LLAMA2-13B-base-fp8"]["prompt_template"] = [
        "The capital of France is",
        "Largest animal in the sea is",
        "Fastest animal in the world is",
    ]
    test_data["LLAMA2-13B-base-fp8"]["expected_keyword"] = ["Paris", "Whale", "Cheetah"]
    test_data["LLAMA2-13B-base-fp8"]["max_output_token"] = 128
    test_data["LLAMA2-13B-base-fp8"]["max_batch_size"] = 10

    test_data["LLAMA2-13B-base-int4"] = {}
    test_data["LLAMA2-13B-base-int4"]["model_type"] = "llama"
    test_data["LLAMA2-13B-base-int4"]["min_gpus"] = 2
    test_data["LLAMA2-13B-base-int4"]["location"] = "Local"
    test_data["LLAMA2-13B-base-int4"]["trt_llm_model_dir"] = "/tmp/LLAMA2-13B-base-int4/trt_llm_model-1/"
    test_data["LLAMA2-13B-base-int4"][
        "checkpoint"
    ] = "/opt/checkpoints/LLAMA2-13B-base-int4/LLAMA2-13B-base-int4-1-qnemo"
    test_data["LLAMA2-13B-base-int4"]["prompt_template"] = [
        "The capital of France is",
        "Largest animal in the sea is",
        "Fastest animal in the world is",
    ]
    test_data["LLAMA2-13B-base-int4"]["expected_keyword"] = ["Paris", "Whale", "Cheetah"]
    test_data["LLAMA2-13B-base-int4"]["max_output_token"] = 128
    test_data["LLAMA2-13B-base-int4"]["max_batch_size"] = 10

    test_data["LLAMA2-70B-base-fp8"] = {}
    test_data["LLAMA2-70B-base-fp8"]["model_type"] = "llama"
    test_data["LLAMA2-70B-base-fp8"]["min_gpus"] = 8
    test_data["LLAMA2-70B-base-fp8"]["location"] = "Local"
    test_data["LLAMA2-70B-base-fp8"]["trt_llm_model_dir"] = "/tmp/LLAMA2-70B-base-fp8/trt_llm_model-1/"
    test_data["LLAMA2-70B-base-fp8"]["checkpoint"] = "/opt/checkpoints/LLAMA2-70B-base-fp8/LLAMA2-70B-base-fp8-1-qnemo"
    test_data["LLAMA2-70B-base-fp8"]["prompt_template"] = [
        "The capital of France is",
        "Largest animal in the sea is",
        "Fastest animal in the world is",
    ]
    test_data["LLAMA2-70B-base-fp8"]["expected_keyword"] = ["Paris", "Whale", "Cheetah"]
    test_data["LLAMA2-70B-base-fp8"]["max_output_token"] = 128
    test_data["LLAMA2-70B-base-fp8"]["max_batch_size"] = 10

    test_data["LLAMA2-70B-base-int4"] = {}
    test_data["LLAMA2-70B-base-int4"]["model_type"] = "llama"
    test_data["LLAMA2-70B-base-int4"]["min_gpus"] = 8
    test_data["LLAMA2-70B-base-int4"]["location"] = "Local"
    test_data["LLAMA2-70B-base-int4"]["trt_llm_model_dir"] = "/tmp/LLAMA2-70B-base-int4/trt_llm_model-1/"
    test_data["LLAMA2-70B-base-int4"][
        "checkpoint"
    ] = "/opt/checkpoints/LLAMA2-70B-base-int4/LLAMA2-70B-base-int4-1-qnemo"
    test_data["LLAMA2-70B-base-int4"]["prompt_template"] = [
        "The capital of France is",
        "Largest animal in the sea is",
        "Fastest animal in the world is",
    ]
    test_data["LLAMA2-70B-base-int4"]["expected_keyword"] = ["Paris", "Whale", "Cheetah"]
    test_data["LLAMA2-70B-base-int4"]["max_output_token"] = 128
    test_data["LLAMA2-70B-base-int4"]["max_batch_size"] = 10

    test_data["FALCON-7B-base"] = {}
    test_data["FALCON-7B-base"]["model_type"] = "falcon"
    test_data["FALCON-7B-base"]["min_gpus"] = 1
    test_data["FALCON-7B-base"]["location"] = "Local"
    test_data["FALCON-7B-base"]["trt_llm_model_dir"] = "/tmp/FALCON-7B-base/trt_llm_model-1/"
    test_data["FALCON-7B-base"]["checkpoint"] = "/opt/checkpoints/FALCON-7B-base/FALCON-7B-base-1.nemo"
    test_data["FALCON-7B-base"]["prompt_template"] = [
        "The capital of France is",
        "Largest animal in the sea is",
        "Fastest animal in the world is",
    ]
    test_data["FALCON-7B-base"]["expected_keyword"] = ["Paris", "Whale", "Cheetah"]
    test_data["FALCON-7B-base"]["max_output_token"] = 128
    test_data["FALCON-7B-base"]["max_batch_size"] = 10

    test_data["FALCON-40B-base"] = {}
    test_data["FALCON-40B-base"]["model_type"] = "falcon"
    test_data["FALCON-40B-base"]["min_gpus"] = 2
    test_data["FALCON-40B-base"]["location"] = "Local"
    test_data["FALCON-40B-base"]["trt_llm_model_dir"] = "/tmp/FALCON-40B-base/trt_llm_model-1/"
    test_data["FALCON-40B-base"]["checkpoint"] = "/opt/checkpoints/FALCON-40B-base/FALCON-40B-base-1.nemo"
    test_data["FALCON-40B-base"]["prompt_template"] = [
        "The capital of France is",
        "Largest animal in the sea is",
        "Fastest animal in the world is",
    ]
    test_data["FALCON-40B-base"]["expected_keyword"] = ["Paris", "Whale", "Cheetah"]
    test_data["FALCON-40B-base"]["max_output_token"] = 128
    test_data["FALCON-40B-base"]["max_batch_size"] = 10

    test_data["FALCON-180B-base"] = {}
    test_data["FALCON-180B-base"]["model_type"] = "falcon"
    test_data["FALCON-180B-base"]["min_gpus"] = 8
    test_data["FALCON-180B-base"]["location"] = "Local"
    test_data["FALCON-180B-base"]["trt_llm_model_dir"] = "/tmp/FALCON-180B-base/trt_llm_model-1/"
    test_data["FALCON-180B-base"]["checkpoint"] = "/opt/checkpoints/FALCON-180B-base/FALCON-180B-base-1.nemo"
    test_data["FALCON-180B-base"]["prompt_template"] = [
        "The capital of France is",
        "Largest animal in the sea is",
        "Fastest animal in the world is",
    ]
    test_data["FALCON-180B-base"]["expected_keyword"] = ["Paris", "Whale", "Cheetah"]
    test_data["FALCON-180B-base"]["max_output_token"] = 128
    test_data["FALCON-180B-base"]["max_batch_size"] = 10

    test_data["STARCODER1-15B-base"] = {}
    test_data["STARCODER1-15B-base"]["model_type"] = "starcoder"
    test_data["STARCODER1-15B-base"]["min_gpus"] = 1
    test_data["STARCODER1-15B-base"]["location"] = "Local"
    test_data["STARCODER1-15B-base"]["trt_llm_model_dir"] = "/tmp/STARCODER1-15B-base/trt_llm_model-1/"
    test_data["STARCODER1-15B-base"]["checkpoint"] = "/opt/checkpoints/STARCODER1-15B-base/STARCODER1-15B-base-1.nemo"
    test_data["STARCODER1-15B-base"]["prompt_template"] = ["def fibonnaci(n"]
    test_data["STARCODER1-15B-base"]["expected_keyword"] = ["fibonnaci"]
    test_data["STARCODER1-15B-base"]["max_output_token"] = 128
    test_data["STARCODER1-15B-base"]["max_batch_size"] = 5

    test_data["GEMMA-base"] = {}
    test_data["GEMMA-base"]["model_type"] = "gemma"
    test_data["GEMMA-base"]["min_gpus"] = 1
    test_data["GEMMA-base"]["location"] = "Local"
    test_data["GEMMA-base"]["trt_llm_model_dir"] = "/tmp/GEMMA-base/trt_llm_model-1/"
    test_data["GEMMA-base"]["checkpoint"] = "/opt/checkpoints/GEMMA-base/GEMMA-base-1.nemo"
    test_data["GEMMA-base"]["prompt_template"] = [
        "The capital of France is",
        "Largest animal in the sea is",
        "Fastest animal in the world is",
    ]
    test_data["GEMMA-base"]["expected_keyword"] = ["Paris", "Whale", "Cheetah"]
    test_data["GEMMA-base"]["max_output_token"] = 128
    test_data["GEMMA-base"]["max_batch_size"] = 10

    return test_data


def download_nemo_checkpoint(checkpoint_link, checkpoint_dir, checkpoint_path):
    if not Path(checkpoint_path).exists():
        print("Checkpoint: {0}, will be downloaded to {1}".format(checkpoint_link, checkpoint_path))
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        ckp_path = Path("/opt/checkpoints/")
        if not ckp_path.exists():
            ckp_path.mkdir(parents=True, exist_ok=False)
        req.urlretrieve(checkpoint_link, checkpoint_path)
        print("Checkpoint: {0}, download completed.".format(checkpoint_link))
    else:
        print("Checkpoint: {0}, has already been downloaded.".format(checkpoint_link))
