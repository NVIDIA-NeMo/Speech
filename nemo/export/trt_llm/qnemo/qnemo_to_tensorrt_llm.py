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
import os
import subprocess

from typing import List, Optional

CONFIG_NAME = "config.json"


def qnemo_to_tensorrt_llm(
    nemo_checkpoint_path: str,
    engine_dir: str,
    max_input_len: int,
    max_output_len: int,
    max_batch_size: int,
    max_prompt_embedding_table_size: int,
    lora_target_modules: Optional[List[str]] = None,
):
    """Build TRT-LLM engine via trtllm-build CLI API in a subprocess."""
    assert not lora_target_modules, f"LoRA is not supported for quantized checkpoints, got {lora_target_modules}"
    print(
        "Note that setting n_gpus, tensor_parallel_size and pipeline_parallel_size parameters"
        " for quantized models is possible only on export step via nemo.export.quantize module."
        " These parameters are ignored when building and running TensorRT-LLM engine below."
    )
    # Load config to explicitly pass selected parameters to trtllm-build command:
    with open(os.path.join(nemo_checkpoint_path, CONFIG_NAME), "r") as f:
        model_config = json.load(f)
    command = [
        "trtllm-build",
        "--checkpoint_dir",
        nemo_checkpoint_path,
        "--output_dir",
        engine_dir,
        "--max_batch_size",
        str(max_batch_size),
        "--max_input_len",
        str(max_input_len),
        "--max_output_len",
        str(max_output_len),
        "--max_prompt_embedding_table_size",
        str(max_prompt_embedding_table_size),
        "--gemm_plugin",
        model_config["dtype"],
        "--gpt_attention_plugin",
        model_config["dtype"],
        "--strongly_typed",
        "--use_custom_all_reduce",
        "disable",
        "--workers",
        str(model_config["mapping"]["world_size"]),
    ]
    command_str = " ".join(command)
    print(f"Build command is:\n{command_str}")
    print("Running trtllm-build, this may take a while...")
    result = subprocess.run(command, capture_output=True)  # TODO: consider streaming logs
    if result.returncode != 0:
        print(result.stdout.decode())
        print(result.stderr.decode())
        raise RuntimeError("Error encountered for trtllm-build command, please check logs.")

    print("Building engine done. Full logs are:")
    print(result.stdout.decode())
