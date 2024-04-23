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

from nemo.export.trt_llm.qnemo import align_config
from nemo.export.trt_llm.tensorrt_llm_build import MODEL_NAME, get_engine_name

CONFIG_NAME = "config.json"
CONFIG_TRTLLM_BUILD_NAME = "config_trtllm_build.json"


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

    # Alignment to make nemo-fw tensorrt_llm.runtime ModelConfig definition compatible with config
    # produced by trtllm-build API. The new config is saved as "config.json" while the source build
    # config is saved as "config_trtllm_build.json" in the engine directory for reference.
    os.rename(os.path.join(engine_dir, CONFIG_NAME), os.path.join(engine_dir, CONFIG_TRTLLM_BUILD_NAME))
    with open(os.path.join(engine_dir, CONFIG_TRTLLM_BUILD_NAME), "r") as f:
        config_trtllm_build = json.load(f)

    config = align_config(config_trtllm_build)

    # Other parameters
    assert lora_target_modules is None
    config["builder_config"]["lora_target_modules"] = lora_target_modules

    with open(os.path.join(engine_dir, CONFIG_NAME), "w") as f:
        json.dump(config, f, indent=2)

    # Rename for consistency with how engine is run later
    for i in range(config["builder_config"]["world_size"]):
        os.rename(
            os.path.join(engine_dir, f"rank{i}.engine"),
            os.path.join(
                engine_dir,
                get_engine_name(
                    MODEL_NAME,
                    config["builder_config"]["precision"],
                    config["builder_config"]["tensor_parallel"],
                    config["builder_config"]["pipeline_parallel"],
                    i,
                ),
            ),
        )
