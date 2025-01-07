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

import os
import subprocess
from pathlib import Path

from nemo.utils import logging


def unset_environment_variables() -> None:
    """
    SLURM_, PMI_, PMIX_ Variables are needed to be unset for trtllm export to work
    on clusters. This method takes care of unsetting these env variables
    """
    logging.info("Unsetting all SLURM_, PMI_, PMIX_ Variables")

    # Function to unset variables with a specific prefix
    def unset_vars_with_prefix(prefix):
        unset_vars = []
        cmd = f"env | grep ^{prefix} | cut -d= -f1"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        vars_to_unset = result.stdout.strip().split('\n')
        for var in vars_to_unset:
            if var:  # Check if the variable name is not empty
                os.environ.pop(var, None)
                unset_vars.append(var)
        return unset_vars

    # Collect all unset variables across all prefixes
    all_unset_vars = []

    # Unset variables for each prefix
    for prefix in ['SLURM_', 'PMI_', 'PMIX_']:
        unset_vars = unset_vars_with_prefix(prefix)
        all_unset_vars.extend(unset_vars)

    if all_unset_vars:
        logging.info(f"Unset env variables: {', '.join(all_unset_vars)}")
    else:
        logging.info("No env variables were unset.")


def get_trtllm_deployable(
    nemo_checkpoint,
    model_type,
    triton_model_repository,
    num_gpus,
    tensor_parallelism_size,
    pipeline_parallelism_size,
    max_input_len,
    max_output_len,
    max_batch_size,
    dtype,
    output_generation_logits,
):
    """
    Exports the nemo checkpoint to trtllm and returns trt_llm_exporter that is used to deploy on PyTriton.
    """
    from nemo.export.tensorrt_llm import TensorRTLLM

    if triton_model_repository is None:
        trt_llm_path = "/tmp/trt_llm_model_dir/"
        Path(trt_llm_path).mkdir(parents=True, exist_ok=True)
    else:
        trt_llm_path = triton_model_repository

    if nemo_checkpoint is None and triton_model_repository is None:
        raise ValueError(
            "The provided model repository is not a valid TensorRT-LLM model "
            "directory. Please provide a --nemo_checkpoint or a TensorRT-LLM engine."
        )

    if nemo_checkpoint is None and not os.path.isdir(triton_model_repository):
        raise ValueError(
            "The provided model repository is not a valid TensorRT-LLM model "
            "directory. Please provide a --nemo_checkpoint or a valid TensorRT-LLM engine."
        )

    if nemo_checkpoint is not None and model_type is None:
        raise ValueError("Model type is required to be defined if a nemo checkpoint is provided.")

    trt_llm_exporter = TensorRTLLM(
        model_dir=trt_llm_path,
        load_model=(nemo_checkpoint is None),
    )

    if nemo_checkpoint is not None:
        try:
            logging.info("Export operation will be started to export the nemo checkpoint to TensorRT-LLM.")
            trt_llm_exporter.export(
                nemo_checkpoint_path=nemo_checkpoint,
                model_type=model_type,
                gpus_per_node=num_gpus,
                tensor_parallelism_size=tensor_parallelism_size,
                pipeline_parallelism_size=pipeline_parallelism_size,
                max_input_len=max_input_len,
                max_output_len=max_output_len,
                max_batch_size=max_batch_size,
                dtype=dtype,
                gather_generation_logits=output_generation_logits,
            )
        except Exception as error:
            raise RuntimeError("An error has occurred during the model export. Error message: " + str(error))

    return trt_llm_exporter
