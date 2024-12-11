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

import nemo_run as run

from nemo.collections import vlm


def configure_recipe(nodes: int = 1, gpus_per_node: int = 1):
    # pylint: disable=C0115,C0116
    recipe = vlm.mllama_11b.finetune_recipe(
        dir="/checkpoints/mllama",  # Path to store checkpoints
        name="mllama",
        num_nodes=nodes,
        num_gpus_per_node=gpus_per_node,
        peft_scheme="lora",
    )
    recipe.trainer.max_steps = 100
    recipe.trainer.val_check_interval = 100
    return recipe


def local_executor_torchrun(nodes: int = 1, devices: int = 1) -> run.LocalExecutor:
    # pylint: disable=C0115,C0116
    # Env vars for jobs are configured here
    env_vars = {}

    executor = run.LocalExecutor(ntasks_per_node=devices, launcher="torchrun", env_vars=env_vars)

    return executor


def run_training():
    # pylint: disable=C0115,C0116
    recipe = configure_recipe()
    executor = local_executor_torchrun(nodes=recipe.trainer.num_nodes, devices=recipe.trainer.devices)

    run.run(recipe, executor=executor)


# This condition is necessary for the script to be compatible with Python's multiprocessing module.
if __name__ == "__main__":
    run_training()
