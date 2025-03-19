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

import argparse
import os
from dataclasses import dataclass
from functools import partial

import fiddle as fdl
import nemo_run as run

from nemo.collections import llm
from nemo.collections.llm.gpt.model.llama import Llama3Config, LlamaModel
from nemo.collections.llm.tools.auto_configurator import AutoConfigurator, generate_configs, get_results


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_number", type=int, help="Number of config to run")
    parser.add_argument("--log_dir", type=str, help="Path where to save training logs")
    parser.add_argument("--get_results", action="store_true")

    return parser.parse_args()


@dataclass
class Llama3Config145M(Llama3Config):
    num_layers: int = 12
    hidden_size: int = 768
    num_attention_heads: int = 16
    num_query_groups: int = 8
    ffn_hidden_size: int = 2688


@run.cli.factory(target=llm.pretrain, name="llama3_145m")
def llama3_145m(num_nodes=1, num_gpus_per_node=1):
    # Setup Llama3 145M config
    recipe = partial(llm.llama3_8b.pretrain_recipe, num_nodes=num_nodes, num_gpus_per_node=num_gpus_per_node)()
    recipe.data.global_batch_size = 16
    recipe.data.seq_length = 2048

    recipe.trainer.strategy.context_parallel_size = 1
    recipe.model.config.seq_length = recipe.data.seq_length

    recipe = run.Partial(
        llm.pretrain,
        model=run.Config(LlamaModel, config=run.Config(Llama3Config145M)),
        trainer=recipe.trainer,
        data=recipe.data,
        log=recipe.log,
        optim=recipe.optim,
        resume=None,
    )

    return recipe


def train_config(args):
    # Llama3 145M
    # This example will generate 3 configs.
    # It is expected that this script will be run 3 times with changing --run_number flag for each run from 1 to 3.
    # After all configurations are trained, please trigger the script using --get_results flag.

    # Get Auto Conf runner
    runner = AutoConfigurator(
        recipe=partial(llama3_145m)(),
        gpu_memory_gb=40,
        tensor_parallel_sizes=[1],
        pipeline_parallel_sizes=[1],
        micro_batch_sizes=[1, 2, 4],
        max_training_days=1,
        max_steps_per_run=10,
        num_tokens_in_b=10,
        vocab_size=32000,
        path_to_logs=args.log_dir,
    )

    base_cfg, configs = generate_configs(runner)
    if not args.get_results:
        # Get generated configs
        partials = list(configs.values())
        names = list(configs.keys())

        # Run pre-training
        pretrain_cfg = partials[args.run_number - 1]  # partial(llama3_145m)() #
        pretrain = fdl.build(pretrain_cfg)
        pretrain()
    else:
        # # Get Auto Configurator results
        get_results(base_cfg, runner, args.log_dir, log_file_prefix="nemo_error")
        print(f"The results were successfully saved to {args.log_dir}.")


def main():
    args = get_args()
    train_config(args)


if __name__ == '__main__':
    main()
