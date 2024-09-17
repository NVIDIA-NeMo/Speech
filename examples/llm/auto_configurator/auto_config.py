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

import fiddle as fdl
import nemo_run as run

from nemo.collections.llm import GPTConfig126M
from nemo.collections.llm.tools.auto_configurator import AutoConfigurator, generate_configs, get_results


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_number", type=int, help="Number of config to run")
    parser.add_argument("--logs_dir", type=str, help="Path where to save training logs")
    parser.add_argument("--data_path", type=str, help="Path to the dataset")
    parser.add_argument("--tokenizer_path", type=str, help="Path to the tokenizer")
    parser.add_argument("--get_results", action="store_true")

    return parser.parse_args()


def train_config(args):
    # GPT-3 126M
    # This example will generate 3 configs.
    # It is expected that this script will be run 3 times with changing --run_number flag for each run from 0 to 2.
    # After all configurations are trained, please trigger the script using --get_results flag.
    runner = AutoConfigurator(
        model=run.Config(GPTConfig126M),
        num_nodes=1,
        gpus_per_node=1,
        gpu_memory_gb=40,
        global_batch_size=16,
        seq_length=512,
        tensor_parallel_sizes=[1],
        pipeline_parallel_sizes=[1],
        micro_batch_sizes=[1, 2, 4],
        max_training_days=1,
        max_steps_per_run=25,
        num_tokens_in_b=10,
        vocab_size=51200,
        tokenizer_type="autotokenizer",
        tokenizer_path=args.tokenizer_path,
        data_paths=args.data_path,
        path_to_logs=args.logs_dir,
    )

    base_cfg, configs = generate_configs(runner)
    if not args.get_results:
        # Get generated configs
        partials = list(configs.values())
        names = list(configs.keys())

        # Run pre-training
        partial = partials[args.run_number - 1]
        partial.log.log_dir = os.path.join(args.logs_dir, names[args.run_number - 1])
        pretrain = fdl.build(partial)
        pretrain()
    else:
        # # Get Auto Configurator results
        get_results(base_cfg, runner, args.logs_dir)
        print(f"The results were successfully saved to {args.logs_dir}.")


def main():
    args = get_args()
    train_config(args)


if __name__ == '__main__':
    main()
