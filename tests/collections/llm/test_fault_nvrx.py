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

"""
Test fault tolerance with LLaMA3 recipe and a smaller model.
"""

import argparse
import os

import nemo_run as run

from lightning.pytorch.callbacks import Callback

from nemo.collections import llm
from nemo.collections.llm.recipes.callbacks.common import straggler_det_callback
from nemo.lightning.run.plugins import FaultTolerancePlugin
from nemo.utils.exp_manager import TimingCallback
from tests.collections.llm.common import small_llama_cfg, train_data


class CrashCallback(Callback):
    def __init__(self, crash_step=16):
        self.crash_step = crash_step
        self.current_step = 0
        print(f"Setup to simulate a crash if step == {self.crash_step}")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.current_step = self.current_step + 1
        if self.crash_step and self.current_step == self.crash_step:
            raise Exception(f"Simulating a crash at step {self.crash_step}!")


def get_args():
    parser = argparse.ArgumentParser(prog="", description="")
    parser.add_argument('--devices', type=int, required=True, help="Number of devices to use for training")
    parser.add_argument(
        '--crash-step',
        type=int,
        help="Step when a crash should be simulated",
    )
    parser.add_argument(
        '--check-report', type=bool, default=False, help="Check if StragglerDetection reports performance scores"
    )
    parser.add_argument(
        '--experiment-dir', type=str, required=True, help="directory to write results and checkpoints to"
    )
    parser.add_argument(
        '--data-path', type=str, default=None, help="Path to data file. If not specified, uses mock data."
    )
    parser.add_argument(
        '--tokenizer-path',
        type=str,
        default=None,
        help="Path to a sentencepiece tokenizer model file. If not specified, uses mock data.",
    )
    parser.add_argument('--index-mapping-dir', type=str, help="directory to write index mappings to")

    return parser.parse_args()


def main():
    args = get_args()

    exp_name = "L2_llama3_small_pretrain_fault_tolerance_test"
    pretrain_recipe = llm.llama3_8b.pretrain_recipe(
        dir=args.experiment_dir, name=exp_name, num_gpus_per_node=args.devices
    )

    pretrain_recipe.model = run.Config(llm.LlamaModel, small_llama_cfg(1024))

    if args.data_path and args.tokenizer_path:
        pretrain_recipe.data = train_data(
            data_path=args.data_path,
            tokenizer_path=args.tokenizer_path,
            index_mapping_dir=args.index_mapping_dir,
            seq_length=1024,
        )

    # Recipe Overrides
    pretrain_recipe.trainer.max_steps = 20
    pretrain_recipe.trainer.log_every_n_steps = 1
    # Enable ckpt save so that after the simulated crash, training can resume from ckpt
    pretrain_recipe.log.ckpt.every_n_train_steps = 10
    pretrain_recipe.log.ckpt.train_time_interval = None
    # Disable async ckpt because the simulated crash happens during ckpt save
    # So only an unfinished ckpt would be available for resume which can cause errors
    pretrain_recipe.trainer.strategy.ckpt_async_save = False
    pretrain_recipe.trainer.val_check_interval = 30
    pretrain_recipe.trainer.limit_val_batches = 2

    executor: run.SlurmExecutor = run.LocalExecutor(ntasks_per_node=args.devices, launcher="ft")
    # Add the fault tolerance plugin which enables restart after a crash
    run_plugins: list[run.Plugin] = [FaultTolerancePlugin(num_in_job_restarts=1, num_job_retries_on_failure=0)]
    pretrain_recipe.trainer.callbacks = [
        run.Config(TimingCallback),
        straggler_det_callback(straggler_report_time_interval=0.5),
    ]

    if args.crash_step:
        pretrain_recipe.trainer.callbacks.append(run.Config(CrashCallback, crash_step=args.crash_step))

    run.run(pretrain_recipe, plugins=run_plugins, executor=executor)

    # Assumes that NeMo logs are written into "run.log"
    # When a crash a simulated, error shows up on the terminal but it is not written to a file
    # So the test appends run output to run.log in the experiment-dir
    log_content = None
    with open(os.path.join(args.experiment_dir, "run.log")) as f:
        log_content = f.read()

    if args.check_report:
        assert "GPU relative performance" in log_content
        assert "GPU individual performance" in log_content
        assert "Straggler report processing time" in log_content
    if args.crash_step:
        assert f"Exception: Simulating a crash at step {args.crash_step}!" in log_content
        assert "Restored all states from the checkpoint" in log_content
        assert "`Trainer.fit` stopped: `max_steps=20` reached" in log_content


if __name__ == '__main__':
    main()
