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

import copy
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import nemo_run as run
import yaml
from nemo_run.core.serialization.yaml import YamlSerializer
from pytorch_lightning import Callback
from pytorch_lightning.loggers import WandbLogger

from nemo.lightning.pytorch.callbacks import NsysCallback, PreemptionCallback
from nemo.lightning.pytorch.strategies.megatron_strategy import MegatronStrategy
from nemo.utils import logging

# This file contains plugins based on NeMo-Run's run.Plugin API.
# Plugins operate both on a configured task and an executor at the same time, and are specific to NeMo-Run.
# If you are adding functionality that goes directly into the Pytorch Lightning trainer, you may consider adding a callback instead of a plugin.


def _merge_callbacks(partial: run.Partial, callbacks: list[run.Config[Callback]]):
    if hasattr(partial, "trainer"):
        if hasattr(partial.trainer, "callbacks") and partial.trainer.callbacks:
            for callback in callbacks:
                if callback not in partial.trainer.callbacks:
                    partial.trainer.callbacks.append(callback)
        else:
            partial.trainer.callbacks = copy.deepcopy(callbacks)


@dataclass(kw_only=True)
class PreemptionPlugin(run.Plugin):
    """
    A plugin for setting up Preemption callback and preemption signals.

    Args:
        preempt_time (int): The time, in seconds, before the task's time limit at which the executor
                             will send a SIGTERM preemption signal. This allows tasks to be gracefully
                             stopped before reaching their time limit, reducing waste and
                             promoting fair resource usage. The default value is 300 seconds (5 minutes).
                             This is only supported for ``run.SlurmExecutor``.
        callbacks (list[run.Config[Callback]]): A list of callback configurations that the plugin
                                                will merge with the task's existing callbacks.
                                                By default, the list includes NeMo's preemption callback.
    """

    preempt_time: int = 300
    callbacks: list[run.Config[Callback]] = field(default_factory=lambda: [run.Config(PreemptionCallback)])

    def setup(self, task: run.Partial | run.Script, executor: run.Executor):
        if isinstance(task, run.Script):
            logging.warning(
                f"The {self.__class__.__name__} will have no effect on the task as it's an instance of run.Script"
            )
            return

        if isinstance(executor, run.SlurmExecutor):
            # Sends a SIGTERM self.preempt_time seconds before hitting time limit
            logging.info(
                f"{self.__class__.__name__} will send a SIGTERM {self.preempt_time} seconds before the job's time limit for your Slurm executor."
            )
            executor.signal = f"TERM@{self.preempt_time}"

        _merge_callbacks(task, callbacks=self.callbacks)


@dataclass(kw_only=True)
class NsysPlugin(run.Plugin):
    """
    A plugin for nsys profiling.

    The NsysPlugin allows you to profile your run using nsys.
    You can specify when to start and end the profiling, on which ranks to run the profiling,
    and what to trace during profiling.

    Args:
        start_step (int): The step at which to start the nsys profiling.
        end_step (int): The step at which to end the nsys profiling.
        ranks (Optional[list[int]]): The ranks on which to run the nsys profiling. If not specified,
            profiling will be run on rank 0.
        nsys_trace (Optional[list[str]]): The events to trace during profiling. If not specified,
            'nvtx' and 'cuda' events will be traced.
    """

    start_step: int
    end_step: int
    ranks: Optional[list[int]] = None
    nsys_trace: Optional[list[str]] = None

    def setup(self, task: run.Partial | run.Script, executor: run.Executor):
        if isinstance(task, run.Partial):
            nsys_callback = run.Config(
                NsysCallback,
                start_step=self.start_step,
                end_step=self.end_step,
                ranks=self.ranks or [0],
            )
            callbacks: list[run.Config[Callback]] = [nsys_callback]  # type: ignore
            _merge_callbacks(task, callbacks=callbacks)

        launcher = executor.get_launcher()
        launcher.nsys_profile = True
        launcher.nsys_trace = self.nsys_trace or ["nvtx", "cuda"]


@dataclass(kw_only=True)
class WandbPlugin(run.Plugin):
    """
    A plugin for setting up Weights & Biases.

    This plugin sets a ``WandbLogger`` to ``NeMoLogger``'s ``wandb`` arg,
    which in turn initializes the Pytorch Lightning `WandbLogger <https://lightning.ai/docs/pytorch/stable/extensions/generated/lightning.pytorch.loggers.WandbLogger.html>`_.

    This plugin is only activated if the ``WANDB_API_KEY`` environment variable is set.
    The ``WANDB_API_KEY`` environment variables will also be set in the executor's environment variables.
    Follow https://docs.wandb.ai/quickstart to retrieve your ``WANDB_API_KEY``.

    If `log_task_config` is True, the plugin will log the task configuration as a config dictionary
    to the Weights and Biases logger.

    Args:
        name (str): The name for the Weights & Biases run.
        logger_fn (Callable[..., run.Config[WandbLogger]]): A callable that returns a Config of ``WandbLogger``
        log_task_config (bool, optional): Whether to log the task configuration to the logger.
            Defaults to True.

    Raises:
        logging.warning: If the task is an instance of `run.Script`, as the plugin has no effect on such tasks.
    """

    name: str
    logger_fn: Callable[..., run.Config[WandbLogger]]
    log_task_config: bool = True

    def setup(self, task: run.Partial | run.Script, executor: run.Executor):
        if isinstance(task, run.Script):
            logging.warning(
                f"The {self.__class__.__name__} will have no effect on the task as it's an instance of run.Script"
            )
            return

        if "WANDB_API_KEY" in os.environ:
            executor.env_vars["WANDB_API_KEY"] = os.environ["WANDB_API_KEY"]

            if hasattr(task, "log") and hasattr(task.log, "wandb"):
                task.log.wandb = self.logger_fn(name=self.name)
                if self.log_task_config:
                    partial_config = yaml.safe_load(YamlSerializer().serialize(task))
                    partial_config["experiment"] = {
                        "id": self.experiment_id,
                        "task_name": self.name,
                        "executor": executor.info(),
                        "remote_directory": (
                            os.path.join(executor.tunnel.job_dir, Path(executor.job_dir).name)
                            if isinstance(executor, run.SlurmExecutor)
                            else None
                        ),
                        "local_directory": executor.job_dir,
                    }
                    task.log.wandb.config = partial_config
        else:
            logging.warning(
                f"The {self.__class__.__name__} will have no effect as WANDB_API_KEY environment variable is not set."
            )


@dataclass(kw_only=True)
class ConfigValidationPlugin(run.Plugin):
    """
    A plugin for validating a NeMo task with its executor.

    This plugin is used to ensure that the NeMo environment, task, and executor meet certain criteria.
    The validation checks include preemption, checkpoint directory,
    serialization, and Weights and Biases (wandb) integration.

    Attributes:
        validate_preemption (bool): Whether to validate the preemption callback. If set to True, the plugin will
            assert that the task has a `PreemptionCallback`. Defaults to True.
        validate_checkpoint_dir (bool): Whether to validate the checkpoint directory. If set to True and the executor
            is a `SlurmExecutor`, the plugin will assert that the task's log directory exists in the mounts
            specified in the `SlurmExecutor`. Defaults to True.
        validate_serialization (bool): Whether to validate task serialization. If set to True, the plugin will
            assert that the task can be successfully serialized and deserialized using NeMo-Run's
            `ZlibJSONSerializer`. Defaults to True.
        validate_wandb (bool): Whether to validate Weights and Biases integration. If set to True, the plugin will
            assert that the executor's environment variables contain a `WANDB_API_KEY`
            and that NeMo Logger's `wandb` is set. Defaults to False.
        validate_nodes_and_devices (bool): Whether to validate the number of devices and nodes. If set to True, the plugin will assert that the task's
            trainer is configured to use the same number of nodes and devices as the executor. Defaults to True.
    """

    validate_preemption: bool = True
    validate_checkpoint_dir: bool = True
    validate_serialization: bool = True
    validate_wandb: bool = False
    validate_nodes_and_devices: bool = True

    def setup(self, task: run.Partial | run.Script, executor: run.Executor):
        assert isinstance(task, run.Partial)
        logging.info(f"Validating {task.__fn_or_cls__.__qualname__} and {executor.__class__.__qualname__}.")
        if self.validate_preemption:
            logging.info("Validating preemption callback")
            assert any(map(lambda callback: callback.__fn_or_cls__ == PreemptionCallback, task.trainer.callbacks))

        if self.validate_checkpoint_dir:
            if isinstance(executor, run.SlurmExecutor):
                mounts = executor.container_mounts + ["/nemo_run"]
                mounts = list(map(lambda m: m.split(":")[-1], mounts))
                logging.info(f"Validating checkpoint dir {task.log.log_dir} exists in {mounts}")
                assert task.log.log_dir
                assert any(map(lambda mount: Path(mount) in Path(task.log.log_dir).parents, mounts))

        if self.validate_serialization:
            from nemo_run.core.serialization.zlib_json import ZlibJSONSerializer

            logging.info("Validating serialization/de-serialization of task")
            serializer = ZlibJSONSerializer()
            assert serializer.deserialize(serializer.serialize(task)) == task

        if self.validate_wandb:
            logging.info("Validating that Weights and Biases is enabled for task")
            assert "WANDB_API_KEY" in executor.env_vars.keys()
            assert task.log.wandb

        if self.validate_nodes_and_devices:
            logging.info("Validating that nodes and devices match for task and executor")
            if isinstance(executor, run.SlurmExecutor):
                assert task.trainer.num_nodes == executor.nodes
                assert task.trainer.devices == executor.nproc_per_node()


@dataclass(kw_only=True)
class PerfEnvPlugin(run.Plugin):
    """
    A plugin for setting up performance optimized environments.

    Attributes:
        enable_layernorm_sm_margin (bool): Set SM margin for TransformerEngine's Layernorm, so
            in order to not block DP level communication overlap.
        layernorm_sm_margin (int): The SM margin for TransformerEngine Layernorm.
        enable_vboost (bool): Whether to steer more power towards tensor cores via
            `sudo nvidia-smi boost-slider --vboost 1`. May not work on all systems.
    """

    enable_layernorm_sm_margin: bool = True
    layernorm_sm_margin: int = 16
    enable_vboost: bool = False

    def get_vboost_srun_cmd(self, nodes, job_dir):
        import shlex

        vboost_cmd = " ".join(
            [
                "\n# Command 0: enable vboost\n\n",
                "srun",
                f"--ntasks={nodes}",
                "--output",
                os.path.join(job_dir, "vboost.out"),
                "--error",
                os.path.join(job_dir, "vboost.err"),
                "bash -c ",
                shlex.quote("sudo nvidia-smi boost-slider --vboost 1"),
            ],
        )

        return vboost_cmd

    def setup(self, task: run.Partial | run.Script, executor: run.Executor):

        if task.trainer.strategy.__fn_or_cls__ == MegatronStrategy:
            # Force program order kernel launch for TP, CP overlap
            tp_size = task.trainer.strategy.tensor_model_parallel_size
            cp_size = task.trainer.strategy.context_parallel_size
            if tp_size > 1 and cp_size > 1:
                executor.env_vars["CUDA_DEVICE_MAX_CONNECTIONS"] = 1

            # Set LayerNorm SM margin to support the overlap with LayerNorm kernel
            if self.enable_layernorm_sm_margin:
                executor.env_vars["NVTE_FWD_LAYERNORM_SM_MARGIN"] = self.layernorm_sm_margin
                executor.env_vars["NVTE_BWD_LAYERNORM_SM_MARGIN"] = self.layernorm_sm_margin

        # Force Transformer Engine to use cuDNN attention over HazyResearch's Flash Attention
        executor.env_vars["NVTE_FLASH_ATTN"] = 0
        executor.env_vars["NVTE_FUSED_ATTN"] = 1

        # Improve perf by steering power to tensor cores, may not work on all systems
        if self.enable_vboost and isinstance(executor, run.SlurmExecutor):
            vboost_cmd = self.get_vboost_srun_cmd(executor.nodes, executor.job_dir)
            executor.setup_lines = (
                executor.setup_lines + vboost_cmd
                if (executor.setup_lines and len(executor.setup_lines) > 0)
                else vboost_cmd
            )
