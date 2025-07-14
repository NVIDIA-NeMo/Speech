# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
from lightning.pytorch import Callback
from lightning.pytorch.loggers import WandbLogger
from nemo_run.core.serialization.yaml import YamlSerializer

from nemo.lightning.pytorch.callbacks import MemoryProfileCallback, NsysCallback, PreemptionCallback
from nemo.lightning.pytorch.strategies.megatron_strategy import MegatronStrategy
from nemo.utils import logging
from nemo.utils.import_utils import safe_import

res_module, HAVE_RES = safe_import('nvidia_resiliency_ext.ptl_resiliency')

# This file contains plugins based on NeMo-Run's run.Plugin API.
# Plugins operate both on a configured task and an executor at the same time, and are specific to NeMo-Run.
# If you are adding functionality that goes directly into the Pytorch Lightning trainer,
# you may consider adding a callback instead of a plugin.


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
                             promoting fair resource usage. The default value is 60 seconds (1 minute).
                             This is only supported for ``run.SlurmExecutor``.
        callbacks (list[run.Config[Callback]]): A list of callback configurations that the plugin
                                                will merge with the task's existing callbacks.
                                                By default, the list includes NeMo's preemption callback.
    """

    preempt_time: int = 60
    callbacks: list[run.Config[Callback]] = field(default_factory=lambda: [run.Config(PreemptionCallback)])

    def setup(self, task: run.Partial | run.Script, executor: run.Executor):
        """Set up the preemption plugin."""
        if isinstance(task, run.Script):
            logging.warning(
                f"The {self.__class__.__name__} will have no effect on the task as it's an instance of run.Script"
            )
            return

        if isinstance(executor, run.SlurmExecutor):
            # Sends a SIGTERM self.preempt_time seconds before hitting time limit
            logging.info(
                f"{self.__class__.__name__} will send a SIGTERM {self.preempt_time} seconds before the job's time limit for your Slurm executor."  # pylint: disable=C0301
            )
            executor.signal = f"TERM@{self.preempt_time}"

        _merge_callbacks(task, callbacks=self.callbacks)


@dataclass(kw_only=True)
class FaultTolerancePlugin(run.Plugin):
    """
    A plugin for setting up the fault tolerance callback from nvidia-resiliency-ext.
    This plugin enables workload hang detection, automatic calculation of timeouts used for hang detection,
    detection of rank(s) terminated due to an error and workload respawning in case of a failure.
    Note: FaultTolerancePlugin does not work with the NsysPlugin.
    Args:
        num_in_job_restarts (int): Max number of restarts on failure, within the same job. Default is 3.
        num_job_retries_on_failure (int): Max number of new job restarts on failure. Default is 2.
        initial_rank_heartbeat_timeout (int): Timeouts are time intervals used by a rank monitor to detect
            that a rank is not alive. This is the max timeout for the initial heartbeat. Default is 1800.
        rank_heartbeat_timeout (int): This is the timeout for subsequent hearbeats after the initial heartbeat.
            Default is 300.
    """

    num_in_job_restarts: int = 3
    num_job_retries_on_failure: int = 2
    initial_rank_heartbeat_timeout: int = 1800
    rank_heartbeat_timeout: int = 300

    def setup(self, task: run.Partial | run.Script, executor: run.Executor):
        """Set up the fault tolerance plugin."""
        assert HAVE_RES, "nvidia-resiliency-ext.ptl_resiliency is required to use the FaultTolerancePlugin."

        executor.launcher = run.FaultTolerance(
            max_restarts=self.num_in_job_restarts,
            initial_rank_heartbeat_timeout=self.initial_rank_heartbeat_timeout,
            rank_heartbeat_timeout=self.rank_heartbeat_timeout,
        )
        executor.retries = self.num_job_retries_on_failure

        assert isinstance(task, run.Partial)

        callbacks = [
            run.Config(
                res_module.FaultToleranceCallback, autoresume=True, calculate_timeouts=True, exp_dir=task.log.log_dir
            )
        ]

        assert not executor.launcher.nsys_profile, "Nsys not supported with the FaultTolerancePlugin."
        if hasattr(task, "trainer") and hasattr(task.trainer, "callbacks"):
            assert all(
                map(
                    lambda cb: not cb.__fn_or_cls__ == NsysCallback if "__fn_or_cls__" in dir(cb) else True,
                    task.trainer.callbacks,
                )
            ), "Nsys not supported with FaultTolerancePlugin."

        _merge_callbacks(task, callbacks=callbacks)


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
    gen_shape: bool = False

    def setup(self, task: run.Partial | run.Script, executor: run.Executor):
        """Set up the nsys profiling plugin."""
        if isinstance(task, run.Partial):
            nsys_callback = run.Config(
                NsysCallback,
                start_step=self.start_step,
                end_step=self.end_step,
                ranks=self.ranks or [0],
                gen_shape=self.gen_shape,
            )
            callbacks: list[run.Config[Callback]] = [nsys_callback]  # type: ignore
            _merge_callbacks(task, callbacks=callbacks)

        launcher = executor.get_launcher()
        launcher.nsys_profile = True
        launcher.nsys_trace = self.nsys_trace or ["nvtx", "cuda"]
        if isinstance(executor, run.SlurmExecutor):
            # NOTE: DO NOT change to f-string, `%q{}` is Slurm placeholder
            launcher.nsys_filename = "profile_%p_%q{SLURM_JOB_ID}_node%q{SLURM_NODEID}_rank%q{SLURM_PROCID}"


@dataclass(kw_only=True)
class MemoryProfilePlugin(run.Plugin):
    """
    A plugin for memory profiling.

    The MemoryProfilePlugin allows you to profile a timeline of memory allocations during you run.
    The memory profiling plugin creates snapshots during the entire training. You can specify
    which ranks to run the profiling.

    Args:
        dir (str): Directory to store the memory profile dump .pickle files
        ranks (Optional[list[int]]): The ranks on which to run the memory profiling. If not specified,
            profiling will be run on rank 0.
    """

    dir: str
    ranks: Optional[list[int]] = None

    def setup(self, task: run.Partial | run.Script, executor: run.Executor):
        """Set up the memory profiling plugin."""
        if isinstance(task, run.Partial):
            memprof_callback = run.Config(
                MemoryProfileCallback,
                dir=self.dir,
                ranks=self.ranks or [0],
            )
            callbacks: list[run.Config[Callback]] = [memprof_callback]  # type: ignore
            _merge_callbacks(task, callbacks=callbacks)


@dataclass(kw_only=True)
class WandbPlugin(run.Plugin):
    """
    A plugin for setting up Weights & Biases.

    This plugin sets a ``WandbLogger`` to ``NeMoLogger``'s ``wandb`` arg,
    which in turn initializes the Pytorch Lightning `WandbLogger
    <https://lightning.ai/docs/pytorch/stable/extensions/generated/lightning.pytorch.loggers.WandbLogger.html>`_.

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
        """Set up the wandb plugin."""
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
        validate_nodes_and_devices (bool): Whether to validate the number of devices and nodes. If set to True,
            the plugin will assert that the task's trainer is configured to use the same number of nodes and devices
            as the executor. Defaults to True.
    """

    validate_preemption: bool = True
    validate_checkpoint_dir: bool = True
    validate_serialization: bool = True
    validate_wandb: bool = False
    validate_nodes_and_devices: bool = True

    def setup(self, task: run.Partial | run.Script, executor: run.Executor):
        """Set up the plugin to configure validation."""
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
    nccl_pp_comm_chunksize: Optional[int] = None
    gpu_sm100_or_newer: bool = False
    user_buffer_registration: bool = False

    def get_vboost_srun_cmd(self, nodes, job_dir):
        "Create the vboost `sudo nvidia-smi boost-slider --vboost 1` command"

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
        """Enable the performance environment settings"""

        if task.trainer.strategy.__fn_or_cls__ == MegatronStrategy:
            # Force program order kernel launch for TP, CP overlap
            tp_size = task.trainer.strategy.tensor_model_parallel_size
            cp_size = task.trainer.strategy.context_parallel_size
            if self.gpu_sm100_or_newer and (tp_size > 1 or cp_size > 1):
                executor.env_vars["CUDA_DEVICE_MAX_CONNECTIONS"] = "32"
            elif (not self.gpu_sm100_or_newer) and (tp_size > 1 or cp_size > 1):
                executor.env_vars["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"

            # Set LayerNorm SM margin to support the overlap with LayerNorm kernel
            if self.enable_layernorm_sm_margin:
                executor.env_vars["NVTE_FWD_LAYERNORM_SM_MARGIN"] = str(self.layernorm_sm_margin)
                executor.env_vars["NVTE_BWD_LAYERNORM_SM_MARGIN"] = str(self.layernorm_sm_margin)

            # Set the chunk size of P2P communications. Using a large chunk size reduces the
            # buffering overhead from the communication kernel execution time
            pp_size = task.trainer.strategy.pipeline_model_parallel_size
            if pp_size > 1 and self.nccl_pp_comm_chunksize is not None:
                assert isinstance(self.nccl_pp_comm_chunksize, int) and self.nccl_pp_comm_chunksize > 1
                executor.env_vars["NCCL_P2P_NET_CHUNKSIZE"] = str(self.nccl_pp_comm_chunksize)

            # Enable high priority for NCCL communications
            executor.env_vars["TORCH_NCCL_HIGH_PRIORITY"] = "1"

            if self.user_buffer_registration:
                # Enable NCCL NVLS ALGO, which could increase GPU memory usage
                executor.env_vars["NCCL_NVLS_ENABLE"] = "1"
                # This option makes NCCL to prefer SM efficient ALGOS if available
                # With this option, NCCL will use NVLS if user buffer is registered
                executor.env_vars["NCCL_CTA_POLICY"] = "1"
                if "PYTORCH_CUDA_ALLOC_CONF" in executor.env_vars:
                    pytorch_cuda_alloc_conf = executor.env_vars["PYTORCH_CUDA_ALLOC_CONF"].split(',')
                    if "expandable_segments:True" in pytorch_cuda_alloc_conf:
                        logging.warning(
                            "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True is not currently compatible with"
                            "user buffer registration. Removing expandable_segments:True from the list."
                        )
                        pytorch_cuda_alloc_conf.remove("expandable_segments:True")
                        executor.env_vars["PYTORCH_CUDA_ALLOC_CONF"] = ",".join(pytorch_cuda_alloc_conf)

        # Improve perf by steering power to tensor cores, may not work on all systems
        if self.enable_vboost and isinstance(executor, run.SlurmExecutor):
            vboost_cmd = self.get_vboost_srun_cmd(executor.nodes, executor.tunnel.job_dir)
            executor.setup_lines = (
                executor.setup_lines + vboost_cmd
                if (executor.setup_lines and len(executor.setup_lines) > 0)
                else vboost_cmd
            )


@dataclass(kw_only=True)
class TritonCacheSetup(run.Plugin):
    """
    A plugin for setting up Triton cache environment variables.
    This should not be neccessay for Triton 3.2.0 and above.
    """

    from nemo.core.utils.optional_libs import TRITON_AVAILABLE

    if TRITON_AVAILABLE:
        from triton import __version__ as triton_version

        if triton_version <= "3.1.0":

            def setup(self, task: run.Partial | run.Script, executor: run.Executor):
                """Set up the Triton cache environment variables."""
                executor.env_vars["TRITON_CACHE_DIR"] = executor.job_dir + "triton_cahce"
                executor.env_vars["TRITON_CACHE_MANAGER"] = (
                    "megatron.core.ssm.triton_cache_manager:ParallelFileCacheManager"
                )
