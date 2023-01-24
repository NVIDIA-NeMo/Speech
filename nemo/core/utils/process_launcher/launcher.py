# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
import hashlib
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import torch
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from hydra.core.plugins import Plugins
from hydra.core.singleton import Singleton
from hydra.core.utils import JobReturn, JobStatus, configure_log, filter_overrides, setup_globals
from hydra.plugins.launcher import Launcher
from hydra.types import HydraContext, TaskFunction
from omegaconf import DictConfig, OmegaConf, open_dict

from nemo.utils import logging


# monkey-patch hydra func
def is_in_toplevel_plugins_module(*args, **kwargs) -> bool:
    return True


# Monkey-patch Hydra
Plugins.instance().is_in_toplevel_plugins_module = is_in_toplevel_plugins_module


@dataclass
class ProcessLauncherConfig:
    _target_: str = "nemo.core.utils.process_launcher.launcher.ProcessLauncher"
    num_gpus: int = -1
    jobs_per_gpu: int = 1


def execute_job(
    idx: int,
    overrides: Sequence[str],
    hydra_context: HydraContext,
    config: DictConfig,
    singleton_state: Dict[Any, Any],
    gpu_idx: int,
):
    """
    Creates a process that launches a "single" job that is identical in config + updated with sweep hyperparams.
    Since a different process is being used, CUDA can work in non-ddp mode without issue.
    Attempting ddp when using this script will not work as ddp cannot be used in shared contexts.

    Args:
        idx: Global index of the job.
        overrides: List of str overrides that correspond to this job
        hydra_context: Hydra Context used to load the sweep params into the global config
        config: Global config that will be updated with sweep hyper parameters.
        singleton_state: Hydra state.
        gpu_idx: The GPU ID on which this process will be run.

    Returns:
        - The Process object that corresponds to this sweep
        - The JobReturn object holding some metadata about this run
    """
    # Required by Hydra (lookup other Hydra Launchers for details)
    setup_globals()
    Singleton.set_state(singleton_state)

    # Update base config with overrides to create sweep config
    sweep_config = hydra_context.config_loader.load_sweep_config(config, list(overrides))
    with open_dict(sweep_config):
        sweep_config.hydra.job.id = "{}_{}".format(sweep_config.hydra.job.name, idx)
        sweep_config.hydra.job.num = idx
    HydraConfig.instance().set_config(sweep_config)

    # Setup a directory where the config will temporarily be stored.
    script_path = os.path.join(os.getcwd(), sys.argv[0])
    script_path = os.path.abspath(script_path)

    hash_salt = "|".join([script_path, str(OmegaConf.to_yaml(config))]).encode('utf-8')
    hash_val = hashlib.sha256(hash_salt).hexdigest()

    config_dir = os.path.join(os.getcwd(), "hydra_cfg", str(hash_val))
    if not os.path.exists(config_dir):
        os.makedirs(config_dir, exist_ok=True)

    task_cfg = copy.deepcopy(sweep_config)

    # Remove hydra from sweep config
    # This is done to prevent recursive call to multirun launcher in Hydra.
    with open_dict(task_cfg):
        task_cfg.pop('hydra', '')

    # Save the current jobs config to directory
    temp_config_name = f"config_{idx}.yaml"
    temp_config = os.path.join(config_dir, temp_config_name)
    OmegaConf.save(task_cfg, temp_config)

    # Compute the overides as a dict
    overrides = OmegaConf.to_container(config.hydra.overrides.task)

    # Check and replace trainer.devices in config with gpu_idx
    found_devices = False
    gpu_override = f'trainer.devices=[{gpu_idx}]'
    for oidx, val in enumerate(overrides):
        if 'trainer.devices' in val:
            overrides[oidx] = gpu_override
            found_devices = True

    if not found_devices:
        overrides.append(gpu_override)

    # Build launch command
    # Note: We depend on PTL doing the right thing since this command has global visibility of all CUDA_VISIBLE_DEVICES
    cmd = [
        'python',
        script_path,
        "--config-path",
        config_dir,
        "--config-name",
        temp_config_name,
        *overrides,
    ]

    # Launch the subprocess; pipe the stderr
    # NOTE: If this hangs due to some reason after prolonged training, it means that the stderr pipe buffer
    # has become full at the OS level and we need to explicitly empty it (either parallel thread or manually
    # call proc.communicate(). It should not happen in general case as stderr is filled only in case retcode != 0
    # If it does happen though, implement the code here
    # https://stackoverflow.com/questions/39607172/python-subprocess-popen-poll-seems-to-hang-but-communicate-works
    proc = subprocess.Popen(cmd, stderr=subprocess.PIPE)

    # Setup data thread for stderr
    std_error_buffer = []
    # Trivial thread just reads lines from stdout into the list
    drainerthread = threading.Thread(target=std_error_buffer.extend, args=(proc.stderr,))
    drainerthread.daemon = True
    drainerthread.start()

    # Construct JobReturn object for Hydra
    res = JobReturn()
    res.cfg = task_cfg
    res.overrides = overrides
    res.hydra_cfg = config
    res.working_dir = os.getcwd()
    res.return_value = None

    return proc, res, (std_error_buffer, drainerthread)


def launch(launcher, job_overrides: Sequence[Sequence[str]], initial_job_idx: int,) -> Sequence[JobReturn]:
    """
    Args:
        launcher: Reference to the Launched subclass
        job_overrides: A List of List<String>, where each inner list is the arguments for one job run
        initial_job_idx: Initial job idx in batch

    Returns:
        A list of JobReturn objects.
    """
    # Needed for Hydra, lookup JoblibLauncher in Hydra
    setup_globals()
    assert launcher.config is not None
    assert launcher.task_function is not None
    assert launcher.hydra_context is not None

    configure_log(launcher.config.hydra.hydra_logging, launcher.config.hydra.verbose)
    sweep_dir = Path(str(launcher.config.hydra.sweep.dir))
    sweep_dir.mkdir(parents=True, exist_ok=True)

    # Extraact the runner's config (its actually a DictConfig, but type is used for autocomplete)
    runner_cfg = launcher.runner  # type: ProcessLauncherConfig

    logging.info(
        "ProcessLauncher({}) is launching {} jobs".format(
            ",".join([f"{k}={v}" for k, v in runner_cfg.items()]), len(job_overrides),
        )
    )
    logging.info("Launching jobs, sweep output dir : {}".format(sweep_dir))
    for idx, overrides in enumerate(job_overrides):
        logging.info("\t#{} : {}".format(idx, " ".join(filter_overrides(overrides))))

    # Needed by Hydra
    singleton_state = Singleton.get_state()

    # Process the runner's config to build up the multiplex config
    num_gpus = runner_cfg.get('num_gpus', -1)
    jobs_per_gpu = runner_cfg.get('jobs_per_gpu', 1)

    # Only GPUs are supported for now.
    if num_gpus <= 0:
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
        else:
            raise ValueError(f"{launcher.__class__.__name__} only supports GPU operations.")

    # Setup arguments for multiplex runner
    overrides = list(job_overrides)
    num_overrides = len(overrides)

    job_idx = 0
    batch_size = num_gpus * jobs_per_gpu
    gpu_idx = 0

    ret = []  # List of returned JobResult
    subprocess_list = []  # Buffer of subprocess
    results = []  # Buffer of JobResult

    # STD ERROR cache
    std_error_buffers = []  # type: List[List[str]]
    std_error_threads = []  # type: threading.Thread

    # Run over all job combinations
    while job_idx < num_overrides:
        # Fill up subprocess buffer while its size is smaller than multiplex batch size
        while len(subprocess_list) < batch_size:
            # If we run out of jobs, stop trying to submit more jobs
            if job_idx >= num_overrides:
                break

            # Submit a job as a new process
            process, res, error_tup = execute_job(
                initial_job_idx + job_idx,
                overrides[job_idx],
                launcher.hydra_context,
                launcher.config,
                singleton_state,
                gpu_idx % num_gpus,  # This will evenly distribute GPU load
            )

            # Store the subprocesses and JobResults
            subprocess_list.append(process)
            results.append(res)

            # Manage stderror thread data
            std_error_buffers.append(error_tup[0])
            std_error_threads.append(error_tup[1])

            job_idx += 1
            gpu_idx += 1

        # Poll for samples in batch to finish.
        if len(subprocess_list) > 0:
            finished_processes = [0] * len(subprocess_list)

            # Check if all processes are completed or not
            # This is busy waiting, this is actually quite necessary
            # Turns out that when you do proc.communicate(), you block all other threads immediately.
            # IE they may fill up their buffers entirely, and hang while they wait for the first thread
            # who called communicate() to finish its work or crash.
            # Effectively it entirely stops multiprocessing jobs or multiplexed runs.
            # Must poll and busy wait to keep threads alive, along with drain the pipes with thread buffers.
            while sum(finished_processes) < len(subprocess_list):
                # Check all processes to make sure they have a retcode (doesnt matter yet if 0 or not)
                for proc_idx, proc in enumerate(subprocess_list):
                    # poll() is cheaper op than communicate()
                    retcode = proc.poll()

                    if retcode is not None:
                        # Log that the process with some ID has finished
                        if finished_processes[proc_idx] == 0:
                            logging.info(f"Processed job : {len(ret) + proc_idx} :: Ret code = {retcode}")

                        finished_processes[proc_idx] = 1

                        # Join this thread and merge its stderror buffer
                        proc.wait()
                        std_error_threads[proc_idx].join()
                        error_data = std_error_buffers[proc_idx]
                        error_data = [
                            str(data, encoding='utf-8').encode('utf-8').decode('utf-8').encode('utf-8')
                            for data in error_data
                        ]

                        std_error_buffers[proc_idx] = error_data

                time.sleep(1.0)

            # Process all the subprocess results
            for proc_idx, (proc, res) in enumerate(zip(subprocess_list, results)):
                # Wait until completion of process
                output, error = proc.communicate()

                # 0 is for successful run
                if proc.returncode == 0:
                    res.status = JobStatus.COMPLETED
                else:
                    # > 0 is for error, log the error.
                    # Note: For the sake of efficiency while we log the error and raise an exception,
                    # It will only raise the 1st wrong job in all the jobs.
                    # If multiple jobs fail, it will still try to execute every job first before
                    # raising the error for the first one.
                    # This is done so that even if some jobs fail (say OOM or something),
                    # other jobs can still run.
                    err_buffer = std_error_buffers[proc_idx]
                    if isinstance(err_buffer, (list, tuple)):
                        err_string = ""
                        for err_line in err_buffer:
                            err_string = (
                                err_string + f"{str(err_line, encoding='utf-8').encode('utf-8').decode('utf-8')}"
                            )
                    else:
                        err_string = err_buffer

                    error_msg = (
                        f"\nHyperparameter Arguments : {proc.args}\n"
                        f"Process Return code : {proc.returncode}\n"
                        f"Error Trace :\n"
                        f"{err_string}"
                    )
                    res.return_value = Exception(error_msg)
                    res.status = JobStatus.FAILED

                logging.info(f"Finished executing job : {len(ret)}. Return Code = {proc.returncode}")
                ret.append(res)

            # Reset for next batch
            subprocess_list.clear()
            results.clear()

    return ret


class ProcessLauncher(Launcher):
    def __init__(self, **kwargs: Any) -> None:
        """Process Launcher
        Based on the JoblibLauncher, but uses processes to scatter jobs in a multiplexed manner across
        some number of GPUs on a single machine.
        """
        self.config: Optional[DictConfig] = None
        self.task_function: Optional[TaskFunction] = None
        self.hydra_context: Optional[HydraContext] = None

        self.runner = kwargs  # type: ProcessLauncherConfig

    def setup(self, *, hydra_context: HydraContext, task_function: TaskFunction, config: DictConfig,) -> None:
        self.config = config
        self.task_function = task_function
        self.hydra_context = hydra_context

    def launch(self, job_overrides: Sequence[Sequence[str]], initial_job_idx: int) -> Sequence[JobReturn]:

        return launch(launcher=self, job_overrides=job_overrides, initial_job_idx=initial_job_idx)


ConfigStore.instance().store(
    group="hydra/launcher", name="nemo_launcher", node=ProcessLauncherConfig, provider="nemo_process_launcher",
)

Plugins.instance().register(ProcessLauncher)
