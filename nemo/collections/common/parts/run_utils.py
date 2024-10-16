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
import shlex
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import nemo_run as run
from nemo_run.config import NEMORUN_HOME
from nemo_run.core.execution.docker import DockerExecutor
from nemo_run.core.execution.slurm import SlurmJobDetails
from nemo_run.core.serialization.zlib_json import ZlibJSONSerializer
from nemo_run.core.tunnel import LocalTunnel, SSHTunnel
from omegaconf import DictConfig, OmegaConf

from nemo.utils import logging


@lru_cache(maxsize=2)
def get_tunnel(**ssh_tunnel):
    return SSHTunnel(**ssh_tunnel)


def get_mounts_from_config(cluster_config: dict, env_vars: dict = None):
    """
    Determines if there are mount paths that are being passed via environment variables.
    Selects the key in the cluster config called `mounts` which is a list of strings.
    Each string is in the format of `<str | {env_var}>:<str | {env_var}>` where `env_var`
    is the name of the environment variable.

    Args:
        cluster_config (dict): cluster config dictionary
        env_vars (dict): dictionary of environment variables

    Returns:
        list: updated list of mounts
    """
    mounts = cluster_config.get('mounts', [])

    # if there are env_mounts, we will add the mounts from the env_mounts
    for mount_id in range(len(mounts)):
        mount = mounts[mount_id]

        if ":" not in mount:
            raise ValueError(f"Invalid mount format: {mount}. The mount path must be separated by a colon.")

        mount_source, mount_target = mount.split(":")

        if mount_source[0] == "{" and mount_source[-1] == "}":
            # Resolve the environment variable for the mount source
            mount_source = mount_source[1:-1]

            if mount_source not in os.environ:
                raise ValueError(
                    f"Required environment variable {mount_source} not found in env variables passed in cluster configs."
                )

            mount_source = os.environ[mount_source]

        if mount_target[0] == "{" and mount_target[-1] == "}":
            # Resolve the environment variable for the mount target
            mount_target = mount_target[1:-1]

            if mount_target not in os.environ:
                raise ValueError(
                    f"Required environment variable {mount_target} not found in env variables passed in cluster configs."
                )

            mount_target = os.environ[mount_target]

        # add the mount to the list of mounts
        resolved_mount = f"{mount_source}:{mount_target}"
        mounts[mount_id] = resolved_mount

    return mounts


def check_if_mounted(cluster_config, path_to_check):
    """Will check that path_to_check is referenced inside one of the mounts."""
    for mount in get_mounts_from_config(cluster_config) + ['/nemo_run/code:/nemo_run/code']:
        if path_to_check.startswith(mount.split(":")[1]):
            return
    raise ValueError(f"The path '{path_to_check}' is not mounted. Check cluster config and add appropriate mounts.")


def add_mount_path(mount_source: str, mount_dest: str, cluster_config):
    """Add a mount path to the cluster configuration."""

    if cluster_config is None:
        raise ValueError("Cluster config is not provided.")

    if 'mounts' in cluster_config:
        original_mounts = get_mounts_from_config(cluster_config)
        added_mount = False
        for mount_path in original_mounts:
            source, destination = mount_path.split(':')

            if source == mount_source and destination == mount_dest:
                return

        if not added_mount:
            cluster_config['mounts'].append(f"{mount_source}:{mount_dest}")
            logging.info(f"Added mount path: `{mount_source}:{mount_dest}`")

    else:
        raise ValueError("No mounts found in cluster config, can only add to existing mount list.")


def create_remote_directory(directory: str | list, cluster_config: dict):
    """Create a remote directory on the cluster."""

    if cluster_config is None:
        raise ValueError("Cluster config is not provided.")

    if isinstance(directory, str):
        directory = [directory]

    if cluster_config.get('executor') == 'local':
        tunnel = LocalTunnel(job_dir=directory[0])
        for dir_path in directory:
            tunnel.run(f'mkdir -p {dir_path}', hide=False, warn=True)
            logging.info(f"Created directory: {dir_path} in local filesystem.")

        # tunnel.cleanup()

    elif cluster_config.get('executor') == 'slurm':
        ssh_tunnel_config = cluster_config.get('ssh_tunnel', None)
        if ssh_tunnel_config is None:
            raise ValueError("`ssh_tunnel` sub-config is not provided in cluster_config.")

        # Check for pre-existing job_dir in the ssh_tunnel_config
        if 'job_dir' not in ssh_tunnel_config:
            ssh_tunnel_config['job_dir'] = directory[0]

        tunnel = get_tunnel(**cluster_config['ssh_tunnel'])
        for dir_path in directory:
            tunnel.run(f'mkdir -p {dir_path}', hide=False, warn=True)
            logging.info(f"Created directory: {dir_path} on remote cluster.")
        # tunnel.cleanup()

    else:
        raise ValueError(f"Unsupported executor: {cluster_config.get('executor')}")


def create_remote_config(config: dict | DictConfig, config_name: str, config_directory: str, cluster_config: dict):
    """Create a remote yaml config at the result directory on the cluster."""

    if cluster_config is None:
        raise ValueError("Cluster config is not provided.")

    if not config_name.endswith('.yaml'):
        config_name = f"{config_name}.yaml"

    if isinstance(config_directory, str):
        config_directory = [config_directory]

    if cluster_config.get('executor') == 'local':
        tunnel = LocalTunnel(job_dir=config_directory[0])
        for dir_path in config_directory:
            config_filepath = os.path.join(dir_path, config_name)
            tunnel.run(f'mkdir -p {dir_path}', hide=False, warn=True)
            tunnel.run(f"touch {config_filepath}", hide=False, warn=True)
            tunnel.run(f"echo '{OmegaConf.to_yaml(config)}' > {config_filepath}", hide=False, warn=True)
            logging.info(f"Created config file: {dir_path} in local filesystem.")

        # tunnel.cleanup()

    elif cluster_config.get('executor') == 'slurm':
        ssh_tunnel_config = cluster_config.get('ssh_tunnel', None)
        if ssh_tunnel_config is None:
            raise ValueError("`ssh_tunnel` sub-config is not provided in cluster_config.")

        # Check for pre-existing job_dir in the ssh_tunnel_config
        if 'job_dir' not in ssh_tunnel_config:
            ssh_tunnel_config['job_dir'] = config_directory[0]

        tunnel = get_tunnel(**cluster_config['ssh_tunnel'])
        for dir_path in config_directory:
            config_filepath = os.path.join(dir_path, config_name)
            tunnel.run(f'mkdir -p {dir_path}', hide=False, warn=True)
            tunnel.run(f"touch {config_filepath}", hide=False, warn=True)
            tunnel.run(f"echo '{OmegaConf.to_yaml(config)}' > {config_filepath}", hide=False, warn=True)
            logging.info(f"Created config file: {dir_path} on remote cluster.")
        # tunnel.cleanup()

    else:
        raise ValueError(f"Unsupported executor: {cluster_config.get('executor')}")


def check_remote_mount_directories(directories: list, cluster_config: dict, exit_on_failure: bool = True):
    """Create a remote directory on the cluster."""

    if cluster_config is None:
        raise ValueError("Cluster config is not provided.")

    if isinstance(directories, str):
        directories = [directories]

    if cluster_config.get('executor') == 'local':
        tunnel = LocalTunnel(job_dir=None)

        all_dirs_exist = True
        missing_source_locations = []
        for directory in directories:
            result = tunnel.run(f'test -e {directory} && echo "Directory Exists"', hide=True, warn=True)

            if "Directory Exists" not in result.stdout:
                missing_source_locations.append(directory)

        # tunnel.cleanup()

        if len(missing_source_locations) > 0 and exit_on_failure:
            missing_source_locations = [
                f"{loc} DOES NOT exist at source destination" for loc in missing_source_locations
            ]
            missing_source_locations = "\n".join(missing_source_locations)
            raise FileNotFoundError(
                f"Some files or directories do not exist at the source location for mounting !!\n\n"
                f"{missing_source_locations}"
            )

    elif cluster_config.get('executor') == 'slurm':
        ssh_tunnel_config = cluster_config.get('ssh_tunnel', None)
        if ssh_tunnel_config is None:
            raise ValueError("`ssh_tunnel` sub-config is not provided in cluster_config.")

        # Check for pre-existing job_dir in the ssh_tunnel_config
        if 'job_dir' not in ssh_tunnel_config:
            ssh_tunnel_config['job_dir'] = os.getcwd()

        tunnel = get_tunnel(**cluster_config['ssh_tunnel'])
        missing_source_locations = []

        for directory in directories:
            result = tunnel.run(f'test -e {directory} && echo "Directory Exists"', hide=True, warn=True)

            if "Directory Exists" not in result.stdout:
                missing_source_locations.append(directory)

        # tunnel.cleanup()

        if len(missing_source_locations) > 0 and exit_on_failure:
            missing_source_locations = [
                f"{loc} DOES NOT exist at source destination" for loc in missing_source_locations
            ]
            missing_source_locations = "\n".join(missing_source_locations)
            raise FileNotFoundError(
                f"Some files or directories do not exist at the source location for mounting !!\n\n"
                f"{missing_source_locations}"
            )

    else:
        raise ValueError(f"Unsupported executor: {cluster_config.get('executor')}")


def get_unmounted_filepath(cluster_config: dict, filepath: str):
    """
    Resolve the mounted filepath using the cluster config to merge the mount source path to the filepath.
    Args:
        filepath:
        cluster_config:

    Returns:

    """
    # Find which mount path matches the filepaths prefix
    mount_path = None
    for mount in cluster_config['mounts']:
        mount_source, mount_dest = mount.split(':')
        if filepath.startswith(mount_dest):
            mount_path = mount
            break

    if mount_path is None:
        raise ValueError(
            f"Could not find a mount path for the file path `{filepath}`. Below paths are mounted: \n"
            f"{cluster_config['mounts']}"
        )

    # replace the mount destination inside the filepath with the mount source
    mount_source, mount_dest = mount_path.split(':')
    filepath = mount_source + filepath[len(mount_dest) :]  # replace the mount destination with the mount source

    return filepath


def get_mounted_filepath(cluster_config: dict, filepath: str):
    """
    Resolve the mounted filepath using the cluster config to merge the mount destination path to the filepath.

    Args:
        cluster_config:
        filepath:

    Returns:

    """
    # Find which mount path matches the filepaths prefix
    mount_path = None
    for mount in cluster_config['mounts']:
        mount_source, mount_dest = mount.split(':')
        if filepath.startswith(mount_source):
            mount_path = mount
            break

    if mount_path is None:
        raise ValueError(
            f"Could not find a mount path for the file path `{filepath}`. Below paths are mounted: \n"
            f"{cluster_config['mounts']}"
        )

    # replace the mount destination inside the filepath with the mount source
    mount_source, mount_dest = mount_path.split(':')
    filepath = mount_dest + filepath[len(mount_source) :]  # replace the mount destination with the mount source

    return filepath


def get_env_variables(cluster_config):
    """
    Will get the environment variables from the cluster config and the user environment.

    The following items in the cluster config are supported:
    - `required_env_vars` - list of required environment variables
    - `env_vars` - list of optional environment variables

    Args:
        cluster_config: cluster config dictionary

    Returns:
        dict: dictionary of environment
    """
    env_vars = {}
    # Check for user requested env variables
    required_env_vars = cluster_config.get("required_env_vars", [])
    for env_var in required_env_vars:
        if env_var not in os.environ:
            raise ValueError(f"Required environment variable {env_var} not found.")

        env_vars[env_var] = os.environ[env_var]
        logging.info(f"Adding required environment variable {env_var} (value={os.environ[env_var]})")

    # Add optional env variables
    optional_env_vars = cluster_config.get("env_vars", [])
    for env_var in optional_env_vars:
        if env_var in os.environ:
            logging.info(f"Adding optional environment variable {env_var} (value={os.environ[env_var]})")
            env_vars[env_var] = os.environ[env_var]
        else:
            logging.info(f"Optional environment variable {env_var} not found in user environment; skipping.")

    return env_vars


def _get_latest_dir(path, expname, job_id) -> str:
    if job_id is not None:
        return os.path.join(path, f"{expname}_{job_id}")

    dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    latest_dir = max(dirs, key=lambda d: os.path.getctime(os.path.join(path, d)))
    return os.path.join(path, latest_dir)


def get_exp_handles(expname):
    # TODO: remove this after we can properly use .from_title api
    if "_" in expname:
        try:
            job_id = int(expname.split("_")[-1])
            expname = expname[: expname.rfind("_")]
        except:
            job_id = None

    parent_dir = os.path.join(NEMORUN_HOME, "experiments", expname)
    exp_dir = _get_latest_dir(parent_dir, expname, job_id)

    with open(os.path.join(exp_dir, '_TASKS')) as f:
        serialized_jobs = json.load(f)

    serializer = ZlibJSONSerializer()
    handles = []
    for job in serialized_jobs:
        obj = serializer.deserialize(job[0])
        if hasattr(obj, 'handle'):
            handles.append(obj.handle)
        elif hasattr(obj, 'handles'):
            handles.extend(obj.handles)
        else:
            raise ValueError(f"Object {obj} does not have a handle or handles attribute.")
    return handles


@dataclass(kw_only=True)
class CustomJobDetails(SlurmJobDetails):
    log_prefix: str = "main"

    @property
    def stdout(self) -> Path:
        return Path(self.folder) / f"{self.log_prefix}_sbatch.log"

    @property
    def srun_stdout(self) -> Path:
        return Path(self.folder) / f"{self.log_prefix}_srun.log"

    @property
    def stderr(self) -> Path:
        return Path(self.folder) / f"{self.log_prefix}_sbatch.log"

    @property
    def srun_stderr(self) -> Path:
        return Path(self.folder) / f"{self.log_prefix}_srun.log"

    @property
    def ls_term(self) -> str:
        """This term will be used to fetch the logs.

        The command used to list the files is ls -1 {ls_term} 2> /dev/null
        """
        assert self.folder
        return os.path.join(self.folder, "*_srun.log")


def get_packager():
    """Will check if we are running from a git repo and use git packager or default packager otherwise."""
    return run.GitArchivePackager(
        check_uncommitted_changes=True,
    )


def get_executor(
    cluster_config,
    container,
    num_nodes,
    tasks_per_node,
    gpus_per_node,
    job_name,
    log_dir,
    log_prefix: str = "main",
    mounts=None,
    partition=None,
    dependencies=None,
):
    env_vars = get_env_variables(cluster_config)
    config_mounts = get_mounts_from_config(cluster_config, env_vars)

    mounts = mounts or config_mounts
    packager = get_packager()
    if cluster_config["executor"] == "local":
        if num_nodes > 1:
            raise ValueError("Local executor does not support multi-node execution")

        env_vars["PYTHONUNBUFFERED"] = "1"  # this makes sure logs are streamed right away

        return DockerExecutor(
            container_image=container,
            packager=packager,
            ipc_mode="host",
            volumes=mounts,
            ntasks_per_node=1,
            num_gpus=gpus_per_node,
            network="host",
            env_vars=env_vars,
        )

    partition = partition or cluster_config.get("partition")
    if 'timeouts' not in cluster_config:
        timeout = "10000:00:00:00"
    else:
        timeout = cluster_config["timeouts"][partition]

    return run.SlurmExecutor(
        account=cluster_config["account"],
        partition=partition,
        nodes=num_nodes,
        ntasks_per_node=tasks_per_node,
        tunnel=get_tunnel(**cluster_config["ssh_tunnel"]),
        container_image=container,
        container_mounts=mounts,
        time=timeout,
        packager=packager,
        gpus_per_node=gpus_per_node if not cluster_config.get("disable_gpus_per_node", False) else None,
        srun_args=[
            "--no-container-mount-home",
            "--overlap",
            "--mpi=pmix",
            '--wait=10',
            # we need to be explicit about this in srun as commands might need to run in parallel
            f"--ntasks={tasks_per_node * num_nodes}",
            f"--nodes={num_nodes}",
        ],
        # TODO: can we relax this to allow partial node allocation?
        exclusive=True,
        mem=0,
        job_details=CustomJobDetails(
            job_name=cluster_config.get("job_name_prefix", "") + job_name,
            folder=get_unmounted_filepath(cluster_config, log_dir),
            log_prefix=log_prefix + '_' + job_name,
        ),
        wait_time_for_group_job=0.01,
        monitor_group_job_wait_time=20,
        dependencies=dependencies,
        dependency_type="afterany",
        env_vars=env_vars,
    )


def add_task(
    exp,
    cmd,
    task_name,
    cluster_config,
    container,
    # TODO: these are good defaults for generation jobs, but probably not the best overall?
    num_tasks=1,
    num_gpus=1,
    num_nodes=1,
    log_dir=None,
    partition=None,
    run_after=None,
):
    if run_after is not None and cluster_config["executor"] == "slurm":
        dependencies = tuple(get_exp_handles(run_after))
    else:
        dependencies = None
    commands = []
    executors = []
    # then goes the main task unless it's empty
    if cmd:
        if cluster_config["executor"] == "local" and num_tasks > 1:
            cmd = f"mpirun --allow-run-as-root -np {num_tasks} bash -c {shlex.quote(cmd)}"
        commands.append(cmd)
        executors.append(
            get_executor(
                cluster_config=cluster_config,
                container=container,
                num_nodes=num_nodes,
                tasks_per_node=num_tasks,
                gpus_per_node=num_gpus,
                partition=partition,
                dependencies=dependencies,
                job_name=task_name,
                log_dir=log_dir,
                log_prefix="main",
            )
        )

    if len(commands) == 1:
        # to keep sbatch script simpler, we don't wrap in a list in this case
        exp.add(run.Script(inline=commands[0]), executor=executors[0], name="nemo-run")
    else:
        exp.add(
            [run.Script(inline=command) for command in commands],
            executor=executors,
            name="nemo-run",
        )


def run_exp(exp, cluster_config, sequential=False):
    if cluster_config['executor'] == 'local':
        # locally we are always running sequentially - does that need to be changed?
        exp.run(detach=False, tail_logs=True, sequential=True)
    else:
        exp.run(detach=True, sequential=sequential)
