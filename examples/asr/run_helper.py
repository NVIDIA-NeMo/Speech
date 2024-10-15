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
from pathlib import Path

import nemo_run as run
from omegaconf import OmegaConf, open_dict

from nemo.collections.common.parts import run_utils
from nemo.core.config import hydra_runner
from nemo.utils import logging


NEMO_ROOT = Path(__file__).absolute().parents[2]


def gather_mounts(cluster_cfg):
    # Gather all mounts from the cluster config including ones which are disjoint from the cluster_cfg.mounts list.
    mounts = cluster_cfg.get('mounts', [])

    # Resolve any mounts in th cluster config that need user expansion
    mounts = [os.path.expanduser(m) for m in mounts]

    keys = list(cluster_cfg.keys())
    with open_dict(cluster_cfg):
        for k in keys:
            if k.startswith("mount_"):
                logging.info(f"Found additional mount flag in the cluster config `{k}`. Adding it to the mounts list.")
                mounts.append(cluster_cfg[k])
                del cluster_cfg[k]

        cluster_cfg['mounts'] = mounts
        logging.info(f"Final Mounts: {mounts}")


def check_root_path(path, nemo_root):
    path = str(path)
    nemo_root = str(nemo_root)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Path {path} does not exist.")

    if not path.startswith(nemo_root):
        raise ValueError(f"Path {path} is not in the NeMo root directory.")

    new_path = path.replace(nemo_root, '/nemo_run/code/')
    return new_path


def merge_configs(script_config, run_config):
    script_config = OmegaConf.load(script_config)
    original_script_keys = set(script_config.keys())
    result = OmegaConf.merge(script_config, run_config)

    # delete cluster config keys from the merged config
    with open_dict(result):
        for k in run_config.keys():
            if k in result and k not in original_script_keys:
                del result[k]

    # Check for any ??? missing values in result recursively and raise an error if found
    def check_missing_values(cfg):
        if hasattr(cfg, 'items'):
            for k, v in cfg.items():
                if hasattr(v, 'items'):
                    check_missing_values(v)
                elif v == '???':
                    raise ValueError(f"Missing value for key {k} in the config file")

    check_missing_values(result)
    return result

def check_config_mount_paths(script_config, cluster_config):
    # recursively walk all values of the script_config, checking if its a path-like string and if so, check if the path is a mounted path
    # if it is not, raise an error

    def check_mounted_path(cfg, cluster_cfg):
        if hasattr(cfg, 'items'):
            for k, v in cfg.items():
                if hasattr(v, 'items'):
                    check_mounted_path(v, cluster_cfg)
                elif isinstance(v, str):
                    if v.startswith(os.path.sep):
                        run_utils.check_if_mounted(cluster_cfg, v)

    check_mounted_path(script_config, cluster_config)


def get_execution_script(cluster_script_path, config_name):
    # Create the command to run the script
    cmd = """
nvidia-smi && \
export PYTHONPATH=$PYTHONPATH:/nemo_run/code && \
export HF_TOKEN={HF_TOKEN} && \
export WANDB_API_KEY={WANDB} && \
cd {cluster_script_dir} && \
python {cluster_script_path} --config-path "/results" --config-name "{config_name}" && \
cd /results && \
ls -l;
    """
    wandb_key = os.environ.get("WANDB", os.environ.get("WANDB_API_KEY", os.environ.get("WANDB_KEY", "")))
    format_dict = dict(
        cluster_script_dir=os.path.dirname(cluster_script_path),
        cluster_script_path=os.path.basename(cluster_script_path),
        config_name=config_name,
        HF_TOKEN=os.getenv('HF_TOKEN', ''),
        WANDB=wandb_key,
    )

    cmd = cmd.format(**format_dict)
    return cmd


@hydra_runner(config_path='conf', config_name='run_local')
def main(cluster_cfg):
    script_path = cluster_cfg.script
    script_config = cluster_cfg.script_config
    results_dir = cluster_cfg.results_dir

    script_path = Path(script_path).absolute()
    script_config = Path(script_config).absolute()

    gather_mounts(cluster_cfg)

    # Add the results directory to the cluster config as a mount path
    run_utils.add_mount_path(results_dir, '/results', cluster_cfg)

    cluster_script_path = check_root_path(script_path, NEMO_ROOT)

    # Create results and logdir
    log_dir = cluster_cfg.get('log_dir', os.path.join(results_dir, 'logs'))
    run_utils.create_remote_directory([results_dir, log_dir], cluster_cfg)

    merged_config = merge_configs(script_config, cluster_cfg)
    run_utils.create_remote_config(merged_config, "config.yaml", results_dir, cluster_cfg)

    check_config_mount_paths(merged_config, cluster_cfg)

    # Resolve experiment name
    exp_name = cluster_cfg.exp_name
    if exp_name is None:
        if 'exp_manager' in merged_config and 'name' in merged_config['exp_manager']:
            exp_name = merged_config['exp_manager']['name']
        else:
            raise ValueError("Experiment name not provided in the run config file (`exp_name`)) or the cluster config (inside exp_manager.name)")

    with run.Experiment(exp_name) as exp:
        cmd = get_execution_script(cluster_script_path, "config.yaml")

        job_name = f"{exp_name}_job"
        num_gpus = cluster_cfg.get('num_gpus', merged_config['trainer']['devices'])
        if isinstance(num_gpus, list):
            num_gpus = len(num_gpus)
        num_nodes = cluster_cfg.get('num_nodes', merged_config['trainer'].get('num_nodes', 1))
        cluster_cfg = OmegaConf.to_object(cluster_cfg)

        run_utils.add_task(exp,
            cmd=cmd,
            task_name=job_name,
            cluster_config=cluster_cfg,
            container=cluster_cfg['containers']['asr'],
            num_tasks=cluster_cfg.get('num_tasks', cluster_cfg.get('num_tasks_per_node', 1)),
            num_gpus=num_gpus,
            num_nodes=num_nodes,
            log_dir=run_utils.get_mounted_filepath(cluster_cfg, log_dir),
            partition=cluster_cfg.get('partition', None),
            run_after=cluster_cfg.get('run_after', None),
        )

        run_utils.run_exp(exp, cluster_cfg)


if __name__ == '__main__':
    main()
