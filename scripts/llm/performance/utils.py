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

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import nemo_run as run
import pandas as pd
from lightning.pytorch.callbacks.callback import Callback
from nemo_run.config import NEMORUN_HOME
from numpy import nan

from nemo.collections.common.tokenizers.huggingface import AutoTokenizer
from nemo.collections.llm.gpt.data.squad import SquadDataModule
from nemo.collections.llm.gpt.model import GPTModel
from nemo.collections.llm.recipes.llama3_8b import MegatronCommOverlapCallback
from nemo.lightning.base import DEFAULT_NEMO_CACHE_HOME
from nemo.utils import logging

DEFAULT_NEMO_HOME = os.getenv('NEMO_HOME', DEFAULT_NEMO_CACHE_HOME)


def slurm_executor(
    account: str,
    partition: str,
    log_dir: str,
    nodes: int,
    num_gpus_per_node: int,
    time_limit: str = "00:30:00",
    container_image: str = "nvcr.io/nvidia/nemo:dev",
    custom_mounts: List[str] = [],
    custom_env_vars: Dict[str, str] = {},
    custom_srun_args: List[str] = [],
    hf_token: str = None,
    nemo_home: str = DEFAULT_NEMO_HOME,
) -> run.SlurmExecutor:
    """
    Slurm cluster definition with appropriate cluster params and NeMo container params needed for pre-training
    and fine-tuning experiments
    """
    err_msgs = []
    if log_dir != NEMORUN_HOME:
        err_msgs.append(f"\nRun `export NEMORUN_HOME={log_dir}` in your shell environment and rerun this script.")
    if len(err_msgs) > 0:
        logging.error("\n".join(err_msgs))
        sys.exit(1)

    env_vars = {
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "False",
        "NCCL_NVLS_ENABLE": "0",
        "NVTE_DP_AMAX_REDUCE_INTERVAL": "0",
        "NVTE_ASYNC_AMAX_REDUCTION": "1",
        "NVTE_FUSED_ATTN": "1",
        "NVTE_FLASH_ATTN": "1",
        "NEMO_LOG_MEMORY_USAGE": "1",
        "NEMORUN_HOME": log_dir,
    }
    mounts = []
    srun_args = ["--mpi=pmix"]

    if nemo_home != DEFAULT_NEMO_CACHE_HOME:  # DO NOT change this to 'DEFAULT_NEMO_HOME'/'NEMO_HOME'
        env_vars.update({"NEMO_HOME": nemo_home})
        mounts.extend([f"{nemo_home}:{nemo_home}"])
    if hf_token is not None:
        env_vars.update({"HF_TOKEN": hf_token, "TRANSFORMERS_OFFLINE": "0"})

    env_vars |= custom_env_vars
    mounts.extend(custom_mounts)
    srun_args.extend(custom_srun_args)

    executor = run.SlurmExecutor(
        account=account,
        partition=partition,
        tunnel=run.LocalTunnel(
            job_dir=os.path.join(log_dir, "experiments"),
        ),
        nodes=nodes,
        ntasks_per_node=num_gpus_per_node,
        container_image=container_image,
        container_mounts=mounts,
        env_vars=env_vars,
        srun_args=srun_args,
        time=time_limit,
        mem="0",
        exclusive=True,
        packager=run.GitArchivePackager(),
    )

    return executor


def hf_tokenizer(model_name: str) -> run.Config[AutoTokenizer]:
    """
    HuggingFace tokenizer.

    Args:
        model_name (str): corresponds to HuggingFace-AutoTokenizer's 'pretrained_model_name_or_path' input argument.
                For more details please refer to-
                huggingface.co/docs/transformers/v4.47.1/en/model_doc/auto#transformers.AutoTokenizer
    """
    log_msg = [
        f"`AutoTokenizer` first searches for tokenizer files locally stored in {DEFAULT_NEMO_HOME}.",
        "(from env var `NEMO_HOME`- can be changed using '-nh/--nemo_home' CLI arg).",
        "If files are missing locally, `AutoTokenizer` will try downloading from HuggingFace. In this case-",
        "make sure env vars 'TRANSFORMERS_OFFLINE':'0' and 'HF_TOKEN':'<token_value>' are set in your sbatch script.",
        "Both of these will be set automatically if you provide '-hf/--hf_token' CLI arg.",
    ]
    logging.warning(" ".join(log_msg))

    return run.Config(
        AutoTokenizer,
        pretrained_model_name=model_name,
        use_fast=True,
    )


def get_user_configs(gpu: str, task: str, model_name: str, model_size: str, args) -> List[int]:
    """
    Choose recommended configs tuned for performance from a csv file if available.
    User (command line) provided args override the recommended configs.

    NOTE: pre-train and PEFT recommended configs available for H100 and B200.

    Args:
        gpu (str): target GPU machine for experiment. Options- ['h100', 'b200']
        task (str): experiment task. Options- ['pre_train', 'sft', 'lora']
        model_name (str): target model for experiment. E.g.: 'llama3', 'mixtral'
        model_size (str): size of target model. E.g.: '8b' (for llama3)
    """
    script_dir = str(Path(__file__).parent.absolute())
    recommended_configs_csv = os.path.join(script_dir, "recommended_model_configs", f"model_configs_{gpu}.csv")
    logging.info(f"Using {recommended_configs_csv} for loading default recommended model configs")

    config_df = pd.DataFrame()
    if os.path.isfile(recommended_configs_csv):
        df = pd.read_csv(recommended_configs_csv)
        config_df = df[
            (df["task"] == task)
            & (df["model"] == model_name)
            & (df["size"] == model_size)
            & (df["dtype"] == args.compute_dtype)
        ]
        config_df = config_df.replace({nan: None})
        if len(config_df) == 0:
            logging.warning(f"Missing performance configs for {task}-{model_name}-{model_size}-{args.compute_dtype}")
            logging.warning("Make sure you provide all necessary arguments in the command line")

    config = config_df.to_dict(orient='records')[0] if len(config_df) > 0 else {}

    num_gpus = config.get("num_gpus") if args.num_gpus is None else args.num_gpus
    num_nodes = -(num_gpus // -args.gpus_per_node)  # ceil division
    mbs = config.get("mbs") if args.micro_batch_size is None else args.micro_batch_size
    gbs = config.get("gbs") if args.global_batch_size is None else args.global_batch_size
    tp_size = config.get("tp_size") if args.tensor_parallel_size is None else args.tensor_parallel_size
    pp_size = config.get("pp_size") if args.pipeline_parallel_size is None else args.pipeline_parallel_size
    cp_size = config.get("cp_size") if args.context_parallel_size is None else args.context_parallel_size
    ep_size = config.get("ep_size") if args.expert_parallel_size is None else args.expert_parallel_size
    vp_size = args.virtual_pipeline_parallel_size
    vp_size = config.get("vp_size") if vp_size is None else vp_size
    etp_size = args.expert_tensor_parallel_size
    etp_size = config.get("etp_size") if etp_size is None else etp_size

    kwargs = num_nodes, mbs, gbs, tp_size, pp_size, cp_size, vp_size, ep_size, etp_size
    kwargs = [int(arg) if arg is not None else arg for arg in kwargs]

    return kwargs


def set_primary_perf_configs(
    recipe,
    enable_tb: bool,
    num_nodes: int,
    num_gpus_per_node: int,
    mbs: int,
    gbs: int,
    max_steps: int,
    tp_size: int,
    pp_size: int,
    cp_size: int,
    vp_size: int,
    ep_size: int,
    etp_size: Optional[int] = None,
):
    """Set experiment configs we usually tune for performance of all models."""
    # nemo.lightning.Trainer configs
    recipe.trainer.num_nodes = num_nodes
    recipe.trainer.devices = num_gpus_per_node
    recipe.trainer.max_steps = max_steps

    # lightning.pytorch.LightningDataModule configs
    recipe.data.micro_batch_size = mbs
    recipe.data.global_batch_size = gbs

    # parallelism configs
    recipe.trainer.strategy.tensor_model_parallel_size = tp_size
    recipe.trainer.strategy.pipeline_model_parallel_size = pp_size
    recipe.trainer.strategy.context_parallel_size = cp_size
    recipe.trainer.strategy.virtual_pipeline_model_parallel_size = None if vp_size == 1 else vp_size
    recipe.trainer.strategy.expert_model_parallel_size = ep_size
    recipe.trainer.strategy.expert_tensor_parallel_size = etp_size

    recipe.trainer.strategy.sequence_parallel = bool(tp_size > 1)

    # callback configs
    comm_overlap_callback_idx = get_comm_overlap_callback_idx(recipe.trainer.callbacks)
    dp_size = (num_nodes * num_gpus_per_node) / (tp_size * pp_size * cp_size)
    if comm_overlap_callback_idx is not None:
        # WARNING: If True, checkpointing (if enabled) might not work
        recipe.trainer.callbacks[comm_overlap_callback_idx].overlap_param_gather_with_optimizer_step = bool(
            dp_size > 1 and pp_size > 1 and vp_size and vp_size > 1
        )

    if not enable_tb:  # tensorboard adds performance overhead.
        recipe.log.tensorboard = None
        recipe.trainer.logger = False
    else:
        # default path is NOT intuitive- `<log_dir>/code/nemo_experiments/tb_logs/default/<tfevents_file>`
        recipe.log.log_dir = "/nemo_run/lightning_logs"  # saves file at- `<log_dir>/lightning_logs/tb_logs

    # Misc. for overall faster experiment runtime
    recipe.log.ckpt = None
    recipe.trainer.enable_checkpointing = False
    recipe.trainer.val_check_interval = max_steps
    recipe.trainer.log_every_n_steps = 1

    return recipe


def import_ckpt_experiment(executor: run.SlurmExecutor, model: run.Config[GPTModel], source: str):
    """
    Downloads/Acceses checkpoint to be used for fine-tuning. `import_ckpt` first tries find the nemo checkpoint in
    <NEMO_HOME>/models/. For eg: for llama3 8b, the path will look like- <NEMO_HOME>/models/meta-llama/Meta-Llama-3-8B
    If missing, tries to downloads at the same location from HuggingFace and converts it nemo format.

    Args:
        source (str): HuggingFace URL. For eg- hf://meta-llama/Meta-Llama-3-70B
    """
    from copy import deepcopy

    from nemo.collections.llm import import_ckpt

    import_executor = deepcopy(executor)
    import_executor.ntasks_per_node = 1
    import_executor.nodes = 1

    return run.Partial(import_ckpt, model=model, source=source, overwrite=False), import_executor, "import_ckpt_exp"


def isfile_train_pack_metadata(hf_model_uri: str, data_config: run.Config[SquadDataModule]) -> bool:
    """
    This method is used for fine-tuning. It checks if packed train data for a partiular
    sequence length exists locally. This is needed to set data flag (force_redownload=True)
    which avoids experiment crash in case files are missing.
    """
    datasets_dir = os.getenv("NEMO_DATASETS_CACHE", os.path.join(DEFAULT_NEMO_HOME, "datasets"))
    model_dir = hf_model_uri.replace("/", "--")
    metadata_filename = f"{data_config.seq_length}_metadata.jsonl"

    train_pack_metadata_filepath = os.path.join(datasets_dir, "squad", "packed", model_dir, metadata_filename)

    return os.path.exists(train_pack_metadata_filepath) and os.path.isfile(train_pack_metadata_filepath)


def get_comm_overlap_callback_idx(callbacks: List[Callback]) -> int | None:
    """
    nemo.lightning.Trainer has a list of callbacks defined. This method identifies index of MegatronCommOverlapCallback
    from the list defined in recipes in nemo.collections.llm.recipes. The index is needed to override ddp communication
    params
    """
    if callbacks:  # default is None in lightning
        for idx, callback in enumerate(callbacks):
            if callback.__fn_or_cls__ == MegatronCommOverlapCallback:
                return idx
    return None
