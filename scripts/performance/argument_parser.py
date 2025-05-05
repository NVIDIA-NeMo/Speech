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

import argparse

from nemo_run.config import get_nemorun_home

from .utils import DEFAULT_NEMO_HOME


def parse_cli_args():
    """
    Command line arguments correspong to Slurm cluster and NeMo2.0 for running pre-training and
    fine-tuning experiments.
    """
    parser = argparse.ArgumentParser(description="NeMo2.0 Performance Pretraining and Fine-Tuning")

    parser.add_argument(
        "-a",
        "--account",
        type=str,
        help="Slurm account to use for experiment",
        required=True,
    )
    parser.add_argument(
        "-p",
        "--partition",
        type=str,
        help="Slurm partition to use for experiment",
        required=True,
    )
    parser.add_argument(
        "-l",
        "--log_dir",
        type=str,
        help=f"Directory for logging experiment results. Defaults to {get_nemorun_home()}",
        required=False,
        default=get_nemorun_home(),
    )
    parser.add_argument(
        "-t",
        "--time_limit",
        type=str,
        help="Maximum time limit to run experiment for. Defaults to 30 minutes (format- 'HH:MM:SS')",
        required=False,
        default="00:30:00",
    )
    container_img_msg = [
        "NeMo container to use for experiment. Defaults to latest dev container- 'nvcr.io/nvidia/nemo:dev'",
        "Make sure your NGC credentials are accessible in your environment.",
    ]
    parser.add_argument(
        "-i",
        "--container_image",
        type=str,
        help=" ".join(container_img_msg),
        required=False,
        default="nvcr.io/nvidia/nemo:dev",
    )
    parser.add_argument(
        "-c",
        "--compute_dtype",
        type=str,
        choices=["bf16", "fp8"],
        help="Compute precision. Options- bf16 or fp8. Defaults to bf16",
        required=False,
        default="bf16",
    )
    fp8_recipe_msg = (
        "FP8 recipe. Options- ds (per-tensor delayed scaling), cs (per-tensor current scaling), mxfp8. Defaults to ds"
    )
    parser.add_argument(
        "-fr",
        "--fp8_recipe",
        type=str,
        choices=["ds", "cs", "mxfp8"],
        help=fp8_recipe_msg,
        required=False,
        default="ds",
    )
    parser.add_argument(
        "-en",
        "--enable_nsys",
        help="Enable Nsys profiling. Diabled by default",
        action="store_true",
    )
    parser.add_argument(
        "-tb",
        "--tensorboard",
        help="Enable tensorboard logging. Disabled by default",
        action="store_true",
    )
    parser.add_argument(
        "-wd",
        "--wandb",
        help="Enable wandb logging. Disabled by default",
        action="store_true",
    )
    parser.add_argument(
        "-wdk",
        "--wandb_key",
        type=str,
        help="wandb key. Needed for wandb logger projetion to server",
        required=False,
        default=None,
    )
    parser.add_argument(
        "-wdp",
        "--wandb_prj_name",
        type=str,
        help="wandb project name",
        required=False,
        default=None,
    )
    parser.add_argument(
        "-wdj",
        "--wandb_job_name",
        type=str,
        help="wandb job name",
        required=False,
        default=None,
    )
    parser.add_argument(
        "-f",
        "--finetuning",
        choices=["sft", "lora"],
        help="Finetuning scheme to use. Defaults to 'lora'",
        default='lora',
    )
    parser.add_argument(
        "-hf",
        "--hf_token",
        type=str,
        help="HuggingFace token. Defaults to None. Required for accessing tokenizers and checkpoints.",
        default=None,
    )
    nemo_home_msg = [
        "Sets env var `NEMO_HOME` (on compute node using sbatch script)- directory where NeMo searches",
        "for models and checkpoints. This saves a lot of time (especially for bigger models) if checkpoints already",
        f"exist here. Missing files will be downloaded here from HuggingFace. Defaults to {DEFAULT_NEMO_HOME}",
    ]
    parser.add_argument(
        "-nh",
        "--nemo_home",
        type=str,
        help=" ".join(nemo_home_msg),
        default=DEFAULT_NEMO_HOME,
    )
    parser.add_argument(
        "-d",
        "--dryrun",
        help="If true, prints sbatch script to terminal without launching experiment.",
        required=False,
        action="store_true",
    )
    parser.add_argument(
        "-tp",
        "--tensor_parallel_size",
        type=int,
        help="Intra-layer model parallelism. Splits tensors across GPU ranks.",
        required=False,
        default=None,
    )
    parser.add_argument(
        "-pp",
        "--pipeline_parallel_size",
        type=int,
        help="Inter-layer model parallelism. Splits transformer layers across GPU ranks.",
        required=False,
        default=None,
    )
    parser.add_argument(
        "-cp",
        "--context_parallel_size",
        type=int,
        help="Splits network input along sequence dimension across GPU ranks.",
        required=False,
        default=None,
    )
    parser.add_argument(
        "-vp",
        "--virtual_pipeline_parallel_size",
        type=int,
        help="Number of virtual blocks per pipeline model parallel rank is the virtual model parallel size.",
        required=False,
        default=None,
    )
    parser.add_argument(
        "-ep",
        "--expert_parallel_size",
        type=int,
        help="Distributes Moe Experts across sub data parallel dimension.",
        required=False,
        default=None,
    )
    parser.add_argument(
        "-et",
        "--expert_tensor_parallel_size",
        type=lambda x: int(x) if x is not None else None,
        nargs="?",
        const=None,
        help="Intra-layer tensor model parallelsm for expert layer. Splits tensors across GPU ranks.\
            Use -et/--expert_tensor_parallel_size <space> for None or -et/--expert_tensor_parallel_size <int>",
        required=False,
        default=None,
    )
    parser.add_argument(
        "-mb",
        "--micro_batch_size",
        type=int,
        required=False,
        default=None,
    )
    parser.add_argument(
        "-gb",
        "--global_batch_size",
        type=int,
        required=False,
        default=None,
    )
    parser.add_argument(
        "-g",
        "--gpu",
        type=str,
        help="Target gpu type. Defaults to 'h100'.",
        required=False,
        default="h100",
    )
    parser.add_argument(
        "-ng",
        "--num_gpus",
        type=int,
        help="Number of gpus.",
        required=False,
        default=None,
    )
    parser.add_argument(
        "-gn",
        "--gpus_per_node",
        type=int,
        help="Number of gpus per node. Defaults to 8",
        required=False,
        default=8,
    )
    parser.add_argument(
        "-ms",
        "--max_steps",
        type=int,
        help="Number of train steps. Defaults to 100",
        required=False,
        default=100,
    )

    def bool_arg(arg):
        if arg.lower() in ['true', '1', 't', 'yes', 'y']:
            return True
        elif arg.lower() in ['false', '0', 'f', 'no', 'n']:
            return False
        else:
            raise ValueError(f"Invalid value for boolean argument: {arg}")

    parser.add_argument(
        "-cg",
        "--cuda_graphs",
        help="Enable CUDA graphs. Disabled by default",
        type=bool_arg,
        required=False,
        default=None,  # NOTE: DO NOT SET DEFAULT TO FALSE, IT WILL BE OVERRIDDEN BY THE RECOMMENDED MODEL CONFIGS
    )
    parser.add_argument(
        "-fsdp",
        "--use_mcore_fsdp",
        help="Enable Megatron Core (Mcore) FSDP. Disabled by default",
        type=bool_arg,
        required=False,
        default=None,
    )
    parser.add_argument(
        "-rl",
        "--recompute_layers",
        type=int,
        help="Number of Transformer layers to recompute, where all the intermediate "
        "activations of a Transformer layer are computed. Defaults to None",
        required=False,
        default=None,
    )
    parser.add_argument(
        "-ol",
        "--activation_offload_layers",
        type=int,
        help="Number of Transformer layers to offload to the CPU memory. Defaults to None",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--nccl_communicator_config_path",
        type=str,
        help="Path to NCCL communicator config yaml file",
        required=False,
        default=None,
    )

    def list_of_strings(arg):
        return arg.split(',')

    parser.add_argument(
        "-rm",
        "--recompute_modules",
        nargs="*",
        const=None,
        type=str,
        help="List of modules to perform selective activation recompute. Users can provide 0 or any number of arguments. Defaults to None",
        required=False,
        default=None,
    )

    parser.add_argument(
        "-cm",
        "--custom_mounts",
        type=list_of_strings,
        help="Comma separated string of mounts",
        required=False,
        default=[],
    )

    return parser
