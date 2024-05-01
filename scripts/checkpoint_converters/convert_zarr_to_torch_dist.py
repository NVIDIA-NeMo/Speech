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

r"""
Conversion script to convert zarr checkpoints into torch distributed checkpoint.
  Example to run this conversion script:
    python -m torch.distributed.launch --nproc_per_node=<tensor_model_parallel_size> * <pipeline_model_parallel_size> \
     megatron_zarr_ckpt_to_torch_dist.py \
     --model_type <model_type> \
     --checkpoint_folder <path_to_PTL_checkpoints_folder> \
     --checkpoint_name <checkpoint_name> \
     --path_to_save <path_to_output_ckpt_files> \
     --tensor_model_parallel_size <tensor_model_parallel_size> \
     --pipeline_model_parallel_size <pipeline_model_parallel_size> \
     --hparams_file <path_to_model_yaml_config> \
     --gpus_per_node <gpus_per_node>
"""

import os
from argparse import ArgumentParser

import torch
from megatron.core import parallel_state
from omegaconf import OmegaConf, open_dict

from nemo.collections.nlp.models.language_modeling.megatron_bert_model import MegatronBertModel
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.models.language_modeling.megatron_gpt_sft_model import MegatronGPTSFTModel
from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronTrainerBuilder
from nemo.collections.nlp.parts.nlp_overrides import NLPSaveRestoreConnector
from nemo.utils import AppState, logging
from nemo.utils.distributed import initialize_distributed


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--checkpoint_folder",
        type=str,
        default=None,
        required=True,
        help="Path to PTL checkpoints saved during training. Ex: /raid/nemo_experiments/megatron_gpt/checkpoints",
    )
    parser.add_argument(
        "--checkpoint_name",
        type=str,
        default=None,
        required=True,
        help="Name of checkpoint to be used. Ex: megatron_gpt--val_loss=0.14-step=20-consumed_samples=160.0-last",
    )

    parser.add_argument(
        "--hparams_file",
        type=str,
        default=None,
        required=True,
        help="Path config for restoring. It's created during training and may need to be modified during restore if restore environment is different than training. Ex: /raid/nemo_experiments/megatron_gpt/hparams.yaml",
    )
    parser.add_argument("--path_to_save", type=str, default=None, required=True, help="Path to output ckpt files.")
    parser.add_argument(
        "--save_to_nemo", action="store_true", help="If passed, output will be written as .nemo file.",
    )
    parser.add_argument("--gpus_per_node", type=int, required=True, default=None)
    parser.add_argument("--tensor_model_parallel_size", type=int, required=True, default=None)
    parser.add_argument("--pipeline_model_parallel_size", type=int, required=True, default=None)
    parser.add_argument(
        "--pipeline_model_parallel_split_rank",
        type=int,
        required=False,
        default=None,
        help="If pipeline parallel size > 1, this is the rank at which the encoder ends and the decoder begins.",
    )
    parser.add_argument("--local_rank", type=int, required=False, default=os.getenv('LOCAL_RANK', -1))
    parser.add_argument("--cluster_type", required=False, default=None, help="Whether on BCP platform")
    parser.add_argument(
        "--precision",
        type=str,
        required=False,
        default='bf16-mixed',
        choices=['32-true', '16-mixed', 'bf16-mixed'],
        help="Precision value for the trainer that matches with precision of the ckpt",
    )

    parser.add_argument(
        "--model_type", type=str, required=True, default="gpt", choices=["gpt", "sft", "bert"],
    )

    args = parser.parse_args()
    return args


def convert(local_rank, rank, world_size, args):

    app_state = AppState()
    app_state.data_parallel_rank = 0
    num_nodes = world_size // args.gpus_per_node

    cfg = {
        'trainer': {
            'devices': args.gpus_per_node,
            'num_nodes': num_nodes,
            'accelerator': 'gpu',
            'precision': args.precision,
        },
        'model': {
            'native_amp_init_scale': 2 ** 32,
            'native_amp_growth_interval': 1000,
            'hysteresis': 2,
            'gradient_as_bucket_view': True,
        },
        'cluster_type': args.cluster_type,
    }
    cfg = OmegaConf.create(cfg)

    # Set precision None after precision plugins are created as PTL >= 2.1 does not allow both
    # precision plugins and precision to exist
    cfg.trainer.precision = None

    trainer = MegatronTrainerBuilder(cfg).create_trainer()

    app_state.pipeline_model_parallel_size = args.pipeline_model_parallel_size
    app_state.tensor_model_parallel_size = args.tensor_model_parallel_size
    app_state.pipeline_model_parallel_split_rank = None

    app_state.model_parallel_size = app_state.tensor_model_parallel_size * app_state.pipeline_model_parallel_size

    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=app_state.tensor_model_parallel_size,
        pipeline_model_parallel_size=app_state.pipeline_model_parallel_size,
        pipeline_model_parallel_split_rank=app_state.pipeline_model_parallel_split_rank,
    )

    app_state.pipeline_model_parallel_rank = parallel_state.get_pipeline_model_parallel_rank()
    app_state.tensor_model_parallel_rank = parallel_state.get_tensor_model_parallel_rank()

    # check for distributed checkpoint
    checkpoint_path = os.path.join(args.checkpoint_folder, args.checkpoint_name)

    logging.info(
        f'rank: {rank}, local_rank: {local_rank}, is loading checkpoint: {checkpoint_path} for tp_rank: {app_state.tensor_model_parallel_rank} and pp_rank: {app_state.pipeline_model_parallel_rank}'
    )

    if args.model_type == "gpt":
        model = MegatronGPTModel.load_from_checkpoint(checkpoint_path, hparams_file=args.hparams_file, trainer=trainer)
    elif args.model_type == "sft":
        model = MegatronGPTSFTModel.load_from_checkpoint(
            checkpoint_path, hparams_file=args.hparams_file, trainer=trainer
        )
        # we force the target for the loaded model to have the correct target
        # because the hparams.yaml sometimes contains MegatronGPTModel as the target.
        with open_dict(model.cfg):
            model.cfg.target = f"{MegatronGPTSFTModel.__module__}.{MegatronGPTSFTModel.__name__}"
    elif args.model_type == 'bert':
        model = MegatronBertModel.load_from_checkpoint(
            checkpoint_path, hparams_file=args.hparams_file, trainer=trainer
        )

    with open_dict(model.cfg):
        model.cfg.torch_distributed_checkpoint = True

    model._save_restore_connector = NLPSaveRestoreConnector()
    save_file_path = args.path_to_save
    if not args.save_to_nemo:
        # With --save_to_nemo, save_to_path is expected to be a directory.
        # Adding a dummy model filename here conforms with SaveRestoreConnector's convention.
        model._save_restore_connector.pack_nemo_file = False
        save_file_path = os.path.join(save_file_path, 'model.nemo')

    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    model.save_to(save_file_path)

    logging.info(f'NeMo model saved to: {args.path_to_save}')


if __name__ == '__main__':
    args = get_args()

    local_rank, rank, world_size = initialize_distributed(args)

    convert(local_rank, rank, world_size, args)
