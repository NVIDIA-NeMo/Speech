# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
Conversion script to convert PTL checkpoints into nemo checkpoint.
  Example to run this conversion script:
    python -m torch.distributed.launch --nproc_per_node=<tensor_model_parallel_size> * <pipeline_model_parallel_size> \
     megatron_ckpt_to_nemo.py \
     --checkpoint_folder <path_to_PTL_checkpoints_folder> \
     --checkpoint_name <checkpoint_name> \
     --nemo_file_path <path_to_output_nemo_file> \
     --tensor_model_parallel_size <tensor_model_parallel_size> \
     --pipeline_model_parallel_size <pipeline_model_parallel_size>
"""

import os
from argparse import ArgumentParser

import torch
from apex.transformer import parallel_state
from lightning_lite.plugins.environments import TorchElasticEnvironment
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.nlp.models.language_modeling.megatron_bart_model import MegatronBARTModel
from nemo.collections.nlp.models.language_modeling.megatron_bert_model import MegatronBertModel
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.models.language_modeling.megatron_retrieval_model import MegatronRetrievalModel
from nemo.collections.nlp.models.language_modeling.megatron_t5_model import MegatronT5Model
from nemo.collections.nlp.models.machine_translation.megatron_nmt_model import MegatronNMTModel
from nemo.collections.nlp.parts.nlp_overrides import NLPSaveRestoreConnector
from nemo.utils import AppState, logging
from nemo.utils.distributed import initialize_distributed
from nemo.utils.model_utils import inject_model_parallel_rank


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
        help="Name of checkpoint to be used. Ex: megatron_gpt--val_loss=6.34-step=649-last.ckpt",
    )

    parser.add_argument(
        "--hparams_file",
        type=str,
        default=None,
        required=False,
        help="Path config for restoring. It's created during training and may need to be modified during restore if restore environment is different than training. Ex: /raid/nemo_experiments/megatron_gpt/hparams.yaml",
    )
    parser.add_argument("--nemo_file_path", type=str, default=None, required=True, help="Path to output .nemo file.")
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
    parser.add_argument(
        "--model_type", type=str, required=True, default="gpt", choices=["gpt", "t5", "bert", "nmt", "bart", "retro"]
    )
    parser.add_argument("--local_rank", type=int, required=False, default=os.getenv('LOCAL_RANK', -1))
    parser.add_argument("--bcp", action="store_true", help="Whether on BCP platform")

    args = parser.parse_args()
    return args


def convert(local_rank, rank, world_size, args):

    app_state = AppState()
    app_state.data_parallel_rank = 0
    num_nodes = world_size // args.gpus_per_node
    if args.bcp:
        trainer = Trainer(
            devices=args.gpus_per_node, num_nodes=num_nodes, accelerator='gpu', plugins=[TorchElasticEnvironment()]
        )
    else:
        trainer = Trainer(devices=args.gpus_per_node, num_nodes=num_nodes, accelerator='gpu')

    app_state.pipeline_model_parallel_size = args.pipeline_model_parallel_size
    app_state.tensor_model_parallel_size = args.tensor_model_parallel_size
    # Auto set split rank for T5, BART, NMT if split rank is None.
    if args.pipeline_model_parallel_size > 1 and args.model_type in ['t5', 'bart', 'nmt']:
        if args.pipeline_model_parallel_split_rank is not None:
            app_state.pipeline_model_parallel_split_rank = args.pipeline_model_parallel_split_rank
        else:
            if args.pipeline_model_parallel_size % 2 != 0:
                raise ValueError(
                    f"Pipeline model parallel size {args.pipeline_model_parallel_size} must be even if split rank is not specified."
                )
            else:
                # If split rank is not set, then we set it to be pipeline_model_parallel_size // 2 - this is because in most cases we have the same number of enc/dec layers.
                app_state.pipeline_model_parallel_split_rank = args.pipeline_model_parallel_size // 2
    else:
        app_state.pipeline_model_parallel_split_rank = None

    app_state.model_parallel_size = app_state.tensor_model_parallel_size * app_state.pipeline_model_parallel_size

    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size_=app_state.tensor_model_parallel_size,
        pipeline_model_parallel_size_=app_state.pipeline_model_parallel_size,
        pipeline_model_parallel_split_rank_=app_state.pipeline_model_parallel_split_rank,
    )

    app_state.pipeline_model_parallel_rank = parallel_state.get_pipeline_model_parallel_rank()
    app_state.tensor_model_parallel_rank = parallel_state.get_tensor_model_parallel_rank()

    # inject model parallel rank
    checkpoint_path = inject_model_parallel_rank(os.path.join(args.checkpoint_folder, args.checkpoint_name))

    logging.info(
        f'rank: {rank}, local_rank: {local_rank}, is loading checkpoint: {checkpoint_path} for tp_rank: {app_state.tensor_model_parallel_rank} and pp_rank: {app_state.pipeline_model_parallel_rank}'
    )

    if args.model_type == 'gpt':
        model = MegatronGPTModel.load_from_checkpoint(checkpoint_path, hparams_file=args.hparams_file, trainer=trainer)
    elif args.model_type == 'bert':
        model = MegatronBertModel.load_from_checkpoint(
            checkpoint_path, hparams_file=args.hparams_file, trainer=trainer
        )
    elif args.model_type == 't5':
        model = MegatronT5Model.load_from_checkpoint(checkpoint_path, hparams_file=args.hparams_file, trainer=trainer)
    elif args.model_type == 'bart':
        model = MegatronBARTModel.load_from_checkpoint(
            checkpoint_path, hparams_file=args.hparams_file, trainer=trainer
        )
    elif args.model_type == 'nmt':
        model = MegatronNMTModel.load_from_checkpoint(checkpoint_path, hparams_file=args.hparams_file, trainer=trainer)
    elif args.model_type == 'retro':
        model = MegatronRetrievalModel.load_from_checkpoint(
            checkpoint_path, hparams_file=args.hparams_file, trainer=trainer
        )
    model._save_restore_connector = NLPSaveRestoreConnector()

    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    model.save_to(args.nemo_file_path)

    logging.info(f'NeMo model saved to: {args.nemo_file_path}')


if __name__ == '__main__':
    args = get_args()

    local_rank, rank, world_size = initialize_distributed(args)

    convert(local_rank, rank, world_size, args)
