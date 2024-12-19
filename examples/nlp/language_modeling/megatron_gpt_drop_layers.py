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
NOTE: This script will be deprecated soon in favor of `megatron_gpt_prune.py`. Please use the new script for trimming layers.

Script to trim model layers.
  Example to run the script with checkpoint:
    python -m torch.distributed.launch --nproc_per_node=<tensor_model_parallel_size> * <pipeline_model_parallel_size> \
     megatron_gpt_drop_layers.py \
     --checkpoint_folder <path_to_PTL_checkpoints_folder> \
     --checkpoint_name <checkpoint_name> \
     --path_to_save <path_to_output_ckpt_files> \
     --tensor_model_parallel_size <tensor_model_parallel_size> \
     --pipeline_model_parallel_size <pipeline_model_parallel_size> \
     --hparams_file <path_to_model_yaml_config> \
     --gpus_per_node <gpus_per_node>
  Example to run the script with .nemo model:
    python -m torch.distributed.launch --nproc_per_node=<tensor_model_parallel_size> * <pipeline_model_parallel_size> \
     megatron_gpt_drop_layers.py \
     --path_to_nemo <path_to_.nemo_model_file>
     --path_to_save <path_to_output_.nemo_model> \
     --tensor_model_parallel_size <tensor_model_parallel_size> \
     --pipeline_model_parallel_size <pipeline_model_parallel_size> \
     --gpus_per_node <gpus_per_node>
"""

import os
from argparse import ArgumentParser

import torch
from megatron.core import parallel_state
from megatron.core.transformer.transformer_block import TransformerBlock
from omegaconf import OmegaConf, open_dict

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
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
        required=False,
        help="Path to PTL checkpoints saved during training. Ex: /raid/nemo_experiments/megatron_gpt/checkpoints",
    )
    parser.add_argument(
        "--checkpoint_name",
        type=str,
        default=None,
        required=False,
        help="Name of checkpoint to be used. Ex: megatron_gpt--val_loss=0.14-step=20-consumed_samples=160.0-last",
    )
    parser.add_argument("--path_to_nemo", type=str, required=False, default=None, help="path to .nemo file")
    parser.add_argument(
        "--hparams_file",
        type=str,
        default=None,
        required=False,
        help="Path config for restoring. It's created during training and may need to be modified during restore if restore environment is different than training. Ex: /raid/nemo_experiments/megatron_gpt/hparams.yaml",
    )
    parser.add_argument(
        "--precision",
        type=str,
        required=False,
        default='16-mixed',
        choices=['32-true', '16-mixed', 'bf16-mixed'],
        help="Precision value for the trainer that matches with precision of the ckpt",
    )
    parser.add_argument("--local-rank", type=int, required=False, default=os.getenv('LOCAL_RANK', -1))
    parser.add_argument("--cluster_type", required=False, default=None, help="Whether on BCP platform")
    parser.add_argument("--gpus_per_node", type=int, required=True, default=None)
    parser.add_argument("--tensor_model_parallel_size", type=int, required=True, default=None)
    parser.add_argument("--pipeline_model_parallel_size", type=int, required=True, default=None)
    parser.add_argument(
        "--drop_layers", type=int, default=None, required=True, nargs="+", help="list of layer numbers to drop."
    )
    parser.add_argument("--path_to_save", type=str, required=True, help="Path to output ckpt files.")
    parser.add_argument("--zarr", action="store_true", help="zarr ckpt strategy usage.")

    args = parser.parse_args()
    return args


# function to trim model layers
def trim_layers(model, layers_to_trim):
    for name, module in model.named_modules():
        if isinstance(module, TransformerBlock):
            print(f'Removing from {name} {len(layers_to_trim)} of {len(module.layers)} layers')
            for i in sorted(layers_to_trim, reverse=True):
                assert i > 0 and i < len(module.layers), "Layers are numbered from 0 to num_layers"
                del module.layers[i - 1]
            module.config.num_layers = len(module.layers)
            for i, layer in enumerate(module.layers):
                layer.layer_number = i + 1

    return model


def main(local_rank, rank, world_size, args):
    logging.warning("This script will be deprecated soon in favor of `megatron_gpt_prune.py`.")

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
            'native_amp_init_scale': 2**32,
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

    if not args.path_to_nemo:
        checkpoint_path = os.path.join(args.checkpoint_folder, args.checkpoint_name)
        model_path = checkpoint_path
    else:
        model_path = args.path_to_nemo
    logging.info(
        f'rank: {rank}, local_rank: {local_rank}, is loading checkpoint: {model_path} for tp_rank: {app_state.tensor_model_parallel_rank} and pp_rank: {app_state.pipeline_model_parallel_rank}'
    )

    if not args.path_to_nemo:
        # check for distributed checkpoint
        # restore model from the checkpoint
        model = MegatronGPTModel.load_from_checkpoint(model_path, hparams_file=args.hparams_file, trainer=trainer)
    else:
        # restore model from the .nemo file
        model = MegatronGPTModel.restore_from(model_path, trainer=trainer)
    model._save_restore_connector = NLPSaveRestoreConnector()

    save_file_path = args.path_to_save
    if not args.path_to_nemo:
        # Without --path_to_nemo, path_to_save is expected to be a directory.
        # Adding a dummy model filename here conforms with SaveRestoreConnector's convention.
        model._save_restore_connector.pack_nemo_file = False
        save_file_path = os.path.join(save_file_path, 'model.nemo')

    model = trim_layers(model, args.drop_layers)

    OmegaConf.set_struct(model.cfg, False)
    model.cfg.dist_ckpt_format = 'zarr' if args.zarr else 'torch_dist'
    OmegaConf.set_struct(model.cfg, True)
    model.cfg.num_layers -= len(args.drop_layers)

    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    model.save_to(save_file_path)

    logging.info(f'NeMo model saved to: {args.path_to_save}')


if __name__ == '__main__':
    args = get_args()

    local_rank, rank, world_size = initialize_distributed(args)

    main(local_rank, rank, world_size, args)
