# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

"""
Usage example:
  python /opt/NeMo/examples/multimodal/generative/stable_diffusion/convert_hf_ckpt_to_nemo.py
    --ckpt_path=path/to/hf.ckpt
    --hparams_file=path/to/saved.yaml
    --nemo_file_path=hf2sd.nemo

Additionally, provide a NeMo hparams file with the correct model architecture arguments. Refer to examples/multimodal/foundation/clip/conf/megatron_clip_config.yaml.
"""

import os
import tempfile
from argparse import ArgumentParser

import torch
from lightning.fabric.utilities.cloud_io import _load as pl_load
from lightning.pytorch.plugins.environments import TorchElasticEnvironment
from lightning.pytorch.trainer.trainer import Trainer
from omegaconf import OmegaConf

from nemo.collections.multimodal.models.text_to_image.controlnet.controlnet import MegatronControlNet
from nemo.collections.multimodal.models.text_to_image.stable_diffusion.diffusion_engine import MegatronDiffusionEngine
from nemo.collections.multimodal.models.text_to_image.stable_diffusion.ldm.ddpm import MegatronLatentDiffusion
from nemo.collections.nlp.parts.nlp_overrides import NLPSaveRestoreConnector
from nemo.utils import AppState, logging
from nemo.utils.distributed import initialize_distributed

try:
    from megatron.core import parallel_state

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    HAVE_MEGATRON_CORE = False


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default=None, required=True, help="Path to checkpoint.")

    parser.add_argument(
        "--hparams_file",
        type=str,
        default=None,
        required=False,
        help="Path config for restoring. It's created during training and may need to be modified during restore if restore environment is different than training. Ex: /raid/nemo_experiments/megatron_gpt/hparams.yaml",
    )
    parser.add_argument("--nemo_file_path", type=str, default=None, required=True, help="Path to output .nemo file.")
    parser.add_argument("--gpus_per_node", type=int, required=False, default=1)
    parser.add_argument("--tensor_model_parallel_size", type=int, required=False, default=1)
    parser.add_argument("--pipeline_model_parallel_size", type=int, required=False, default=1)
    parser.add_argument(
        "--pipeline_model_parallel_split_rank",
        type=int,
        required=False,
        default=None,
        help="If pipeline parallel size > 1, this is the rank at which the encoder ends and the decoder begins.",
    )
    parser.add_argument("--local_rank", type=int, required=False, default=os.getenv('LOCAL_RANK', -1))
    parser.add_argument("--bcp", action="store_true", help="Whether on BCP platform")
    parser.add_argument("--model_type", type=str, required=False, default="stable_diffusion")
    parser.add_argument("--nemo_clip_path", type=str, required=False, help="Path to clip ckpt file in .nemo format")

    args = parser.parse_args()
    return args


def load_config_and_state_from_nemo(nemo_path):
    if torch.cuda.is_available():
        map_location = torch.device('cuda')
    else:
        map_location = torch.device('cpu')
    save_restore_connector = NLPSaveRestoreConnector()
    cwd = os.getcwd()

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            save_restore_connector._unpack_nemo_file(path2file=nemo_path, out_folder=tmpdir)

            # Change current working directory to
            os.chdir(tmpdir)
            config_yaml = os.path.join(tmpdir, save_restore_connector.model_config_yaml)
            cfg = OmegaConf.load(config_yaml)

            model_weights = os.path.join(tmpdir, save_restore_connector.model_weights_ckpt)
            state_dict = save_restore_connector._load_state_dict_from_disk(model_weights, map_location=map_location)
        finally:
            os.chdir(cwd)

    return cfg, state_dict


def mapping_hf_state_dict(hf_state_dict, model, clip_dict=None):
    nemo_state = model.state_dict()
    new_state_dict = {}
    for k, v in hf_state_dict.items():
        k = 'model.' + k
        # This is not necessary when you turn off model.inductor in config file
        # if 'diffusion_model' in k:
        #     k = k.replace('diffusion_model', 'diffusion_model._orig_mod')
        if 'in_layers' in k or 'out_layers' in k:
            s = k.split('.')
            idx = int(s[-2])
            if idx != 0:
                k = ".".join(s[:-2] + [str(int(idx - 1))] + [s[-1]])
        if k in nemo_state:
            new_state_dict[k] = v
    if clip_dict:
        for k, v in clip_dict.items():
            k = k.replace("model.text_encoder", "model.cond_stage_model.model")
            if k in nemo_state:
                new_state_dict[k] = v
    for k in [
        'betas',
        'alphas_cumprod',
        'alphas_cumprod_prev',
        'sqrt_alphas_cumprod',
        'sqrt_one_minus_alphas_cumprod',
        'log_one_minus_alphas_cumprod',
        'sqrt_recip_alphas_cumprod',
        'sqrt_recipm1_alphas_cumprod',
        'posterior_variance',
        'posterior_log_variance_clipped',
        'posterior_mean_coef1',
        'posterior_mean_coef2',
    ]:
        new_state_dict['model.' + k] = nemo_state['model.' + k]

    return new_state_dict


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

    # no use atm, use to split ranks in encoder/decoder models.
    if args.pipeline_model_parallel_size > 1 and args.model_type in []:
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
        tensor_model_parallel_size=app_state.tensor_model_parallel_size,
        pipeline_model_parallel_size=app_state.pipeline_model_parallel_size,
    )

    app_state.pipeline_model_parallel_rank = parallel_state.get_pipeline_model_parallel_rank()
    app_state.tensor_model_parallel_rank = parallel_state.get_tensor_model_parallel_rank()

    if args.ckpt_path.endswith('safetensors'):
        from safetensors.torch import load_file as load_safetensors

        checkpoint = load_safetensors(args.ckpt_path)
    else:
        checkpoint = pl_load(args.ckpt_path, map_location='cpu')
    if 'state_dict' in checkpoint.keys():
        checkpoint = checkpoint['state_dict']
    cfg = OmegaConf.load(args.hparams_file)
    cfg.model.inductor = False
    if args.model_type == 'stable_diffusion':
        model = MegatronLatentDiffusion(cfg.model, trainer)
    elif args.model_type == 'controlnet':
        model = MegatronControlNet(cfg.model, trainer)
    elif args.model_type == 'sdxl':
        cfg.model.unet_config.from_pretrained = args.ckpt_path
        model = MegatronDiffusionEngine(cfg.model, trainer)
    else:
        raise NotImplementedError

    if model.cfg.get('cond_stage_config', None) and 'nemo' in model.cfg.cond_stage_config._target_:
        assert (
            args.nemo_clip_path is not None
        ), "To align with current hparams file, you need to provide .nemo checkpoint of clip model for stable diffusion. If you want to convert HF clip checkpoint to .nemo checkpoint first, please refer to /opt/NeMo/examples/multimodal/foundation/clip/convert_external_clip_to_nemo.py"
        _, clip_dict = load_config_and_state_from_nemo(args.nemo_clip_path)
    else:
        clip_dict = None

    if args.model_type != 'sdxl':
        state_dict = mapping_hf_state_dict(checkpoint, model, clip_dict=clip_dict)

        model._save_restore_connector = NLPSaveRestoreConnector()

        model.load_state_dict(state_dict)

    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    model.save_to(args.nemo_file_path)

    logging.info(f'NeMo model saved to: {args.nemo_file_path}')


if __name__ == '__main__':
    args = get_args()
    local_rank, rank, world_size = initialize_distributed(args)
    convert(local_rank, rank, world_size, args)
