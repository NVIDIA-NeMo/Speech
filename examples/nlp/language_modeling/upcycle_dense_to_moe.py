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

r"""
Conversion script to convert NeMo Mistral-7B checkpoints into HuggingFace checkpoint.
  Example to run this conversion script:
    python3 upcycle_dense_to_moe.py \
        --model <path_to_nemo_checkpoints_folder> \
        --num-experts 8 \
        --output_path <path_to_output_hf_file>
"""

from argparse import ArgumentParser
from pathlib import Path

import torch
import torch.nn
from lightning.pytorch.trainer.trainer import Trainer

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy, NLPSaveRestoreConnector
from nemo.utils import logging


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default=None, required=True, help="Path to NeMo checkpoint")
    parser.add_argument(
        "--output-path", type=str, default='', required=False, help="Path to NeMo save upcycled checkpoint"
    )
    parser.add_argument(
        "--num-experts", type=int, default=8, required=True, help="Number of experts to use in upcycled model."
    )
    args = parser.parse_args()
    assert isinstance(args.num_experts, int)
    assert args.num_experts > 1, "Expected --num-experts to be greater-than 1."
    if args.output_path == '':
        args.output_path = args.model + f'_upcycled_num_exp{args.num_experts}.nemo'
    return args


def make_moe_config_from_dense(config, num_experts=8):
    from copy import deepcopy

    moe_config = deepcopy(config)
    moe_config['num_moe_experts'] = num_experts
    return moe_config


def unwrap(model):
    tmp = model
    while hasattr(tmp, 'module'):
        tmp = tmp.module
    return tmp


def upcycle(in_file, num_experts, cpu_only=True) -> None:
    """
    Upcycle dense checkpoint to MoE.
    """

    logging.info(f'Loading NeMo checkpoint from: {in_file}')

    dummy_trainer = Trainer(devices=1, accelerator='cpu', strategy=NLPDDPStrategy())

    # Load dense model
    model_config = MegatronGPTModel.restore_from(in_file, trainer=dummy_trainer, return_config=True)
    model_config.tensor_model_parallel_size = 1
    model_config.pipeline_model_parallel_size = 1
    model_config.sequence_parallel = False
    if cpu_only:
        map_location = torch.device('cpu')
        model_config.use_cpu_initialization = True
    else:
        map_location = None
    model_config.perform_initialization = False
    dense_model = MegatronGPTModel.restore_from(
        in_file, trainer=dummy_trainer, override_config_path=model_config, map_location=map_location
    )

    # Make upcycled config
    moe_config = make_moe_config_from_dense(model_config, num_experts)
    # print(moe_config)
    # quit()
    dummy_trainer2 = Trainer(devices=1, accelerator='cpu', strategy=NLPDDPStrategy())
    moe_model = MegatronGPTModel(moe_config, trainer=dummy_trainer2)

    # convert state dict dense -> MoE
    from megatron.core.transformer.moe.upcycling_utils import upcycle_state_dict

    moe_state_dict = upcycle_state_dict([unwrap(moe_model.model)], [unwrap(dense_model.model)])
    moe_model.model.module.load_state_dict(moe_state_dict['model'])
    moe_model._save_restore_connector = NLPSaveRestoreConnector()
    # hack
    if Path(args.model).is_dir():
        moe_model._save_restore_connector._model_extracted_dir = args.model

    moe_model.save_to(args.output_path)


if __name__ == '__main__':
    args = get_args()
    upcycle(args.model, args.num_experts)
    logging.info(f'Upcycled checkpoint saved to: {args.output_path}')
