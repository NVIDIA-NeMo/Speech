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
import os

from omegaconf import OmegaConf

from nemo.collections import speechlm
from nemo.collections.common.tokenizers.tokenizer_utils import get_nmt_tokenizer
from nemo.utils import logging

## NOTE: This script is present for github-actions testing only.


def get_args():
    parser = argparse.ArgumentParser(description='Finetune a small GPT model using NeMo 2.0')
    parser.add_argument('--restore_path', type=str, help="Path to model to be finetuned")
    parser.add_argument('--peft', type=str, default='lora', help="none | lora")
    parser.add_argument('--devices', type=int, default=1, help="number of devices")
    parser.add_argument('--max_steps', type=int, default=1, help="number of steps")
    parser.add_argument('--mbs', type=int, default=1, help="micro batch size")
    parser.add_argument('--tp_size', type=int, default=1, help="tensor parallel size")
    parser.add_argument('--pp_size', type=int, default=1, help="pipeline parallel size")
    parser.add_argument('--experiment_dir', type=str, help="directory to write results and checkpoints to")
    parser.add_argument('--train_manifest', type=str, help="path to train manifest")
    parser.add_argument('--val_manifest', type=str, help="path to validation manifest")
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    tokenizer = get_nmt_tokenizer(tokenizer_model=os.path.join(args.restore_path, "dummy_tokenizer.model"))

    cur_dir = os.path.dirname(os.path.realpath(__file__))
    yaml_path = os.path.abspath(os.path.join(cur_dir, 'conf/salm_ci_config.yaml'))
    cfg = OmegaConf.load(yaml_path)

    cfg.data.common.micro_batch_size = args.mbs
    cfg.data.common.global_batch_size = args.mbs * args.devices
    cfg.strategy.tensor_parallel_size = args.tp_size
    cfg.strategy.pipeline_parallel_size = args.pp_size
    cfg.trainer.devices = args.devices
    cfg.trainer.max_steps = args.max_steps
    cfg.logger.log_dir = args.experiment_dir
    cfg.data.train_ds.manifest_filepath = args.train_manifest
    cfg.data.validation_ds.manifest_filepath = args.val_manifest

    if args.peft == "none":
        cfg.model.llm.pop("peft")

    cfg = OmegaConf.to_container(cfg, resolve=True)
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')
    cfg = OmegaConf.create(cfg)
    speechlm.speech_to_text_llm_train(cfg, tokenizer=tokenizer)

    logging.info("Finetuning done.")
