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

## NOTE: This script is present for github-actions testing only.
## There are no guarantees that this script is up-to-date with latest NeMo.

import argparse

import torch
from megatron.core.optimizer import OptimizerConfig

from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.llm.api import _setup
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.lightning import NeMoLogger
from nemo.lightning.pytorch.optim.megatron import MegatronOptimizerModule


def get_args():
    parser = argparse.ArgumentParser(description='Train a small GPT model using NeMo 2.0')
    parser.add_argument('--devices', type=int, help="Number of devices to use for training")
    parser.add_argument('--max-steps', type=int, help="Number of steps to train for")
    parser.add_argument(
        '--experiment-dir', type=str, default=None, help="directory to write results and checkpoints to"
    )
    parser.add_argument('--model-path', type=str, help="Path to model checkpoint")
    parser.add_argument(
        '--tokenizer-model-path', type=str, default=None, help="Path to tokenizer model, defaults to None"
    )
    return parser.parse_args()


if __name__ == "__main__":

    args = get_args()

    # Checkpoint callback setup
    checkpoint_callback = nl.ModelCheckpoint(
        every_n_train_steps=10,
        dirpath=args.experiment_dir,
    )

    trainer = nl.Trainer(
        devices=args.devices,
        max_steps=args.max_steps,
        accelerator="gpu",
        strategy=nl.MegatronStrategy(
            ckpt_load_optimizer=False,
            ckpt_save_optimizer=False,
            tensor_model_parallel_size=1,
        ),
        plugins=nl.MegatronMixedPrecision(
            precision="bf16-mixed",
            params_dtype=torch.bfloat16,
        ),
        callbacks=[checkpoint_callback],
        log_every_n_steps=1,
        limit_val_batches=5,
        val_check_interval=10,
        num_sanity_val_steps=0,
    )

    opt_config = OptimizerConfig(
        optimizer='adam',
        lr=1e-5,
        min_lr=1e-5,
        use_distributed_optimizer=False,
        clip_grad=1.0,
        bf16=True,
    )

    optim = MegatronOptimizerModule(config=opt_config)
    model_config = llm.BaseMambaConfig130M()
    model_config.tokenizer_model_path = args.tokenizer_model_path

    tokenizer = get_nmt_tokenizer(
        library=model_config.tokenizer_library,
        model_name=model_config.tokenizer_name,
        tokenizer_model=model_config.tokenizer_model_path,
        use_fast=True,
    )

    model = llm.GPTModel(model_config, optim=optim, tokenizer=tokenizer)

    ckpt_path = model.import_ckpt(
        path="pytorch://" + args.model_path,
        model_config=model_config,
    )

    nemo_logger = NeMoLogger(
        log_dir=args.experiment_dir,
    )

    data = llm.SquadDataModule(
        seq_length=512,
        micro_batch_size=2,
        global_batch_size=4,
        tokenizer=model.tokenizer,
        num_workers=0,
        pad_to_max_length=True,
    )

    app_state = _setup(
        model=model,
        data=data,
        resume=None,
        trainer=trainer,
        log=nemo_logger,
        optim=optim,
        tokenizer=tokenizer,
        model_transform=None,
    )

    trainer.fit(model, data, ckpt_path=ckpt_path)
