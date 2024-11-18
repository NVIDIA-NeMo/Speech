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
from lightning.pytorch.loggers import TensorBoardLogger
from megatron.core.optimizer import OptimizerConfig

from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.llm.api import train
from nemo.collections.llm.gpt.data import PreTrainingDataModule
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.lightning import NeMoLogger
from nemo.lightning.pytorch.callbacks import ModelCheckpoint
from nemo.lightning.pytorch.optim.megatron import MegatronOptimizerModule


def get_args():
    parser = argparse.ArgumentParser(description='Train a Mamba model using NeMo 2.0')
    parser.add_argument('--devices', type=int, help="Number of devices to use for training")
    parser.add_argument('--max-steps', type=int, help="Number of steps to train for")
    parser.add_argument(
        '--experiment-dir', type=str, default=None, help="directory to write results and checkpoints to"
    )
    parser.add_argument('--data-path', type=str, help="Path to data file")
    parser.add_argument('--tokenizer-path', type=str, default=None, help="Path to tokenizer model")

    return parser.parse_args()


if __name__ == '__main__':

    args = get_args()

    seq_length = 512

    tokenizer = get_nmt_tokenizer(
        "huggingface",
        "EleutherAI/gpt-neox-20b",
        tokenizer_model=None,
        use_fast=True,
    )
    data = PreTrainingDataModule(
        paths=args.data_path,
        seq_length=seq_length,
        micro_batch_size=2,
        global_batch_size=16,
        seed=1234,
        tokenizer=tokenizer,
    )
    ssm_config = llm.SSMConfig(
        hybrid_override_pattern="M-M*",
        num_layers=4,
        hidden_size=1024,
        ffn_hidden_size=1024,
        num_attention_heads=4,
        seq_length=seq_length,
        init_method_std=0.02,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        layernorm_epsilon=1e-5,
        make_vocab_size_divisible_by=16,
    )
    model = llm.GPTModel(ssm_config, tokenizer=data.tokenizer)
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
    )
    checkpoint_callback = ModelCheckpoint(
        every_n_train_steps=10,
        dirpath=args.experiment_dir,
    )
    callbacks = [checkpoint_callback]

    loggers = []
    tensorboard_logger = TensorBoardLogger(
        save_dir='dummy',  ## NOTE: this gets overwritten by default
    )
    loggers.append(tensorboard_logger)

    opt_config = OptimizerConfig(
        optimizer='adam',
        lr=6e-4,
        min_lr=6e-5,
        clip_grad=1.0,
        use_distributed_optimizer=False,
        bf16=True,
    )
    opt = MegatronOptimizerModule(config=opt_config)

    trainer = nl.Trainer(
        devices=args.devices,
        max_steps=args.max_steps,
        accelerator="gpu",
        strategy=strategy,
        logger=loggers,
        callbacks=callbacks,
        log_every_n_steps=1,
        limit_val_batches=2,
        plugins=nl.MegatronMixedPrecision(
            precision="bf16-mixed",
            params_dtype=torch.bfloat16,
        ),
    )

    nemo_logger = NeMoLogger(
        log_dir=args.experiment_dir,
    )

    train(
        model=model,
        data=data,
        trainer=trainer,
        log=nemo_logger,
        tokenizer='data',
        optim=opt,
    )
