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
from dataclasses import dataclass

import torch
from megatron.core.optimizer import OptimizerConfig

from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.common.tokenizers.tokenizer_utils import get_nmt_tokenizer


## NOTE: This script is present for github-actions testing only.
def get_args():
    parser = argparse.ArgumentParser(description='Pretraining a small BERT model using NeMo 2.0')
    parser.add_argument('--experiment_dir', type=str, help="directory to write results and checkpoints to")
    parser.add_argument('--devices', type=int, default=1, help="number of devices")
    parser.add_argument('--max_steps', type=int, default=3, help="number of devices")
    parser.add_argument('--mbs', type=int, default=1, help="micro batch size")
    parser.add_argument('--tp_size', type=int, default=1, help="tensor parallel size")
    parser.add_argument('--pp_size', type=int, default=1, help="pipeline parallel size")
    parser.add_argument('--type', type=str, default='huggingface')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=args.tp_size,
        pipeline_model_parallel_size=args.pp_size,
        # Pipeline dtype is coupled with the bf16 mixed precision plugin
        pipeline_dtype=torch.bfloat16,
        ckpt_load_strictness="log_all",  # Only for CI tests to use older versions of checkpoint
    )

    trainer = nl.Trainer(
        devices=args.devices,
        max_steps=args.max_steps,
        accelerator="gpu",
        strategy=strategy,
        plugins=nl.MegatronMixedPrecision(precision="bf16-mixed"),
        log_every_n_steps=1,
        limit_val_batches=2,
        val_check_interval=2,
        num_sanity_val_steps=0,
    )

    ckpt = nl.ModelCheckpoint(
        save_last=True,
        monitor="reduced_train_loss",
        save_top_k=1,
        save_on_train_epoch_end=True,
        save_optim_on_train_end=True,
    )

    logger = nl.NeMoLogger(
        log_dir=args.experiment_dir,
        use_datetime_version=False,  # must be false if using auto resume
        ckpt=ckpt,
    )

    adam = nl.MegatronOptimizerModule(
        config=OptimizerConfig(
            optimizer="adam",
            lr=0.0001,
            adam_beta2=0.98,
            use_distributed_optimizer=True,
            clip_grad=1.0,
            bf16=True,
        ),
    )

    data = llm.BERTMockDataModule(
        seq_length=512,
        micro_batch_size=args.mbs,
        global_batch_size=8,
        num_workers=0,
    )

    tokenizer = get_nmt_tokenizer("megatron", "BertWordPieceLowerCase")
    if args.type == 'huggingface':
        print('Init HuggingFace Bert Base Model')
        model = llm.BertModel(llm.HuggingFaceBertBaseConfig(), tokenizer=tokenizer)
    elif args.type == 'megatron':
        print('Init Megatron Bert Base Model')
        model = llm.BertModel(llm.MegatronBertBaseConfig(), tokenizer=tokenizer)
    else:
        raise ValueError('Unknown type.')
    resume = nl.AutoResume(
        resume_if_exists=True,
        resume_ignore_no_checkpoint=True,
    )

    llm.pretrain(model=model, data=data, trainer=trainer, log=logger, optim=adam, resume=resume)

    print("Bert Pretraining Succeeded")
