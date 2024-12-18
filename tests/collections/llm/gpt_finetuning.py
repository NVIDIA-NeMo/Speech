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
import argparse
import os
from dataclasses import dataclass

import torch
from megatron.core.optimizer import OptimizerConfig

from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.llm.gpt.data.packed_sequence import PackedSequenceSpecs
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from tests.collections.llm.common import Llama3ConfigCI


## NOTE: This script is present for github-actions testing only.


def get_args():
    parser = argparse.ArgumentParser(description='Finetune a small GPT model using NeMo 2.0')
    parser.add_argument('--restore_path', type=str, help="Path to model to be finetuned")
    parser.add_argument('--experiment_dir', type=str, help="directory to write results and checkpoints to")
    parser.add_argument('--peft', type=str, default='none', help="none | lora")
    parser.add_argument('--devices', type=int, default=1, help="number of devices")
    parser.add_argument('--max_steps', type=int, default=1, help="number of devices")
    parser.add_argument('--mbs', type=int, default=1, help="micro batch size")
    parser.add_argument('--tp_size', type=int, default=1, help="tensor parallel size")
    parser.add_argument('--pp_size', type=int, default=1, help="pipeline parallel size")
    parser.add_argument('--packed', action='store_true', help="use packed sequence dataset")
    parser.add_argument(
        '--chat_dataset_path', type=str, default="", help="path to chat dataset. Uses dolly if this is empty."
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=args.tp_size,
        pipeline_model_parallel_size=args.pp_size,
        # Pipeline dtype is coupled with the bf16 mixed precision plugin
        pipeline_dtype=torch.bfloat16,
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

    if args.peft in llm.peft.PEFT_STR2CLS:
        peft = llm.peft.PEFT_STR2CLS[args.peft]()
    else:
        peft = None

    packed_sequence_specs = (
        PackedSequenceSpecs(packed_sequence_size=2048, tokenizer_model_name="dummy_tokenizer") if args.packed else None
    )
    if args.chat_dataset_path:
        assert not args.packed
        data = llm.ChatDataModule(
            dataset_root=args.chat_dataset_path,
            seq_length=2048,
            micro_batch_size=args.mbs,
            global_batch_size=8,
            num_workers=0,
            packed_sequence_specs=packed_sequence_specs,
        )
    else:
        data = llm.DollyDataModule(
            seq_length=2048,
            micro_batch_size=args.mbs,
            global_batch_size=8,
            num_workers=0,
            packed_sequence_specs=packed_sequence_specs,
        )

    tokenizer = get_nmt_tokenizer(tokenizer_model=os.path.join(args.restore_path, "dummy_tokenizer.model"))
    llama3_8b = llm.LlamaModel(Llama3ConfigCI(), tokenizer=tokenizer)

    resume = nl.AutoResume(
        restore_config=nl.RestoreConfig(path=args.restore_path),
        resume_if_exists=True,
    )

    llm.finetune(
        model=llama3_8b,
        data=data,
        trainer=trainer,
        peft=peft,
        log=logger,
        optim=adam,
        resume=resume,
    )
