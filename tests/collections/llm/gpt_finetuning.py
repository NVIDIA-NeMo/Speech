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

from megatron.core.optimizer import OptimizerConfig

from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer

## NOTE: This script is present for github-actions testing only.


@dataclass
class Llama3Config96M(llm.Llama3Config8B):
    seq_length: int = 2048
    num_layers: int = 2
    hidden_size: int = 768
    ffn_hidden_size: int = 3072
    num_attention_heads: int = 8


def get_args():
    parser = argparse.ArgumentParser(description='Finetune a small GPT model using NeMo 2.0')
    parser.add_argument('--restore_path', type=str, help="Path to model to be finetuned")
    parser.add_argument('--experiment_dir', type=str, help="directory to write results and checkpoints to")
    parser.add_argument('--devices', type=int, default=1, help="number of devices")
    parser.add_argument('--mbs', type=int, default=1, help="micro batch size")
    parser.add_argument('--tp_size', type=int, default=1, help="tensor parallel size")
    parser.add_argument('--pp_size', type=int, default=1, help="pipeline parallel size")

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=args.tp_size,
        pipeline_parallel_size=args.pp_size,
    )

    trainer = nl.Trainer(
        devices=args.devices,
        max_steps=2,
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

    lora = llm.peft.LoRA()

    squad = llm.SquadDataModule(seq_length=2048, micro_batch_size=args.mbs, global_batch_size=8, num_workers=0)

    tokenizer = get_nmt_tokenizer(
        tokenizer_model="/lustre/fsw/coreai_dlalgo_llm/nemo_home/models/llama_96M/dummy_tokenizer.model"
    )
    llama3_8b = llm.LlamaModel(Llama3Config96M(), tokenizer=tokenizer)

    resume = nl.AutoResume(
        restore_config=nl.RestoreConfig(path=args.restore_path),
        resume_if_exists=True,
    )

    llm.finetune(
        model=llama3_8b,
        data=squad,
        trainer=trainer,
        peft=lora,
        log=logger,
        optim=adam,
        resume=resume,
    )
