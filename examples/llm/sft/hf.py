#!/usr/bin/python3
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

import tempfile
from functools import partial

import fiddle as fdl
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader

from nemo import lightning as nl
from nemo.collections import llm
from nemo.lightning.pytorch.callbacks import JitConfig, JitTransform


class SquadDataModuleWithPthDataloader(llm.SquadDataModule):
    """Creates a squad dataset with a PT dataloader"""

    def _create_dataloader(self, dataset, mode, **kwargs) -> DataLoader:
        return DataLoader(
            dataset,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=dataset.collate_fn,
            batch_size=self.micro_batch_size,
            **kwargs,
        )


def squad(tokenizer, mbs=1, gbs=2) -> pl.LightningDataModule:
    """Instantiates a SquadDataModuleWithPthDataloader and return it

    Args:
        tokenizer (AutoTokenizer): the tokenizer to use

    Returns:
        pl.LightningDataModule: the dataset to train with.
    """
    return SquadDataModuleWithPthDataloader(
        tokenizer=tokenizer,
        seq_length=512,
        micro_batch_size=mbs,
        global_batch_size=gbs,
        num_workers=0,
        dataset_kwargs={
            "sanity_check_dist_workers": False,
            "get_attention_mask_from_fusion": True,
        },
    )


def main():
    """Example script to run SFT with a HF transformers-instantiated model on squad."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.2-1B')
    parser.add_argument('--strategy', type=str, default='auto', choices=['auto', 'ddp', 'fsdp2'])
    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--num-nodes', type=int, default=1)
    parser.add_argument('--accelerator', type=str, default='gpu', choices=['gpu'])
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--model-accelerator', type=str, default=None, choices=['te'])
    parser.add_argument('--max-steps', type=int, default=100)
    parser.add_argument("--fp8-autocast", action='store_true')
    parser.add_argument('--wandb-project', type=str, default=None)
    parser.add_argument('--ckpt-folder', type=str, default=tempfile.TemporaryDirectory().name)
    parser.add_argument('--use-torch-jit', action='store_true')
    args = parser.parse_args()

    wandb = None
    if args.wandb_project is not None:
        model = '_'.join(args.model.split('/')[-2:])
        wandb = WandbLogger(
            project=args.wandb_project,
            name=f'{model}_dev{args.devices}_strat_{args.strategy}',
        )

    model_accelerator = None
    if args.model_accelerator == "te":
        from nemo.lightning.pytorch.accelerate.transformer_engine import te_accelerate

        model_accelerator = partial(te_accelerate, fp8_autocast=args.fp8_autocast)

    callbacks = []
    if args.use_torch_jit:
        jit_config = JitConfig(use_torch=True, torch_kwargs={'dynamic': False}, use_thunder=False)
        callbacks = [JitTransform(jit_config)]

    callbacks.append(
        nl.ModelCheckpoint(
            every_n_train_steps=args.max_steps // 2,
            dirpath=args.ckpt_folder,
        )
    )

    if args.strategy == 'fsdp2':
        args.strategy = nl.FSDP2Strategy(data_parallel_size=args.devices * args.num_nodes, tensor_parallel_size=1)

    llm.api.finetune(
        model=llm.HFAutoModelForCausalLM(model_name=args.model, model_accelerator=model_accelerator),
        data=squad(llm.HFAutoModelForCausalLM.configure_tokenizer(args.model), gbs=args.devices),
        trainer=nl.Trainer(
            devices=args.devices,
            num_nodes=args.num_nodes,
            max_steps=args.max_steps,
            accelerator=args.accelerator,
            strategy=args.strategy,
            log_every_n_steps=1,
            limit_val_batches=0.0,
            num_sanity_val_steps=0,
            accumulate_grad_batches=1,
            gradient_clip_val=args.grad_clip,
            use_distributed_sampler=False,
            logger=wandb,
            callbacks=callbacks,
            precision="bf16",
        ),
        optim=fdl.build(llm.adam.pytorch_adam_with_flat_lr(lr=1e-5)),
        log=None,
    )


if __name__ == '__main__':
    main()
