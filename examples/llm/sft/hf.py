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

import fiddle as fdl
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader

from nemo import lightning as nl
from nemo.collections import llm
from nemo.lightning.pytorch.accelerate.transformer_engine import is_te_accelerated
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


def squad(tokenizer) -> pl.LightningDataModule:
    """Instantiates a SquadDataModuleWithPthDataloader and return it

    Args:
        tokenizer (AutoTokenizer): the tokenizer to use

    Returns:
        pl.LightningDataModule: the dataset to train with.
    """
    return SquadDataModuleWithPthDataloader(
        tokenizer=tokenizer,
        seq_length=512,
        micro_batch_size=2,
        global_batch_size=128,  # assert gbs == mbs * accumulate_grad_batches
        num_workers=0,
        dataset_kwargs={
            "sanity_check_dist_workers": False,
            "pad_to_max_length": True,
            "get_attention_mask_from_fusion": True,
        },
    )


def main():
    """Example script to run SFT with a HF transformers-instantiated model on squad."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='meta-llama/Llama-3.2-1B')
    parser.add_argument('--strategy', type=str, default='auto', choices=['auto', 'ddp', 'fsdp'])
    parser.add_argument('--devices', default=1)
    parser.add_argument('--accelerator', default='gpu', choices=['gpu'])
    parser.add_argument('--model-accelerator', default=None, choices=['te'])
    parser.add_argument('--max-steps', type=int, default=100)
    parser.add_argument("--fp8-autocast", default=False, action='store_true')
    parser.add_argument('--wandb-project', type=str, default=None)
    parser.add_argument('--model-save-path', type=str, default=None)
    parser.add_argument('--use-torch-jit', action='store_true')
    args = parser.parse_args()

    wandb = None
    if args.wandb_project is not None:
        model = '_'.join(args.model.split('/')[-2:])
        wandb = WandbLogger(
            project=args.wandb_project,
            name=f'{model}_dev{args.devices}_strat_{args.strategy}',
        )
    grad_clip = 0.5
    if args.strategy == 'fsdp':
        # See: https://github.com/Lightning-AI/pytorch-lightning/blob/8ad3e29816a63d8ce5c00ac104b14729a4176f4f/src/lightning/pytorch/plugins/precision/fsdp.py#L81
        grad_clip = None
    use_dist_samp = False

    model_accelerator = None
    if args.model_accelerator == "te":
        from functools import partial
        from nemo.lightning.pytorch.accelerate.transformer_engine import te_accelerate

        model_accelerator = partial(te_accelerate, fp8_autocast=args.fp8_autocast)

    from nemo.lightning.pytorch.accelerate.transformer_engine import te_accelerate

    model = llm.HFAutoModelForCausalLM(model_name=args.model, model_accelerator=model_accelerator)
    tokenizer = model.tokenizer

    callbacks = []
    if args.use_torch_jit:
        jit_config = JitConfig(use_torch=True, torch_kwargs={'dynamic': False}, use_thunder=False)
        callbacks = [JitTransform(jit_config)]

    llm.api.finetune(
        model=model,
        data=squad(tokenizer),
        trainer=nl.Trainer(
            devices=args.devices,
            max_steps=args.max_steps,
            accelerator=args.accelerator,
            strategy=args.strategy,
            log_every_n_steps=1,
            limit_val_batches=0.0,
            num_sanity_val_steps=0,
            accumulate_grad_batches=10,
            gradient_clip_val=grad_clip,
            use_distributed_sampler=use_dist_samp,
            logger=wandb,
            callbacks=callbacks,
            precision="bf16",
        ),
        optim=fdl.build(llm.adam.pytorch_adam_with_flat_lr(lr=1e-5)),
        log=None,
    )

    if args.model_accelerator:
        if args.model_accelerator == "te":
            te_acc = is_te_accelerated(model.model)
            assert te_acc, "Transformer Engine acceleration was unsuccessful"
            print("TE Accelerated: ", te_acc)

    if args.model_save_path is not None:
        model.save_pretrained(args.model_save_path)


if __name__ == '__main__':
    main()
