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

import fiddle as fdl
from lightning.pytorch.loggers import WandbLogger

from nemo import lightning as nl
from nemo.collections import llm
from nemo.lightning import NeMoLogger
from nemo.lightning.pytorch.callbacks import JitConfig, JitTransform


def make_squad_hf_dataset(tokenizer):
    def formatting_prompts_func(example):
        formatted_text = [
            f"Context: {example['context']} Question: {example['question']} Answer:",
            f" {example['answers']['text'][0].strip()}",
        ]
        context_ids, answer_ids = list(map(tokenizer.text_to_ids, formatted_text))
        if len(context_ids) > 0 and context_ids[0] != tokenizer.bos_id:
            context_ids.insert(0, tokenizer.bos_id)
        if len(answer_ids) > 0 and answer_ids[-1] != tokenizer.eos_id:
            answer_ids.append(tokenizer.eos_id)

        return dict(
            labels=(context_ids + answer_ids)[1:],
            input_ids=(context_ids + answer_ids)[:-1],
            loss_mask=[0] * (len(context_ids) - 1) + [1] * len(answer_ids),
        )

    datamodule = llm.HFDatasetDataModule("rajpurkar/squad", split="train[:100]", pad_token_id=tokenizer.eos_id)
    datamodule.map(
        formatting_prompts_func,
        batched=False,
        batch_size=2,
        remove_columns=["id", "title", "context", "question", 'answers'],
    )
    return datamodule


def main():
    """Example script to run PEFT with a HF transformers-instantiated model on squad."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.2-1B')
    parser.add_argument('--strategy', type=str, default='auto', choices=['auto', 'ddp', 'fsdp2'])
    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--num-nodes', type=int, default=1)
    parser.add_argument('--accelerator', type=str, default='gpu', choices=['gpu'])
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--max-steps', type=int, default=100)
    parser.add_argument('--wandb-project', type=str, default=None)
    parser.add_argument('--use-torch-jit', action='store_true')
    parser.add_argument('--ckpt-folder', type=str, default=tempfile.TemporaryDirectory().name)
    args = parser.parse_args()

    wandb = None
    if args.wandb_project is not None:
        model = '_'.join(args.model.split('/')[-2:])
        wandb = WandbLogger(
            project=args.wandb_project,
            name=f'{model}_dev{args.devices}_strat_{args.strategy}',
        )

    callbacks = []
    if args.use_torch_jit:
        jit_config = JitConfig(use_torch=True, torch_kwargs={'dynamic': True}, use_thunder=False)
        callbacks = [JitTransform(jit_config)]

    if args.strategy == 'fsdp2':
        args.strategy = nl.FSDP2Strategy(data_parallel_size=args.devices * args.num_nodes, tensor_parallel_size=1)

    llm.api.finetune(
        model=llm.HFAutoModelForCausalLM(model_name=args.model),
        data=make_squad_hf_dataset(llm.HFAutoModelForCausalLM.configure_tokenizer(args.model)),
        trainer=nl.Trainer(
            devices=args.devices,
            num_nodes=args.num_nodes,
            max_steps=args.max_steps,
            accelerator=args.accelerator,
            strategy=args.strategy,
            log_every_n_steps=1,
            limit_val_batches=0.0,
            num_sanity_val_steps=0,
            accumulate_grad_batches=10,
            gradient_clip_val=args.grad_clip,
            use_distributed_sampler=False,
            logger=wandb,
            callbacks=callbacks,
            precision="bf16",
        ),
        optim=fdl.build(llm.adam.pytorch_adam_with_flat_lr(lr=1e-5)),
        log=NeMoLogger(log_dir=args.ckpt_folder, use_datetime_version=False),
        peft=llm.peft.LoRA(
            target_modules=['*_proj'],
            dim=8,
        ),
    )


if __name__ == '__main__':
    main()
