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
import torch
from packaging.version import Version as PkgVersion
from utils import get_torch_version_str

from nemo import lightning as nl
from nemo.collections import llm
from nemo.lightning.pytorch.accelerate.transformer_engine import is_te_accelerated

DATA_PATH = '/home/TestData/lite/hf_cache/squad/'


def make_squad_hf_dataset(tokenizer, data_path=DATA_PATH):
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

    datamodule = llm.HFDatasetDataModule(data_path, split="train[:10]", pad_token_id=tokenizer.eos_id)
    datamodule.map(
        formatting_prompts_func,
        batched=False,
        batch_size=2,
        remove_columns=["id", "title", "context", "question", 'answers'],
    )
    return datamodule


if __name__ == '__main__':
    if PkgVersion(get_torch_version_str()) >= PkgVersion("2.4"):
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument('--model', default='meta-llama/Llama-3.2-1B')
        parser.add_argument('--devices', default=2, type=int)
        parser.add_argument('--accelerator', default='gpu', choices=['gpu'])
        parser.add_argument('--model-accelerator', default=None, choices=['te'])
        parser.add_argument('--max-steps', type=int, default=5)
        parser.add_argument("--fp8-autocast", default=False, action='store_true')
        parser.add_argument('--wandb-project', type=str, default=None)
        parser.add_argument('--model-save-path', type=str, default=None)
        args = parser.parse_args()

        wandb = None
        if args.wandb_project is not None:
            from lightning.pytorch.loggers import WandbLogger

            model = '_'.join(args.model.split('/')[-2:])
            wandb = WandbLogger(
                project=args.wandb_project,
                name=f'{model}_dev{args.devices}_strat_{args.strategy}',
            )
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

        llm.api.finetune(
            model=model,
            data=make_squad_hf_dataset(tokenizer),
            trainer=nl.Trainer(
                devices=args.devices,
                max_steps=args.max_steps,
                accelerator=args.accelerator,
                strategy=nl.FSDP2Strategy(data_parallel_size=args.devices),
                log_every_n_steps=1,
                limit_val_batches=0.0,
                num_sanity_val_steps=0,
                accumulate_grad_batches=1,
                gradient_clip_val=grad_clip,
                use_distributed_sampler=use_dist_samp,
                callbacks=[],
                logger=wandb,
            ),
            optim=fdl.build(llm.adam.pytorch_adam_with_flat_lr(lr=1e-5)),
            log=None,
        )

        # Check memory usage compared to non-parallelized version
        assert (
            torch.cuda.max_memory_allocated(device=None) / 1024 / 1024 < 29326
        ), f"using {torch.cuda.max_memory_allocated(device=None)/1024/1024} MB, larger than 29326 MB when not using parallelization."

        if args.model_accelerator:
            if args.model_accelerator == "te":
                te_acc = is_te_accelerated(model.model)
                assert te_acc, "Transformer Engine acceleration was unsuccessful"
                print("TE Accelerated: ", te_acc)

        if args.model_save_path is not None:
            model.save_pretrained(args.model_save_path)
