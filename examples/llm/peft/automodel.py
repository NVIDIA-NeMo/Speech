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
import lightning.pytorch as pl

from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.llm.recipes.optim.adam import pytorch_adam_with_cosine_annealing
from nemo.lightning.pytorch.callbacks import JitConfig, JitTransform


# Run this example with torchrun, for example:
# torchrun --nproc-per-node=8 \
#   examples/llm/peft/automodel.py \
#   --strategy fsdp2 \
#   --devices 8 \
#   --model meta-llama/Llama-3.2-1B \
#   --ckpt-folder "output"
#
# Note: ensure that the --nproc-per-node and --devices values match.


def make_squad_hf_dataset(tokenizer, batch_size, fp8=False):
    def formatting_prompts_func(example):
        formatted_text = [
            f"Context: {example['context']} Question: {example['question']} Answer:",
            f" {example['answers']['text'][0].strip()}",
        ]
        context_ids, answer_ids = list(map(tokenizer.text_to_ids, formatted_text))
        if len(context_ids) > 0 and context_ids[0] != tokenizer.bos_id and tokenizer.bos_id is not None:
            context_ids.insert(0, tokenizer.bos_id)
        if len(answer_ids) > 0 and answer_ids[-1] != tokenizer.eos_id and tokenizer.eos_id is not None:
            answer_ids.append(tokenizer.eos_id)

        return dict(
            labels=(context_ids + answer_ids)[1:],
            input_ids=(context_ids + answer_ids)[:-1],
            loss_mask=[0] * (len(context_ids) - 1) + [1] * len(answer_ids),
        )

    datamodule = llm.HFDatasetDataModule(
        "rajpurkar/squad",
        split="train",
        micro_batch_size=batch_size,
        pad_token_id=tokenizer.eos_id or 0,
        pad_seq_len_divisible=16 if fp8 else None,  # FP8 training requires seq length to be divisible by 16.
    )
    datamodule.map(
        formatting_prompts_func,
        batched=False,
        batch_size=2,
        remove_columns=["id", "title", "context", "question", 'answers'],
    )
    return datamodule


def make_strategy(strategy, model, devices, num_nodes, adapter_only=False, enable_cpu_offload=False):
    if strategy == 'auto':
        return pl.strategies.SingleDeviceStrategy(
            device='cuda:0',
            checkpoint_io=model.make_checkpoint_io(adapter_only=adapter_only),
        )
    elif strategy == 'ddp':
        return pl.strategies.DDPStrategy(
            checkpoint_io=model.make_checkpoint_io(adapter_only=adapter_only),
        )
    elif strategy == 'fsdp2':
        offload_policy = None
        if enable_cpu_offload:
            from nemo.lightning.pytorch.strategies.fsdp2_strategy import HAS_CPU_OFFLOAD_POLICY, CPUOffloadPolicy

            assert HAS_CPU_OFFLOAD_POLICY, "Could not import offload policy"
            offload_policy = CPUOffloadPolicy()

        return nl.FSDP2Strategy(
            data_parallel_size=devices * num_nodes,
            tensor_parallel_size=1,
            checkpoint_io=model.make_checkpoint_io(adapter_only=adapter_only),
            offload_policy=offload_policy,
        )
    else:
        raise NotImplementedError("Encountered unknown strategy")


def logger(ckpt_folder, save_every_n_train_steps) -> nl.NeMoLogger:
    ckpt = nl.ModelCheckpoint(
        save_last=True,
        every_n_train_steps=save_every_n_train_steps,
        monitor="reduced_train_loss",
        save_top_k=1,
        save_on_train_epoch_end=True,
        save_optim_on_train_end=True,
    )

    return nl.NeMoLogger(
        name="nemo2_peft",
        log_dir=ckpt_folder,
        use_datetime_version=False,  # must be false if using auto resume
        ckpt=ckpt,
        wandb=None,
    )


def main():
    """Example script to run PEFT with a HF transformers-instantiated model on squad."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.2-1B', help='Hugging Face model-id to use')
    parser.add_argument(
        '--strategy',
        type=str,
        default='auto',
        choices=['auto', 'ddp', 'fsdp2'],
        help='Training strategy e.g. ddp/fsdp2/single-gpu',
    )
    parser.add_argument('--devices', type=int, default=1, help='Number of GPUs to use')
    parser.add_argument('--num-nodes', type=int, default=1, help='Number of Nodes to use; to be used with torchrun')
    parser.add_argument('--use-te-optimizer', action='store_true', help='Use TE optimizer')
    parser.add_argument('--grad-clip', type=float, default=1.0, help='Grad clip value')
    parser.add_argument(
        '--accumulate_grad_batches', type=int, default=10, help='Number of batches to accumulate gradient over'
    )
    parser.add_argument('--load-in-4bit', action='store_true', help='Use 4-bit quantization for e.g. for qlora')
    parser.add_argument('--max-steps', type=int, default=100, help='Maximum number of training steps')
    parser.add_argument('--wandb-project', type=str, default=None, help='Wandb project to use')
    parser.add_argument('--use-torch-jit', action='store_true', help='Enables torch.compile on model')
    parser.add_argument('--auto-resume', action='store_true', help='Enables autoresume from a previous training job')
    parser.add_argument('--enable-cpu-offload', action='store_true', help='Enabled cpu offloading; requires FSDP2')
    parser.add_argument('--liger', action='store_true', help='Enables Liger-Kernels')
    parser.add_argument('--enable-grad-ckpt', action='store_true', help='Enables gradient checkpoint')
    parser.add_argument(
        '--ckpt-folder', type=str, default=tempfile.TemporaryDirectory().name, help='Directory to save checkpoints'
    )
    parser.add_argument('--batch-size', default=1, type=int, help='Batch size to use for training')
    parser.add_argument(
        '--trust-remote-code',
        action='store_true',
        help='Enables trust_remote_code to load HF models with unverified sources',
    )
    parser.add_argument('--fp8', action='store_true', help='Enables fp8 training')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    args = parser.parse_args()

    # CPUOffload WA for known issue
    if args.enable_cpu_offload and args.use_te_optimizer:
        args.use_te_optimizer = False

    wandb = None
    if args.wandb_project is not None:
        from lightning.pytorch.loggers import WandbLogger

        model = '_'.join(args.model.split('/')[-2:])
        wandb = WandbLogger(
            project=args.wandb_project,
            name=f'{model}_dev{args.devices}_strat_{args.strategy}',
        )

    callbacks = []
    if args.use_torch_jit:
        jit_config = JitConfig(use_torch=True, torch_kwargs={'dynamic': True}, use_thunder=False)
        callbacks = [JitTransform(jit_config)]

    if args.use_te_optimizer:
        # Use TE optimizer
        # Faster convergence but may lead to memory issues
        optimizer = fdl.build(llm.adam.te_adam_with_flat_lr(lr=args.lr))
    else:
        optimizer = fdl.build(pytorch_adam_with_cosine_annealing(max_lr=args.lr, warmup_steps=50))

    if args.fp8:
        from nemo.lightning.pytorch.accelerate.transformer_engine import TEConfig

        model_accelerator = TEConfig(fp8_autocast=True)
    else:
        model_accelerator = None
    model = llm.HFAutoModelForCausalLM(
        model_name=args.model,
        model_accelerator=model_accelerator,
        trust_remote_code=args.trust_remote_code,
        use_liger_kernel=args.liger,
        load_in_4bit=args.load_in_4bit,
        enable_grad_ckpt=args.enable_grad_ckpt,
    )
    strategy = make_strategy(args.strategy, model, args.devices, args.num_nodes, True, args.enable_cpu_offload)

    resume = (
        nl.AutoResume(
            resume_if_exists=True,
            resume_ignore_no_checkpoint=False,
        )
        if args.auto_resume
        else None
    )

    llm.api.finetune(
        model=model,
        data=make_squad_hf_dataset(model.tokenizer, args.batch_size, args.fp8),
        trainer=nl.Trainer(
            devices=args.devices,
            num_nodes=args.num_nodes,
            max_steps=args.max_steps,
            accelerator='gpu',
            strategy=strategy,
            log_every_n_steps=1,
            limit_val_batches=0.0,
            num_sanity_val_steps=0,
            accumulate_grad_batches=args.accumulate_grad_batches,
            gradient_clip_val=args.grad_clip,
            use_distributed_sampler=False,
            logger=wandb,
            callbacks=callbacks,
            precision="bf16-mixed",
        ),
        optim=optimizer,
        log=logger(args.ckpt_folder, args.max_steps // 2),
        peft=llm.peft.LoRA(
            target_modules=['*_proj'],
            dim=8,
        ),
        resume=resume,
    )


if __name__ == '__main__':
    main()
