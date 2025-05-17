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

import tempfile

import fiddle as fdl
import lightning.pytorch as pl

from nemo import lightning as nl
from nemo.automodel.dist_utils import FirstRankPerNode
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


def get_chat_template(tokenizer):
    # attempt to unwrap NeMo's tokenizer wrapper and check if wrapped tokenizer has chat_template
    tmp_tokenizer = getattr(tokenizer, 'tokenizer', tokenizer)
    has_chat_template = getattr(tmp_tokenizer, 'chat_template', None) is not None
    if has_chat_template:
        return tmp_tokenizer, getattr(tmp_tokenizer, 'eos_token_id', None), has_chat_template
    else:
        return tokenizer, getattr(tokenizer, 'eos_id', None), has_chat_template


@FirstRankPerNode()
def make_squad_hf_dataset(
    tokenizer, micro_batch_size, seq_length=None, limit_dataset_samples=None, start_of_turn_token=None, fp8=False
):
    tokenizer, eos_token_id, has_chat_template = get_chat_template(tokenizer)

    def pad_to_seq_length(sample):
        seq_pad_len_ar = max(0, seq_length - len(next(iter(sample.values()))))
        return {k: v + [eos_token_id if v != 'loss_mask' else 0] * seq_pad_len_ar for k, v in sample.items()}

    def formatting_prompts_func(example):
        formatted_text = [
            f"{example['context']} {example['question']} ",
            example['answers']['text'][0].strip(),
        ]
        context_ids, answer_ids = list(map(tokenizer.text_to_ids, formatted_text))
        bos_id = getattr(tokenizer, 'bos_id', None)
        eos_id = getattr(tokenizer, 'eos_id', None)
        if len(context_ids) > 0 and bos_id is not None and context_ids[0] != bos_id:
            context_ids.insert(0, bos_id)
        if len(answer_ids) > 0 and eos_id is not None and answer_ids[-1] != eos_id:
            answer_ids.append(eos_id)

        input_ids = context_ids + answer_ids
        return dict(
            input_ids=input_ids,
            labels=input_ids[1:] + [eos_token_id or input_ids[-1]],
            loss_mask=[0] * len(context_ids) + [1] * len(answer_ids),
        )

    def formatting_prompts_func_with_chat_template(example, start_of_turn_token=None):
        formatted_text = [
            {'role': 'user', 'content': f"{example['context']} {example['question']}"},
            {'role': 'assistant', 'content': example['answers']['text'][0].strip()},
        ]
        input_ids = tokenizer.apply_chat_template(formatted_text)
        if isinstance(start_of_turn_token, str):
            start_of_turn_token_id = tokenizer(start_of_turn_token, add_special_tokens=False)['input_ids'][0]
            first_start_of_turn_token_id = input_ids.index(start_of_turn_token_id)
            response_start = input_ids.index(start_of_turn_token_id, first_start_of_turn_token_id + 1) + 1
        else:
            response_start = 0
        loss_mask = [0] * response_start + [1] * (len(input_ids) - response_start)
        return dict(
            input_ids=input_ids,
            labels=input_ids[1:] + [getattr(tokenizer, 'eos_token_id', None) or input_ids[-1]],
            loss_mask=loss_mask,
        )

    splits = ['train', 'validation']
    if limit_dataset_samples is not None:
        assert isinstance(limit_dataset_samples, int), "Expected limit_dataset_samples to be an int"
        splits = list(map(lambda x: f'{x}[:{limit_dataset_samples}]', splits))

    fmt_fn = formatting_prompts_func
    if has_chat_template:
        fmt_fn = lambda x: formatting_prompts_func_with_chat_template(x, start_of_turn_token)
    if isinstance(seq_length, int):
        fmt_fn_ = fmt_fn
        fmt_fn = lambda x: pad_to_seq_length(fmt_fn_(x))

    datamodule = llm.HFDatasetDataModule(
        "rajpurkar/squad",
        split=splits,
        micro_batch_size=micro_batch_size,
        pad_token_id=getattr(tokenizer, 'eos_id', 0) or 0,
        pad_seq_len_divisible=16 if fp8 else None,  # FP8 training requires seq length to be divisible by 16.
    )
    datamodule.map(
        fmt_fn,
        batched=False,
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
            find_unused_parameters=True,
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
    parser.add_argument(
        '--attn-implementation',
        type=str,
        default="sdpa",
        choices=["flash_attention_2", "sdpa", "eager"],
        help='Attention implementation to use. Default: sdpa',
    )
    parser.add_argument('--wandb-project', type=str, default=None, help='Wandb project to use')
    parser.add_argument('--use-torch-jit', action='store_true', help='Enables torch.compile on model')
    parser.add_argument('--auto-resume', action='store_true', help='Enables autoresume from a previous training job')
    parser.add_argument('--enable-cpu-offload', action='store_true', help='Enabled cpu offloading; requires FSDP2')
    parser.add_argument('--liger', action='store_true', help='Enables Liger-Kernels')
    parser.add_argument('--enable-grad-ckpt', action='store_true', help='Enables gradient checkpoint')
    parser.add_argument(
        '--ckpt-folder', type=str, default=tempfile.TemporaryDirectory().name, help='Directory to save checkpoints'
    )
    parser.add_argument(
        '--batch-size',
        '--micro-batch-size',
        dest='batch_size',
        default=1,
        type=int,
        help='Micro batch size to use for training.',
    )
    parser.add_argument(
        '--trust-remote-code',
        action='store_true',
        help='Enables trust_remote_code to load HF models with unverified sources',
    )
    parser.add_argument('--fp8', action='store_true', help='Enables fp8 training')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--no-lce', action='store_false', help='Disables LCE')
    parser.add_argument('--start-of-turn-token', default=None, help='Chat turn token')
    parser.add_argument(
        '--limit-dataset-samples',
        type=int,
        default=None,
        help='If set will limit num of dataset samples. Default None (disabled)',
    )
    parser.add_argument('--seq-length', default=None, type=int, help='Sequence length to use for training')
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
        attn_implementation=args.attn_implementation,
        trust_remote_code=args.trust_remote_code,
        use_liger_kernel=args.liger,
        load_in_4bit=args.load_in_4bit,
        enable_grad_ckpt=args.enable_grad_ckpt,
        use_linear_ce_loss=not args.no_lce,
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
    dataset = make_squad_hf_dataset(
        tokenizer=model.tokenizer,
        micro_batch_size=args.batch_size,
        seq_length=args.seq_length,
        start_of_turn_token=args.start_of_turn_token,
        limit_dataset_samples=args.limit_dataset_samples,
        fp8=args.fp8,
    )
    llm.api.finetune(
        model=model,
        data=dataset,
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
