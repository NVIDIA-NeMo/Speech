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

import os
import re
import sys
import tempfile
from pathlib import Path

import fiddle as fdl
import lightning.pytorch as pl
import torch
from lightning.pytorch.loggers import WandbLogger
from transformers import AutoModelForCausalLM

from nemo import lightning as nl
from nemo.collections import llm
from nemo.lightning import NeMoLogger
from nemo.lightning.pytorch.callbacks import JitConfig, JitTransform
from nemo.lightning.pytorch.strategies.utils import to_cpu

DATA_PATH = '/home/TestData/lite/hf_cache/squad/'


def make_squad_hf_dataset(data_path, tokenizer):
    EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN

    def formatting_prompts_func(examples):
        alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    {}

    ### Input:
    {}

    ### Response:
    {}"""
        instruction = examples["context"]
        input = examples["question"]
        output = examples["answers"]['text']
        if isinstance(output, list):
            output = output[0]
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        ans = tokenizer(text)
        ans['labels'] = ans['input_ids']
        return ans

    tokenizer = getattr(tokenizer, 'tokenizer', tokenizer)
    datamodule = llm.HFDatasetDataModule(data_path, split="train[:100]", pad_token_id=tokenizer.eos_token_id)
    datamodule.map(
        formatting_prompts_func,
        batched=False,
        batch_size=2,
        remove_columns=["id", "title", "context", "question", 'answers'],
    )

    return datamodule


def make_strategy(strategy, model, devices, num_nodes, adapter_only=False):
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
        return nl.FSDP2Strategy(
            data_parallel_size=devices * num_nodes,
            tensor_parallel_size=1,
            checkpoint_io=model.make_checkpoint_io(adapter_only=adapter_only),
        )
    else:
        raise NotImplementedError("Encountered unknown strategy")


def logger(ckpt_folder) -> nl.NeMoLogger:
    ckpt = nl.ModelCheckpoint(
        save_last=True,
        every_n_train_steps=1,
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


class ValidateCheckpointRestoreCallback(pl.Callback):
    """This callback checks that the model weights and optimizer states are exactly restored
    from the checkpoint on the first training batch.
    """

    def __init__(self, check_weights: bool = True, check_optimizer: bool = True):
        super().__init__()
        self.check_weights = check_weights
        self.check_optimizer = check_optimizer
        self.loaded_model_state = None
        self.loaded_optimizer_states = None

    def on_load_checkpoint(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", checkpoint: dict) -> None:
        """
        Save the loaded model and optimizer states so we can compare them
        to the actual states after resuming.
        """
        self.loaded_model_state = checkpoint["state_dict"]
        print('self.loaded_model_state.keys= ' + str(self.loaded_model_state.keys()))
        self.loaded_optimizer_states = checkpoint["optimizer_states"]

    def on_train_batch_start(self, trainer: "pl.Trainer", lightning_module, batch, batch_idx) -> None:
        """
        Verify that the loaded model weights and optimizer state matches checkpoints'.
        """
        # ------------------------------------------------------------------
        # 1) Check the model weights
        # ------------------------------------------------------------------
        if self.check_weights and self.loaded_model_state is not None:
            for name, current_param in lightning_module.named_parameters():
                # Ensure parameter is in the loaded state dict
                print(name, current_param.requires_grad)
                if current_param.requires_grad == False:
                    assert not current_param in self.loaded_model_state
                elif name not in self.loaded_model_state:
                    raise ValueError(f"Parameter '{name}' not found in checkpoint state dict.")
                else:
                    # Compare current parameter to the checkpointed parameter
                    checkpoint_param = self.loaded_model_state[name]
                    current_param, checkpoint_param = to_cpu(current_param), to_cpu(checkpoint_param)
                    if not torch.allclose(current_param, checkpoint_param, atol=1e-4):
                        raise ValueError(f"Model parameter '{name}' does not match the checkpointed value. {diff}")
            print("model weights match")

        # ------------------------------------------------------------------
        # 2) Check the optimizer states
        # ------------------------------------------------------------------
        if self.check_optimizer and self.loaded_optimizer_states is not None:
            optimizers = trainer.optimizers
            if len(optimizers) != len(self.loaded_optimizer_states):
                raise ValueError(
                    f"Number of optimizers ({len(optimizers)}) does not match "
                    f"the checkpoint ({len(self.loaded_optimizer_states)})."
                )

            for optimizer, loaded_opt_state in zip(optimizers, self.loaded_optimizer_states):
                current_opt_state = optimizer.state_dict()

                # Compare keys in the optimizer state
                if current_opt_state.keys() != loaded_opt_state.keys():
                    raise ValueError("Mismatch in optimizer state keys between current state and checkpoint.")

                # Compare each parameter state group
                for param_id, param_state in current_opt_state["state"].items():
                    loaded_param_state = loaded_opt_state["state"][param_id]
                    for state_key, current_tensor_or_val in param_state.items():
                        if state_key not in loaded_param_state:
                            raise ValueError(f"Key '{state_key}' missing in the loaded optimizer state.")

                        loaded_tensor_or_val = loaded_param_state[state_key]
                        # If it's a tensor, compare contents

                        current_tensor_or_val, loaded_tensor_or_val = to_cpu(current_tensor_or_val), to_cpu(
                            loaded_tensor_or_val
                        )
                        if isinstance(current_tensor_or_val, torch.Tensor):
                            if not torch.allclose(current_tensor_or_val, loaded_tensor_or_val, atol=1e-7):
                                raise ValueError(
                                    f"Optimizer state for param_id={param_id}, "
                                    f"key='{state_key}' does not match the checkpoint."
                                )

                        # Otherwise, compare directly (e.g., scalar values)
                        else:
                            if current_tensor_or_val != loaded_tensor_or_val:
                                raise ValueError(
                                    f"Optimizer state for param_id={param_id}, "
                                    f"key='{state_key}' differs from checkpoint."
                                )
            print("optim weights match")

        # After the first batch, we don't need to check again
        # (Remove if you need ongoing checks)
        trainer.callbacks.remove(self)
        print("All weights match")
        sys.exit(0)


def get_latest_checkpoint(base_dir):
    latest_checkpoint = None
    max_epoch = -1
    max_step = -1

    pattern = re.compile(r"[a-z0-9_]+--reduced_train_loss=([\d\.]+)-epoch=(\d+)-step=(\d+)-last")

    # Traverse through the base directory
    for root, dirs, _ in os.walk(base_dir):
        for dir_name in dirs:
            match = pattern.match(dir_name)
            if match:
                loss, epoch, step = map(float, match.groups())  # Convert to float/int
                epoch, step = int(epoch), int(step)

                # Update the latest checkpoint based on epoch and step
                if (epoch > max_epoch) or (epoch == max_epoch and step > max_step):
                    max_epoch = epoch
                    max_step = step
                    latest_checkpoint = os.path.join(root, dir_name)

    return Path(latest_checkpoint)


def verify_peft_checkpoint_structure(path):
    expected_files = set(
        [
            'adapter_model.safetensors',
            'adapter_config.json',
        ]
    )
    ckpt_dir = Path(path)
    hf_weights = ckpt_dir / "hf_adapter"
    assert hf_weights.exists(), str(hf_weights)
    for file in hf_weights.glob('*'):
        assert file.name in expected_files, file
        expected_files.remove(file.name)
    assert len(expected_files) == 0

    assert (ckpt_dir / 'trainer.pt').exists()

    context_files = ['model.yaml', 'io.json']
    for file in context_files:
        assert (ckpt_dir / 'context' / file).exists()


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
    parser.add_argument('--auto-resume', action='store_true')
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

    model = llm.HFAutoModelForCausalLM(model_name=args.model)
    strategy = make_strategy(args.strategy, model, args.devices, args.num_nodes, True)

    if args.auto_resume:
        callbacks.append(ValidateCheckpointRestoreCallback())

    resume = (
        nl.AutoResume(
            resume_if_exists=True,
            resume_ignore_no_checkpoint=False,
        )
        if args.auto_resume
        else None
    )
    trainer = nl.Trainer(
        devices=args.devices,
        num_nodes=args.num_nodes,
        max_steps=args.max_steps,
        accelerator=args.accelerator,
        strategy=strategy,
        log_every_n_steps=1,
        limit_val_batches=0.0,
        num_sanity_val_steps=0,
        accumulate_grad_batches=1,
        gradient_clip_val=args.grad_clip,
        use_distributed_sampler=False,
        logger=wandb,
        callbacks=callbacks,
        precision="bf16",
    )

    llm.api.finetune(
        model=model,
        data=make_squad_hf_dataset(DATA_PATH, llm.HFAutoModelForCausalLM.configure_tokenizer(args.model)),
        trainer=trainer,
        optim=fdl.build(llm.adam.pytorch_adam_with_flat_lr(lr=1e-5)),
        log=logger(args.ckpt_folder),
        peft=llm.peft.LoRA(
            target_modules=['*_proj'],
            dim=8,
        ),
        resume=resume,
    )

    del model
    del trainer

    path = get_latest_checkpoint(args.ckpt_folder)
    verify_peft_checkpoint_structure(path)

    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.load_adapter(f'{path}/hf_adapter')


if __name__ == '__main__':
    main()
