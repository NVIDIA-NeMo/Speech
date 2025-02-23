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
import os

os.environ['HF_HOME'] = '/tmp/hf_home'
import re
import sys
from pathlib import Path

import fiddle as fdl
import lightning.pytorch as pl
import torch
from lightning.pytorch.loggers import WandbLogger
from transformers import AutoModelForCausalLM

from nemo import lightning as nl
from nemo.collections import llm
from nemo.lightning.pytorch.callbacks import JitConfig, JitTransform
from nemo.lightning.pytorch.strategies.utils import to_cpu

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
        name="nemo2_sft",
        log_dir=ckpt_folder,
        use_datetime_version=False,  # must be false if using auto resume
        ckpt=ckpt,
        wandb=None,
    )


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


def verify_sft_checkpoint_structure(path, has_io_bytes=False):
    expected_files = {
        'config.json',
        'generation_config.json',
        'special_tokens_map.json',
        'tokenizer.model',
        'tokenizer_config.json',
    }

    model_file = "model.safetensors"
    model_shard_prefix = "model-"
    index_file = "model.safetensors.index.json"

    # TODO(@akoumparouli): test cases where there are keys with io bytes
    # if has_io_bytes:
    #     expected_files.add('io_bytes.pt')

    ckpt_dir = Path(path)
    hf_weights = ckpt_dir / "hf_weights"
    assert hf_weights.exists(), f"Missing hf_weights directory: {hf_weights}"

    found_files = set()
    found_shards = []

    for file in hf_weights.glob('*'):
        if file.name.startswith(model_shard_prefix) and file.name.endswith(".safetensors"):
            found_shards.append(file.name)
        else:
            found_files.add(file.name)

    # Ensure mutual exclusivity of model.safetensors and sharded model files
    has_model_file = model_file in found_files
    has_sharded_files = len(found_shards) > 0

    assert not (
        has_model_file and has_sharded_files
    ), "Both model.safetensors and sharded model files exist, which is invalid."

    if has_sharded_files:
        assert index_file in found_files, f"Missing index file: {index_file}"
        shard_numbers = sorted([int(f.split('-')[1]) for f in found_shards])
        expected_shards = [
            f"model-{i:05d}-of-{shard_numbers[-1]:05d}.safetensors" for i in range(1, shard_numbers[-1] + 1)
        ]
        assert set(found_shards) == set(
            expected_shards
        ), f"Missing or extra shard files. Expected: {expected_shards}, Found: {found_shards}"
    else:
        assert has_model_file, "Missing model file(s)"

    assert expected_files.issubset(found_files), f"Missing files: {expected_files - found_files}"

    # Ensure trainer.pt exists
    assert (ckpt_dir / 'trainer.pt').exists(), "Missing trainer.pt file"

    # Validate context files
    context_files = ['model.yaml', 'io.json']
    for file in context_files:
        assert (ckpt_dir / 'context' / file).exists(), f"Missing context file: {file}"

    print("Checkpoint structure verification passed.")


class ValidateCheckpointRestoreCallback(pl.Callback):
    """This callback checks that the model weights and optimizer states are exactly restored
    from the checkpoint on the first training batch.
    """

    def __init__(self, check_weights: bool = True, check_optimizer: bool = True, tolerance: float = 1e-4):
        super().__init__()
        self.check_weights = check_weights
        self.check_optimizer = check_optimizer
        self.loaded_model_state = None
        self.loaded_optimizer_states = None
        self.tolerance = tolerance

    def on_load_checkpoint(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", checkpoint: dict) -> None:
        """
        Save the loaded model and optimizer states so we can compare them
        to the actual states after resuming.
        """
        self.loaded_model_state = checkpoint["state_dict"]
        self.loaded_optimizer_states = checkpoint["optimizer_states"]

    @property
    def has_loaded_checkpoint(self):
        return self.loaded_model_state is not None and self.loaded_optimizer_states is not None

    def on_train_batch_start(self, trainer: "pl.Trainer", lightning_module, batch, batch_idx) -> None:
        """
        Verify that the loaded model weights and optimizer state matches checkpoints'.
        """
        # ------------------------------------------------------------------
        # 1) Check the model weights
        # ------------------------------------------------------------------
        if self.check_weights and self.loaded_model_state is not None:
            for name, current_param in lightning_module.model.named_parameters():
                # Ensure parameter is in the loaded state dict
                if name not in self.loaded_model_state:
                    raise ValueError(f"Parameter '{name}' not found in checkpoint state dict.")

                # Compare current parameter to the checkpointed parameter
                checkpoint_param = self.loaded_model_state[name]
                current_param, checkpoint_param = to_cpu(current_param), to_cpu(checkpoint_param)
                if not torch.allclose(current_param, checkpoint_param, atol=self.tolerance):
                    rank = os.environ.get('LOCAL_RANK', '0')
                    diff = (current_param - checkpoint_param).view(-1).float().abs().cpu().max()
                    raise ValueError(f"{rank}: Model parameter '{name}' does not match the checkpointed value {diff}.")
            print("model weights match")
        else:
            print("Did not test model weights")
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
                        if isinstance(current_tensor_or_val, torch.Tensor):
                            if not torch.allclose(current_tensor_or_val.cpu(), loaded_tensor_or_val.cpu(), atol=1e-5):
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
        else:
            print("did not test optim weights")
        # After the first batch, we don't need to check again
        # (Remove if you need ongoing checks)
        if not self.has_loaded_checkpoint:
            trainer.callbacks.remove(self)
        else:
            print("All weights match")
            sys.exit(0)


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
    parser.add_argument('--ckpt-folder', type=str, default='/tmp/nemo_automodel_sft/')
    parser.add_argument('--use-torch-jit', action='store_true')
    parser.add_argument('--auto-resume', action='store_true')
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
        from nemo.lightning.pytorch.accelerate.transformer_engine import TEConfig

        model_accelerator = TEConfig(fp8_autocast=args.fp8_autocast)

    callbacks = []
    if args.use_torch_jit:
        jit_config = JitConfig(use_torch=True, torch_kwargs={'dynamic': False}, use_thunder=False)
        callbacks = [JitTransform(jit_config)]

    if args.auto_resume:
        callbacks.append(ValidateCheckpointRestoreCallback())

    class ZeroInitHFAutoModelForCausalLM(llm.HFAutoModelForCausalLM):
        def configure_module(self, *args, **kwargs):
            ans = super().configure_model(*args, **kwargs)
            for param in self.parameters():
                param.fill_(0)
            return ans

    model_cls = ZeroInitHFAutoModelForCausalLM if args.auto_resume else llm.HFAutoModelForCausalLM
    model = model_cls(model_name=args.model, model_accelerator=model_accelerator)

    strategy = make_strategy(args.strategy, model, args.devices, args.num_nodes, False)

    resume = (
        nl.AutoResume(
            resume_if_exists=True,
            resume_ignore_no_checkpoint=False,
        )
        if args.auto_resume
        else None
    )
    args.max_steps += int(resume is not None)
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
        data=make_squad_hf_dataset(llm.HFAutoModelForCausalLM.configure_tokenizer(args.model)),
        trainer=trainer,
        optim=fdl.build(llm.adam.pytorch_adam_with_flat_lr(lr=1e-5)),
        log=logger(args.ckpt_folder),
        resume=resume,
    )

    del model
    del trainer

    path = get_latest_checkpoint(args.ckpt_folder)
    verify_sft_checkpoint_structure(path, model_accelerator is not None)

    ans = AutoModelForCausalLM.from_pretrained(path / "hf_weights", output_loading_info=True)
    assert len(ans[1]['missing_keys']) == 0, ("NOT LOADABLE #1", ans)
    assert len(ans[1]['mismatched_keys']) == 0, ("NOT LOADABLE #2", ans)


if __name__ == '__main__':
    main()
