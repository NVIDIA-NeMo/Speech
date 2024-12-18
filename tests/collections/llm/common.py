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
from dataclasses import dataclass

import lightning.pytorch as pl
import nemo_run as run
import torch

from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.common.tokenizers import SentencePieceTokenizer
from nemo.utils import logging


def train_data(
    data_path: str, tokenizer_path: str, index_mapping_dir: str, seq_length: int
) -> llm.PreTrainingDataModule:
    """Single shard dataset tokenized by SentencePiece"""
    tokenizer = run.Config(SentencePieceTokenizer, model_path=tokenizer_path)
    return run.Config(
        llm.PreTrainingDataModule,
        paths=data_path,
        tokenizer=tokenizer,
        seq_length=seq_length,
        micro_batch_size=4,
        global_batch_size=32,
        seed=1234,
        index_mapping_dir=index_mapping_dir,
    )


def small_llama_cfg(seq_length: int) -> llm.GPTConfig:
    """Small 145m model"""
    return run.Config(
        llm.Llama3Config8B,
        rotary_base=500_000,
        seq_length=seq_length,
        num_layers=12,
        hidden_size=768,
        ffn_hidden_size=2688,
        num_attention_heads=16,
        init_method_std=0.023,
    )


class StopBeforeEnd(pl.Callback):
    """Preemptively stop training at a given global step. Allows stopping training before reaching
    the max steps. Useful for testing checkpoint save and resume.

    Args:
        stop_on_step (int): Stop training when trainer.global_step reaches this value.
            Checked at the start of every step.
    """

    def __init__(self, stop_on_step: int):
        self.stop_on_step = stop_on_step

    def on_train_batch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs, batch, batch_idx
    ) -> None:
        if trainer.global_step >= self.stop_on_step:
            logging.info(f"Global step {trainer.global_step} >= {self.stop_on_step}, signaling Trainer to stop.")
            trainer.should_stop = True
            # skip EarlyStopping validation unless val_check_interval met
            if trainer.global_step % trainer.val_check_interval != 0:
                trainer.limit_val_batches = 0


class MCoreModelAttributeValidator(pl.Callback):
    """Walk through submodules and verify user-specified attributes like parallelisms."""

    def __init__(self, attr_dict: dict):
        super().__init__()
        self.attr_dict = attr_dict

    def _check_attrs(self, target):
        for k, v in self.attr_dict.items():
            if hasattr(target, k):
                model_val = getattr(target, k)
                assert (
                    model_val == v
                ), f"Key {k} for model ({model_val}) does not match {v} from provided attribute mapping."

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        def walk_fn(module: torch.nn.Module) -> torch.nn.Module:
            # self._check_attrs(module) # TE DPA has 'sequence_parallel' attribute that is always False. Checking module config should be sufficient
            if hasattr(module, "config"):
                self._check_attrs(module.config)

            return module

        trainer.model.walk(walk_fn)


class MiscAttributeValidator(pl.Callback):
    """Place for any miscellaneous attribute assertions. Extend as needed."""

    def __init__(self, attr_dict: dict):
        super().__init__()
        self.attr_dict = attr_dict

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if 'max_steps' in self.attr_dict:
            sched_max = trainer.model.optim.lr_scheduler._scheduler['lr_scheduler']['scheduler'].max_steps
            assert (
                trainer.max_steps == self.attr_dict['max_steps']
            ), f"Trainer max_steps {trainer.max_steps} did not match provided {self.attr_dict['max_steps']}"
            assert (
                sched_max == self.attr_dict['max_steps']
            ), f"Scheduler max_steps {sched_max} did not match provided {self.attr_dict['max_steps']}"

    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if 'stop_on_step' in self.attr_dict:
            total_steps = trainer.fit_loop.epoch_loop.batch_progress.total.completed
            assert total_steps == self.attr_dict['stop_on_step']


def verify_distcp_dir(ckpt_path: str) -> None:
    ckpt_name = os.path.basename(ckpt_path)

    weights_dir = os.path.join(ckpt_path, 'weights')
    assert os.path.isdir(weights_dir), f"Weights not found in checkpoint {ckpt_name}"
    assert os.path.isfile(os.path.join(weights_dir, 'common.pt')), f"No 'common.pt' file in checkpoint {ckpt_name}"
    assert os.path.isfile(
        os.path.join(weights_dir, 'metadata.json')
    ), f"No 'metadata.json' file in checkpoint {ckpt_name}"

    shards = [shard for shard in os.listdir(weights_dir) if shard.endswith('.distcp')]
    world_size = torch.distributed.get_world_size()
    assert (
        len(shards) == 2 * world_size
    ), f"Wrong number of .distcp files, Expected: {2*world_size} Found: {len(shards)}"


def verify_ckpt_dir(
    model_ckpt: nl.ModelCheckpoint, max_steps: int, val_check_interval: int, exp_dir: str, dist_ckpts: bool = True
) -> None:
    """Ensures that the provided checkpoint directory has
    - correct number of checkpoints
    - no more than top-k checkpoints
    - no unfinished checkpoints
    - a checkpoint for the last step
    - all checkpoints in the correct format
    """

    ckpt_dir = os.path.join(exp_dir, 'checkpoints')
    ckpts = os.listdir(ckpt_dir)

    if model_ckpt.save_last:
        assert any([c.endswith('-last') for c in ckpts]), "No -last checkpoint found after training"

    expected_count = (max_steps // val_check_interval) + model_ckpt.save_last
    if model_ckpt.save_top_k > 0:
        assert (
            len(ckpts) == expected_count or len(ckpts) == model_ckpt.save_top_k + model_ckpt.save_last
        ), f"Expected {expected_count} checkpoints or at most top {model_ckpt.save_top_k} checkpoints besides '-last'"
    else:
        assert len(ckpts) == expected_count, f"Expected {expected_count} checkpoints"

    for ckpt_name in ckpts:
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)

        assert (
            '-unfinished' not in ckpt_name
        ), f"Unfinished checkpoint found. Something went wrong with saving checkpoint {ckpt_name}"

        if ckpt_name.endswith('-last') and 'step' in model_ckpt.filename:
            assert f'step={max_steps-1}' in ckpt_name, f"Last checkpoint {ckpt_name} not for final step {max_steps}"

        if dist_ckpts:
            assert os.path.isdir(ckpt_path), "Checkpoint is not correct type"
            verify_distcp_dir(ckpt_path)
        else:
            assert os.path.isfile(ckpt_path), "Checkpoint is not correct type"


def create_verify_precision(precision: torch.dtype):
    def verify_precision(tensor: torch.Tensor) -> None:
        assert tensor.dtype == precision

    return verify_precision


@dataclass
class Llama3ConfigCI(llm.Llama3Config8B):
    seq_length: int = 2048
    num_layers: int = 2
    hidden_size: int = 768
    ffn_hidden_size: int = 3072
    num_attention_heads: int = 8
