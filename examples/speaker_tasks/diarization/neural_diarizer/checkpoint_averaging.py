# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

from nemo.utils import logging


def average_checkpoints_after_training(trainer: Trainer, model) -> str:
    """Average top-k checkpoints and export an averaged .nemo model.

    Only the global-zero rank performs I/O and averaging; all ranks
    synchronize via a barrier before returning.
    """
    output_path = _rank_zero_average_and_save(trainer, model) if trainer.is_global_zero else ""
    trainer.strategy.barrier("checkpoint_averaging_done")
    return output_path


def _get_checkpoint_callback(trainer: Trainer):
    """Return the first ModelCheckpoint callback attached to the trainer."""
    for callback in trainer.callbacks:
        if isinstance(callback, ModelCheckpoint):
            return callback
    return None


def _rank_zero_average_and_save(trainer: Trainer, model) -> str:
    """Load top-k checkpoints, average them, and save as .nemo (rank 0 only)."""
    checkpoint_callback = _get_checkpoint_callback(trainer)
    if checkpoint_callback is None:
        logging.warning("No ModelCheckpoint callback was found. Skipping post-training checkpoint averaging.")
        return ""

    best_k_models = getattr(checkpoint_callback, "best_k_models", None)
    if not best_k_models:
        logging.warning("No top-k checkpoints were found. Skipping post-training checkpoint averaging.")
        return ""

    reverse = getattr(checkpoint_callback, "mode", "max") == "max"
    ranked_ckpts = sorted(best_k_models.items(), key=lambda item: float(item[1]), reverse=reverse)
    selected_ckpts = [str(path) for path, _ in ranked_ckpts]

    selected_ckpts = [path for path in selected_ckpts if not path.endswith("-last.ckpt")]
    if checkpoint_callback.save_top_k not in (None, -1):
        selected_ckpts = selected_ckpts[: int(checkpoint_callback.save_top_k)]

    if len(selected_ckpts) == 0:
        logging.warning("No checkpoints remained after filtering. Skipping averaging.")
        return ""

    logging.info(f"Averaging {len(selected_ckpts)} checkpoints.")
    device = torch.device("cpu")
    avg_state = {}
    non_float_state = {}
    ref_dtypes = {}
    ref_keys = None

    for idx, path in enumerate(selected_ckpts):
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint

        if idx == 0:
            ref_keys = list(state_dict.keys())
            ref_key_set = set(ref_keys)
            for key, value in state_dict.items():
                tensor = value.detach().cpu()
                ref_dtypes[key] = tensor.dtype
                if torch.is_floating_point(tensor):
                    avg_state[key] = tensor.to(torch.float32)
                else:
                    non_float_state[key] = tensor
            continue

        if set(state_dict.keys()) != ref_key_set:
            raise RuntimeError(f"State dict mismatch while averaging checkpoints: {path}")

        for key in ref_keys:
            tensor = state_dict[key].detach().cpu()
            if torch.is_floating_point(tensor):
                avg_state[key] = avg_state[key] + tensor.to(torch.float32)

    num_ckpts = len(selected_ckpts)
    merged_state = {}
    for key in ref_keys:
        if key in avg_state:
            merged_state[key] = (avg_state[key] / num_ckpts).to(ref_dtypes[key])
        else:
            merged_state[key] = non_float_state[key]

    model.load_state_dict(merged_state, strict=True)

    output_name = "model_averaged.nemo"
    output_dir = str(getattr(checkpoint_callback, "dirpath", "") or os.path.dirname(selected_ckpts[0]))
    output_path = os.path.join(output_dir, output_name)

    model.save_to(output_path)
    logging.info(f"Averaged .nemo model exported to: {output_path}")
    return output_path
