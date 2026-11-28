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
"""Resuming from a ``NeMoModelCheckpoint`` saved mid-epoch keeps Lightning's epoch boundaries in place."""

from pathlib import Path

import lightning.pytorch as pl
import pytest
import torch
from torchdata.stateful_dataloader import StatefulDataLoader

from nemo.utils.callbacks.nemo_model_checkpoint import NeMoModelCheckpoint

LIMIT, SPLIT, TOTAL = 5, 3, 12


@pytest.mark.unit
def test_resume_from_mid_epoch_checkpoint_keeps_epoch_boundaries(tmp_path):
    uninterrupted = _fit(tmp_path / "a", max_steps=TOTAL)
    ckpt_path = tmp_path / "a" / "checkpoints" / f"step={SPLIT}.ckpt"
    assert ckpt_path.exists()

    resumed = _fit(tmp_path / "b", max_steps=TOTAL, ckpt_path=ckpt_path)

    # A checkpoint saved from on_train_batch_end must resume at the next batch; one extra batch in the resumed epoch
    # would shift every later epoch boundary.
    assert resumed == uninterrupted[SPLIT:]


class _RangeDataset(torch.utils.data.Dataset):
    def __len__(self) -> int:
        return 100

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.tensor([float(index)])


class _RecordingModel(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(1, 1)
        self.seen: list[tuple[int, int, int]] = []

    def train_dataloader(self):
        return StatefulDataLoader(_RangeDataset(), batch_size=1, num_workers=0)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)

    def training_step(self, batch, batch_idx):
        self.seen.append((int(self.global_step), int(self.current_epoch), int(batch.item())))
        return self.proj(batch).pow(2).mean()


def _fit(root: Path, max_steps: int, ckpt_path: Path | None = None) -> list[tuple[int, int, int]]:
    checkpoint_callback = NeMoModelCheckpoint(
        dirpath=str(root / "checkpoints"),
        filename="{step}",
        every_n_train_steps=SPLIT,
        save_top_k=-1,
        save_nemo_on_train_end=False,
    )
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        default_root_dir=str(root),
        callbacks=[checkpoint_callback],
        max_steps=max_steps,
        limit_train_batches=LIMIT,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        logger=False,
        enable_model_summary=False,
        enable_progress_bar=False,
    )
    model = _RecordingModel()
    trainer.fit(model, ckpt_path=str(ckpt_path) if ckpt_path is not None else None)
    return model.seen
