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
from types import SimpleNamespace

import lightning.pytorch as pl
import pytest
import torch
from torchdata.stateful_dataloader import StatefulDataLoader

from nemo.utils.callbacks.nemo_model_checkpoint import NeMoModelCheckpoint, _complete_batch_saved_on_train_batch_end

LIMIT, SPLIT, TOTAL = 5, 3, 12


@pytest.mark.unit
def test_resume_from_mid_epoch_checkpoint_keeps_epoch_boundaries(tmp_path):
    uninterrupted = _fit(tmp_path / "a", max_steps=TOTAL)
    ckpt_path = tmp_path / "a" / "checkpoints" / f"step={SPLIT}.ckpt"
    assert ckpt_path.exists()

    resumed = _fit(tmp_path / "b", max_steps=TOTAL, ckpt_path=ckpt_path)

    # Before the fix, Lightning<2.5 ran one extra batch in the resumed epoch, shifting every later epoch boundary.
    assert resumed == uninterrupted[SPLIT:]


@pytest.mark.unit
@pytest.mark.parametrize(
    "current, is_last_batch, expected_completed",
    [
        # Saved from on_train_batch_end mid-epoch: the in-flight batch is counted as completed.
        (dict(ready=3, started=3, processed=3, completed=2), False, 3),
        # Saved after the batch finished (e.g. on validation end): nothing to fix.
        (dict(ready=3, started=3, processed=3, completed=3), False, 3),
        # Saved on the last batch of the epoch: left to Lightning (see the helper's docstring).
        (dict(ready=5, started=5, processed=5, completed=4), True, 4),
    ],
)
def test_complete_batch_saved_on_train_batch_end(current, is_last_batch, expected_completed):
    checkpoint = {
        "loops": {
            "fit_loop": {
                "epoch_loop.batch_progress": {
                    "total": {k: v + 10 for k, v in current.items()},
                    "current": dict(current),
                    "is_last_batch": is_last_batch,
                },
                "epoch_loop.state_dict": {"_batches_that_stepped": 12},
            }
        }
    }
    trainer = SimpleNamespace(fit_loop=SimpleNamespace(epoch_loop=object()), accumulate_grad_batches=1)

    _complete_batch_saved_on_train_batch_end(trainer, checkpoint)

    fit_loop_state = checkpoint["loops"]["fit_loop"]
    progress = fit_loop_state["epoch_loop.batch_progress"]
    assert progress["current"]["completed"] == expected_completed
    assert progress["total"]["completed"] == expected_completed + 10
    changed = expected_completed != current["completed"]
    assert fit_loop_state["epoch_loop.state_dict"]["_batches_that_stepped"] == 12 + int(changed)


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
