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
"""Lightning epoch boundaries must not drop batches from an iterable Lhotse stream.

Lightning calls ``iter()`` on the train loader at every epoch boundary, including the ones set by
``limit_train_batches``. Each batch carries the id of the worker that produced it and that worker's running
batch counter, so a batch the workers produced but nobody consumed shows up as a gap in the counter.
"""

from pathlib import Path

import lightning.pytorch as pl
import pytest
import torch
from lhotse import CutSet
from lhotse.testing.dummies import DummyManifest

from nemo.collections.common.data.lhotse.dataloader import get_lhotse_dataloader_from_config
from nemo.core.utils.lightning_utils import read_batch
from nemo.utils.callbacks.nemo_model_checkpoint import NeMoModelCheckpoint

pytest.importorskip("torchdata.stateful_dataloader")

NUM_WORKERS, LIMIT, SPLIT, TOTAL = 2, 4, 6, 12
NUM_CUTS, BATCH_SIZE = 60, 3


@pytest.fixture(scope="module")
def cuts_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("iterable_stream") / "cuts.jsonl"
    DummyManifest(CutSet, begin_id=0, end_id=NUM_CUTS).to_jsonl(path)
    return path


@pytest.mark.unit
@pytest.mark.parametrize("use_stateful_dataloader", [False, True])
def test_dataloader_does_not_drop_batches_at_epoch_boundaries(cuts_path, use_stateful_dataloader):
    dloader = _make_dataloader(cuts_path, use_stateful_dataloader=use_stateful_dataloader)
    consumed = []
    for _ in range(TOTAL // LIMIT):  # what Lightning does at each epoch boundary
        consumed.extend(batch for _, batch in zip(range(LIMIT), dloader))
    # The stream is finite: the next iter() continues it to the end ...
    consumed.extend(dloader)
    _assert_no_gaps(consumed)
    # ... every cut is seen once per worker (each worker reads its own shuffled copy here) ...
    num_batches_per_pass = len(consumed)
    assert num_batches_per_pass == NUM_WORKERS * NUM_CUTS // BATCH_SIZE
    # ... and once it has ended, iter() starts a new pass.
    assert len(list(dloader)) == num_batches_per_pass


@pytest.mark.unit
def test_lightning_epochs_consume_every_batch_and_resume_exactly(tmp_path, cuts_path):
    uninterrupted = _fit(tmp_path / "a", cuts_path)
    assert len(uninterrupted) == TOTAL
    assert len({epoch for epoch, _ in uninterrupted}) == TOTAL // LIMIT
    _assert_no_gaps([batch for _, batch in uninterrupted])

    resumed = _fit(tmp_path / "b", cuts_path, ckpt_path=tmp_path / "a" / "checkpoints" / f"step={SPLIT}.ckpt")

    def project(records):
        return [(epoch, batch["cut_ids"]) for epoch, batch in records]

    assert project(resumed) == project(uninterrupted[SPLIT:])
    _assert_no_gaps([batch for _, batch in resumed])


class _WorkerCountingDataset(torch.utils.data.Dataset):
    def __init__(self) -> None:
        self.num_batches = 0

    def __getitem__(self, cuts: CutSet) -> dict:
        info = torch.utils.data.get_worker_info()
        batch = {"worker": info.id if info is not None else -1, "seq": self.num_batches, "cut_ids": list(cuts.ids)}
        self.num_batches += 1
        return batch


class _Model(pl.LightningModule):
    def __init__(self, cuts_path: Path) -> None:
        super().__init__()
        self.cuts_path = cuts_path
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.seen: list[tuple[int, dict]] = []

    def train_dataloader(self):
        return _make_dataloader(self.cuts_path, use_stateful_dataloader=True)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)

    def training_step(self, dataloader_iter):
        batch, _ = read_batch(dataloader_iter, self)
        self.seen.append((int(self.current_epoch), batch))
        return self.weight.pow(2).sum()


def _fit(root: Path, cuts_path: Path, ckpt_path: Path | None = None) -> list[tuple[int, dict]]:
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        default_root_dir=str(root),
        callbacks=[
            NeMoModelCheckpoint(
                dirpath=str(root / "checkpoints"),
                filename="{step}",
                every_n_train_steps=SPLIT,
                save_top_k=-1,
                save_nemo_on_train_end=False,
            )
        ],
        max_steps=TOTAL,
        limit_train_batches=LIMIT,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        logger=False,
        enable_model_summary=False,
        enable_progress_bar=False,
    )
    model = _Model(cuts_path)
    trainer.fit(model, ckpt_path=str(ckpt_path) if ckpt_path is not None else None)
    return model.seen


def _make_dataloader(cuts_path: Path, use_stateful_dataloader: bool):
    config = {
        "cuts_path": str(cuts_path),
        "batch_size": BATCH_SIZE,
        "shuffle": True,
        "seed": 0,
        "shard_seed": "randomized",
        "num_workers": NUM_WORKERS,
        "force_iterable_dataset": True,
        "use_stateful_dataloader": use_stateful_dataloader,
    }
    return get_lhotse_dataloader_from_config(config, global_rank=0, world_size=1, dataset=_WorkerCountingDataset())


def _assert_no_gaps(batches: list[dict]) -> None:
    # Each worker must deliver its batches 0, 1, 2, ... in order: a gap means produced batches were dropped.
    assert {batch["worker"] for batch in batches} == set(range(NUM_WORKERS))
    for worker in range(NUM_WORKERS):
        seqs = [batch["seq"] for batch in batches if batch["worker"] == worker]
        assert seqs == list(range(seqs[0], seqs[0] + len(seqs))), f"worker {worker} skipped batches: {seqs}"
