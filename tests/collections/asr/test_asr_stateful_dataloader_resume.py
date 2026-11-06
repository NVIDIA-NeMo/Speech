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
"""Mid-epoch resume of a ModelPT model (EncDecCTCModel) trained with a stateful Lhotse dataloader."""

import inspect
from pathlib import Path

import pytest
from lhotse import CutSet
from lhotse.testing.dummies import dummy_cut, dummy_recording, dummy_supervision
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import OmegaConf

from nemo.collections.asr.models import EncDecCTCModel

LABELS = list(" abcdefghijklmnopqrstuvwxyz'")
NUM_CUTS = 12
BATCH_SIZE = 2
SPLIT = 3
TOTAL = 6  # == NUM_CUTS // BATCH_SIZE: both phases stay within the first epoch


class _RecordingCTCModel(EncDecCTCModel):
    """Records the transcripts of every training batch, in order."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seen_batches = []

    def training_step(self, batch, batch_idx):
        _, _, tokens, token_lens = batch
        self.seen_batches.append(
            ["".join(LABELS[t] for t in toks[:n].tolist()) for toks, n in zip(tokens, token_lens.tolist())]
        )
        return super().training_step(batch, batch_idx)


@pytest.fixture(scope="module")
def cuts_path(tmp_path_factory) -> str:
    root = tmp_path_factory.mktemp("stateful_resume")
    path = root / "cuts.jsonl.gz"
    CutSet(
        [
            dummy_cut(
                i,
                recording=dummy_recording(i, with_data=True),
                supervisions=[dummy_supervision(i, text=LABELS[i + 1] * 3)],
            )
            for i in range(NUM_CUTS)
        ]
    ).save_audios(root / "audio").to_file(path)
    return str(path)


@pytest.mark.unit
def test_ctc_stateful_dataloader_mid_epoch_resume_continues_at_next_batch(cuts_path, tmp_path):
    ckpt_dir = tmp_path / "ckpt"

    model_a = _fit(cuts_path, ckpt_dir, stateful=True)
    assert len(model_a.seen_batches) == TOTAL

    ckpt = next(ckpt_dir.glob(f"*step={SPLIT}.ckpt"))
    model_b = _fit(cuts_path, tmp_path / "ckpt_b", stateful=True, ckpt_path=str(ckpt))

    # The resumed run continues at exactly the next batch: no skipped or replayed batch.
    assert model_b.seen_batches == model_a.seen_batches[SPLIT:]
    # The model reads its own batches (no prefetch) when the train dataloader is stateful.
    assert list(inspect.signature(model_b.training_step).parameters) == ["dataloader_iter"]
    # self.log calls without batch_size (e.g. in training_step) get the current batch's size.
    assert model_b._current_train_batch_size() == BATCH_SIZE


@pytest.mark.unit
def test_ctc_non_stateful_dataloader_keeps_batch_training_step(cuts_path, tmp_path):
    model = _fit(cuts_path, tmp_path / "ckpt", stateful=False, max_steps=1)
    assert len(model.seen_batches) == 1
    assert list(inspect.signature(model.training_step).parameters) == ["batch", "batch_idx"]


def _fit(cuts_path: str, ckpt_dir: Path, stateful: bool, ckpt_path: str = None, max_steps: int = TOTAL):
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        max_steps=max_steps,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        use_distributed_sampler=False,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        callbacks=[ModelCheckpoint(dirpath=ckpt_dir, every_n_train_steps=SPLIT, save_top_k=-1)],
    )
    model = _RecordingCTCModel(cfg=_model_config(cuts_path, stateful), trainer=trainer)
    trainer.fit(model, ckpt_path=ckpt_path)
    return model


def _model_config(cuts_path: str, stateful: bool):
    return OmegaConf.create(
        {
            "sample_rate": 16000,
            "labels": LABELS,
            "train_ds": {
                "use_lhotse": True,
                "cuts_path": cuts_path,
                "sample_rate": 16000,
                "labels": LABELS,
                "batch_size": BATCH_SIZE,
                "shuffle": True,
                "seed": 1234,
                "shard_seed": 0,
                "num_workers": 0,
                "force_map_dataset": True,
                "use_stateful_dataloader": stateful,
            },
            "preprocessor": {"_target_": "nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor"},
            "encoder": {
                "_target_": "nemo.collections.asr.modules.ConvASREncoder",
                "feat_in": 64,
                "activation": "relu",
                "conv_mask": True,
                "jasper": [
                    {
                        "filters": 8,
                        "repeat": 1,
                        "kernel": [1],
                        "stride": [1],
                        "dilation": [1],
                        "dropout": 0.0,
                        "residual": False,
                        "separable": True,
                        "se": False,
                        "se_context_size": -1,
                    }
                ],
            },
            "decoder": {
                "_target_": "nemo.collections.asr.modules.ConvASRDecoder",
                "feat_in": 8,
                "num_classes": len(LABELS),
                "vocabulary": LABELS,
            },
            "optim": {"name": "adam", "lr": 1e-4},
        }
    )
