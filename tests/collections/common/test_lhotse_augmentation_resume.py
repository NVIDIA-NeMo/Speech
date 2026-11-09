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
"""Noise mixing and RIR reverb must give bit-identical audio after a mid-stream StatefulDataLoader resume."""

import copy
import hashlib

import numpy as np
import pytest
import soundfile as sf
from lhotse import CutSet, Recording, RecordingSet
from lhotse.dataset.iterable_dataset import IdentityDataset
from omegaconf import OmegaConf

from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config

pytestmark = pytest.mark.unit


def _wav_cuts(root, prefix, count, rng, duration=1.0):
    cuts = []
    for i in range(count):
        path = root / f"{prefix}_{i}.wav"
        sf.write(path, (0.1 * rng.standard_normal(int(16000 * duration))).astype(np.float32), 16000)
        cuts.append(Recording.from_file(path, recording_id=f"{prefix}_{i}").to_cut())
    return CutSet.from_cuts(cuts)


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    """Speech and noise as Shar (tarred audio; noise also in an indexed variant), plus a RecordingSet of RIRs."""
    root = tmp_path_factory.mktemp("aug_resume")
    rng = np.random.default_rng(0)
    for name, count, shard_size in [("speech", 24, 8), ("noise", 12, 4)]:
        (root / name).mkdir()
        _wav_cuts(root, name, count, rng).to_shar(root / name, fields={"recording": "wav"}, shard_size=shard_size)
    (root / "noise_indexed").mkdir()
    _wav_cuts(root, "noise_indexed", 12, rng).to_shar(
        root / "noise_indexed", fields={"recording": "wav"}, shard_size=4, compress_jsonl=False, create_index=True
    )
    rir_recordings = []
    for i in range(4):
        path = root / f"rir_{i}.wav"
        sf.write(path, (rng.standard_normal(1600) * np.exp(-np.arange(1600) / 200.0)).astype(np.float32), 16000)
        rir_recordings.append(Recording.from_file(path))
    RecordingSet.from_recordings(rir_recordings).to_file(root / "rirs.jsonl")
    return root


def _loader(data_dir, mode, num_workers, aug):
    """aug: "noise" / "noise_indexed" (noise mixing from Shar) or "rir" (reverb with RIR recordings)."""
    cfg = {
        "shar_path": str(data_dir / "speech"),
        "batch_size": 2,
        "shuffle": True,
        "num_workers": num_workers,
        "seed": 1,
        "shard_seed": 1,
        "use_stateful_dataloader": True,
        "force_map_dataset": mode == "map",
        "force_iterable_dataset": mode == "iterable",
        "sample_rate": 16000,
    }
    if aug.startswith("noise"):
        cfg.update(noise_path={"shar_path": str(data_dir / aug)}, noise_mix_prob=1.0)
    else:
        cfg.update(rir_enabled=True, rir_prob=1.0, rir_path=str(data_dir / "rirs.jsonl"))
    return get_lhotse_dataloader_from_config(
        OmegaConf.create(cfg), global_rank=0, world_size=1, dataset=IdentityDataset()
    )


def _take(data_dir, mode, num_workers, aug, num_batches, state=None):
    dloader = _loader(data_dir, mode, num_workers, aug)
    if state is not None:
        dloader.load_state_dict(state)
    it = iter(dloader)
    hashes = []
    for _ in range(num_batches):
        batch = next(it)
        hashes.append([hashlib.md5(np.ascontiguousarray(c.load_audio()).tobytes()).hexdigest() for c in batch])
    return hashes, copy.deepcopy(dloader.state_dict())


@pytest.mark.parametrize(
    "aug,num_workers",
    [("noise", 0), ("noise_indexed", 0), ("noise_indexed", 2), ("rir", 0)],
)
@pytest.mark.parametrize("mode", ["map", "iterable"])
def test_augmented_audio_is_bit_identical_after_resume(data_dir, mode, num_workers, aug):
    uninterrupted, _ = _take(data_dir, mode, num_workers, aug, num_batches=6)
    _, state = _take(data_dir, mode, num_workers, aug, num_batches=2)
    resumed, _ = _take(data_dir, mode, num_workers, aug, num_batches=4, state=state)

    assert resumed == uninterrupted[2:]
