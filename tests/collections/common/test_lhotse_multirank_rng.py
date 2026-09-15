# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from io import BytesIO
from pathlib import Path

import pytest
from lhotse import CutSet
from lhotse.serialization import load_jsonl, save_to_jsonl
from lhotse.shar.writers import JsonlShardWriter, TarWriter
from lhotse.testing.dummies import DummyManifest
from omegaconf import OmegaConf

from nemo.collections.common.data.lhotse.dataloader import get_lhotse_dataloader_from_config


class _Identity:
    def __getitem__(self, cuts):
        return cuts


@pytest.fixture(scope="session")
def cutset_path(tmp_path_factory) -> Path:
    """10 utterances of length 1s as a Lhotse CutSet."""
    cuts = DummyManifest(CutSet, begin_id=0, end_id=10, with_data=True)
    for c in cuts:
        c.features = None
        c.custom = None
        c.supervisions[0].custom = None

    tmp_path = tmp_path_factory.mktemp("data")
    p = tmp_path / "cuts.jsonl.gz"
    pa = tmp_path / "audio"
    cuts.save_audios(pa).to_file(p)
    return p


@pytest.fixture(scope="session")
def nemo_manifest_path(cutset_path: Path):
    """10 utterances of length 1s as a NeMo manifest."""
    nemo = []
    for idx, c in enumerate(CutSet.from_file(cutset_path)):
        nemo.append(
            {
                "audio_filepath": c.recording.sources[0].source,
                "text": f"irrelevant-{idx}",
                "duration": c.duration,
            }
        )
    p = cutset_path.parent / "nemo_manifest.json"
    save_to_jsonl(nemo, p)
    return p


@pytest.fixture(scope="session")
def nemo_tarred_manifest_path(nemo_manifest_path: Path) -> tuple[str, str]:
    """5 shards, each with 2 utterances."""
    root = nemo_manifest_path.parent / "nemo_tar"
    root.mkdir(exist_ok=True)
    with (
        TarWriter(f"{root}/audios_%01d.tar", shard_size=2) as tar_writer,
        JsonlShardWriter(f"{root}/manifest_%01d.jsonl", shard_size=2) as mft_writer,
    ):
        for idx, d in enumerate(load_jsonl(nemo_manifest_path)):
            p = d["audio_filepath"]
            name = Path(p).name
            with open(p, "rb") as f:
                tar_writer.write(name, BytesIO(f.read()))
            mft_writer.write({**d, "audio_filepath": name, "shard_id": idx // 2})
    return f"{root}/manifest__OP_0..4_CL_.jsonl", f"{root}/audios__OP_0..4_CL_.tar"


def test_dataloader_multiple_ranks_deterministic_rng(nemo_tarred_manifest_path: tuple[str, str]):
    json_mft, tar_mft = nemo_tarred_manifest_path
    config = OmegaConf.create(
        {
            "manifest_filepath": json_mft,
            "tarred_audio_filepaths": tar_mft,
            "sample_rate": 16000,
            "shuffle": True,
            "use_lhotse": True,
            "num_workers": 1,
            # lhotse specific
            "use_bucketing": True,
            "concurrent_bucketing": False,
            "num_buckets": 2,
            "drop_last": False,
            "batch_duration": 4.0,  # seconds
            "quadratic_duration": 15.0,  # seconds
            "shuffle_buffer_size": 10,
            "bucket_buffer_size": 100,
            "seed": 0,
            "shard_seed": "randomized",
        }
    )

    # Data parallel, rank 0
    dp0 = get_lhotse_dataloader_from_config(config=config, global_rank=0, world_size=2, dataset=_Identity())

    # Data parallel, rank 0 copy (is the iteration deterministic? -> yes)
    dp0_cpy = get_lhotse_dataloader_from_config(
        config=config,
        global_rank=0,
        world_size=2,
        dataset=_Identity(),
    )

    # Data parallel, rank 0, incremented seed (paranoia mode: does the iteration order change with the seed? -> yes)
    config2 = config.copy()
    config2["seed"] = config2["seed"] + 1
    dp0_incrseed = get_lhotse_dataloader_from_config(
        config=config2,
        global_rank=0,
        world_size=2,
        dataset=_Identity(),
    )

    # Data parallel, rank 1 (is data different on each DP rank? -> yes)
    dp1 = get_lhotse_dataloader_from_config(config=config, global_rank=1, world_size=2, dataset=_Identity())

    dloaders = zip(*[iter(dl) for dl in (dp0, dp0_cpy, dp0_incrseed, dp1)])

    for i in range(5):
        b0, b0_cpy, b0_incrseed, b1 = next(dloaders)
        assert b0 == b0_cpy
        assert b0 != b1
        assert b0_incrseed != b1
        assert b0 != b0_incrseed


def test_dataloader_multiple_ranks_trng(nemo_tarred_manifest_path: tuple[str, str]):
    """
    This test is the same as ``test_dataloader_multiple_ranks_deterministic_rng``,
    except that we set ``shard_seed="trng"`` which causes the seed to be lazily
    resolved in subprocesses (resolved => being drawn using OS's TRNG).
    Therefore, we don't expect any reproducibility.
    """
    json_mft, tar_mft = nemo_tarred_manifest_path
    config = OmegaConf.create(
        {
            "manifest_filepath": json_mft,
            "tarred_audio_filepaths": tar_mft,
            "sample_rate": 16000,
            "shuffle": True,
            "use_lhotse": True,
            "num_workers": 1,
            # lhotse specific
            "use_bucketing": True,
            "concurrent_bucketing": False,
            "num_buckets": 2,
            "drop_last": False,
            "batch_duration": 4.0,  # seconds
            "quadratic_duration": 15.0,  # seconds
            "shuffle_buffer_size": 10,
            "bucket_buffer_size": 100,
            "seed": 0,
            "shard_seed": "trng",
        }
    )

    # Data parallel, rank 0
    dp0 = get_lhotse_dataloader_from_config(config=config, global_rank=0, world_size=2, dataset=_Identity())

    # Data parallel, rank 0 copy (is the iteration deterministic? -> no, trng)
    dp0_cpy = get_lhotse_dataloader_from_config(
        config=config,
        global_rank=0,
        world_size=2,
        dataset=_Identity(),
    )

    # Data parallel, rank 0, incremented seed (paranoia mode: does the iteration order change with the seed? -> yes)
    config2 = config.copy()
    config2["seed"] = config2["seed"] + 1
    dp0_incrseed = get_lhotse_dataloader_from_config(
        config=config2,
        global_rank=0,
        world_size=2,
        dataset=_Identity(),
    )

    # Data parallel, rank 1 (is data different on each DP rank? -> yes)
    dp1 = get_lhotse_dataloader_from_config(config=config, global_rank=1, world_size=2, dataset=_Identity())

    dloaders = zip(*[iter(dl) for dl in (dp0, dp0_cpy, dp0_incrseed, dp1)])
    batches = [next(dloaders) for _ in range(5)]
    b0_batches, b0_cpy_batches, b0_incrseed_batches, b1_batches = map(list, zip(*batches))

    assert b0_batches != b0_cpy_batches
    assert b0_batches != b1_batches
    assert b0_incrseed_batches != b1_batches
    assert b0_batches != b0_incrseed_batches


def _find_transform(sampler, cls):
    """Return the first sampler-level transform of type ``cls`` (they are attached via ``sampler.map``)."""
    matches = [t for t in sampler._transforms if isinstance(t, cls)]
    assert matches, f"{cls.__name__} was not attached to the sampler; transforms={sampler._transforms}"
    return matches[0]


def _rir_dataloader(cutset_path: Path, rank: int, shard_seed):
    config = OmegaConf.create(
        {
            "cuts_path": str(cutset_path),
            "sample_rate": 16000,
            "use_lhotse": True,
            "num_workers": 0,
            "batch_size": 2,
            "seed": 0,
            "shard_seed": shard_seed,
            # RIR is the transform under test; lowpass is a sibling used as a control.
            "rir_enabled": True,
            "rir_prob": 0.5,
            "lowpass_enabled": True,
            "lowpass_prob": 0.5,
        }
    )
    return get_lhotse_dataloader_from_config(config=config, global_rank=rank, world_size=2, dataset=_Identity())


def test_rir_augmentation_rng_differs_across_ranks(cutset_path: Path):
    """RIR augmentation must draw its randomness from ``shard_seed``, like every other online
    augmentation attached to the sampler.

    ``seed`` is resolved to a concrete integer before the sampler is built, so it is bit-identical
    on every data-parallel rank; ``shard_seed`` defaults to ``"trng"`` and is resolved per process.
    Seeding the RIR transform from ``seed`` therefore made every rank apply the same reverberation
    coin flips and pick the same impulse responses, silently collapsing RIR augmentation diversity.
    """
    from lhotse.dataset import LowpassUsingResampling, ReverbWithImpulseResponse

    dl0 = _rir_dataloader(cutset_path, rank=0, shard_seed="trng")
    dl1 = _rir_dataloader(cutset_path, rank=1, shard_seed="trng")

    rir0 = _find_transform(dl0.sampler, ReverbWithImpulseResponse)
    rir1 = _find_transform(dl1.sampler, ReverbWithImpulseResponse)
    assert [rir0.random.random() for _ in range(32)] != [rir1.random.random() for _ in range(32)]

    # Control: a sibling augmentation that already used ``shard_seed`` differentiates the same way,
    # so the assertion above is testing the seed source and not the harness.
    lowpass0 = _find_transform(dl0.sampler, LowpassUsingResampling)
    lowpass1 = _find_transform(dl1.sampler, LowpassUsingResampling)
    assert [lowpass0.rng.random() for _ in range(32)] != [lowpass1.rng.random() for _ in range(32)]


def test_rir_augmentation_rng_reproducible_with_fixed_shard_seed(cutset_path: Path):
    """Counterpart to the test above: an explicit integer ``shard_seed`` must still give every rank
    the same RIR randomness, so that fully reproducible runs remain possible."""
    from lhotse.dataset import ReverbWithImpulseResponse

    dl0 = _rir_dataloader(cutset_path, rank=0, shard_seed=1234)
    dl1 = _rir_dataloader(cutset_path, rank=1, shard_seed=1234)

    rir0 = _find_transform(dl0.sampler, ReverbWithImpulseResponse)
    rir1 = _find_transform(dl1.sampler, ReverbWithImpulseResponse)
    assert [rir0.random.random() for _ in range(32)] == [rir1.random.random() for _ in range(32)]
