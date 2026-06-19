# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
