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
from collections import Counter
from io import BytesIO
from itertools import islice
from pathlib import Path
from typing import Dict, List, Tuple

import lhotse
import numpy as np
import pytest
import torch
from lhotse import CutSet, MonoCut, NumpyFilesWriter, Recording
from lhotse.audio import AudioLoadingError
from lhotse.cut import Cut, MixedCut
from lhotse.cut.text import TextPairExample
from omegaconf import OmegaConf

from nemo.collections.asr.data.audio_to_text_lhotse import TokenizerWrapper
from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config
from nemo.collections.common.data.lhotse.text_adapters import TextExample
from nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer, create_spt_model


@pytest.fixture(scope="session")
def cutset_path(tmp_path_factory) -> Path:
    """10 utterances of length 1s as a Lhotse CutSet."""
    from lhotse import CutSet
    from lhotse.testing.dummies import DummyManifest

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
def cutset_shar_path(cutset_path: Path) -> Path:
    """10 utterances of length 1s as a Lhotse Shar (tarred) CutSet."""
    from lhotse import CutSet

    cuts = CutSet.from_file(cutset_path)
    p = cutset_path.parent / "shar"
    p.mkdir(exist_ok=True)
    cuts.to_shar(p, fields={"recording": "wav"}, shard_size=5)
    return p


@pytest.fixture(scope="session")
def cutset_shar_path_other(cutset_path: Path) -> Path:
    """10 utterances of length 1s as a Lhotse Shar (tarred) CutSet, but with different IDs."""
    from lhotse import CutSet

    cuts = CutSet.from_file(cutset_path).modify_ids(lambda id: f"other-{id}")
    p = cutset_path.parent / "shar-other"
    p.mkdir(exist_ok=True)
    cuts.to_shar(p, fields={"recording": "wav"}, shard_size=5)
    return p


@pytest.fixture(scope="session")
def nemo_manifest_path(cutset_path: Path):
    """10 utterances of length 1s as a NeMo manifest."""
    from lhotse import CutSet
    from lhotse.serialization import save_to_jsonl

    nemo = []
    for c in CutSet.from_file(cutset_path):
        nemo.append(
            {
                "audio_filepath": c.recording.sources[0].source,
                "text": "irrelevant",
                "text-other": "not relevant",
                "duration": c.duration,
                "my-custom-field": "irrelevant",
                "lang": "en",
                "custom-lang": "pl",
            }
        )
    p = cutset_path.parent / "nemo_manifest.json"
    save_to_jsonl(nemo, p)
    return p


@pytest.fixture(scope="session")
def mc_cutset_path(tmp_path_factory) -> Path:
    """10 two-channel utterances of length 1s as a Lhotse CutSet."""
    from lhotse import CutSet, MultiCut
    from lhotse.testing.dummies import DummyManifest

    num_examples = 10  # number of examples
    num_channels = 2  # number of channels per example

    # create a dummy manifest with single-channel examples
    sc_cuts = DummyManifest(CutSet, begin_id=0, end_id=num_examples * num_channels, with_data=True)
    mc_cuts = []

    for n in range(num_examples):
        # sources for individual channels
        mc_sources = []
        for channel in range(num_channels):
            source = sc_cuts[n * num_channels + channel].recording.sources[0]
            source.channels = [channel]
            mc_sources.append(source)

        # merge recordings
        rec = Recording(
            sources=mc_sources,
            id=f'mc-dummy-recording-{n:02d}',
            num_samples=sc_cuts[0].num_samples,
            duration=sc_cuts[0].duration,
            sampling_rate=sc_cuts[0].sampling_rate,
        )

        # multi-channel cut
        cut = MultiCut(
            recording=rec, id=f'mc-dummy-cut-{n:02d}', start=0, duration=1.0, channel=list(range(num_channels))
        )
        mc_cuts.append(cut)

    mc_cuts = CutSet.from_cuts(mc_cuts)

    tmp_path = tmp_path_factory.mktemp("data")
    p = tmp_path / "mc_cuts.jsonl.gz"
    pa = tmp_path / "mc_audio"
    mc_cuts.save_audios(pa).to_file(p)
    return p


@pytest.fixture(scope="session")
def nemo_tarred_manifest_path(nemo_manifest_path: Path) -> Tuple[str, str]:
    """10 utterances of length 1s as a NeMo tarred manifest."""
    from lhotse.serialization import SequentialJsonlWriter, load_jsonl
    from lhotse.shar.writers import TarWriter

    root = nemo_manifest_path.parent / "nemo_tar"
    root.mkdir(exist_ok=True)

    with (
        TarWriter(f"{root}/audios_%01d.tar", shard_size=5) as tar_writer,
        SequentialJsonlWriter(root / "tarred_audio_filepaths.jsonl") as mft_writer,
    ):
        for idx, d in enumerate(load_jsonl(nemo_manifest_path)):
            p = d["audio_filepath"]
            name = Path(p).name
            with open(p, "rb") as f:
                tar_writer.write(name, BytesIO(f.read()))
            mft_writer.write({**d, "audio_filepath": name, "shard_id": int(idx > 4)})
    return mft_writer.path, f"{root}/audios__OP_0..1_CL_.tar"


@pytest.fixture(scope="session")
def nemo_tarred_manifest_path_multi(nemo_tarred_manifest_path: tuple[str, str]) -> Tuple[str, str]:
    """10 utterances of length 1s as a NeMo tarred manifest. Stored in one manifest per shard."""
    from lhotse.serialization import load_jsonl
    from lhotse.shar.writers import JsonlShardWriter

    json_p, tar_p = nemo_tarred_manifest_path

    json_dir = json_p.parent / "shard_manifests"
    json_dir.mkdir(exist_ok=True)
    with JsonlShardWriter(f"{json_dir}/manifest_%d.jsonl", shard_size=5) as mft_writer:
        for item in load_jsonl(json_p):
            mft_writer.write(item)
    return f"{json_dir}/manifest__OP_0..1_CL_.jsonl", tar_p


class UnsupervisedAudioDataset(torch.utils.data.Dataset):
    def __getitem__(self, cuts: lhotse.CutSet) -> Dict[str, torch.Tensor]:
        audio, audio_lens = lhotse.dataset.collation.collate_audio(cuts)
        return {"audio": audio, "audio_lens": audio_lens, "ids": [c.id for c in cuts]}


def test_dataloader_from_lhotse_cuts(cutset_path: Path):
    config = OmegaConf.create(
        {
            "cuts_path": cutset_path,
            "sample_rate": 16000,
            "shuffle": True,
            "use_lhotse": True,
            "num_workers": 0,
            # lhotse specific
            "use_bucketing": True,
            "num_buckets": 2,
            "drop_last": False,
            "batch_duration": 4.0,  # seconds
            "quadratic_duration": 15.0,  # seconds
            "shuffle_buffer_size": 10,
            "bucket_buffer_size": 100,
            "seed": 0,
        }
    )

    dl = get_lhotse_dataloader_from_config(
        config=config, global_rank=0, world_size=1, dataset=UnsupervisedAudioDataset()
    )

    batches = [batch for batch in dl]
    assert len(batches) == 4

    b = batches[0]
    assert set(b.keys()) == {"audio", "audio_lens", "ids"}
    assert b["audio"].shape[0] == b["audio_lens"].shape[0] == 3

    b = batches[1]
    assert set(b.keys()) == {"audio", "audio_lens", "ids"}
    assert b["audio"].shape[0] == b["audio_lens"].shape[0] == 3

    b = batches[2]
    assert set(b.keys()) == {"audio", "audio_lens", "ids"}
    assert b["audio"].shape[0] == b["audio_lens"].shape[0] == 3

    b = batches[3]
    assert set(b.keys()) == {"audio", "audio_lens", "ids"}
    assert b["audio"].shape[0] == b["audio_lens"].shape[0] == 1


def test_dataloader_from_lhotse_cuts_truncate(cutset_path: Path):
    config = OmegaConf.create(
        {
            "cuts_path": cutset_path,
            "truncate_duration": 0.5,
            "sample_rate": 16000,
            "shuffle": True,
            "use_lhotse": True,
            "num_workers": 0,
            "batch_size": 4,
            "seed": 0,
        }
    )

    dl = get_lhotse_dataloader_from_config(
        config=config, global_rank=0, world_size=1, dataset=UnsupervisedAudioDataset()
    )

    batches = [b for b in dl]
    assert len(batches) == 3
    # 0.5s = 8000 samples, note the constant duration and batch size except for last batch
    assert batches[0]["audio"].shape == (4, 8000)
    assert batches[1]["audio"].shape == (4, 8000)
    assert batches[2]["audio"].shape == (2, 8000)
    # exactly 10 cuts were used


def test_dataloader_from_lhotse_cuts_cut_into_windows(cutset_path: Path):
    config = OmegaConf.create(
        {
            "cuts_path": cutset_path,
            "cut_into_windows_duration": 0.5,
            "sample_rate": 16000,
            "shuffle": True,
            "use_lhotse": True,
            "num_workers": 0,
            "batch_size": 4,
            "seed": 0,
        }
    )

    dl = get_lhotse_dataloader_from_config(
        config=config, global_rank=0, world_size=1, dataset=UnsupervisedAudioDataset()
    )

    batches = [b for b in dl]
    assert len(batches) == 5
    # 0.5s = 8000 samples, note the constant duration and batch size
    assert batches[0]["audio"].shape == (4, 8000)
    assert batches[1]["audio"].shape == (4, 8000)
    assert batches[2]["audio"].shape == (4, 8000)
    assert batches[3]["audio"].shape == (4, 8000)
    assert batches[4]["audio"].shape == (4, 8000)
    # exactly 20 cuts were used because we cut 10x 1s cuts into 20x 0.5s cuts


def test_dataloader_from_lhotse_cuts_channel_selector(mc_cutset_path: Path):
    # Dataloader without channel selector
    config = OmegaConf.create(
        {
            "cuts_path": mc_cutset_path,
            "sample_rate": 16000,
            "shuffle": True,
            "use_lhotse": True,
            "num_workers": 0,
            "batch_size": 4,
            "seed": 0,
        }
    )

    dl = get_lhotse_dataloader_from_config(
        config=config, global_rank=0, world_size=1, dataset=UnsupervisedAudioDataset()
    )
    batches = [b for b in dl]
    assert len(batches) == 3

    # 1.0s = 16000 samples, two channels, note the constant duration and batch size
    assert batches[0]["audio"].shape == (4, 2, 16000)
    assert batches[1]["audio"].shape == (4, 2, 16000)
    assert batches[2]["audio"].shape == (2, 2, 16000)
    # exactly 10 cuts were used

    # Apply channel selector
    for channel_selector in [None, 0, 1]:

        config_cs = OmegaConf.create(
            {
                "cuts_path": mc_cutset_path,
                "channel_selector": channel_selector,
                "sample_rate": 16000,
                "shuffle": True,
                "use_lhotse": True,
                "num_workers": 0,
                "batch_size": 4,
                "seed": 0,
            }
        )

        dl_cs = get_lhotse_dataloader_from_config(
            config=config_cs, global_rank=0, world_size=1, dataset=UnsupervisedAudioDataset()
        )

        for n, b_cs in enumerate(dl_cs):
            if channel_selector is None:
                # no channel selector, needs to match the original dataset
                assert torch.equal(b_cs["audio"], batches[n]["audio"])
            else:
                # channel selector, needs to match the selected channel
                assert torch.equal(b_cs["audio"], batches[n]["audio"][:, channel_selector, :])


def test_dataloader_from_lhotse_shar_cuts(cutset_shar_path: Path):
    config = OmegaConf.create(
        {
            "shar_path": cutset_shar_path,
            "sample_rate": 16000,
            "shuffle": True,
            "use_lhotse": True,
            "num_workers": 0,
            # lhotse specific
            "use_bucketing": True,
            "num_buckets": 2,
            "drop_last": False,
            "batch_duration": 4.0,  # seconds
            "quadratic_duration": 15.0,  # seconds
            "shuffle_buffer_size": 10,
            "bucket_buffer_size": 100,
            "seed": 0,
            "shard_seed": 0,
        }
    )

    dl = get_lhotse_dataloader_from_config(
        config=config, global_rank=0, world_size=1, dataset=UnsupervisedAudioDataset()
    )

    # Note: we use islice here because with Lhotse Shar the dataloader will always be infinite.
    batches = [batch for batch in islice(dl, 4)]
    assert len(batches) == 4

    b = batches[0]
    assert set(b.keys()) == {"audio", "audio_lens", "ids"}
    assert b["audio"].shape[0] == b["audio_lens"].shape[0] == 3

    b = batches[1]
    assert set(b.keys()) == {"audio", "audio_lens", "ids"}
    assert b["audio"].shape[0] == b["audio_lens"].shape[0] == 3

    b = batches[2]
    assert set(b.keys()) == {"audio", "audio_lens", "ids"}
    assert b["audio"].shape[0] == b["audio_lens"].shape[0] == 3

    b = batches[3]
    assert set(b.keys()) == {"audio", "audio_lens", "ids"}
    assert b["audio"].shape[0] == b["audio_lens"].shape[0] == 3


def test_dataloader_from_nemo_manifest(nemo_manifest_path: Path):
    config = OmegaConf.create(
        {
            "manifest_filepath": nemo_manifest_path,
            "sample_rate": 16000,
            "shuffle": True,
            "use_lhotse": True,
            "num_workers": 0,
            # lhotse specific
            "use_bucketing": True,
            "num_buckets": 2,
            "drop_last": False,
            "batch_duration": 4.0,  # seconds
            "quadratic_duration": 15.0,  # seconds
            "shuffle_buffer_size": 10,
            "bucket_buffer_size": 100,
            "seed": 0,
            "shard_seed": 0,
        }
    )

    dl = get_lhotse_dataloader_from_config(
        config=config, global_rank=0, world_size=1, dataset=UnsupervisedAudioDataset()
    )

    batches = [batch for batch in dl]
    assert len(batches) == 4

    b = batches[0]
    assert set(b.keys()) == {"audio", "audio_lens", "ids"}
    assert b["audio"].shape[0] == b["audio_lens"].shape[0] == 3

    b = batches[1]
    assert set(b.keys()) == {"audio", "audio_lens", "ids"}
    assert b["audio"].shape[0] == b["audio_lens"].shape[0] == 3

    b = batches[2]
    assert set(b.keys()) == {"audio", "audio_lens", "ids"}
    assert b["audio"].shape[0] == b["audio_lens"].shape[0] == 3

    b = batches[3]
    assert set(b.keys()) == {"audio", "audio_lens", "ids"}
    assert b["audio"].shape[0] == b["audio_lens"].shape[0] == 1


class _Identity:
    def __getitem__(self, cuts):
        return cuts


def test_dataloader_from_nemo_manifest_has_custom_fields(nemo_manifest_path: Path):
    config = OmegaConf.create(
        {
            "manifest_filepath": nemo_manifest_path,
            "sample_rate": 16000,
            "shuffle": True,
            "use_lhotse": True,
            "num_workers": 0,
            # lhotse specific
            "use_bucketing": False,
            "batch_duration": 4.0,  # seconds
            "shuffle_buffer_size": 10,
            "seed": 0,
            "shard_seed": 0,
        }
    )

    dl = get_lhotse_dataloader_from_config(config=config, global_rank=0, world_size=1, dataset=_Identity())

    batch = next(iter(dl))
    for cut in batch:
        assert isinstance(cut.custom, dict)
        assert "my-custom-field" in cut.custom


def test_dataloader_from_tarred_nemo_manifest(nemo_tarred_manifest_path: tuple[str, str]):
    json_mft, tar_mft = nemo_tarred_manifest_path
    config = OmegaConf.create(
        {
            "manifest_filepath": json_mft,
            "tarred_audio_filepaths": tar_mft,
            "sample_rate": 16000,
            "shuffle": True,
            "use_lhotse": True,
            "num_workers": 0,
            # lhotse specific
            "use_bucketing": True,
            "num_buckets": 2,
            "drop_last": False,
            "batch_duration": 4.0,  # seconds
            "quadratic_duration": 15.0,  # seconds
            "shuffle_buffer_size": 10,
            "bucket_buffer_size": 100,
            "seed": 0,
            "shard_seed": 0,
        }
    )

    dl = get_lhotse_dataloader_from_config(
        config=config, global_rank=0, world_size=1, dataset=UnsupervisedAudioDataset()
    )

    batches = [batch for batch in islice(dl, 4)]
    assert len(batches) == 4

    b = batches[0]
    assert set(b.keys()) == {"audio", "audio_lens", "ids"}
    assert b["audio"].shape[0] == b["audio_lens"].shape[0] == 3

    b = batches[1]
    assert set(b.keys()) == {"audio", "audio_lens", "ids"}
    assert b["audio"].shape[0] == b["audio_lens"].shape[0] == 3

    b = batches[2]
    assert set(b.keys()) == {"audio", "audio_lens", "ids"}
    assert b["audio"].shape[0] == b["audio_lens"].shape[0] == 3

    b = batches[3]
    assert set(b.keys()) == {"audio", "audio_lens", "ids"}
    assert b["audio"].shape[0] == b["audio_lens"].shape[0] == 3


def test_dataloader_from_tarred_nemo_manifest_weighted_combination(nemo_tarred_manifest_path: tuple[str, str]):
    json_mft, tar_mft = nemo_tarred_manifest_path
    config = OmegaConf.create(
        {
            "manifest_filepath": [[json_mft, 0.8], [json_mft, 0.2]],
            "tarred_audio_filepaths": [[tar_mft], [tar_mft]],
            "sample_rate": 16000,
            "shuffle": True,
            "use_lhotse": True,
            "num_workers": 0,
            # lhotse specific
            "use_bucketing": True,
            "num_buckets": 2,
            "drop_last": False,
            "batch_duration": 4.0,  # seconds
            "quadratic_duration": 15.0,  # seconds
            "shuffle_buffer_size": 10,
            "bucket_buffer_size": 100,
            "seed": 0,
            "shard_seed": 0,
        }
    )

    dl = get_lhotse_dataloader_from_config(
        config=config, global_rank=0, world_size=1, dataset=UnsupervisedAudioDataset()
    )

    b = next(iter(dl))
    assert set(b.keys()) == {"audio", "audio_lens", "ids"}
    assert b["audio"].shape[0] == b["audio_lens"].shape[0] == 3


def test_dataloader_from_tarred_nemo_manifest_multi(nemo_tarred_manifest_path_multi: tuple[str, str]):
    json_mft, tar_mft = nemo_tarred_manifest_path_multi
    config = OmegaConf.create(
        {
            "manifest_filepath": json_mft,
            "tarred_audio_filepaths": tar_mft,
            "sample_rate": 16000,
            "shuffle": True,
            "use_lhotse": True,
            "num_workers": 0,
            # lhotse specific
            "use_bucketing": True,
            "num_buckets": 2,
            "drop_last": False,
            "batch_duration": 4.0,  # seconds
            "quadratic_duration": 15.0,  # seconds
            "shuffle_buffer_size": 10,
            "bucket_buffer_size": 100,
            "seed": 0,
            "shard_seed": 0,
        }
    )

    dl = get_lhotse_dataloader_from_config(
        config=config, global_rank=0, world_size=1, dataset=UnsupervisedAudioDataset()
    )

    batches = [batch for batch in islice(dl, 4)]
    assert len(batches) == 4

    b = batches[0]
    assert set(b.keys()) == {"audio", "audio_lens", "ids"}
    assert b["audio"].shape[0] == b["audio_lens"].shape[0] == 3

    b = batches[1]
    assert set(b.keys()) == {"audio", "audio_lens", "ids"}
    assert b["audio"].shape[0] == b["audio_lens"].shape[0] == 3

    b = batches[2]
    assert set(b.keys()) == {"audio", "audio_lens", "ids"}
    assert b["audio"].shape[0] == b["audio_lens"].shape[0] == 3

    b = batches[3]
    assert set(b.keys()) == {"audio", "audio_lens", "ids"}
    assert b["audio"].shape[0] == b["audio_lens"].shape[0] == 3


def test_dataloader_from_tarred_nemo_manifest_multi_max_open_streams(nemo_tarred_manifest_path_multi: tuple[str, str]):
    json_mft, tar_mft = nemo_tarred_manifest_path_multi
    config = OmegaConf.create(
        {
            "manifest_filepath": [[json_mft], [json_mft]],
            "tarred_audio_filepaths": [[tar_mft], [tar_mft]],
            "sample_rate": 16000,
            "shuffle": True,
            "use_lhotse": True,
            "num_workers": 0,
            # lhotse specific
            "use_bucketing": True,
            "num_buckets": 2,
            "max_open_streams": 1,
            "drop_last": False,
            "batch_duration": 4.0,  # seconds
            "quadratic_duration": 15.0,  # seconds
            "shuffle_buffer_size": 10,
            "bucket_buffer_size": 100,
            "seed": 0,
            "shard_seed": 0,
        }
    )

    dl = get_lhotse_dataloader_from_config(
        config=config, global_rank=0, world_size=1, dataset=UnsupervisedAudioDataset()
    )

    _ = next(iter(dl))


def test_dataloader_from_tarred_nemo_manifest_concat(nemo_tarred_manifest_path: tuple[str, str]):
    json_mft, tar_mft = nemo_tarred_manifest_path
    config = OmegaConf.create(
        {
            "manifest_filepath": json_mft,
            "tarred_audio_filepaths": tar_mft,
            "sample_rate": 16000,
            "shuffle": True,
            "use_lhotse": True,
            "num_workers": 0,
            # lhotse specific
            "concatenate_samples": True,
            "concatenate_duration_factor": 3.0,
            "batch_duration": 4.0,
            "quadratic_duration": 15.0,  # seconds
            "use_bucketing": False,
            "drop_last": False,
            "shuffle_buffer_size": 10,
            "seed": 0,
            "shard_seed": 0,
        }
    )

    dl = get_lhotse_dataloader_from_config(
        config=config, global_rank=0, world_size=1, dataset=UnsupervisedAudioDataset()
    )

    batches = [batch for batch in islice(dl, 4)]

    assert len(batches) == 4

    # the first element has been concatenated: 2x16000 speech (2x1s) + 1600 gap (0.1s)
    expected_audio_lens = torch.tensor([33600, 16000], dtype=torch.int32)

    b = batches[0]
    assert set(b.keys()) == {"audio", "audio_lens", "ids"}
    assert b["audio"].shape[0] == b["audio_lens"].shape[0] == 2
    torch.testing.assert_close(b["audio_lens"], expected_audio_lens)

    b = batches[1]
    assert set(b.keys()) == {"audio", "audio_lens", "ids"}
    assert b["audio"].shape[0] == b["audio_lens"].shape[0] == 2
    torch.testing.assert_close(b["audio_lens"], expected_audio_lens)

    b = batches[2]
    assert set(b.keys()) == {"audio", "audio_lens", "ids"}
    assert b["audio"].shape[0] == b["audio_lens"].shape[0] == 2
    torch.testing.assert_close(b["audio_lens"], expected_audio_lens)

    b = batches[3]
    assert set(b.keys()) == {"audio", "audio_lens", "ids"}
    assert b["audio"].shape[0] == b["audio_lens"].shape[0] == 2
    torch.testing.assert_close(b["audio_lens"], expected_audio_lens)


def test_dataloader_from_lhotse_shar_cuts_combine_datasets_unweighted(
    cutset_shar_path: Path, cutset_shar_path_other: Path
):
    """
    Note: if we iterated more mini-batches in this test, in the expectation there
    will be 50-50 % mini-batch occupancy of examples from both datasets.
    """
    config = OmegaConf.create(
        {
            "shar_path": [cutset_shar_path, cutset_shar_path_other],
            "sample_rate": 16000,
            "shuffle": True,
            "use_lhotse": True,
            "num_workers": 0,
            # lhotse specific
            "use_bucketing": True,
            "num_buckets": 2,
            "drop_last": False,
            "batch_duration": 4.0,  # seconds
            "quadratic_duration": 15.0,  # seconds
            "shuffle_buffer_size": 10,
            "bucket_buffer_size": 100,
            "seed": 0,
            "shard_seed": 0,
        }
    )

    dl = get_lhotse_dataloader_from_config(
        config=config, global_rank=0, world_size=1, dataset=UnsupervisedAudioDataset()
    )

    # Note: we use islice here because with Lhotse Shar the dataloader will always be infinite.
    batches = [batch for batch in islice(dl, 4)]
    assert len(batches) == 4

    b = batches[0]
    assert len([cid for cid in b["ids"] if cid.startswith("dummy")]) == 1  # dataset 1
    assert len([cid for cid in b["ids"] if cid.startswith("other")]) == 2  # dataset 2

    b = batches[1]
    assert len([cid for cid in b["ids"] if cid.startswith("dummy")]) == 0  # dataset 1
    assert len([cid for cid in b["ids"] if cid.startswith("other")]) == 3  # dataset 2

    b = batches[2]
    assert len([cid for cid in b["ids"] if cid.startswith("dummy")]) == 2  # dataset 1
    assert len([cid for cid in b["ids"] if cid.startswith("other")]) == 1  # dataset 2

    b = batches[3]
    assert len([cid for cid in b["ids"] if cid.startswith("dummy")]) == 1  # dataset 1
    assert len([cid for cid in b["ids"] if cid.startswith("other")]) == 2  # dataset 2


def test_dataloader_from_lhotse_shar_cuts_combine_datasets_weighted(
    cutset_shar_path: Path, cutset_shar_path_other: Path
):
    """
    Note: if we iterated more mini-batches in this test, in the expectation there
    will be 90-10 % mini-batch occupancy of examples from both datasets.
    """
    config = OmegaConf.create(
        {
            "shar_path": [[cutset_shar_path, 90], [cutset_shar_path_other, 10]],
            "sample_rate": 16000,
            "shuffle": True,
            "use_lhotse": True,
            "num_workers": 0,
            # lhotse specific
            "use_bucketing": True,
            "num_buckets": 2,
            "drop_last": False,
            "batch_duration": 4.0,  # seconds
            "quadratic_duration": 15.0,  # seconds
            "shuffle_buffer_size": 10,
            "bucket_buffer_size": 100,
            "seed": 0,
            "shard_seed": 0,
        }
    )

    dl = get_lhotse_dataloader_from_config(
        config=config, global_rank=0, world_size=1, dataset=UnsupervisedAudioDataset()
    )

    # Note: we use islice here because with Lhotse Shar the dataloader will always be infinite.
    batches = [batch for batch in islice(dl, 6)]
    assert len(batches) == 6

    b = batches[0]
    assert len([cid for cid in b["ids"] if cid.startswith("dummy")]) == 3  # dataset 1
    assert len([cid for cid in b["ids"] if cid.startswith("other")]) == 0  # dataset 2

    b = batches[1]
    assert len([cid for cid in b["ids"] if cid.startswith("dummy")]) == 1  # dataset 1
    assert len([cid for cid in b["ids"] if cid.startswith("other")]) == 2  # dataset 2

    b = batches[2]
    assert len([cid for cid in b["ids"] if cid.startswith("dummy")]) == 2  # dataset 1
    assert len([cid for cid in b["ids"] if cid.startswith("other")]) == 1  # dataset 2

    b = batches[3]
    assert len([cid for cid in b["ids"] if cid.startswith("dummy")]) == 3  # dataset 1
    assert len([cid for cid in b["ids"] if cid.startswith("other")]) == 0  # dataset 2

    b = batches[4]
    assert len([cid for cid in b["ids"] if cid.startswith("dummy")]) == 3  # dataset 1
    assert len([cid for cid in b["ids"] if cid.startswith("other")]) == 0  # dataset 2

    b = batches[5]
    assert len([cid for cid in b["ids"] if cid.startswith("dummy")]) == 3  # dataset 1
    assert len([cid for cid in b["ids"] if cid.startswith("other")]) == 0  # dataset 2


class TextDataset(torch.utils.data.Dataset):
    def __getitem__(self, cuts: lhotse.CutSet) -> List[str]:
        return [c.supervisions[0].text for c in cuts]


@pytest.mark.parametrize(["text_field", "text_value"], [(None, "irrelevant"), ("text-other", "not relevant")])
def test_dataloader_from_nemo_manifest_with_text_field(nemo_manifest_path: Path, text_field: str, text_value: str):
    kwarg = {"text_field": text_field} if text_field is not None else {}
    config = OmegaConf.create(
        {
            "manifest_filepath": nemo_manifest_path,
            "sample_rate": 16000,
            "shuffle": True,
            "use_lhotse": True,
            "num_workers": 0,
            "batch_size": 2,
            # lhotse specific
            "use_bucketing": False,
            **kwarg,
        }
    )

    dl = get_lhotse_dataloader_from_config(config=config, global_rank=0, world_size=1, dataset=TextDataset())
    b = next(iter(dl))
    assert b == [text_value] * 2


class LangDataset(torch.utils.data.Dataset):
    def __getitem__(self, cuts: lhotse.CutSet) -> List[str]:
        return [c.supervisions[0].language for c in cuts]


@pytest.mark.parametrize(["lang_field", "lang_value"], [(None, "en"), ("custom-lang", "pl")])
def test_dataloader_from_nemo_manifest_with_lang_field(nemo_manifest_path: Path, lang_field: str, lang_value: str):
    kwarg = {"lang_field": lang_field} if lang_field is not None else {}
    config = OmegaConf.create(
        {
            "manifest_filepath": nemo_manifest_path,
            "sample_rate": 16000,
            "shuffle": True,
            "use_lhotse": True,
            "num_workers": 0,
            "batch_size": 2,
            # lhotse specific
            "use_bucketing": False,
            **kwarg,
        }
    )

    dl = get_lhotse_dataloader_from_config(config=config, global_rank=0, world_size=1, dataset=LangDataset())
    b = next(iter(dl))
    assert b == [lang_value] * 2


def test_lazy_nemo_iterator_with_offset_field(tmp_path: Path):
    import numpy as np
    import soundfile as sf

    from nemo.collections.common.data.lhotse.nemo_adapters import LazyNeMoIterator

    # Have to generate as INT16 to avoid quantization error after saving to 16-bit WAV
    INT16MAX = 2**15
    expected_audio = np.random.randint(low=-INT16MAX - 1, high=INT16MAX, size=(16000,)).astype(np.float32) / INT16MAX
    audio_path = str(tmp_path / "dummy.wav")
    sf.write(audio_path, expected_audio, 16000)

    manifest_path = str(tmp_path / "manifest.json")
    lhotse.serialization.save_to_jsonl(
        [
            {"audio_filepath": audio_path, "offset": 0.0, "duration": 0.5, "text": "irrelevant"},
            {"audio_filepath": audio_path, "offset": 0.5, "duration": 0.5, "text": "irrelevant"},
        ],
        manifest_path,
    )

    cuts = lhotse.CutSet(LazyNeMoIterator(manifest_path))

    cut = cuts[0]
    assert isinstance(cut, lhotse.MonoCut)
    assert cut.start == 0.0
    assert cut.duration == 0.5
    assert cut.sampling_rate == 16000
    assert cut.num_samples == 8000
    assert cut.supervisions[0].text == "irrelevant"
    audio = cut.load_audio()
    assert audio.shape == (1, 8000)
    np.testing.assert_equal(audio[0], expected_audio[:8000])

    cut = cuts[1]
    assert isinstance(cut, lhotse.MonoCut)
    assert cut.start == 0.5
    assert cut.duration == 0.5
    assert cut.sampling_rate == 16000
    assert cut.num_samples == 8000
    assert cut.supervisions[0].text == "irrelevant"
    audio = cut.load_audio()
    assert audio.shape == (1, 8000)
    np.testing.assert_allclose(audio[0], expected_audio[8000:], atol=5e-5)

    assert cuts[0].id != cuts[1].id


def test_lazy_nemo_iterator_with_relative_paths(tmp_path: Path):
    import numpy as np
    import soundfile as sf

    from nemo.collections.common.data.lhotse.nemo_adapters import LazyNeMoIterator

    # Have to generate as INT16 to avoid quantization error after saving to 16-bit WAV
    INT16MAX = 2**15
    expected_audio = np.random.randint(low=-INT16MAX - 1, high=INT16MAX, size=(16000,)).astype(np.float32) / INT16MAX
    audio_path = str(tmp_path / "dummy.wav")
    sf.write(audio_path, expected_audio, 16000)

    manifest_path = str(tmp_path / "manifest.json")
    lhotse.serialization.save_to_jsonl(
        [
            # note: relative path
            {"audio_filepath": "dummy.wav", "offset": 0.0, "duration": 0.5, "text": "irrelevant"},
        ],
        manifest_path,
    )

    cuts = lhotse.CutSet(LazyNeMoIterator(manifest_path))
    cut = cuts[0]
    audio = cut.load_audio()

    assert isinstance(cut, lhotse.MonoCut)
    assert cut.start == 0.0
    assert cut.duration == 0.5
    assert cut.sampling_rate == 16000
    assert cut.num_samples == 8000
    assert cut.supervisions[0].text == "irrelevant"
    assert audio.shape == (1, 8000)
    np.testing.assert_equal(audio[0], expected_audio[:8000])


def test_lhotse_cuts_resolve_relative_paths(tmp_path: Path):
    cuts_path = tmp_path / "cuts.jsonl.gz"
    audio_path = tmp_path / "_relative_test_audio_.wav"
    lhotse.audio.save_audio(audio_path, np.random.rand(16000) - 0.5, 16000)
    cut = Recording.from_file(audio_path).to_cut()
    cut.recording.sources[0].source = str(audio_path.name)  # make the path relative
    cut.target_recording = cut.recording  # assign a custom field with relative path
    with NumpyFilesWriter(tmp_path) as w:
        cut.some_array = w.store_array(cut.id, np.random.randn(32))
        cut.some_array.storage_path = ""  # relative path

    with pytest.raises(AudioLoadingError):
        cut.load_audio()  # Lhotse doesn't know about what the path should be relative to
        cut.load_target_recording()

    CutSet([cut]).to_file(cuts_path)

    config = OmegaConf.create(
        {
            "cuts_path": cuts_path,
            "sample_rate": 16000,
            "use_lhotse": True,
            "num_workers": 0,
            "batch_size": 2,
        }
    )

    dl = get_lhotse_dataloader_from_config(config=config, global_rank=0, world_size=1, dataset=_Identity())

    batches = [batch for batch in dl]
    assert len(batches) == 1

    for cut in batches[0]:
        assert cut.has_recording
        cut.load_audio()  # works
        assert cut.has_custom("target_recording")
        cut.load_target_recording()
        assert cut.has_custom("some_array")
        cut.load_some_array()


class Identity(torch.utils.data.Dataset):
    def __getitem__(self, cuts: lhotse.CutSet) -> lhotse.CutSet:
        return cuts


def test_extended_data_input_cfg(cutset_shar_path, nemo_tarred_manifest_path_multi):
    config = OmegaConf.create(
        {
            "input_cfg": [
                {
                    "type": "nemo_tarred",
                    "manifest_filepath": nemo_tarred_manifest_path_multi[0],
                    "tarred_audio_filepaths": nemo_tarred_manifest_path_multi[1],
                    "weight": 0.5,
                    "tags": {
                        "language": "en",
                        "modality": "audio",
                        "dataset_name": "D1",
                    },
                },
                {
                    "type": "lhotse_shar",
                    "shar_path": cutset_shar_path,
                    "weight": 0.5,
                    "tags": {
                        "language": "en",
                        "modality": "audio",
                        "dataset_name": "D2",
                    },
                },
            ],
            "sample_rate": 16000,
            "shuffle": True,
            "num_workers": 0,
            "batch_size": 4,
            "seed": 0,
            "shard_seed": 0,
        }
    )

    dl = get_lhotse_dataloader_from_config(config=config, global_rank=0, world_size=1, dataset=Identity())

    # Note: we use islice here because the dataloader will be infinite.
    batches = [batch for batch in islice(dl, 2)]

    b = batches[0]
    assert isinstance(b, lhotse.CutSet)
    assert all(c.custom["language"] == "en" for c in b)
    assert all(c.custom["modality"] == "audio" for c in b)
    assert sum(c.custom["dataset_name"] == "D1" for c in b) == 2
    assert sum(c.custom["dataset_name"] == "D2" for c in b) == 2

    b = batches[1]
    assert isinstance(b, lhotse.CutSet)
    assert all(c.custom["language"] == "en" for c in b)
    assert all(c.custom["modality"] == "audio" for c in b)
    assert sum(c.custom["dataset_name"] == "D1" for c in b) == 1
    assert sum(c.custom["dataset_name"] == "D2" for c in b) == 3


def test_extended_data_input_cfg_subgroup(cutset_shar_path, nemo_tarred_manifest_path_multi):
    config = OmegaConf.create(
        {
            "input_cfg": [
                {
                    "type": "group",
                    "input_cfg": [
                        {
                            "type": "nemo_tarred",
                            "manifest_filepath": nemo_tarred_manifest_path_multi[0],
                            "tarred_audio_filepaths": nemo_tarred_manifest_path_multi[1],
                            "weight": 0.5,
                            "tags": {
                                "language": "en",
                                "modality": "audio",
                                "dataset_name": "D1",
                            },
                        },
                        {
                            "type": "lhotse_shar",
                            "shar_path": cutset_shar_path,
                            "weight": 0.5,
                            "tags": {
                                "language": "en",
                                "modality": "audio",
                                "dataset_name": "D2",
                            },
                        },
                    ],
                    "weight": 0.2,
                    "tags": {
                        "group_name": "G1",
                    },
                },
                {
                    "type": "group",
                    "weight": 0.8,
                    "input_cfg": [
                        {
                            "type": "nemo_tarred",
                            "manifest_filepath": nemo_tarred_manifest_path_multi[0],
                            "tarred_audio_filepaths": nemo_tarred_manifest_path_multi[1],
                            "weight": 0.5,
                            "tags": {
                                "language": "en",
                                "modality": "audio",
                                "dataset_name": "D3",
                            },
                        },
                        {
                            "type": "lhotse_shar",
                            "shar_path": cutset_shar_path,
                            "weight": 0.5,
                            "tags": {
                                "language": "en",
                                "modality": "audio",
                                "dataset_name": "D4",
                            },
                        },
                    ],
                    "tags": {
                        "group_name": "G2",
                    },
                },
            ],
            "sample_rate": 16000,
            "shuffle": True,
            "num_workers": 0,
            "batch_size": 32,
            "seed": 0,
            "shard_seed": 0,
        }
    )

    dl = get_lhotse_dataloader_from_config(config=config, global_rank=0, world_size=1, dataset=Identity())

    # Sample 100 mini-batches and test statistical properties
    group_occurrences = Counter()
    dataset_occurrences = Counter()
    for batch in islice(dl, 100):
        for cut in batch:
            group_occurrences[cut.group_name] += 1
            dataset_occurrences[cut.dataset_name] += 1

    tot = sum(group_occurrences.values())
    for k in group_occurrences:
        group_occurrences[k] /= tot
    for k in dataset_occurrences:
        dataset_occurrences[k] /= tot

    def almost(number):
        return pytest.approx(number, abs=0.02)

    assert group_occurrences["G1"] == almost(0.2)  # group weight: 0.2
    assert group_occurrences["G2"] == almost(0.8)  # group weight: 0.8
    assert dataset_occurrences["D1"] == almost(0.1)  # group weight: 0.2 * dataset weight 0.5 => 0.1
    assert dataset_occurrences["D2"] == almost(0.1)  # group weight: 0.2 * dataset weight 0.5 => 0.1
    assert dataset_occurrences["D3"] == almost(0.4)  # group weight: 0.8 * dataset weight 0.5 => 0.4
    assert dataset_occurrences["D4"] == almost(0.4)  # group weight: 0.8 * dataset weight 0.5 => 0.4


def test_extended_data_input_cfg_yaml_path(tmp_path, cutset_shar_path, nemo_tarred_manifest_path_multi):
    input_cfg = [
        {
            "type": "nemo_tarred",
            "manifest_filepath": str(nemo_tarred_manifest_path_multi[0]),
            "tarred_audio_filepaths": str(nemo_tarred_manifest_path_multi[1]),
            "weight": 0.5,
            "tags": {
                "language": "en",
                "modality": "audio",
                "dataset_name": "D1",
            },
        },
        {
            "type": "lhotse_shar",
            "shar_path": str(cutset_shar_path),
            "weight": 0.5,
            "tags": {
                "language": "en",
                "modality": "audio",
                "dataset_name": "D2",
            },
        },
    ]

    yaml_path = tmp_path / "input_cfg.yaml"
    lhotse.serialization.save_to_yaml(input_cfg, yaml_path)

    config = OmegaConf.create(
        {
            "input_cfg": input_cfg,
            "sample_rate": 16000,
            "shuffle": True,
            "num_workers": 0,
            "batch_size": 32,
            "seed": 0,
            "shard_seed": 0,
        }
    )

    dl = get_lhotse_dataloader_from_config(config=config, global_rank=0, world_size=1, dataset=Identity())

    batch = next(iter(dl))
    assert isinstance(batch, lhotse.CutSet)
    for cut in batch:
        assert cut.dataset_name in ("D1", "D2")


@pytest.fixture(scope="session")
def txt_en_path(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("text_data")
    en_path = tmp_path / "text.en"
    en_path.write_text(
        """Example text in English.
Another sentence.
        """
    )
    return en_path


@pytest.fixture(scope="session")
def txt_es_path(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("text_data")
    es_path = tmp_path / "text.es"
    es_path.write_text(
        """Otro texto en ingles.
Otra frase."""
    )
    return es_path


def test_text_file_input(txt_en_path, txt_es_path):
    config = OmegaConf.create(
        {
            "input_cfg": [
                {
                    "type": "txt",
                    "paths": txt_en_path,
                    "language": "en",
                },
            ],
            "shuffle": True,
            "num_workers": 0,
            "batch_size": 4,
            "seed": 0,
            "shard_seed": 0,
        }
    )

    # Note: this test does not need to pass a tokenizer because we use static batch sizes
    dl = get_lhotse_dataloader_from_config(config=config, global_rank=0, world_size=1, dataset=Identity())

    # Note: we use islice here because the dataloader will be infinite.
    batches = [batch for batch in islice(dl, 2)]

    b = batches[0]
    assert isinstance(b, lhotse.CutSet)
    assert all(isinstance(c, TextExample) for c in b)
    assert all(c.language == "en" for c in b)

    b = batches[1]
    assert isinstance(b, lhotse.CutSet)
    assert all(isinstance(c, TextExample) for c in b)
    assert all(c.language == "en" for c in b)


def test_text_file_pairs_input(txt_en_path, txt_es_path):
    config = OmegaConf.create(
        {
            "input_cfg": [
                {
                    "type": "txt_pair",
                    "source_paths": txt_en_path,
                    "target_paths": txt_es_path,
                    "source_language": "en",
                    "target_language": "es",
                },
            ],
            "shuffle": True,
            "num_workers": 0,
            "batch_size": 4,
            "seed": 0,
            "shard_seed": 0,
        }
    )

    # Note: this test does not need to pass a tokenizer because we use static batch sizes
    dl = get_lhotse_dataloader_from_config(config=config, global_rank=0, world_size=1, dataset=Identity())

    # Note: we use islice here because the dataloader will be infinite.
    batches = [batch for batch in islice(dl, 2)]

    b = batches[0]
    assert isinstance(b, lhotse.CutSet)
    assert all(isinstance(c, TextPairExample) for c in b)
    assert all(c.source.language == "en" for c in b)
    assert all(c.target.language == "es" for c in b)

    b = batches[1]
    assert isinstance(b, lhotse.CutSet)
    assert all(isinstance(c, TextPairExample) for c in b)
    assert all(c.source.language == "en" for c in b)
    assert all(c.target.language == "es" for c in b)


@pytest.fixture(scope="session")
def txt_pair_paths_shards(tmp_path_factory, txt_en_path, txt_es_path):
    tmp_path = tmp_path_factory.mktemp("text_data_shards")

    en_text = txt_en_path.read_text().splitlines()
    (tmp_path / "en_0.txt").write_text("\n".join(en_text[:5]))
    (tmp_path / "en_1.txt").write_text("\n".join(en_text[5:]))

    es_text = txt_es_path.read_text().splitlines()
    (tmp_path / "es_0.txt").write_text("\n".join(es_text[:5]))
    (tmp_path / "es_1.txt").write_text("\n".join(es_text[5:]))

    return f"{tmp_path}/en__OP_0..1_CL_.txt", f"{tmp_path}/es__OP_0..1_CL_.txt"


def test_text_file_pairs_shards_input(txt_pair_paths_shards: tuple[str, str]):
    en_paths, es_paths = txt_pair_paths_shards

    config = OmegaConf.create(
        {
            "input_cfg": [
                {
                    "type": "txt_pair",
                    "source_paths": en_paths,
                    "target_paths": es_paths,
                    "source_language": "en",
                    "target_language": "es",
                },
            ],
            "shuffle": True,
            "num_workers": 0,
            "batch_size": 4,
            "seed": 0,
            "shard_seed": 0,
        }
    )

    # Note: this test does not need to pass a tokenizer because we use static batch sizes
    dl = get_lhotse_dataloader_from_config(config=config, global_rank=0, world_size=1, dataset=Identity())

    # Note: we use islice here because the dataloader will be infinite.
    batches = [batch for batch in islice(dl, 2)]

    b = batches[0]
    assert isinstance(b, lhotse.CutSet)
    assert all(isinstance(c, TextPairExample) for c in b)
    assert all(c.source.language == "en" for c in b)
    assert all(c.target.language == "es" for c in b)

    b = batches[1]
    assert isinstance(b, lhotse.CutSet)
    assert all(isinstance(c, TextPairExample) for c in b)
    assert all(c.source.language == "en" for c in b)
    assert all(c.target.language == "es" for c in b)


@pytest.fixture(scope="session")
def en_es_tokenizer(tmp_path_factory, txt_en_path, txt_es_path) -> TokenizerWrapper:
    tmpdir = tmp_path_factory.mktemp("en_es_tokenizer")
    text_path = tmpdir / "text.txt"
    text_path.write_text(txt_en_path.read_text() + "\n" + txt_es_path.read_text())
    create_spt_model(text_path, vocab_size=128, sample_size=-1, do_lower_case=False, output_dir=str(tmpdir))
    return TokenizerWrapper(SentencePieceTokenizer(str(tmpdir / "tokenizer.model")))


def test_multimodal_text_audio_dataloading(
    txt_pair_paths_shards: tuple[str, str],
    nemo_tarred_manifest_path_multi: tuple[str, str],
    en_es_tokenizer: TokenizerWrapper,
):
    en_paths, es_paths = txt_pair_paths_shards
    manifest_filepath, tarred_audio_filepaths = nemo_tarred_manifest_path_multi
    config = OmegaConf.create(
        {
            "input_cfg": [
                {
                    "type": "txt_pair",
                    "source_paths": en_paths,
                    "target_paths": es_paths,
                    "source_language": "en",
                    "target_language": "es",
                    "tags": {
                        "modality": "text",
                    },
                },
                {
                    "type": "nemo_tarred",
                    "manifest_filepath": manifest_filepath,
                    "tarred_audio_filepaths": tarred_audio_filepaths,
                    "tags": {
                        "modality": "audio",
                    },
                },
            ],
            "shuffle": True,
            "num_workers": 0,
            "use_multimodal_sampling": True,
            "batch_tokens": 1024,
            # How to set token equivalent duration in actual training?
            #   assuming fbank frames: 0.01 is the base due to frame shift;
            #       + subsampling x8 gives us 0.08
            #   assuming discrete audio tokens, with frame rate 50Hz,
            #       we'd get 0.02
            #   in this test we'll just use 0.1 for simplicity
            "token_equivalent_duration": 0.1,
            "quadratic_factor": 50,
            "seed": 0,
            "shard_seed": 0,
        }
    )

    dl = get_lhotse_dataloader_from_config(
        config=config,
        global_rank=0,
        world_size=1,
        dataset=Identity(),
        tokenizer=en_es_tokenizer,
    )

    # Note: we use islice here because the dataloader will be infinite.
    batches = [batch for batch in islice(dl, 2)]

    b = batches[0]
    assert isinstance(b, lhotse.CutSet)
    assert len(b) == 48
    assert sum(ex.num_tokens for ex in b) == pytest.approx(574.0)
    assert min(ex.num_tokens for ex in b) == pytest.approx(10)
    assert max(ex.num_tokens for ex in b) == pytest.approx(16)
    assert sum(isinstance(ex, Cut) for ex in b) == 29
    assert sum(isinstance(ex, TextPairExample) for ex in b) == 19
    for ex in b:
        if isinstance(ex, Cut):
            assert ex.modality == "audio"
            assert isinstance(ex.load_audio(), np.ndarray)
            assert isinstance(ex.supervisions[0].text, str)
        if isinstance(ex, TextPairExample):
            assert ex.modality == "text"
            assert ex.source.language == "en"
            assert ex.target.language == "es"
            assert isinstance(ex.source.text, str)
            assert isinstance(ex.target.text, str)
            assert isinstance(ex.source.tokens, np.ndarray)
            assert isinstance(ex.target.tokens, np.ndarray)

    b = batches[1]
    assert isinstance(b, lhotse.CutSet)
    assert len(b) == 48
    assert sum(ex.num_tokens for ex in b) == pytest.approx(614.0)
    assert min(ex.num_tokens for ex in b) == pytest.approx(10)
    assert max(ex.num_tokens for ex in b) == pytest.approx(16)
    assert sum(isinstance(ex, Cut) for ex in b) == 21
    assert sum(isinstance(ex, TextPairExample) for ex in b) == 27
    for ex in b:
        if isinstance(ex, Cut):
            assert ex.modality == "audio"
            assert isinstance(ex.load_audio(), np.ndarray)
            assert isinstance(ex.supervisions[0].text, str)
        if isinstance(ex, TextPairExample):
            assert ex.modality == "text"
            assert ex.source.language == "en"
            assert ex.target.language == "es"
            assert isinstance(ex.source.text, str)
            assert isinstance(ex.target.text, str)
            assert isinstance(ex.source.tokens, np.ndarray)
            assert isinstance(ex.target.tokens, np.ndarray)


def test_dataloader_with_noise_nemo_json(cutset_path: Path, nemo_manifest_path: Path):
    config = OmegaConf.create(
        {
            "cuts_path": str(cutset_path),
            "noise_path": str(nemo_manifest_path),
            "noise_mix_prob": 1.0,
            "noise_snr": [-5.0, 5.0],
            "batch_size": 2,
            "seed": 0,
            "shard_seed": 0,
        }
    )
    dl = get_lhotse_dataloader_from_config(
        config=config,
        global_rank=0,
        world_size=1,
        dataset=Identity(),
    )
    batch = next(iter(dl))
    assert isinstance(batch, CutSet)
    assert len(batch) == 2
    cut = batch[0]
    assert isinstance(cut, MixedCut)
    assert -5.0 < cut.tracks[1].snr < 5.0
    cut = batch[1]
    assert isinstance(cut, MixedCut)
    assert -5.0 < cut.tracks[1].snr < 5.0


def test_dataloader_with_noise_lhotse_jsonl(cutset_path: Path):
    config = OmegaConf.create(
        {
            "cuts_path": str(cutset_path),
            "noise_path": str(cutset_path),
            "noise_mix_prob": 1.0,
            "noise_snr": [-5.0, 5.0],
            "batch_size": 2,
            "seed": 0,
            "shard_seed": 0,
        }
    )
    dl = get_lhotse_dataloader_from_config(
        config=config,
        global_rank=0,
        world_size=1,
        dataset=Identity(),
    )
    batch = next(iter(dl))
    assert isinstance(batch, CutSet)
    assert len(batch) == 2
    cut = batch[0]
    assert isinstance(cut, MixedCut)
    assert -5.0 < cut.tracks[1].snr < 5.0
    cut = batch[1]
    assert isinstance(cut, MixedCut)
    assert -5.0 < cut.tracks[1].snr < 5.0


def test_dataloader_with_noise_nemo_tar(cutset_path: Path, nemo_tarred_manifest_path_multi: Path):
    noise_json, noise_tar = nemo_tarred_manifest_path_multi
    config = OmegaConf.create(
        {
            "cuts_path": str(cutset_path),
            "noise_path": {
                "manifest_filepath": noise_json,
                "tarred_audio_filepaths": noise_tar,
            },
            "noise_mix_prob": 1.0,
            "noise_snr": [-5.0, 5.0],
            "batch_size": 2,
            "seed": 0,
            "shard_seed": 0,
        }
    )
    dl = get_lhotse_dataloader_from_config(
        config=config,
        global_rank=0,
        world_size=1,
        dataset=Identity(),
    )
    batch = next(iter(dl))
    assert isinstance(batch, CutSet)
    assert len(batch) == 2
    cut = batch[0]
    assert isinstance(cut, MixedCut)
    assert -5.0 < cut.tracks[1].snr < 5.0
    cut = batch[1]
    assert isinstance(cut, MixedCut)
    assert -5.0 < cut.tracks[1].snr < 5.0


def test_dataloader_with_synth_rir(cutset_path: Path):
    from lhotse.augmentation import ReverbWithImpulseResponse

    config = OmegaConf.create(
        {
            "cuts_path": str(cutset_path),
            "rir_enabled": True,
            "rir_prob": 0.5,
            "batch_size": 4,
            "seed": 0,
            "shard_seed": 0,
        }
    )
    dl = get_lhotse_dataloader_from_config(
        config=config,
        global_rank=0,
        world_size=1,
        dataset=Identity(),
    )
    batch = next(iter(dl))
    assert isinstance(batch, CutSet)
    assert len(batch) == 4
    cut = batch[0]
    assert isinstance(cut, MonoCut)
    assert cut.recording.transforms is None
    cut = batch[1]
    assert isinstance(cut, MonoCut)
    assert cut.recording.transforms is None
    cut = batch[2]
    assert isinstance(cut, MonoCut)
    assert isinstance(cut.recording.transforms, list) and len(cut.recording.transforms) == 1
    tfnm = cut.recording.transforms[0]
    if isinstance(tfnm, dict):  # lhotse<=1.23.0
        assert tfnm["name"] == "ReverbWithImpulseResponse"
    else:  # lhotse>=1.24.0
        assert isinstance(tfnm, ReverbWithImpulseResponse)
    cut = batch[3]
    assert isinstance(cut, MonoCut)
    assert isinstance(cut.recording.transforms, list) and len(cut.recording.transforms) == 1
    tfnm = cut.recording.transforms[0]
    if isinstance(tfnm, dict):  # lhotse<=1.23.0
        assert tfnm["name"] == "ReverbWithImpulseResponse"
    else:  # lhotse>=1.24.0
        assert isinstance(tfnm, ReverbWithImpulseResponse)


@pytest.fixture(scope="session")
def questions_path(tmp_path_factory) -> Path:
    """A text file with 10 lines containing question values"""
    qdir = tmp_path_factory.mktemp("questions")
    path = qdir / "questions.txt"
    path.write_text("\n".join(f"some question number {i}" for i in range(10)))
    return path


def test_dataloader_from_nemo_nontarred_manifest_with_extra_questions_field_iter(
    nemo_manifest_path: Path, questions_path: Path
):
    config = OmegaConf.create(
        {
            "input_cfg": [
                {
                    "manifest_filepath": nemo_manifest_path,
                    "type": "nemo",
                    "extra_fields": [
                        {
                            "type": "text_iter",
                            "name": "question",
                            "path": questions_path,
                        }
                    ],
                },
            ],
            "sample_rate": 16000,
            "shuffle": False,
            "use_lhotse": True,
            "num_workers": 0,
            "batch_size": 2,
            "use_bucketing": False,
        }
    )

    dl = get_lhotse_dataloader_from_config(config=config, global_rank=0, world_size=1, dataset=Identity())
    b = next(iter(dl))
    c = b[0]
    assert isinstance(c, MonoCut)
    assert hasattr(c, "question")
    assert c.question == "some question number 0"
    c = b[1]
    assert isinstance(c, MonoCut)
    assert hasattr(c, "question")
    assert c.question == "some question number 1"


def test_dataloader_from_nemo_manifest_with_extra_questions_field_iter(
    nemo_tarred_manifest_path: tuple, questions_path: Path
):
    config = OmegaConf.create(
        {
            "input_cfg": [
                {
                    "manifest_filepath": nemo_tarred_manifest_path[0],
                    "tarred_audio_filepaths": nemo_tarred_manifest_path[1],
                    "type": "nemo_tarred",
                    "extra_fields": [
                        {
                            "type": "text_iter",
                            "name": "question",
                            "path": questions_path,
                        }
                    ],
                },
            ],
            "sample_rate": 16000,
            "shuffle": False,
            "use_lhotse": True,
            "num_workers": 0,
            "batch_size": 2,
            "use_bucketing": False,
        }
    )

    dl = get_lhotse_dataloader_from_config(config=config, global_rank=0, world_size=1, dataset=Identity())
    b = next(iter(dl))
    c = b[0]
    assert isinstance(c, MonoCut)
    assert hasattr(c, "question")
    assert c.question == "some question number 0"
    c = b[1]
    assert isinstance(c, MonoCut)
    assert hasattr(c, "question")
    assert c.question == "some question number 1"


def test_dataloader_from_nemo_manifest_with_extra_questions_field_sample(
    nemo_tarred_manifest_path: tuple, questions_path: Path
):
    config = OmegaConf.create(
        {
            "input_cfg": [
                {
                    "manifest_filepath": nemo_tarred_manifest_path[0],
                    "tarred_audio_filepaths": nemo_tarred_manifest_path[1],
                    "type": "nemo_tarred",
                    "extra_fields": [
                        {
                            "type": "text_sample",
                            "name": "question",
                            "path": questions_path,
                        }
                    ],
                },
            ],
            "sample_rate": 16000,
            "shuffle": False,
            "use_lhotse": True,
            "num_workers": 0,
            "batch_size": 5,
            "seed": 0,
            "shard_seed": 0,
            "use_bucketing": False,
        }
    )

    # Note: despite shuffle=True, it is sampling lines from questions_path because of type: "text_sample"
    dl = get_lhotse_dataloader_from_config(config=config, global_rank=0, world_size=1, dataset=Identity())
    b = next(iter(dl))
    c = b[0]
    assert isinstance(c, MonoCut)
    assert hasattr(c, "question")
    assert c.question == "some question number 6"
    c = b[1]
    assert isinstance(c, MonoCut)
    assert hasattr(c, "question")
    assert c.question == "some question number 6"
    c = b[2]
    assert isinstance(c, MonoCut)
    assert hasattr(c, "question")
    assert c.question == "some question number 0"
    c = b[3]
    assert isinstance(c, MonoCut)
    assert hasattr(c, "question")
    assert c.question == "some question number 4"
    c = b[4]
    assert isinstance(c, MonoCut)
    assert hasattr(c, "question")
    assert c.question == "some question number 8"
