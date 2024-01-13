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

from io import BytesIO
from itertools import islice
from pathlib import Path
from typing import Dict, List, Tuple

import pytest
import torch
from omegaconf import OmegaConf

from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config

lhotse = pytest.importorskip("lhotse", reason="Lhotse + NeMo tests require Lhotse (pip install lhotse).")

requires_torchaudio = pytest.mark.skipif(
    not lhotse.utils.is_torchaudio_available(), reason="Lhotse Shar format support requires torchaudio."
)


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
def nemo_tarred_manifest_path(nemo_manifest_path: Path) -> Tuple[str, str]:
    """10 utterances of length 1s as a NeMo tarred manifest."""
    from lhotse.serialization import SequentialJsonlWriter, load_jsonl
    from lhotse.shar.writers import TarWriter

    root = nemo_manifest_path.parent / "nemo_tar"
    root.mkdir(exist_ok=True)

    with TarWriter(f"{root}/audios_%01d.tar", shard_size=5) as tar_writer, SequentialJsonlWriter(
        root / "tarred_audio_filepaths.jsonl"
    ) as mft_writer:
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


@requires_torchaudio
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
            "buffer_size": 100,
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
        }
    )

    class _IdentityDataset:
        def __getitem__(self, cuts):
            return cuts

    dl = get_lhotse_dataloader_from_config(config=config, global_rank=0, world_size=1, dataset=_IdentityDataset())

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
        }
    )

    dl = get_lhotse_dataloader_from_config(
        config=config, global_rank=0, world_size=1, dataset=UnsupervisedAudioDataset()
    )

    batches = [batch for batch in dl]

    print(batches)
    print(batches[0])

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
    assert b["audio"].shape[0] == b["audio_lens"].shape[0] == 1
    torch.testing.assert_close(b["audio_lens"], torch.tensor([16000], dtype=torch.int32))


@requires_torchaudio
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
            "shar_seed": 0,
        }
    )

    dl = get_lhotse_dataloader_from_config(
        config=config, global_rank=0, world_size=1, dataset=UnsupervisedAudioDataset()
    )

    # Note: we use islice here because with Lhotse Shar the dataloader will always be infinite.
    batches = [batch for batch in islice(dl, 4)]
    assert len(batches) == 4

    b = batches[0]
    assert len([cid for cid in b["ids"] if cid.startswith("dummy")]) == 3  # dataset 1
    assert len([cid for cid in b["ids"] if cid.startswith("other")]) == 0  # dataset 2

    b = batches[1]
    assert len([cid for cid in b["ids"] if cid.startswith("dummy")]) == 0  # dataset 1
    assert len([cid for cid in b["ids"] if cid.startswith("other")]) == 3  # dataset 2

    b = batches[2]
    assert len([cid for cid in b["ids"] if cid.startswith("dummy")]) == 1  # dataset 1
    assert len([cid for cid in b["ids"] if cid.startswith("other")]) == 2  # dataset 2

    b = batches[3]
    assert len([cid for cid in b["ids"] if cid.startswith("dummy")]) == 2  # dataset 1
    assert len([cid for cid in b["ids"] if cid.startswith("other")]) == 1  # dataset 2


@requires_torchaudio
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
            "shar_seed": 0,
        }
    )

    dl = get_lhotse_dataloader_from_config(
        config=config, global_rank=0, world_size=1, dataset=UnsupervisedAudioDataset()
    )

    # Note: we use islice here because with Lhotse Shar the dataloader will always be infinite.
    batches = [batch for batch in islice(dl, 4)]
    assert len(batches) == 4

    b = batches[0]
    assert len([cid for cid in b["ids"] if cid.startswith("dummy")]) == 3  # dataset 1
    assert len([cid for cid in b["ids"] if cid.startswith("other")]) == 0  # dataset 2

    b = batches[1]
    assert len([cid for cid in b["ids"] if cid.startswith("dummy")]) == 2  # dataset 1
    assert len([cid for cid in b["ids"] if cid.startswith("other")]) == 1  # dataset 2

    b = batches[2]
    assert len([cid for cid in b["ids"] if cid.startswith("dummy")]) == 3  # dataset 1
    assert len([cid for cid in b["ids"] if cid.startswith("other")]) == 0  # dataset 2

    b = batches[3]
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
