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
from lhotse.cut import Cut
from lhotse.cut.text import TextPairExample
from omegaconf import OmegaConf

from nemo.collections.asr.data.audio_to_text_lhotse import TokenizerWrapper
from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config
from nemo.collections.common.data.lhotse.text_adapters import TextExample
from nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer, create_spt_model

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
    assert len([cid for cid in b["ids"] if cid.startswith("dummy")]) == 2  # dataset 1
    assert len([cid for cid in b["ids"] if cid.startswith("other")]) == 1  # dataset 2

    b = batches[2]
    assert len([cid for cid in b["ids"] if cid.startswith("dummy")]) == 1  # dataset 1
    assert len([cid for cid in b["ids"] if cid.startswith("other")]) == 2  # dataset 2

    b = batches[3]
    assert len([cid for cid in b["ids"] if cid.startswith("dummy")]) == 1  # dataset 1
    assert len([cid for cid in b["ids"] if cid.startswith("other")]) == 2  # dataset 2


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
    assert len([cid for cid in b["ids"] if cid.startswith("dummy")]) == 3  # dataset 1
    assert len([cid for cid in b["ids"] if cid.startswith("other")]) == 0  # dataset 2

    b = batches[2]
    assert len([cid for cid in b["ids"] if cid.startswith("dummy")]) == 3  # dataset 1
    assert len([cid for cid in b["ids"] if cid.startswith("other")]) == 0  # dataset 2

    b = batches[3]
    assert len([cid for cid in b["ids"] if cid.startswith("dummy")]) == 3  # dataset 1
    assert len([cid for cid in b["ids"] if cid.startswith("other")]) == 0  # dataset 2

    b = batches[4]
    assert len([cid for cid in b["ids"] if cid.startswith("dummy")]) == 3  # dataset 1
    assert len([cid for cid in b["ids"] if cid.startswith("other")]) == 0  # dataset 2

    b = batches[5]
    assert len([cid for cid in b["ids"] if cid.startswith("dummy")]) == 1  # dataset 1
    assert len([cid for cid in b["ids"] if cid.startswith("other")]) == 2  # dataset 2


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
    INT16MAX = 2 ** 15
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
    INT16MAX = 2 ** 15
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
    np.testing.assert_allclose(audio[0], expected_audio[:8000], atol=5e-5)


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
                    "tags": {"language": "en", "modality": "audio", "dataset_name": "D1",},
                },
                {
                    "type": "lhotse_shar",
                    "shar_path": cutset_shar_path,
                    "weight": 0.5,
                    "tags": {"language": "en", "modality": "audio", "dataset_name": "D2",},
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
                            "tags": {"language": "en", "modality": "audio", "dataset_name": "D1",},
                        },
                        {
                            "type": "lhotse_shar",
                            "shar_path": cutset_shar_path,
                            "weight": 0.5,
                            "tags": {"language": "en", "modality": "audio", "dataset_name": "D2",},
                        },
                    ],
                    "weight": 0.2,
                    "tags": {"group_name": "G1",},
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
                            "tags": {"language": "en", "modality": "audio", "dataset_name": "D3",},
                        },
                        {
                            "type": "lhotse_shar",
                            "shar_path": cutset_shar_path,
                            "weight": 0.5,
                            "tags": {"language": "en", "modality": "audio", "dataset_name": "D4",},
                        },
                    ],
                    "tags": {"group_name": "G2",},
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
            "tags": {"language": "en", "modality": "audio", "dataset_name": "D1",},
        },
        {
            "type": "lhotse_shar",
            "shar_path": str(cutset_shar_path),
            "weight": 0.5,
            "tags": {"language": "en", "modality": "audio", "dataset_name": "D2",},
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
            "input_cfg": [{"type": "txt", "paths": txt_en_path, "language": "en",},],
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
                    "tags": {"modality": "text",},
                },
                {
                    "type": "nemo_tarred",
                    "manifest_filepath": manifest_filepath,
                    "tarred_audio_filepaths": tarred_audio_filepaths,
                    "tags": {"modality": "audio",},
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
        config=config, global_rank=0, world_size=1, dataset=Identity(), tokenizer=en_es_tokenizer,
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
