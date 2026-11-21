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

import io
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
from omegaconf import OmegaConf

# We need this try/except because the test environment might not have pyarrow/pandas
try:
    import pyarrow as pa
    import pyarrow.parquet as pq

    HAVE_PARQUET = True
except ImportError:
    HAVE_PARQUET = False

from nemo.collections.common.data.lhotse.cutset import read_cutset_from_config
from nemo.collections.common.data.lhotse.nemo_adapters import ParquetRowError


@pytest.fixture
def parquet_dataset(tmp_path):
    """
    Creates a dummy Parquet dataset with 5 audio files embedded as bytes.
    Returns the path to the .parquet file.
    """
    if not HAVE_PARQUET:
        pytest.skip("PyArrow not installed, skipping Parquet tests.")

    data = []
    sr = 16000
    for i in range(5):
        # Generate 1 second of random noise
        audio = np.random.uniform(-1, 1, sr).astype(np.float32)

        # Write to bytes using SoundFile
        buf = io.BytesIO()
        sf.write(buf, audio, sr, format='WAV')
        audio_bytes = buf.getvalue()

        data.append(
            {
                "id": f"audio_{i}",
                "audio_col": audio_bytes,  # Custom column name to test mapping
                "text_col": f"This is sentence {i}",
                "duration": 1.0,
                "lang": "en",
            }
        )

    df = pd.DataFrame(data)
    path = tmp_path / "test_data.parquet"

    # Save using pyarrow engine
    table = pa.Table.from_pandas(df)
    pq.write_table(table, path)

    return str(path)


@pytest.mark.skipif(not HAVE_PARQUET, reason="PyArrow required for this test")
def test_read_parquet_manifest(parquet_dataset):
    """
    Test that we can read the parquet file using the registry mechanism.
    """
    config = OmegaConf.create(
        {
            "input_cfg": [
                {
                    "type": "parquet",
                    "manifest_filepath": parquet_dataset,
                    "audio_field": "audio_col",
                    "text_field": "text_col",
                }
            ],
            "batch_size": 2,
            "force_finite": True,
        }
    )

    # This calls read_dataset_config internally
    cuts, is_tarred = read_cutset_from_config(config)

    assert is_tarred is True

    # Iterate and check data
    cuts_list = list(cuts)
    assert len(cuts_list) == 5

    first_cut = cuts_list[0]
    assert first_cut.id == "audio_0"
    assert first_cut.supervisions[0].text == "This is sentence 0"

    # Check if we can actually decode audio
    audio = first_cut.load_audio()
    assert audio.shape[1] == 16000  # 1 second at 16khz


@pytest.mark.skipif(not HAVE_PARQUET, reason="PyArrow required for this test")
def test_read_parquet_sharding(parquet_dataset):
    """
    Test that we can handle a list of files (sharding) using LazyIteratorChain.
    """
    config = OmegaConf.create(
        {
            "input_cfg": [
                {
                    "type": "parquet",
                    "manifest_filepath": [parquet_dataset, parquet_dataset],
                    "audio_field": "audio_col",
                    "text_field": "text_col",
                    "shuffle": True,
                    "shard_seed": 42,
                }
            ],
            "force_finite": True,
        }
    )

    cuts, is_tarred = read_cutset_from_config(config)
    cuts_list = list(cuts)

    # 5 items per file * 2 files = 10 items
    assert len(cuts_list) == 10


@pytest.fixture
def parquet_with_corrupt_row(tmp_path):
    """A Parquet file whose first row has valid audio and whose second row has undecodable audio bytes."""
    if not HAVE_PARQUET:
        pytest.skip("PyArrow not installed, skipping Parquet tests.")
    buf = io.BytesIO()
    sf.write(buf, np.zeros(1600, dtype=np.float32), 16000, format='WAV')
    table = pa.table(
        {
            "id": ["good", "corrupt"],
            "audio": [buf.getvalue(), b"this is not audio"],
            "text": ["hello", "world"],
            "duration": [0.1, 0.1],
        }
    )
    path = tmp_path / "corrupt.parquet"
    pq.write_table(table, path)
    return str(path)


@pytest.mark.unit
@pytest.mark.skipif(not HAVE_PARQUET, reason="PyArrow required for this test")
@pytest.mark.parametrize("indexed", [False, True])
def test_parquet_corrupt_audio_raises_by_default(parquet_with_corrupt_row, indexed):
    it = _parquet_iterator(parquet_with_corrupt_row, indexed=indexed)
    with pytest.raises(RuntimeError, match=r"row 1 \(id='corrupt'\) of Parquet file .*corrupt\.parquet"):
        list(it)


@pytest.mark.unit
@pytest.mark.skipif(not HAVE_PARQUET, reason="PyArrow required for this test")
def test_parquet_corrupt_audio_getitem_raises(parquet_with_corrupt_row):
    it = _parquet_iterator(parquet_with_corrupt_row, indexed=True, fault_tolerant_audio_loading=True)
    assert it[0].id == "good"
    # Random access cannot skip a row, so it raises a descriptive error even in fault-tolerant mode.
    with pytest.raises(RuntimeError, match=r"row 1 \(id='corrupt'\) of Parquet file .*failed to decode audio"):
        it.__getitem__(1)


@pytest.mark.unit
@pytest.mark.skipif(not HAVE_PARQUET, reason="PyArrow required for this test")
@pytest.mark.parametrize("indexed", [False, True])
def test_parquet_corrupt_audio_skipped_when_fault_tolerant(parquet_with_corrupt_row, indexed, monkeypatch):
    from nemo.collections.common.data.lhotse import nemo_adapters

    warnings = []
    monkeypatch.setattr(nemo_adapters.logging, "warning", lambda msg, *a, **kw: warnings.append(msg))
    it = _parquet_iterator(parquet_with_corrupt_row, indexed=indexed, fault_tolerant_audio_loading=True)
    assert [c.id for c in it] == ["good"]
    assert any("Skipped 1 of 2 rows" in w for w in warnings), warnings


@pytest.mark.unit
@pytest.mark.skipif(not HAVE_PARQUET, reason="PyArrow required for this test")
@pytest.mark.parametrize("indexed", [False, True])
@pytest.mark.parametrize("fault_tolerant", [False, True])
# Parameters that need pyarrow are lambdas, so collection works without pyarrow (the tests are then skipped).
@pytest.mark.parametrize(
    "make_columns,match",
    [
        (
            lambda: {"audio": pa.array([None], type=pa.binary())},
            r"audio column 'audio' holds NoneType, expected bytes",
        ),
        (lambda: {"audio": pa.array([1], type=pa.int64())}, r"audio column 'audio' holds int, expected bytes"),
        (lambda: {"text": pa.array([None], type=pa.string())}, r"text column 'text' holds NoneType, expected str"),
        (lambda: {"duration": [-1.0]}, r"duration column 'duration' holds -1.0, expected a non-negative number"),
        (lambda: {"duration": [float("nan")]}, r"duration column 'duration' holds nan"),
        (
            lambda: {"duration": pa.array([Decimal("-0.10")], type=pa.decimal128(8, 2))},
            r"holds Decimal\('-0.10'\)",
        ),
        (lambda: {"duration": ["0.1"]}, r"duration column 'duration' holds '0.1'"),
        (lambda: {"duration": [True]}, r"duration column 'duration' holds True"),
    ],
    ids=["null_audio", "int_audio", "null_text", "negative", "nan", "negative_decimal", "string", "bool"],
)
def test_parquet_malformed_row_always_raises(tmp_path, indexed, fault_tolerant, make_columns, match):
    # Row-level problems are not audio-decoding failures, so fault-tolerant audio loading never skips them.
    path = tmp_path / "malformed.parquet"
    buf = io.BytesIO()
    sf.write(buf, np.zeros(1600, dtype=np.float32), 16000, format='WAV')
    table = {"audio": [buf.getvalue()], "text": ["x"], "duration": [0.1]}
    table.update(make_columns())
    pq.write_table(pa.table(table), path)
    it = _parquet_iterator(str(path), indexed=indexed, fault_tolerant_audio_loading=fault_tolerant)
    with pytest.raises(ParquetRowError, match=r"row 0 .*" + match):
        list(it)


@pytest.mark.unit
@pytest.mark.skipif(not HAVE_PARQUET, reason="PyArrow required for this test")
@pytest.mark.parametrize("indexed", [False, True])
@pytest.mark.parametrize("fault_tolerant", [False, True])
@pytest.mark.parametrize(
    "make_duration",
    [
        lambda: pa.array([Decimal("0.10")], type=pa.decimal128(8, 2)),
        lambda: pa.array([0], type=pa.int64()),
        lambda: pa.array([0.1], type=pa.float64()),
        lambda: pa.array([0.0], type=pa.float32()),
        lambda: None,
    ],
    ids=["decimal", "int_zero", "float", "float32_zero", "omitted"],
)
def test_parquet_valid_duration_values_are_accepted(tmp_path, indexed, fault_tolerant, make_duration):
    # The cut duration always comes from the decoded audio; the duration column is only validated metadata.
    duration = make_duration()
    path = tmp_path / "valid_duration.parquet"
    buf = io.BytesIO()
    sf.write(buf, np.zeros(1600, dtype=np.float32), 16000, format='WAV')
    table = {"audio": [buf.getvalue()], "text": ["x"]}
    if duration is not None:
        table["duration"] = duration
    pq.write_table(pa.table(table), path)
    it = _parquet_iterator(str(path), indexed=indexed, fault_tolerant_audio_loading=fault_tolerant)
    cuts = list(it) if not indexed else [it[0]]
    assert len(cuts) == 1
    cut = cuts[0]
    assert cut.duration == pytest.approx(0.1)
    assert cut.supervisions[0].duration == pytest.approx(0.1)
    assert cut.load_audio().shape == (1, 1600)
    if duration is not None:
        assert cut.custom["duration"] == duration[0].as_py()


@pytest.mark.unit
@pytest.mark.skipif(not HAVE_PARQUET, reason="PyArrow required for this test")
@pytest.mark.parametrize("indexed", [False, True])
@pytest.mark.parametrize("policy", [False, True, None])
@pytest.mark.parametrize("malformed", [False, True])
def test_parquet_loader_audio_policy(tmp_path, indexed, policy, malformed, monkeypatch):
    # Covers the real dataloader path: config defaults, sampling, and two-item audio collation.
    from lhotse.dataset.collation import collate_audio

    from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config, nemo_adapters

    class AudioDataset:
        def __getitem__(self, cuts):
            audio, lengths = collate_audio(cuts)
            return {
                "audio": audio,
                "lengths": lengths,
                "ids": [c.id for c in cuts],
                "texts": [c.supervisions[0].text for c in cuts],
            }

    buf = io.BytesIO()
    sf.write(buf, np.zeros(1600, dtype=np.float32), 16000, format="WAV")
    middle_audio = None if malformed else b"not audio"
    path = tmp_path / "loader_policy.parquet"
    pq.write_table(
        pa.table(
            {
                "id": ["good0", "corrupt", "good1"],
                "audio": pa.array([buf.getvalue(), middle_audio, buf.getvalue()], type=pa.binary()),
                "text": ["first", "bad", "second"],
                "duration": [0.1, 0.1, 0.1],
            }
        ),
        path,
    )
    warnings = []
    monkeypatch.setattr(nemo_adapters.logging, "warning", lambda msg, *a, **kw: warnings.append(msg))
    config = {
        "input_cfg": [{"type": "parquet", "manifest_filepath": str(path)}],
        "indexed": indexed,
        "batch_size": 2,
        "num_workers": 0,
        "force_finite": True,
        "shuffle": False,
        "use_bucketing": False,
        "sample_rate": 16000,
        "seed": 0,
        "shard_seed": 0,
        "cuda_expandable_segments": False,
    }
    if policy is not None:
        config["fault_tolerant_audio_loading"] = policy
    loader = get_lhotse_dataloader_from_config(
        OmegaConf.create(config), global_rank=0, world_size=1, dataset=AudioDataset()
    )
    if malformed or policy is False:
        error = nemo_adapters.ParquetRowError if malformed else nemo_adapters.ParquetAudioDecodeError
        reason = "audio column 'audio' holds NoneType" if malformed else "failed to decode audio bytes"
        with pytest.raises(error, match=r"row 1 \(id='corrupt'\).*loader_policy\.parquet.*" + reason):
            list(loader)
    else:
        batches = list(loader)
        assert len(batches) == 1
        batch = batches[0]
        assert batch["ids"] == ["good0", "good1"]
        assert batch["texts"] == ["first", "second"]
        assert tuple(batch["audio"].shape) == (2, 1600)
        assert batch["lengths"].tolist() == [1600, 1600]
        assert any("Skipped 1 of 3 rows" in w for w in warnings), warnings


@pytest.mark.unit
@pytest.mark.skipif(not HAVE_PARQUET, reason="PyArrow required for this test")
@pytest.mark.parametrize("indexed", [False, True])
@pytest.mark.parametrize("field", ["audio", "text"])
def test_parquet_missing_column_raises_even_when_fault_tolerant(parquet_with_corrupt_row, indexed, field):
    with pytest.raises(ParquetRowError, match=f"has no {field} column '{field}_col'"):
        list(
            _parquet_iterator(
                parquet_with_corrupt_row,
                **{f"{field}_field": f"{field}_col"},
                indexed=indexed,
                fault_tolerant_audio_loading=True,
            )
        )


@pytest.mark.unit
@pytest.mark.skipif(not HAVE_PARQUET, reason="PyArrow required for this test")
@pytest.mark.parametrize("field", ["audio", "text"])
def test_parquet_missing_column_raises_on_every_metadata_query(parquet_with_corrupt_row, field):
    # A failed schema check must not leave cached metadata behind that lets later queries bypass it.
    it = _parquet_iterator(parquet_with_corrupt_row, **{f"{field}_field": f"{field}_col"}, indexed=False)
    for _ in range(2):
        with pytest.raises(ParquetRowError, match=f"has no {field} column '{field}_col'"):
            len(it)


@pytest.mark.unit
@pytest.mark.skipif(not HAVE_PARQUET, reason="PyArrow required for this test")
def test_parquet_len_counts_physical_rows(tmp_path):
    buf = io.BytesIO()
    sf.write(buf, np.zeros(160, dtype=np.float32), 16000, format='WAV')
    path = tmp_path / "rg.parquet"
    pq.write_table(pa.table({"audio": [buf.getvalue()] * 5, "text": ["x"] * 5}), path, row_group_size=2)
    it = _parquet_iterator(str(path), indexed=False)
    assert len(it) == 5
    assert len(it) == 5
    empty = tmp_path / "empty.parquet"
    pq.write_table(pa.table({"audio": pa.array([], type=pa.binary()), "text": pa.array([], type=pa.string())}), empty)
    assert len(_parquet_iterator(str(empty), indexed=False)) == 0


@pytest.mark.unit
@pytest.mark.skipif(not HAVE_PARQUET, reason="PyArrow required for this test")
@pytest.mark.parametrize("indexed", [False, True])
def test_read_parquet_manifest_respects_fault_tolerant_audio_loading(parquet_with_corrupt_row, indexed):
    def make_config(fault_tolerant):
        return OmegaConf.create(
            {
                "input_cfg": [{"type": "parquet", "manifest_filepath": parquet_with_corrupt_row}],
                "indexed": indexed,
                "force_finite": True,
                "fault_tolerant_audio_loading": fault_tolerant,
            }
        )

    cuts, _ = read_cutset_from_config(make_config(False))
    with pytest.raises(RuntimeError, match=r"row 1 \(id='corrupt'\)"):
        list(cuts)

    cuts, _ = read_cutset_from_config(make_config(True))
    assert [c.id for c in cuts] == ["good"]


@pytest.mark.unit
@pytest.mark.skipif(not HAVE_PARQUET, reason="PyArrow required for this test")
def test_parquet_fallback_ids_are_global_and_match_across_modes(tmp_path, monkeypatch):
    # Rows without an ``id`` column get IDs from their position in the file. Streaming reads the file in
    # record batches; IDs must not restart in each batch and must match the IDs from indexed mode.
    buf = io.BytesIO()
    sf.write(buf, np.zeros(160, dtype=np.float32), 16000, format='WAV')
    path = tmp_path / "noid.parquet"
    pq.write_table(pa.table({"audio": [buf.getvalue()] * 5, "text": [str(i) for i in range(5)]}), path)

    original_iter_batches = pq.ParquetFile.iter_batches

    def iter_small_batches(self, *args, **kwargs):
        return original_iter_batches(self, *args, **{**kwargs, "batch_size": 2})

    monkeypatch.setattr(pq.ParquetFile, "iter_batches", iter_small_batches)

    expected = [f"noid_{i}" for i in range(5)]
    assert [c.id for c in _parquet_iterator(str(path), indexed=False)] == expected
    assert [c.id for c in _parquet_iterator(str(path), indexed=True)] == expected


@pytest.mark.unit
@pytest.mark.skipif(not HAVE_PARQUET, reason="PyArrow required for this test")
@pytest.mark.parametrize("indexed", [False, True])
def test_parquet_fallback_ids_reach_collation(tmp_path, monkeypatch, indexed):
    # Same as above, but through the real dataloader: sampling plus audio collation of two-item batches.
    from lhotse.dataset.collation import collate_audio

    from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config

    class CollatingDataset:
        def __getitem__(self, cuts):
            audio, lengths = collate_audio(cuts)
            return audio, lengths, [c.id for c in cuts]

    buf = io.BytesIO()
    sf.write(buf, np.zeros(160, dtype=np.float32), 16000, format="WAV")
    path = tmp_path / "noid.parquet"
    pq.write_table(pa.table({"audio": [buf.getvalue()] * 5, "text": [str(i) for i in range(5)]}), path)

    original_iter_batches = pq.ParquetFile.iter_batches

    def iter_small_batches(self, *args, **kwargs):
        return original_iter_batches(self, *args, **{**kwargs, "batch_size": 2})

    monkeypatch.setattr(pq.ParquetFile, "iter_batches", iter_small_batches)

    loader = get_lhotse_dataloader_from_config(
        OmegaConf.create(
            {
                "input_cfg": [{"type": "parquet", "manifest_filepath": str(path)}],
                "indexed": indexed,
                "force_finite": True,
                "batch_size": 2,
                "sample_rate": 16000,
                "num_workers": 0,
                "use_bucketing": False,
                "shuffle": False,
                "seed": 0,
                "shard_seed": 0,
            }
        ),
        global_rank=0,
        world_size=1,
        dataset=CollatingDataset(),
    )
    batches = list(loader)
    assert [cut_id for _, _, ids in batches for cut_id in ids] == [f"noid_{i}" for i in range(5)]
    assert [n for _, lengths, _ in batches for n in lengths.tolist()] == [160] * 5
    assert [tuple(audio.shape) for audio, _, _ in batches] == [(2, 160), (2, 160), (1, 160)]


def _parquet_iterator(path, **kwargs):
    from nemo.collections.common.data.lhotse.nemo_adapters import LazyParquetIterator

    return LazyParquetIterator(path=path, **kwargs)
