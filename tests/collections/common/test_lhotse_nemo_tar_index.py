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
import io
import json
import struct
import tarfile

import numpy as np
import pytest
import soundfile as sf
from click.testing import CliRunner
from lhotse.indexing import create_jsonl_index
from lhotse.indexing import create_tar_index as create_lhotse_tar_index
from lhotse.indexing import read_index
from scripts.dataloading import build_indexes

from nemo.collections.common.data.lhotse import indexed_adapters
from nemo.collections.common.data.lhotse.indexed_adapters import (
    IndexedTarMemberReader,
    create_tar_index,
    validate_tar_member_index_layout,
)
from nemo.collections.common.data.lhotse.nemo_adapters import LazyNeMoTarredIterator


def test_nemo_tar_index_sentinel_includes_trailing_record_padding(tmp_path):
    tar_path = tmp_path / "data.tar"
    with tarfile.open(tar_path, "w") as archive:
        payload = b"sample"
        info = tarfile.TarInfo("sample.json")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    with tar_path.open("ab") as stream:
        stream.write(b"\0" * 10240)

    idx_path = tmp_path / "data.tar.idx"
    create_tar_index(tar_path, idx_path)

    with idx_path.open("rb") as stream:
        stream.seek(-8, io.SEEK_END)
        (sentinel,) = struct.unpack("<Q", stream.read(8))
    assert sentinel == tar_path.stat().st_size

    # The .idx remains exactly the raw uint64 layout. This is the same
    # operation performed by older Lhotse readers.
    offsets = read_index(idx_path)
    assert offsets.tolist() == [0, tar_path.stat().st_size]
    assert idx_path.read_bytes() == struct.pack("<QQ", 0, tar_path.stat().st_size)
    assert not (tmp_path / "data.tar.idx.meta").exists()

    reader = IndexedTarMemberReader(tar_path, idx_path, auto_create_index=False)
    assert reader[0] == ("sample.json", b"sample")
    reader.close()


def test_nemo_tar_index_and_reader_use_s3_local_mirror(tmp_path, monkeypatch):
    mirror_root = tmp_path / "mirror"
    tar_path = mirror_root / "bucket" / "nested" / "data.tar"
    tar_path.parent.mkdir(parents=True)
    with tarfile.open(tar_path, "w") as archive:
        payload = b"sample"
        info = tarfile.TarInfo("sample.json")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))

    monkeypatch.setenv("LHOTSE_S3_LOCAL_MIRROR_ROOTS", str(mirror_root))

    def fail_ais(*args, **kwargs):
        raise AssertionError("AIS must not be opened when an S3 source has a local mirror")

    monkeypatch.setattr("lhotse.ais.AISRangeReader", fail_ais)
    remote_tar = "s3://bucket/nested/data.tar"
    idx_path = tmp_path / "data.tar.idx"

    create_tar_index(remote_tar, idx_path)
    assert read_index(idx_path).tolist() == [0, tar_path.stat().st_size]
    assert build_indexes._source_size(remote_tar) == tar_path.stat().st_size

    reader = IndexedTarMemberReader(remote_tar, idx_path, auto_create_index=False)
    assert reader[0] == ("sample.json", b"sample")
    reader.close()


@pytest.mark.parametrize("scheme", ["https", "s3", "ais"])
def test_nemo_tar_index_uses_streaming_opener(tmp_path, monkeypatch, scheme):
    tar_path = tmp_path / "data.tar"
    with tarfile.open(tar_path, "w") as archive:
        payload = b"sample"
        info = tarfile.TarInfo("sample.json")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    raw = tar_path.read_bytes()
    opened = []

    def open_http(path, mode):
        opened.append((path, mode))
        return io.BytesIO(raw)

    def fail_ais(*args, **kwargs):
        raise AssertionError("Sequential tar scans must not use AISRangeReader")

    monkeypatch.setattr("lhotse.serialization.open_best", open_http)
    monkeypatch.setattr("lhotse.ais.AISRangeReader", fail_ais)
    idx_path = tmp_path / "http.idx"

    create_tar_index(f"{scheme}://example.test/data.tar", idx_path)

    assert opened == [(f"{scheme}://example.test/data.tar", "rb")]
    assert read_index(idx_path).tolist() == [0, len(raw)]


def test_nemo_tar_index_seekable_matches_streaming_and_skips_payloads(tmp_path, monkeypatch):
    tar_path = tmp_path / "data.tar"
    with tarfile.open(tar_path, "w") as archive:
        for name, payload in (
            ("one.json", b"{}"),
            ("one.wav", b"x" * (2 * 1024 * 1024)),
            ("nested/two.json", b'{"text": "two"}'),
            ("nested/two.flac", b"audio-two"),
        ):
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    with tar_path.open("ab") as stream:
        stream.write(b"\0" * 10240)

    class ReadCountingBytesIO(io.BytesIO):
        def __init__(self, data):
            super().__init__(data)
            self.bytes_read = 0

        def read(self, size=-1):
            data = super().read(size)
            self.bytes_read += len(data)
            return data

    raw_tar = tar_path.read_bytes()
    local_source = ReadCountingBytesIO(raw_tar)
    original_open = indexed_adapters._open_data_path

    def open_source(path):
        return local_source if path == str(tar_path) else original_open(path)

    monkeypatch.setattr(indexed_adapters, "_open_data_path", open_source)
    local_idx = tmp_path / "local.idx"
    create_tar_index(tar_path, local_idx)
    assert local_source.bytes_read < 64 * 1024
    assert local_source.bytes_read < len(raw_tar) // 16

    monkeypatch.setattr("lhotse.serialization.open_best", lambda path, mode: io.BytesIO(raw_tar))
    remote_idx = tmp_path / "remote.idx"
    create_tar_index("ais://bucket/data.tar", remote_idx)

    assert remote_idx.read_bytes() == local_idx.read_bytes()
    offsets = read_index(local_idx).tolist()
    assert len(offsets) == 3
    assert offsets[0] == 0 and offsets[-1] == len(raw_tar)


def test_nemo_tar_index_dense_local_tar_selects_streaming(tmp_path, monkeypatch):
    tar_path = tmp_path / "dense.tar"
    with tarfile.open(tar_path, "w") as archive:
        for idx in range(indexed_adapters._TAR_INDEX_PROBE_REGULAR_MEMBERS):
            payload = bytes([idx % 256]) * (64 * 1024)
            info = tarfile.TarInfo(f"sample-{idx}.wav")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))

    with tar_path.open("rb") as stream:
        assert indexed_adapters._select_local_tar_index_mode(stream) == "r|"

    local_idx = tmp_path / "dense-local.idx"
    create_tar_index(tar_path, local_idx)

    raw_tar = tar_path.read_bytes()
    monkeypatch.setattr("lhotse.serialization.open_best", lambda path, mode: io.BytesIO(raw_tar))
    streaming_idx = tmp_path / "dense-streaming.idx"
    create_tar_index("ais://bucket/dense.tar", streaming_idx)
    assert local_idx.read_bytes() == streaming_idx.read_bytes()


def test_nemo_tar_index_sparse_local_tar_selects_seekable(tmp_path):
    tar_path = tmp_path / "sparse.tar"
    payload_size = 4 * 1024 * 1024
    with tar_path.open("wb") as stream:
        for idx in range(indexed_adapters._TAR_INDEX_STREAMING_MIN_REGULAR_MEMBERS):
            info = tarfile.TarInfo(f"sample-{idx}.wav")
            info.size = payload_size
            stream.write(info.tobuf())
            stream.seek(payload_size, io.SEEK_CUR)
        stream.write(b"\0" * (2 * tarfile.BLOCKSIZE))

    with tar_path.open("rb") as stream:
        assert indexed_adapters._select_local_tar_index_mode(stream) == "r:"

    idx_path = tmp_path / "sparse.idx"
    create_tar_index(tar_path, idx_path)
    offsets = read_index(idx_path).tolist()
    assert len(offsets) == indexed_adapters._TAR_INDEX_STREAMING_MIN_REGULAR_MEMBERS + 1
    assert offsets[-1] == tar_path.stat().st_size


def test_nemo_tar_index_seekable_and_streaming_reject_truncated_member(tmp_path, monkeypatch):
    complete_path = tmp_path / "complete.tar"
    with tarfile.open(complete_path, "w") as archive:
        payload = b"x" * 4096
        info = tarfile.TarInfo("sample.json")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    truncated = complete_path.read_bytes()[:1024]

    truncated_path = tmp_path / "truncated.tar"
    truncated_path.write_bytes(truncated)
    with pytest.raises(tarfile.ReadError, match="unexpected end of data"):
        create_tar_index(truncated_path, tmp_path / "local-truncated.idx")

    monkeypatch.setattr("lhotse.serialization.open_best", lambda path, mode: io.BytesIO(truncated))
    with pytest.raises(tarfile.ReadError, match="unexpected end of data"):
        create_tar_index("ais://bucket/truncated.tar", tmp_path / "remote-truncated.idx")


@pytest.mark.unit
def test_indexed_nemo_tarred_rejects_lhotse_paired_tar_index_at_construction(tmp_path):
    manifest_path, tar_path = _write_nemo_tarred_shard(tmp_path)
    # ``lhotse index tar`` writes the Shar paired layout: 2 entries for this 4-member tar.
    create_lhotse_tar_index(tar_path)
    assert len(read_index(f"{tar_path}.idx")) == 3

    with pytest.raises(ValueError, match=r"has 4 rows.*lists only 2 members.*wrong layout.*build_indexes\.py"):
        LazyNeMoTarredIterator(manifest_path=str(manifest_path), tar_paths=str(tar_path), indexed=True)


@pytest.mark.unit
def test_indexed_nemo_tarred_accepts_nemo_tar_index(tmp_path):
    manifest_path, tar_path = _write_nemo_tarred_shard(tmp_path)
    create_tar_index(tar_path, f"{tar_path}.idx")

    iterator = LazyNeMoTarredIterator(manifest_path=str(manifest_path), tar_paths=str(tar_path), indexed=True)

    assert [cut.recording.id for cut in iterator] == [f"utt{i}.wav" for i in range(4)]


@pytest.mark.unit
def test_validate_tar_member_index_layout(tmp_path):
    _, tar_path = _write_nemo_tarred_shard(tmp_path)
    nemo_idx = tmp_path / "nemo.idx"
    lhotse_idx = tmp_path / "lhotse.idx"
    create_tar_index(tar_path, nemo_idx)
    create_lhotse_tar_index(tar_path, lhotse_idx)

    validate_tar_member_index_layout(tar_path, nemo_idx)
    with pytest.raises(ValueError, match="paired-member layout"):
        validate_tar_member_index_layout(tar_path, lhotse_idx)


@pytest.mark.unit
def test_build_indexes_rebuilds_nemo_tar_index_in_lhotse_paired_layout(tmp_path):
    manifest_path, tar_path = _write_nemo_tarred_shard(tmp_path)
    create_lhotse_tar_index(tar_path)
    cfg_path = tmp_path / "input_cfg.yaml"
    entry = {"type": "nemo_tarred", "manifest_filepath": str(manifest_path), "tarred_audio_filepaths": str(tar_path)}
    cfg_path.write_text(json.dumps([entry]))
    job = build_indexes.IndexJob(str(tar_path), build_indexes.NEMO_TAR)
    assert not build_indexes._is_indexed(job)

    result = CliRunner().invoke(build_indexes.main, [str(cfg_path), "--executor", "thread"])

    assert result.exit_code == 0, result.output
    assert build_indexes._is_indexed(job)
    assert len(read_index(f"{tar_path}.idx")) == 5


@pytest.mark.unit
def test_indexed_nemo_tarred_accepts_skipped_rows_past_tar_index(tmp_path):
    audio = np.linspace(-0.5, 0.5, 1600, dtype=np.float32)
    tar_path = _write_tar(tmp_path / "audio_0.tar", [("A.wav", _wav_bytes(audio))])
    manifest_path = _write_manifest(
        tmp_path / "manifest_0.json",
        [_row("A.wav"), _row("B.wav", _skipme=True), _row("C.wav", custom={"_skipme": True})],
    )
    create_tar_index(tar_path, f"{tar_path}.idx")

    iterator = LazyNeMoTarredIterator(manifest_path=str(manifest_path), tar_paths=str(tar_path), indexed=True)
    cuts = list(iterator)

    assert len(iterator) == 3
    assert [cut.id for cut in cuts] == ["A.wav"]
    np.testing.assert_allclose(cuts[0].load_audio()[0], audio, atol=1e-4)


@pytest.mark.unit
def test_indexed_nemo_tarred_accepts_empty_tar_with_only_skipped_rows(tmp_path):
    tar_path = _write_tar(tmp_path / "audio_0.tar", [])
    manifest_path = _write_manifest(
        tmp_path / "manifest_0.json", [_row("A.wav", _skipme=True), _row("B.wav", custom={"_skipme": 1})]
    )
    create_tar_index(tar_path, f"{tar_path}.idx")

    iterator = LazyNeMoTarredIterator(manifest_path=str(manifest_path), tar_paths=str(tar_path), indexed=True)

    assert len(iterator) == 2
    assert list(iterator) == []


@pytest.mark.unit
@pytest.mark.parametrize("case", ["marker-then-active", "falsy-marker"])
def test_indexed_nemo_tarred_rejects_active_rows_past_tar_index(tmp_path, case):
    if case == "marker-then-active":
        # Rows are not renumbered: the active row sits at physical index 1 of a one-member tar.
        rows = [_row("B.wav", _skipme=True), _row("A.wav")]
    else:
        rows = [_row("A.wav"), _row("B.wav", _skipme=False)]
    tar_path = _write_tar(tmp_path / "audio_0.tar", [("A.wav", b"audio")])
    manifest_path = _write_manifest(tmp_path / "manifest_0.json", rows)
    create_tar_index(tar_path, f"{tar_path}.idx")

    with pytest.raises(ValueError, match=r"has 2 rows.*lists only 1 members.*not skipped"):
        LazyNeMoTarredIterator(manifest_path=str(manifest_path), tar_paths=str(tar_path), indexed=True)


@pytest.mark.unit
@pytest.mark.parametrize(
    "entries",
    [
        [("utt0.wav", b"0"), ("empty/", None), ("utt1.wav", b"1")],
        [("utt0.wav", b"0"), ("empty/", None)],
        [("utt0.wav", b"0"), ("empty/", {"comment": "pax metadata on a directory"}), ("utt1.wav", b"1")],
        [("sample.json", b"{}"), ("sample.wav", b"0"), ("other.json", b"{}"), ("other.wav", b"1")],
        [("d/" + "x" * 150 + ".wav", b"0"), ("d/" + "y" * 150 + ".wav", b"1")],
        [],
    ],
    ids=["directory-between", "trailing-directory", "pax-directory", "same-stem-group", "long-names", "empty"],
)
@pytest.mark.parametrize("tar_format", [tarfile.PAX_FORMAT, tarfile.GNU_FORMAT], ids=["pax", "gnu"])
def test_validate_tar_member_index_layout_accepts_nemo_index(tmp_path, entries, tar_format):
    tar_path = _write_tar(tmp_path / "audio_0.tar", entries, tar_format=tar_format)
    idx_path = tmp_path / "audio_0.tar.idx"
    create_tar_index(tar_path, idx_path)

    validate_tar_member_index_layout(tar_path, idx_path)

    assert build_indexes._is_indexed(build_indexes.IndexJob(str(tar_path), build_indexes.NEMO_TAR))


@pytest.mark.unit
def test_build_indexes_keeps_valid_nemo_tar_index_with_directory_entries(tmp_path):
    tar_path = _write_tar(tmp_path / "audio_0.tar", [("utt0.wav", b"0"), ("empty/", None), ("utt1.wav", b"1")])
    manifest_path = _write_manifest(tmp_path / "manifest_0.json", [_row("utt0.wav"), _row("utt1.wav")])
    create_tar_index(tar_path, f"{tar_path}.idx")
    idx_mtime = (tmp_path / "audio_0.tar.idx").stat().st_mtime_ns
    cfg_path = tmp_path / "input_cfg.yaml"
    entry = {"type": "nemo_tarred", "manifest_filepath": str(manifest_path), "tarred_audio_filepaths": str(tar_path)}
    cfg_path.write_text(json.dumps([entry]))

    result = CliRunner().invoke(build_indexes.main, [str(cfg_path), "--executor", "thread"])

    assert result.exit_code == 0, result.output
    assert (tmp_path / "audio_0.tar.idx").stat().st_mtime_ns == idx_mtime


@pytest.mark.unit
def test_validate_tar_member_index_layout_rejects_metadata_past_indexed_range(tmp_path):
    tar_path = _write_tar(tmp_path / "audio_0.tar", [("d/" + "x" * 150 + ".wav", b"0")], tar_format=tarfile.PAX_FORMAT)
    idx_path = tmp_path / "bad.idx"
    # The first range ends right after the PAX header block, inside the PAX payload.
    idx_path.write_bytes(struct.pack("<QQQ", 0, 512, tar_path.stat().st_size))

    with pytest.raises(ValueError, match="extends past indexed range"):
        validate_tar_member_index_layout(tar_path, idx_path)


def _write_nemo_tarred_shard(tmp_path, num_samples=4):
    """Write manifest_0.json + audio_0.tar with one tar member per manifest row (NeMo tarred layout)."""
    manifest_path = tmp_path / "manifest_0.json"
    tar_path = tmp_path / "audio_0.tar"
    rows = []
    with tarfile.open(tar_path, "w") as archive:
        for i in range(num_samples):
            payload = f"audio-{i}".encode()
            info = tarfile.TarInfo(f"utt{i}.wav")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
            rows.append({"audio_filepath": f"utt{i}.wav", "duration": 1.0, "text": f"t{i}", "sampling_rate": 16000})
    manifest_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    create_jsonl_index(manifest_path)
    return manifest_path, tar_path


def _row(audio_filepath, **extra):
    return {"audio_filepath": audio_filepath, "duration": 0.1, "text": audio_filepath, "sampling_rate": 16000, **extra}


def _write_manifest(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    create_jsonl_index(path)
    return path


def _write_tar(path, entries, tar_format=tarfile.DEFAULT_FORMAT):
    """Write a tar from ``(name, payload)`` entries; a None or dict payload makes a directory (dict = PAX headers)."""
    with tarfile.open(path, "w", format=tar_format) as archive:
        for name, payload in entries:
            if payload is None or isinstance(payload, dict):
                info = tarfile.TarInfo(name.rstrip("/"))
                info.type = tarfile.DIRTYPE
                if isinstance(payload, dict) and tar_format == tarfile.PAX_FORMAT:
                    info.pax_headers = payload
                archive.addfile(info)
            else:
                info = tarfile.TarInfo(name)
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
    return path


def _wav_bytes(audio):
    buffer = io.BytesIO()
    sf.write(buffer, audio, 16000, format="WAV", subtype="PCM_16")
    return buffer.getvalue()
