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
import struct
import tarfile

import pytest
from lhotse.indexing import read_index

from nemo.collections.common.data.lhotse import indexed_adapters
from nemo.collections.common.data.lhotse.indexed_adapters import IndexedTarMemberReader, create_tar_index
from scripts.dataloading import build_indexes


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


def test_nemo_tar_index_uses_generic_streaming_opener_for_http(
    tmp_path, monkeypatch
):
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
        raise AssertionError("generic URLs must not be routed through AISRangeReader")

    monkeypatch.setattr("lhotse.serialization.open_best", open_http)
    monkeypatch.setattr("lhotse.ais.AISRangeReader", fail_ais)
    idx_path = tmp_path / "http.idx"

    create_tar_index("https://example.test/data.tar", idx_path)

    assert opened == [("https://example.test/data.tar", "rb")]
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

    monkeypatch.setattr("lhotse.ais.AISRangeReader", lambda path: io.BytesIO(raw_tar))
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
    monkeypatch.setattr("lhotse.ais.AISRangeReader", lambda path: io.BytesIO(raw_tar))
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

    monkeypatch.setattr("lhotse.ais.AISRangeReader", lambda path: io.BytesIO(truncated))
    with pytest.raises(tarfile.ReadError, match="unexpected end of data"):
        create_tar_index("ais://bucket/truncated.tar", tmp_path / "remote-truncated.idx")
