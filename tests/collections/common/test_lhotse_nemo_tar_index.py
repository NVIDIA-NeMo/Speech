# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

from lhotse.indexing import read_index

from nemo.collections.common.data.lhotse.indexed_adapters import IndexedTarMemberReader, create_tar_index


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
