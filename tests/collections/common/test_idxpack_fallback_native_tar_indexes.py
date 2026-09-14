import io
import tarfile

from lhotse.index_pack import IndexPackCollectionSpec
from lhotse.indexing import index_file_path

from nemo.collections.common.data.lhotse.indexed_adapters import create_tar_index
from scripts.dataloading.convert_indexes_to_idxpack import (
    _preflight_native_tar_sidecars,
)


def test_native_tar_preflight_uses_read_only_fallback_when_primary_is_missing(tmp_path):
    tar_path = tmp_path / "audio.tar"
    with tarfile.open(tar_path, "w") as archive:
        payload = b"audio"
        info = tarfile.TarInfo("sample.wav")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))

    primary_root = tmp_path / "primary"
    fallback_root = tmp_path / "fallback"
    fallback_idx = index_file_path(tar_path, fallback_root)
    fallback_idx.parent.mkdir(parents=True)
    create_tar_index(tar_path, fallback_idx)
    primary_idx = index_file_path(tar_path, primary_root)

    collection = IndexPackCollectionSpec(
        role="tar",
        kind="nemo_tar",
        source_spec=str(tar_path),
        paths=(str(tar_path),),
    )
    source_sizes, overrides = _preflight_native_tar_sidecars(
        [collection],
        primary_root,
        fallback_indexes_root=fallback_root,
    )

    assert source_sizes == {}
    assert overrides == {str(tar_path): fallback_idx}
    assert not primary_idx.exists()
