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

"""Coverage for indexed ShareGPT tar-route CLI integration."""

import io
import json
import tarfile
from pathlib import Path

import yaml
from click.testing import CliRunner
from lhotse.index_pack import IndexPack, index_pack_collection_key
from lhotse.indexing import create_jsonl_index
from scripts.dataloading import build_indexes, convert_indexes_to_idxpack

from nemo.collections.common.data.lhotse.indexed_adapters import create_tar_index as create_nemo_tar_index
from nemo.collections.common.data.lhotse.sharegpt_tar_routing import ShareGptTarRoutingIndex, TarRoute


def _write_sharegpt_collection(tmp_path: Path) -> tuple[Path, Path, Path]:
    manifest = tmp_path / "rows.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "sound": "sample.wav",
                "conversations": [
                    {"from": "human", "value": "<sound>"},
                    {"from": "gpt", "value": "answer"},
                ],
            }
        )
        + "\n"
    )
    tar_path = tmp_path / "audio.tar"
    with tarfile.open(tar_path, "w:") as archive:
        payload = b"audio"
        info = tarfile.TarInfo("sample.wav")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    config_path = tmp_path / "sharegpt.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "type": "share_gpt",
                "manifest_filepath": str(manifest),
                "tarred_audio_filepaths": str(tar_path),
                "tar_lookup_mode": "collection",
                "tar_routing_filepath": "rows.sgroute",
            }
        )
    )
    return manifest, tar_path, config_path


def test_build_indexes_builds_sharegpt_route_after_companion_indexes(tmp_path):
    _, _, config_path = _write_sharegpt_collection(tmp_path)
    indexes_root = tmp_path / "indexes"

    result = CliRunner().invoke(
        build_indexes.main,
        [
            "--executor",
            "thread",
            "--workers",
            "2",
            "--indexes-root",
            str(indexes_root),
            str(config_path),
        ],
    )

    assert result.exit_code == 0, result.output
    with ShareGptTarRoutingIndex(indexes_root / "rows.sgroute") as routing:
        assert routing.routes_for_row(0) == (TarRoute(0, 0),)


def test_converter_builds_and_validates_route_beside_idxpack(tmp_path):
    manifest, tar_path, config_path = _write_sharegpt_collection(tmp_path)
    create_jsonl_index(manifest)
    create_nemo_tar_index(tar_path, Path(f"{tar_path}.idx"))
    pack_dir = tmp_path / "packs"
    output = pack_dir / "sharegpt.idxpack"

    result = CliRunner().invoke(
        convert_indexes_to_idxpack.main,
        ["--output", str(output), str(config_path)],
    )

    assert result.exit_code == 0, result.output
    route_path = pack_dir / "rows.sgroute"
    with ShareGptTarRoutingIndex(route_path) as routing:
        assert routing.routes_for_row(0) == (TarRoute(0, 0),)
    with IndexPack(output) as pack:
        key = index_pack_collection_key("tar_collection", "nemo_tar", str(tar_path))
        assert pack.collection(key).offsets_required
