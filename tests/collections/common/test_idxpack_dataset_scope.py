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
import json

import yaml
from click.testing import CliRunner
from lhotse.indexing import create_jsonl_index
from omegaconf import OmegaConf
from scripts.dataloading.convert_indexes_to_idxpack import main

from nemo.collections.common.data.lhotse import nemo_adapters
from nemo.collections.common.data.lhotse.cutset import read_dataset_config


def test_explicit_outer_dataset_pack_is_propagated_to_nested_leaves(tmp_path, monkeypatch):
    manifests = []
    expected = []
    for leaf in range(2):
        manifest = tmp_path / f"leaf-{leaf}.jsonl"
        text = f"leaf-{leaf}"
        manifest.write_text(
            json.dumps(
                {
                    "audio_filepath": f"audio-{leaf}.wav",
                    "duration": 1.0,
                    "sampling_rate": 16000,
                    "text": text,
                }
            )
            + "\n"
        )
        create_jsonl_index(manifest)
        manifests.append(str(manifest))
        expected.append(text)

    outer_input_cfg = tmp_path / "outer-dataset.yaml"
    outer_input_cfg.write_text(
        yaml.safe_dump(
            [
                {
                    "type": "group",
                    "input_cfg": [
                        {
                            "type": "nemo",
                            "manifest_filepath": manifest,
                        }
                        for manifest in manifests
                    ],
                }
            ]
        )
    )
    pack = tmp_path / "outer-dataset.idxpack"
    result = CliRunner().invoke(main, ["--output", str(pack), str(outer_input_cfg)])
    assert result.exit_code == 0, result.output

    def fail_expand(*args, **kwargs):
        raise AssertionError("nested packed leaves must not expand source shard paths")

    monkeypatch.setattr(nemo_adapters, "expand_sharded_filepaths", fail_expand)
    train_cfg = OmegaConf.create(
        {
            "indexed": True,
            "metadata_only": True,
            "force_finite": True,
            "shuffle": False,
            "index_pack_root": str(tmp_path),
            "input_cfg": [
                {
                    "type": "group",
                    "input_cfg": str(outer_input_cfg),
                    "index_pack": pack.name,
                }
            ],
        }
    )
    cuts, _ = read_dataset_config(train_cfg)
    assert sorted(cut.supervisions[0].text for cut in cuts) == expected
