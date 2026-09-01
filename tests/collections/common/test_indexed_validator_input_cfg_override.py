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

import pytest
import yaml
from click.testing import CliRunner
from omegaconf import OmegaConf
from scripts.dataloading import validate_dataloader


@pytest.mark.parametrize("as_list", [False, True])
def test_input_cfg_override_preserves_base_recipe_and_resolves_interpolation(
    tmp_path, monkeypatch, as_list
):
    base_config = tmp_path / "base-recipe.yaml"
    base_config.write_text(
        yaml.safe_dump(
            {
                "data_blend_dir": "/stale/blend/root",
                "model": {"retained_model_setting": "unchanged"},
                "data": {
                    "train_ds": {
                        "input_cfg": "${data_blend_dir}/original.yaml",
                        "batch_duration": 90,
                        "num_workers": 7,
                    }
                },
            }
        )
    )
    leaf = {
        "type": "share_gpt",
        "manifest_filepath": "${data_blend_dir}/generated.jsonl",
    }
    override = tmp_path / "generated-leaf.yaml"
    override.write_text(yaml.safe_dump([leaf] if as_list else leaf))

    captured = {}

    def build_dataset(full_cfg, tokenizer, **kwargs):
        captured["model"] = OmegaConf.to_container(full_cfg.model, resolve=True)
        return object()

    monkeypatch.setattr(validate_dataloader, "_build_validation_dataset", build_dataset)

    from nemo.collections.common.data.lhotse import dataloader as dataloader_module

    def build_dataloader(*, config, **kwargs):
        captured["section"] = OmegaConf.to_container(config, resolve=True)
        return []

    monkeypatch.setattr(
        dataloader_module,
        "get_lhotse_dataloader_from_config",
        build_dataloader,
    )
    resolved_blend_dir = tmp_path / "resolved-blends"
    result = CliRunner().invoke(
        validate_dataloader.cli,
        [
            "--config",
            str(base_config),
            "--data-blend-dir",
            str(resolved_blend_dir),
            "--input-cfg",
            str(override),
            "--output-dir",
            str(tmp_path / "out"),
            "--phase",
            "baseline",
            "--steps",
            "0",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["model"] == {"retained_model_setting": "unchanged"}
    assert captured["section"]["batch_duration"] == 90
    assert captured["section"]["num_workers"] == 7
    expected_leaf = {
        **leaf,
        "manifest_filepath": str(resolved_blend_dir / "generated.jsonl"),
        "force_finite": True,
        "metadata_only": True,
    }
    assert captured["section"]["input_cfg"] == (
        [expected_leaf] if as_list else expected_leaf
    )
