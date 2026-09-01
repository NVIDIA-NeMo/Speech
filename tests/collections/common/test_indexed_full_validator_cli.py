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

import pytest
import torch
import yaml
from click.testing import CliRunner
from scripts.dataloading import validate_dataloader

from tests.collections.common.test_indexed_full_validator_stats import _batch


class _Tokenizer:
    def token_to_id(self, token):
        assert token == "<audio>"
        return 7


class _DataLoader:
    def __init__(self, batches):
        self.batches = batches

    def __iter__(self):
        return iter(self.batches)

    def state_dict(self):
        return {}


def test_validator_initializes_process_group_for_multi_rank(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr(
        torch.distributed,
        "init_process_group",
        lambda **kwargs: calls.append(kwargs),
    )

    initialized = validate_dataloader._ensure_validation_process_group(
        rank=1, world_size=2
    )

    assert initialized is True
    assert calls == [{"backend": "gloo", "rank": 1, "world_size": 2}]


def test_validator_reuses_matching_process_group(monkeypatch):
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)

    assert (
        validate_dataloader._ensure_validation_process_group(rank=1, world_size=2)
        is False
    )


def test_validator_rejects_mismatched_process_group(monkeypatch):
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)

    with pytest.raises(RuntimeError, match="process-group mismatch"):
        validate_dataloader._ensure_validation_process_group(rank=1, world_size=2)


def _invoke(tmp_path, monkeypatch, *, num_batches, requested_batches):
    config_path = tmp_path / "recipe.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {"audio_locator_tag": "<audio>"},
                "data": {
                    "train_ds": {
                        "input_cfg": [
                            {
                                "type": "share_gpt",
                                "tar_lookup_mode": "collection",
                            }
                        ]
                    }
                },
            }
        )
    )
    monkeypatch.setattr(
        validate_dataloader, "_build_tokenizer", lambda *args, **kwargs: _Tokenizer()
    )
    monkeypatch.setattr(
        validate_dataloader,
        "_build_validation_dataset",
        lambda *args, **kwargs: object(),
    )

    from nemo.collections.common.data.lhotse import dataloader as dataloader_module

    monkeypatch.setattr(
        dataloader_module,
        "get_lhotse_dataloader_from_config",
        lambda **kwargs: _DataLoader([_batch() for _ in range(num_batches)]),
    )
    output_dir = tmp_path / "out"
    result = CliRunner().invoke(
        validate_dataloader.cli,
        [
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
            "--phase",
            "baseline",
            "--steps",
            str(requested_batches),
            "--checkpoint-at",
            "-1",
            "--mode",
            "full",
            "--no-metadata-only",
        ],
    )
    summary_path = output_dir / "baseline" / "run0" / "full_summary_rank_000.json"
    return result, json.loads(summary_path.read_text())


def test_full_cli_writes_passed_summary_after_exact_completion(tmp_path, monkeypatch):
    result, summary = _invoke(tmp_path, monkeypatch, num_batches=2, requested_batches=2)

    assert result.exit_code == 0, result.output
    assert summary["status"] == "passed"
    assert summary["requested_batches"] == 2
    assert summary["completed_batches"] == 2
    assert summary["counters"]["examples"] == 6
    assert summary["audio"]["path_resolution_modes"] == {
        "status": "configured",
        "values": ["tar_collection_route"],
    }


def test_full_cli_writes_failed_summary_when_loader_exhausts_early(
    tmp_path, monkeypatch
):
    result, summary = _invoke(tmp_path, monkeypatch, num_batches=1, requested_batches=2)

    assert result.exit_code != 0
    assert "materialized 1/2" in result.output
    assert summary["status"] == "failed"
    assert summary["requested_batches"] == 2
    assert summary["completed_batches"] == 1
    assert summary["failures"] == [
        {"step": 1, "stage": "materialize_or_measure", "error_type": "ClickException"}
    ]
