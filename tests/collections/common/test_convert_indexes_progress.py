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

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from scripts.dataloading import convert_indexes_to_idxpack as converter


def test_conversion_reports_while_sidecar_preflight_is_blocked(tmp_path, monkeypatch, caplog):
    from click.testing import CliRunner

    config = tmp_path / "input.yaml"
    config.write_text("[]\n")
    reported = threading.Event()

    class Handler(logging.Handler):
        def emit(self, record):
            if "Checking source indexes: running" in record.getMessage():
                reported.set()

    def preflight(*args, **kwargs):
        assert reported.wait(2), "No progress report during blocked preflight"
        raise RuntimeError("preflight sentinel")

    monkeypatch.setattr(converter, "_PROGRESS_INTERVAL", 0.02, raising=False)
    monkeypatch.setattr(converter, "_preflight_native_tar_sidecars", preflight)
    handler = Handler()
    logging.getLogger().addHandler(handler)
    try:
        with caplog.at_level(logging.INFO):
            result = CliRunner().invoke(converter.main, [str(config), "--output", str(tmp_path / "out.idxpack")])
    finally:
        logging.getLogger().removeHandler(handler)
    assert str(result.exception) == "preflight sentinel"
    assert "Checking source indexes: failed" in caplog.text
    assert not any(t.name == "idxpack-progress" for t in threading.enumerate())


@pytest.mark.parametrize("workers", [1, 2])
def test_routing_reports_actual_completions_and_waits(tmp_path, monkeypatch, caplog, workers):
    reported = threading.Event()

    class Handler(logging.Handler):
        def emit(self, record):
            message = record.getMessage()
            if "Building native-tar routes: running" in message and "completed=1/2" in message:
                reported.set()

    def route(task):
        map_index, shard_index, *_ = task
        if shard_index == 1:
            assert reported.wait(2), "No progress report while waiting for remaining shard"
        return (
            map_index,
            shard_index,
            SimpleNamespace(
                records_checked=1, skip_marker_records=0, top_level_skip_marker_records=0, custom_skip_marker_records=0
            ),
        )

    snapshot = SimpleNamespace(validate=lambda: None)
    monkeypatch.setattr(converter.NativeTarOrdinalMapInputSnapshot, "capture", lambda *a: snapshot)
    monkeypatch.setattr(converter, "_resolve_local_sidecar", lambda p, r: tmp_path / "unused.idx")
    monkeypatch.setattr(converter, "_write_native_tar_ordinal_map_shard_task", route)
    monkeypatch.setattr(converter, "ProcessPoolExecutor", lambda max_workers, **k: ThreadPoolExecutor(max_workers))
    monkeypatch.setattr(converter, "_PROGRESS_INTERVAL", 0.02, raising=False)
    spec = converter.NativeTarOrdinalMapSpec("manifest", "tar", ("m0", "m1"), ("t0", "t1"))
    handler = Handler()
    logging.getLogger().addHandler(handler)
    try:
        with caplog.at_level(logging.INFO):
            _, summary, _, _ = converter._build_native_tar_ordinal_array_specs(
                [spec],
                tmp_path,
                indexes_root=tmp_path,
                index_path_overrides={},
                source_size_overrides={},
                native_tar_route_workers=workers,
            )
    finally:
        logging.getLogger().removeHandler(handler)
    assert summary.records_checked == 2
    assert "completed=2/2" in caplog.text
    assert not any(t.name == "idxpack-progress" for t in threading.enumerate())
