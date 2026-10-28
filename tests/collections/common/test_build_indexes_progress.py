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

import pytest
import yaml
from click.testing import CliRunner
from scripts.dataloading import build_indexes


def test_build_indexes_reports_progress_while_a_file_is_blocked(tmp_path, monkeypatch, caplog):
    import threading

    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("{}\n")
    config = tmp_path / "input.yaml"
    config.write_text(yaml.safe_dump({"type": "nemo", "manifest_filepath": str(manifest)}))
    reported = threading.Event()
    completed = []

    class ProgressHandler(logging.Handler):
        def emit(self, record):
            if "built 0/1" in record.getMessage():
                reported.set()

    def build(job, *, force):
        assert reported.wait(2), "No progress was logged while a file was unfinished"
        completed.append(job)
        return job, "built"

    monkeypatch.setattr(build_indexes, "_PROGRESS_INTERVAL", 0.02, raising=False)
    monkeypatch.setattr(build_indexes, "_build_one", build)
    handler = ProgressHandler()
    logger = logging.getLogger()
    logger.addHandler(handler)
    try:
        with caplog.at_level(logging.INFO):
            result = CliRunner().invoke(
                build_indexes.main, ["--force", "--executor", "thread", "--workers", "1", str(config)]
            )
    finally:
        logger.removeHandler(handler)
    assert result.exit_code == 0, result.output
    assert len(completed) == 1
    assert "built 1/1" in caplog.text


@pytest.mark.parametrize("error_type", [RuntimeError, TimeoutError])
def test_build_indexes_counts_worker_failures(tmp_path, monkeypatch, caplog, error_type):
    manifests = [tmp_path / "good.jsonl", tmp_path / "bad.jsonl"]
    for manifest in manifests:
        manifest.write_text("{}\n")
    config = tmp_path / "input.yaml"
    config.write_text(yaml.safe_dump([{"type": "nemo", "manifest_filepath": str(p)} for p in manifests]))

    def build(job, *, force):
        if job.path == str(manifests[1]):
            raise error_type("worker failure sentinel")
        return job, "built"

    monkeypatch.setattr(build_indexes, "_build_one", build)
    with caplog.at_level(logging.INFO):
        result = CliRunner().invoke(
            build_indexes.main, ["--force", "--executor", "thread", "--workers", "2", str(config)]
        )
    assert result.exit_code == 1
    assert "built 1/2  failures=1  pending=0" in caplog.text
    assert "worker failure sentinel" in caplog.text
