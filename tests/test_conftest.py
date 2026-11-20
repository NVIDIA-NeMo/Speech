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

import importlib.util
import json
import os
import subprocess
import sys
import textwrap
import urllib.error
from pathlib import Path
from unittest.mock import Mock, call, patch

import pytest
import torch

ROOT_CONFTEST_PATH = Path(__file__).with_name("conftest.py")
ROOT_CONFTEST_SPEC = importlib.util.spec_from_file_location("root_conftest", ROOT_CONFTEST_PATH)
assert ROOT_CONFTEST_SPEC is not None
assert ROOT_CONFTEST_SPEC.loader is not None
root_conftest = importlib.util.module_from_spec(ROOT_CONFTEST_SPEC)
ROOT_CONFTEST_SPEC.loader.exec_module(root_conftest)


class TestRetryUrlOperation:
    @pytest.mark.unit
    def test_retries_with_exponential_backoff(self):
        result = object()
        operation = Mock(side_effect=[urllib.error.URLError("transient"), OSError("transient"), result])

        with patch.object(root_conftest.time, "sleep") as sleep:
            assert root_conftest._retry_url_operation(operation) is result

        assert operation.call_count == 3
        assert sleep.call_args_list == [call(2), call(4)]

    @pytest.mark.unit
    def test_raises_after_five_attempts(self):
        operation = Mock(side_effect=urllib.error.URLError("unavailable"))

        with patch.object(root_conftest.time, "sleep") as sleep, pytest.raises(urllib.error.URLError):
            root_conftest._retry_url_operation(operation)

        assert operation.call_count == 5
        assert sleep.call_args_list == [call(2), call(4), call(8), call(16)]


NIGHTLY_CONFTEST_PATH = Path(__file__).parent / "e2e_nightly" / "conftest.py"
NIGHTLY_CONFTEST_SPEC = importlib.util.spec_from_file_location("nightly_conftest", NIGHTLY_CONFTEST_PATH)
nightly_conftest = importlib.util.module_from_spec(NIGHTLY_CONFTEST_SPEC)
NIGHTLY_CONFTEST_SPEC.loader.exec_module(nightly_conftest)


def _item_at(path, timeout=None):
    item = Mock()
    item.path = path
    item.get_closest_marker.side_effect = lambda name: (
        pytest.mark.timeout(timeout).mark if name == "timeout" and timeout is not None else None
    )
    return item


class TestNightlyTimeout:
    @pytest.mark.unit
    def test_only_unmarked_nightly_items_get_the_longer_timeout(self):
        nightly_dir = NIGHTLY_CONFTEST_PATH.parent
        nightly = _item_at(nightly_dir / "test_model_support_x.py")
        nightly_marked = _item_at(nightly_dir / "test_model_support_y.py", timeout=60)
        ordinary = _item_at(nightly_dir.parent / "test_data_dir.py")
        ordinary_marked = _item_at(nightly_dir.parent / "test_data_dir.py", timeout=30)

        nightly_conftest.pytest_collection_modifyitems(None, [nightly, nightly_marked, ordinary, ordinary_marked])

        nightly.add_marker.assert_called_once_with(pytest.mark.timeout(1200))
        nightly_marked.add_marker.assert_not_called()
        ordinary.add_marker.assert_not_called()
        ordinary_marked.add_marker.assert_not_called()


def _fake_item(markers, check_gpu_markers=True, cpu=False):
    item = Mock()
    item.nodeid = "tests/test_fake.py::test_fake"
    item.get_closest_marker.side_effect = lambda name: markers.get(name)
    item.config.getoption.side_effect = {"--check-gpu-markers": check_gpu_markers, "--cpu": cpu}.__getitem__
    item.config.stash = {}
    return item


def _run_protocol(item, allocations):
    with patch.object(root_conftest, "_num_cuda_allocations", side_effect=allocations):
        wrapper = root_conftest.pytest_runtest_protocol(item, None)
        next(wrapper)
        with pytest.raises(StopIteration):
            next(wrapper)


class TestGpuMarker:
    @pytest.mark.unit
    def test_gpu_only_tests_get_gpu_marker(self):
        gpu_only = _fake_item({"run_only_on": pytest.mark.run_only_on("GPU").mark})
        cpu_only = _fake_item({"run_only_on": pytest.mark.run_only_on("CPU").mark})
        unrestricted = _fake_item({})

        root_conftest._add_gpu_marker_to_gpu_only_tests([gpu_only, cpu_only, unrestricted])

        gpu_only.add_marker.assert_called_once_with(pytest.mark.gpu)
        cpu_only.add_marker.assert_not_called()
        unrestricted.add_marker.assert_not_called()

    @pytest.mark.unit
    def test_already_marked_gpu_only_test_is_not_marked_twice(self):
        item = _fake_item({"run_only_on": pytest.mark.run_only_on("GPU").mark, "gpu": pytest.mark.gpu.mark})

        root_conftest._add_gpu_marker_to_gpu_only_tests([item])

        item.add_marker.assert_not_called()

    @pytest.mark.unit
    def test_audit_records_unmarked_test_that_allocates_cuda_memory(self):
        item = _fake_item({})

        # Pin the CPU default device: other test modules may set a CUDA default at import time.
        with patch("torch.get_default_device", return_value=torch.device("cpu")):
            _run_protocol(item, allocations=[10, 12])

        assert item.config.stash[root_conftest._UNMARKED_CUDA_TESTS] == [item.nodeid]

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "markers,allocations",
        [({"gpu": pytest.mark.gpu.mark}, [10, 12]), ({}, [10, 10])],
        ids=["marked", "no-cuda-allocations"],
    )
    def test_audit_ignores_marked_or_cpu_only_tests(self, markers, allocations):
        item = _fake_item(markers)

        _run_protocol(item, allocations=allocations)

        assert root_conftest._UNMARKED_CUDA_TESTS not in item.config.stash

    @pytest.mark.unit
    def test_audit_ignores_tests_run_with_leaked_cuda_default_device(self):
        item = _fake_item({})

        with patch("torch.get_default_device", return_value=torch.device("cuda")):
            _run_protocol(item, allocations=[10, 12])

        assert root_conftest._UNMARKED_CUDA_TESTS not in item.config.stash

    @pytest.mark.unit
    @pytest.mark.parametrize("check_gpu_markers,cpu", [(False, False), (True, True)], ids=["disabled", "cpu-mode"])
    def test_audit_is_inactive_without_option_or_in_cpu_mode(self, check_gpu_markers, cpu):
        item = _fake_item({}, check_gpu_markers=check_gpu_markers, cpu=cpu)

        with patch.object(root_conftest, "_num_cuda_allocations") as num_allocations:
            wrapper = root_conftest.pytest_runtest_protocol(item, None)
            next(wrapper)
            with pytest.raises(StopIteration):
                next(wrapper)

        num_allocations.assert_not_called()

    @pytest.mark.unit
    def test_session_fails_when_unmarked_cuda_tests_were_found(self):
        session = Mock()
        session.config.stash = {root_conftest._UNMARKED_CUDA_TESTS: ["tests/test_fake.py::test_fake"]}
        session.exitstatus = pytest.ExitCode.OK

        root_conftest.pytest_sessionfinish(session, pytest.ExitCode.OK)

        assert session.exitstatus == pytest.ExitCode.TESTS_FAILED

    @pytest.mark.unit
    def test_num_cuda_allocations_is_zero_before_cuda_initialization(self):
        torch = Mock()
        torch.cuda.is_available.return_value = True
        torch.cuda.is_initialized.return_value = False

        assert root_conftest._num_cuda_allocations(torch) == 0
        torch.cuda.memory_stats.assert_not_called()


# Run in a separate pytest session: it replaces checkpoint restoration with a lightweight model that has a reference
# cycle (like ModelPT) and disables automatic garbage collection, so only the fixture's own cleanup can free a model.
_CHECKPOINT_LIFETIME_PLUGIN = textwrap.dedent(
    """
    import gc
    import json
    import os
    import types
    import weakref

    import torch

    _refs = []
    _alive_at_restore = []


    class _FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(2, 2)
            self.cycle = types.MethodType(lambda self: self, self)
            self.joint = types.SimpleNamespace(num_classes_with_blank=5)
            self.decoder = types.SimpleNamespace(vocab_size=4)
            self.tokenizer = types.SimpleNamespace(vocab_size=4)

        def to_config_dict(self):
            return {}

        def training_step(self, batch, batch_idx):
            return self.lin(torch.ones(2)).sum()

        def transcribe(self, audio, batch_size, return_hypotheses=False):
            return [types.SimpleNamespace(text="x")] if return_hypotheses else ["x"]


    def _fake_restore(*args, **kwargs):
        _alive_at_restore.append(sum(ref() is not None for ref in _refs))
        model = _FakeModel()
        _refs.append(weakref.ref(model))
        return model


    def pytest_configure(config):
        gc.disable()
        from nemo.collections.asr.models import ASRModel

        ASRModel.restore_from = staticmethod(_fake_restore)


    def pytest_sessionfinish(session, exitstatus):
        with open(os.environ["LIFETIME_REPORT"], "w") as f:
            json.dump({"alive_at_restore": _alive_at_restore}, f)
    """
)


@pytest.mark.unit
def test_shared_architecture_fixture_frees_previous_checkpoint_before_next_restore(tmp_path):
    (tmp_path / "checkpoint_lifetime_plugin.py").write_text(_CHECKPOINT_LIFETIME_PLUGIN)
    report = tmp_path / "report.json"
    nightly_dir = NIGHTLY_CONFTEST_PATH.parent
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(tmp_path), str(nightly_dir), os.environ.get("PYTHONPATH", "")]),
        "LIFETIME_REPORT": str(report),
        "CUDA_VISIBLE_DEVICES": "",
        # Select two checkpoints so the fixture switches parameters once.
        "NEMO_MODEL_SUPPORT_SHARD": "0",
        "NEMO_MODEL_SUPPORT_NUM_SHARDS": "23",
    }
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "checkpoint_lifetime_plugin",
            "-p",
            "no:cacheprovider",
            "--noconftest",
            "-c",
            os.devnull,
            "--rootdir",
            str(tmp_path),
            str(nightly_dir / "test_model_support_asr_shared_architectures.py"),
        ],
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    # One restore per checkpoint, shared by its three tests; no earlier checkpoint is alive at the second restore.
    assert json.loads(report.read_text()) == {"alive_at_restore": [0, 0]}
