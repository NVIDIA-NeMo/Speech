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


class TestFileShards:
    @pytest.mark.unit
    def test_files_are_balanced_by_duration(self):
        durations = {"a.py": 10.0, "b.py": 6.0, "c.py": 4.0, "d.py": 1.0}

        shards = root_conftest._assign_files_to_shards(list(durations), durations, num_shards=2)

        # Longest first, each to the least-loaded shard: a -> 0 (10), b -> 1 (6), c -> 1 (10), d -> 0 (tie, lowest id).
        assert shards == {"a.py": 0, "b.py": 1, "c.py": 1, "d.py": 0}

    @pytest.mark.unit
    def test_every_file_gets_exactly_one_shard(self):
        files = [f"test_{i}.py" for i in range(20)]
        durations = {f: float(i) for i, f in enumerate(files)}

        shards = root_conftest._assign_files_to_shards(files, durations, num_shards=3)

        assert set(shards) == set(files)
        assert set(shards.values()) == {0, 1, 2}

    @pytest.mark.unit
    def test_unknown_files_count_as_median_duration(self):
        durations = {"a.py": 1.0, "b.py": 5.0, "c.py": 9.0}

        shards = root_conftest._assign_files_to_shards(["a.py", "b.py", "c.py", "new.py"], durations, num_shards=2)

        # new.py counts as the median (5): c -> 0 (9), b -> 1 (5), new -> 1 (10), a -> 0 (10).
        assert shards == {"c.py": 0, "b.py": 1, "new.py": 1, "a.py": 0}

    @pytest.mark.unit
    def test_committed_durations_table_has_both_modes(self):
        for mode in ("cpu", "gpu"):
            durations = root_conftest._load_test_file_durations(mode)
            assert durations
            assert all(path.startswith("tests/") and path.endswith(".py") for path in durations)
