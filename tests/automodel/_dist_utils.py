# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import contextlib
import importlib

import sys
import types

import pytest


################################################################################
#  A light-weight dummy torch.distributed implementation
################################################################################
class _DummyDist:
    def __init__(self):
        self._init = False
        self._rank = 0
        self.barrier_calls = 0
        self.init_calls = 0
        self.destroy_calls = 0

    # --- public API the context uses -----------------------------------------
    def is_initialized(self):
        return self._init

    def get_rank(self):
        if not self._init:
            raise RuntimeError("process-group not initialised")
        return self._rank

    def barrier(self):
        if not self._init:
            raise RuntimeError("process-group not initialised")
        self.barrier_calls += 1

    def init_process_group(self, *, backend, world_size, rank):
        assert backend == "gloo"
        self._init = True
        self._rank = rank
        self.init_calls += 1

    def destroy_process_group(self):
        if self._init:
            self._init = False
            self.destroy_calls += 1


################################################################################
#  Inject the dummy module before importing FirstRankPerNode
################################################################################
dummy_dist = _DummyDist()

torch_mod = types.ModuleType("torch")
torch_mod.distributed = dummy_dist
sys.modules["torch"] = torch_mod
sys.modules["torch.distributed"] = dummy_dist  # for `import torch.distributed as dist`

# --------------------------------------------------------------------------- #
# Import the code under test *after* the monkey-patch so that it picks up the
# dummy torch.distributed instance.
# --------------------------------------------------------------------------- #
FirstRankPerNode = importlib.import_module("nemo.automodel.dist_utils").FirstRankPerNode
################################################################################


# Helper: reset dummy state between tests
@contextlib.contextmanager
def fresh_dummy_dist():
    dummy_dist.__init__()  # reset counters / flags
    yield
    dummy_dist.__init__()  # clean up afterwards


###############################################################################
#                                   TESTS
###############################################################################


def test_single_gpu(monkeypatch):
    """No process-group, no env → behaves like single-GPU."""
    for k in ("WORLD_SIZE", "RANK", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
        monkeypatch.delenv(k, raising=False)

    with fresh_dummy_dist():
        with FirstRankPerNode() as first:
            assert first is True  # LOCAL_RANK==0 by default
        assert dummy_dist.barrier_calls == 0
        assert dummy_dist.destroy_calls == 0


def test_auto_bootstrap(monkeypatch):
    """
    dist not initialised, but env variables present → _try_bootstrap_pg path.
    Use rank 1 so that `first` is False and the first barrier is executed.
    """
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("MASTER_ADDR", "localhost")
    monkeypatch.setenv("MASTER_PORT", "29500")
    monkeypatch.delenv("LOCAL_RANK", raising=False)

    with fresh_dummy_dist():
        with FirstRankPerNode() as first:
            assert first is False  # local_rank == 1
        assert dummy_dist.init_calls == 1
        assert dummy_dist.barrier_calls == 1  # only the enter-barrier
        assert dummy_dist.destroy_calls == 1  # _created_pg → destroyed


def test_preinitialised_rank0(monkeypatch):
    """Process-group already created; we are local-rank 0."""
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")

    # simulate an already-initialised PG
    dummy_dist._init = True
    dummy_dist._rank = 0

    with fresh_dummy_dist():
        dummy_dist._init = True
        dummy_dist._rank = 0

        with FirstRankPerNode() as first:
            assert first is True
            assert dummy_dist.barrier_calls == 0  # nothing yet

        # exit-barrier for the rest of the ranks
        assert dummy_dist.barrier_calls == 1
        assert dummy_dist.destroy_calls == 0  # PG remains


def test_exception_path(monkeypatch):
    """
    Exception raised inside context → PG is destroyed (only when first==True).
    """
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")

    with fresh_dummy_dist():
        dummy_dist._init = True
        dummy_dist._rank = 0

        with pytest.raises(RuntimeError):
            with FirstRankPerNode():
                raise RuntimeError("boom")

        # barrier executed before destroy
        assert dummy_dist.barrier_calls == 1
        assert dummy_dist.destroy_calls == 1
