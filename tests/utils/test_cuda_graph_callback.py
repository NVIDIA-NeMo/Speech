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

import inspect
from types import MethodType, SimpleNamespace

import pytest
import torch
import torch._dynamo
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities import CombinedLoader
from lightning.pytorch.utilities.signature_utils import is_param_in_hook_signature
from torch.utils.data import DataLoader, TensorDataset

from nemo.utils.callbacks import cuda_graph as cuda_graph_module
from nemo.utils.callbacks.cuda_graph import CUDAGraphCallback, CUDAGraphState, get_training_step


class _Module(LightningModule):
    def training_step(self, batch, batch_idx):
        return {"loss": batch.sum() + batch_idx}


def _copy_on_the_host(src):
    """``struct_copy_one`` for a CPU test of the wrapper: the callback's copy moves the step output to CUDA."""
    return {key: value.clone() for key, value in src.items()}


class _Fetcher:
    """The two calls of the trainer's data fetcher that the callback makes."""

    def setup(self, combined_loader):
        """Nothing to prepare."""

    def __iter__(self):
        return iter(())


class TestCUDAGraphCallbackPatches:
    @pytest.mark.unit
    def test_autocast_replacement_keeps_the_signature_and_compiles_autocast_scopes(self):
        """Dynamo binds the arguments of an autocast scope against ``inspect.signature(torch.autocast)``,
        so the replacement installed by ``setup`` keeps the original parameters and still turns the cache off."""
        original = torch.autocast.__init__
        callback = CUDAGraphCallback(capture_iteration=12)
        callback.setup(trainer=None, pl_module=None, stage="fit")
        torch._dynamo.reset()
        try:
            assert torch.autocast.__init__ is not original
            assert list(inspect.signature(torch.autocast.__init__).parameters) == list(
                inspect.signature(original).parameters
            )
            with torch.autocast("cpu", dtype=torch.bfloat16):
                assert torch.is_autocast_cache_enabled() is False
            with torch.autocast("cpu", dtype=torch.bfloat16, cache_enabled=True):
                assert torch.is_autocast_cache_enabled() is True

            def step(x):
                with torch.autocast(device_type="cpu", enabled=False):
                    y = x.float() * 2
                return y + 1

            # the eager backend: the binding of the scope's arguments is Dynamo's, no kernel is generated
            out = torch.compile(step, backend="eager", fullgraph=True)(torch.ones(4))
            torch.testing.assert_close(out, torch.full((4,), 3.0))
        finally:
            callback.teardown(trainer=None, pl_module=None, stage="fit")
            torch._dynamo.reset()
        assert torch.autocast.__init__ is original

    @pytest.mark.unit
    def test_training_step_wrapper_forwards_the_batch_index(self, monkeypatch):
        monkeypatch.setattr(cuda_graph_module, "struct_copy_one", _copy_on_the_host)
        module = _Module()
        module.__orig_training_step__ = module.training_step
        module.training_step = MethodType(get_training_step(CUDAGraphState(capture_iteration=12)), module)
        assert float(module.training_step(torch.ones(2), 3)["loss"]) == 5.0

    @pytest.mark.unit
    def test_wrapped_training_step_keeps_the_signature_lightning_reads(self, monkeypatch):
        """Lightning passes the batch index only when ``training_step`` takes a second parameter after ``self``."""
        # the static-buffer loader and the output copy need CUDA; the signature and the forwarding do not
        monkeypatch.setattr(cuda_graph_module, "StaticBufferLoader", lambda loader: loader)
        monkeypatch.setattr(cuda_graph_module, "struct_copy_one", _copy_on_the_host)
        module = _Module()
        loader = DataLoader(TensorDataset(torch.zeros(2, 1)), batch_size=1)
        fit_loop = SimpleNamespace(_combined_loader=CombinedLoader(loader), _data_fetcher=_Fetcher())
        trainer = SimpleNamespace(fit_loop=fit_loop, optimizers=[], lr_scheduler_configs=[])
        callback = CUDAGraphCallback(capture_iteration=12)
        callback.on_train_start(trainer, module)
        try:
            assert is_param_in_hook_signature(module.training_step, "batch_idx", min_args=2)
            assert list(inspect.signature(module.training_step).parameters) == ["batch", "batch_idx"]
            assert float(module.training_step(torch.ones(2), 3)["loss"]) == 5.0
        finally:
            callback.on_train_end(trainer, module)
        assert list(inspect.signature(module.training_step).parameters) == ["batch", "batch_idx"]
        assert fit_loop._combined_loader._iterables is loader
