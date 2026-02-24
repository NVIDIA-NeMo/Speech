# Copyright (c) 2025, NVIDIA CORPORATION.
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

"""Shared fixtures and utilities for per-model functional tests."""

import torch


class _MinimalTrainerStub:
    """Provides just enough Trainer-like interface for training_step()."""

    global_step = 0
    log_every_n_steps = 1_000_000  # large value to skip WER computation during test

    # Lightning's log() checks these attributes:
    training = True
    sanity_checking = False
    barebones = False

    @property
    def callback_metrics(self):
        return {}


def prepare_for_training_step(model):
    """Prepare a model for a direct training_step() call without a full Trainer."""
    model.train()

    # Attach minimal trainer stub for models that access self.trainer
    stub = _MinimalTrainerStub()
    model._trainer = stub

    # Suppress Lightning logging (requires active Trainer control flow).
    # We don't need logging in tests — we only care about loss computation.
    model.log = lambda *a, **kw: None
    model.log_dict = lambda *a, **kw: None

    # Many NeMo models access self._optimizer.param_groups[0]['lr'] in training_step
    # to log the learning rate. Provide a minimal stand-in if no optimizer is set.
    if getattr(model, '_optimizer', None) is None:
        model._optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1e-4)


def prepare_for_transcribe(model):
    """Placeholder for any pre-transcribe model preparation.

    Previously disabled CUDA graph decoding, but the root cause
    (CUDA 12/13 API mismatch in cudaStreamGetCaptureInfo) is now
    fixed in the NeMo source. Kept as a no-op for compatibility
    with test files that call it.
    """
    pass
