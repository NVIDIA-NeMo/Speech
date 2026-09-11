# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import copy
from contextlib import contextmanager

import librosa
import torch

from nemo.collections.asr.models import ASRModel


@contextmanager
def preserve_decoding_cfg_and_cpu_device(model: ASRModel):
    """
    Context manager to preserve the decoding strategy and device of the model.
    This is useful for tests that modify the model's decoding strategy or device
    to avoid side effects or costly model reloading.
    """
    backup_decoding_cfg = copy.deepcopy(model.cfg.decoding)

    try:
        yield
    finally:
        model.to(device="cpu")
        if model.cfg.decoding != backup_decoding_cfg:
            model.change_decoding_strategy(backup_decoding_cfg)


def load_audio(file_path, target_sr=16000) -> tuple[torch.Tensor, int]:
    audio, sr = librosa.load(file_path, sr=target_sr)
    return torch.tensor(audio, dtype=torch.float32), sr


@contextmanager
def avoid_sync_operations(device: torch.device):
    try:
        if device.type == "cuda":
            torch.cuda.set_sync_debug_mode(2)  # fail if a blocking operation
        yield
    finally:
        if device.type == "cuda":
            torch.cuda.set_sync_debug_mode(0)  # default, blocking operations are allowed


def make_preprocessor_deterministic(model: ASRModel):
    model.preprocessor.featurizer.dither = 0.0
    model.preprocessor.featurizer.pad_to = 0
