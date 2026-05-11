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
"""Shared fakes for SALM / SALMAutomodel encoder-chunking tests."""
from types import SimpleNamespace

import torch


def chunking_test_devices():
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    return devices


class ChunkingTestTokenizer:
    pad = 0
    unk_id = None
    bos_id = 1
    eos_id = 2

    def __init__(self, audio_locator_tag):
        self._audio_locator_tag = audio_locator_tag

    def token_to_id(self, token):
        assert token == self._audio_locator_tag
        return 99


class ChunkingTestPerception(torch.nn.Module):
    def __init__(self, sampling_rate, hop_length):
        super().__init__()
        self.preprocessor = SimpleNamespace(featurizer=_ChunkingTestFeaturizer(sampling_rate, hop_length))
        self.calls = []

    def forward(self, input_signal=None, input_signal_length=None):
        self.calls.append((input_signal.detach().clone(), input_signal_length.detach().clone()))
        max_len = int(input_signal_length.max().item()) if input_signal_length.numel() > 0 else 0
        return input_signal[:, :max_len].unsqueeze(-1), input_signal_length.clone()


class _ChunkingTestFeaturizer:
    def __init__(self, sampling_rate, hop_length):
        self.sample_rate = sampling_rate
        self.hop_length = hop_length

    def get_seq_len(self, seq_len):
        return torch.floor_divide(seq_len, self.hop_length).to(dtype=torch.long)
