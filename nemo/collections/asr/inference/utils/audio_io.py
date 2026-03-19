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

import librosa
import torch


def read_audio(audio_file: str, target_sr: int, mono: bool = True) -> torch.Tensor:
    """
    Read audio file and return samples with target sampling rate
    Args:
        audio_file: (str) audio file path
        target_sr: (int) target sampling rate
        mono: (bool) whether to convert to mono
    Returns:
        (torch.Tensor) audio samples
    """
    return torch.tensor(librosa.load(audio_file, sr=target_sr, mono=mono)[0]).float()
