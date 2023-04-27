# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import functools
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from einops import rearrange
from scipy import ndimage
from torch.special import gammaln


def get_abs_rel_paths(input_path: Path, base_path: Path) -> Tuple[Path, Path]:
    """
    Get the absolute and relative paths of input file path.

    Args:
        input_path: An absolute or relative path.
        base_path: base directory the input is relative to.

    Returns:
        The absolute and relative paths of the file.
    """
    if os.path.isabs(input_path):
        abs_path = input_path
        rel_path = input_path.relative_to(base_path)
    else:
        rel_path = input_path
        abs_path = base_path / rel_path

    return abs_path, rel_path


def get_audio_filepaths(manifest_entry: dict, audio_dir: Path) -> Tuple[Path, Path]:
    """
    Get the absolute and relative paths of audio from a manifest entry.

    Args:
        manifest_entry: Manifest entry dictionary.
        audio_dir: base directory where audio is stored.

    Returns:
        The absolute and relative paths of the audio.
    """
    audio_filepath = Path(manifest_entry["audio_filepath"])
    audio_filepath_abs, audio_filepath_rel = get_abs_rel_paths(input_path=audio_filepath, base_path=audio_dir)
    return audio_filepath_abs, audio_filepath_rel


def normalize_volume(audio: np.array, volume_level: float) -> np.array:
    """Apply peak normalization to the input audio.
    """
    if not (0.0 <= volume_level <= 1.0):
        raise ValueError(f"Volume must be in range [0.0, 1.0], received {volume_level}")

    max_sample = np.max(np.abs(audio))
    if max_sample == 0:
        return audio

    return volume_level * (audio / np.max(np.abs(audio)))


class BetaBinomialInterpolator:
    """
    This module calculates alignment prior matrices (based on beta-binomial distribution) using cached popular sizes and image interpolation.
    The implementation is taken from https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/SpeechSynthesis/FastPitch/fastpitch/data_function.py
    """

    def __init__(self, round_mel_len_to=50, round_text_len_to=10, cache_size=500):
        self.round_mel_len_to = round_mel_len_to
        self.round_text_len_to = round_text_len_to
        self.bank = functools.lru_cache(maxsize=cache_size)(beta_binomial_prior_distribution)

    @staticmethod
    def round(val, to):
        return max(1, int(np.round((val + 1) / to))) * to

    def __call__(self, w, h):
        bw = BetaBinomialInterpolator.round(w, to=self.round_mel_len_to)
        bh = BetaBinomialInterpolator.round(h, to=self.round_text_len_to)
        ret = ndimage.zoom(self.bank(bw, bh).T, zoom=(w / bw, h / bh), order=1)
        assert ret.shape[0] == w, ret.shape
        assert ret.shape[1] == h, ret.shape
        return ret


def general_padding(item, item_len, max_len, pad_value=0):
    if item_len < max_len:
        item = torch.nn.functional.pad(item, (0, max_len - item_len), value=pad_value)
    return item


def logbeta(x, y):
    return gammaln(x) + gammaln(y) - gammaln(x + y)


def logcombinations(n, k):
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)


def logbetabinom(n, a, b, x):
    return logcombinations(n, x) + logbeta(x + a, n - x + b) - logbeta(a, b)


def beta_binomial_prior_distribution(phoneme_count: int, mel_count: int, scaling_factor: float = 1.0) -> np.array:
    x = rearrange(torch.arange(0, phoneme_count), "b -> 1 b")
    y = rearrange(torch.arange(1, mel_count + 1), "b -> b 1")
    a = scaling_factor * y
    b = scaling_factor * (mel_count + 1 - y)
    n = torch.FloatTensor([phoneme_count - 1])

    return logbetabinom(n, a, b, x).exp().numpy()


def get_base_dir(paths):
    def is_relative_to(path1, path2):
        try:
            path1.relative_to(path2)
            return True
        except ValueError:
            return False

    def common_path(path1, path2):
        while path1 is not None:
            if is_relative_to(path2, path1):
                return path1
            path1 = path1.parent if path1 != path1.parent else None
        return None

    base_dir = None
    for p in paths:
        audio_dir = Path(p).parent
        if base_dir is None:
            base_dir = audio_dir
            continue
        base_dir = common_path(base_dir, audio_dir)

    return base_dir
