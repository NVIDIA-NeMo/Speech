# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Drift check for the vLLM plugin's audio-token estimator.

The plugin hand-rolls
``NeMoSpeechLMProcessingInfo._estimate_audio_tokens`` in pure Python for
speed (~90x faster than the equivalent tensor-ops path). Accuracy
matters: the result drives how many ``<|audio|>`` placeholders the
prompt processor inserts AND the encoder's real output frame count at
forward time; a mismatch breaks placeholder-to-feature alignment.

This test locks the hand-rolled math to NeMo's ``calc_length`` on a
canonical set of audio lengths. If FastConformer's subsampling ever
changes upstream (different kernel/stride/repeat), this test fails and
forces an update to the estimator.
"""

import pytest
import torch

pytest.importorskip("vllm")

from nemo.collections.asr.parts.submodules.subsampling import calc_length
from nemo.collections.speechlm2.vllm.salm.audio import NeMoSpeechLMProcessingInfo


def _reference(audio_length_samples: int) -> int:
    """Reference impl using NeMo's calc_length for the conv chain."""
    n_fft = 512
    hop_length = 160
    stft_pad = n_fft // 2
    fbank_len = (audio_length_samples + 2 * stft_pad - n_fft) // hop_length
    out = calc_length(
        torch.tensor([fbank_len], dtype=torch.float),
        all_paddings=2,
        kernel_size=3,
        stride=2,
        ceil_mode=False,
        repeat_num=3,
    )
    return max(1, int(out.item()))


@pytest.mark.parametrize(
    "samples",
    [
        1_600,  # 0.1 s
        16_000,  # 1 s
        80_000,  # 5 s
        160_000,  # 10 s
        320_000,  # 20 s
        640_000,  # 40 s, the typical max
        12_345,  # arbitrary small
        54_321,  # arbitrary mid
        100_001,  # arbitrary (odd)
    ],
)
def test_estimator_matches_calc_length(samples: int) -> None:
    ours = NeMoSpeechLMProcessingInfo._estimate_audio_tokens(samples)
    ref = _reference(samples)
    assert ours == ref, (
        f"audio_token estimator diverged from NeMo calc_length for "
        f"samples={samples}: ours={ours}, ref={ref}. "
        f"Check if FastConformer's subsampling stack changed upstream."
    )


def test_estimator_min_one() -> None:
    """Even for very short audio the estimator must return at least 1."""
    assert NeMoSpeechLMProcessingInfo._estimate_audio_tokens(1) >= 1
