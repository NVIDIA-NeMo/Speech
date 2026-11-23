# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Check packed perception features against a fixed synthetic encoder fixture."""

import os
from pathlib import Path

import pytest
import soundfile as sf
import torch
from safetensors.torch import load_file

from nemo.collections.asr.modules import AudioToMelSpectrogramPreprocessor
from nemo.collections.asr.modules.parallel_expert_encoder import ParallelExpertEncoderPT


@pytest.fixture(scope="module")
def fixture_dir():
    return Path(os.environ["NEMOTRON35_ENCODER_DIR"])


@pytest.fixture(scope="module")
def reference(fixture_dir):
    return load_file(fixture_dir / "encoder-reference.safetensors")


@pytest.fixture(scope="module")
def encoder(fixture_dir):
    encoder = ParallelExpertEncoderPT.load_from_nemo(
        str(fixture_dir / "parallel-expert-encoder.nemo"),
        config_overrides={"chunk_size_seconds": 30.0, "sync_max_audio_length": False},
    )
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    return encoder.to(device="cuda", dtype=torch.bfloat16).eval()


@pytest.fixture(scope="module")
def packed_mels(fixture_dir):
    waveforms = []
    for name in ("short.wav", "long.wav"):
        wave, sample_rate = sf.read(fixture_dir / name, dtype="float32")
        assert sample_rate == 16000 and wave.ndim == 1, name
        waveforms.append(torch.from_numpy(wave))
    assert waveforms[0].numel() < 30 * 16000 < waveforms[1].numel()
    lengths = torch.tensor([wave.numel() for wave in waveforms], device="cuda")
    cu_seqlens = torch.cat((lengths.new_zeros(1), lengths.cumsum(0)))
    preprocessor = AudioToMelSpectrogramPreprocessor(
        sample_rate=16000, window_size=0.025, features=128, n_fft=512, normalize=None, dither=0.0, pad_to=0
    ).to(device="cuda", dtype=torch.bfloat16)
    return preprocessor.forward_packed(torch.cat(waveforms).cuda(), lengths, cu_seqlens)


@pytest.fixture
def targets():
    # Two clips, padded to the longer clip on the 80 ms grid, and eight speakers.
    return torch.zeros((2, 377, 8), device="cuda")


def test_missing_rttm_matches_predicted_speaker_reference(encoder, packed_mels, targets, reference):
    targets.fill_(-1)
    _assert_packed_features(encoder, packed_mels, targets, reference, "missing")


def test_silent_rttm_matches_zero_activity_reference(encoder, packed_mels, targets, reference):
    _assert_packed_features(encoder, packed_mels, targets, reference, "silent")


def test_mixed_rttm_matches_reference_with_speaker_seven(encoder, packed_mels, targets, reference):
    targets[0] = -1
    targets[1, 40:100, 7] = 1
    _assert_packed_features(encoder, packed_mels, targets, reference, "mixed")


@torch.inference_mode()
def _assert_packed_features(encoder, packed_mels, targets, reference, case):
    encoded = encoder.forward_sequence_packed(packed_mels, spk_targets=targets)
    expected_lengths = reference[f"{case}.lengths"]
    torch.testing.assert_close(encoded.lengths.cpu(), expected_lengths, rtol=0, atol=0)
    torch.testing.assert_close(encoded.cu_seqlens.cpu(), reference[f"{case}.cu_seqlens"], rtol=0, atol=0)
    actual_rows = encoded.data.float().cpu().split(expected_lengths.tolist())
    expected_rows = reference[f"{case}.features"].split(expected_lengths.tolist())
    for row, (actual, expected) in enumerate(zip(actual_rows, expected_rows, strict=True)):
        assert actual.shape == expected.shape
        relative_rms_error = ((actual - expected).norm() / expected.norm()).item()
        # Allow BF16 kernel roundoff across GPUs.
        assert relative_rms_error <= 0.02, f"{case} utterance {row}: relative RMS error {relative_rms_error:.2%}"
