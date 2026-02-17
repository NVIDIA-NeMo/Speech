# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import pytest
import torch

from nemo.collections.tts.metrics.frechet_codec_distance import FrechetCodecDistance
from nemo.collections.tts.models import AudioCodecModel


class TestFrechetCodecDistance:
    codec_name = "nvidia/low-frame-rate-speech-codec-22khz"

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.fixture
    def codec(self, device, scope="session"):
        return AudioCodecModel.from_pretrained(self.codec_name).to(device)

    @pytest.fixture
    def metric(self, codec, device):
        return FrechetCodecDistance(codec_name=self.codec_name).to(device)

    @pytest.mark.unit
    def test_same_distribution(self, metric, device, codec):
        """Test that FCD is close to zero when comparing identical distributions."""
        B, C, T = 3, codec.num_codebooks, 20
        codes = torch.randint(low=0, high=codec.codebook_size, size=(B, C, T), device=device)
        codes_len = torch.randint(low=1, high=T, size=(B,), device=device)

        # Update with same codes for both real and fake
        metric.update(codes, codes_len, is_real=True)
        metric.update(codes, codes_len, is_real=False)

        eps = 0.01
        fcd = metric.compute()
        assert fcd < eps and fcd >= 0, f"FCD value is {fcd} but should be close to 0"
        metric.reset()

    @pytest.mark.unit
    def test_different_distribution(self, metric, device, codec):
        """Test that FCD is positive when comparing different distributions."""
        B, C, T = 3, codec.num_codebooks, 20

        # Generate two different sets of codes
        codes1 = torch.randint(low=0, high=codec.codebook_size, size=(B, C, T), device=device)
        codes2 = torch.randint(low=0, high=codec.codebook_size, size=(B, C, T), device=device)
        codes_len = torch.randint(low=1, high=T, size=(B,), device=device)

        metric.update(codes1, codes_len, is_real=True)
        metric.update(codes2, codes_len, is_real=False)

        fcd = metric.compute()
        assert fcd > 0, f"FCD value is {fcd} but should be positive for different distributions"
        metric.reset()

    @pytest.mark.filterwarnings("ignore:The.*compute.*method of metric.*was called before the.*update.*method")
    @pytest.mark.unit
    def test_empty_distribution(self, metric):
        """Test that computing the FCD on empty distributions returns 0."""
        fcd = metric.compute()
        assert fcd == 0.0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.unit
    def test_gpu_compatibility(self, metric, device, codec):
        """Test that the metric works correctly on GPU."""
        assert metric.device.type == "cuda"
        B, C, T = 3, codec.num_codebooks, 20
        codes = torch.randint(low=0, high=codec.codebook_size, size=(B, C, T), device=device)
        codes_len = torch.randint(low=1, high=T, size=(B,), device=device)

        metric.update(codes, codes_len, is_real=True)
        metric.update(codes, codes_len, is_real=False)

        fcd = metric.compute()

        eps = 0.01
        assert isinstance(fcd, torch.Tensor)
        assert fcd.device.type == "cuda"
        assert fcd < eps and fcd >= 0, f"FCD value is {fcd} but should be close to 0"

    @pytest.mark.unit
    def test_update_from_audio_file(self, metric):
        """Test the update_from_audio_file method."""

        # Test with both "real" and "fake" audio files (different files)
        metric.update_from_audio_file("tests/.data/tts/mini_ljspeech/wavs/LJ019-0373.wav", is_real=True)
        metric.update_from_audio_file("tests/.data/tts/mini_ljspeech/wavs/LJ050-0234.wav", is_real=False)

        fcd = metric.compute()
        assert isinstance(fcd, torch.Tensor)
        assert fcd > 0, f"FCD value is {fcd} but should be positive given that we tested different audio files"

    @pytest.mark.unit
    def test_empty_codes_update(self, metric, device):
        """Test that the FCD metric doesn't crash when provided with empty codes."""
        B, C, T = 1, 0, 100
        codes = torch.ones(B, C, T, device=device)
        codes_len = T * torch.ones(B, device=device)
        # if it crashes PyTest will report it
        metric.update(codes, codes_len, is_real=True)

    @pytest.mark.unit
    def test_codebooks_mismatch_update(self, metric, device, codec):
        """Test that the FCD metric doesn't crash when provided with incorrect number ofcodebooks."""
        B = 2
        C = codec.num_codebooks - 1  # intentionally missing one codebook
        T = 10
        codes = torch.ones(B, C, T, device=device)
        codes_len = T * torch.ones(B, device=device, dtype=torch.long)
        # if it crashes PyTest will report it
        metric.update(codes, codes_len, is_real=True)
