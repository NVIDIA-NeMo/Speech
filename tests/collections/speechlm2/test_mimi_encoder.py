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
"""Test MimiEncoder wrapper with mocked HuggingFace MimiModel."""

import pytest
import torch
import torch.nn as nn


@pytest.fixture
def mock_mimi_model(monkeypatch):
    """Mock transformers.MimiModel and AutoFeatureExtractor."""

    class FakeEncoderOutput:
        def __init__(self, codes):
            self.audio_codes = codes

    class FakeMimiModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = nn.Linear(1, 1)  # need at least one param

        def encode(self, audio, padding_mask=None):
            B, channels, T = audio.shape  # HF Mimi expects (B, channels, T)
            num_frames = T // 1920  # Mimi frame = 1920 samples at 24kHz
            codes = torch.randint(0, 2048, (B, 8, num_frames))
            return FakeEncoderOutput(codes)

    class FakeFeatureExtractor:
        pass

    monkeypatch.setattr(
        "transformers.MimiModel.from_pretrained",
        staticmethod(lambda *a, **kw: FakeMimiModel()),
    )
    monkeypatch.setattr(
        "transformers.AutoFeatureExtractor.from_pretrained",
        staticmethod(lambda *a, **kw: FakeFeatureExtractor()),
    )


class TestMimiEncoder:
    def test_properties(self, mock_mimi_model):
        from nemo.collections.speechlm2.modules.mimi_encoder import MimiEncoder

        enc = MimiEncoder(pretrained_model="fake")
        assert enc.token_equivalent_duration == 0.08
        assert enc.codebook_size == 2048
        assert enc.num_codebooks == 8
        assert enc.SAMPLE_RATE == 24000

    def test_encode_output_shape(self, mock_mimi_model):
        from nemo.collections.speechlm2.modules.mimi_encoder import MimiEncoder

        enc = MimiEncoder(pretrained_model="fake")
        audio = torch.randn(2, 48000)  # 2 seconds at 24kHz
        audio_lens = torch.tensor([48000, 24000])
        codes, code_lens = enc.encode(audio, audio_lens)
        assert codes.shape[0] == 2
        assert codes.shape[1] == 8  # all codebooks
        assert codes.ndim == 3

    def test_all_params_frozen(self, mock_mimi_model):
        from nemo.collections.speechlm2.modules.mimi_encoder import MimiEncoder

        enc = MimiEncoder(pretrained_model="fake")
        for p in enc.parameters():
            assert not p.requires_grad

    def test_fewer_codebooks(self, mock_mimi_model):
        from nemo.collections.speechlm2.modules.mimi_encoder import MimiEncoder

        enc = MimiEncoder(pretrained_model="fake", num_codebooks=4)
        assert enc.num_codebooks == 4
        audio = torch.randn(1, 24000)
        audio_lens = torch.tensor([24000])
        codes, code_lens = enc.encode(audio, audio_lens)
        assert codes.shape[1] == 4

    # ---- New tests below: verify real logic, not mock output ----

    def test_code_lens_computation(self, mock_mimi_model):
        """Verify code_lens formula: floor(audio_lens / (sr * frame_shift)), clamped to actual T."""
        from nemo.collections.speechlm2.modules.mimi_encoder import MimiEncoder

        enc = MimiEncoder(pretrained_model="fake")
        # 2.5 seconds → 2.5 / 0.08 = 31.25 → int = 31
        # But actual T from mock = 60000 // 1920 = 31 frames
        audio = torch.randn(2, 60000)
        audio_lens = torch.tensor([60000, 30000])  # 2.5s and 1.25s
        codes, code_lens = enc.encode(audio, audio_lens)

        expected_0 = min(int(60000 / (24000 * 0.08)), codes.shape[2])  # 31
        expected_1 = min(int(30000 / (24000 * 0.08)), codes.shape[2])  # 15
        assert code_lens[0].item() == expected_0
        assert code_lens[1].item() == expected_1

    def test_code_lens_clamped_to_actual_frames(self, mock_mimi_model):
        """When audio_lens would imply more frames than the encoder produces, code_lens is clamped."""
        from nemo.collections.speechlm2.modules.mimi_encoder import MimiEncoder

        enc = MimiEncoder(pretrained_model="fake")
        # audio is padded to 48000 (25 frames) but audio_lens says full length
        audio = torch.randn(1, 48000)
        audio_lens = torch.tensor([48000])
        codes, code_lens = enc.encode(audio, audio_lens)
        # code_lens should never exceed actual number of frames
        assert code_lens[0].item() <= codes.shape[2]

    def test_padding_mask_shape(self, mock_mimi_model):
        """Verify that encode passes correct (B, 1, T) padding mask to the HF model."""
        from nemo.collections.speechlm2.modules.mimi_encoder import MimiEncoder

        received_masks = []

        class FakeEncoderOutput:
            def __init__(self, codes):
                self.audio_codes = codes

        class InspectingMimiModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.dummy = nn.Linear(1, 1)

            def encode(self, audio, padding_mask=None):
                received_masks.append(padding_mask)
                B, channels, T = audio.shape
                return FakeEncoderOutput(torch.zeros(B, 8, T // 1920, dtype=torch.long))

        enc = MimiEncoder.__new__(MimiEncoder)
        nn.Module.__init__(enc)
        enc.model = InspectingMimiModel()
        enc.feature_extractor = None
        enc.num_codebooks = 8
        for p in enc.parameters():
            p.requires_grad = False

        audio = torch.randn(2, 48000)
        audio_lens = torch.tensor([48000, 24000])
        enc.encode(audio, audio_lens)

        mask = received_masks[0]
        assert mask.shape == (2, 1, 48000)
        # First sample: all valid
        assert mask[0, 0, :48000].sum() == 48000
        # Second sample: first 24000 valid, rest masked
        assert mask[1, 0, :24000].sum() == 24000
        assert mask[1, 0, 24000:].sum() == 0

    def test_encode_passes_3d_audio(self, mock_mimi_model):
        """Verify encode reshapes (B, T) audio to (B, 1, T) for HF model."""
        from nemo.collections.speechlm2.modules.mimi_encoder import MimiEncoder

        received_audio = []

        class FakeEncoderOutput:
            def __init__(self, codes):
                self.audio_codes = codes

        class InspectingMimiModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.dummy = nn.Linear(1, 1)

            def encode(self, audio, padding_mask=None):
                received_audio.append(audio)
                B, channels, T = audio.shape
                return FakeEncoderOutput(torch.zeros(B, 8, T // 1920, dtype=torch.long))

        enc = MimiEncoder.__new__(MimiEncoder)
        nn.Module.__init__(enc)
        enc.model = InspectingMimiModel()
        enc.feature_extractor = None
        enc.num_codebooks = 8
        for p in enc.parameters():
            p.requires_grad = False

        audio_2d = torch.randn(1, 19200)
        enc.encode(audio_2d, torch.tensor([19200]))
        assert received_audio[0].ndim == 3
        assert received_audio[0].shape[1] == 1  # channels dimension
