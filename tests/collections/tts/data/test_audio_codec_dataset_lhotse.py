# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from pathlib import Path

import lhotse
import numpy as np
import pytest
import soundfile as sf
import torch
from lhotse import CutSet

from nemo.collections.tts.data.audio_codec_dataset_lhotse import AudioCodecLhotseDataset


SOURCE_SAMPLE_RATE = 16000
TARGET_SAMPLE_RATE = 24000
DEFAULT_DURATION = 1.0


def _write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    """Write a mono wav file."""
    sf.write(str(path), samples, sample_rate, format="WAV")


def _make_cut(
    tmp_path: Path,
    cut_id: str,
    duration: float,
    sample_rate: int,
    tone_frequency: float,
) -> lhotse.MonoCut:
    """Create a MonoCut whose `target_audio` field points to a wav file on disk."""
    num_samples = int(duration * sample_rate)
    t = np.arange(num_samples, dtype=np.float32) / sample_rate
    # Two sine waves at different frequencies to make sure we're reading the right field.
    f_main = tone_frequency - 100.0
    f_target = tone_frequency
    tone_target = (0.5 * np.sin(2.0 * np.pi * f_target * t)).astype(np.float32)
    tone_main = (0.5 * np.sin(2.0 * np.pi * f_main * t)).astype(np.float32)
    main_path = tmp_path / f"{cut_id}_main.wav"
    target_path = tmp_path / f"{cut_id}_target.wav"
    _write_wav(main_path, tone_main, sample_rate)
    _write_wav(target_path, tone_target, sample_rate)

    cut = lhotse.MonoCut(
        id=cut_id,
        start=0.0,
        duration=duration,
        channel=0,
        recording=lhotse.Recording.from_file(main_path, recording_id=f"{cut_id}_main_rec"),
    )
    cut.custom = {
        "target_audio": lhotse.Recording.from_file(target_path, recording_id=f"{cut_id}_target_rec"),
        "target_tone_frequency": f_target,
    }
    return cut


@pytest.fixture()
def cutset(tmp_path) -> CutSet:
    """A small CutSet of 3 cuts with `target_audio` recordings on disk."""
    cuts = [
        _make_cut(
            tmp_path,
            f"cut_{i}",
            sample_rate=SOURCE_SAMPLE_RATE,
            duration=DEFAULT_DURATION,
            tone_frequency=(i + 1) * 440.0,
        )
        for i in range(3)
    ]
    return CutSet.from_cuts(cuts)


@pytest.fixture()
def dataset() -> AudioCodecLhotseDataset:
    return AudioCodecLhotseDataset(
        sample_rate=TARGET_SAMPLE_RATE,
        min_samples_for_sanity=DEFAULT_DURATION * TARGET_SAMPLE_RATE,
    )


class TestAudioCodecLhotseDataset:
    @pytest.mark.unit
    def test_init(self):
        ds = AudioCodecLhotseDataset(sample_rate=22050, min_samples_for_sanity=512)
        assert ds.sample_rate == 22050
        assert ds.min_samples_for_sanity == 512

    @pytest.mark.unit
    def test_getitem_returns_expected_keys_and_shapes(self, dataset, cutset):
        batch = dataset[cutset]

        assert set(batch.keys()) == {"audio", "audio_lens"}
        audio = batch["audio"]
        audio_lens = batch["audio_lens"]

        assert isinstance(audio, torch.Tensor)
        assert isinstance(audio_lens, torch.Tensor)

        batch_size = len(cutset)
        assert audio.shape[0] == batch_size
        assert audio_lens.shape == (batch_size,)
        # Number of samples should match the (resampled) target length, up to a fractional sample
        expected_n_samples = int(round(DEFAULT_DURATION * TARGET_SAMPLE_RATE))
        assert abs(audio_lens.max().item() - expected_n_samples) <= 1
        assert audio.shape[1] == audio_lens.max().item()

    @pytest.mark.unit
    def test_getitem_resampling_preserves_frequency(self, dataset, cutset):
        # Each cut contains a tone at a different frequency. After resampling from
        # SOURCE_SAMPLE_RATE to TARGET_SAMPLE_RATE the dominant FFT bin should still
        # correspond to the original frequency.
        batch = dataset[cutset]
        audio = batch["audio"]
        audio_lens = batch["audio_lens"]

        for i in range(audio.shape[0]):
            print(i)
            n = int(audio_lens[i].item())
            signal = audio[i, :n].numpy()
            magnitude_spectrum = np.abs(np.fft.rfft(signal))
            peak_bin = int(np.argmax(magnitude_spectrum))
            peak_freq_hz = peak_bin * TARGET_SAMPLE_RATE / n
            # FFT bin width is TARGET_SAMPLE_RATE / n; allow ~1 bin of tolerance.
            bin_width_hz = TARGET_SAMPLE_RATE / n
            assert abs(peak_freq_hz - cutset[i].target_tone_frequency) <= bin_width_hz
