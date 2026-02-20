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

import os
import tempfile

import pytest
import soundfile as sf
import torch
from lhotse import CutSet, Recording, SupervisionSegment
from lhotse.testing.dummies import dummy_cut, dummy_recording

from nemo.collections.speechlm2.data.force_align import ForceAligner


def synthesize_speech(text: str, output_path: str, target_sample_rate: int = 16000):
    """Synthesize speech from text using NeMo FastPitch + HiFiGAN and save to wav file."""
    from nemo.collections.tts.models import FastPitchModel, HifiGanModel

    spec_gen = FastPitchModel.from_pretrained("tts_en_fastpitch")
    vocoder = HifiGanModel.from_pretrained("tts_en_hifigan")
    spec_gen.eval()
    vocoder.eval()

    parsed = spec_gen.parse(text)
    spectrogram = spec_gen.generate_spectrogram(tokens=parsed)
    audio = vocoder.convert_spectrogram_to_audio(spec=spectrogram)

    audio_np = audio.squeeze().cpu().detach().numpy()

    # FastPitch + HiFiGAN output at 22050 Hz; resample to target sample rate
    model_sample_rate = 22050
    if model_sample_rate != target_sample_rate:
        from scipy import signal

        num_samples = int(len(audio_np) * target_sample_rate / model_sample_rate)
        audio_np = signal.resample(audio_np, num_samples)

    sf.write(output_path, audio_np, target_sample_rate)
    print(f"Synthesized '{text[:50]}...' -> {len(audio_np)/target_sample_rate:.2f}s ({len(audio_np)} samples @ {target_sample_rate}Hz)")


@pytest.fixture(scope="module")
def force_aligner():
    """Create a ForceAligner instance with CPU device for testing"""
    aligner = ForceAligner(device='cpu', frame_length=0.08)
    return aligner


@pytest.fixture(scope="module")
def test_cutset_synthetic_audio():
    """Create a test cutset with on-the-fly TTS-generated audio at 16kHz"""
    texts = [
        'ten companies that let you teach english online without a degree',
        'yeah i really would like to see canada of course their border is not open right now but',
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        cuts = []
        for i, text in enumerate(texts):
            audio_path = os.path.join(tmpdir, f"{i}.wav")
            synthesize_speech(text, audio_path)

            rec = Recording.from_file(audio_path)
            cut = rec.to_cut()
            cut.supervisions = [
                SupervisionSegment(
                    id=f"{cut.id}-0",
                    recording_id=cut.recording_id,
                    start=0.0,
                    duration=rec.duration,
                    text=text,
                    speaker="user",
                ),
            ]
            cuts.append(cut)

        yield CutSet(cuts)


def test_force_align_synthetic_audio(force_aligner, test_cutset_synthetic_audio):
    """Test force alignment with TTS-generated synthetic audio."""
    import re

    # Store original texts before alignment
    original_texts = {}
    for cut in test_cutset_synthetic_audio:
        for sup in cut.supervisions:
            if sup.speaker == "user":
                original_texts[sup.id] = sup.text

    result_cuts = force_aligner.batch_force_align_user_audio(test_cutset_synthetic_audio, source_sample_rate=16000)

    assert len(result_cuts) == len(test_cutset_synthetic_audio)
    assert len(result_cuts) == 2

    for cut in result_cuts:
        user_supervisions = [s for s in cut.supervisions if s.speaker == "user"]
        assert len(user_supervisions) > 0

        for sup in user_supervisions:
            original_text = original_texts.get(sup.id, "")
            aligned_text = sup.text

            print(f"\n{'='*80}")
            print(f"Supervision ID: {sup.id}")
            print(f"{'='*80}")
            print(f"ORIGINAL TEXT:\n  {original_text}")
            print(f"\nALIGNED TEXT:\n  {aligned_text}")
            print(f"{'='*80}")

            assert "<|" in aligned_text, "Aligned text should contain timestamp markers"

            # Extract timestamp-word-timestamp patterns: <|start|> word <|end|>
            pattern = r'<\|(\d+)\|>\s+(\S+)\s+<\|(\d+)\|>'
            matches = re.findall(pattern, aligned_text)

            words_only = re.sub(r'<\|\d+\|>', '', aligned_text).split()
            words_only = [w for w in words_only if w]

            print(f"\nValidation: Found {len(matches)} timestamped words out of {len(words_only)} total words")

            assert len(matches) > 0, "Should have at least one timestamped word"
            assert len(matches) == len(
                words_only
            ), f"Every word should have timestamps. Found {len(matches)} timestamped words but {len(words_only)} total words"

            for start_frame, word, end_frame in matches:
                start_frame = int(start_frame)
                end_frame = int(end_frame)

                assert (
                    start_frame <= end_frame
                ), f"Start frame {start_frame} should be before or equal to end frame {end_frame} for word '{word}'"
                assert start_frame >= 0, f"Start frame should be non-negative for word '{word}'"
                assert end_frame >= 0, f"End frame should be non-negative for word '{word}'"


def test_force_align_no_user_supervisions(force_aligner):
    """Test with cutset containing no user supervisions"""
    cut = dummy_cut(0, recording=dummy_recording(0, with_data=True))
    cut.supervisions = [
        SupervisionSegment(
            id=cut.id,
            recording_id=cut.recording_id,
            start=0,
            duration=0.5,
            text='hello',
            speaker="assistant",
        ),
    ]
    cutset = CutSet([cut])

    result_cuts = force_aligner.batch_force_align_user_audio(cutset)

    assert len(result_cuts) == 1
    result_supervisions = list(result_cuts)[0].supervisions
    assert result_supervisions[0].text == 'hello'


def test_force_align_empty_cutset(force_aligner):
    """Test with empty cutset"""
    empty_cutset = CutSet.from_cuts([])
    result_cuts = force_aligner.batch_force_align_user_audio(empty_cutset)
    assert len(result_cuts) == 0


def test_strip_timestamps(force_aligner):
    """Test timestamp stripping utility"""
    text_with_timestamps = "<|10|> hello <|20|> world <|30|>"
    result = force_aligner._strip_timestamps(text_with_timestamps)
    assert result == "hello world"
    assert "<|" not in result

    text_without_timestamps = "hello world"
    result = force_aligner._strip_timestamps(text_without_timestamps)
    assert result == "hello world"


def test_normalize_transcript(force_aligner):
    """Test transcript normalization"""
    assert force_aligner._normalize_transcript("Hello World!") == "hello world"
    assert force_aligner._normalize_transcript("don't worry") == "don't worry"
    assert force_aligner._normalize_transcript("test123") == "test"
    assert force_aligner._normalize_transcript("A,B.C!D?E") == "a b c d e"
