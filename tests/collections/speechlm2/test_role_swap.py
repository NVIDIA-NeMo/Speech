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

from io import BytesIO
from unittest.mock import MagicMock, PropertyMock

import numpy as np
import pytest
import soundfile as sf
from lhotse import AudioSource, MonoCut, Recording, SupervisionSegment

from nemo.collections.speechlm2.data import DuplexSTTDataset


def _make_in_memory_recording(recording_id, duration, sampling_rate, signal_freq=440.0):
    """Create a lhotse Recording backed by in-memory WAV bytes with a sine wave."""
    num_samples = int(duration * sampling_rate)
    t = np.linspace(0, duration, num_samples, endpoint=False, dtype=np.float32)
    audio = 0.5 * np.sin(2 * np.pi * signal_freq * t)
    buf = BytesIO()
    sf.write(buf, audio, sampling_rate, format='wav')
    buf.seek(0)
    return Recording(
        id=recording_id,
        sampling_rate=sampling_rate,
        num_samples=num_samples,
        duration=duration,
        sources=[AudioSource(type="memory", channels=[0], source=buf.getvalue())],
    )


def test_create_role_swapped_cut():
    """
    Test _create_role_swapped_cut with a 6-turn conversation.

    Original:
        Turn 1: User       0.0-1.0  "hello"
        Turn 2: Assistant  1.5-2.5  "hi there"
        Turn 3: User       3.0-4.0  "how are you"
        Turn 4: Assistant  4.5-5.5  "doing well"
        Turn 5: User       6.0-7.0  "great"
        Turn 6: Assistant  7.5-8.5  "bye"

    After swap + filter (remove first Assistant, last User):
        User       0.0-1.0  "hi there"     (originally Assistant turn 2)
        Assistant  1.5-2.5  "how are you"  (originally User turn 3)
        User       3.0-4.0  "doing well"   (originally Assistant turn 4)
        Assistant  4.5-5.5  "great"        (originally User turn 5)
    """
    tokenizer = MagicMock()
    type(tokenizer).bos = PropertyMock(return_value=1)
    type(tokenizer).eos = PropertyMock(return_value=2)
    type(tokenizer).pad = PropertyMock(return_value=0)
    type(tokenizer).pad_id = PropertyMock(return_value=0)
    type(tokenizer).unk_id = PropertyMock(return_value=None)
    tokenizer.text_to_ids = MagicMock(return_value=[1])

    dataset = DuplexSTTDataset(
        tokenizer=tokenizer,
        frame_length=0.08,
        source_sample_rate=16000,
        input_roles=["User"],
        output_roles=["Assistant"],
        aug_by_swap_role=True,
        cfg={"early_interruption_prob": 0.0},
        model_cfg={"predict_user_text": False, "force_align_user_text": False},
    )

    # Build a 6-turn cut with source (440Hz) and target (880Hz) audio
    sampling_rate = 16000
    total_duration = 9.0
    source_rec = _make_in_memory_recording("src", total_duration, sampling_rate, signal_freq=440.0)
    target_rec = _make_in_memory_recording("tgt", total_duration, sampling_rate, signal_freq=880.0)

    supervisions = [
        SupervisionSegment(id="s1", recording_id="src", start=0.0, duration=1.0, speaker="User", text="hello"),
        SupervisionSegment(id="s2", recording_id="src", start=1.5, duration=1.0, speaker="Assistant", text="hi there"),
        SupervisionSegment(id="s3", recording_id="src", start=3.0, duration=1.0, speaker="User", text="how are you"),
        SupervisionSegment(
            id="s4", recording_id="src", start=4.5, duration=1.0, speaker="Assistant", text="doing well"
        ),
        SupervisionSegment(id="s5", recording_id="src", start=6.0, duration=1.0, speaker="User", text="great"),
        SupervisionSegment(id="s6", recording_id="src", start=7.5, duration=1.0, speaker="Assistant", text="bye"),
    ]

    cut = MonoCut(
        id="test_cut",
        start=0,
        duration=total_duration,
        channel=0,
        supervisions=supervisions,
        recording=source_rec,
        custom={'target_audio': target_rec, 'total_turns': 6},
    )

    swapped_cut = dataset._create_role_swapped_cut(cut)
    assert swapped_cut is not None

    sups = swapped_cut.supervisions

    # 4 turns remain after removing first Assistant and last User
    assert len(sups) == 4

    # Roles alternate User, Assistant, User, Assistant
    assert [s.speaker for s in sups] == ["User", "Assistant", "User", "Assistant"]

    # Texts match the kept turns
    assert [s.text for s in sups] == ["hi there", "how are you", "doing well", "great"]

    # Timestamps adjusted (first_remaining_start=1.5 subtracted), all start at 0
    assert abs(sups[0].start - 0.0) < 1e-6
    assert abs(sups[1].start - 1.5) < 1e-6
    assert abs(sups[2].start - 3.0) < 1e-6
    assert abs(sups[3].start - 4.5) < 1e-6

    # Source audio: User regions have signal (from target_audio), Assistant regions are silent
    audio = swapped_cut.recording.to_cut().load_audio().squeeze()
    for s in sups:
        start_sample = int(s.start * sampling_rate)
        end_sample = int((s.start + s.duration) * sampling_rate)
        rms = np.sqrt(np.mean(audio[start_sample:end_sample] ** 2))
        if s.speaker == 'User':
            assert rms > 0.1, f"User turn '{s.text}' should have audio from target_audio, RMS={rms:.4f}"
        else:
            assert rms < 0.01, f"Assistant turn '{s.text}' should be silent in STT source, RMS={rms:.4f}"

    # Verify User audio is similar to target audio (880Hz)
    # Load the original target audio for comparison
    target_audio = target_rec.to_cut().load_audio().squeeze()

    for s in sups:
        if s.speaker == 'User':
            # User turns should be from target_audio (originally Assistant turns)
            # The original supervisions had these start times: 1.5 (hi there), 4.5 (doing well)
            # which map to indices 0 and 2 in sups (User turns after swap)
            start_sample = int(s.start * sampling_rate)
            end_sample = int((s.start + s.duration) * sampling_rate)
            segment = audio[start_sample:end_sample]
            idx = list(sups).index(s)

            original_start_times = {0: 1.5, 2: 4.5}  # map swapped index to original time
            if idx in original_start_times:
                orig_start = original_start_times[idx]
                orig_start_sample = int(orig_start * sampling_rate)
                orig_end_sample = int((orig_start + s.duration) * sampling_rate)
                target_segment = target_audio[orig_start_sample:orig_end_sample]

                # Verify lengths match exactly
                len_diff = abs(len(segment) - len(target_segment))
                assert len_diff == 0, (
                    f"User turn '{s.text}' length must match exactly: "
                    f"segment={len(segment)} samples, "
                    f"target_segment={len(target_segment)} samples, "
                    f"diff={len_diff}"
                )

                # Compute correlation between user audio and target audio
                # Waveforms should be nearly identical
                correlation = np.corrcoef(segment, target_segment)[0, 1]
                assert correlation > 0.999, (
                    f"User turn '{s.text}' should have nearly identical waveform to target audio. "
                    f"Correlation={correlation:.6f}, expected > 0.999"
                )

    # Metadata
    assert swapped_cut.custom['role_swapped'] is True
    assert swapped_cut.custom['total_turns'] == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
