# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import numpy as np
import pytest

from nemo.collections.tts.parts.preprocessing.audio_trimming import (
    get_start_and_end_of_speech_frames,
    pad_sample_indices,
)


class TestAudioTrimming:
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_get_start_and_end_of_speech_frames_frames(self):
        # First speech frame is index 2 (inclusive) and last one is index 8 (exclusive).
        is_speech = np.array([True, False, True, True, False, True, True, True, False, True, False])
        speech_frame_threshold = 2

        start_frame, end_frame = get_start_and_end_of_speech_frames(
            is_speech=is_speech, speech_frame_threshold=speech_frame_threshold
        )

        assert start_frame == 2
        assert end_frame == 8

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_get_start_and_end_of_speech_frames_not_frames_found(self):
        is_speech = np.array([False, True, True, False])
        speech_frame_threshold = 3

        start_frame, end_frame = get_start_and_end_of_speech_frames(
            is_speech=is_speech, speech_frame_threshold=speech_frame_threshold, audio_id="test"
        )

        assert start_frame == 0
        assert end_frame == 0

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_pad_sample_indices(self):
        start_sample, end_sample = pad_sample_indices(
            start_sample=1000, end_sample=2000, max_sample=5000, sample_rate=100, pad_seconds=3
        )
        assert start_sample == 700
        assert end_sample == 2300

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_pad_sample_indices_boundaries(self):
        start_sample, end_sample = pad_sample_indices(
            start_sample=100, end_sample=1000, max_sample=1150, sample_rate=100, pad_seconds=2
        )
        assert start_sample == 0
        assert end_sample == 1150
