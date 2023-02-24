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

from nemo.collections.tts.parts.utils.tts_dataset_utils import normalize_volume


class TestDataUtils:
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_normalize_volume(self):
        input_audio = np.array([0.0, 0.1, 0.3, 0.5])
        expected_output = np.array([0.0, 0.18, 0.54, 0.9])

        output_audio = normalize_volume(audio=input_audio, volume_level=0.9)

        np.testing.assert_array_almost_equal(output_audio, expected_output)

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_normalize_volume_negative_peak(self):
        input_audio = np.array([0.0, 0.1, -0.3, -1.0, 0.5])
        expected_output = np.array([0.0, 0.05, -0.15, -0.5, 0.25])

        output_audio = normalize_volume(audio=input_audio, volume_level=0.5)

        np.testing.assert_array_almost_equal(output_audio, expected_output)

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_normalize_volume_zero(self):
        input_audio = np.array([0.0, 0.1, 0.3, 0.5])
        expected_output = np.array([0.0, 0.0, 0.0, 0.0])

        output_audio = normalize_volume(audio=input_audio, volume_level=0.0)

        np.testing.assert_array_almost_equal(output_audio, expected_output)

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_normalize_volume_max(self):
        input_audio = np.array([0.0, 0.1, 0.3, 0.5])
        expected_output = np.array([0.0, 0.2, 0.6, 1.0])

        output_audio = normalize_volume(audio=input_audio, volume_level=1.0)

        np.testing.assert_array_almost_equal(output_audio, expected_output)

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_normalize_volume_zeros(self):
        input_audio = np.array([0.0, 0.0, 0.0])

        output_audio = normalize_volume(audio=input_audio, volume_level=0.5)

        np.testing.assert_array_almost_equal(input_audio, input_audio)

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_normalize_volume_out_of_range(self):
        input_audio = np.array([0.0, 0.1, 0.3, 0.5])
        with pytest.raises(ValueError, match="Volume must be in range"):
            normalize_volume(audio=input_audio, volume_level=2.0)
