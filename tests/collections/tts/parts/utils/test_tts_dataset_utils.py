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

from pathlib import Path

import pytest

from nemo.collections.tts.parts.utils.tts_dataset_utils import get_abs_rel_paths, get_audio_filepaths


class TestTTSDatasetUtils:
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_get_abs_rel_paths_input_abs(self):
        input_path = Path("/home/data/audio/test")
        base_path = Path("/home/data")

        abs_path, rel_path = get_abs_rel_paths(input_path=input_path, base_path=base_path)

        assert abs_path == input_path
        assert rel_path == Path("audio/test")

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_get_abs_rel_paths_input_rel(self):
        input_path = Path("audio/test")
        base_path = Path("/home/data")

        abs_path, rel_path = get_abs_rel_paths(input_path=input_path, base_path=base_path)

        assert abs_path == Path("/home/data/audio/test")
        assert rel_path == input_path

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_get_audio_paths(self):
        audio_dir = Path("/home/audio")
        audio_rel_path = Path("examples/example.wav")
        manifest_entry = {"audio_filepath": str(audio_rel_path)}

        abs_path, rel_path = get_audio_filepaths(manifest_entry=manifest_entry, audio_dir=audio_dir)

        assert abs_path == Path("/home/audio/examples/example.wav")
        assert rel_path == audio_rel_path
