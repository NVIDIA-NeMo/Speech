# Copyright (c) 2020-2025, NVIDIA CORPORATION.
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
from nemo.collections.diffusion.data.diffusion_taskencoder import cook_raw_images


class TestTaskEncoder:
    @pytest.mark.unit
    def test_cook_raw_images(self):
        sample = {"jpg": "original_image_data", "png": "control_image_data", "txt": "raw_text_data"}

        processed_sample = cook_raw_images(sample)

        assert "images" in processed_sample
        assert "hint" in processed_sample
        assert "txt" in processed_sample

        assert processed_sample["images"] == sample["jpg"]
        assert processed_sample["hint"] == sample["png"]
        assert processed_sample["txt"] == sample["txt"]
