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

from unittest.mock import patch

import pytest

from nemo.utils.file_utils import robust_copy


class TestRobustCopy:
    @pytest.mark.unit
    def test_robust_copy_success(self, tmp_path):
        """Tests that robust_copy uses shutil.copy2 and does not fall back."""
        src_file = tmp_path / "source.txt"
        src_file.write_text("test content")
        dest_file = tmp_path / "dest.txt"

        with (
            patch('nemo.utils.file_utils.shutil.copy2') as mock_copy2,
            patch('nemo.utils.file_utils.shutil.copy') as mock_copy,
        ):
            robust_copy(src_file, dest_file)
            mock_copy2.assert_called_once_with(src_file, dest_file)
            mock_copy.assert_not_called()

    @pytest.mark.unit
    def test_robust_copy_fallback(self, tmp_path):
        """Tests that robust_copy falls back to shutil.copy if shutil.copy2 fails."""
        src_file = tmp_path / "source.txt"
        src_file.write_text("test content")
        dest_file = tmp_path / "dest.txt"

        with (
            patch('nemo.utils.file_utils.shutil.copy2', side_effect=PermissionError("copy2 fails")) as mock_copy2,
            patch('nemo.utils.file_utils.shutil.copy') as mock_copy,
        ):
            robust_copy(src_file, dest_file)
            mock_copy2.assert_called_once_with(src_file, dest_file)
            mock_copy.assert_called_once_with(src_file, dest_file)
