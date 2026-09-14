# SPDX-FileCopyrightText: Copyright (c) 2020, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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


from os.path import exists, join

import pytest


class TestDataDir:
    @pytest.mark.unit
    def test_test_data_dir(self, test_data_dir):
        """Just a dummy tests showing how to use the test_data_dir fixture."""
        # test_data_dir contains the absolute path to nemo -> tests/.data
        assert exists(test_data_dir)
        assert exists(join(test_data_dir, "test_data.tar.gz"))
