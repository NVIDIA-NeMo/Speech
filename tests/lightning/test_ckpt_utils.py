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

from pathlib import Path

import pytest

from nemo.lightning.ckpt_utils import ckpt_to_context_subdir, ckpt_to_dir, idempotent_path_append
from nemo.lightning.io.pl import ckpt_to_weights_subdir


@pytest.mark.parametrize("base_dir", [Path("test/path"), "test/path", "msc://default/tmp/test/path"])
def test_idempotent_path_append_path_no_suffix(base_dir):
    suffix = "new_suffix"
    result = idempotent_path_append(base_dir, suffix)
    assert str(result).endswith("test/path/new_suffix")


@pytest.mark.parametrize(
    "filepath", [Path("test/checkpoints"), "test/checkpoints", "msc://default/tmp/test/checkpoints"]
)
def test_ckpt_to_context_subdir(filepath):
    result = ckpt_to_context_subdir(filepath)
    assert str(result).endswith("test/checkpoints/context")


@pytest.mark.parametrize(
    "filepath", [Path("test/checkpoints"), "test/checkpoints", "msc://default/tmp/test/checkpoints"]
)
def test_ckpt_to_dir(filepath):
    result = ckpt_to_dir(filepath)
    assert str(result).endswith("test/checkpoints")


@pytest.mark.parametrize(
    "filepath", [Path("test/checkpoints"), "test/checkpoints", "msc://default/tmp/test/checkpoints"]
)
def test_ckpt_to_weights_subdir(filepath):
    result = ckpt_to_weights_subdir(filepath, is_saving=True)
    assert str(result).endswith("test/checkpoints/weights")
