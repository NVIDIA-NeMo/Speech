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
from pathlib import Path

import multistorageclient as msc

from nemo.lightning.resume import AutoResume


def test_auto_resume_get_weights_path():
    auto_resume = AutoResume()
    assert auto_resume.get_weights_path(Path("test/checkpoints")) == Path("test/checkpoints/weights")
    assert auto_resume.get_weights_path(msc.Path("msc://default/tmp/test/checkpoints")) == msc.Path(
        "msc://default/tmp/test/checkpoints/weights"
    )


def test_auto_resume_get_context_path():
    auto_resume = AutoResume()

    auto_resume.resume_if_exists = False
    assert auto_resume.get_context_path() is None

    auto_resume.resume_if_exists = True
    assert auto_resume.get_context_path() is None

    # test with filesystem path
    with tempfile.TemporaryDirectory() as tmpdir:
        os.makedirs(os.path.join(tmpdir, "checkpoints", "step=10-epoch=0-last", "weights"))
        os.makedirs(os.path.join(tmpdir, "checkpoints", "step=10-epoch=0-last", "context"))
        auto_resume.resume_from_directory = os.path.join(tmpdir, "checkpoints")
        assert str(auto_resume.get_context_path()) == os.path.join(
            tmpdir, "checkpoints", "step=10-epoch=0-last", "context"
        )

    # test with MSC URL
    with tempfile.TemporaryDirectory() as tmpdir:
        os.makedirs(os.path.join(tmpdir, "checkpoints", "step=10-epoch=0-last", "weights"))
        os.makedirs(os.path.join(tmpdir, "checkpoints", "step=10-epoch=0-last", "context"))
        auto_resume.resume_from_directory = f"msc://default{tmpdir}/checkpoints"
        path = auto_resume.get_context_path()
        assert isinstance(path, msc.Path)
        assert str(path) == os.path.join(tmpdir, "checkpoints", "step=10-epoch=0-last", "context")


def test_auto_resume_get_trainer_ckpt_path():
    auto_resume = AutoResume()

    auto_resume.resume_if_exists = False
    assert auto_resume.get_trainer_ckpt_path() is None

    auto_resume.resume_if_exists = True
    assert auto_resume.get_trainer_ckpt_path() is None

    # test with filesystem path
    with tempfile.TemporaryDirectory() as tmpdir:
        os.makedirs(os.path.join(tmpdir, "checkpoints", "step=10-epoch=0-last", "weights"))
        os.makedirs(os.path.join(tmpdir, "checkpoints", "step=10-epoch=0-last", "context"))
        auto_resume.resume_from_path = os.path.join(tmpdir, "checkpoints", "step=10-epoch=0-last")
        assert str(auto_resume.get_trainer_ckpt_path()) == os.path.join(
            tmpdir, "checkpoints", "step=10-epoch=0-last", "weights"
        )

    # test with MSC URL
    with tempfile.TemporaryDirectory() as tmpdir:
        os.makedirs(os.path.join(tmpdir, "checkpoints", "step=10-epoch=0-last", "weights"))
        os.makedirs(os.path.join(tmpdir, "checkpoints", "step=10-epoch=0-last", "context"))
        auto_resume.resume_from_path = f"msc://default{tmpdir}/checkpoints/step=10-epoch=0-last"
        path = auto_resume.get_trainer_ckpt_path()
        assert isinstance(path, msc.Path)
        assert str(path) == os.path.join(tmpdir, "checkpoints", "step=10-epoch=0-last", "weights")
