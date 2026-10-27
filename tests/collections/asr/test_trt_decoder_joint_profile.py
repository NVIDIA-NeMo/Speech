# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

"""Check decoder-joint profiles without building an engine or allocating a GPU."""

from pathlib import Path

import onnx
import pytest
from examples.asr.export.build_trt_decoder_joint import decoder_joint_profile
from onnx import TensorProto, helper


def _model(tmp_path: Path, batch: str | int = "batch", hidden: str | int = 640) -> Path:
    inputs = [
        helper.make_tensor_value_info("encoder_outputs", TensorProto.FLOAT, [batch, 1024, "frames"]),
        helper.make_tensor_value_info("targets", TensorProto.INT64, [batch, "labels"]),
        helper.make_tensor_value_info("target_length", TensorProto.INT64, [batch]),
        helper.make_tensor_value_info("input_states_1", TensorProto.FLOAT, [2, batch, hidden]),
        helper.make_tensor_value_info("input_states_2", TensorProto.FLOAT, [2, batch, hidden]),
    ]
    model = helper.make_model(helper.make_graph([], "profile", inputs, []))
    path = tmp_path / "decoder.onnx"
    onnx.save(model, path)
    return path


@pytest.mark.unit
def test_profile_keeps_recurrent_batch_on_axis_one(tmp_path: Path) -> None:
    profile = decoder_joint_profile(_model(tmp_path), 4, 16)
    assert profile["encoder_outputs"] == [(1, 1024, 1), (4, 1024, 1), (16, 1024, 1)]
    assert profile["targets"] == [(1, 1), (4, 1), (16, 1)]
    assert profile["input_states_1"] == [(2, 1, 640), (2, 4, 640), (2, 16, 640)]
    assert profile["target_length"] == [(1,), (4,), (16,)]


@pytest.mark.unit
def test_reject_static_batch_conflict(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="conflicts with static dimension"):
        decoder_joint_profile(_model(tmp_path, batch=1), 4, 16)


@pytest.mark.unit
def test_reject_unknown_hidden_dimension(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="static non-batch dimension"):
        decoder_joint_profile(_model(tmp_path, hidden="hidden"), 4, 16)


@pytest.mark.unit
@pytest.mark.parametrize("opt_batch,max_batch", [(0, 16), (16, 4)])
def test_reject_invalid_batch_bounds(tmp_path: Path, opt_batch: int, max_batch: int) -> None:
    with pytest.raises(ValueError, match="opt_batch"):
        decoder_joint_profile(_model(tmp_path), opt_batch, max_batch)
