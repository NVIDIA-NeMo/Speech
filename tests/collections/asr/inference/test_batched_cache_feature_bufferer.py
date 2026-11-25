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

import pytest
import torch
from omegaconf import OmegaConf

from nemo.collections.asr.inference.streaming.buffering.cache_feature_bufferer import BatchedCacheFeatureBufferer
from nemo.collections.asr.inference.streaming.framing.request import Frame

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 3200  # 0.2 s, 20 feature frames at a 10 ms stride
PREPROCESSOR_CFG = OmegaConf.create(
    {
        "_target_": "nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor",
        "sample_rate": SAMPLE_RATE,
        "window_size": 0.025,
        "window_stride": 0.01,
        "features": 16,
        "n_fft": 512,
        "dither": 0.0,
        "pad_to": 0,
        "log": True,
    }
)


def _bufferer(num_slots=3):
    return BatchedCacheFeatureBufferer(
        num_slots=num_slots,
        sample_rate=SAMPLE_RATE,
        buffer_size_in_secs=0.4,
        chunk_size_in_secs=0.2,
        preprocessor_cfg=PREPROCESSOR_CFG,
        device=torch.device("cpu"),
    )


def _frame(stream_id, valid_samples=CHUNK_SAMPLES, is_last=False):
    return Frame(
        samples=torch.randn(CHUNK_SAMPLES),
        stream_id=stream_id,
        is_first=False,
        is_last=is_last,
        length=valid_samples if valid_samples != CHUNK_SAMPLES else -1,
    )


@pytest.mark.unit
def test_update_feature_buffer_shifts_only_the_slots_of_the_batch_whatever_their_order():
    bufferer = _bufferer(num_slots=5)
    torch.manual_seed(0)
    bufferer.feature_buffer = torch.randn_like(bufferer.feature_buffer)
    before = bufferer.feature_buffer.clone()
    chunk = torch.randn(3, bufferer.n_feat, bufferer.feature_chunk_len)
    slot_ids = [3, 0, 2]

    returned = bufferer._update_feature_buffer(torch.tensor(slot_ids), chunk)

    for i, slot in enumerate(slot_ids):
        expected = torch.cat([before[slot, :, bufferer.feature_chunk_len :], chunk[i]], dim=1)
        assert torch.equal(bufferer.feature_buffer[slot], expected)
        assert torch.equal(returned[i], expected)
    for slot in (1, 4):
        assert torch.equal(bufferer.feature_buffer[slot], before[slot])


@pytest.mark.unit
def test_update_feature_buffer_rejects_a_chunk_longer_than_the_buffer():
    bufferer = _bufferer()
    chunk = torch.zeros(1, bufferer.n_feat, bufferer.feature_buffer_len + 1)

    with pytest.raises(ValueError, match="longer than buffer"):
        bufferer._update_feature_buffer(torch.tensor([0]), chunk)


@pytest.mark.unit
def test_update_returns_batched_buffers_that_match_the_stored_state_and_tensor_paddings():
    bufferer = _bufferer()
    frames = [_frame(stream_id=7), _frame(stream_id=9, valid_samples=1600, is_last=True)]

    buffers, right_paddings = bufferer.update(frames)

    assert buffers.shape == (2, bufferer.n_feat, bufferer.feature_buffer_len)
    assert isinstance(right_paddings, torch.Tensor) and right_paddings.shape == (2,)
    # stream 9 is on its last frame, which frees its slot, so only stream 7 is still held by the bufferer
    assert torch.equal(buffers[0], bufferer.feature_buffer[bufferer.streamidx2slotidx[7]])
    # only the short last frame is padded
    assert right_paddings[0] == 0 and right_paddings[1] > 0
    assert 9 not in bufferer.streamidx2slotidx
