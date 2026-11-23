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

from types import SimpleNamespace

import pytest
import torch

from nemo.collections.asr.inference.model_wrappers.cache_aware_rnnt_inference_wrapper import (
    CacheAwareRNNTInferenceWrapper,
)
from nemo.collections.asr.inference.utils.context_manager import CacheAwareContext

NUM_FRAMES = 14


class _FakeEncoder:
    """Encoder stub: returns NUM_FRAMES output frames per stream and records the keep_all_outputs it was given."""

    def __init__(self, valid_out_len):
        self.streaming_cfg = SimpleNamespace(valid_out_len=valid_out_len)
        self.keep_all_outputs = None

    def cache_aware_stream_step(self, processed_signal, processed_signal_length, keep_all_outputs, **kwargs):
        self.keep_all_outputs = keep_all_outputs
        batch = processed_signal.shape[0]
        encoded = torch.zeros(batch, 4, NUM_FRAMES)
        return encoded, torch.full((batch,), NUM_FRAMES), None, None, None


def _encoder_step(encoder_valid_out_len, keep, valid_out_len=None):
    wrapper = object.__new__(CacheAwareRNNTInferenceWrapper)
    encoder = _FakeEncoder(encoder_valid_out_len)
    wrapper.asr_model = SimpleNamespace(encoder=encoder)
    _, encoded_len, _ = wrapper.encoder_step(
        processed_signal=torch.zeros(len(keep), 8, 20),
        processed_signal_length=torch.full((len(keep),), 20),
        context=CacheAwareContext(),
        drop_extra_pre_encoded=None,
        keep_all_outputs=False,
        keep_all_outputs_mask=torch.tensor(keep),
        valid_out_len=valid_out_len,
    )
    return encoded_len.tolist(), encoder


@pytest.mark.unit
@pytest.mark.parametrize("valid_out_len", [None, 1])
def test_mixed_batch_trims_the_right_context_of_the_streams_that_are_not_last(valid_out_len):
    """With the cache on, the pipeline passes no valid_out_len: the encoder's own one must still trim the streams."""
    lens, encoder = _encoder_step(encoder_valid_out_len=1, keep=[True, False, False], valid_out_len=valid_out_len)

    assert lens == [NUM_FRAMES, 1, 1]
    assert encoder.keep_all_outputs is True


@pytest.mark.unit
def test_mixed_batch_keeps_everything_when_the_encoder_has_no_right_context_to_drop():
    lens, _ = _encoder_step(encoder_valid_out_len=0, keep=[True, False])

    assert lens == [NUM_FRAMES, NUM_FRAMES]
