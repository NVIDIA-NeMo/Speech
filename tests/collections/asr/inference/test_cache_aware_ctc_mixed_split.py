# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from nemo.collections.asr.inference.model_wrappers.cache_aware_ctc_inference_wrapper import (
    CacheAwareCTCInferenceWrapper,
)
from nemo.collections.asr.inference.pipelines.cache_aware_ctc_pipeline import CacheAwareCTCPipeline

LEFT_CONTEXT = 70


def _pipeline(use_cache, num_left_context):
    wrapper = object.__new__(CacheAwareCTCInferenceWrapper)
    wrapper.asr_model = SimpleNamespace(encoder=SimpleNamespace(streaming_cfg=SimpleNamespace(valid_out_len=14)))
    pipeline = object.__new__(CacheAwareCTCPipeline)
    pipeline.asr_model = wrapper
    pipeline.use_cache = use_cache
    pipeline.drop_left_context = num_left_context or None
    pipeline.valid_out_len = 13
    pipeline.return_tail_result = True
    return pipeline


@pytest.mark.unit
@pytest.mark.parametrize(
    "use_cache, left_context, frames, expected_head, expected_tail",
    [
        # cached: the encoder caps its output at 14 frames, no left context in the buffer
        (True, 0, 14, 13, (13, 14)),
        # uncached: no cap, 70 left context frames are dropped and the 16 frames that remain are all kept
        (False, LEFT_CONTEXT, 16, 13, (13, 16)),
        # uncached without left context: no cap either
        (False, 0, 16, 13, (13, 16)),
    ],
)
def test_non_last_requests_of_a_mixed_batch_keep_the_frames_a_step_of_their_own_would(
    use_cache, left_context, frames, expected_head, expected_tail
):
    pipeline = _pipeline(use_cache, left_context)
    log_probs = torch.arange(2 * frames).reshape(2, frames, 1).float()

    heads, tails = pipeline.split_mixed_log_probs(log_probs, [True, False])

    assert heads[0].shape[0] == frames  # the last request keeps everything
    assert heads[1].shape[0] == expected_head
    assert torch.equal(heads[1], log_probs[1, :expected_head])
    assert torch.equal(tails[1], log_probs[1, expected_tail[0] : expected_tail[1]])
    assert tails[0] is None
