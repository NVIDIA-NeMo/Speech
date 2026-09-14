# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
"""Focused regression tests for low-precision packed normalization statistics."""

from __future__ import annotations

import pytest
import torch

from nemo.collections.asr.parts.packed_sequence import pack_encoder_output
from nemo.collections.asr.parts.preprocessing.features import normalize_batch, normalize_packed_batch


def make_features(device: str, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(2746317213)
    lengths = torch.tensor([1200, 2500, 4500], dtype=torch.int64, device=device)
    time = torch.arange(4500, dtype=torch.float32, device=device)
    channels = torch.arange(128, dtype=torch.float32, device=device).unsqueeze(1)
    features = torch.zeros((3, 128, 4500), dtype=dtype, device=device)
    for index, length in enumerate(lengths.tolist()):
        row = (
            -12.0
            + channels * 0.015
            + 2.0 * torch.sin(time.unsqueeze(0) * (0.003 + channels * 0.00001) + index)
            + 1.5 * torch.randn((128, 4500), device=device)
        )
        features[index, :, :length] = row[:, :length].to(dtype)
    return features, lengths


@pytest.mark.parametrize("normalize_type", ["per_feature", "all_features"])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")),
    ],
)
def test_low_precision_packed_statistics_match_fp32_reference(device, normalize_type):
    features, lengths = make_features(device, torch.bfloat16)
    packed = pack_encoder_output(features.transpose(1, 2), lengths)

    actual = normalize_packed_batch(packed, normalize_type)
    expected, _, _ = normalize_batch(features.float(), lengths, normalize_type)
    expected = pack_encoder_output(expected.transpose(1, 2), lengths).data.to(torch.bfloat16)

    assert actual.data.dtype == torch.bfloat16
    torch.testing.assert_close(actual.data, expected, rtol=1e-2, atol=1.5625e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_legacy_bf16_segmented_statistics_are_not_silently_equivalent_to_fp32_fix():
    features, lengths = make_features("cuda", torch.bfloat16)
    packed = pack_encoder_output(features.transpose(1, 2), lengths)
    stable = normalize_packed_batch(packed, "per_feature")
    sequence_ids = torch.repeat_interleave(torch.arange(3, device="cuda"), lengths)
    denominator = lengths.unsqueeze(1)
    mean = torch.segment_reduce(packed.data, "sum", lengths=lengths, unsafe=True) / denominator
    centered = packed.data - mean[sequence_ids]
    variance = torch.segment_reduce(centered.square(), "sum", lengths=lengths, unsafe=True) / (denominator - 1)
    std = torch.sqrt(variance).masked_fill(variance.isnan(), 0.0) + 1e-5
    legacy = centered / std[sequence_ids]

    cosine = torch.nn.functional.cosine_similarity(stable.data.float().flatten(), legacy.float().flatten(), dim=0)
    assert cosine < 0.9
