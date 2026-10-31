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

import copy

import pytest
import torch

from tests.collections.speechlm2.test_perception_packed_sequence import _make_perception


class _SpecAugment(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, input_spec, length):
        self.calls += 1
        return input_spec + 0.125


class _NoPackedEncoder(torch.nn.Module):
    def forward(self, audio_signal, length):
        return audio_signal, length


@pytest.mark.parametrize("unsupported", ["rote", "multilayer", "marker", "method"])
def test_perception_packed_capability_rejects_each_unsupported_stack(unsupported):
    perception = _make_perception()
    if unsupported == "rote":
        perception.rote = torch.nn.Identity()
    elif unsupported == "multilayer":
        wrapper = torch.nn.Module()
        wrapper.encoder = perception.encoder
        perception.encoder_multilayer = wrapper
    elif unsupported == "marker":
        perception.encoder.supports_sequence_packed_output = False
    else:
        perception._modules["encoder"] = _NoPackedEncoder()

    assert not perception.supports_sequence_packed_output
    with pytest.raises(ValueError, match="Packed encoder sequences"):
        perception.forward_sequence_packed(
            input_signal=torch.randn(1, 8, 8),
            input_signal_length=torch.tensor([8]),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Perception packed-gradient parity requires CUDA")
def test_perception_packed_spec_augmentation_and_projection_gradients_match_legacy():
    torch.manual_seed(0)
    legacy = _make_perception().cuda().train()
    legacy.spec_augmentation = _SpecAugment()
    packed = copy.deepcopy(legacy)
    legacy_features = torch.randn(3, 8, 12, device="cuda", requires_grad=True)
    packed_features = legacy_features.detach().clone().requires_grad_()
    lengths = torch.tensor([12, 7, 4], device="cuda")

    legacy_output, output_lengths = legacy(input_signal=legacy_features, input_signal_length=lengths)
    packed_output = packed.forward_sequence_packed(input_signal=packed_features, input_signal_length=lengths)
    valid = torch.arange(legacy_output.shape[1], device="cuda")[None, :] < output_lengths[:, None]
    legacy_output[valid].square().mean().backward()
    packed_output.data.square().mean().backward()

    assert legacy.spec_augmentation.calls == packed.spec_augmentation.calls == 1
    torch.testing.assert_close(packed_features.grad, legacy_features.grad, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(packed.proj.weight.grad, legacy.proj.weight.grad, rtol=1e-5, atol=1e-6)


def test_perception_packed_api_rejects_legacy_encoder_return_request():
    perception = _make_perception()

    with pytest.raises(TypeError, match="return_encoder_emb"):
        perception.forward_sequence_packed(
            input_signal=torch.randn(1, 8, 8),
            input_signal_length=torch.tensor([8]),
            return_encoder_emb=True,
        )
