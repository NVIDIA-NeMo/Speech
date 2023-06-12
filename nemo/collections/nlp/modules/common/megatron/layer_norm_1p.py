# coding=utf-8
# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import torch
from nemo.collections.nlp.modules.common.megatron.utils import _cast_if_autocast_enabled

try:
    from apex.contrib.layer_norm.layer_norm import FastLayerNorm as OrigFastLayerNorm
    from apex.contrib.layer_norm.layer_norm import _fast_layer_norm
    from apex.transformer.layers.layer_norm import FastLayerNorm

    HAVE_APEX = True
except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False


if HAVE_APEX:
    # TODO: use Apex implementation
    class LayerNorm1P(FastLayerNorm):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            assert isinstance(
                self, OrigFastLayerNorm
            ), 'LayerNorm1P implemented only as an apex.contrib.layer_norm.FastLayerNorm extension'

        def reset_parameters(self):
            torch.nn.init.zeros_(self.weight)
            torch.nn.init.zeros_(self.bias)

        def forward(self, x):
            return _fast_layer_norm(x, self.weight + 1, self.bias, self.epsilon)


else:

    class LayerNorm1P(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            raise NotImplementedError('LayerNorm1P available only with apex installed')


class LPLayerNorm(torch.nn.LayerNorm):
    def __init__(self, normalized_shape, eps=1e-05, elementwise_affine=True, device=None, dtype=None):
        super().__init__(
            normalized_shape=normalized_shape,
            eps=eps,
            elementwise_affine=elementwise_affine,
            device=device,
            dtype=dtype,
        )

    def forward(self, x):
        module_device = x.device
        downcast_x = _cast_if_autocast_enabled(x)
        downcast_weight = _cast_if_autocast_enabled(self.weight) if self.weight is not None else self.weight
        downcast_bias = _cast_if_autocast_enabled(self.bias) if self.bias is not None else self.bias
        with torch.autocast(enabled=False, device_type=module_device.type):
            return torch.nn.functional.layer_norm(
                downcast_x, self.normalized_shape, downcast_weight, downcast_bias, self.eps
            )
