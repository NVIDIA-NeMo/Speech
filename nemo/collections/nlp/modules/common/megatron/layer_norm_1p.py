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

from torch import nn

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
            nn.init.zeros_(self.weight)
            nn.init.zeros_(self.bias)

        def forward(self, x):
            return _fast_layer_norm(x, self.weight + 1, self.bias, self.epsilon)


else:

    class LayerNorm1P(nn.Module):
        def __init__(self, *args, **kwargs):
            raise NotImplementedError('LayerNorm1P available only with apex installed')
