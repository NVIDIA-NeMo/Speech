# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

try:
    from apex._autocast_utils import _cast_if_autocast_enabled

    HAVE_APEX = True
except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False

###### BIAS GELU FUSION/ NO AUTOGRAD ################
# 1/sqrt(2*pi)-> 0.3989423
# 1/sqrt(2)   -> 0.70710678
# sqrt(2/pi)  -> 0.79788456
# this function is tanh approximation of gelu
# actual gelu is:
# x * 0.5 * (1.0 + torch.erf(x * 0.70710678))


@torch.jit.script
def bias_gelu(bias, y):
    x = bias + y
    return x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))


# gradient of tanh approximation of gelu
# gradient of actual gelu is:
# 0.5 * (1. + torch.erf(x * 0.70710678)) + 0.3989423 * x * torch.exp(-0.5 * x * x)
@torch.jit.script
def bias_gelu_back(g, bias, y):
    x = bias + y
    tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    # sqrt(2/pi) * 3 * 0.044715 -> 0.1070322243
    ff = 0.5 * x * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)) + 0.5 * (1 + tanh_out)
    return ff * g


class GeLUFunction(torch.autograd.Function):
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, bias):
        ctx.save_for_backward(input, bias)
        return bias_gelu(bias, input)

    @staticmethod
    def backward(ctx, grad_output):
        input, bias = ctx.saved_tensors
        tmp = bias_gelu_back(grad_output, bias, input)
        return tmp, tmp

    @staticmethod
    def symbolic(g: torch.Graph, input: torch.Value, bias: torch.Value):
        # define constants and variables
        x = g.op("Add", input, bias)
        const_1 = g.op("Constant", value_t=torch.tensor(0.5, dtype=torch.float16))
        const_2 = g.op("Constant", value_t=torch.tensor(1.0, dtype=torch.float16))
        const_3 = g.op("Constant", value_t=torch.tensor(0.79788456, dtype=torch.float16))
        const_4 = g.op("Constant", value_t=torch.tensor(0.044715, dtype=torch.float16))

        # calculates (1 + 0.044715 * x * x)
        p_1 = g.op("Add", const_2, g.op("Mul", x, g.op("Mul", const_4, x)))

        # calculates torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
        p_2 = g.op("Tanh", g.op("Mul", const_3, g.op("Mul", x, p_1)))

        # calculates x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))
        return g.op("Mul", const_1, g.op("Mul", x, g.op("Add", const_2, p_2)))


def fused_bias_gelu(input, bias):
    args = _cast_if_autocast_enabled(input, bias)
    with torch.cuda.amp.autocast(enabled=False):
        return GeLUFunction.apply(*args)
