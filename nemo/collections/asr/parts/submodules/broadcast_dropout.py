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

"""Dropout with one mask over the feature axis, shared by every position of the batch."""

import torch
from torch import nn

from nemo.core.utils.optional_libs import TRITON_AVAILABLE

if TRITON_AVAILABLE:
    from nemo.collections.asr.parts.triton.activation_dropout import ACTIVATION_DROPOUT_DTYPES, activation_dropout

__all__ = ['BroadcastDropout']


class BroadcastDropout(nn.Module):
    """Dropout with one mask over the last axis, shared by every position of the batch.

    ``torch.nn.Dropout`` draws a mask per element.
    This module draws one mask of the feature width per call and broadcasts it over the leading axes, batch and time,
    as RETURNN's ``rf_dropout_broadcast`` does for the input, feed-forward and residual dropouts of its Conformer;
    the kept features are scaled by ``1 / (1 - p)``.
    In training on CUDA with Triton installed the mask and the scaling run in one kernel
    (:func:`nemo.collections.asr.parts.triton.activation_dropout.activation_dropout` with the identity activation),
    which saves the mask alone for the backward; elsewhere the mask is drawn with ``torch.rand`` and multiplied in.

    Args:
        p: dropout probability in ``[0, 1)``
    """

    def __init__(self, p: float = 0.5):
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError(f"dropout probability has to be in [0, 1), got {p}")
        self.p = p

    def extra_repr(self):
        return f"p={self.p}"

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        if TRITON_AVAILABLE and x.is_cuda and x.dtype in ACTIVATION_DROPOUT_DTYPES:
            return activation_dropout(x, "identity", self.p, broadcast=True)
        keep = torch.rand(x.shape[-1], device=x.device) >= self.p
        return x * keep.to(x.dtype) * (1.0 / (1.0 - self.p))
