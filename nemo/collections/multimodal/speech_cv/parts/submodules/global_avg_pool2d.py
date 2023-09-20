# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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


class GlobalAvgPool2d(nn.Module):
    def __init__(self, dim=(2, 3), keepdim=False):
        super(GlobalAvgPool2d, self).__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x):

        assert x.dim() == 4, "input signal should have 4 dims, has {}".format(x.dim())

        return x.mean(dim=self.dim, keepdim=self.keepdim)
