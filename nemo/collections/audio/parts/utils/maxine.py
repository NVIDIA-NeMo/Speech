# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from torch.nn.utils import remove_weight_norm, weight_norm


def apply_weight_norm_lstm(lstm_module):
    bidirectional = lstm_module.bidirectional
    lstm_wn = weight_norm(lstm_module, name='weight_ih_l0')
    lstm_wn = weight_norm(lstm_wn, name='weight_hh_l0')
    if bidirectional:
        lstm_wn = weight_norm(lstm_wn, name='weight_ih_l0_reverse')
        lstm_wn = weight_norm(lstm_wn, name='weight_hh_l0_reverse')
    return lstm_wn


def remove_weight_norm_lstm(lstm_module):
    bidirectional = lstm_module.bidirectional
    lstm = remove_weight_norm(lstm_module, name='weight_ih_l0')
    lstm = remove_weight_norm(lstm, name='weight_hh_l0')
    if bidirectional:
        lstm = remove_weight_norm(lstm, name='weight_ih_l0_reverse')
        lstm = remove_weight_norm(lstm, name='weight_hh_l0_reverse')
    return lstm
