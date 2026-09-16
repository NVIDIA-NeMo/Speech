# SPDX-FileCopyrightText: Copyright (c) 2020, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from nemo.collections.common.losses.aggregator import AggregatorLoss
from nemo.collections.common.losses.bce_logits_loss import BCEWithLogitsLoss
from nemo.collections.common.losses.cross_entropy import CrossEntropyLoss, NLLLoss
from nemo.collections.common.losses.mse_loss import MSELoss
from nemo.collections.common.losses.multi_similarity_loss import MultiSimilarityLoss
from nemo.collections.common.losses.smoothed_cross_entropy import SmoothedCrossEntropyLoss, SmoothedNLLLoss
from nemo.collections.common.losses.spanning_loss import SpanningLoss
