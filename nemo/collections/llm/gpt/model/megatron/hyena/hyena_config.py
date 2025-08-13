# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2024 Arc Institute. All rights reserved.
# Copyright (c) 2024 Michael Poli. All rights reserved.
# Copyright (c) 2024 Stanford University. All rights reserved
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

from dataclasses import dataclass


@dataclass
class HyenaConfig:
    """Configuration object for Hyena model and operators"""

    tie_projection_weights: bool = False
    """
    Tie projection weights between QKV for attn and hyena (will repeat output 3 times).
    """
    #
    to_upper: str = "normalized_weighted"
    """
    "upper"
    "weighted"
    Whether to convert all text to uppercase.
    """
    #
    lowercase_loss_reweighting: float = 0.1
    # """
    # If to_upper == "weighted"
    # Weight to apply to lowercase tokens in the loss function, 1.0 is no reweighting.
    # """

    short_conv_L: int = 3
    """
    For Hyena models, length of the short convolution.
    """

    use_hyena_filter: bool = False
    """
    Whether to use the Hyena filter.
    """

    normalize_hyena_filters: bool = False

    conv_proj_bias: bool = True  # Maybe this should be false
    """
    Use bias in the short conv1D, needed for model parallel for the short conv.
    """

    use_fast_heads: bool = False
    """
    Use external fast heads in Hyena mixer (reduce BEFORE fftconv)
    """

    num_groups_hyena: int = None
    """
    Determines number of unique filters to have, for the hyena long filter.
    """

    num_groups_hyena_medium: int = None
    """
    Determines number of unique filters to have, for the hyena medium filter.
    """

    num_groups_hyena_short: int = None
    """
    Determines number of unique filters to have, for the hyena short filter.
    """

    use_depthwise_short_conv_grouping: bool = True
    """
    Whether to use depthwise convolution grouping for short conv and hyena mlp filters.
    """

    hyena_filter_cls: str = "implicit_modal"
    """
    """

    hyena_width_expansion: float = 1.0
    """
    Factor to expand the projections width within hyena layers.
    """

    hyena_medium_filter_cls: str = 'explicit_single_decay'
    """
    For medium hyena filters specifically, None defaults ot same as hyena_filter_cls (long filters).
    """

    hyena_filter_order: int = 16

    explicit_filter_decay_preset: str = "weak"

    modal_gamma_min: float = 0.01

    modal_gamma_max: float = 0.1

    bidirectional: bool = False
    """
    A bidirectional version of hyena fftconv
    """

    hyena_short_conv_len: int = 7
    """
    Length of the hyena short conv layer, if using
    """

    fast_conv_proj: bool = True
    """
    Use a custom causal conv layer for the hyena projection convs.
    """

    hyena_medium_conv_len: int = 128
    """
    Length of the medium hyena filter.
    """

    fast_conv_mixer: bool = False
    """
    Use a custom causal conv layer for the hyena short conv layer.
    """

    use_cuhyena: bool = False
    """
    Use a back-to-back causal convolution CUDA kernel for the hyena short conv layers for improved performance.
    """

    hyena_short_conv_pregate: bool = True
    """
    Use a pre-gate in the hyena short conv layer.
    """

    hyena_short_conv_postgate: bool = True
    """
    Use a post-gate in the hyena short conv layer.
    """

    proj_groups: int = 1

    grouped_attention: bool = False
