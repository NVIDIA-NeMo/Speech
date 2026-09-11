# SPDX-FileCopyrightText: Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from typing import TYPE_CHECKING, Optional

from omegaconf import DictConfig, open_dict

from nemo.collections.asr.modules import conv_asr
from nemo.collections.asr.parts.submodules import jasper
from nemo.utils import logging

if TYPE_CHECKING:
    from nemo.collections.asr.models.asr_model import ASRModel


def change_conv_asr_se_context_window(model: 'ASRModel', context_window: int, update_config: bool = True):
    """
    Update the context window of the SqueezeExcitation module if the provided model contains an
    `encoder` which is an instance of `ConvASREncoder`.

    Args:
        model: A subclass of `ASRModel`, itself a subclass of `ModelPT`.
        context_window:  An integer representing the number of input timeframes that will be used
            to compute the context. Each timeframe corresponds to a single window stride of the
            STFT features.

            Say the window_stride = 0.01s, then a context window of 128 represents 128 * 0.01 s
            of context to compute the Squeeze step.
        update_config: Whether to update the config or not with the new context window.
    """
    if update_config and not hasattr(model.cfg, 'encoder'):
        logging.info(
            "Could not change the context window in SqueezeExcite module "
            "since the model provided does not contain an `encoder` module in its config."
        )
        return

    if not isinstance(model.encoder, conv_asr.ConvASREncoder):
        logging.info(
            f"Could not change the context window in SqueezeExcite module "
            f"since the `encoder` module is not an instance of `ConvASREncoder`.\n"
            f"Provided encoder class = {model.encoder.__class__.__name__}"
        )
        return

    enc_cfg = model.cfg.encoder if update_config else None

    if enc_cfg is not None:
        with open_dict(enc_cfg):
            _update_se_context_window(model, context_window, cfg=enc_cfg)
    else:
        _update_se_context_window(model, context_window)

    # Update model config
    if update_config:
        model.cfg.encoder = enc_cfg


def _update_se_context_window(model: 'ASRModel', context_window: int, cfg: Optional[DictConfig] = None):
    jasper_block_counter = -1
    for name, m in model.named_modules():
        if type(m) == jasper.JasperBlock:
            jasper_block_counter += 1

        if type(m) == jasper.MaskedConv1d:
            if m.conv.stride[0] > 1 and 'mconv' in name:
                context_window = context_window // m.conv.stride[0]

        if type(m) == jasper.SqueezeExcite:
            m.change_context_window(context_window=context_window)

            # update config
            if cfg is not None:
                cfg.jasper[jasper_block_counter].se_context_size = context_window
