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
import torch.nn as nn
import torch.nn.functional as F

from nemo.collections.tts.modules.common import ConvLSTMLinear
from nemo.collections.tts.modules.submodules import ConvNorm, MaskedInstanceNorm1d
from nemo.collections.tts.modules.transformer import FFTransformer
from nemo.collections.tts.parts.utils.helpers import get_mask_from_lengths


def get_attribute_prediction_model(config):
    name = config['name']
    hparams = config['hparams']
    if name == 'dap':
        model = DAP(**hparams)
    else:
        raise Exception("{} model is not supported".format(name))

    return model


class AttributeProcessing(nn.Module):
    def __init__(self, take_log_of_input=False):
        super(AttributeProcessing, self).__init__()
        self.take_log_of_input = take_log_of_input

    def normalize(self, x):
        if self.take_log_of_input:
            x = torch.log(x + 1)
        return x

    def denormalize(self, x):
        if self.take_log_of_input:
            x = torch.exp(x) - 1
        return x


class BottleneckLayerLayer(nn.Module):
    def __init__(self, in_dim, reduction_factor, norm='weightnorm', non_linearity='relu', use_pconv=False):
        super(BottleneckLayerLayer, self).__init__()

        self.reduction_factor = reduction_factor
        reduced_dim = int(in_dim / reduction_factor)
        self.out_dim = reduced_dim
        if self.reduction_factor > 1:
            if norm == 'weightnorm':
                norm_args = {"use_weight_norm": True}
            elif norm == 'instancenorm':
                norm_args = {"norm_fn": MaskedInstanceNorm1d}
            else:
                norm_args = {}
            fn = ConvNorm(in_dim, reduced_dim, kernel_size=3, **norm_args)
            self.projection_fn = fn
            self.non_linearity = non_linearity

    def forward(self, x, lens):
        if self.reduction_factor > 1:
            # borisf: here, float() instead of to(x.dtype) to work arounf ONNX exporter bug
            mask = get_mask_from_lengths(lens, x).unsqueeze(1).float()
            x = self.projection_fn(x, mask)
            if self.non_linearity == 'relu':
                x = F.relu(x)
            elif self.non_linearity == 'leakyrelu':
                x = F.leaky_relu(x)
        return x


class DAP(AttributeProcessing):
    def __init__(self, n_speaker_dim, bottleneck_hparams, take_log_of_input, arch_hparams, use_transformer=False):
        super(DAP, self).__init__(take_log_of_input)
        self.bottleneck_layer = BottleneckLayerLayer(**bottleneck_hparams)
        arch_hparams['in_dim'] = self.bottleneck_layer.out_dim + n_speaker_dim
        if use_transformer:
            self.feat_pred_fn = FFTransformer(**arch_hparams)
        else:
            self.feat_pred_fn = ConvLSTMLinear(**arch_hparams)

    def forward(self, txt_enc, spk_emb, x, lens):
        if x is not None:
            x = self.normalize(x)

        txt_enc = self.bottleneck_layer(txt_enc, lens)
        spk_emb_expanded = spk_emb[..., None].expand(-1, -1, txt_enc.shape[2])
        context = torch.cat((txt_enc, spk_emb_expanded), 1)
        x_hat = self.feat_pred_fn(context, lens)
        outputs = {'x_hat': x_hat, 'x': x}
        return outputs

    def infer(self, txt_enc, spk_emb, lens=None):
        x_hat = self.forward(txt_enc, spk_emb, x=None, lens=lens)['x_hat']
        x_hat = self.denormalize(x_hat)
        return x_hat
