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
###############################################################################
import torch
from torch.nn import functional as F

from nemo.collections.tts.losses.aligner_loss import ForwardSumLoss
from nemo.collections.tts.parts.utils.helpers import get_mask_from_lengths
from nemo.core.classes import Loss


def compute_flow_loss(z, log_det_W_list, log_s_list, n_elements, n_dims, mask, sigma=1.0):

    log_det_W_total = 0.0
    for i, log_s in enumerate(log_s_list):
        if i == 0:
            log_s_total = torch.sum(log_s * mask)
            if len(log_det_W_list):
                log_det_W_total = log_det_W_list[i]
        else:
            log_s_total = log_s_total + torch.sum(log_s * mask)
            if len(log_det_W_list):
                log_det_W_total += log_det_W_list[i]

    if len(log_det_W_list):
        log_det_W_total *= n_elements

    z = z * mask
    prior_NLL = torch.sum(z * z) / (2 * sigma * sigma)

    loss = prior_NLL - log_s_total - log_det_W_total

    denom = n_elements * n_dims
    loss = loss / denom
    loss_prior = prior_NLL / denom
    return loss, loss_prior


def compute_regression_loss(x_hat, x, mask, name=False):
    x = x[:, None] if len(x.shape) == 2 else x  # add channel dim
    mask = mask[:, None] if len(mask.shape) == 2 else mask  # add channel dim
    assert len(x.shape) == len(mask.shape)

    x = x * mask
    x_hat = x_hat * mask

    if name == 'vpred':
        loss = F.binary_cross_entropy_with_logits(x_hat, x, reduction='sum')
    else:
        loss = F.mse_loss(x_hat, x, reduction='sum')
    loss = loss / mask.sum()

    loss_dict = {"loss_{}".format(name): loss}

    return loss_dict


class AttributePredictionLoss(torch.nn.Module):
    def __init__(self, name, model_config, loss_weight, sigma=1.0):
        super(AttributePredictionLoss, self).__init__()
        self.name = name
        self.sigma = sigma
        self.model_name = model_config['name']
        self.loss_weight = loss_weight
        self.n_group_size = 1
        if 'n_group_size' in model_config['hparams']:
            self.n_group_size = model_config['hparams']['n_group_size']

    def forward(self, model_output, lens):
        mask = get_mask_from_lengths(lens // self.n_group_size)
        mask = mask[:, None].float()
        loss_dict = {}
        if 'z' in model_output:
            n_elements = lens.sum() // self.n_group_size
            n_dims = model_output['z'].size(1)

            loss, loss_prior = compute_flow_loss(
                model_output['z'],
                model_output['log_det_W_list'],
                model_output['log_s_list'],
                n_elements,
                n_dims,
                mask,
                self.sigma,
            )
            loss_dict = {
                "loss_{}".format(self.name): (loss, self.loss_weight),
                "loss_prior_{}".format(self.name): (loss_prior, 0.0),
            }
        elif 'x_hat' in model_output:
            loss_dict = compute_regression_loss(model_output['x_hat'], model_output['x'], mask, self.name)
            for k, v in loss_dict.items():
                loss_dict[k] = (v, self.loss_weight)

        if len(loss_dict) == 0:
            raise Exception("loss not supported")

        return loss_dict


class AttentionBinarizationLoss(torch.nn.Module):
    def __init__(self):
        super(AttentionBinarizationLoss, self).__init__()

    def forward(self, hard_attention, soft_attention):
        log_sum = torch.log(soft_attention[hard_attention == 1]).sum()
        return -log_sum / hard_attention.sum()


class RADTTSLoss(Loss):
    def __init__(
        self,
        sigma=1.0,
        n_group_size=1,
        dur_model_config=None,
        f0_model_config=None,
        energy_model_config=None,
        vpred_model_config=None,
        loss_weights=None,
    ):
        super(RADTTSLoss, self).__init__()
        self.sigma = sigma
        self.n_group_size = n_group_size
        self.loss_weights = loss_weights
        self.attn_ctc_loss = ForwardSumLoss()
        self.loss_weights = loss_weights
        self.loss_fns = {}
        if dur_model_config is not None:
            self.loss_fns['duration_model_outputs'] = AttributePredictionLoss(
                'duration', dur_model_config, loss_weights['dur_loss_weight']
            )

        if f0_model_config is not None:
            self.loss_fns['f0_model_outputs'] = AttributePredictionLoss(
                'f0', f0_model_config, loss_weights['f0_loss_weight'], sigma=1.0
            )

        if energy_model_config is not None:
            self.loss_fns['energy_model_outputs'] = AttributePredictionLoss(
                'energy', energy_model_config, loss_weights['energy_loss_weight']
            )

        if vpred_model_config is not None:
            self.loss_fns['vpred_model_outputs'] = AttributePredictionLoss(
                'vpred', vpred_model_config, loss_weights['vpred_loss_weight']
            )

    def forward(self, model_output, in_lens, out_lens):
        loss_dict = {}
        if len(model_output['z_mel']):
            n_elements = out_lens.sum() // self.n_group_size
            mask = get_mask_from_lengths(out_lens // self.n_group_size)
            mask = mask[:, None].float()
            n_dims = model_output['z_mel'].size(1)
            loss_mel, loss_prior_mel = compute_flow_loss(
                model_output['z_mel'],
                model_output['log_det_W_list'],
                model_output['log_s_list'],
                n_elements,
                n_dims,
                mask,
                self.sigma,
            )
            loss_dict['loss_mel'] = (loss_mel, 1.0)  # loss, weight
            loss_dict['loss_prior_mel'] = (loss_prior_mel, 0.0)

        ctc_cost = self.attn_ctc_loss(attn_logprob=model_output['attn_logprob'], in_lens=in_lens, out_lens=out_lens)
        loss_dict['loss_ctc'] = (ctc_cost, self.loss_weights['ctc_loss_weight'])

        for k in model_output:
            if k in self.loss_fns:
                if model_output[k] is not None and len(model_output[k]) > 0:
                    t_lens = in_lens if 'dur' in k else out_lens
                    mout = model_output[k]
                    for loss_name, v in self.loss_fns[k](mout, t_lens).items():
                        loss_dict[loss_name] = v

        return loss_dict
