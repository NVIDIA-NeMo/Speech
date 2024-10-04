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

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from torch import nn

from nemo.collections.asr.parts.submodules.jasper import jasper_activations
from nemo.core import NeuralModule
from nemo.core.neural_types import EncodedRepresentation, LossType, NeuralType


class GumbelVectorQuantizer(NeuralModule):
    def __init__(
        self,
        dim,
        num_vars,
        temp,
        groups,
        combine_groups,
        vq_dim,
        time_first,
        activation="gelu",
        weight_proj_depth=1,
        weight_proj_factor=1,
    ):
        """Vector quantization using gumbel softmax

        Args:
            dim: input dimension (channels)
            num_vars: number of quantized vectors per group
            temp: temperature for training. this should be a tuple of 3 elements: (start, stop, decay factor)
            groups: number of groups for vector quantization
            combine_groups: whether to use the vectors for all groups
            vq_dim: dimensionality of the resulting quantized vector
            time_first: if true, expect input in BxTxC format, otherwise in BxCxT
            activation: what activation to use (should be a module). this is only used if weight_proj_depth is > 1
            weight_proj_depth: number of layers (with activation in between) to project input before computing logits
            weight_proj_factor: this is used only if weight_proj_depth is > 1. scales the inner dimensionality of
                                projections by this factor
        """
        super().__init__()

        self.groups = groups
        self.combine_groups = combine_groups
        self.input_dim = dim
        self.num_vars = num_vars
        self.time_first = time_first

        assert vq_dim % groups == 0, f"dim {vq_dim} must be divisible by groups {groups} for concatenation"

        var_dim = vq_dim // groups
        num_groups = groups if not combine_groups else 1

        self.vars = nn.Parameter(torch.FloatTensor(1, num_groups * num_vars, var_dim))
        nn.init.uniform_(self.vars)

        if weight_proj_depth > 1:
            activation = jasper_activations["gelu"]

            def block(input_dim, output_dim):
                return nn.Sequential(nn.Linear(input_dim, output_dim), activation)

            inner_dim = self.input_dim * weight_proj_factor
            self.weight_proj = nn.Sequential(
                *[block(self.input_dim if i == 0 else inner_dim, inner_dim) for i in range(weight_proj_depth - 1)],
                nn.Linear(inner_dim, groups * num_vars),
            )
        else:
            self.weight_proj = nn.Linear(self.input_dim, groups * num_vars)
            nn.init.normal_(self.weight_proj.weight, mean=0, std=1)
            nn.init.zeros_(self.weight_proj.bias)

        assert len(temp) == 3, "Quantize temperature should be a tuple of 3 elements: (start, stop, decay factor)"

        self.max_temp, self.min_temp, self.temp_decay = temp
        self.curr_temp = self.max_temp
        self.codebook_indices = None

    def set_num_updates(self, num_updates):
        self.curr_temp = max(self.max_temp * self.temp_decay**num_updates, self.min_temp)

    def get_codebook_indices(self):
        if self.codebook_indices is None:
            from itertools import product

            p = [range(self.num_vars)] * self.groups
            inds = list(product(*p))
            self.codebook_indices = torch.tensor(inds, dtype=torch.long, device=self.vars.device).flatten()

            if not self.combine_groups:
                self.codebook_indices = self.codebook_indices.view(self.num_vars**self.groups, -1)
                for b in range(1, self.groups):
                    self.codebook_indices[:, b] += self.num_vars * b
                self.codebook_indices = self.codebook_indices.flatten()
        return self.codebook_indices

    def sample_from_codebook(self, b, n):
        indices = self.get_codebook_indices()
        indices = indices.view(-1, self.groups)
        cb_size = indices.size(0)
        assert n < cb_size, f"sample size {n} is greater than size of codebook {cb_size}"
        sample_idx = torch.randint(low=0, high=cb_size, size=(b * n,))
        indices = indices[sample_idx]

        z = self.vars.squeeze(0).index_select(0, indices.flatten()).view(b, n, -1)
        return z

    @property
    def input_types(self):
        """Returns definitions of module input ports."""
        if self.time_first:
            return {"x": NeuralType(('B', 'T', 'D'), EncodedRepresentation())}
        return {"x": NeuralType(('B', 'D', 'T'), EncodedRepresentation())}

    @property
    def output_types(self):
        """Returns definitions of module output ports."""
        if self.time_first:
            return {
                "x": NeuralType(('B', 'T', 'D'), EncodedRepresentation()),
                "quantize_prob_ppl": NeuralType(elements_type=LossType()),
            }
        return {
            "x": NeuralType(('B', 'D', 'T'), EncodedRepresentation()),
            "quantize_prob_ppl": NeuralType(elements_type=LossType()),
        }

    def forward(self, x, return_ids=False):

        if not self.time_first:
            x = x.transpose(1, 2)

        bsz, tsz, fsz = x.shape
        x = x.reshape(-1, fsz)
        x = self.weight_proj(x)
        x = x.view(bsz * tsz * self.groups, -1)

        _, k = x.max(-1)
        hard_x = x.new_zeros(*x.shape).scatter_(-1, k.view(-1, 1), 1.0).view(bsz * tsz, self.groups, -1)

        # Calculate quantize prob perplexity
        num_vars = self.num_vars * self.groups
        avg_probs = torch.softmax(x.view(bsz * tsz, self.groups, -1).float(), dim=-1).mean(dim=0)
        quantize_prob_ppl = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-7), dim=-1)).sum()
        quantize_prob_ppl = (num_vars - quantize_prob_ppl) / num_vars

        if self.training:
            x = F.gumbel_softmax(x.float(), tau=self.curr_temp, hard=True).type_as(x)
        else:
            x = hard_x

        x = x.view(bsz * tsz, -1)

        vars = self.vars
        if self.combine_groups:
            vars = vars.repeat(1, self.groups, 1)

        x = x.unsqueeze(-1) * vars
        x = x.view(bsz * tsz, self.groups, self.num_vars, -1)
        x = x.sum(-2)
        x = x.view(bsz, tsz, -1)

        cur_codebook_temp = self.curr_temp

        if not self.time_first:
            x = x.transpose(1, 2)  # BTC -> BCT

        if return_ids:
            hard_x_max = hard_x.argmax(-1).reshape(bsz, tsz, -1)
            # BxTxG

            # create single id from multiple group ids
            target_ids = hard_x.new_zeros(bsz, tsz).long()

            for i in range(self.groups):
                target_ids *= self.num_vars
                target_ids += hard_x_max[:, :, i]

            return x, quantize_prob_ppl, cur_codebook_temp, target_ids
        else:
            return x, quantize_prob_ppl, cur_codebook_temp
