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


import math

import torch

from nemo.collections.nlp.modules.common.megatron.utils import ApexGuardDefaults, init_method_normal

try:
    from megatron.core import ModelParallelConfig, tensor_parallel

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):
    # fake missing classes with None attributes
    ModelParallelConfig = ApexGuardDefaults()
    tensor_parallel = ApexGuardDefaults()

    HAVE_MEGATRON_CORE = False

__all__ = ["MegatronBaseHiddenTransform", "MegatronGaussianHiddenTransform"]


class MegatronBaseHiddenTransform(torch.nn.Module):
    """Base class to apply hidden state transformations"""

    def __init__(self, name: str = "", model_parallel_cfg: ModelParallelConfig = None):
        super().__init__()

        self.name = name
        self.model_parallel_cfg = model_parallel_cfg

    def __str__(self):
        return super().__str__() + f"(name={self.name})"

    @property
    def input_names(self):
        """
        Provide here all required inputs
        """
        return []

    @property
    def output_names(self):
        """
        Provide here all generated outputs
        """
        return []

    def _validate_inputs(self, inputs):
        """Validate inputs"""
        # validate inputs
        if not set(self.input_names).issubset(set(inputs.keys())):
            raise ValueError(f"Inputs should contain {self.input_names}, but got {inputs.keys()}")

    def _transform(self, inputs, batch_data=None):
        """
        Implement your own transformations.
        We expect here shapes to be [S x B x H] for Sequence, Batch, Hidden sizes (due to tensor parallel support).
        """
        # by default we pass inputs.
        outputs = inputs.copy()

        return outputs

    def transform(self, inputs, batch_data=None):
        """Apply a transformations on the inputs (hiddens is always assumed)"""
        # validate inputs
        self._validate_inputs(inputs)

        outputs = self._transform(inputs, batch_data=batch_data)

        return outputs


class MegatronGaussianHiddenTransform(MegatronBaseHiddenTransform):
    """
    Constructes a diagonal Gaussian distribution from the hidden states and samples from it using reparametrization.
    """

    def __init__(
        self,
        hidden_size,
        ffn_hidden_size=None,
        min_logvar=-6,
        init_method_std=0.02,
        name="cond_gaussian",
        model_parallel_cfg: ModelParallelConfig = None,
    ):
        super().__init__(name=name, model_parallel_cfg=model_parallel_cfg)
        # limit smaller allowed variance (for numerical stability)
        self.min_logvar = min_logvar
        self.hidden_size = hidden_size
        if ffn_hidden_size is None:
            ffn_hidden_size = hidden_size * 2
        self.ffn_hidden_size = ffn_hidden_size

        # project hiddens to mean and log variance (support tensor parallelism)
        self.hiddens_to_mean_logvar = tensor_parallel.ColumnParallelLinear(
            hidden_size,
            ffn_hidden_size,  # NOTE: When using *glu, divide ffn dim by 2/3 to keep overall params the same.
            gather_output=True,
            init_method=init_method_normal(init_method_std),
            skip_bias_add=False,
            bias=True,
            config=self.model_parallel_cfg,
        )

    @property
    def input_names(self):
        """
        Provide here all required inputs
        """
        return ["hiddens", "hiddens_mask"]

    @property
    def output_names(self):
        """
        Provide here all generated outputs
        """
        return ["z_mean", "z_logvar", "z", "z_log_prob"]

    def _transform(self, inputs, batch_data=None):
        """
        We expect here shapes to be [S x B x H] for Sequence, Batch, Hidden sizes (due to tensor parallel support).

        inputs:
            hiddens: accepts a tensor of shape [S x B x H]
        
        outputs:
            z: a sample from Gaussian a tensor of shape [S x B x H]
            z_mean: mean of Gaussian a tensor of shape [S x B x H]
            z_logvar: log variance of Gaussian a tensor of shape [S x B x H]
            z_log_prob: log probability of z over posterior log q(z|x) a tensor of shape [S x B x H]
        """
        hiddens = inputs["hiddens"]
        # compute distribution's parameters (or use cached ones)
        if "z_mean" in inputs and "z_logvar" in inputs:
            z_mean = inputs["z_mean"]
            z_logvar = inputs["z_logvar"]
        else:
            # ColumnLinear returns output and bias, we ignore bias here (already added to hiddens)
            z_mean, z_logvar = self.hiddens_to_mean_logvar(hiddens)[0].chunk(2, dim=-1)
        # clamp logvar
        z_logvar = z_logvar.clamp(min=self.min_logvar)
        # sample z with reparametrization (or use cached one)
        if "z" in inputs:
            z = inputs["z"]
            z_log_prob = inputs.get("z_log_prob", None)
        else:
            e = torch.randn_like(hiddens)
            z = (z_logvar * 0.5).exp() * e + z_mean
            z_log_prob = None

        if z_log_prob is None:
            # compute log probability of z under a diagonal Gaussian distribution
            z_log_prob = -0.5 * (math.log(2 * math.pi) + z_logvar + (z - z_mean).pow(2) / z_logvar.exp())
            # sum over the last dimension (hidden_size)
            z_log_prob = z_log_prob.sum(dim=-1)

        return {
            "z": z,  # [S x B x H]
            "z_mean": z_mean,  # [S x B x H]
            "z_logvar": z_logvar,  # [S x B x H]
            "z_log_prob": z_log_prob,  # [S x B]
        }
