# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from importlib.metadata import version
from typing import TYPE_CHECKING, Dict, Optional

import packaging
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from nemo.collections.nlp.parts.peft_config import LORA_CONFIG_TO_MCORE_MAP, get_target_modules
from nemo.utils import logging
from nemo.utils.import_utils import safe_import_from

te_version = packaging.version.Version(version("transformer-engine"))

if TYPE_CHECKING:
    from megatron.core.models.gpt import MCoreGPTModel
    from omegaconf import DictConfig


class NF4Weight(nn.Parameter):
    def __new__(
        cls,
        data: torch.Tensor,
        is_nf4_quantized: bool = False,
        block_size: int = 64,
        scale_block_size: int = 256,
    ):
        self = torch.Tensor._make_subclass(cls, data, require_grad=False)
        self._nf4_quantizer = None
        self.is_nf4_quantized = is_nf4_quantized
        self.block_size = block_size
        self.scale_block_size = scale_block_size
        return self

    def quantize(self, device='cuda') -> torch.Tensor:
        from modelopt.torch.quantization.nn import TensorQuantizer
        from modelopt.torch.quantization.tensor_quant import QuantDescriptor

        # initialize the quantizer
        nf4_desc = QuantDescriptor(
            num_bits=4,
            block_sizes={-1: self.block_size, "scale_bits": 8, "scale_block_sizes": {-1: self.scale_block_size}},
            fake_quant=False,
        )
        self._nf4_quantizer = TensorQuantizer(nf4_desc)

        # quantize on GPU directly
        nf4_tensor = self._nf4_quantizer(self.data.to(device))
        self.quantized_data = nf4_tensor
        self.is_nf4_quantized = True
        return self

    def dequantize(self):
        assert self.is_nf4_quantized, "NF4 Tensor is not yet quantized, cannot dequantize."
        return self._nf4_quantizer(self.quantized_data)

    def cuda(self, device=None, non_blocking=False):
        return self.to(device="cuda" if device is None else device, non_blocking=non_blocking)

    def to(self, *args, **kwargs):
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)

        if device is not None and device.type == "cuda":
            # Note: self.data remains on CPU. Only self.quantized_data is on GPU
            return self.quantize() if not self.is_nf4_quantized else self
        else:
            return NF4Weight(
                super().to(device=device, dtype=dtype, non_blocking=non_blocking),
                self.is_nf4_quantized,
                self.block_size,
                self.scale_block_size,
            )

    def __repr__(self, *, tensor_contents=None):
        if self.is_nf4_quantized:
            return f"NF4Weight(is_nf4_quantized=True, quantized_data={self.quantized_data}"
        else:
            return f"NF4Weight(is_nf4_quantized=False, data={self.data}"


class _LinearNF4(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, weight: NF4Weight):
        ctx.nf4_weight = weight
        return F.linear(input, weight.dequantize().to(input.device))

    @staticmethod
    def backward(ctx, grad_output):
        weight: NF4Weight = ctx.nf4_weight
        return grad_output @ weight.dequantize().to(grad_output.device), None


def nf4_quantize(x: torch.Tensor):
    return NF4Weight(x).cuda()


class NF4LinearWrapper(nn.Module):
    """
    NF4 Linear Layer for QLoRA as introduced in `QLORA: Efficient Finetuning of Quantized LLMs <https://arxiv.org/abs/2305.14314>`_.
    This wrapper module is instantiated in `on_load_checkpoint` and replaces TERowParallelLinear
    Tensor Parallel is not supported.

    Args:
        bf16_linear_weight: Weight tensor in BF16 to wrap with NF4Weight
    """

    def __init__(self, bf16_linear_weight: torch.Tensor):
        super().__init__()

        # quantize the weight upon initialization
        self.weight = nf4_quantize(bf16_linear_weight)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (Tensor): input tensor with shape ``(..., in_dim)``

        Returns:
            Tensor: output tensor with shape ``(..., out_dim)``

        """
        return _LinearNF4.apply(x, self.weight), None


class NF4LayerNormLinearWrapper(NF4LinearWrapper):
    """
    Layernorm + NF4 Linear for QLoRA.
    This class only combines the two modules for compatibility with TE's LayernormLinear layer, so that
    the implementation for LoRA and QLoRA can share the same code path.
    It does NOT fuse the two operations like TE does.
    This wrapper module is instantiated in `on_load_checkpoint` and replaces TELayerNormColumnParallelLinear
    Tensor Parallel is not supported.

    Args:
        bf16_linear_weight: Weight tensor in BF16 to wrap with NF4Weight
        layer_norm_weight: layernorm weight tensor
        layer_norm_bias: layernorm bias tensor, only if normalization is LayerNorm
        normalization: Same as TELayerNormColumnParallelLinear.config.normalization
        zero_centered_gamma: Same as TELayerNormColumnParallelLinear.config.zero_centered_gamma
    """

    def __init__(
        self,
        bf16_linear_weight: torch.Tensor,
        layer_norm_weight: torch.Tensor,
        layer_norm_bias: Optional[torch.Tensor],
        normalization: str,
        zero_centered_gamma: bool,
    ):
        super().__init__(bf16_linear_weight)
        self.layer_norm_weight = nn.Parameter(layer_norm_weight)
        if normalization != "RMSNorm":
            self.layer_norm_bias = nn.Parameter(layer_norm_bias)
        else:
            self.layer_norm_bias = None

        self.zero_centered_gamma = zero_centered_gamma
        self.normalization = normalization
        self.layer_norm_fn = self._create_layer_norm_fn()
        self.te_return_bias = False

    def _create_layer_norm_fn(self):
        '''
        create the layernorm function signature in TE. Assume this layer is already running without gradients
        since this is for QLoRA.
        '''
        if self.normalization == 'LayerNorm':
            _LayerNorm, _ = safe_import_from("transformer_engine.pytorch.module.layernorm", "_LayerNorm")

            layer_norm_fn = _LayerNorm.apply
        elif self.normalization == 'RMSNorm':
            _RMSNorm, _ = safe_import_from("transformer_engine.pytorch.module.layernorm", "_RMSNorm")

            layer_norm_fn = _RMSNorm.apply
        else:
            raise ValueError("Unsupported normalization type:", self.normalization)

        return layer_norm_fn

    def forward(self, x):
        layer_norm_args = [
            x,  # inp
            self.layer_norm_weight,
            1e-5,  # eps,
            0,  # fwd_rmsnorm_sm_margin,
            0,  # bwd_rmsnorm_sm_margin,
            self.zero_centered_gamma,
            True,  # is_grad_enabled,
            x.dtype,  # activation_dtype,
        ]
        if te_version >= packaging.version.Version("1.6"):
            layer_norm_args.insert(5, 0)  # inf_rmsnorm_sm_margin
        if self.normalization == "LayerNorm":
            layer_norm_args.insert(2, self.layer_norm_bias)
        layernorm_output = self.layer_norm_fn(*layer_norm_args)
        linear_output = _LinearNF4.apply(layernorm_output, self.weight)
        return (linear_output, layernorm_output), None


def qlora_load_model(model: 'MCoreGPTModel', model_cfg: 'DictConfig', checkpoint: Dict[str, Tensor]):
    # swap linear layer and cast weight to nf4
    qlora_targets = [
        LORA_CONFIG_TO_MCORE_MAP[x] for x in get_target_modules(model_cfg.peft.lora_tuning, default=('all',))
    ]

    # if not load directly on device, need to load the rest of the model
    # this block should only load word_embeddings, final_layernorm and output_layer weights.
    if not model_cfg.get("dist_ckpt_load_on_device", True):
        checkpoint_state_dict = {}
        for key, value in checkpoint.items():
            if not any(qlora_target in key for qlora_target in qlora_targets):
                checkpoint_state_dict[key.replace('model.', '')] = value
        model.load_state_dict(checkpoint_state_dict, strict=False)

    def replace_linear(module: nn.Module, prefix=""):
        for name, child in module.named_children():
            if name in qlora_targets:
                bf16_weight = checkpoint[f"{prefix}.{name}.weight"].to(torch.bfloat16)
                logging.info(f'QLoRA: Quantizing linear layer: {prefix}.{name}')
                layer_norm_weight = checkpoint.get(f"{prefix}.{name}.layer_norm_weight", None)
                if layer_norm_weight is None:
                    setattr(module, name, NF4LinearWrapper(bf16_weight))
                else:
                    layer_norm_bias = checkpoint.get(f"{prefix}.{name}.layer_norm_bias", None)
                    normalization = module.config.normalization
                    zero_centered_gamma = module.config.layernorm_zero_centered_gamma
                    setattr(
                        module,
                        name,
                        NF4LayerNormLinearWrapper(
                            bf16_weight, layer_norm_weight, layer_norm_bias, normalization, zero_centered_gamma
                        ),
                    )
            else:
                replace_linear(child, prefix=f"{prefix}.{name}")

    replace_linear(model, prefix="model")
