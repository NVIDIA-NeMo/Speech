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

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, Optional

import torch
from megatron.core import parallel_state

from nemo.utils.import_utils import safe_import

if torch.cuda.is_available():
    bitsandbytes, HAVE_BNB = safe_import("bitsandbytes")
else:
    bitsandbytes = None
    HAVE_BNB = False

import torch.nn.functional as F
from torch import nn

from nemo.utils.import_utils import safe_import_from

te, HAVE_TE = safe_import_from("transformer_engine", "pytorch")

from nemo.collections.llm.peft.module_matcher import ModuleMatcher
from nemo.collections.llm.peft.utils import get_adapter_attributes_from_linear, is_expert_linear
from nemo.lightning.pytorch.callbacks.peft import PEFT, AdapterWrapper
from nemo.utils import logging
from nemo.utils.te_utils import te_version


class LoRALinear(AdapterWrapper):
    """An adapter wrapper that adds the output of the adapter to the output of the wrapped module.

    This class is designed to be used with LoRA (Low-Rank Adaptation) and similar techniques
    where the adapter's output is added to the main module's output. It extends the AdapterWrapper
    class to provide a specific implementation of the forward method.
    """

    def forward(
        self,
        x: torch.Tensor,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        # pylint: disable=C0115,C0116
        linear_output, bias, layernorm_output = self.base_linear_forward(x, *args, **kwargs)
        adapter_output = self.adapter(layernorm_output.contiguous())
        adapter_output = adapter_output.reshape(linear_output.shape)
        return linear_output + adapter_output, bias


# Fused LoRA requires Transformer Engine 2.7+
HAVE_TE_FUSED_LORA: bool = HAVE_TE and te_version() >= (2, 7)

if HAVE_TE_FUSED_LORA:

    class TEFusedLoRALinear(LoRALinear):
        """LoRA adapter wrapper using Transformer Engine operation fuser"""

        def __init__(self, to_wrap: nn.Module, adapter: nn.Module):
            super().__init__(to_wrap, adapter)
            self._fused_branches: Optional[tuple[te.ops.Sequential, te.ops.Sequential]] = None

        def _make_fused_branches(self) -> tuple[te.ops.Sequential, te.ops.Sequential]:
            """Construct fused modules for main and LoRA branches"""

            # Extract layer size and tensor parallel config
            kwargs = {
                "in_features": self.to_wrap.weight.size(1),
                "out_features": self.to_wrap.weight.size(0),
                "tensor_parallel_mode": None,
                "tensor_parallel_group": None,
                "sequence_parallel": False,
            }
            tensor_parallel_size = parallel_state.get_tensor_model_parallel_world_size()
            if tensor_parallel_size > 1:
                kwargs["tensor_parallel_group"] = parallel_state.get_tensor_model_parallel_group()
                if isinstance(self.to_wrap, (te.Linear, te.LayerNormLinear)):
                    kwargs["tensor_parallel_mode"] = self.to_wrap.parallel_mode
                    kwargs["sequence_parallel"] = self.to_wrap.sequence_parallel
                if kwargs["tensor_parallel_mode"] == "row":
                    kwargs["in_features"] *= tensor_parallel_size
                elif kwargs["tensor_parallel_mode"] == "column":
                    kwargs["out_features"] *= tensor_parallel_size

            # wgrad accumulation fusion
            accumulate_into_main_grad = False
            if isinstance(self.to_wrap, (te.Linear, te.LayerNormLinear)):
                accumulate_into_main_grad = self.to_wrap.fuse_wgrad_accumulation
            kwargs["accumulate_into_main_grad"] = accumulate_into_main_grad

            # Construct fused branches
            main_branch = self._make_main_branch(**kwargs)
            lora_branch = self._make_lora_branch(**kwargs)

            # Get submodule forward hooks
            forward_pre_hooks = []
            forward_post_hooks = []
            for submodule in self.modules():
                for hook in submodule._forward_pre_hooks.values():
                    forward_pre_hooks.append((submodule, hook))
                for hook in submodule._forward_hooks.values():
                    forward_post_hooks.append((submodule, hook))

            # Attempt to emulate submodule forward hooks if needed
            # Note: Assume hooks do not interact with submodule inputs
            # or outputs since they are internal to the op fuser.
            if forward_pre_hooks:

                def forward_pre_hook(module, *_) -> None:
                    for submodule, hook in forward_pre_hooks:
                        # Assume that hook does not interact with
                        # input
                        hook(submodule, None)

                main_branch.register_forward_pre_hook(forward_pre_hook)
            if forward_post_hooks:

                def forward_post_hook(module, *_) -> None:
                    for submodule, hook in forward_post_hooks:
                        # Assume that hook does not interact with
                        # input or output
                        hook(submodule, None, None)

                lora_branch.register_forward_hook(forward_post_hook)

            return main_branch, lora_branch

        def _make_main_branch(
            self,
            *,
            in_features: int,
            out_features: int,
            tensor_parallel_mode: Optional[str],
            tensor_parallel_group: Optional[torch.distributed.ProcessGroup],
            sequence_parallel: bool,
            accumulate_into_main_grad: bool,
        ) -> te.ops.Sequential:
            """Construct fused module for main branch (norm + fork + linear)"""

            # Check wrapped linear class
            if not isinstance(self.to_wrap, (te.Linear, te.LayerNormLinear, torch.nn.Linear)):
                raise ValueError(f"Unsupported class for wrapped linear ({self.to_wrap.__class__.__name__})")

            # Ops in main branch
            main_branch = te.ops.Sequential()

            # Norm op
            if isinstance(self.to_wrap, te.LayerNormLinear):
                norm_type = self.to_wrap.normalization
                kwargs = {
                    "eps": self.to_wrap.eps,
                    "device": "meta",
                    "dtype": self.to_wrap.layer_norm_weight.dtype,
                    "zero_centered_gamma": self.to_wrap.zero_centered_gamma,
                }
                op = None
                if norm_type == "LayerNorm":
                    op = te.ops.LayerNorm(in_features, **kwargs)
                    op.weight = self.to_wrap.layer_norm_weight
                    op.bias = self.to_wrap.layer_norm_bias
                elif norm_type == "RMSNorm":
                    op = te.ops.RMSNorm(in_features, **kwargs)
                    op.weight = self.to_wrap.layer_norm_weight
                else:
                    raise ValueError(f"Unsupported normalization ({norm_type})")
                main_branch.append(op)
                main_branch.append(te.ops.Quantize(forward=True, backward=False))

            # Fork to LoRA branch
            # Note: GEMM with beta=1 in backward pass
            main_branch.append(te.ops.MakeExtraOutput(in_place=True))

            # Linear op
            weight = self.to_wrap.weight
            bias = self.to_wrap.bias
            if isinstance(bias, torch.Tensor) and bias.numel() == 0:
                bias = None
            op = te.ops.Linear(
                in_features,
                out_features,
                bias=bias is not None,
                device="meta",
                dtype=weight.dtype,
                tensor_parallel_mode=tensor_parallel_mode,
                tensor_parallel_group=tensor_parallel_group,
                sequence_parallel=sequence_parallel,
                accumulate_into_main_grad=accumulate_into_main_grad,
            )
            op.weight = weight
            op.bias = bias
            main_branch.append(op)

            return main_branch

        def _make_lora_branch(
            self,
            *,
            in_features: int,
            out_features: int,
            tensor_parallel_mode: Optional[str],
            tensor_parallel_group: Optional[torch.distributed.ProcessGroup],
            sequence_parallel: bool,
            accumulate_into_main_grad: bool,
        ) -> te.ops.Sequential:
            """Construct fused module for LoRA branch (lora_a + lora_b + add)"""

            from nemo.collections.llm.peft.utils import ParallelLinearAdapter

            # Extract params from LoRA adapter
            lora_a_weight = None
            lora_b_weight = None
            lora_dim = None
            dropout = 0
            dropout_position = None
            scale = None
            if isinstance(self.adapter, (LinearAdapter, TELinearAdapter)):
                lora_a_weight = self.adapter.lora_a.weight
                lora_b_weight = self.adapter.lora_b.weight
                lora_dim = lora_b_weight.size(1)
                dropout = self.adapter.dropout.p
                dropout_position = self.adapter.dropout_position
                scale = self.adapter.scale
            elif isinstance(self.adapter, ParallelLinearAdapter):
                lora_a_weight = self.adapter.linear_in.weight
                lora_b_weight = self.adapter.linear_out.weight
                lora_dim = lora_b_weight.size(1)
                if self.adapter.dropout is not None:
                    dropout = self.adapter.dropout.p
                dropout_position = self.adapter.dropout_position
                scale = self.adapter.alpha / self.adapter.dim
            else:
                raise ValueError(f"Unsupported class for LoRA adapter ({self.adapter.__class__.__name__})")

            # Ops in LoRA branch
            lora_branch = te.ops.Sequential()

            # LoRA pre-processing
            if dropout > 0 and dropout_position == "pre":
                lora_branch.append(te.ops.Dropout(dropout))

            # LoRA A linear op
            op = te.ops.Linear(
                in_features,
                lora_dim,
                bias=False,
                device="meta",
                dtype=lora_a_weight.dtype,
                tensor_parallel_mode=tensor_parallel_mode,
                tensor_parallel_group=tensor_parallel_group,
                sequence_parallel=sequence_parallel,
                accumulate_into_main_grad=accumulate_into_main_grad,
            )
            op.weight = lora_a_weight
            lora_branch.append(op)

            # LoRA B linear op
            if tensor_parallel_mode == "column":
                # All-gather along dim -1
                raise NotImplementedError("Column tensor parallelism is not yet supported")
            op = te.ops.Linear(
                lora_dim,
                out_features,
                bias=False,
                device="meta",
                dtype=lora_b_weight.dtype,
                tensor_parallel_mode=None if tensor_parallel_mode is None else "column",
                tensor_parallel_group=tensor_parallel_group,
                sequence_parallel=False,
                accumulate_into_main_grad=accumulate_into_main_grad,
            )
            op.weight = lora_b_weight
            lora_branch.append(op)

            # LoRA post-processing
            if scale != 1:
                lora_branch.append(te.ops.ConstantScale(scale))
            if dropout > 0 and dropout_position == "post":
                lora_branch.append(te.ops.Dropout(dropout))
            if tensor_parallel_mode == "row":
                # All-gather along dim -1
                raise NotImplementedError("Row tensor parallelism is not yet supported")

            # Add with main branch
            # Note: GEMM with beta=1 in forward pass
            lora_branch.append(te.ops.AddExtraInput(in_place=True))

            return lora_branch

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
            # pylint: disable=C0115,C0116

            # Construct fused impl if needed
            # Note: We initialize during the first forward pass in
            # case the params are modified after the constructor.
            # Note: The fused impl is stored in a tuple to avoid
            # registering submodules.
            if self._fused_branches is None:
                self._fused_branches = self._make_fused_branches()

            # Apply fused impl
            main_branch, lora_branch = self._fused_branches
            linear_output, linear_input = main_branch(x)
            with te.fp8_autocast(enabled=False):
                out = lora_branch(linear_input, linear_output)
            return out, None


if HAVE_TE:

    class TELinearAdapter(te.Linear):
        """
        TELinear + LoRA, maintains ckpts structrue (i.e. Linear's weight/bias remain at the same FQN)

        The _init_wrapper and _forward methods provide the LoRA functionality. We want to be able to
        use those inside LinearAdapter but also for monkey-patching modules, without repeating the
        same code -> therefore those are decorated with @staticmethod.

        Args:
            orig_linear (nn.Module): the linear module to augment.
            dim (int): lora's dim in_features -> dim -> out_features.
            alpha (int): lora's scaling alpha.
            dropout (float): dropout prob (default: 0.0).
            dropout_position (str): where to apply dropout rel. to lora (choices= ['pre', 'post'], default=post)
            lora_A_init_method (str): init method for lora_A (choices= ['xavier', 'uniform'])
            lora_dtype (torch.dtype): weight's dtype, by default will use orig_linear's but if they
            are quantized weights (e.g. 4bit) needs to be specified explicitly.
        """

        def __init__(
            self,
            orig_linear,
            dim=8,
            alpha=32,
            dropout=0.0,
            dropout_position='post',
            lora_A_init_method='xavier',
            lora_dtype=None,
        ):
            assert orig_linear.__class__ == te.Linear
            # TELinear has bias set to empty tensor
            has_bias = orig_linear.bias is not None and orig_linear.bias.shape[0] != 0
            super(TELinearAdapter, self).__init__(
                in_features=orig_linear.in_features,
                out_features=orig_linear.out_features,
                bias=has_bias,
                device=orig_linear.weight.device,
                params_dtype=orig_linear.weight.dtype,
            )
            # copy weights
            self.weight.data.copy_(orig_linear.weight.data)
            if has_bias:
                self.bias.data.copy_(orig_linear.bias.data)
            # initialize the adapter
            TELinearAdapter._init_adapter(
                self,
                dim=dim,
                alpha=alpha,
                dropout=dropout,
                dropout_position=dropout_position,
                lora_A_init_method=lora_A_init_method,
                lora_dtype=lora_dtype,
            )

        @torch.no_grad
        @staticmethod
        def _init_adapter(
            obj,
            dim=8,
            alpha=32,
            dropout=0.0,
            dropout_position='post',
            lora_A_init_method='xavier',
            lora_dtype=None,
        ):
            """Adds LoRA weights to obj. The obj is either a LinearAdapter or an nn.Module (when
            monkey-patching).

            Args:
                obj (LinearAdapter | nn.Module): input module to adapt.
                dim (int): lora's dim in_features -> dim -> out_features.
                alpha (int): lora's scaling alpha.
                dropout (float): dropout prob (default: 0.0).
                dropout_position (str): where to apply dropout rel. to lora (choices= ['pre', 'post'], default=post)
                lora_A_init_method (str): init method for lora_A (choices= ['xavier', 'uniform'])
                lora_dtype (torch.dtype): weight's dtype, by default will use orig_linear's but if they
                are quantized weights (e.g. 4bit) needs to be specified explicitly.
            """
            obj.dim = dim
            obj.scale = alpha / dim

            # Freezer
            device = obj.weight.device
            obj.weight.requires_grad = False
            if obj.bias is not None:
                obj.bias.requires_grad = False

            in_features = obj.in_features
            out_features = obj.out_features
            dtype = lora_dtype or obj.weight.dtype

            obj.lora_a = nn.Linear(in_features, dim, bias=False, dtype=dtype, device=device)
            obj.lora_b = nn.Linear(dim, out_features, bias=False, dtype=dtype, device=device)
            if lora_A_init_method == 'xavier':
                torch.nn.init.uniform_(obj.lora_a.weight.data)
            else:
                nn.init.kaiming_uniform_(obj.lora_a.weight.data, a=math.sqrt(5))
            obj.lora_b.weight.data.fill_(0)
            obj.dropout = nn.Dropout(p=dropout)
            assert dropout_position in ['pre', 'post'], dropout_position
            obj.dropout_position = dropout_position

        def forward(self, x):
            # pylint: disable=C0115,C0116
            res = super(TELinearAdapter, self).forward(x)
            if self.dropout_position == 'pre':
                x = self.dropout(x)
            # LoRA fwd is performed in original precision regardless of FP8 enabled
            lora_res = self.lora_b(self.lora_a(x))
            lora_res = lora_res * self.scale
            if self.dropout_position == 'post':
                lora_res = self.dropout(lora_res)
            return res + lora_res


class LinearAdapter(nn.Linear):
    """
    Linear + LoRA, maintains ckpts structrue (i.e. Linear's weight/bias remain at the same FQN)

    The _init_wrapper and _forward methods provide the LoRA functionality. We want to be able to
    use those inside LinearAdapter but also for monkey-patching modules, without repeating the
    same code -> therefore those are decorated with @staticmethod.

    Args:
        orig_linear (nn.Module): the linear module to augment.
        dim (int): lora's dim in_features -> dim -> out_features.
        alpha (int): lora's scaling alpha.
        dropout (float): dropout prob (default: 0.0).
        dropout_position (str): where to apply dropout rel. to lora (choices= ['pre', 'post'], default=post)
        lora_A_init_method (str): init method for lora_A (choices= ['xavier', 'uniform'])
        lora_dtype (torch.dtype): weight's dtype, by default will use orig_linear's but if they
        are quantized weights (e.g. 4bit) needs to be specified explicitly.
    """

    def __init__(
        self,
        orig_linear,
        dim=8,
        alpha=32,
        dropout=0.0,
        dropout_position='post',
        lora_A_init_method='xavier',
        lora_dtype=None,
    ):
        assert isinstance(orig_linear, nn.Linear)
        super(LinearAdapter, self).__init__(
            in_features=orig_linear.in_features,
            out_features=orig_linear.out_features,
            bias=orig_linear.bias is not None,
            device=orig_linear.weight.device,
            dtype=orig_linear.weight.dtype,
        )
        # copy weights
        self.weight.data.copy_(orig_linear.weight.data)
        if orig_linear.bias is not None:
            self.bias.data.copy_(orig_linear.bias.data)
        # initialize the adapte
        LinearAdapter._init_adapter(
            self,
            dim=dim,
            alpha=alpha,
            dropout=dropout,
            dropout_position=dropout_position,
            lora_A_init_method=lora_A_init_method,
            lora_dtype=lora_dtype,
        )

    @torch.no_grad
    @staticmethod
    def _init_adapter(
        obj,
        dim=8,
        alpha=32,
        dropout=0.0,
        dropout_position='post',
        lora_A_init_method='xavier',
        lora_dtype=None,
    ):
        """Adds LoRA weights to obj. The obj is either a LinearAdapter or an nn.Module (when
        monkey-patching).

        Args:
            obj (LinearAdapter | nn.Module): input module to adapt.
            dim (int): lora's dim in_features -> dim -> out_features.
            alpha (int): lora's scaling alpha.
            dropout (float): dropout prob (default: 0.0).
            dropout_position (str): where to apply dropout rel. to lora (choices= ['pre', 'post'], default=post)
            lora_A_init_method (str): init method for lora_A (choices= ['xavier', 'uniform'])
            lora_dtype (torch.dtype): weight's dtype, by default will use orig_linear's but if they
            are quantized weights (e.g. 4bit) needs to be specified explicitly.
        """
        obj.dim = dim
        obj.scale = alpha / dim

        # Freezer
        device = obj.weight.device
        obj.weight.requires_grad = False
        if obj.bias is not None:
            obj.bias.requires_grad = False

        in_features = obj.in_features
        out_features = obj.out_features
        dtype = lora_dtype or obj.weight.dtype

        obj.lora_a = nn.Linear(in_features, dim, bias=False, dtype=dtype, device=device)
        obj.lora_b = nn.Linear(dim, out_features, bias=False, dtype=dtype, device=device)
        if lora_A_init_method == 'xavier':
            torch.nn.init.uniform_(obj.lora_a.weight.data)
        else:
            nn.init.kaiming_uniform_(obj.lora_a.weight.data, a=math.sqrt(5))
        obj.lora_b.weight.data.fill_(0)
        obj.dropout = nn.Dropout(p=dropout)
        assert dropout_position in ['pre', 'post'], dropout_position
        obj.dropout_position = dropout_position

    def forward(self, x):
        # pylint: disable=C0115,C0116
        # If LinearAdapter is used to monkey-patch a nn.Linear module, we want to use nn.Linear's
        # forward in the case where it uses quantized weights. We store a reference to nn.Linear's
        # forward in `super_fwd` attribute. If the attribute does not exist we do the usual linear.
        if (fwd := getattr(self, 'super_fwd', None)) is not None:
            assert fwd != self.forward
            res = fwd(x)
        else:
            res = F.linear(x, self.weight, self.bias)

        if self.dropout_position == 'pre':
            x = self.dropout(x)
        lora_res = self.lora_b(self.lora_a(x))
        lora_res = lora_res * self.scale
        if self.dropout_position == 'post':
            lora_res = self.dropout(lora_res)
        return res + lora_res


def patch_linear_module(
    orig_linear,
    dim=8,
    alpha=32,
    dropout=0.0,
    dropout_position='post',
    lora_A_init_method='xavier',
    lora_dtype=None,
):
    """Monkey-patches a nn.Linear (orig_linear param) to be a LinearAdapter, for all purposes
    think of this function as replacing a nn.Linear with a LinearAdapter defined above.

    The orig_linear might not contain valid weights, for example, the given orig_linear was
    initialized within a context-manager that uses a "meta" device. Therefore, we cannot copy
    the weight/bias from the orig_linear to the LinearAdapter, since those have not been allocated,

    To circumvent this scenario, LinearAdapter's additional functionality (_init_adapter, _forward)
    is based on static functions, so that we can use them for patching or when allocating a
    new LinearAdapter object.

    Args:
        orig_linear (nn.Linear): the module we add adapter to.
        dim (int, optional): Lora dim. Defaults to 8.
        alpha (int, optional): Lora alpha scale. Defaults to 32.
        dropout (float, optional): dropout prob. Defaults to 0.0.
        dropout_position (str, optional): location to apply dropout wrt lora.
            Defaults to 'post' (choices: 'pre', 'post').
        lora_A_init_method (str, optional): lora_a init method. Defaults to 'xavier'.
        lora_dtype (_type_, optional): Lora weights' dtype. By default will use orig_linear's dtype
        but orig_linear might use non-trainable dtype (e.g., 4bit), in which case the user must
        specify the dtype manually. Defaults to None.

    Returns:
        (nn.Module): the monkey-patched (nn.Linear + LoRA) nn.Module
    """

    assert isinstance(orig_linear, nn.Linear) or orig_linear.__class__ == te.Linear
    assert not hasattr(orig_linear, 'super_fwd'), orig_linear.super_fwd

    if isinstance(orig_linear, nn.Linear):
        LinearAdapter._init_adapter(orig_linear, dim, alpha, dropout, dropout_position, lora_A_init_method, lora_dtype)
        cls = orig_linear.__class__
        new_cls = type('PatchedLinearAdapter', (LinearAdapter, cls), {})
    elif orig_linear.__class__ == te.Linear:
        TELinearAdapter._init_adapter(
            orig_linear, dim, alpha, dropout, dropout_position, lora_A_init_method, lora_dtype
        )
        cls = orig_linear.__class__
        new_cls = type('PatchedTELinearAdapter', (TELinearAdapter, cls), {})
    else:
        raise NotImplementedError("Expected isinstance(orig_linear, (nn.Linear, te.Linear))")

    # If the model uses quantized weights, we want to use orig_linear's forward
    if (
        getattr(orig_linear, 'quant_state', None) is not None
        and orig_linear.quant_state.__class__ == bitsandbytes.functional.QuantState
    ):
        orig_linear.super_fwd = orig_linear.forward

    orig_linear.__class__ = new_cls
    return orig_linear


@dataclass
class LoRA(PEFT, ModuleMatcher):
    """
    Implements the LoRA (Low-Rank Adaptation) module for parameter-efficient fine-tuning.

    LoRA uses a low-rank projection to adapt the weights of a pre-trained model to a new downstream task.
    This class facilitates the application of LoRA to specific modules within the model architecture.

    Args:
        target_modules (list[str], optional): A list of module names to apply LoRA to.
            Defaults to all linear layers ['linear_qkv', 'linear_proj', 'linear_fc1', 'linear_fc2'].
                - 'linear_qkv': Apply LoRA to the fused linear layer used for query, key, and value projections
                                in self-attention.
                - 'linear_proj': Apply LoRA to the linear layer used for projecting the output of self-attention.
                - 'linear_fc1': Apply LoRA to the first fully-connected layer in MLP.
                - 'linear_fc2': Apply LoRA to the second fully-connected layer in MLP.
            Target modules can also contain wildcards. For example, you can specify
                target_modules=['*.layers.0.*.linear_qkv', '*.layers.1.*.linear_qkv'] to add LoRA to only linear_qkv
                on the first two layers.
        exclude_modules (list[str], optional): A list of module names not to apply LoRa to. It will
            match all nn.Linear & nn.Linear-adjacent modules whose name does not match any string in
            exclude_modules. If used, will require target_modules to be empty list or None.
        dim (int): Dimension of the low-rank projection space. Defaults to 32.
        alpha (int): Weighting factor for the low-rank projection. Defaults to 32.
        dropout (float): Dropout rate for the low-rank projection. Defaults to 0.0.
        dropout_position (Literal['pre', 'post'], optional): Position for applying dropout.
            Can be 'pre' (before the low-rank projection) or 'post' (after). Defaults to 'pre'.
        a2a_experimental (bool): Enables the experimental All-to-All (A2A) communication strategy. Defaults to False.
        dropout_recompute (bool): Enables dropout recompute using Thunder JIT compilation. When True,
            applies thunder.jit() to the dropout layer for memory-efficient training by recomputing
            dropout activations during backward pass instead of storing them.
        lora_dtype (torch.dtype): Parameter data type for LoRA weights. Default None (will use model's dtype).

    Example:
    --------
        >>> from nemo.collections import llm
        >>> lora = llm.peft.LoRA(target_modules=['linear_qkv', 'linear_proj'], dim=32)
        >>> model = llm.Mistral7BModel(model_transform=lora)
        >>> # (set up trainer and data)
        >>> trainer.fit(model, data)

    References:
    -----------
        Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2021).
        LoRA: Low-Rank Adaptation of Large Language Models. arXiv preprint arXiv:2106.09685.
        https://arxiv.org/abs/2106.09685

    )
    """

    target_modules: list[str] = field(
        default_factory=lambda: ['linear_qkv', 'linear_proj', 'linear_fc1', 'linear_fc2']
    )
    dim: int = 32
    alpha: int = 32
    dropout: float = 0.0
    dropout_position: Literal['pre', 'post'] = 'pre'
    lora_A_init_method: str = "xavier"
    lora_B_init_method: str = "zero"
    a2a_experimental: bool = False
    lora_dtype: torch.dtype = None
    dropout_recompute: bool = False

    def transform(self, m: nn.Module, name=None, prefix=None):
        """
        Applies LoRA to a specific module within the model architecture.

        Args:
            m (nn.Module): The module to apply LoRA to.
            name (str, optional): Name of the module (if applicable). Defaults to None.
            prefix (str, optional): Prefix for the module name (if applicable). Defaults to None.

        Returns:
            nn.Module: The modified module with LoRA applied, or the original module if not a target.
        """
        from nemo.collections.llm.peft.utils import ParallelLinearAdapter

        if (ans := self.match(m, name, prefix)) is not None:
            (match, full_name) = ans
            if isinstance(m, nn.Linear) or m.__class__ == te.Linear:
                # Will use the `patch_linear_module` function if:
                # - is FSDP v1
                # - is DTensor (has _local_tensor attribute)
                # - has quant_state attribute
                if (
                    self._add_via_setattr
                    or hasattr(m.weight.data, '_local_tensor')
                    or (
                        getattr(m, 'quant_state', None) is not None
                        and m.quant_state.__class__ == bitsandbytes.functional.QuantState
                    )
                ):
                    lora_cls = patch_linear_module
                elif HAVE_TE and m.__class__ == te.Linear:
                    lora_cls = TELinearAdapter
                else:
                    lora_cls = LinearAdapter

                # Construct LoRA module
                return lora_cls(
                    m,
                    dim=self.dim,
                    alpha=self.alpha,
                    dropout=self.dropout,
                    lora_A_init_method=self.lora_A_init_method,
                    lora_dtype=self.lora_dtype,
                )

            input_is_parallel, in_features, out_features, disable_sp_comm, base_linear_is_parallel = (
                get_adapter_attributes_from_linear(m)
            )
            enable_op_fuser = (
                HAVE_TE_FUSED_LORA
                and hasattr(m, "config")
                and getattr(m.config, "use_transformer_engine_op_fuser", False)
                # TP not yet supported
                and parallel_state.get_tensor_model_parallel_world_size() == 1
            )
            logging.info(f"Adding lora to: {full_name}")
            adapter = ParallelLinearAdapter(
                in_features,
                out_features,
                self.dim,
                base_linear_name=full_name,
                activation='identity',
                norm_type=None,
                column_init_method=self.lora_A_init_method,
                row_init_method=self.lora_B_init_method,
                gather_output=False,
                input_is_parallel=input_is_parallel,
                dropout=self.dropout,
                dropout_position=self.dropout_position,
                model_parallel_config=getattr(m, "config", None),
                alpha=self.alpha,
                is_expert=is_expert_linear(full_name),
                a2a_experimental=self.a2a_experimental,
                disable_sequence_parallel_comm=disable_sp_comm,
                dropout_recompute=self.dropout_recompute,
                base_linear_is_parallel=base_linear_is_parallel,
            )
            if enable_op_fuser:
                return TEFusedLoRALinear(m, adapter)
            else:
                return LoRALinear(m, adapter)
        return m


class LoRAMerge(PEFT):
    """
    Implements the LoRA weight merge for parameter-efficient fine-tuning.

    Example:
    --------
        >>> from nemo.collections.llm.peft.lora import LoRAMerge
        >>> lora_merge = LoRAMerge()
        >>> merged_model = lora_merge(trainer.strategy.megatron_parallel)
    """

    @torch.no_grad()
    def transform(self, m: nn.Module, name=None, prefix=None):
        """
        Merges the LoRA adapter with the base model weights.

        Args:
            m (nn.Module): The module to apply LoRA merge to.
            name (str, optional): Name of the module to merge. Defaults to None.
            prefix (str, optional): Prefix for the module name. Defaults to None.

        Returns:
            nn.Module: The modified module with the LoRA adapter merged into the base model weights.
        """

        if not isinstance(m, LoRALinear):
            return m
        logging.info(f'merging {(prefix if prefix else "") + "." + (name if name else "")}')
        lora_weight = m.adapter.alpha / m.adapter.dim * m.adapter.linear_out.weight @ m.adapter.linear_in.weight
        if hasattr(m.to_wrap, "weight"):
            base_weight = m.to_wrap.weight
            merged_weight = base_weight + lora_weight.to(base_weight.device)
            m.to_wrap.weight.data = merged_weight
        else:  # TE Grouped Linear
            for i in range(m.to_wrap.num_gemms):
                base_weight = getattr(m.to_wrap, f"weight{i}")
                merged_weight = base_weight + lora_weight.to(base_weight.device)
                getattr(m.to_wrap, f"weight{i}").data = merged_weight
        return m
