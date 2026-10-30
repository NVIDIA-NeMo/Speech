# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

from nemo.collections.asr.parts.turbo_transducer.backend import TurboTransducerBackend
from nemo.core.utils.optional_libs import CUTILE_AVAILABLE, TRITON_AVAILABLE

if TRITON_AVAILABLE:
    from nemo.collections.asr.parts.turbo_transducer.rnnt_fused_joint_triton import rnnt_fused_joint_triton


def rnnt_fused_joint(
    encoder_output_projected: torch.Tensor,
    predictor_output_projected: torch.Tensor,
    targets: torch.Tensor,
    tgt_lengths: torch.Tensor,
    src_lengths: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    blank_id: int,
    activation: str = "relu",
    dropout_p: float = 0.0,
    dropout_seed: int | None = None,
    use_high_precision_mm: bool = False,
    input_grad_full_precision: bool = True,
    weight_grad_full_precision: bool = True,
    backend: str | TurboTransducerBackend | None = TurboTransducerBackend.TRITON,
    mm_full_precision: bool = False,
):
    """Fuse Joint hidden computation and vocabulary projection into target/blank log-probabilities.

    BF16 projected inputs use BF16 matmul operands even with FP32 master weights. ``mm_full_precision=True``
    selects FP32 operands; ``use_high_precision_mm=True`` selects FP32 operands and IEEE input precision.
    Gradient accumulation precision is controlled separately by the gradient flags.
    """
    backend = TurboTransducerBackend(backend) if backend is not None else None

    if backend is None:
        if TRITON_AVAILABLE:
            backend = TurboTransducerBackend.TRITON
        elif CUTILE_AVAILABLE:
            backend = TurboTransducerBackend.CUTILE
        else:
            raise Exception("No backend")

    match backend:
        case TurboTransducerBackend.TRITON:
            if not TRITON_AVAILABLE:
                raise ModuleNotFoundError("Triton is unavailable")
            target_logprobs, blank_logprobs = rnnt_fused_joint_triton(
                encoder_output_projected=encoder_output_projected,
                predictor_output_projected=predictor_output_projected,
                targets=targets,
                tgt_lengths=tgt_lengths,
                src_lengths=src_lengths,
                weight=weight,
                bias=bias,
                blank_id=blank_id,
                activation=activation,
                dropout_p=dropout_p,
                dropout_seed=dropout_seed,
                use_high_precision_mm=use_high_precision_mm,
                input_grad_full_precision=input_grad_full_precision,
                weight_grad_full_precision=weight_grad_full_precision,
                mm_full_precision=mm_full_precision,
            )
        case TurboTransducerBackend.CUTILE:
            if not CUTILE_AVAILABLE:
                raise ModuleNotFoundError("cuTile is unavailable")
            raise NotImplementedError("cuTile support not yet implemented")
        case _:
            raise Exception("Not implemented backend")
    return target_logprobs, blank_logprobs
