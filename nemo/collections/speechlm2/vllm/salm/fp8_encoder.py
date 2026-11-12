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

"""FP8 audio encoder for SpeechLM checkpoints quantized with ModelOpt FP8.

A checkpoint describes its audio encoder's quantization the way it describes its
decoder's, in the ModelOpt ``quantization_config`` of ``config.json``
(``quant_algo: FP8``): every Linear of the perception module that no ``ignore``
entry matches is quantized. Such a layer stores an FP8 E4M3 ``weight``, a float32
scalar ``weight_scale`` and a float32 scalar ``input_scale``, the static per-tensor
activation scale, both in the divisor convention ``x_fp8 = x / scale``. A
speaker-aware checkpoint keeps its diarizer, the ASR encoder's input projection and
the connector unquantized by listing them::

    "ignore": [..., "perception.encoder.diarization_model*",
               "perception.encoder.asr_encoder.pre_encode*", "perception.proj"]

The audio encoder runs as plain PyTorch inside the plugin rather than through vLLM's
quantized layers, so ``build_fp8_encoder`` applies vLLM's exclusion rule to the
perception Linears itself and builds ``FP8Linear`` layers for the quantized ones
when the model is constructed; the checkpoint then loads into them directly.

A ``quantization_config`` without any ``perception`` entry predates encoder
quantization (its decoder was quantized on its own), so its encoder stays unquantized.

A ModelOpt ``MIXED_PRECISION`` checkpoint (for example an NVFP4 decoder with FP8 layers)
has no such rule: it names every quantized layer in ``quantized_layers``. There an
encoder Linear is FP8 when listed with ``quant_algo: FP8`` and unquantized when not
listed::

    "quantized_layers": {..., "perception.encoder.asr_encoder.layers.0.attn.w_qkv": {"quant_algo": "FP8"}}
"""

from collections.abc import Callable, Mapping
from typing import Any, Optional

import torch
from torch import nn

from nemo.utils import logging

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = 448.0
_PERCEPTION_PREFIX = "perception."


class FP8Linear(nn.Linear):
    """Linear layer with FP8 E4M3 weights and static per-tensor scales, run with vLLM's CUTLASS FP8 GEMM.

    The module is built empty and filled by ``load_state_dict``: ``weight``,
    ``weight_scale`` and ``input_scale`` are persistent buffers named and shaped like
    ModelOpt's FP8 checkpoint tensors. Dtype casts of an enclosing module convert only
    the bias; the FP8 weight and the float32 scales keep their dtypes and follow
    device moves. It subclasses ``nn.Linear`` because encoders choose how to call an
    input projection with ``isinstance(module, nn.Linear)``.

    Args:
        in_features: Input width. CUTLASS needs a multiple of 16.
        out_features: Output width.
        bias: Whether the layer has a bias.
        bias_dtype: Dtype of the bias parameter.
        device: Device for the empty buffers.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool,
        bias_dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None,
    ) -> None:
        # nn.Linear.__init__ would allocate full-precision parameters that the FP8 buffers replace.
        nn.Module.__init__(self)
        if in_features % 16 != 0:
            raise ValueError(f"in_features={in_features} must be a multiple of 16 for the CUTLASS FP8 GEMM")
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("weight", torch.empty(out_features, in_features, dtype=FP8_DTYPE, device=device))
        self.register_buffer("weight_scale", torch.empty((), dtype=torch.float32, device=device))
        self.register_buffer("input_scale", torch.empty((), dtype=torch.float32, device=device))
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=bias_dtype, device=device), requires_grad=False)

    def _apply(self, fn: Callable[[torch.Tensor], torch.Tensor], recurse: bool = True) -> "FP8Linear":
        """Apply ``fn`` to the bias as usual but only its device change to the buffers.

        The perception module is cast to bfloat16 as a whole; applied to the buffers,
        that cast would turn the FP8 weight into bfloat16 values and round the scales.
        """
        buffers = dict(self._buffers)
        self._buffers.clear()
        try:
            super()._apply(fn, recurse)
        finally:
            for name, buf in buffers.items():
                if buf is not None:
                    device = fn(torch.empty(0, dtype=buf.dtype, device=buf.device)).device
                    buf = buf.to(device=device)
                self._buffers[name] = buf
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize ``x`` to FP8 with the static input scale and multiply by the FP8 weight, returning ``x.dtype``."""
        import vllm._custom_ops as ops

        out_shape = (*x.shape[:-1], self.out_features)
        # The quantization kernel needs a contiguous last dimension; a transposed input reshapes to a strided view.
        xq, x_scale = ops.scaled_fp8_quant(x.reshape(-1, self.in_features).contiguous(), self.input_scale)
        # CUTLASS wants a column-major B, which the transpose gives as a view.
        out = ops.cutlass_scaled_mm(xq, self.weight.t(), x_scale, self.weight_scale, x.dtype, self.bias)
        return out.reshape(out_shape)

    def extra_repr(self) -> str:
        """Summarize the layer shape and quantization."""
        return f"in_features={self.in_features}, out_features={self.out_features}, fp8_e4m3, static per-tensor scales"


def build_fp8_encoder(perception: nn.Module, quant_config: Optional[Any]) -> int:
    """Build empty ``FP8Linear`` layers in place of the perception Linears that ``quant_config`` quantizes.

    Call this when the model is constructed, before any weights load: the
    checkpoint's FP8 tensors then load into these layers directly, and later dtype
    casts of the perception module leave them intact.

    Args:
        perception: The SpeechLM perception module, whose names start with ``perception.`` in the checkpoint.
        quant_config: vLLM's quantization config for the checkpoint, or None.

    Returns:
        The number of Linears replaced; 0 when ``quant_config`` does not quantize the encoder.

    Raises:
        NotImplementedError: If the encoder is quantized with anything but ModelOpt FP8.
        ValueError: If a quantized Linear's input width does not suit the CUTLASS FP8 GEMM.
    """
    if hasattr(quant_config, "get_name") and quant_config.get_name() == "modelopt_mixed":
        targets = _listed_fp8_targets(perception, quant_config)
    else:
        targets = _unexcluded_fp8_targets(perception, quant_config)
    if not targets:
        return 0

    replacements = [
        (
            name,
            FP8Linear(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                bias_dtype=module.bias.dtype if module.bias is not None else torch.bfloat16,
                device=module.weight.device,
            ),
        )
        for name, module in targets
    ]
    for name, layer in replacements:
        parent_name, _, child_name = name.rpartition(".")
        parent = perception.get_submodule(parent_name) if parent_name else perception
        setattr(parent, child_name, layer)

    logging.info(f"FP8 audio encoder: {len(replacements)} Linears built from quantization_config")
    return len(replacements)


def _unexcluded_fp8_targets(perception: nn.Module, quant_config: Optional[Any]) -> list[tuple[str, nn.Linear]]:
    """Perception Linears that a ModelOpt FP8 config quantizes: those no exclusion entry matches."""
    exclude_modules = getattr(quant_config, "exclude_modules", None) or ()
    if not hasattr(quant_config, "is_layer_excluded") or not any(
        str(entry).startswith("perception") for entry in exclude_modules
    ):
        return []
    targets = [
        (name, module)
        for name, module in perception.named_modules()
        if isinstance(module, nn.Linear)
        and not isinstance(module, FP8Linear)
        and not quant_config.is_layer_excluded(_PERCEPTION_PREFIX + name)
    ]
    if not targets:
        return []
    scheme = (quant_config.get_name(), getattr(quant_config, "quant_method", None))
    if scheme != ("modelopt", "FP8"):
        raise NotImplementedError(
            f"quantization_config quantizes {len(targets)} audio encoder Linear(s), e.g. "
            f"{_PERCEPTION_PREFIX + targets[0][0]!r}, as {scheme[0]} {scheme[1]}; the audio encoder supports only "
            "ModelOpt FP8 with static per-tensor scales"
        )
    return targets


def _listed_fp8_targets(perception: nn.Module, quant_config: Any) -> list[tuple[str, nn.Linear]]:
    """Perception Linears that a ModelOpt MIXED_PRECISION config lists in ``quantized_layers``."""
    quantized_layers = quant_config.quantized_layers
    targets = []
    others = {}
    for name, module in perception.named_modules():
        prefix = _PERCEPTION_PREFIX + name
        if not isinstance(module, nn.Linear) or isinstance(module, FP8Linear) or prefix not in quantized_layers:
            continue
        # vLLM's mixed-precision config checks exclusions before the per-layer table.
        if quant_config.is_layer_excluded(prefix):
            continue
        algo = str(quantized_layers[prefix].get("quant_algo", "")).upper()
        if algo == "FP8":
            targets.append((name, module))
        else:
            others[prefix] = algo
    if others:
        raise NotImplementedError(
            f"quantized_layers lists {len(others)} audio encoder Linear(s) as {sorted(set(others.values()))}, "
            f"e.g. {next(iter(others))!r}; the audio encoder supports only FP8 with static per-tensor scales"
        )
    return targets


def check_fp8_encoder_weights(perception: nn.Module, weights: Mapping[str, torch.Tensor]) -> None:
    """Fail if the checkpoint's FP8 encoder tensors do not match the layers ``build_fp8_encoder`` built.

    ``load_state_dict`` converts dtypes silently, so FP8 data copied into a bfloat16
    Linear, or bfloat16 data into an ``FP8Linear``, would load without an error.

    Args:
        perception: The perception module after ``build_fp8_encoder``.
        weights: The checkpoint's perception tensors, named relative to the perception module.

    Raises:
        ValueError: If an ``FP8Linear`` lacks an FP8 weight or a per-tensor scale, or the
            checkpoint stores an FP8 weight for a layer that is not quantized.
    """
    fp8_layers = {name for name, module in perception.named_modules() if isinstance(module, FP8Linear)}
    problems = []
    for name in sorted(fp8_layers):
        weight = weights.get(f"{name}.weight")
        if weight is None or weight.dtype != FP8_DTYPE:
            problems.append(f"{name}.weight is {'missing' if weight is None else weight.dtype}, not {FP8_DTYPE}")
        for scale_name in ("weight_scale", "input_scale"):
            scale = weights.get(f"{name}.{scale_name}")
            if scale is None or scale.numel() != 1:
                problems.append(
                    f"{name}.{scale_name} is {'missing' if scale is None else tuple(scale.shape)}, not one scale"
                )
    unquantized = sorted(
        key
        for key, tensor in weights.items()
        if tensor.dtype == FP8_DTYPE and key.rpartition(".")[0] not in fp8_layers
    )
    problems.extend(f"{key} is FP8 but quantization_config leaves its layer unquantized" for key in unquantized)
    if problems:
        shown = "; ".join(problems[:5]) + (f"; and {len(problems) - 5} more" if len(problems) > 5 else "")
        hint = (
            " (config.json's quantization_config must describe the encoder's FP8 layers; an "
            "encoder_quantization block is not read)"
            if unquantized
            else ""
        )
        raise ValueError(f"FP8 audio encoder weights do not match quantization_config: {shown}{hint}")
