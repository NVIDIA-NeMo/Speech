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

from collections.abc import Mapping
from contextlib import nullcontext
from math import gcd
from typing import Any

import torch
from omegaconf import DictConfig, ListConfig, OmegaConf


def get_config_value(cfg: Any, path: str, default: Any = None) -> Any:
    """Read a dotted config path from ``dict``/OmegaConf/object configs."""
    if cfg is None:
        return default
    if isinstance(cfg, (DictConfig, ListConfig)):
        return OmegaConf.select(cfg, path, default=default)

    current = cfg
    for key in path.split("."):
        if current is None:
            return default
        if isinstance(current, Mapping):
            if key not in current:
                return default
            current = current[key]
        else:
            current = getattr(current, key, default)
            if current is default:
                return default
    return current


def as_plain_container(cfg: Any) -> Any:
    """Convert OmegaConf containers to plain Python containers."""
    if isinstance(cfg, (DictConfig, ListConfig)):
        return OmegaConf.to_container(cfg, resolve=True)
    return cfg


def has_torchao_fp8(cfg: Any) -> bool:
    """Return whether TorchAO FP8 training is enabled in the model config."""
    return bool(get_config_value(cfg, "fp8.enabled", False))


def has_te_fp8(automodel_backend_config: Any) -> bool:
    """Return whether Transformer Engine FP8 is configured in an Automodel backend config."""
    return get_config_value(automodel_backend_config, "te_fp8", None) is not None


def is_te_fp8_enabled(te_fp8_config: Any) -> bool:
    """Return whether a direct ``automodel_backend.te_fp8`` config is present."""
    return te_fp8_config is not None


def validate_fp8_config(cfg: Any) -> None:
    """Validate model FP8 config combinations."""
    if has_torchao_fp8(cfg) and has_te_fp8(get_config_value(cfg, "automodel_backend", None)):
        raise ValueError(
            "only one FP8 mode may be configured at a time. Configure either "
            "fp8 for TorchAO FP8 or automodel_backend.te_fp8 for "
            "Transformer Engine FP8, but not both."
        )


def maybe_apply_te_patches(automodel_backend_config: Any) -> None:
    """Apply Automodel's Transformer Engine runtime patches when TE FP8 is configured."""
    if not has_te_fp8(automodel_backend_config):
        return

    from nemo_automodel.shared.te_patches import apply_te_patches

    apply_te_patches()


def make_fp8_config(cfg: Any) -> Any:
    """Build Automodel's TorchAO FP8Config from SALMAutomodel config, or return ``None``."""
    if not has_torchao_fp8(cfg):
        return None

    from nemo_automodel.components.quantization.fp8 import build_fp8_config

    return build_fp8_config(as_plain_container(get_config_value(cfg, "fp8", None)))


def te_fp8_context(automodel_backend_config: Any):
    """Return a Transformer Engine FP8 autocast context for an Automodel backend config."""
    te_fp8_config = get_config_value(automodel_backend_config, "te_fp8", None)
    if te_fp8_config is None:
        return nullcontext()

    te_fp8_config = as_plain_container(te_fp8_config)
    if hasattr(te_fp8_config, "maybe_te_autocast"):
        return te_fp8_config.maybe_te_autocast()

    from nemo_automodel.components.models.common.utils import TEFp8Config

    if isinstance(te_fp8_config, Mapping):
        te_fp8_kwargs = dict(te_fp8_config)
        te_fp8_kwargs.pop("_target_", None)
        return TEFp8Config(**te_fp8_kwargs).maybe_te_autocast()
    if isinstance(te_fp8_config, str):
        return TEFp8Config(recipe=te_fp8_config).maybe_te_autocast()

    raise TypeError(
        "automodel_backend.te_fp8 must be null, a mapping, a recipe string, "
        "or a TEFp8Config-like object with maybe_te_autocast()."
    )


def validate_te_fp8_hidden_size(te_fp8_config: Any, hidden_size: int) -> None:
    """Validate TE FP8's GEMM alignment requirement for activation hidden size."""
    if is_te_fp8_enabled(te_fp8_config) and hidden_size % 16 != 0:
        raise ValueError(
            "Transformer Engine FP8 requires input hidden size to be divisible by 16; "
            f"got hidden_size={hidden_size}."
        )


def get_te_fp8_bshd_sequence_multiple(batch_size: int) -> int:
    """Return the minimal sequence-length multiple so B*T is divisible by 8."""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive; got {batch_size}.")
    return 8 // gcd(batch_size, 8)


def maybe_pad_bshd_inputs_for_te_fp8(
    te_fp8_config: Any,
    input_embeds: torch.Tensor,
    attention_mask: torch.Tensor | None,
    llm_kwargs: Mapping[str, Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, Any], int]:
    """Pad BSHD LLM inputs for TE FP8 and return the original sequence length.

    TE FP8 Linear requires the product of all input dimensions except the last
    to be divisible by 8 and the last dimension to be divisible by 16. For
    BSHD inputs this means ``B * T`` must be divisible by 8. Padding is appended
    on the sequence dimension and can be trimmed from logits after the LLM.
    """
    llm_kwargs = dict(llm_kwargs or {})
    if input_embeds.dim() != 3:
        return input_embeds, attention_mask, llm_kwargs, input_embeds.shape[0]
    original_seq_len = input_embeds.shape[1]
    if not is_te_fp8_enabled(te_fp8_config):
        return input_embeds, attention_mask, llm_kwargs, original_seq_len

    batch_size, seq_len, hidden_size = input_embeds.shape
    validate_te_fp8_hidden_size(te_fp8_config, hidden_size)

    seq_multiple = get_te_fp8_bshd_sequence_multiple(batch_size)
    pad = (-seq_len) % seq_multiple
    if pad == 0:
        return input_embeds, attention_mask, llm_kwargs, original_seq_len

    pad_embeds = torch.zeros(
        batch_size,
        pad,
        hidden_size,
        dtype=input_embeds.dtype,
        device=input_embeds.device,
    )
    input_embeds = torch.cat([input_embeds, pad_embeds], dim=1)

    if attention_mask is not None:
        # These are appended causal dummy tokens, not data-loader padding.
        # Mark them valid so their query rows are finite; real tokens cannot
        # attend to future dummy tokens through the causal mask.
        pad_mask = torch.ones(
            batch_size,
            pad,
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        attention_mask = torch.cat([attention_mask, pad_mask], dim=1)

    for key in ("position_ids", "cache_position"):
        value = llm_kwargs.get(key, None)
        if isinstance(value, torch.Tensor):
            llm_kwargs[key] = pad_sequence_tensor(value, seq_len, pad)

    return input_embeds, attention_mask, llm_kwargs, original_seq_len


def pad_sequence_tensor(tensor: torch.Tensor, seq_len: int, pad: int, pad_value: int = 0) -> torch.Tensor:
    """Right-pad a tensor when one of its sequence dimensions matches ``seq_len``."""
    if pad <= 0:
        return tensor
    if tensor.dim() == 1 and tensor.shape[0] == seq_len:
        padding = torch.full((pad,), pad_value, dtype=tensor.dtype, device=tensor.device)
        return torch.cat([tensor, padding], dim=0)
    if tensor.dim() >= 2 and tensor.shape[1] == seq_len:
        pad_shape = list(tensor.shape)
        pad_shape[1] = pad
        padding = torch.full(pad_shape, pad_value, dtype=tensor.dtype, device=tensor.device)
        return torch.cat([tensor, padding], dim=1)
    return tensor


def trim_fp8_padded_logits(logits: torch.Tensor, original_seq_len: int) -> torch.Tensor:
    """Trim sequence padding introduced by ``maybe_pad_bshd_inputs_for_te_fp8``."""
    if logits.dim() >= 3 and logits.shape[1] > original_seq_len:
        return logits[:, :original_seq_len]
    if logits.dim() == 2 and logits.shape[0] > original_seq_len:
        return logits[:original_seq_len]
    return logits


def maybe_pad_thd_padded_lengths_for_te_fp8(
    te_fp8_config: Any,
    padded_lens: list[int],
    *,
    cp_size: int = 1,
    tp_size: int = 1,
) -> list[int]:
    """Pad THD packed sequence lengths so local TE FP8 Linear inputs are aligned.

    This must run before context-parallel THD partitioning because ``cu_seqlens``
    is global metadata and CP partitioning derives local token indices from it.
    """
    if not is_te_fp8_enabled(te_fp8_config):
        return padded_lens
    if not padded_lens:
        return padded_lens

    cp_size = max(int(cp_size), 1)
    tp_size = max(int(tp_size), 1)
    total_multiple = 8 * cp_size * tp_size
    total_len = sum(padded_lens)
    pad = (-total_len) % total_multiple
    if pad == 0:
        return padded_lens

    padded_lens = list(padded_lens)
    padded_lens[-1] += pad
    if cp_size > 1:
        cp_multiple = 2 * cp_size
        if padded_lens[-1] % cp_multiple != 0:
            raise AssertionError(
                "Internal error: TE FP8 THD padding did not preserve context-parallel " f"alignment to {cp_multiple}."
            )
    return padded_lens


def maybe_precompute_float8_dynamic_scale_for_fsdp(cfg: Any, llm: Any, device_mesh: Any, use_fsdp: bool) -> None:
    """Run TorchAO's FSDP FP8 scale precompute hook when the config and mesh require it."""
    if not has_torchao_fp8(cfg):
        return
    if not bool(get_config_value(cfg, "fp8.precompute_float8_dynamic_scale_for_fsdp", False)):
        return
    if llm is None or not use_fsdp or device_mesh is None:
        return

    mesh_dim_names = getattr(device_mesh, "mesh_dim_names", ()) or ()
    if "dp_shard" not in mesh_dim_names:
        return
    try:
        dp_shard_size = device_mesh["dp_shard"].size()
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
        return
    if dp_shard_size <= 1:
        return

    from torchao.float8 import precompute_float8_dynamic_scale_for_fsdp

    precompute_float8_dynamic_scale_for_fsdp(llm)
