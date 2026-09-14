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

"""Token-flat encoder activations and conversions for sequence-packed ASR execution."""

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.utils._pytree import GetAttrKey, register_pytree_node

__all__ = [
    "PackedEncoderActivations",
    "pack_encoder_output",
    "packed_encoder_position_ids",
    "split_packed_data",
    "split_encoder_output",
    "unpack_encoder_output",
]


@dataclass(frozen=True)
class PackedEncoderActivations:
    """An ASR encoder batch stored without inter-utterance padding.

    Args:
        data: Valid encoder states concatenated in batch order, shape ``(T_total, D)``.
        lengths: Per-utterance token counts, shape ``(B,)`` and dtype ``int64``.
        cu_seqlens: Cumulative token counts, shape ``(B + 1,)`` and dtype ``int32``.
        max_seqlen: Largest value in ``lengths`` (zero for an empty batch).
        padding_value: Value to use when a downstream frame-stacking operation
            needs to complete the final partial group. A scalar applies to the
            whole batch; a ``(B, D)`` tensor supports sequence-specific normalized
            feature padding. Encoder outputs use zero.
        padded_length: Optional dense time dimension associated with the packed
            data. Frontends preserve it solely for dense-compatible augmentation
            RNG ranges; packed kernels continue to use ``max_seqlen``.

    The representation deliberately has an ASR-specific name rather than reusing
    :class:`torch.nn.utils.rnn.PackedSequence`: the latter stores sort/unsort metadata
    for recurrent networks and has a different layout contract.
    """

    data: Tensor
    lengths: Tensor
    cu_seqlens: Tensor
    max_seqlen: int
    padding_value: float | Tensor = 0.0
    padded_length: int | None = None

    def __post_init__(self) -> None:
        _validate_packed_encoder_activations(self)

    @property
    def batch_size(self) -> int:
        """Number of utterances represented by this output."""

        return int(self.lengths.numel())

    @property
    def total_tokens(self) -> int:
        """Number of valid (unpadded) encoder tokens."""

        return int(self.data.shape[0])

    def with_data(self, data: Tensor) -> "PackedEncoderActivations":
        """Reuse validated sequence metadata with replacement token data.

        Internal encoder stages use this to avoid revalidating unchanged CUDA
        offsets (and synchronizing the host) after every projection or fusion.
        """
        if data.ndim != 2 or data.shape[0] != self.total_tokens:
            raise ValueError(f"replacement data must have shape ({self.total_tokens}, D), got {tuple(data.shape)}.")
        if data.device != self.data.device:
            raise ValueError(f"replacement data must be on {self.data.device}, got {data.device}.")
        return _new_packed_encoder_activations(
            data,
            self.lengths,
            self.cu_seqlens,
            self.max_seqlen,
            padding_value=self.padding_value,
            padded_length=self.padded_length,
        )


def _packed_encoder_activations_flatten(packed: PackedEncoderActivations):
    values = [
        packed.data,
        packed.lengths,
        packed.cu_seqlens,
        packed.max_seqlen,
        packed.padding_value,
        packed.padded_length,
    ]
    return values, None


def _packed_encoder_activations_flatten_with_keys(packed: PackedEncoderActivations):
    values, context = _packed_encoder_activations_flatten(packed)
    names = (
        "data",
        "lengths",
        "cu_seqlens",
        "max_seqlen",
        "padding_value",
        "padded_length",
    )
    keyed_values = [(GetAttrKey(name), value) for name, value in zip(names, values, strict=True)]
    return keyed_values, context


def _packed_encoder_activations_unflatten(values, context):
    del context
    # Pytree transforms may temporarily replace every leaf with a sentinel. In
    # particular, torch.utils.checkpoint._infer_device_type maps all leaves to
    # None merely to inspect CUDA tensors. Re-running public-constructor
    # validation on that transient object would reject the transform itself.
    # Real PackedEncoderActivations still enter through the validating public
    # constructor; pytree reconstruction preserves already-validated metadata.
    return _new_packed_encoder_activations(*values)


# FSDP2 discovers tensors that need post-forward/pre-backward hooks by walking
# the output pytree. Register this container so gradients from ``data`` reach
# the optimizer-owned sharded parameters when a custom packed forward is used.
# Use a validation-free unflatten function because generic pytree transforms
# are allowed to replace leaves with temporary non-domain values.
register_pytree_node(
    PackedEncoderActivations,
    _packed_encoder_activations_flatten,
    _packed_encoder_activations_unflatten,
    serialized_type_name="nemo.collections.asr.parts.packed_sequence.PackedEncoderActivations",
    flatten_with_keys_fn=_packed_encoder_activations_flatten_with_keys,
)


def pack_encoder_output(padded: Tensor, lengths: Tensor) -> PackedEncoderActivations:
    """Compact valid prefixes from a channels-last encoder batch.

    Args:
        padded: Encoder states with shape ``(B, T, D)``.
        lengths: Valid prefix lengths with shape ``(B,)``.

    Returns:
        A :class:`PackedEncoderActivations` whose data has shape ``(sum(lengths), D)``.
    """

    if padded.ndim != 3:
        raise ValueError(f"padded must have shape (B, T, D), got {tuple(padded.shape)}.")
    lengths, max_seqlen = _normalize_lengths(
        lengths, batch_size=padded.shape[0], max_length=padded.shape[1], device=padded.device
    )
    mask = _length_mask(lengths, padded.shape[1])
    data = padded[mask]
    cu_seqlens = _lengths_to_cu_seqlens(lengths)
    return _new_packed_encoder_activations(data, lengths, cu_seqlens, max_seqlen, padded_length=padded.shape[1])


def unpack_encoder_output(packed: PackedEncoderActivations, *, total_length: int | None = None) -> Tensor:
    """Restore a packed encoder output to channels-last ``(B, T, D)`` form.

    ``total_length`` defaults to ``packed.max_seqlen``. A larger value is useful at
    legacy boundaries that must preserve an externally chosen padded width.
    """

    if total_length is None:
        total_length = packed.max_seqlen
    if int(total_length) != total_length or total_length < packed.max_seqlen:
        raise ValueError(f"total_length must be an integer >= max_seqlen ({packed.max_seqlen}), got {total_length}.")
    total_length = int(total_length)
    if isinstance(packed.padding_value, Tensor):
        padding = packed.padding_value.to(device=packed.data.device, dtype=packed.data.dtype)
        output = padding.unsqueeze(1).expand(-1, total_length, -1).clone()
    else:
        output = packed.data.new_full((packed.batch_size, total_length, packed.data.shape[-1]), packed.padding_value)
    if packed.total_tokens:
        output[_length_mask(packed.lengths, total_length)] = packed.data
    return output


def split_encoder_output(packed: PackedEncoderActivations) -> tuple[Tensor, ...]:
    """Return one unpadded ``(T_i, D)`` view per utterance."""

    offsets = packed.cu_seqlens.tolist()
    return tuple(packed.data[offsets[i] : offsets[i + 1]] for i in range(packed.batch_size))


def packed_encoder_position_ids(packed: PackedEncoderActivations) -> Tensor:
    """Return zero-based positions for every packed token, resetting per utterance."""

    if packed.total_tokens == 0:
        return packed.lengths.new_empty((0,))
    token_indices = torch.arange(packed.total_tokens, device=packed.data.device, dtype=torch.int64)
    sequence_ids = torch.bucketize(token_indices, packed.cu_seqlens[1:], right=True)
    return token_indices - packed.cu_seqlens[sequence_ids]


def split_packed_data(data: Tensor, lengths: Tensor, cu_seqlens: Tensor) -> tuple[Tensor, ...]:
    """Validate packed leading-dimension metadata and return one view per sequence.

    This lower-level representation is useful before encoder features exist, such as
    for concatenated waveform samples. Unlike :class:`PackedEncoderActivations`, it allows
    any data rank and integer offset dtype.
    """

    if data.ndim == 0:
        raise ValueError("packed data must have at least one dimension.")
    if lengths.ndim != 1:
        raise ValueError(f"lengths must be 1D, got shape {tuple(lengths.shape)}.")
    if lengths.dtype == torch.bool or lengths.is_floating_point() or lengths.is_complex():
        raise TypeError(f"lengths must have an integer dtype, got {lengths.dtype}.")
    if cu_seqlens.ndim != 1 or cu_seqlens.numel() != lengths.numel() + 1:
        raise ValueError(f"cu_seqlens must have shape ({lengths.numel() + 1},), got {tuple(cu_seqlens.shape)}.")
    if cu_seqlens.dtype == torch.bool or cu_seqlens.is_floating_point() or cu_seqlens.is_complex():
        raise TypeError(f"cu_seqlens must have an integer dtype, got {cu_seqlens.dtype}.")
    if data.device != lengths.device or data.device != cu_seqlens.device:
        raise ValueError("data, lengths, and cu_seqlens must be on the same device.")

    offsets = cu_seqlens.to(torch.int64)
    if not torch.equal(offsets[1:] - offsets[:-1], lengths.to(torch.int64)):
        raise ValueError("Differences in cu_seqlens must equal lengths.")
    host_offsets = offsets.detach().cpu().tolist()
    if host_offsets[0] != 0:
        raise ValueError("cu_seqlens must start at zero.")
    if any(end < begin for begin, end in zip(host_offsets, host_offsets[1:])):
        raise ValueError("cu_seqlens must be non-decreasing.")
    if host_offsets[-1] != data.shape[0]:
        raise ValueError(f"data has {data.shape[0]} entries, but cu_seqlens ends at {host_offsets[-1]}.")
    return tuple(data[begin:end] for begin, end in zip(host_offsets, host_offsets[1:]))


def _length_mask(lengths: Tensor, total_length: int) -> Tensor:
    positions = torch.arange(total_length, device=lengths.device)
    return positions.unsqueeze(0) < lengths.unsqueeze(1)


def _lengths_to_cu_seqlens(lengths: Tensor) -> Tensor:
    return torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device=lengths.device),
            lengths.cumsum(dim=0, dtype=torch.int32),
        ]
    ).contiguous()


def _normalize_lengths(lengths: Tensor, *, batch_size: int, max_length: int, device: torch.device):
    if lengths.ndim != 1 or lengths.numel() != batch_size:
        raise ValueError(f"lengths must have shape ({batch_size},), got {tuple(lengths.shape)}.")
    if lengths.device != device:
        raise ValueError(f"lengths must be on {device}, got {lengths.device}.")
    if lengths.dtype == torch.bool or lengths.is_floating_point() or lengths.is_complex():
        raise TypeError(f"lengths must have an integer dtype, got {lengths.dtype}.")
    lengths = lengths.to(torch.int64)
    # Varlen kernels need a host max length. Validate the same host copy so the
    # hot packing path pays one device synchronization rather than one per check.
    host_lengths = lengths.detach().to(device="cpu")
    if bool(((host_lengths < 0) | (host_lengths > max_length)).any()):
        raise ValueError(f"lengths must be between 0 and padded time {max_length}, got {host_lengths.tolist()}.")
    max_seqlen = int(host_lengths.max()) if host_lengths.numel() else 0
    return lengths, max_seqlen


def _validate_packed_encoder_activations(packed: PackedEncoderActivations) -> None:
    if packed.data.ndim != 2:
        raise ValueError(f"data must have shape (T_total, D), got {tuple(packed.data.shape)}.")
    if packed.lengths.ndim != 1 or packed.lengths.dtype != torch.int64:
        raise ValueError(f"lengths must be 1D int64, got shape={tuple(packed.lengths.shape)}, {packed.lengths.dtype}.")
    if packed.cu_seqlens.ndim != 1 or packed.cu_seqlens.dtype != torch.int32:
        raise ValueError(
            "cu_seqlens must be 1D int32, got " f"shape={tuple(packed.cu_seqlens.shape)}, {packed.cu_seqlens.dtype}."
        )
    if not packed.cu_seqlens.is_contiguous():
        raise ValueError("cu_seqlens must be contiguous for variable-length attention kernels.")
    if packed.data.device != packed.lengths.device or packed.data.device != packed.cu_seqlens.device:
        raise ValueError("data, lengths, and cu_seqlens must be on the same device.")
    if packed.cu_seqlens.numel() != packed.lengths.numel() + 1:
        raise ValueError("cu_seqlens must have exactly batch_size + 1 entries.")
    expected = _lengths_to_cu_seqlens(packed.lengths)
    if not torch.equal(packed.cu_seqlens, expected):
        raise ValueError("cu_seqlens must start at zero and have differences equal to lengths.")
    if packed.data.shape[0] != int(expected[-1].item()):
        raise ValueError(f"data has {packed.data.shape[0]} tokens, but cu_seqlens ends at {int(expected[-1].item())}.")
    expected_max = int(packed.lengths.max().item()) if packed.lengths.numel() else 0
    if int(packed.max_seqlen) != packed.max_seqlen or packed.max_seqlen != expected_max:
        raise ValueError(f"max_seqlen must equal max(lengths)={expected_max}, got {packed.max_seqlen}.")
    if isinstance(packed.padding_value, Tensor):
        expected_shape = (packed.batch_size, packed.data.shape[1])
        if packed.padding_value.shape != expected_shape:
            raise ValueError(
                f"tensor padding_value must have shape {expected_shape}, got {tuple(packed.padding_value.shape)}."
            )
        if packed.padding_value.device != packed.data.device:
            raise ValueError(
                f"tensor padding_value must be on {packed.data.device}, got {packed.padding_value.device}."
            )
    if packed.padded_length is not None and (
        int(packed.padded_length) != packed.padded_length or packed.padded_length < packed.max_seqlen
    ):
        raise ValueError(
            f"padded_length must be an integer >= max_seqlen ({packed.max_seqlen}), got {packed.padded_length}."
        )


def _new_packed_encoder_activations(data, lengths, cu_seqlens, max_seqlen, padding_value=0.0, padded_length=None):
    """Construct from metadata that a public constructor already validated."""
    packed = object.__new__(PackedEncoderActivations)
    object.__setattr__(packed, "data", data)
    object.__setattr__(packed, "lengths", lengths)
    object.__setattr__(packed, "cu_seqlens", cu_seqlens)
    object.__setattr__(packed, "max_seqlen", max_seqlen)
    object.__setattr__(packed, "padding_value", padding_value)
    object.__setattr__(packed, "padded_length", padded_length)
    return packed
