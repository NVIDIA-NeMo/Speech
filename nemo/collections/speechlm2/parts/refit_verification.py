# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional check that RL weight updates ("refits") arrive intact in the vLLM model.

In NeMo RL, a refit copies the trainer's updated weights into the vLLM generation
engine after each training step. Enable the check with ``speechlm_refit_verification``
in the model's HF config (or vLLM ``hf_overrides``)::

    {"speechlm_refit_verification": {
        "output_dir": "/path/to/refit",
        "samples": {"language_model.backbone.layers.0.mixer.D": 0, "perception.proj.weight": null}}}

``samples`` maps received weight names to their tensor-parallel shard axis, or ``null``
for replicated weights. On every load after the first full one, up to four values of each
sampled weight are recorded as received and compared exactly with the loaded vLLM
parameter; a mismatch fails the load and each process appends a JSONL receipt.

This checks received against loaded values only, not that the trainer sent the right
weights. Only replicated weights and the unfused Mamba ``D``/``dt_bias`` and row-parallel
``out_proj`` layouts are supported.
"""

import json
import math
import os
import socket
from pathlib import Path

import torch


class RefitWeightVerifier:
    """Retain at most four expected scalar values per explicitly chosen weight."""

    def __init__(self, model, specs: dict[str, int | None], *, tp_rank: int, tp_size: int):
        if tp_size < 1 or not 0 <= tp_rank < tp_size:
            raise ValueError('Invalid tensor-parallel rank/size')
        self.model = model
        self.specs = specs
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.samples = {}

    def capture(self, name: str, received: torch.Tensor) -> None:
        if name not in self.specs:
            return
        if name in self.samples:
            raise ValueError(f'Duplicate refit sample: {name}')
        destination_name = name
        if name.startswith("language_model."):
            receiver = getattr(self.model, "language_model", None)
            mapper = getattr(receiver, "hf_to_vllm_mapper", None)
            if mapper is not None:
                mapped = mapper.apply_list([name.removeprefix("language_model.")])
                if len(mapped) != 1:
                    raise ValueError(f'Refit sample has no unique receiver destination: {name}')
                destination_name = "language_model." + mapped[0]
        destination = self.model.get_parameter(destination_name)
        axis = self.specs[name]
        if received.dtype != torch.bfloat16:
            raise ValueError(f'Refit verification expects BF16 transport for {name}, got {received.dtype}')
        if received.ndim != destination.ndim or destination.numel() == 0:
            raise ValueError(f'Unsupported refit dimensions for {name}')
        expected_shape = list(destination.shape)
        if axis is not None:
            if axis == 0 and name.endswith(('.mixer.D', '.mixer.dt_bias')):
                if destination.ndim != 1:
                    raise ValueError(f'Expected one-dimensional Mamba weight: {name}')
            elif axis == 1 and name.endswith('.mixer.out_proj.weight'):
                if destination.ndim != 2 or getattr(destination, 'input_dim', None) != 1:
                    raise ValueError(f'Expected unfused RowParallelLinear input_dim=1: {name}')
            else:
                raise ValueError(f'Unsupported refit shard mapping: {name}, axis={axis}')
            expected_shape[axis] *= self.tp_size
        if list(received.shape) != expected_shape:
            raise ValueError(f'Refit shape mismatch for {name}: {tuple(received.shape)} != {tuple(expected_shape)}')
        coordinates = _sample_coordinates(tuple(destination.shape))
        source_coordinates = []
        expected = []
        for coordinate in coordinates:
            source_coordinate = list(coordinate)
            if axis is not None:
                source_coordinate[axis] += self.tp_rank * destination.shape[axis]
            value = received[tuple(source_coordinate)].to(destination.dtype).item()
            if not math.isfinite(value):
                raise ValueError(f'Nonfinite received refit sample: {name}')
            source_coordinates.append(tuple(source_coordinate))
            expected.append(value)
        self.samples[name] = {
            'destination_name': destination_name,
            'received_shape': list(received.shape),
            'destination_shape': list(destination.shape),
            'destination_dtype': str(destination.dtype),
            'transport_dtype': str(received.dtype),
            'shard_axis': axis,
            'coordinates': coordinates,
            'source_coordinates': source_coordinates,
            'expected': expected,
        }

    def verify(self) -> dict:
        result = {}
        for name, sample in self.samples.items():
            destination = self.model.get_parameter(sample['destination_name'])
            if (
                list(destination.shape) != sample['destination_shape']
                or str(destination.dtype) != sample['destination_dtype']
            ):
                raise ValueError(f'Destination shape/dtype changed during refit: {name}')
            actual = [destination[tuple(c)].item() for c in sample['coordinates']]
            if any(not math.isfinite(v) for v in actual) or actual != sample['expected']:
                raise ValueError(f'Destination does not match received refit samples: {name}')
            result[name] = {**sample, 'actual': actual, 'verified': True}
        return {'tp_rank': self.tp_rank, 'tp_size': self.tp_size, 'samples': result}


def append_refit_receipt(output_dir: str, receipt: dict) -> None:
    """One process/rank owns each JSONL; separate buckets append separate rows."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    filename = f'refit-{socket.gethostname()}-{os.getpid()}-tp{receipt["tp_rank"]}.jsonl'
    with (output / filename).open('a') as stream:
        stream.write(json.dumps(receipt, sort_keys=True) + '\n')


def _sample_coordinates(shape: tuple[int, ...]) -> list[tuple[int, ...]]:
    size = math.prod(shape)
    result = []
    for index in sorted({0, size // 3, 2 * size // 3, size - 1}):
        coordinate = []
        for length in reversed(shape):
            coordinate.append(index % length)
            index //= length
        result.append(tuple(reversed(coordinate)))
    return result
