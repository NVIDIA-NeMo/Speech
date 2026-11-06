# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared contract for collection-owned NeMo Speech throughput policies.

A throughput policy describes the useful work present in one actual training
batch. Policies belong to the collection that owns the batch schema; this module
only provides the contract, registration decorator, and schema-independent
helpers.

Policy implementations must:

* return additive measurements in explicit units, such as audio seconds or text
  tokens;
* return the actual rank-local example count from ``num_examples`` when the batch
  schema provides one without device synchronization;
* use real per-example lengths rather than configured batch sizes or sequence
  limits;
* omit a measurement when its unit cannot be determined safely;
* keep CUDA tensors deferred--do not reduce them, call item(), or copy them to
  the host in SpeechThroughputPolicy.measure; and
* avoid collectives and mutable model state other than an existing exact work
  counter maintained by the model itself.

Register a policy on the narrowest model base class that shares its batch
semantics. Registration is inherited normally by subclasses::

    from nemo.lightning.speech_throughput import (
        SpeechThroughputPolicy,
        add_sum,
        batch_value,
        register_throughput_policy,
        value_count,
    )

    class MyPolicy(SpeechThroughputPolicy):
        name = "my_collection"

        def measure(self, model, batch):
            del model
            measurements = {}
            add_sum(measurements, "target_units", batch_value(batch, "target_lengths"))
            return measurements

        def num_examples(self, model, batch):
            del model
            return value_count(batch_value(batch, "target_lengths"))

    @register_throughput_policy(MyPolicy)
    class MyModelBase:
        pass

    class MyModel(MyModelBase):
        pass  # Automatically inherits MyPolicy.

A specialized subclass can use the decorator again to replace the inherited
policy. Downstream models can equivalently expose a
one_logger_throughput_policy policy class or instance.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

__all__ = [
    "SpeechThroughputPolicy",
    "ThroughputMeasurements",
    "ThroughputValue",
    "add_audio_seconds",
    "add_sum",
    "add_value",
    "audio_lengths",
    "batch_value",
    "first_positive",
    "register_throughput_policy",
    "select_throughput_policy",
    "value_count",
]


@dataclass(frozen=True)
class ThroughputValue:
    """An additive value whose reduction and unit conversion are deferred.

    Args:
        value: A number, a numeric sequence, or a tensor containing additive
            per-example measurements. CUDA tensors must stay on-device.
        scale: Constant applied after summation, for example
            1 / sample_rate to convert waveform samples to seconds.
    """

    value: Any
    scale: float = 1.0


ThroughputMeasurements = dict[str, ThroughputValue]


class SpeechThroughputPolicy:
    """Base contract for extracting work units from one dynamic speech batch.

    measure runs on the OneLogger exporter rank after a training batch.
    Implementations should only select and detach small existing length tensors;
    reductions and materialization are deferred to the callback reporting
    window. Returning an empty mapping skips throughput for that batch.
    """

    name = "speech"

    def measure(self, model: Any, batch: Any) -> ThroughputMeasurements:
        """Return additive, explicitly named work units for batch."""

        del model, batch
        return {}

    def num_examples(self, model: Any, batch: Any) -> int | None:
        """Return the number of rank-local examples in this completed training batch.

        Use host-visible shape metadata only. For a batch containing disjoint
        cohorts, return their combined count. Return None when the schema does
        not provide a safe count.
        """

        del model, batch
        return None


def register_throughput_policy(policy_type: type[SpeechThroughputPolicy]):
    """Attach policy_type to a model class and all of its subclasses.

    Apply this decorator to a collection model, base class, or semantic mixin.
    Decorating a subclass overrides a policy inherited from its parents.
    """

    if not isinstance(policy_type, type) or not issubclass(policy_type, SpeechThroughputPolicy):
        raise TypeError("Throughput policies must subclass SpeechThroughputPolicy")

    def decorator(model_type: type) -> type:
        model_type.one_logger_throughput_policy = policy_type
        return model_type

    return decorator


def select_throughput_policy(model: Any) -> SpeechThroughputPolicy | None:
    """Instantiate the policy registered on model or inherited by it."""

    registered = getattr(model, "one_logger_throughput_policy", None)
    if registered is None:
        return None
    policy = registered() if isinstance(registered, type) else registered
    if not isinstance(policy, SpeechThroughputPolicy):
        raise TypeError("one_logger_throughput_policy must be a SpeechThroughputPolicy class or instance")
    return policy


def batch_value(batch: Any, *names: str) -> Any:
    """Return the first named value exposed by a mapping or batch object."""

    for name in names:
        if isinstance(batch, Mapping) and name in batch:
            return batch[name]
        if hasattr(batch, name):
            return getattr(batch, name)
    return None


def value_count(value: Any) -> int | None:
    """Return the number of entries using host-visible shape metadata only."""

    if value is None or isinstance(value, (str, bytes, Mapping)):
        return None
    if torch.is_tensor(value):
        return value.numel() if value.ndim > 0 else None
    try:
        return len(value)
    except TypeError:
        return None


def audio_lengths(batch: Any) -> Any:
    """Return common waveform-length fields without guessing from tensor shape."""

    lengths = batch_value(batch, "audio_lens", "audio_lengths", "audio_len", "input_length")
    if lengths is not None:
        return lengths
    if isinstance(batch, (tuple, list)) and len(batch) > 1:
        return batch[1]
    return None


def first_positive(obj: Any, *paths: str) -> float | None:
    """Return the first positive numeric value found at dotted attribute paths."""

    for path in paths:
        value = obj
        for part in path.split("."):
            if isinstance(value, Mapping):
                value = value.get(part)
            else:
                value = getattr(value, part, None)
            if value is None:
                break
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return None


def add_value(
    measurements: ThroughputMeasurements,
    name: str,
    value: Any,
    scale: float = 1.0,
) -> None:
    """Add a deferred measurement when value is available."""

    if value is not None:
        value = value.detach() if torch.is_tensor(value) else value
        measurements[name] = ThroughputValue(value, scale)


def add_sum(measurements: ThroughputMeasurements, name: str, value: Any) -> None:
    """Add values that the callback will sum across examples and batches."""

    add_value(measurements, name, value)


def add_audio_seconds(
    measurements: ThroughputMeasurements,
    name: str,
    lengths: Any,
    sample_rate: float | None,
) -> None:
    """Add waveform lengths converted to seconds when the rate is known."""

    if lengths is not None and sample_rate is not None:
        add_value(measurements, name, lengths, scale=1.0 / sample_rate)
