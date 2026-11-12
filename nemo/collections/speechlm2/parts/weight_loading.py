# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
"""SpeechLM checkpoint and perception weight loading with strict name and shape checks."""

from itertools import chain
from typing import Mapping

import torch


def canonical_parameter_name(name: str) -> str:
    """Remove PyTorch activation-checkpoint wrapper segments from a state name."""
    return ".".join(part for part in name.split(".") if part != "_checkpoint_wrapped_module")


def checkpoint_load_state(
    model: torch.nn.Module, checkpoint_shapes: Mapping[str, tuple[int, ...]]
) -> dict[str, torch.Tensor]:
    """Build a direct-load state, rejecting missing parameters and unexpected tensors.

    Nonpersistent synthesized buffers may be absent from the checkpoint. Every
    parameter, including frozen and MTP parameters, and every persistent buffer
    must have a matching tensor.
    Shape mismatches and collisions fail before DCP writes any model storage.
    """
    parameters = dict(model.named_parameters(remove_duplicate=False))
    all_tensors = dict(chain(parameters.items(), model.named_buffers(remove_duplicate=False)))
    canonical = {}
    for name, tensor in all_tensors.items():
        key = canonical_parameter_name(name)
        if key in canonical:
            raise ValueError(f"Duplicate canonical checkpoint name: {key}")
        canonical[key] = tensor
    # Module extra state is a separate PyTorch protocol, not a parameter or
    # buffer. Non-FP8 Transformer Engine modules return None and DCP stores a
    # zero-length uint8 tensor. Recognize only modules declaring that protocol.
    for module_name, module in model.named_modules(remove_duplicate=False):
        if type(module).get_extra_state is torch.nn.Module.get_extra_state:
            continue
        key = canonical_parameter_name(f"{module_name}._extra_state" if module_name else "_extra_state")
        extra = module.get_extra_state()
        if extra is None:
            extra = torch.empty(0, dtype=torch.uint8)
        if not isinstance(extra, torch.Tensor):
            raise ValueError(f"Unsupported non-tensor module extra state: {key}")
        canonical[key] = extra
    required = set(parameters)
    for module_name, module in model.named_modules(remove_duplicate=False):
        for buffer_name, buffer in module._buffers.items():
            if buffer is not None and buffer_name not in module._non_persistent_buffers_set:
                required.add(f"{module_name}.{buffer_name}" if module_name else buffer_name)
    required.update(key for key, tensor in canonical.items() if key.endswith("_extra_state") and tensor.numel())
    missing = sorted(
        canonical_parameter_name(name) for name in required if canonical_parameter_name(name) not in checkpoint_shapes
    )
    unexpected = sorted(set(checkpoint_shapes) - set(canonical))
    if missing or unexpected:
        raise ValueError(
            f"SpeechLM checkpoint coverage mismatch: missing parameters/buffers={missing}; unexpected tensors={unexpected}"
        )
    state = {}
    for key, shape in checkpoint_shapes.items():
        tensor = canonical[key]
        if tuple(tensor.shape) != tuple(shape):
            raise ValueError(
                f"Checkpoint shape mismatch for {key}: model={tuple(tensor.shape)}, checkpoint={tuple(shape)}"
            )
        state[key] = tensor
    return state


def apply_checkpoint_extra_state(model: torch.nn.Module, state: Mapping[str, torch.Tensor]) -> None:
    """Restore declared tensor extra state after DCP, keeping genuine empty None state."""
    for module_name, module in model.named_modules(remove_duplicate=False):
        if type(module).get_extra_state is torch.nn.Module.get_extra_state:
            continue
        key = canonical_parameter_name(f"{module_name}._extra_state" if module_name else "_extra_state")
        if key not in state:
            continue
        if module.get_extra_state() is None:
            if state[key].numel():
                raise ValueError(f"Nonempty checkpoint extra state cannot load into None state: {key}")
        else:
            module.set_extra_state(state[key])


def load_nemo_rl_checkpoint(model: torch.nn.Module, checkpoint_path: str) -> None:
    """Load native flat DCP tensors from NeMo-RL's weights/model directory.

    Use the same strict canonical coverage as the HF seed loader; this avoids
    inventing a second conversion/export path or a Lightning state namespace.
    """
    import torch.distributed.checkpoint as dcp

    reader = dcp.FileSystemReader(checkpoint_path)
    metadata = reader.read_metadata().state_dict_metadata
    state = checkpoint_load_state(model, {key: tuple(value.size) for key, value in metadata.items()})
    dcp.load(state, storage_reader=reader)
    apply_checkpoint_extra_state(model, state)
    ctc_timestamp = model._modules.get("ctc_timestamp")
    if ctc_timestamp is not None:
        ctc_timestamp.validate_tokenizer()


def load_perception_state(
    perception: torch.nn.Module,
    weights: Mapping[str, torch.Tensor],
    *,
    require_complete: bool,
) -> set[str]:
    """Load perception weights, checking every name and shape before anything is copied.

    Args:
        perception: The vLLM model's perception module.
        weights: Perception tensors keyed by names relative to ``perception``.
        require_complete: Also require every perception tensor. The first load must be
            complete; later RL weight updates may send only some tensors.

    Returns:
        The loaded names, prefixed with ``perception.``.
    """
    expected = perception.state_dict()
    missing = (
        sorted(key for key in expected if key not in weights and not key.endswith("._extra_state"))
        if require_complete
        else []
    )
    unexpected = sorted(key for key in weights if key not in expected)
    wrong_shapes = [
        key
        for key, tensor in weights.items()
        if key in expected and isinstance(expected[key], torch.Tensor) and expected[key].shape != tensor.shape
    ]
    if missing or unexpected or wrong_shapes:
        raise RuntimeError(
            "Speech encoder checkpoint does not exactly match its exported architecture: "
            f"missing={missing}, unexpected={unexpected}, shape_mismatches={wrong_shapes}."
        )
    perception.load_state_dict(weights, strict=False)
    return {"perception." + key for key in weights}
