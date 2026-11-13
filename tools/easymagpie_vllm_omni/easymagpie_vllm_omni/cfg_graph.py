# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Address-stable, fixed-shape CFG inputs for decode graph replay."""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from easymagpie_vllm_omni.cfg_pairing import CFGPairingLayout
from vllm.config import CompilationMode, CUDAGraphMode

if TYPE_CHECKING:
    from vllm.config import CompilationConfig, VllmConfig


def supports_cfg_graphs(compilation: CompilationConfig) -> bool:
    """Allow decode-only capture or compiled backbone prefill with full decode capture.

    The outer model and V2 predictor stay eager during piecewise prefill; only
    the decorated Nemotron-H backbone is compiled. Both policies need the same
    address-stable pairing buffers for full decode replay.

    Args:
        compilation: Compilation and CUDA graph policy.

    Returns:
        Whether the requested condition holds.
    """
    return (getattr(compilation, "mode", None), getattr(compilation, "cudagraph_mode", None)) in (
        (CompilationMode.NONE, CUDAGraphMode.FULL_DECODE_ONLY),
        (CompilationMode.VLLM_COMPILE, CUDAGraphMode.FULL_AND_PIECEWISE),
    )


def uses_cfg_decode_graphs(config: VllmConfig) -> bool:
    """Select the fixed-shape path only for the explicit paired synthesis scheduler.

    Args:
        config: Engine configuration to validate or inspect.

    Returns:
        Whether the requested condition holds.
    """
    scheduler = config.scheduler_config.scheduler_cls
    if isinstance(scheduler, type):
        scheduler = f"{scheduler.__module__}.{scheduler.__qualname__}"
    return (
        (
            (
                scheduler == "easymagpie_vllm_omni.cfg_scheduler.EasyMagpieCFGScheduler"
                and not getattr(config.scheduler_config, "async_scheduling", False)
            )
            or (
                scheduler == "easymagpie_vllm_omni.cfg_async_scheduler.EasyMagpieCFGAsyncScheduler"
                and config.scheduler_config.async_scheduling
            )
        )
        and not config.model_config.enforce_eager
        and supports_cfg_graphs(config.compilation_config)
    )


class CFGGraphBuffers:
    """Physical-row ownership staged by the host before each model invocation.

    Capture always executes the same operations, including when no pair is live.
    Private rows gather their public owner's sampled frame; unpaired rows own
    themselves. Inactive padding is masked independently from branch ownership.
    These are scratch inputs, never cache allocations or checkpoint weights.
    """

    def __init__(
        self,
        capacity: int,
        *,
        device: str | torch.device = None,
    ) -> None:
        """Initialize CFGGraphBuffers state.

        Args:
            capacity: Maximum number of physical token rows or sequence slots.
            device: Device on which tensors are allocated.
        """
        self.owners = torch.arange(capacity, device=device)
        self.partners = torch.arange(capacity, device=device)
        self.active = torch.zeros(capacity, dtype=torch.bool, device=device)
        # Keep host storage alive until both uploads from its slot have completed.
        # Two slots allow preparation to overlap the preceding model invocation.
        cuda = self.owners.is_cuda
        self._host_slots = [
            torch.empty((2, capacity), dtype=torch.long, device="cpu", pin_memory=cuda) for _ in range(2)
        ]
        self._host_views = [slot.numpy() for slot in self._host_slots]
        self._identity = torch.arange(capacity, device="cpu").numpy()
        self._copy_events = [torch.cuda.Event() for _ in self._host_slots] if cuda else None
        self._next_slot = 0

    def update(
        self,
        layout: CFGPairingLayout | None,
        num_tokens: int,
        padded_tokens: int,
    ) -> None:
        """Stage the current unpadded mapping outside graph capture/replay.

        Rebuild the entire padded prefix, including identity entries, so slot
        reuse and transitions to conditional-only traffic cannot retain a pair.
        Layout alignment and request identities are validated by the runner.
        Uploads and the consuming forward must use the same ordered CUDA stream.
        A pending staging slot waits only for its own copies before host reuse;
        an available slot introduces no host synchronization.

        Args:
            layout: Current unpadded mapping of public and private token rows.
            num_tokens: Number of active unpadded physical token rows.
            padded_tokens: Physical token count including graph padding.

        Raises:
            ValueError: If padded capacity or active branch row alignment is invalid.
        """
        if not 0 <= num_tokens <= padded_tokens <= self.owners.numel():
            raise ValueError("CFG graph token count exceeds buffer capacity")
        if layout is not None:
            for conditional, unconditional in zip(layout.conditional_rows, layout.unconditional_rows):
                if len(conditional) != len(unconditional):
                    raise ValueError("CFG graph branch lengths must agree")
                for public, private in zip(conditional, unconditional):
                    if not (0 <= public < num_tokens and 0 <= private < num_tokens):
                        raise ValueError("CFG graph row is outside the active batch")
        if padded_tokens == 0:
            return
        # Pinned uploads return before CUDA has finished reading the host maps.
        # Reusing that memory too early can corrupt an earlier forward's pairing.
        # Alternate slots to overlap work; wait only when reusing a still-busy slot.
        slot = self._next_slot
        event = self._copy_events[slot] if self._copy_events is not None else None
        if event is not None and not event.query():
            event.synchronize()
        host = self._host_slots[slot]
        owners, partners = self._host_views[slot]
        owners[:padded_tokens] = self._identity[:padded_tokens]
        partners[:padded_tokens] = self._identity[:padded_tokens]
        if layout is not None:
            for conditional, unconditional in zip(layout.conditional_rows, layout.unconditional_rows):
                for public, private in zip(conditional, unconditional):
                    owners[private] = public
                    partners[public] = private
        self.owners[:padded_tokens].copy_(host[0, :padded_tokens], non_blocking=True)
        self.partners[:padded_tokens].copy_(host[1, :padded_tokens], non_blocking=True)
        if event is not None:
            event.record(torch.cuda.current_stream(self.owners.device))
        self._next_slot = (slot + 1) % len(self._host_slots)
        self.active[:padded_tokens].zero_()
        self.active[:num_tokens].fill_(True)
