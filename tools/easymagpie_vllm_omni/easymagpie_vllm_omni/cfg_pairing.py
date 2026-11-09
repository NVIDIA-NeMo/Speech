# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Identity, alignment, and output routing for paired CFG engine sequences.

This module deliberately has no engine or tensor dependencies. Cache allocation
belongs to the scheduler; row mappings are rebuilt from request identities each step.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from uuid import uuid4

if TYPE_CHECKING:
    from collections.abc import Iterable

    from vllm.v1.engine import EngineCoreOutputs
    from vllm.v1.request import Request, RequestStatus
    from vllm_omni.distributed.omni_connectors.transfer_adapter.chunk_transfer_adapter import ChunkTransferAdapter

CFG_PAIR_METADATA_KEY = "_cfg_pair"
_HIDDEN_PREFIX = "__easymagpie_cfg_unconditional__:"


class CFGPairingError(RuntimeError):
    """A physical pair cannot safely execute the next model step."""


@dataclass(frozen=True)
class CFGPair:
    """One public conditional sequence and one private unconditional sequence.

    Attributes:
        conditional_id: Public conditional request identity.
        unconditional_id: Private unconditional request identity.
    """

    conditional_id: str
    unconditional_id: str

    @classmethod
    def create(cls, logical_id: str) -> CFGPair:
        """Create distinct public and private identities for one logical request.

        Args:
            logical_id: Public identity of the logical synthesis request.

        Returns:
            The new pair with a unique private identity.

        Raises:
            ValueError: If the public identity uses the reserved private CFG namespace.
        """
        if is_cfg_hidden_id(logical_id):
            raise ValueError("request ID uses the reserved CFG branch namespace")
        return cls(logical_id, f"{_HIDDEN_PREFIX}{uuid4().hex}")

    @property
    def physical_ids(self) -> tuple[str, str]:
        """Return the public and private identities in branch order.

        Returns:
            Conditional and unconditional physical request identities.
        """
        return self.conditional_id, self.unconditional_id

    def metadata(self, physical_id: str) -> dict[str, dict[str, str]]:
        """Build the scheduler-owned branch metadata for a physical request.

        Args:
            physical_id: Conditional or unconditional identity belonging to this pair.

        Returns:
            Scheduler-owned CFG branch metadata.

        Raises:
            ValueError: If the physical identity does not belong to this pair.
        """
        if physical_id not in self.physical_ids:
            raise ValueError("physical request does not belong to this CFG pair")
        return {
            CFG_PAIR_METADATA_KEY: {
                "logical_id": self.conditional_id,
                "branch": "conditional" if physical_id == self.conditional_id else "unconditional",
            }
        }

    def validate_step(
        self,
        scheduled: Mapping[str, int],
        computed: Mapping[str, int],
    ) -> None:
        """Require paired branches to advance together at matching cache positions.

        Args:
            scheduled: Scheduled token counts by physical request ID.
            computed: Computed positions by physical request ID.

        Raises:
            CFGPairingError: If only one branch is scheduled or branch positions and lengths disagree.
        """
        conditional, unconditional = self.physical_ids
        counts = scheduled.get(conditional, 0), scheduled.get(unconditional, 0)
        if counts == (0, 0):
            return
        if min(counts) <= 0 or counts[0] != counts[1] or computed[conditional] != computed[unconditional]:
            raise CFGPairingError("CFG branches must be scheduled together at aligned positions and query lengths")


@dataclass(frozen=True)
class CFGPairingLayout:
    """Unpadded token rows for each logical pair, independent of batch order.

    Attributes:
        logical_ids: Logical identities in the order used by both row mappings.
        conditional_rows: Physical token rows belonging to each public branch.
        unconditional_rows: Physical token rows belonging to each private branch.
    """

    logical_ids: tuple[str, ...]
    conditional_rows: tuple[tuple[int, ...], ...]
    unconditional_rows: tuple[tuple[int, ...], ...]


def build_cfg_pairing_layout(
    request_ids: Sequence[str],
    scheduled: Sequence[int],
    computed: Sequence[int],
    metadata: Mapping[str, dict[str, Any]],
) -> CFGPairingLayout:
    """Build explicit token-row pairs from a runner's current physical batch.

    Args:
        request_ids: Physical request identities in current runner batch order.
        scheduled: Query lengths aligned with request_ids.
        computed: Computed-token positions aligned with request_ids.
        metadata: Additional information indexed by physical request identity.

    Returns:
        Aligned public and private token rows grouped by logical identity.

    Raises:
        CFGPairingError: If identities, branch metadata, positions, or query lengths are inconsistent.
    """
    if not len(request_ids) == len(scheduled) == len(computed) or len(set(request_ids)) != len(request_ids):
        raise CFGPairingError("invalid CFG batch lengths or duplicate request identities")
    pairs: dict[str, dict] = {}
    offset = 0
    for rid, count, position in zip(request_ids, scheduled, computed):
        count, position = int(count), int(position)
        if count <= 0 or position < 0:
            raise CFGPairingError("invalid CFG batch query length or position")
        tag = metadata.get(rid, {}).get(CFG_PAIR_METADATA_KEY)
        if tag is not None:
            logical_id, branch = tag.get("logical_id"), tag.get("branch")
            if not isinstance(logical_id, str) or branch not in {"conditional", "unconditional"}:
                raise CFGPairingError("invalid CFG branch metadata")
            branches = pairs.setdefault(logical_id, {})
            if branch in branches:
                raise CFGPairingError("duplicate CFG branch identity")
            branches[branch] = (tuple(range(offset, offset + count)), position)
        offset += count

    logical_ids = tuple(sorted(pairs))
    conditional_rows, unconditional_rows = [], []
    for logical_id in logical_ids:
        branches = pairs[logical_id]
        if set(branches) != {"conditional", "unconditional"}:
            raise CFGPairingError("CFG batch is missing a branch partner")
        conditional, conditional_pos = branches["conditional"]
        unconditional, unconditional_pos = branches["unconditional"]
        if len(conditional) != len(unconditional) or conditional_pos != unconditional_pos:
            raise CFGPairingError("CFG batch partners are not aligned")
        conditional_rows.append(conditional)
        unconditional_rows.append(unconditional)
    return CFGPairingLayout(logical_ids, tuple(conditional_rows), tuple(unconditional_rows))


def is_cfg_hidden_id(request_id: str) -> bool:
    """Check whether an identity belongs to the private CFG namespace.

    Args:
        request_id: Public request identity.

    Returns:
        Whether the identity uses the reserved private prefix.
    """
    return request_id.startswith(_HIDDEN_PREFIX)


def filter_cfg_outputs(outputs: dict[int, EngineCoreOutputs]) -> None:
    """Remove private outputs and completion notifications after engine accounting.

    Args:
        outputs: Outputs returned by the scheduler or supplied by the test engine.
    """
    for output in outputs.values():
        output.outputs = [item for item in output.outputs if not is_cfg_hidden_id(item.request_id)]
        finished = getattr(output, "finished_requests", None)
        if finished is not None:
            output.finished_requests = {rid for rid in finished if not is_cfg_hidden_id(rid)}


class CFGPairingTransferFilter:
    """Suppress private payloads before Omni's scheduler calls the codec connector."""

    def __init__(self, adapter: ChunkTransferAdapter) -> None:
        """Initialize CFGPairingTransferFilter state.

        Args:
            adapter: Wrapped codec transfer adapter or adapter test fixture.
        """
        self._adapter = adapter

    def __getattr__(self, name: str) -> Any:
        """Forward connector attributes to the wrapped transfer adapter.

        Args:
            name: Request or attribute name.

        Returns:
            The corresponding attribute of the wrapped codec transfer adapter.
        """
        return getattr(self._adapter, name)

    def save_async(
        self,
        multimodal_output: dict[str, Any] | None = None,
        request: Request | None = None,
        is_segment_finished: bool = False,
    ) -> None:
        """Save async.

        Args:
            multimodal_output: Model payload to transfer to the next stage.
            request: Incoming engine request.
            is_segment_finished: Whether the current resumable segment has ended.
        """
        if not is_cfg_hidden_id(request.request_id):
            return self._adapter.save_async(multimodal_output, request, is_segment_finished)
        return None

    def finish_requests(
        self,
        request_ids: str | Iterable[str] | None,
        finished_status: RequestStatus,
        requests: dict[str, Request],
    ) -> list[tuple[str, int]]:
        """Finish selected requests and release their paired histories.

        Args:
            request_ids: Request identities to finish, or None for all requests.
            finished_status: Terminal status applied to the selected requests.
            requests: Live requests indexed by physical identity.

        Returns:
            Public request identities and their client indices for completed aborts.
        """
        ids = requests.keys() if request_ids is None else request_ids
        if isinstance(ids, str):
            ids = [ids]
        public = [rid for rid in ids if not is_cfg_hidden_id(rid)]
        return self._adapter.finish_requests(public, finished_status, requests)
