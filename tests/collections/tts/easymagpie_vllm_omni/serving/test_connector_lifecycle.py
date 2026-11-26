# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native CPU shared-memory lifetime tests; no serving model is constructed."""
from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, NoReturn
from uuid import uuid4

import pytest
from easymagpie_vllm_omni.cfg_pairing import CFGPairingTransferFilter
from easymagpie_vllm_omni.connector_lifecycle import _publication_absent
from vllm.v1.request import RequestStatus
from vllm_omni.distributed.omni_connectors.connectors.shm_connector import SharedMemoryConnector

if TYPE_CHECKING:
    from vllm.v1.request import Request


def test_consumed_shm_keys_do_not_accumulate_across_completed_streams() -> None:
    """Verify consumed SHM keys are pruned so the sender's pending keys do not accumulate."""
    sender = SharedMemoryConnector({"stage_id": 0})
    receiver = SharedMemoryConnector({"stage_id": 1})
    CFGPairingTransferFilter(SimpleNamespace(connector=sender))
    prefix = "easymagpie_cpu_" + uuid4().hex
    try:
        for index in range(64):
            key = f"{prefix}_{index}_0_0"
            assert sender.put("0", "1", key, {"value": index})[0]
            assert len(sender._pending_keys) <= 1
            assert receiver.get("0", "1", key)[0] == {"value": index}
    finally:
        sender.close()
        receiver.close()


def test_consumed_key_pruning_preserves_unconsumed_publications_and_other_owners() -> None:
    """Verify pruning keeps unconsumed publications and leaves another sender's publication readable."""
    sender = SharedMemoryConnector({"stage_id": 0})
    receiver = SharedMemoryConnector({"stage_id": 1})
    other = SharedMemoryConnector({"stage_id": 0})
    CFGPairingTransferFilter(SimpleNamespace(connector=sender))
    prefix = "easymagpie_cpu_" + uuid4().hex
    held = f"{prefix}_held_0_0"
    unrelated = f"{prefix}_other_0_0"
    try:
        assert sender.put("0", "1", held, {"held": True})[0]
        assert other.put("0", "1", unrelated, {"other": True})[0]
        for index in range(16):
            key = f"{prefix}_{index}_0_0"
            assert sender.put("0", "1", key, {"value": index})[0]
            assert sender._pending_keys == {held, key}
            assert receiver.get("0", "1", key)[0] == {"value": index}
        assert receiver.get("0", "1", held)[0] == {"held": True}
        assert receiver.get("0", "1", unrelated)[0] == {"other": True}
    finally:
        sender.close()
        receiver.close()
        other.close()


def test_stage_zero_cleanup_does_not_retain_receiver_cancellation_identities() -> None:
    """Verify stage-zero receiver cleanup and abort do not retain cancellation identities."""
    cancelled = set()

    def finish(
        ids: list[str],
        status: RequestStatus,
        requests: dict[str, Request],
    ) -> list[tuple[str, int]]:
        """Record finished identities as cancelled, as a native receiver would.

        Args:
            ids: Public request identities to finish.
            status: Terminal status applied to the requests.
            requests: Live requests indexed by physical identity.

        Returns:
            An empty list of completed aborts.
        """
        cancelled.update(ids)
        return []

    adapter = SimpleNamespace(
        connector=SimpleNamespace(stage_id=0),
        _cancelled_load_reqs=cancelled,
        cleanup_receiver=cancelled.add,
        finish_requests=finish,
    )
    transfer = CFGPairingTransferFilter(adapter)
    for index in range(64):
        rid = f"stage-zero-{index}"
        adapter.cleanup_receiver(rid)
        transfer.finish_requests([rid], RequestStatus.FINISHED_ABORTED, {})
        assert not cancelled


@pytest.mark.parametrize("present", ["segment", "lock", "neither", "unreadable"])
def test_bookkeeping_only_forgets_confirmed_absent_publications(
    monkeypatch: pytest.MonkeyPatch,
    present: str,
) -> None:
    """Verify a publication is reported absent only when both its segment and lockfile are missing.

    Args:
        monkeypatch: Fixture used to replace `os.stat` in the connector lifecycle module.
        present: Which resource exists ("segment", "lock", "neither"), or "unreadable" when stat fails.
    """

    def stat(path: str) -> object:
        """Simulate `os.stat` for the selected publication resources.

        Args:
            path: Filesystem path being probed.

        Returns:
            A placeholder stat result for an existing resource.

        Raises:
            PermissionError: When the publication is unreadable.
            FileNotFoundError: When the resource does not exist.
        """
        if present == "unreadable":
            raise PermissionError(path)
        if (present == "segment" and path == "/dev/shm/owner") or (
            present == "lock" and path == "/dev/shm/shm_owner_lockfile.lock"
        ):
            return object()
        raise FileNotFoundError(path)

    with monkeypatch.context() as context:
        context.setattr("easymagpie_vllm_omni.connector_lifecycle.os.stat", stat)
        assert _publication_absent("owner") is (present == "neither")


@pytest.mark.parametrize("key", ["", ".", "..", "../other", "/absolute", "nested/key"])
def test_bookkeeping_does_not_probe_invalid_shm_names(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
) -> None:
    """Verify invalid SHM names are never reported absent and never probed on the filesystem.

    Args:
        monkeypatch: Fixture used to replace `os.stat` in the connector lifecycle module.
        key: Invalid SHM publication name.
    """

    def stat(path: str) -> NoReturn:
        """Fail on any filesystem access.

        Args:
            path: Filesystem path being probed.

        Raises:
            AssertionError: Always.
        """
        raise AssertionError(f"Unexpected filesystem access: {path}")

    with monkeypatch.context() as context:
        context.setattr("easymagpie_vllm_omni.connector_lifecycle.os.stat", stat)
        assert _publication_absent(key) is False
