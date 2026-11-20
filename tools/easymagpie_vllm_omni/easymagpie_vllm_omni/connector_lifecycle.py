# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bound native connector bookkeeping without unlinking unread publications."""
from __future__ import annotations

import os
from functools import wraps
from typing import Any


def install_sender_metadata_cleanup(adapter: Any) -> None:
    """Prune consumed SHM identities and unused stage-zero receive tombstones.

    Args:
        adapter: The initialized native stage-zero chunk adapter.
    """
    connector = getattr(adapter, "connector", None)
    if getattr(connector, "stage_id", None) != 0:
        return
    # Keep imports local: CFG pairing/layout utilities can be imported without
    # the optional serving runtime. Only this native compatibility hook needs it.
    from vllm_omni.distributed.omni_connectors.connectors.shm_connector import SharedMemoryConnector

    if isinstance(connector, SharedMemoryConnector) and not isinstance(connector._pending_keys, _PendingSHMKeys):
        connector._pending_keys = _PendingSHMKeys(connector._pending_keys)
    cleanup = getattr(adapter, "cleanup_receiver", None)
    cancelled = getattr(adapter, "_cancelled_load_reqs", None)
    if cleanup is None or cancelled is None or getattr(adapter, "_easymagpie_sender_metadata_cleanup", False):
        return

    @wraps(cleanup)
    def cleanup_receiver(request_id):
        try:
            return cleanup(request_id)
        finally:
            # Stage zero has no upstream receiver or pending receive tasks.
            cancelled.discard(request_id)

    adapter.cleanup_receiver = cleanup_receiver
    adapter._easymagpie_sender_metadata_cleanup = True


class _PendingSHMKeys(set[str]):
    """Retain outstanding native keys, rather than every publication ever made.

    Native put/add runs on the sole save thread after the new SHM write. Before
    each add, forget only older keys whose SHM object AND lockfile are absent.
    No resource is unlinked here. Between publications the set cannot grow; at
    each publication it contains the observed outstanding resources plus the new
    key. It is not a hard bound on unread data or an acknowledgement protocol.
    """

    def add(self, key: str) -> None:
        self.difference_update(previous for previous in tuple(self) if _publication_absent(previous))
        super().add(key)


def _publication_absent(key: str) -> bool:
    """Observe exact native POSIX names; keep entries on any uncertain result."""
    if not isinstance(key, str) or not key or "/" in key or key in {".", ".."}:
        return False
    for path in (f"/dev/shm/{key}", f"/dev/shm/shm_{key}_lockfile.lock"):
        try:
            os.stat(path)
        except FileNotFoundError:
            continue
        except OSError:
            return False
        return False
    return True
