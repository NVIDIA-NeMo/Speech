# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ownership-map upload ordering and host staging lifetime."""
from __future__ import annotations

import pytest
import torch
from easymagpie_vllm_omni.cfg_graph import CFGGraphBuffers
from easymagpie_vllm_omni.cfg_pairing import CFGPairingLayout
from torch.profiler import ProfilerActivity, profile


def test_graph_map_updates_reuse_host_storage(monkeypatch: pytest.MonkeyPatch) -> None:
    """Repeated updates must not allocate fresh pageable tensor upload sources."""
    mapping = CFGGraphBuffers(8, device="cpu")
    addresses = [x.data_ptr() for x in (mapping.owners, mapping.partners, mapping.active)]

    def reject_allocation(*args, **kwargs):
        raise AssertionError("graph update allocated a fresh CPU tensor")

    monkeypatch.setattr(torch, "tensor", reject_allocation)
    for _ in range(3):
        mapping.update(CFGPairingLayout(("a",), ((4, 5),), ((0, 1),)), 6, 8)
        assert mapping.owners.tolist() == [4, 5, 2, 3, 4, 5, 6, 7]
        assert mapping.partners.tolist() == [0, 1, 2, 3, 0, 1, 6, 7]
        mapping.update(None, 1, 4)
        assert mapping.owners[:4].tolist() == list(range(4))
        mapping.update(None, 0, 0)
        mapping.update(CFGPairingLayout(("b",), ((0,),), ((6,),)), 7, 8)
        assert mapping.owners.tolist() == [0, 1, 2, 3, 4, 5, 0, 7]
        assert mapping.partners.tolist() == [6, 1, 2, 3, 4, 5, 6, 7]
        assert mapping.active.tolist() == [True] * 7 + [False]
    assert addresses == [x.data_ptr() for x in (mapping.owners, mapping.partners, mapping.active)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_graph_map_uploads_do_not_block_the_host() -> None:
    """An available staging slot needs no host wait for its two uploads."""
    mapping = CFGGraphBuffers(64, device="cuda")
    mapping.update(None, 64, 64)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        mapping.update(CFGPairingLayout(("a",), ((1,),), ((0,),)), 63, 64)
    waits = {e.key: e.count for e in prof.key_averages() if "synchronize" in e.key.lower()}
    assert not waits, waits
    assert mapping.owners[:3].tolist() == [1, 1, 2]
    assert mapping.partners[:3].tolist() == [0, 0, 2]
    assert mapping.active.tolist() == [True] * 63 + [False]


def test_cpu_staging_never_uses_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """CPU buffers remain usable without initializing a CUDA runtime."""

    def reject_cuda(*args, **kwargs):
        raise AssertionError("CPU staging used CUDA")

    monkeypatch.setattr(torch.cuda, "Event", reject_cuda)
    monkeypatch.setattr(torch.cuda, "current_stream", reject_cuda)
    mapping = CFGGraphBuffers(4, device="cpu")
    for _ in range(4):
        mapping.update(None, 0, 4)
        assert not mapping.active.any()
        assert not any(slot.is_pinned() for slot in mapping._host_slots)


@pytest.mark.parametrize("already_complete", [False, True])
def test_staging_slot_is_not_overwritten_before_copy_completion(monkeypatch, already_complete) -> None:
    """Deferred copies must read the original slot even after other updates."""
    mapping = CFGGraphBuffers(8, device="cpu")
    original_copy = torch.Tensor.copy_
    pending = []
    consumed = []
    stream = object()

    class CopyEvent:
        def __init__(self):
            self.copies = []
            self.waits = 0

        def query(self):
            return not self.copies

        def record(self, actual_stream):
            assert actual_stream is stream
            assert not self.copies
            self.copies = pending[:]
            pending.clear()

        def finish(self):
            # Read the host memory only now, as a deferred DMA transfer would.
            consumed.append([source.clone() for source in self.copies])
            self.copies.clear()

        def synchronize(self):
            self.waits += 1
            self.finish()

    def defer_copy(destination, source, *, non_blocking=False):
        assert non_blocking
        pending.append(source)
        return destination

    mapping._copy_events = events = [CopyEvent(), CopyEvent()]
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: stream)
    monkeypatch.setattr(torch.Tensor, "copy_", defer_copy)
    layout = CFGPairingLayout(("a",), ((4, 5),), ((0, 1),))
    mapping.update(layout, 6, 8)
    first_sources = [source.clone() for source in events[0].copies]
    mapping.update(None, 1, 4)
    assert not consumed
    for source, expected in zip(events[0].copies, first_sources):
        assert torch.equal(source, expected)
    # Empty and rejected updates must neither acquire nor overwrite a slot.
    mapping.update(None, 0, 0)
    with pytest.raises(ValueError):
        mapping.update(CFGPairingLayout(("bad",), ((0,),), ((8,),)), 8, 8)
    assert not consumed
    if already_complete:
        events[0].finish()
    mapping.update(CFGPairingLayout(("b",), ((0,),), ((6,),)), 7, 8)
    assert events[0].waits == (0 if already_complete else 1)
    for actual, expected in zip(consumed[0], first_sources):
        assert torch.equal(actual, expected)
    mapping.update(None, 8, 8)
    assert events[1].waits == 1
    assert consumed[1][0].tolist() == list(range(4))
    assert consumed[1][1].tolist() == list(range(4))
    assert not pending
    monkeypatch.setattr(torch.Tensor, "copy_", original_copy)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_delayed_cuda_consumers_observe_each_mapping() -> None:
    """Queued consumers see their own maps across slot reuse and batch changes."""
    mapping = CFGGraphBuffers(8, device="cuda")
    assert all(slot.is_pinned() for slot in mapping._host_slots)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    cases = [
        (CFGPairingLayout(("a",), ((4, 5),), ((0, 1),)), 6, 8),
        (None, 1, 4),
        (None, 0, 0),
        (CFGPairingLayout(("b",), ((0,),), ((6,),)), 7, 8),
        (None, 0, 8),
        (None, 8, 8),
    ]
    snapshots = []
    expected = CFGGraphBuffers(8, device="cpu")
    with torch.cuda.stream(stream):
        for _ in range(3):
            for layout, active, padded in cases:
                torch.cuda._sleep(2_000_000)
                mapping.update(layout, active, padded)
                expected.update(layout, active, padded)
                snapshots.append(
                    (
                        [value.clone() for value in (mapping.owners, mapping.partners, mapping.active)],
                        [value.clone() for value in (expected.owners, expected.partners, expected.active)],
                    )
                )
    stream.synchronize()
    for actual, reference in snapshots:
        for value, wanted in zip(actual, reference):
            torch.testing.assert_close(value.cpu(), wanted)
