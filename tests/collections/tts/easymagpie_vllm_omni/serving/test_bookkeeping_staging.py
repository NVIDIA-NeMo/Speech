# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sampled request-row alignment must not rebuild blocking device indices."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from easymagpie_vllm_omni.cfg_pairing import CFGPair
from easymagpie_vllm_omni.runner import EasyMagpieGPUARModelRunner
from torch.profiler import ProfilerActivity, profile
from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner


def test_bookkeeping_reuses_index_storage(monkeypatch):
    runner, sampled = _fixture("cpu")

    def reject_tensor(*args, **kwargs):
        raise AssertionError("bookkeeping rebuilt index tensors from Python lists")

    monkeypatch.setattr(GPUARModelRunner, "_bookkeeping_sync", lambda *a, **k: None)
    monkeypatch.setattr(torch, "tensor", reject_tensor)
    storage = None
    for _ in range(4):
        runner._bookkeeping_sync(None, sampled)
        assert sampled.sampled_token_ids.tolist() == [[7, 8], [7, 8], [4, 5]]
        staging = runner._cfg_bookkeeping_indices
        addresses = [x.data_ptr() for x in [staging._device, *staging._host_slots]]
        if storage is not None:
            assert addresses == storage
        storage = addresses


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_bookkeeping_available_slot_does_not_wait(monkeypatch):
    runner, sampled = _fixture("cuda")
    monkeypatch.setattr(GPUARModelRunner, "_bookkeeping_sync", lambda *a, **k: None)
    runner._bookkeeping_sync(None, sampled)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        runner._bookkeeping_sync(None, sampled)
    waits = {event.key: event.count for event in prof.key_averages() if "synchronize" in event.key.lower()}
    assert not waits, waits
    assert sampled.sampled_token_ids.tolist() == [[7, 8], [7, 8], [4, 5]]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_changing_request_rows_preserves_draws_discard_and_native_order(monkeypatch, device):
    runner, _ = _fixture(device)
    pairs = [CFGPair.create("a"), CFGPair.create("b")]
    runner.requests = {
        p.unconditional_id: SimpleNamespace(additional_information_cpu=p.metadata(p.unconditional_id)) for p in pairs
    }
    observed = []
    marker = None

    def native(self, step, output, *args, **kwargs):
        assert step is marker and args == ("native-argument",) and kwargs == {"native_keyword": True}
        observed.append(output.sampled_token_ids.clone())
        return "native-result"

    monkeypatch.setattr(GPUARModelRunner, "_bookkeeping_sync", native)
    for ids, discarded in [
        ([pairs[1].unconditional_id, "a", "ordinary", pairs[0].unconditional_id, "b"], {"a"}),
        (["b", pairs[0].unconditional_id, "a", pairs[1].unconditional_id, "ordinary"], set()),
        (["ordinary"], set()),
        (["a", pairs[0].unconditional_id], {"a"}),
        ([pairs[1].unconditional_id, "b"], set()),
    ] * 3:
        runner.input_batch.req_ids = ids
        mask = np.array([rid in discarded or rid == pairs[0].unconditional_id and "a" in discarded for rid in ids])
        runner.discard_request_mask.np = mask
        tokens = torch.arange(len(ids) * 3, device=device).reshape(len(ids), 3)
        before = tokens.clone()
        expected = before.clone()
        for pair in pairs:
            if pair.unconditional_id in ids and pair.conditional_id not in discarded:
                expected[ids.index(pair.unconditional_id)] = before[ids.index(pair.conditional_id)]
        # Multi-token prefill counts are unrelated to these sampled request rows.
        marker = SimpleNamespace(num_scheduled_tokens=dict(zip(ids, range(2, 2 + len(ids)))))
        assert sum(marker.num_scheduled_tokens.values()) > len(ids)
        sampled = SimpleNamespace(sampled_token_ids=tokens)
        rng = torch.random.get_rng_state().clone()
        assert runner._bookkeeping_sync(marker, sampled, "native-argument", native_keyword=True) == "native-result"
        assert torch.equal(torch.random.get_rng_state(), rng)
        torch.testing.assert_close(observed[-1], expected)
        torch.testing.assert_close(sampled.sampled_token_ids, expected)


def test_rejected_or_unpaired_batches_do_not_acquire_staging(monkeypatch):
    runner, sampled = _fixture("cpu")
    calls = []
    monkeypatch.setattr(GPUARModelRunner, "_bookkeeping_sync", lambda *a, **k: calls.append(True))
    private = runner.input_batch.req_ids[0]
    runner.input_batch.req_ids = [private, "ordinary"]
    with pytest.raises(RuntimeError, match="partner missing"):
        runner._bookkeeping_sync(None, sampled)
    assert not hasattr(runner, "_cfg_bookkeeping_indices")
    runner.input_batch.req_ids = [private, "a", "ordinary"]
    runner.discard_request_mask.np[0] = True
    with pytest.raises(RuntimeError, match="discard masks disagree"):
        runner._bookkeeping_sync(None, sampled)
    assert not hasattr(runner, "_cfg_bookkeeping_indices") and not calls
    runner.discard_request_mask.np[1] = True
    runner._bookkeeping_sync(None, sampled)
    assert not hasattr(runner, "_cfg_bookkeeping_indices") and calls == [True]


def test_cpu_indices_never_use_cuda_and_keep_storage(monkeypatch):
    from easymagpie_vllm_omni.runner import _CFGBookkeepingIndices

    def reject_cuda(*args, **kwargs):
        raise AssertionError("CPU bookkeeping used CUDA")

    monkeypatch.setattr(torch.cuda, "Event", reject_cuda)
    monkeypatch.setattr(torch.cuda, "current_stream", reject_cuda)
    staging = _CFGBookkeepingIndices(8, torch.device("cpu"))
    addresses = [x.data_ptr() for x in [staging._device, *staging._host_slots]]
    for _ in range(3):
        for sources, destinations in [([4, 0], [1, 3]), ([2], [0]), ([1, 0, 4], [3, 5, 6])]:
            src, dst = staging.stage(sources, destinations)
            assert src.tolist() == sources and dst.tolist() == destinations
    assert addresses == [x.data_ptr() for x in [staging._device, *staging._host_slots]]
    assert not any(x.is_pinned() for x in staging._host_slots)


def test_invalid_indices_do_not_overwrite_staged_storage():
    from easymagpie_vllm_omni.runner import _CFGBookkeepingIndices

    staging = _CFGBookkeepingIndices(2, torch.device("cpu"))
    src, dst = staging.stage([1], [0])
    next_slot = staging._next_slot
    for sources, destinations in [([0, 1], [1]), ([0, 1, 2], [2, 1, 0])]:
        with pytest.raises(ValueError, match="request capacity or disagree"):
            staging.stage(sources, destinations)
        assert staging._next_slot == next_slot
        assert src.tolist() == [1] and dst.tolist() == [0]


@pytest.mark.parametrize("already_complete", [False, True])
def test_packed_host_indices_survive_pending_slot_reuse(monkeypatch, already_complete):
    from easymagpie_vllm_omni.runner import _CFGBookkeepingIndices

    staging = _CFGBookkeepingIndices(8, torch.device("cpu"))
    pending, consumed = [], []
    stream = object()

    class Event:
        def __init__(self):
            self.sources = []
            self.waits = 0

        def query(self):
            return not self.sources

        def record(self, actual_stream):
            assert actual_stream is stream and not self.sources
            self.sources[:] = pending
            pending.clear()

        def finish(self):
            consumed.extend(source.clone() for source in self.sources)
            self.sources.clear()

        def synchronize(self):
            self.waits += 1
            self.finish()

    def defer_copy(destination, source, *, non_blocking=False):
        assert non_blocking and source.is_contiguous()
        pending.append(source)
        return destination

    staging._copy_events = events = [Event(), Event()]
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: stream)
    monkeypatch.setattr(torch.Tensor, "copy_", defer_copy)
    staging.stage([4, 0], [1, 3])
    staging.stage([2], [0])
    assert not consumed
    assert events[0].sources[0].tolist() == [4, 0, 1, 3]
    if already_complete:
        events[0].finish()
    staging.stage([1, 0, 4], [3, 5, 6])
    assert events[0].waits == (0 if already_complete else 1)
    assert consumed[0].tolist() == [4, 0, 1, 3]
    staging.stage([0], [7])
    assert events[1].waits == 1
    assert consumed[1].tolist() == [2, 0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_queued_cuda_bookkeeping_consumers_keep_their_request_rows(monkeypatch):
    runner, sampled = _fixture("cuda")
    monkeypatch.setattr(GPUARModelRunner, "_bookkeeping_sync", lambda *a, **k: None)
    private = runner.input_batch.req_ids[0]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    snapshots = []
    with torch.cuda.stream(stream):
        for i in range(8):
            ids = [private, "a", "ordinary"] if i % 2 == 0 else ["a", "ordinary", private]
            runner.input_batch.req_ids = ids
            sampled.sampled_token_ids = torch.full((3, 2), -1, device="cuda", dtype=torch.long)
            sampled.sampled_token_ids[ids.index("a")].fill_(i)
            torch.cuda._sleep(2_000_000)
            runner._bookkeeping_sync(None, sampled)
            snapshots.append((sampled.sampled_token_ids.clone(), ids, i))
    stream.synchronize()
    for tokens, ids, i in snapshots:
        assert tokens[ids.index(private)].tolist() == [i, i]
        assert tokens[ids.index("a")].tolist() == [i, i]
        assert tokens[ids.index("ordinary")].tolist() == [-1, -1]


def _fixture(device):
    runner = object.__new__(EasyMagpieGPUARModelRunner)
    pair = CFGPair.create("a")
    runner.input_batch = SimpleNamespace(req_ids=[pair.unconditional_id, "a", "ordinary"], max_num_reqs=8)
    runner.requests = {
        pair.unconditional_id: SimpleNamespace(additional_information_cpu=pair.metadata(pair.unconditional_id))
    }
    runner.discard_request_mask = SimpleNamespace(np=np.zeros(3, dtype=bool))
    sampled = SimpleNamespace(sampled_token_ids=torch.tensor([[90, 91], [7, 8], [4, 5]], device=device))
    return runner, sampled
