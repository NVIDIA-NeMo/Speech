# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the CFG physical-sequence coordination contract."""
from types import SimpleNamespace

import pytest
from easymagpie_vllm_omni.cfg_pairing import (
    CFGPair,
    CFGPairingError,
    CFGPairingTransferFilter,
    build_cfg_pairing_layout,
    filter_cfg_outputs,
)


def test_pair_has_distinct_physical_identity_and_rejects_public_reserved_id():
    pair = CFGPair.create("speech-1")
    assert pair.conditional_id == "speech-1"
    assert pair.unconditional_id != pair.conditional_id
    assert CFGPair.create("speech-1").unconditional_id != pair.unconditional_id
    with pytest.raises(ValueError, match="reserved"):
        CFGPair.create(pair.unconditional_id)


@pytest.mark.parametrize("lengths,positions", [((3, 2), (0, 0)), ((3, 3), (0, 1)), ((3, 0), (0, 0))])
def test_pair_rejects_partial_or_misaligned_scheduling(lengths, positions):
    pair = CFGPair.create("speech-1")
    counts = dict(zip(pair.physical_ids, lengths))
    computed = dict(zip(pair.physical_ids, positions))
    with pytest.raises(CFGPairingError, match="aligned"):
        pair.validate_step(counts, computed)


def test_pair_accepts_joint_pause_and_aligned_prefill_or_decode():
    pair = CFGPair.create("speech-1")
    for count, position in ((0, 0), (7, 0), (1, 7)):
        pair.validate_step(dict.fromkeys(pair.physical_ids, count), dict.fromkeys(pair.physical_ids, position))


def test_layout_pairs_token_rows_by_identity_after_reordering():
    pair = CFGPair.create("speech-1")
    other = CFGPair.create("speech-2")
    ids = [other.unconditional_id, "plain", pair.conditional_id, other.conditional_id, pair.unconditional_id]
    metadata = {rid: p.metadata(rid) for p in (pair, other) for rid in p.physical_ids}
    layout = build_cfg_pairing_layout(ids, [2, 1, 3, 2, 3], [4, 8, 0, 4, 0], metadata)
    assert layout.logical_ids == ("speech-1", "speech-2")
    assert layout.conditional_rows == ((3, 4, 5), (6, 7))
    assert layout.unconditional_rows == ((8, 9, 10), (0, 1))


def test_layout_is_empty_for_ordinary_batch_and_rejects_missing_partner():
    assert build_cfg_pairing_layout(["plain"], [1], [0], {}).logical_ids == ()
    pair = CFGPair.create("speech-1")
    with pytest.raises(CFGPairingError, match="partner"):
        build_cfg_pairing_layout(
            [pair.conditional_id], [1], [0], {pair.conditional_id: pair.metadata(pair.conditional_id)}
        )


def test_transfer_filter_suppresses_hidden_payloads_and_terminal_notifications():
    pair = CFGPair.create("speech-1")
    calls = []
    adapter = SimpleNamespace(
        save_async=lambda *args, **kwargs: calls.append((args, kwargs)),
        finish_requests=lambda *args: calls.append(args),
        marker=object(),
    )
    filtered = CFGPairingTransferFilter(adapter)
    for rid in reversed(pair.physical_ids):
        filtered.save_async("codes", SimpleNamespace(request_id=rid), True)
    assert len(calls) == 1
    assert calls[0][0][1].request_id == pair.conditional_id
    filtered.finish_requests(pair.physical_ids, "abort", {})
    assert calls[-1][0] == [pair.conditional_id]
    assert filtered.marker is adapter.marker


def test_output_filter_removes_hidden_outputs_and_finished_ids_after_registry_cleanup():
    pair = CFGPair.create("speech-1")
    result = {
        0: SimpleNamespace(
            outputs=[SimpleNamespace(request_id=rid) for rid in reversed(pair.physical_ids)],
            finished_requests=set(pair.physical_ids),
        )
    }
    filter_cfg_outputs(result)
    assert [output.request_id for output in result[0].outputs] == [pair.conditional_id]
    assert result[0].finished_requests == {pair.conditional_id}
