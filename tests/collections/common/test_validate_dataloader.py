# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Unit tests for ``scripts/dataloading/_validate_dataloader/{pre_validation,consolidate}``.

These cover the parts that can run without a SLURM cluster or real
Lhotse manifests:

  * pre-validation static checks across hand-crafted config snippets
  * consolidate() against synthesized JSONL rows (PASS / FAIL / SKIP)
  * config_inject recursive walker
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

# The validator lives under scripts/, which isn't on PYTHONPATH by default.
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "dataloading"))

from _validate_dataloader import config_inject
from _validate_dataloader import consolidate as cons
from _validate_dataloader import pre_validation as pv
from _validate_dataloader.cut_id_dataset import CutIdDataset
from validate_dataloader import (
    _extract_cuts,
    _extract_numeric_list,
    _extract_semantic_cut_ids,
)

from nemo.collections.common.data.lhotse import cutset as cutset_module

# --------------------------------------------------------------------------- #
# graph identity
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_cut_id_dataset_uses_graph_origin_when_semantic_ids_repeat():
    class Example:
        def __init__(self, semantic_id, graph_origin, duration, num_tokens):
            self.id = semantic_id
            self._graph_origin = graph_origin
            self.duration = duration
            self.num_tokens = num_tokens

    row = CutIdDataset()[
        [Example("0", 17, 1.25, 10), Example("0", (3, 9), 2.5, 20)]
    ]

    assert row["cut_ids"] == ["graph:17", "graph:[3,9]"]
    assert row["semantic_cut_ids"] == ["0", "0"]
    assert row["declared_duration_seconds"] == [1.25, 2.5]
    assert row["sampled_num_tokens"] == [10, 20]
    assert len(set(row["cut_ids"])) == 2


@pytest.mark.unit
def test_cut_id_dataset_legacy_fallback_has_separate_namespace():
    class Example:
        id = "17"

    row = CutIdDataset()[[Example()]]

    assert row["cut_ids"] == ['semantic:"17"']
    assert row["semantic_cut_ids"] == ["17"]
    assert row["declared_duration_seconds"] == [0.0]
    assert row["sampled_num_tokens"] == [-1]


@pytest.mark.unit
def test_cut_id_dataset_sums_declared_conversation_audio_without_loading_it():
    class Cut:
        def __init__(self, duration):
            self.duration = duration

    class Conversation:
        id = "conversation"
        _graph_origin = (4, 2)
        num_tokens = 17

        def list_cuts(self):
            return [Cut(1.25), Cut(2.5)]

    row = CutIdDataset()[[Conversation()]]

    assert row["declared_duration_seconds"] == [3.75]
    assert row["sampled_num_tokens"] == [17]


@pytest.mark.unit
def test_extract_numeric_list_handles_collated_vectors_and_checks_cardinality():
    batch = {
        "declared_duration_seconds": [[1.25, 2.5]],
        "sampled_num_tokens": [[10, 20]],
    }

    assert _extract_numeric_list(batch, "declared_duration_seconds", 2, float) == [
        1.25,
        2.5,
    ]
    assert _extract_numeric_list(batch, "sampled_num_tokens", 2, int) == [10, 20]
    with pytest.raises(Exception, match="cardinality mismatch"):
        _extract_numeric_list(batch, "sampled_num_tokens", 3, int)


@pytest.mark.unit
def test_full_mode_conversations_use_graph_origin_when_semantic_ids_repeat():
    class Conversation:
        def __init__(self, semantic_id, graph_origin):
            self.id = semantic_id
            self._graph_origin = graph_origin

    conversations = [Conversation("0", 17), Conversation("0", (3, 9))]
    batch = {"conversations": conversations}

    cut_ids, worker_id = _extract_cuts(batch)

    assert cut_ids == ["graph:17", "graph:[3,9]"]
    assert worker_id == 0
    assert _extract_semantic_cut_ids(batch, len(cut_ids)) == ["0", "0"]


# --------------------------------------------------------------------------- #
# config_inject
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_config_inject_top_level_and_nested():
    cfg = OmegaConf.create(
        {
            "input_cfg": [
                {
                    "type": "lhotse_as_conversation",
                    "input_cfg": [
                        {"type": "lhotse_shar", "weight": 1.0},
                        {"type": "nemo_tarred", "weight": 0.5},
                    ],
                },
                {
                    "type": "group",
                    "input_cfg": [{"type": "lhotse_shar", "weight": 0.3}],
                },
            ],
        }
    )
    config_inject.inject_validator_flags(cfg, force_finite=True, metadata_only=True)
    assert cfg["force_finite"] is True
    assert cfg["metadata_only"] is True
    for transform in cfg["input_cfg"]:
        assert transform["force_finite"] is True
        assert transform["metadata_only"] is True
        for leaf in transform["input_cfg"]:
            assert leaf["force_finite"] is True
            assert leaf["metadata_only"] is True


@pytest.mark.unit
def test_config_inject_validator_modes_override_explicit_nested_false():
    cfg = OmegaConf.create(
        {
            "input_cfg": [
                {
                    "type": "group",
                    "force_finite": False,
                    "metadata_only": False,
                    "input_cfg": [
                        {
                            "type": "share_gpt_webdataset",
                            "force_finite": False,
                            "metadata_only": False,
                        }
                    ],
                }
            ]
        }
    )

    config_inject.inject_validator_flags(cfg, force_finite=True, metadata_only=True)

    group = cfg.input_cfg[0]
    leaf = group.input_cfg[0]
    assert group.force_finite is True
    assert group.metadata_only is True
    assert leaf.force_finite is True
    assert leaf.metadata_only is True


@pytest.mark.unit
def test_groundtruth_overrides_iterable_recipe_with_map_loader():
    cfg = OmegaConf.create(
        {
            "num_workers": 4,
            "use_stateful_dataloader": True,
            "force_iterable_dataset": True,
            "force_map_dataset": False,
        }
    )

    config_inject.inject_groundtruth_flags(cfg)

    assert cfg.num_workers == 0
    assert cfg.use_stateful_dataloader is False
    assert cfg.force_iterable_dataset is False
    assert cfg.force_map_dataset is True


@pytest.mark.unit
def test_missing_manifest_policy_overrides_explicit_nested_values():
    cfg = OmegaConf.create(
        {
            "skip_missing_manifest_entries": True,
            "input_cfg": [
                {
                    "type": "group",
                    "skip_missing_manifest_entries": True,
                    "input_cfg": [
                        {
                            "type": "share_gpt_webdataset",
                            "skip_missing_manifest_entries": True,
                        }
                    ],
                }
            ],
        }
    )

    config_inject.inject_missing_manifest_policy(
        cfg, skip_missing_manifest_entries=False
    )

    assert cfg.skip_missing_manifest_entries is False
    assert cfg.input_cfg[0].skip_missing_manifest_entries is False
    assert cfg.input_cfg[0].input_cfg[0].skip_missing_manifest_entries is False


@pytest.mark.unit
def test_missing_manifest_policy_overrides_external_input_cfg_yaml(
    tmp_path, monkeypatch
):
    external = tmp_path / "external.yaml"
    external.write_text(
        OmegaConf.to_yaml(
            [
                {
                    "type": "nemo",
                    "manifest_filepath": "unused.jsonl",
                    "skip_missing_manifest_entries": True,
                }
            ]
        )
    )
    cfg = OmegaConf.create(
        {
            "input_cfg": str(external),
            "skip_missing_manifest_entries": True,
        }
    )
    config_inject.inject_missing_manifest_policy(
        cfg, skip_missing_manifest_entries=False
    )
    observed = []

    def capture_group(item, propagate_attrs):
        observed.append((item, propagate_attrs.copy()))
        return object(), False

    monkeypatch.setattr(cutset_module, "parse_group", capture_group)
    cutset_module.read_cutset_from_config(cfg)

    assert len(observed) == 1
    item, propagated = observed[0]
    assert item.skip_missing_manifest_entries is False
    assert propagated["skip_missing_manifest_entries"] is False


@pytest.mark.unit
@pytest.mark.parametrize("loader_policy", [False, True])
@pytest.mark.parametrize("leaf_policy", [False, True])
def test_audio_loading_policy_overrides_external_input_cfg_yaml(tmp_path, monkeypatch, loader_policy, leaf_policy):
    external = tmp_path / "external.yaml"
    external.write_text(
        OmegaConf.to_yaml(
            [
                {
                    "type": "nemo",
                    "manifest_filepath": "unused.jsonl",
                    "fault_tolerant_audio_loading": leaf_policy,
                }
            ]
        )
    )
    cfg = OmegaConf.create(
        {
            "input_cfg": str(external),
            "fault_tolerant_audio_loading": loader_policy,
        }
    )
    observed = []

    def capture_group(item, propagate_attrs):
        observed.append((item, propagate_attrs.copy()))
        return object(), False

    monkeypatch.setattr(cutset_module, "parse_group", capture_group)
    cutset_module.read_cutset_from_config(cfg)

    assert len(observed) == 1
    item, propagated = observed[0]
    assert item.fault_tolerant_audio_loading is loader_policy
    assert propagated["fault_tolerant_audio_loading"] is loader_policy


# --------------------------------------------------------------------------- #
# pre_validation
# --------------------------------------------------------------------------- #


def _base_cfg():
    return OmegaConf.create(
        {
            "seed": 42,
            "shard_seed": 42,
            "use_stateful_dataloader": True,
            "indexed": True,
            "indexes_root": "/tmp/idx_does_not_exist_locally",
            "use_bucketing": True,
            "num_buckets": 20,
            "bucket_buffer_size": 20000,
            "force_map_dataset": False,
            "text_field": "answer",
            "input_cfg": [
                {
                    "type": "lhotse_as_conversation",
                    "input_cfg": [
                        {"type": "lhotse_shar", "weight": 1.0, "corpus": "ami"},
                        {
                            "type": "nemo_tarred",
                            "weight": 0.13,
                            "corpus": "librilight",
                            "text_field": "answer",
                            "manifest_filepath": "s3://x/manifest__OP_0..15_CL_.jsonl",
                            "tarred_audio_filepaths": "s3://x/audio__OP_0..15_CL_.tar",
                        },
                    ],
                },
            ],
        }
    )


@pytest.mark.unit
def test_pre_validation_passing_config():
    report = pv.run_pre_validation(_base_cfg())
    fails = [c for c in report.checks if c.status == pv.FAIL]
    assert not fails, f"unexpected FAILs: {[(c.check_id, c.detail) for c in fails]}"


@pytest.mark.unit
def test_pre_validation_seed_int_fail():
    cfg = _base_cfg()
    cfg.seed = "randomized"
    report = pv.run_pre_validation(cfg)
    seed_check = next(c for c in report.checks if c.check_id == "seed-int")
    assert seed_check.status == pv.FAIL


@pytest.mark.unit
def test_pre_validation_shard_seed_int_fail():
    cfg = _base_cfg()
    cfg.shard_seed = "randomized"
    report = pv.run_pre_validation(cfg)
    shard_check = next(c for c in report.checks if c.check_id == "shard-seed-int")
    assert shard_check.status == pv.FAIL
    mux_check = next(
        c for c in report.checks if c.check_id == "mux-seed-not-randomized"
    )
    # force_map_dataset is False in base config, so this also fires.
    assert mux_check.status == pv.FAIL


@pytest.mark.unit
def test_pre_validation_stateful_off_fail():
    cfg = _base_cfg()
    cfg.use_stateful_dataloader = False
    report = pv.run_pre_validation(cfg)
    check = next(c for c in report.checks if c.check_id == "stateful-on")
    assert check.status == pv.FAIL


@pytest.mark.unit
def test_pre_validation_indexed_implies_root_fail():
    cfg = _base_cfg()
    cfg.indexes_root = None
    report = pv.run_pre_validation(cfg)
    check = next(c for c in report.checks if c.check_id == "indexed-implies-root")
    assert check.status == pv.FAIL


@pytest.mark.unit
def test_pre_validation_constant_time_leaves_fail_when_streaming():
    cfg = _base_cfg()
    cfg.indexed = False  # turns off propagation -> all leaves go streaming
    cfg.indexes_root = (
        None  # avoid the dependent indexed-implies-root failing on its own.
    )
    report = pv.run_pre_validation(cfg)
    check = next(c for c in report.checks if c.check_id == "constant-time-leaves")
    assert check.status == pv.FAIL


@pytest.mark.unit
def test_pre_validation_constant_time_leaves_fail_for_map_style_too():
    """User's correction: constant-time leaves are required for both
    map (force_map_dataset=True) and iterable (force_map_dataset=False)."""
    cfg = _base_cfg()
    cfg.force_map_dataset = True
    cfg.indexed = False
    cfg.indexes_root = None
    report = pv.run_pre_validation(cfg)
    check = next(c for c in report.checks if c.check_id == "constant-time-leaves")
    assert check.status == pv.FAIL


@pytest.mark.unit
def test_pre_validation_slice_length_with_indexed_fail():
    cfg = _base_cfg()
    cfg["input_cfg"][0]["input_cfg"][0]["slice_length"] = 50
    report = pv.run_pre_validation(cfg)
    check = next(c for c in report.checks if c.check_id == "slice-length-vs-indexed")
    assert check.status == pv.FAIL


@pytest.mark.unit
def test_pre_validation_mux_weights_sum_fail():
    cfg = _base_cfg()
    cfg["input_cfg"][0]["input_cfg"][0]["weight"] = -1.0
    report = pv.run_pre_validation(cfg)
    check = next(c for c in report.checks if c.check_id == "mux-weights-sum")
    assert check.status == pv.FAIL


@pytest.mark.unit
def test_pre_validation_ignore_fail_downgrades_to_warn():
    cfg = _base_cfg()
    cfg.seed = "randomized"
    report = pv.run_pre_validation(cfg, ignore_fail=["seed-int"])
    check = next(c for c in report.checks if c.check_id == "seed-int")
    assert check.status == pv.WARN


@pytest.mark.unit
def test_pre_validation_bucketer_buffer_warn():
    cfg = _base_cfg()
    cfg.bucket_buffer_size = 50  # < 20 * 10
    report = pv.run_pre_validation(cfg)
    check = next(c for c in report.checks if c.check_id == "bucketer-buffer")
    assert check.status == pv.WARN


# --------------------------------------------------------------------------- #
# consolidate
# --------------------------------------------------------------------------- #


def _write_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)


def _row(rank, step, cut_ids, *, worker_id=0):
    return {
        "step": step,
        "rank": rank,
        "world_size": 2,
        "worker_id": worker_id,
        "cut_ids": cut_ids,
        "batch_size": len(cut_ids),
        "t_total_ms": 1.0,
        "t_first_batch_ms": None,
    }


@pytest.mark.unit
def test_consolidate_q1_q3_pass(tmp_path):
    """Two ranks, disjoint cuts, no duplication."""
    base = tmp_path / "baseline" / "run0"
    _write_jsonl(
        base / "rank_000.jsonl",
        [
            _row(0, 0, ["a", "b"]),
            _row(0, 1, ["c"]),
        ],
    )
    _write_jsonl(
        base / "rank_001.jsonl",
        [
            _row(1, 0, ["d", "e"]),
            _row(1, 1, ["f"]),
        ],
    )
    report = cons.consolidate(tmp_path, checkpoint_at=0, num_determinism_runs=1)
    q_by_id = {q.q_id: q for q in report.questions}
    assert q_by_id["Q1"].status == cons.PASS
    assert q_by_id["Q3"].status == cons.PASS


@pytest.mark.unit
def test_consolidate_q1_cross_rank_leak(tmp_path):
    base = tmp_path / "baseline" / "run0"
    _write_jsonl(base / "rank_000.jsonl", [_row(0, 0, ["shared", "a"])])
    _write_jsonl(base / "rank_001.jsonl", [_row(1, 0, ["shared", "b"])])
    report = cons.consolidate(tmp_path, checkpoint_at=0, num_determinism_runs=1)
    q1 = next(q for q in report.questions if q.q_id == "Q1")
    assert q1.status == cons.FAIL
    assert q1.tag == "partition-rank-leak"


@pytest.mark.unit
def test_consolidate_q1_repeated_id_in_same_worker_fails(tmp_path):
    base = tmp_path / "baseline" / "run0"
    _write_jsonl(base / "rank_000.jsonl", [_row(0, 0, ["same", "same"])])

    report = cons.consolidate(tmp_path, checkpoint_at=0, num_determinism_runs=1)

    q1 = next(q for q in report.questions if q.q_id == "Q1")
    assert q1.status == cons.FAIL
    assert q1.tag == "duplicate-within-worker"


@pytest.mark.unit
def test_consolidate_q3_full_broadcast(tmp_path):
    """Every rank sees the same cuts → broadcast tag."""
    base = tmp_path / "baseline" / "run0"
    same = ["a", "b", "c"]
    _write_jsonl(base / "rank_000.jsonl", [_row(0, 0, same)])
    _write_jsonl(base / "rank_001.jsonl", [_row(1, 0, same)])
    report = cons.consolidate(tmp_path, checkpoint_at=0, num_determinism_runs=1)
    q3 = next(q for q in report.questions if q.q_id == "Q3")
    assert q3.status == cons.FAIL
    assert "BROADCAST" in q3.detail


@pytest.mark.unit
def test_consolidate_q2_skip_without_groundtruth(tmp_path):
    base = tmp_path / "baseline" / "run0"
    _write_jsonl(base / "rank_000.jsonl", [_row(0, 0, ["a"])])
    report = cons.consolidate(tmp_path, checkpoint_at=0, num_determinism_runs=1)
    q2 = next(q for q in report.questions if q.q_id == "Q2")
    assert q2.status == cons.SKIP


@pytest.mark.unit
def test_consolidate_q2_skip_detects_missing(tmp_path):
    base = tmp_path / "baseline" / "run0"
    _write_jsonl(base / "rank_000.jsonl", [_row(0, 0, ["a", "b"])])
    _write_jsonl(
        tmp_path / "groundtruth" / "cuts.jsonl", [{"cut_ids": ["a", "b", "c"]}]
    )
    report = cons.consolidate(tmp_path, checkpoint_at=0, num_determinism_runs=1)
    q2 = next(q for q in report.questions if q.q_id == "Q2")
    assert q2.status == cons.FAIL
    assert q2.tag == "skip"


@pytest.mark.unit
def test_consolidate_q4_resume_match(tmp_path):
    """State is saved AFTER yielding baseline step ``checkpoint_at``, so
    resumed[0] should match baseline[checkpoint_at + 1]."""
    base = tmp_path / "baseline" / "run0"
    res = tmp_path / "resumed" / "run0"
    _write_jsonl(
        base / "rank_000.jsonl",
        [
            _row(0, 0, ["a"]),
            _row(0, 1, ["b"]),
            _row(0, 2, ["c"]),
        ],
    )
    # checkpoint_at=0 -> resumed[0] == baseline[1] == ["b"], resumed[1] == baseline[2] == ["c"]
    _write_jsonl(
        res / "rank_000.jsonl",
        [
            _row(0, 0, ["b"]),
            _row(0, 1, ["c"]),
        ],
    )
    report = cons.consolidate(tmp_path, checkpoint_at=0, num_determinism_runs=1)
    q4 = next(q for q in report.questions if q.q_id == "Q4")
    assert q4.status == cons.PASS


@pytest.mark.unit
def test_consolidate_q4_resume_diverges(tmp_path):
    base = tmp_path / "baseline" / "run0"
    res = tmp_path / "resumed" / "run0"
    _write_jsonl(
        base / "rank_000.jsonl",
        [_row(0, 0, ["a"]), _row(0, 1, ["b"]), _row(0, 2, ["c"])],
    )
    # checkpoint_at=0 -> resumed[0] should == baseline[1] == ["b"], but it's "DIFFERENT".
    _write_jsonl(
        res / "rank_000.jsonl",
        [_row(0, 0, ["DIFFERENT"]), _row(0, 1, ["c"])],
    )
    report = cons.consolidate(tmp_path, checkpoint_at=0, num_determinism_runs=1)
    q4 = next(q for q in report.questions if q.q_id == "Q4")
    assert q4.status == cons.FAIL
    assert q4.tag == "resume-rng-divergence"


@pytest.mark.unit
def test_consolidate_q4_resume_reordered_batch_fails(tmp_path):
    base = tmp_path / "baseline" / "run0"
    res = tmp_path / "resumed" / "run0"
    _write_jsonl(
        base / "rank_000.jsonl",
        [_row(0, 0, ["a"]), _row(0, 1, ["b", "c"])],
    )
    _write_jsonl(res / "rank_000.jsonl", [_row(0, 0, ["c", "b"])])

    report = cons.consolidate(tmp_path, checkpoint_at=0, num_determinism_runs=1)

    q4 = next(q for q in report.questions if q.q_id == "Q4")
    assert q4.status == cons.FAIL
    assert q4.tag == "resume-rng-divergence"


@pytest.mark.unit
@pytest.mark.parametrize(
    "resumed_rows",
    [
        [],
        [_row(0, 0, ["b"]), _row(0, 1, ["c"]), _row(0, 2, ["extra"])],
    ],
)
def test_consolidate_q4_requires_exact_baseline_tail_coverage(tmp_path, resumed_rows):
    base = tmp_path / "baseline" / "run0"
    res = tmp_path / "resumed" / "run0"
    _write_jsonl(
        base / "rank_000.jsonl",
        [_row(0, 0, ["a"]), _row(0, 1, ["b"]), _row(0, 2, ["c"])],
    )
    _write_jsonl(res / "rank_000.jsonl", resumed_rows)

    report = cons.consolidate(tmp_path, checkpoint_at=0, num_determinism_runs=1)

    q4 = next(q for q in report.questions if q.q_id == "Q4")
    assert q4.status == cons.FAIL
    assert q4.tag == "resume-length-mismatch"


@pytest.mark.unit
def test_consolidate_q5_determinism_match(tmp_path):
    for run in ("run0", "run1"):
        _write_jsonl(
            tmp_path / "baseline" / run / "rank_000.jsonl",
            [_row(0, 0, ["a"]), _row(0, 1, ["b"])],
        )
    report = cons.consolidate(tmp_path, checkpoint_at=0, num_determinism_runs=2)
    q5 = next(q for q in report.questions if q.q_id == "Q5")
    assert q5.status == cons.PASS


@pytest.mark.unit
def test_consolidate_q5_determinism_diverges(tmp_path):
    _write_jsonl(tmp_path / "baseline" / "run0" / "rank_000.jsonl", [_row(0, 0, ["a"])])
    _write_jsonl(
        tmp_path / "baseline" / "run1" / "rank_000.jsonl", [_row(0, 0, ["DIFFERENT"])]
    )
    report = cons.consolidate(tmp_path, checkpoint_at=0, num_determinism_runs=2)
    q5 = next(q for q in report.questions if q.q_id == "Q5")
    assert q5.status == cons.FAIL
    assert q5.tag == "non-determinism"


@pytest.mark.unit
def test_consolidate_q5_reordered_batch_fails(tmp_path):
    _write_jsonl(
        tmp_path / "baseline" / "run0" / "rank_000.jsonl",
        [_row(0, 0, ["a", "b"])],
    )
    _write_jsonl(
        tmp_path / "baseline" / "run1" / "rank_000.jsonl",
        [_row(0, 0, ["b", "a"])],
    )

    report = cons.consolidate(tmp_path, checkpoint_at=0, num_determinism_runs=2)

    q5 = next(q for q in report.questions if q.q_id == "Q5")
    assert q5.status == cons.FAIL
    assert q5.tag == "non-determinism"
