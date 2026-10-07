# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
"""Pin the contract that makes splitting inference from scoring worth anything.

Scoring a manifest offline must produce exactly what scoring it inline produced. If the two can
disagree, the split has bought nothing: every offline number would need re-checking against a GPU
run, which is the situation this work exists to end.

Also pins the scorer's refusals. Each one exists because the alternative is a *wrong number rather
than an error* -- the failure mode that is impossible to notice downstream.
"""

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.speechlm2.streaming_stt_score import (  # noqa: E402
    CpWERScoreConfig,
    _default_path,
    _json_safe,
    _refuse_unscorable,
    main,
)
from nemo.collections.speechlm2.parts.metrics import CpWER, score_rows  # noqa: E402
from nemo.collections.speechlm2.parts.metrics.cpwer_report import cpwer_metrics_dict  # noqa: E402

_ROWS = [
    {"id": "a", "text_raw": "<spk:0> hello there <spk:1> hi", "pred_text_raw": "<spk:0> hello world <spk:1> hi"},
    {"id": "b", "text_raw": "<spk:0> good day", "pred_text_raw": "<spk:0> good day"},
    {"id": "c", "text_raw": "<spk:0> one two three", "pred_text_raw": "<spk:0> one two"},
]


def _with_run(rows, **run):
    base = {"placement": "prefix", "seg_mode": False, "inference_normalizer": "whisper"}
    return [{**r, "_run": {**base, **run}} for r in rows]


# --------------------------------------------------------------------------------------------
# the contract
# --------------------------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"cpwer_normalizer": "chime8"},
        {"cpwer_speaker_order": "first_seen", "cpwer_keep_empty_streams": False},
        {"cpwer_placement_ref": "prefix", "cpwer_placement_hyp": "suffix"},
    ],
)
def test_offline_scoring_equals_inline_scoring(overrides):
    """The whole point of the split: the same settings give the same numbers either way."""
    cfg = CpWERScoreConfig(manifest="unused", **overrides)

    # Inline: what the generate script does -- score each session, then aggregate.
    metric = CpWER.from_config(cfg)
    inline_rows = [metric.score_session(r["text_raw"], r["pred_text_raw"]) for r in _ROWS]
    metric.update("corpus", [r["text_raw"] for r in _ROWS], [r["pred_text_raw"] for r in _ROWS])
    inline_corpus = metric.compute()

    # Offline: what the scorer does -- read a manifest and score it.
    offline_rows, offline_corpus, _ = score_rows(_ROWS, cfg)

    assert offline_corpus["cpwer_corpus"] == inline_corpus["cpwer_corpus"]
    assert offline_corpus["cpwer_macro_corpus"] == inline_corpus["cpwer_macro_corpus"]
    assert offline_corpus["cpwer_errors_corpus"] == inline_corpus["cpwer_errors_corpus"]
    for expected, actual in zip(inline_rows, offline_rows):
        assert actual["cpwer"] == expected.cpwer
        assert actual["cpwer_errors"] == expected.errors
        assert actual["cpwer_ref_words"] == expected.ref_words


@pytest.mark.unit
def test_the_metrics_dict_is_identical_from_either_path():
    cfg = CpWERScoreConfig(manifest="unused")
    metric = CpWER.from_config(cfg)
    metric.update("corpus", [r["text_raw"] for r in _ROWS], [r["pred_text_raw"] for r in _ROWS])
    inline = cpwer_metrics_dict(metric.compute(), {}, cfg)
    _, corpus, subsets = score_rows(_ROWS, cfg)
    offline = cpwer_metrics_dict(corpus, subsets, cfg)
    # `cpwer_rows_without_subset` differs only because the inline path did not count rows.
    for key in ("cpwer", "cpwer_macro", "cpwer_errors", "cpwer_ref_words", "cpwer_axes"):
        assert inline[key] == offline[key], key


# --------------------------------------------------------------------------------------------
# refusals
# --------------------------------------------------------------------------------------------


@pytest.mark.unit
def test_pre_split_manifest_is_refused_not_scored():
    """`text` was normalized and tag-stripped; scoring it gives a wrong number, not an error."""
    with pytest.raises(ValueError, match="no raw reference"):
        _refuse_unscorable([{"id": "a", "text": "hello there", "pred_text": "hello there"}])


@pytest.mark.unit
def test_segmented_manifest_is_refused():
    """Speaker indices are arrival-ordered per decode window, so they are not session-global."""
    with pytest.raises(ValueError, match="globally consistent speaker indices"):
        _refuse_unscorable(_with_run(_ROWS, seg_mode=True))


@pytest.mark.unit
def test_rows_disagreeing_on_placement_are_refused():
    """A concatenation of two incomparable runs; scoring it would average them silently."""
    rows = _with_run(_ROWS[:1]) + _with_run(_ROWS[1:], placement="suffix")
    with pytest.raises(ValueError, match="disagree on tag placement"):
        _refuse_unscorable(rows)


@pytest.mark.unit
def test_rows_disagreeing_on_a_role_placement_are_refused():
    """The same `placement`, but one run read its references as prefix: two runs, not one."""
    rows = _with_run(_ROWS[:1], placement="suffix") + _with_run(_ROWS[1:], placement="suffix", placement_ref="prefix")
    match = r"disagree on tag placement \(reference, hypothesis: \[\('prefix', 'suffix'\), \('suffix', 'suffix'\)\]\)"
    with pytest.raises(ValueError, match=match):
        _refuse_unscorable(rows)


@pytest.mark.unit
def test_rows_that_resolve_to_the_same_placements_are_accepted():
    """What is compared is the placement each role was read with, not how the run spelled it."""
    rows = _with_run(_ROWS[:1], placement="prefix", placement_hyp="suffix")
    rows += _with_run(_ROWS[1:], placement="suffix", placement_ref="prefix")
    _refuse_unscorable(rows)


@pytest.mark.unit
@pytest.mark.parametrize("dry_run", [False, True])
@pytest.mark.parametrize(
    "run,overrides,warned",
    [
        # Read as ref=prefix, hyp=suffix, re-scored with the defaults: prefix/prefix, a different number.
        ({"placement_hyp": "suffix"}, {}, True),
        ({"placement": "suffix"}, {}, True),
        ({"placement_hyp": "suffix"}, {"cpwer_placement_ref": "prefix", "cpwer_placement_hyp": "suffix"}, False),
        # The pair is compared, not its spelling.
        ({"placement_hyp": "suffix"}, {"cpwer_placement": "suffix", "cpwer_placement_ref": "prefix"}, False),
        ({}, {}, False),
        # No stamp to compare with.
        ({"placement": None}, {"cpwer_placement": "suffix"}, False),
    ],
    ids=["roles-vs-defaults", "suffix-vs-defaults", "roles-match", "roles-respelled", "defaults", "unstamped"],
)
def test_a_placement_other_than_the_runs_is_warned_about(tmp_path, monkeypatch, run, overrides, warned, dry_run):
    """Re-scoring under another placement is allowed, since references may be joined from elsewhere, but not silent."""
    from omegaconf import OmegaConf

    manifest = tmp_path / "run.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in _with_run(_ROWS, **run)))
    seen = []
    # NeMo's logger does not propagate to caplog, so capture the warning at the source.
    target = "examples.speechlm2.streaming_stt_score.logging.warning"
    monkeypatch.setattr(target, lambda msg, *a, **k: seen.append(msg))
    main(OmegaConf.structured(CpWERScoreConfig(manifest=str(manifest), dry_run=dry_run, **overrides)))
    assert any("tag placement" in msg for msg in seen) == warned, seen


@pytest.mark.unit
def test_a_well_formed_manifest_is_accepted():
    _refuse_unscorable(_with_run(_ROWS))


@pytest.mark.unit
def test_empty_manifest_is_refused():
    with pytest.raises(ValueError, match="empty"):
        _refuse_unscorable([])


# --------------------------------------------------------------------------------------------
# artifacts
# --------------------------------------------------------------------------------------------


@pytest.mark.unit
def test_outputs_never_overwrite_the_input():
    """A mis-configured re-score must not damage a GPU run's artifact."""
    manifest = Path("/tmp/eval/run.jsonl")
    assert _default_path(manifest, ".scored.jsonl") == Path("/tmp/eval/run.scored.jsonl")
    assert _default_path(manifest, ".metrics.json") == Path("/tmp/eval/run.metrics.json")


@pytest.mark.unit
def test_rescoring_a_scored_manifest_does_not_stack_suffixes():
    assert _default_path(Path("/tmp/run.scored.jsonl"), ".scored.jsonl") == Path("/tmp/run.scored.jsonl")


@pytest.mark.unit
def test_non_finite_floats_become_null_so_the_output_is_valid_json():
    """A bare `Infinity` is not JSON; a strict reader rejects the whole file over one row."""
    safe = _json_safe({"wer": float("inf"), "nested": [float("nan"), 1.0], "ok": "x"})
    assert safe["wer"] is None
    assert safe["nested"] == [None, 1.0]
    json.dumps(safe, allow_nan=False)  # would raise if anything non-finite survived


@pytest.mark.unit
def test_scorer_config_shares_every_axis_with_the_inference_config():
    """An override copied from one command line must work on the other."""
    from dataclasses import fields

    from examples.speechlm2.streaming_stt_generate import StreamingSTTEvalConfig
    from nemo.collections.speechlm2.parts.metrics.cpwer_scoring import AXIS_FIELDS

    score_fields = {f.name for f in fields(CpWERScoreConfig)}
    gen_fields = {f.name for f in fields(StreamingSTTEvalConfig)}
    assert set(AXIS_FIELDS) <= score_fields
    assert set(AXIS_FIELDS) <= gen_fields
