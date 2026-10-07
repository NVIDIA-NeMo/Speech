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
"""Assert the scoring path against checked-in golden output.

The unit tests elsewhere pin individual behaviours; this one pins the whole pipeline end to end on
a fixed corpus, so a change that is individually defensible but collectively moves a number fails
here. Exact integers, no tolerance: every count is discrete, and a tolerance would hide precisely
the off-by-one this is meant to catch.

Regenerate with::

    python scripts/speechlm2/cpwer_devtools.py fixture --out tests/collections/speechlm2/fixtures

Regenerating to make a failure go away defeats the point. A failure means the scoring path changed;
confirm the new behaviour is the intended one, and only then regenerate.
"""

import json
from pathlib import Path

import pytest

from nemo.collections.speechlm2.parts.metrics.cpwer_report import REFERENCE_AXES, cpwer_metrics_dict
from nemo.collections.speechlm2.parts.metrics.cpwer_scoring import CpWERScoringConfig, score_rows

FIXTURES = Path(__file__).parent / "fixtures"
_COUNT_KEYS = ("cpwer_errors", "cpwer_ref_words", "cpwer_insertions", "cpwer_deletions", "cpwer_substitutions")


def _golden():
    with open(FIXTURES / "cpwer_golden.jsonl") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _rescore():
    rows = [
        {k: r[k] for k in ("id", "text_raw", "pred_text_raw") if k in r}
        | ({"subset_for_metrics": r["subset_for_metrics"]} if "subset_for_metrics" in r else {})
        for r in _golden()
    ]
    cfg = CpWERScoringConfig(**REFERENCE_AXES)
    return rows, cfg, score_rows(rows, cfg)


@pytest.mark.unit
def test_every_golden_row_rescores_to_its_recorded_counts():
    golden = _golden()
    _, _, (per_row, _, _) = _rescore()
    assert len(per_row) == len(golden)
    for expected, actual in zip(golden, per_row):
        for key in _COUNT_KEYS:
            assert (key in expected) == (key in actual), f"{expected['id']}: {key} presence changed"
            if key in expected:
                assert actual[key] == expected[key], f"{expected['id']}: {key}"


@pytest.mark.unit
def test_abstained_rows_omit_the_counts_and_admitted_rows_carry_all_five():
    """Key ABSENCE is the marker. A fixture of only scorable rows would never exercise it."""
    golden = _golden()
    abstained = [r for r in golden if "cpwer_errors" not in r]
    admitted = [r for r in golden if "cpwer_errors" in r]
    assert abstained, "the fixture must contain abstained rows or it cannot pin their shape"
    assert admitted
    for row in abstained:
        assert row["cpwer"] is None
        for key in _COUNT_KEYS:
            assert key not in row
    for row in admitted:
        assert all(key in row for key in _COUNT_KEYS)
        assert row["cpwer_errors"] == row["cpwer_insertions"] + row["cpwer_deletions"] + row["cpwer_substitutions"]


@pytest.mark.unit
def test_corpus_metrics_match_the_recorded_block():
    with open(FIXTURES / "cpwer_golden_metrics.json") as handle:
        expected = json.load(handle)
    _, cfg, (_, corpus, subsets) = _rescore()
    actual = cpwer_metrics_dict(corpus, subsets, cfg)
    for key in ("cpwer", "cpwer_macro", "cpwer_errors", "cpwer_ref_words", "cpwer_admitted", "cpwer_abstained"):
        assert actual[key] == expected[key], key


@pytest.mark.unit
def test_rescoring_reproduces_the_golden_manifest_byte_for_byte():
    """Settings added later, at their defaults, must not move one byte of an existing scored manifest.

    The rows are written as the fixture tool writes them. Every value the metrics file records is reproduced too,
    the ``cpwer_axes`` stamp included; it predates ``cpwer_by_num_speakers``, so that file is compared by value.
    """
    rows, cfg, (per_row, corpus, subsets) = _rescore()
    written = "".join(json.dumps({**row, **scored}, sort_keys=True) + "\n" for row, scored in zip(rows, per_row))
    assert written == (FIXTURES / "cpwer_golden.jsonl").read_text()
    with open(FIXTURES / "cpwer_golden_metrics.json") as handle:
        recorded = json.load(handle)
    actual = cpwer_metrics_dict(corpus, subsets, cfg)
    assert {key: actual[key] for key in recorded} == recorded


@pytest.mark.unit
def test_the_fixture_was_generated_at_the_reference_corner():
    """Otherwise it would pin whatever the defaults happened to be, which is a weaker claim."""
    with open(FIXTURES / "cpwer_golden_metrics.json") as handle:
        metrics = json.load(handle)
    assert metrics["cpwer_axes"]["verdict"].startswith("yes")


@pytest.mark.unit
def test_provenance_pins_the_normalizer_tables_the_scores_depend_on():
    """A table edit changes every score here; the hashes make that a named failure, not a mystery."""
    import hashlib

    from nemo.collections.asr.parts.utils.chime8_spelling_data import ENGLISH_SPELLING, PRE_ENGLISH_SPELLING

    with open(FIXTURES / "cpwer_golden_provenance.json") as handle:
        provenance = json.load(handle)

    def table_hash(table):
        return hashlib.sha256(json.dumps(table, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    assert provenance["chime8_english_spelling_sha256"] == table_hash(ENGLISH_SPELLING)
    assert provenance["chime8_pre_english_spelling_sha256"] == table_hash(PRE_ENGLISH_SPELLING)


@pytest.mark.unit
def test_fixtures_carry_no_internal_references():
    """These ship in a public repository."""
    for path in FIXTURES.glob("cpwer_golden*"):
        text = path.read_text().lower()
        for marker in ("gitlab", "internal", "speechlm-2026h1"):
            assert marker not in text, f"{path.name} mentions {marker!r}"


@pytest.mark.unit
def test_defaults_still_reproduce_the_pre_series_snapshot():
    """The feature-preservation gate: today's defaults must score exactly as the code did before.

    The snapshot was produced by the pre-series implementation at the immutable base commit, so it
    cannot drift with the working tree. Every axis added since defaults to the behaviour recorded
    here, and this is what proves that claim rather than asserting it.
    """
    with open(FIXTURES / "legacy_streaming_snapshot.jsonl") as handle:
        snapshot = [json.loads(line) for line in handle if line.strip()]
    rows = [{k: r[k] for k in ("id", "text_raw", "pred_text_raw")} for r in _golden()]
    per_row, _, _ = score_rows(rows, CpWERScoringConfig(use_normalizer="whisper"))
    assert len(per_row) == len(snapshot)
    for expected, actual in zip(snapshot, per_row):
        for key in _COUNT_KEYS:
            assert actual.get(key) == expected[key], f"{expected['id']}: {key}"
        assert actual["cpwer"] == expected["cpwer"], expected["id"]
