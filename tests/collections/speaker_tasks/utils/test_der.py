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

"""
Tests for DER calculation in nemo.collections.asr.metrics.der and
nemo.collections.asr.metrics.md_eval. cpWER coverage lives in
test_cpwer.py for nemo.collections.asr.metrics.cpwer.

All expected values are pre-verified against an external annotation
library (3.x, the historical NeMo dependency that exposed
``Annotation`` / ``Segment`` / ``Timeline`` and a reference
``DiarizationErrorRate``). The values are hardcoded here so that
**this file does not import any external annotation library**.

md-eval (NIST md-eval-22.pl) and the external reference engine share the
same DER semantics (optimal speaker mapping via the Hungarian algorithm,
same collar / overlap conventions) and produce identical results when
the UEM is equivalent. The only behavioural difference captured by these
tests is that md-eval derives the evaluation region (UEM) from the
*reference* extent whereas the external engine uses the *union* of
reference and hypothesis extents. Tests for that case use an explicit
UEM to keep the two engines aligned.

The :class:`TestLhotseAnnotation` group additionally covers the
lhotse-based replacement for the external annotation library types
(``Annotation`` / ``Segment`` / ``Timeline``) introduced in
:mod:`nemo.collections.asr.metrics.der`. Annotations are built as lists
of :class:`lhotse.SupervisionSegment` and must produce bit-identical DER
to the legacy label-string path.
"""

import io

import pytest
from lhotse import SupervisionSegment, SupervisionSet

from nemo.collections.asr.metrics.der import (
    make_diar_annotation,
    make_diar_segment,
    make_uem_timeline,
    score_labels,
    score_labels_from_rttm_labels,
    unique_speakers,
    write_supervisions_to_rttm,
)
from nemo.collections.asr.metrics.md_eval import (
    EPSILON,
    DiarizationErrorResult,
    _iter_annotation_segments,
    _labels_to_rttm_data,
    _merge_rttm_dicts,
    _merge_uem_dicts,
    _uem_list_to_uem_data,
    evaluate,
)

# ─── Helpers ──────────────────────────────────────────────────────────────


def _seg(start: float, end: float, spk: str) -> str:
    """Create a ``"start end speaker"`` label string."""
    return f"{start} {end} {spk}"


def _labels(*segments):
    """Convert ``(start, end, speaker)`` tuples to label strings."""
    return [_seg(s, e, k) for s, e, k in segments]


def _score(
    ref_segs,
    hyp_segs,
    collar=0.0,
    ignore_overlap=False,
    uem_segs=None,
    file_id="file1",
):
    """Score a single file through the public ``score_labels_from_rttm_labels`` API."""
    ref_labels = _labels(*ref_segs)
    hyp_labels = _labels(*hyp_segs)
    ref_list = [(file_id, ref_labels)]
    hyp_list = [(file_id, hyp_labels)]
    uem_list = [(file_id, uem_segs)] if uem_segs else None
    result = score_labels_from_rttm_labels(
        ref_list,
        hyp_list,
        uem_segments_list=uem_list,
        collar=collar,
        ignore_overlap=ignore_overlap,
        verbose=False,
    )
    assert result is not None, "score_labels_from_rttm_labels returned None"
    return result


def _score_raw(
    ref_segs,
    hyp_segs,
    collar=0.0,
    ignore_overlap=False,
    uem_segs=None,
    file_id="file1",
):
    """Score a single file through the low-level ``evaluate`` API in md_eval."""
    ref_labels = _labels(*ref_segs)
    hyp_labels = _labels(*hyp_segs)
    ref_data = _merge_rttm_dicts([_labels_to_rttm_data(file_id, ref_labels)])
    sys_data = _merge_rttm_dicts([_labels_to_rttm_data(file_id, hyp_labels)])
    uem_data = None
    if uem_segs:
        uem_data = _merge_uem_dicts([_uem_list_to_uem_data(file_id, uem_segs)])
    _, cum = evaluate(
        ref_data,
        sys_data,
        uem_data=uem_data,
        collar=collar,
        opt_1=ignore_overlap,
        verbose=False,
    )
    scored = cum.get("SCORED_SPEAKER", 0.0) or EPSILON
    missed = cum.get("MISSED_SPEAKER", 0.0)
    falarm = cum.get("FALARM_SPEAKER", 0.0)
    error = cum.get("SPEAKER_ERROR", 0.0)
    return {
        "DER": (missed + falarm + error) / scored,
        "CER": error / scored,
        "FA": falarm / scored,
        "MISS": missed / scored,
        "scored": scored,
    }


def assert_der(actual, expected, tol=1e-6):
    diff = abs(actual - expected)
    assert diff <= tol, f"DER mismatch: actual={actual:.8f}, expected={expected:.8f}"


def _score_lhotse(
    ref_segs,
    hyp_segs,
    collar=0.0,
    ignore_overlap=False,
    uem_segs=None,
    file_id="file1",
):
    """Score a single file through ``score_labels`` using lhotse-based annotations.

    Mirrors :func:`_score` but builds the reference and hypothesis as lists of
    ``lhotse.SupervisionSegment`` (via :func:`make_diar_annotation`) instead of
    label strings, exercising the new lhotse-based pipeline end-to-end.
    """
    ref_labels = _labels(*ref_segs)
    hyp_labels = _labels(*hyp_segs)
    ref_ann = make_diar_annotation(ref_labels, uniq_name=file_id)
    hyp_ann = make_diar_annotation(hyp_labels, uniq_name=file_id)
    all_uem = [make_uem_timeline(uem_segs, uniq_id=file_id)] if uem_segs else None
    audio_rttm_map = {file_id: {}}
    result = score_labels(
        audio_rttm_map,
        [(file_id, ref_ann)],
        [(file_id, hyp_ann)],
        all_uem=all_uem,
        collar=collar,
        ignore_overlap=ignore_overlap,
        verbose=False,
    )
    assert result is not None, "score_labels returned None"
    return result


# ─── Tests: md_eval low-level engine ──────────────────────────────────────


class TestMdEvalBasic:
    """Verify the md_eval engine produces correct DER for basic scenarios.

    Expected values verified against the external annotation library's
    reference ``DiarizationErrorRate`` implementation.
    """

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "ref_segs, hyp_segs, expected",
        [
            pytest.param(
                [(0, 5, "A"), (5, 10, "B")],
                [(0, 5, "A"), (5, 10, "B")],
                {"DER": 0.0, "scored": 10.0},
                id="perfect_match",
            ),
            pytest.param(
                [(0, 5, "A"), (5, 10, "B")],
                [],
                {"DER": 1.0, "CER": 0.0, "FA": 0.0, "MISS": 1.0, "scored": 10.0},
                id="complete_miss",
            ),
            pytest.param(
                [(0, 5, "A"), (5, 10, "B")],
                [(0, 5, "B"), (5, 10, "A")],
                {"DER": 0.0, "scored": 10.0},
                id="speaker_swap_optimal_mapping",
            ),
            pytest.param(
                [(0, 10, "A")],
                [(0, 5, "A")],
                {"DER": 0.5, "MISS": 0.5, "scored": 10.0},
                id="partial_miss",
            ),
            pytest.param(
                [(0, 10, "A")],
                [(0, 10, "B")],
                {"DER": 0.0},
                id="single_speaker_confusion",
            ),
            pytest.param(
                [(0, 3, "A"), (7, 10, "B")],
                [(0, 3, "A"), (7, 10, "B")],
                {"DER": 0.0, "scored": 6.0},
                id="gap_perfect",
            ),
            pytest.param(
                [(0, 3, "A"), (7, 10, "B")],
                [(0, 3, "A"), (4, 6, "X"), (7, 10, "B")],
                {"DER": 1 / 3, "FA": 1 / 3, "scored": 6.0},
                id="false_alarm_in_gap",
            ),
            pytest.param(
                [(0, 5, "A")],
                [(0, 10, "A")],
                {"DER": 0.0, "scored": 5.0},
                id="false_alarm_extend_no_uem",
            ),
        ],
    )
    def test_basic_no_uem(self, ref_segs, hyp_segs, expected):
        r = _score_raw(ref_segs, hyp_segs)
        for key, val in expected.items():
            assert_der(r[key], val)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "ref_segs, hyp_segs, uem_segs, expected",
        [
            pytest.param(
                [(0, 5, "A")],
                [(0, 10, "A")],
                [[0, 10]],
                {"DER": 1.0, "FA": 1.0, "scored": 5.0},
                id="false_alarm_extend_with_uem",
            ),
        ],
    )
    def test_basic_with_uem(self, ref_segs, hyp_segs, uem_segs, expected):
        r = _score_raw(ref_segs, hyp_segs, uem_segs=uem_segs)
        for key, val in expected.items():
            assert_der(r[key], val)


class TestMdEvalCollar:
    """Verify collar (no-score zone) handling."""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "ref_segs, hyp_segs, collar, expected",
        [
            pytest.param(
                [(0, 5, "A"), (5, 10, "B")],
                [(0, 5, "A"), (5, 10, "B")],
                0.25,
                {"DER": 0.0, "scored": 9.0},
                id="collar_perfect",
            ),
            pytest.param(
                [(0, 5, "A"), (5, 10, "B")],
                [(0, 5.2, "A"), (5.2, 10, "B")],
                0.25,
                {"DER": 0.0, "scored": 9.0},
                id="collar_absorbs_offset",
            ),
            pytest.param(
                [(0, 5, "A"), (5, 10, "B")],
                [(0, 4.5, "A"), (5.5, 10, "B")],
                0.25,
                {"DER": 0.5 / 9.0, "MISS": 0.5 / 9.0},
                id="collar_boundary_error_within",
            ),
            pytest.param(
                [(0, 5, "A"), (5, 10, "B")],
                [(0, 4, "A"), (6, 10, "B")],
                0.25,
                {"DER": 1.5 / 9.0, "MISS": 1.5 / 9.0},
                id="collar_boundary_error_exceeds",
            ),
            pytest.param(
                [(0, 4, "A"), (4, 7, "B"), (7, 10, "C")],
                [(0, 4, "A"), (4, 7, "B"), (7, 10, "C")],
                0.5,
                {"DER": 0.0, "scored": 7.0},
                id="collar_3spk_perfect",
            ),
        ],
    )
    def test_collar(self, ref_segs, hyp_segs, collar, expected):
        r = _score_raw(ref_segs, hyp_segs, collar=collar)
        for key, val in expected.items():
            assert_der(r[key], val)


class TestMdEvalOverlap:
    """Verify overlap handling with skip_overlap / ignore_overlap."""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "ref_segs, hyp_segs, ignore_overlap, expected",
        [
            pytest.param(
                [(0, 7, "A"), (5, 10, "B")],
                [(0, 7, "A"), (5, 10, "B")],
                True,
                {"DER": 0.0, "scored": 8.0},
                id="overlap_perfect_skip",
            ),
            pytest.param(
                [(0, 7, "A"), (5, 10, "B")],
                [(0, 7, "A"), (5, 10, "B")],
                False,
                {"DER": 0.0, "scored": 12.0},
                id="overlap_perfect_noskip",
            ),
            pytest.param(
                [(0, 7, "A"), (5, 10, "B")],
                [(0, 10, "A")],
                True,
                {"DER": 0.375, "CER": 0.375, "scored": 8.0},
                id="overlap_miss_one_speaker_skip",
            ),
            pytest.param(
                [(0, 7, "A"), (5, 10, "B")],
                [(0, 10, "A")],
                False,
                {"DER": 5 / 12, "CER": 3 / 12, "MISS": 2 / 12, "scored": 12.0},
                id="overlap_miss_one_speaker_noskip",
            ),
        ],
    )
    def test_overlap(self, ref_segs, hyp_segs, ignore_overlap, expected):
        r = _score_raw(ref_segs, hyp_segs, ignore_overlap=ignore_overlap)
        for key, val in expected.items():
            assert_der(r[key], val)


class TestMdEvalSpeakerCount:
    """Verify speaker count mismatch scenarios."""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "ref_segs, hyp_segs, expected",
        [
            pytest.param(
                [(0, 3, "A"), (3, 7, "B"), (7, 10, "C")],
                [(0, 3, "A"), (3, 6, "B"), (6, 10, "C")],
                {"DER": 0.1, "CER": 0.1, "scored": 10.0},
                id="three_speakers_boundary_shift",
            ),
            pytest.param(
                [(0, 5, "A"), (5, 10, "B")],
                [(0, 5, "A"), (5, 8, "B"), (8, 10, "C")],
                {"DER": 0.2, "CER": 0.2},
                id="extra_hyp_speaker",
            ),
            pytest.param(
                [(0, 5, "A"), (5, 8, "B"), (8, 10, "C")],
                [(0, 5, "A"), (5, 10, "B")],
                {"DER": 0.2, "CER": 0.2},
                id="missing_hyp_speaker",
            ),
        ],
    )
    def test_speaker_count(self, ref_segs, hyp_segs, expected):
        r = _score_raw(ref_segs, hyp_segs)
        for key, val in expected.items():
            assert_der(r[key], val)


class TestMdEvalUEM:
    """Verify UEM (Un-partitioned Evaluation Map) handling."""

    @pytest.mark.unit
    def test_uem_restricts_evaluation(self):
        """UEM restricts to [2, 8] out of [0, 10].

        ref: A=[0,10]; hyp: A=[0,5], B=[5,10]. UEM=[2,8].
        Scored region: ref A in [2,8] = 6.0.
        B covers [5,8] of ref A → confusion = 3.0. DER = 3/6 = 0.5.
        """
        r = _score_raw(
            [(0, 10, "A")],
            [(0, 5, "A"), (5, 10, "B")],
            uem_segs=[[2, 8]],
        )
        assert_der(r["DER"], 0.5)
        assert_der(r["CER"], 0.5)
        assert_der(r["scored"], 6.0)


# ─── Tests: der.py public API (score_labels_from_rttm_labels) ────────────


class TestScoreLabelsFromRttmLabels:
    """Test the public ``score_labels_from_rttm_labels`` function in der.py.

    Verifies: return type, DiarizationErrorResult interface, and DER values.
    """

    @pytest.mark.unit
    def test_perfect_match_returns_correct_types(self):
        metric, mapping, (DER, CER, FA, MISS) = _score(
            [(0, 5, "A"), (5, 10, "B")],
            [(0, 5, "A"), (5, 10, "B")],
        )
        assert isinstance(metric, DiarizationErrorResult)
        assert isinstance(mapping, dict)
        assert_der(DER, 0.0)
        assert_der(CER, 0.0)
        assert_der(FA, 0.0)
        assert_der(MISS, 0.0)

    @pytest.mark.unit
    def test_result_abs_interface(self):
        """``abs(metric)`` returns overall DER."""
        metric, _, _ = _score(
            [(0, 5, "A"), (5, 10, "B")],
            [(0, 5, "A"), (5, 10, "B")],
        )
        assert_der(abs(metric), 0.0)

    @pytest.mark.unit
    def test_result_getitem_interface(self):
        """``metric['total']`` etc. return correct values."""
        metric, _, _ = _score(
            [(0, 10, "A")],
            [(0, 5, "A")],
        )
        assert_der(metric["total"], 10.0)
        assert_der(metric["confusion"], 0.0)
        assert_der(metric["false alarm"], 0.0)
        assert_der(metric["missed detection"], 5.0)
        assert_der(abs(metric), 0.5)

    @pytest.mark.unit
    def test_result_optimal_mapping(self):
        """Speaker mapping is accessible via ``metric.optimal_mapping()``."""
        metric, _, _ = _score(
            [(0, 5, "A"), (5, 10, "B")],
            [(0, 5, "B"), (5, 10, "A")],
        )
        file_mapping = metric.optimal_mapping("file1", None)
        assert "A" in file_mapping
        assert file_mapping["A"] == "B"
        assert file_mapping["B"] == "A"

    @pytest.mark.unit
    def test_result_report(self):
        """``metric.report()`` returns a non-empty string."""
        metric, _, _ = _score(
            [(0, 5, "A"), (5, 10, "B")],
            [(0, 5, "A"), (5, 10, "B")],
        )
        report = metric.report()
        assert isinstance(report, str)
        assert len(report) > 0
        assert "file1" in report

    @pytest.mark.unit
    def test_results_list(self):
        """``metric.results_`` contains per-file score dicts."""
        metric, _, _ = _score(
            [(0, 5, "A"), (5, 10, "B")],
            [(0, 5, "A"), (5, 10, "B")],
        )
        assert len(metric.results_) == 1
        file_id, scores = metric.results_[0]
        assert file_id == "file1"
        assert_der(scores["total"], 10.0)
        assert_der(scores["confusion"], 0.0)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "ref_segs, hyp_segs, kwargs, checks",
        [
            pytest.param(
                [(0, 5, "A"), (5, 10, "B")],
                [],
                {},
                {"DER": 1.0, "MISS": 1.0},
                id="complete_miss",
            ),
            pytest.param(
                [(0, 5, "A"), (5, 10, "B")],
                [(0, 5, "B"), (5, 10, "A")],
                {},
                {"DER": 0.0},
                id="speaker_swap",
            ),
            pytest.param(
                [(0, 10, "A")],
                [(0, 5, "A")],
                {},
                {"DER": 0.5, "MISS": 0.5},
                id="partial_miss",
            ),
            pytest.param(
                [(0, 5, "A"), (5, 10, "B")],
                [(0, 5, "A"), (5, 10, "B")],
                {"collar": 0.25},
                {"DER": 0.0},
                id="collar",
            ),
            pytest.param(
                [(0, 5, "A"), (5, 10, "B")],
                [(0, 5.2, "A"), (5.2, 10, "B")],
                {"collar": 0.25},
                {"DER": 0.0},
                id="collar_offset",
            ),
            pytest.param(
                [(0, 7, "A"), (5, 10, "B")],
                [(0, 7, "A"), (5, 10, "B")],
                {"ignore_overlap": True},
                {"DER": 0.0},
                id="overlap_skip",
            ),
            pytest.param(
                [(0, 7, "A"), (5, 10, "B")],
                [(0, 10, "A")],
                {"ignore_overlap": True},
                {"DER": 0.375, "CER": 0.375},
                id="overlap_miss_skip",
            ),
            pytest.param(
                [(0, 7, "A"), (5, 10, "B")],
                [(0, 10, "A")],
                {"ignore_overlap": False},
                {"DER": 5 / 12, "CER": 3 / 12, "MISS": 2 / 12},
                id="overlap_miss_noskip",
            ),
            pytest.param(
                [(0, 3, "A"), (3, 7, "B"), (7, 10, "C")],
                [(0, 3, "A"), (3, 6, "B"), (6, 10, "C")],
                {},
                {"DER": 0.1},
                id="three_speakers",
            ),
            pytest.param(
                [(0, 5, "A"), (5, 10, "B")],
                [(0, 5, "A"), (5, 8, "B"), (8, 10, "C")],
                {},
                {"DER": 0.2, "CER": 0.2},
                id="extra_hyp_speaker",
            ),
            pytest.param(
                [(0, 5, "A"), (5, 8, "B"), (8, 10, "C")],
                [(0, 5, "A"), (5, 10, "B")],
                {},
                {"DER": 0.2, "CER": 0.2},
                id="missing_hyp_speaker",
            ),
            pytest.param(
                [(0, 3, "A"), (7, 10, "B")],
                [(0, 3, "A"), (4, 6, "X"), (7, 10, "B")],
                {},
                {"DER": 1 / 3, "FA": 1 / 3},
                id="false_alarm_in_gap",
            ),
            pytest.param(
                [(0, 10, "A")],
                [(0, 5, "A"), (5, 10, "B")],
                {"uem_segs": [[2, 8]]},
                {"DER": 0.5, "CER": 0.5},
                id="uem_restrict",
            ),
        ],
    )
    def test_der_values(self, ref_segs, hyp_segs, kwargs, checks):
        _, _, (DER, CER, FA, MISS) = _score(ref_segs, hyp_segs, **kwargs)
        name_to_val = {"DER": DER, "CER": CER, "FA": FA, "MISS": MISS}
        for key, val in checks.items():
            assert_der(name_to_val[key], val)

    @pytest.mark.unit
    def test_length_mismatch_returns_none(self):
        """Mismatched ref/hyp list lengths should return None."""
        result = score_labels_from_rttm_labels(
            [("f1", _labels((0, 5, "A")))],
            [("f1", _labels((0, 5, "A"))), ("f2", _labels((0, 5, "B")))],
            verbose=False,
        )
        assert result is None


# ─── Tests: Multi-file scoring ───────────────────────────────────────────


class TestMultiFile:
    """Verify multi-file cumulative scoring."""

    @pytest.mark.unit
    def test_two_files_one_perfect_one_confusion(self):
        """File1: perfect. File2: all confusion (mapped away).

        Combined: scored=10, DER=0 (optimal mapping maps C→B).
        """
        ref_list = [
            ("file1", _labels((0, 5, "A"))),
            ("file2", _labels((0, 5, "B"))),
        ]
        hyp_list = [
            ("file1", _labels((0, 5, "A"))),
            ("file2", _labels((0, 5, "C"))),
        ]
        result = score_labels_from_rttm_labels(
            ref_list,
            hyp_list,
            collar=0.0,
            ignore_overlap=False,
            verbose=False,
        )
        assert result is not None
        metric, _, (DER, _, _, _) = result
        assert_der(DER, 0.0)
        assert_der(metric["total"], 10.0)
        assert len(metric.results_) == 2

    @pytest.mark.unit
    def test_two_files_one_miss(self):
        """File1: perfect 5s. File2: complete miss 5s.

        Combined: scored=10, missed=5, DER=0.5.
        """
        ref_list = [
            ("file1", _labels((0, 5, "A"))),
            ("file2", _labels((0, 5, "B"))),
        ]
        hyp_list = [
            ("file1", _labels((0, 5, "A"))),
            ("file2", []),
        ]
        result = score_labels_from_rttm_labels(
            ref_list,
            hyp_list,
            collar=0.0,
            ignore_overlap=False,
            verbose=False,
        )
        assert result is not None
        _, _, (DER, _, _, MISS) = result
        assert_der(DER, 0.5)
        assert_der(MISS, 0.5)


# ─── Tests: External-engine-verified values (cross-validated) ────────────


class TestExternalEngineVerifiedValues:
    """Cross-validation against an external annotation library.

    All expected values in this class have been computed with the external
    annotation library's reference ``DiarizationErrorRate`` (3.x), using
    its ``collar=2*collar_value`` convention, and hardcoded here.

    This class does **not** import the external library; it only checks
    that the md-eval engine reproduces the same numbers.
    """

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "ref_segs, hyp_segs, collar, ignore_overlap, uem_segs, expected",
        [
            pytest.param(
                [(0, 5, "A"), (5, 10, "B")],
                [(0, 5, "A"), (5, 10, "B")],
                0.0,
                False,
                None,
                {"DER": 0.0},
                id="external_perfect",
            ),
            pytest.param(
                [(0, 5, "A"), (5, 10, "B")],
                [],
                0.0,
                False,
                None,
                {"DER": 1.0, "MISS": 1.0},
                id="external_complete_miss",
            ),
            pytest.param(
                [(0, 5, "A"), (5, 10, "B")],
                [(0, 5, "B"), (5, 10, "A")],
                0.0,
                False,
                None,
                {"DER": 0.0},
                id="external_swap",
            ),
            pytest.param(
                [(0, 10, "A")],
                [(0, 5, "A")],
                0.0,
                False,
                None,
                {"DER": 0.5},
                id="external_partial_miss",
            ),
            pytest.param(
                [(0, 5, "A"), (5, 10, "B")],
                [(0, 5, "A"), (5, 10, "B")],
                0.25,
                False,
                None,
                {"DER": 0.0, "scored": 9.0},
                id="external_collar_perfect",
            ),
            pytest.param(
                [(0, 5, "A"), (5, 10, "B")],
                [(0, 5.2, "A"), (5.2, 10, "B")],
                0.25,
                False,
                None,
                {"DER": 0.0},
                id="external_collar_offset",
            ),
            pytest.param(
                [(0, 7, "A"), (5, 10, "B")],
                [(0, 7, "A"), (5, 10, "B")],
                0.0,
                True,
                None,
                {"DER": 0.0, "scored": 8.0},
                id="external_overlap_skip_perfect",
            ),
            pytest.param(
                [(0, 7, "A"), (5, 10, "B")],
                [(0, 7, "A"), (5, 10, "B")],
                0.0,
                False,
                None,
                {"DER": 0.0, "scored": 12.0},
                id="external_overlap_noskip_perfect",
            ),
            pytest.param(
                [(0, 7, "A"), (5, 10, "B")],
                [(0, 10, "A")],
                0.0,
                True,
                None,
                {"DER": 0.375, "CER": 0.375},
                id="external_overlap_miss_skip",
            ),
            pytest.param(
                [(0, 7, "A"), (5, 10, "B")],
                [(0, 10, "A")],
                0.0,
                False,
                None,
                {"DER": 5 / 12, "CER": 3 / 12, "MISS": 2 / 12, "scored": 12.0},
                id="external_overlap_miss_noskip",
            ),
            pytest.param(
                [(0, 3, "A"), (3, 7, "B"), (7, 10, "C")],
                [(0, 3, "A"), (3, 6, "B"), (6, 10, "C")],
                0.0,
                False,
                None,
                {"DER": 0.1, "CER": 0.1},
                id="external_3spk_boundary",
            ),
            pytest.param(
                [(0, 5, "A"), (5, 10, "B")],
                [(0, 5, "A"), (5, 8, "B"), (8, 10, "C")],
                0.0,
                False,
                None,
                {"DER": 0.2, "CER": 0.2},
                id="external_extra_hyp",
            ),
            pytest.param(
                [(0, 5, "A"), (5, 8, "B"), (8, 10, "C")],
                [(0, 5, "A"), (5, 10, "B")],
                0.0,
                False,
                None,
                {"DER": 0.2, "CER": 0.2},
                id="external_missing_hyp",
            ),
            pytest.param(
                [(0, 3, "A"), (7, 10, "B")],
                [(0, 3, "A"), (7, 10, "B")],
                0.0,
                False,
                None,
                {"DER": 0.0, "scored": 6.0},
                id="external_gap",
            ),
            pytest.param(
                [(0, 3, "A"), (7, 10, "B")],
                [(0, 3, "A"), (4, 6, "X"), (7, 10, "B")],
                0.0,
                False,
                None,
                {"DER": 1 / 3, "FA": 1 / 3},
                id="external_false_alarm_in_gap",
            ),
            pytest.param(
                [(0, 10, "A")],
                [(0, 5, "A"), (5, 10, "B")],
                0.0,
                False,
                [[2, 8]],
                {"DER": 0.5, "CER": 0.5, "scored": 6.0},
                id="external_uem",
            ),
            pytest.param(
                [(0, 4, "A"), (4, 7, "B"), (7, 10, "C")],
                [(0, 4, "A"), (4, 7, "B"), (7, 10, "C")],
                0.5,
                False,
                None,
                {"DER": 0.0, "scored": 7.0},
                id="external_collar_3spk",
            ),
            pytest.param(
                [(0, 5, "A"), (5, 10, "B")],
                [(0, 4.5, "A"), (5.5, 10, "B")],
                0.25,
                False,
                None,
                {"DER": 0.5 / 9.0, "MISS": 0.5 / 9.0},
                id="external_collar_boundary_error",
            ),
            pytest.param(
                [(0, 5, "A"), (5, 10, "B")],
                [(0, 4, "A"), (6, 10, "B")],
                0.25,
                False,
                None,
                {"DER": 1.5 / 9.0, "MISS": 1.5 / 9.0},
                id="external_collar_boundary_error_large",
            ),
            pytest.param(
                [(0, 10, "A")],
                [(0, 10, "B")],
                0.0,
                False,
                None,
                {"DER": 0.0},
                id="external_single_speaker_confusion",
            ),
        ],
    )
    def test_external_values(self, ref_segs, hyp_segs, collar, ignore_overlap, uem_segs, expected):
        r = _score_raw(ref_segs, hyp_segs, collar=collar, ignore_overlap=ignore_overlap, uem_segs=uem_segs)
        for key, val in expected.items():
            assert_der(r[key], val)

    @pytest.mark.unit
    def test_external_multi_file(self):
        """External engine, multi-file: file1 perfect + file2 relabelled → DER=0.

        Both engines map C→B via Hungarian algorithm.
        """
        ref_dicts = [
            _labels_to_rttm_data("file1", _labels((0, 5, "A"))),
            _labels_to_rttm_data("file2", _labels((0, 5, "B"))),
        ]
        sys_dicts = [
            _labels_to_rttm_data("file1", _labels((0, 5, "A"))),
            _labels_to_rttm_data("file2", _labels((0, 5, "C"))),
        ]
        ref_data = _merge_rttm_dicts(ref_dicts)
        sys_data = _merge_rttm_dicts(sys_dicts)
        _, cum = evaluate(ref_data, sys_data, uem_data=None, collar=0.0, opt_1=False, verbose=False)
        scored = cum.get("SCORED_SPEAKER", 0.0) or EPSILON
        DER = (
            cum.get("MISSED_SPEAKER", 0.0) + cum.get("FALARM_SPEAKER", 0.0) + cum.get("SPEAKER_ERROR", 0.0)
        ) / scored
        assert_der(DER, 0.0)
        assert_der(scored, 10.0)


# ─── Tests: regression for no-UEM scoring (parity with external lib) ─────


class TestNoUemAutoUnion:
    """Regression tests for the auto-derived UEM used when no UEM is provided.

    Historically NeMo's DER was computed via an external annotation library
    that, in the no-UEM path, built its scoring map from the union of the
    reference and system extents. NIST ``md-eval-22.pl`` (which our
    :func:`md_eval.evaluate` faithfully ports) instead defaults to the
    reference extent only. The high-level wrappers in :mod:`der` bridge the
    two by auto-deriving a ``ref ∪ sys`` UEM whenever the caller does not
    supply one. These tests pin down that behaviour with hardcoded values
    independently verified by hand and previously by the external library.
    """

    # Sortformer Diar 4spk-v1 dihard3-dev tutorial sample.
    _REF = [(0.299, 2.770, "A"), (3.164, 5.147, "B")]
    _HYP_RAW = [(0.400, 2.880, "spk0"), (3.200, 5.190, "spk1")]
    _HYP_PP = [(0.340, 2.800, "spk0"), (3.220, 5.190, "spk1")]

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "hyp_attr, expected_der, expected_fa_num, expected_fa_den, expected_miss_num, expected_miss_den",
        [
            pytest.param(
                "_HYP_RAW",
                0.065110,
                0.153,
                4.454,
                0.137,
                4.454,
                id="raw_binarization_matches_external_lib",
            ),
            pytest.param(
                "_HYP_PP",
                0.038168,
                0.073,
                4.454,
                0.097,
                4.454,
                id="post_processed_matches_external_lib",
            ),
        ],
    )
    def test_score_matches_external_lib(
        self, hyp_attr, expected_der, expected_fa_num, expected_fa_den, expected_miss_num, expected_miss_den
    ):
        hyp = getattr(self, hyp_attr)
        r = _score(self._REF, hyp, collar=0.0, ignore_overlap=False)
        DER, _CER, FA, MISS = r[2]
        assert_der(DER, expected_der, tol=1e-5)
        assert_der(FA, expected_fa_num / expected_fa_den, tol=1e-5)
        assert_der(MISS, expected_miss_num / expected_miss_den, tol=1e-5)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "hyp_attr, expected_der",
        [
            pytest.param("_HYP_RAW", 0.065110, id="lhotse_raw"),
            pytest.param("_HYP_PP", 0.038168, id="lhotse_post"),
        ],
    )
    def test_score_labels_lhotse_path_matches_external_lib(self, hyp_attr, expected_der):
        hyp = getattr(self, hyp_attr)
        r = _score_lhotse(self._REF, hyp, collar=0.0, ignore_overlap=False)
        DER, _CER, _FA, _MISS = r[2]
        assert_der(DER, expected_der, tol=1e-5)

    @pytest.mark.unit
    def test_low_level_evaluate_keeps_nist_semantics(self):
        """The low-level ``evaluate`` API must keep the NIST ref-extent default.

        Power users that call ``md_eval.evaluate`` directly should still see
        the strict NIST behaviour (eval map = ref extent only) when they pass
        ``uem_data=None``. The auto-union behaviour is intentionally limited
        to the high-level wrappers in :mod:`der`.
        """
        r = _score_raw(self._REF, self._HYP_RAW, collar=0.0, ignore_overlap=False)
        assert_der(r["DER"], 0.055456, tol=1e-5)
        r = _score_raw(self._REF, self._HYP_PP, collar=0.0, ignore_overlap=False)
        assert_der(r["DER"], 0.028514, tol=1e-5)

    @pytest.mark.unit
    def test_explicit_uem_overrides_auto_union(self):
        """An explicit UEM must always take precedence over the auto-derived one."""
        r = _score(
            self._REF,
            self._HYP_RAW,
            collar=0.0,
            ignore_overlap=False,
            uem_segs=[[0.299, 5.147]],
        )
        DER, _CER, _FA, _MISS = r[2]
        assert_der(DER, 0.055456, tol=1e-5)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "collar, hyp_attr, expected_der",
        [
            pytest.param(0.05, "_HYP_RAW", 0.026093, id="collar_nist_half_width_raw"),
            pytest.param(0.05, "_HYP_PP", 0.001410, id="collar_nist_half_width_post"),
        ],
    )
    def test_collar_is_nist_half_width(self, collar, hyp_attr, expected_der):
        hyp = getattr(self, hyp_attr)
        r = _score(self._REF, hyp, collar=collar, ignore_overlap=False)
        DER, _CER, _FA, _MISS = r[2]
        assert_der(DER, expected_der, tol=1e-5)

    @pytest.mark.unit
    def test_collar_2x_equivalence_to_external_lib(self):
        """Cross-engine equivalence: NeMo ``collar=X`` ≡ external lib ``collar=2X``.

        The external library reports RAW DER = 0.043638 / POST DER = 0.016077
        when called directly with ``collar=0.10``. NeMo at ``collar=0.05``
        must produce the same numbers — the historical doubling-then-halving
        round trip, made explicit by passing ``collar`` straight through to
        :func:`md_eval.evaluate` (which uses NIST half-width semantics
        natively). Equivalently, NeMo at ``collar=0.025`` must match the
        external lib at ``collar=0.05``.
        """
        r = _score(self._REF, self._HYP_RAW, collar=0.025, ignore_overlap=False)
        DER, _CER, _FA, _MISS = r[2]
        assert_der(DER, 0.043638, tol=1e-5)
        r = _score(self._REF, self._HYP_PP, collar=0.025, ignore_overlap=False)
        DER, _CER, _FA, _MISS = r[2]
        assert_der(DER, 0.016077, tol=1e-5)

    @pytest.mark.unit
    def test_collar_lhotse_path_matches_string_path(self):
        """The lhotse-backed ``score_labels`` collar semantics must agree with ``score_labels_from_rttm_labels``."""
        for collar, expected_raw, expected_post in [
            (0.05, 0.026093, 0.001410),
            (0.025, 0.043638, 0.016077),
        ]:
            r_raw = _score_lhotse(self._REF, self._HYP_RAW, collar=collar, ignore_overlap=False)
            assert_der(r_raw[2][0], expected_raw, tol=1e-5)
            r_post = _score_lhotse(self._REF, self._HYP_PP, collar=collar, ignore_overlap=False)
            assert_der(r_post[2][0], expected_post, tol=1e-5)

    @pytest.mark.unit
    def test_default_uem_helper_builds_union(self):
        """The internal ``_default_uem_from_ref_sys`` builds the right span."""
        from nemo.collections.asr.metrics.der import _default_uem_from_ref_sys

        ref_data = _merge_rttm_dicts([_labels_to_rttm_data("file1", _labels(*self._REF))])
        sys_data = _merge_rttm_dicts([_labels_to_rttm_data("file1", _labels(*self._HYP_RAW))])
        uem = _default_uem_from_ref_sys(ref_data, sys_data)
        assert "file1" in uem
        seg = uem["file1"]["1"][0]
        assert abs(seg["TBEG"] - 0.299) < 1e-9
        assert abs(seg["TEND"] - 5.190) < 1e-9


# ─── Tests: lhotse-based replacement for the external annotation lib ─────


class TestLhotseShimHelpers:
    """Unit tests for the lhotse-based shim helpers in der.py.

    These helpers (``make_diar_segment``, ``make_diar_annotation``,
    ``make_uem_timeline``, ``unique_speakers``, ``write_supervisions_to_rttm``)
    replace the ``Annotation`` / ``Segment`` / ``Timeline`` types from the
    external annotation library that NeMo previously depended on.
    """

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "start, end, spk, rec_id, expected_duration, expected_end",
        [
            pytest.param(1.5, 4.0, "spk0", "rec1", 2.5, 4.0, id="basic"),
            pytest.param(5.0, 5.0, "A", None, 0.0, 5.0, id="zero_duration"),
            pytest.param(5.0, 4.0, "A", None, 0.0, 5.0, id="inverted_clamped"),
        ],
    )
    def test_make_diar_segment(self, start, end, spk, rec_id, expected_duration, expected_end):
        kwargs = {"recording_id": rec_id} if rec_id is not None else {}
        seg = make_diar_segment(start, end, spk, **kwargs)
        assert isinstance(seg, SupervisionSegment)
        assert seg.start == start
        assert seg.duration == expected_duration
        assert seg.speaker == spk
        if rec_id is not None:
            assert seg.recording_id == rec_id

    @pytest.mark.unit
    def test_make_diar_segment_auto_id(self):
        """When ``segment_id`` is None, a deterministic id is generated."""
        s1 = make_diar_segment(0.0, 1.0, "A", recording_id="r")
        s2 = make_diar_segment(0.0, 1.0, "A", recording_id="r")
        assert s1.id == s2.id
        s3 = make_diar_segment(0.0, 2.0, "A", recording_id="r")
        assert s1.id != s3.id

    @pytest.mark.unit
    def test_make_diar_annotation_from_labels(self):
        labels = ["0.0 5.0 A", "5.0 10.0 B", "10.0 12.5 A"]
        ann = make_diar_annotation(labels, uniq_name="rec42")
        assert isinstance(ann, list)
        assert len(ann) == 3
        assert all(isinstance(s, SupervisionSegment) for s in ann)
        assert all(s.recording_id == "rec42" for s in ann)
        assert [s.speaker for s in ann] == ["A", "B", "A"]
        assert [s.start for s in ann] == [0.0, 5.0, 10.0]
        assert [s.end for s in ann] == [5.0, 10.0, 12.5]

    @pytest.mark.unit
    def test_make_diar_annotation_skips_malformed(self):
        """Lines with fewer than 3 tokens are ignored (defensive)."""
        labels = ["0.0 5.0 A", "garbage", "", "5.0 10.0 B"]
        ann = make_diar_annotation(labels, uniq_name="r")
        assert len(ann) == 2
        assert [s.speaker for s in ann] == ["A", "B"]

    @pytest.mark.unit
    def test_make_uem_timeline_basic(self):
        uem = make_uem_timeline([[0.0, 5.0], [10.0, 12.0]], uniq_id="rec1")
        assert len(uem) == 2
        assert all(isinstance(s, SupervisionSegment) for s in uem)
        assert all(s.speaker == "UEM" for s in uem)
        assert all(s.recording_id == "rec1" for s in uem)
        assert (uem[0].start, uem[0].end) == (0.0, 5.0)
        assert (uem[1].start, uem[1].end) == (10.0, 12.0)

    @pytest.mark.unit
    def test_make_uem_timeline_empty(self):
        assert make_uem_timeline([], uniq_id="r") == []

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "labels, expected",
        [
            pytest.param(
                ["0 1 B", "1 2 A", "2 3 B", "3 4 C", "4 5 A"],
                ["B", "A", "C"],
                id="preserves_first_seen_order",
            ),
            pytest.param(
                [],
                [],
                id="empty",
            ),
        ],
    )
    def test_unique_speakers(self, labels, expected):
        if labels:
            ann = make_diar_annotation(labels, uniq_name="r")
        else:
            ann = []
        result = unique_speakers(ann)
        assert result == expected

    @pytest.mark.unit
    def test_unique_speakers_on_supervision_set(self):
        ann = make_diar_annotation(["0 1 A", "1 2 B"], uniq_name="r")
        ss = SupervisionSet.from_segments(ann)
        assert sorted(unique_speakers(ss)) == ["A", "B"]

    @pytest.mark.unit
    def test_write_supervisions_to_rttm_format(self):
        ann = make_diar_annotation(["0.0 1.5 A", "1.5 3.0 B"], uniq_name="rec1")
        buf = io.StringIO()
        write_supervisions_to_rttm(ann, buf)
        lines = [ln for ln in buf.getvalue().splitlines() if ln.strip()]
        assert len(lines) == 2
        for ln in lines:
            parts = ln.split()
            assert parts[0] == "SPEAKER"
            assert parts[1] == "rec1"
            assert parts[2] == "1"
            assert parts[5] == "<NA>" and parts[6] == "<NA>"
            assert parts[8] == "<NA>" and parts[9] == "<NA>"
        p0 = lines[0].split()
        assert float(p0[3]) == pytest.approx(0.0)
        assert float(p0[4]) == pytest.approx(1.5)
        assert p0[7] == "A"

    @pytest.mark.unit
    def test_write_supervisions_to_rttm_skips_zero_duration(self):
        ann = [
            make_diar_segment(0.0, 1.0, "A", recording_id="rec1"),
            make_diar_segment(2.0, 2.0, "B", recording_id="rec1"),  # zero-duration
            make_diar_segment(3.0, 4.5, "C", recording_id="rec1"),
        ]
        buf = io.StringIO()
        write_supervisions_to_rttm(ann, buf)
        lines = [ln for ln in buf.getvalue().splitlines() if ln.strip()]
        assert len(lines) == 2
        speakers = [ln.split()[7] for ln in lines]
        assert speakers == ["A", "C"]

    @pytest.mark.unit
    def test_write_supervisions_to_rttm_explicit_recording_id_override(self):
        """Explicit ``recording_id`` overrides per-segment ids."""
        ann = make_diar_annotation(["0 1 A"], uniq_name="orig")
        buf = io.StringIO()
        write_supervisions_to_rttm(ann, buf, recording_id="overridden")
        line = buf.getvalue().strip()
        assert line.split()[1] == "overridden"

    @pytest.mark.unit
    def test_write_supervisions_to_rttm_round_trip(self):
        """Write annotations to RTTM, then read them back via lhotse.

        Verifies our RTTM output is parseable by lhotse's RTTM reader,
        confirming we follow the same format conventions.
        """
        ann = make_diar_annotation(["0.0 2.5 alice", "2.5 5.0 bob", "5.0 7.25 alice"], uniq_name="conv1")
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".rttm", delete=False) as fh:
            write_supervisions_to_rttm(ann, fh)
            tmp_path = fh.name
        try:
            parsed = SupervisionSet.from_rttm(tmp_path)
            parsed_segs = sorted(list(parsed), key=lambda s: s.start)
        finally:
            import os

            os.unlink(tmp_path)
        assert len(parsed_segs) == 3
        assert [s.speaker for s in parsed_segs] == ["alice", "bob", "alice"]
        assert [s.start for s in parsed_segs] == pytest.approx([0.0, 2.5, 5.0])
        assert [s.end for s in parsed_segs] == pytest.approx([2.5, 5.0, 7.25])


class TestIterAnnotationSegments:
    """Verify ``md_eval._iter_annotation_segments`` accepts every supported type."""

    @pytest.mark.unit
    def test_iter_list_of_supervision_segments(self):
        ann = make_diar_annotation(["0 1 A", "1 3 B"], uniq_name="r")
        out = list(_iter_annotation_segments(ann))
        assert out == [(0.0, 1.0, "A"), (1.0, 3.0, "B")]

    @pytest.mark.unit
    def test_iter_supervision_set(self):
        ann = make_diar_annotation(["0 1 A", "1 3 B"], uniq_name="r")
        ss = SupervisionSet.from_segments(ann)
        out = list(_iter_annotation_segments(ss))
        assert sorted(out) == [(0.0, 1.0, "A"), (1.0, 3.0, "B")]

    @pytest.mark.unit
    def test_iter_duck_typed_objects_with_end(self):
        """Plain dataclass-like objects with ``.start``, ``.end``, ``.speaker``."""

        class _DT:
            def __init__(self, start, end, speaker):
                self.start = start
                self.end = end
                self.speaker = speaker

        ann = [_DT(0.0, 2.0, "X"), _DT(2.0, 5.0, "Y")]
        assert list(_iter_annotation_segments(ann)) == [(0.0, 2.0, "X"), (2.0, 5.0, "Y")]

    @pytest.mark.unit
    def test_iter_duck_typed_objects_with_duration(self):
        """Objects exposing ``.duration`` (no ``.end``) are also accepted."""

        class _DT:
            def __init__(self, start, duration, speaker):
                self.start = start
                self.duration = duration
                self.speaker = speaker
                self.end = None

        ann = [_DT(0.0, 2.0, "X"), _DT(2.0, 3.0, "Y")]
        assert list(_iter_annotation_segments(ann)) == [(0.0, 2.0, "X"), (2.0, 5.0, "Y")]

    @pytest.mark.unit
    def test_iter_legacy_itertracks_object(self):
        """Objects exposing ``.itertracks(yield_label=True)`` (legacy path).

        This duck-typed fallback keeps backwards compatibility with the
        external annotation library's ``Annotation`` API.
        """

        class _Seg:
            def __init__(self, s, e):
                self.start = s
                self.end = e

        class _Ann:
            def __init__(self, items):
                self._items = items

            def itertracks(self, yield_label=True):
                for s, e, spk in self._items:
                    yield _Seg(s, e), "track", spk

        ann = _Ann([(0.0, 1.5, "A"), (1.5, 4.0, "B")])
        assert list(_iter_annotation_segments(ann)) == [(0.0, 1.5, "A"), (1.5, 4.0, "B")]

    @pytest.mark.unit
    def test_iter_missing_end_and_duration_raises(self):
        class _Bad:
            def __init__(self):
                self.start = 0.0
                self.speaker = "A"

        with pytest.raises(TypeError, match="end.*duration"):
            list(_iter_annotation_segments([_Bad()]))

    @pytest.mark.unit
    def test_iter_missing_speaker_raises(self):
        class _Bad:
            def __init__(self):
                self.start = 0.0
                self.end = 1.0

        with pytest.raises(TypeError, match="speaker"):
            list(_iter_annotation_segments([_Bad()]))


class TestLhotseAnnotation:
    """End-to-end DER tests using lhotse SupervisionSegment annotations.

    Every scenario here is also covered by the legacy label-string tests
    above (``TestScoreLabelsFromRttmLabels``); we re-run them through the
    new lhotse pipeline (``score_labels`` + ``make_diar_annotation`` +
    ``make_uem_timeline``) and assert **bit-identical** DER/CER/FA/MISS.
    Any divergence here means the lhotse adapter has regressed.
    """

    @pytest.mark.unit
    def test_perfect_match(self):
        metric, mapping, (DER, CER, FA, MISS) = _score_lhotse(
            [(0, 5, "A"), (5, 10, "B")],
            [(0, 5, "A"), (5, 10, "B")],
        )
        assert isinstance(metric, DiarizationErrorResult)
        assert isinstance(mapping, dict)
        assert_der(DER, 0.0)
        assert_der(CER, 0.0)
        assert_der(FA, 0.0)
        assert_der(MISS, 0.0)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "ref_segs, hyp_segs, kwargs, checks",
        [
            pytest.param(
                [(0, 5, "A"), (5, 10, "B")],
                [],
                {},
                {"DER": 1.0, "MISS": 1.0},
                id="complete_miss",
            ),
            pytest.param(
                [(0, 5, "A"), (5, 10, "B")],
                [(0, 5, "B"), (5, 10, "A")],
                {},
                {"DER": 0.0},
                id="speaker_swap_optimal_mapping",
            ),
            pytest.param(
                [(0, 10, "A")],
                [(0, 5, "A")],
                {},
                {"DER": 0.5, "MISS": 0.5},
                id="partial_miss",
            ),
            pytest.param(
                [(0, 5, "A")],
                [(0, 5, "A"), (5, 10, "A")],
                {"uem_segs": [[0, 10]]},
                {"DER": 1.0, "FA": 1.0},
                id="partial_false_alarm",
            ),
            pytest.param(
                [(0, 5, "A"), (5, 10, "B")],
                [(0, 5, "A"), (5, 10, "A")],
                {},
                {"DER": 0.5, "CER": 0.5},
                id="confusion",
            ),
            pytest.param(
                [(0, 5, "A"), (5, 10, "B")],
                [(0, 4.9, "A"), (5.1, 10, "B")],
                {"collar": 0.25},
                {"DER": 0.0},
                id="collar_eliminates_boundary_error",
            ),
            pytest.param(
                [(0, 5, "A"), (5, 10, "B")],
                [(0, 4, "A"), (6, 10, "B")],
                {"collar": 0.25},
                {"DER": 1.5 / 9.0, "MISS": 1.5 / 9.0},
                id="collar_partial",
            ),
            pytest.param(
                [(0, 10, "A")],
                [(0, 5, "A"), (5, 10, "B")],
                {"uem_segs": [[2, 8]]},
                {"DER": 0.5, "CER": 0.5},
                id="uem_restricts",
            ),
            pytest.param(
                [(0, 5, "A"), (5, 10, "B")],
                [(0, 5, "A"), (5, 8, "B"), (8, 10, "C")],
                {},
                {"DER": 0.2, "CER": 0.2},
                id="extra_hyp_speaker",
            ),
            pytest.param(
                [(0, 5, "A"), (5, 8, "B"), (8, 10, "C")],
                [(0, 5, "A"), (5, 10, "B")],
                {},
                {"DER": 0.2, "CER": 0.2},
                id="missing_hyp_speaker",
            ),
        ],
    )
    def test_der_values(self, ref_segs, hyp_segs, kwargs, checks):
        _, _, (DER, CER, FA, MISS) = _score_lhotse(ref_segs, hyp_segs, **kwargs)
        name_to_val = {"DER": DER, "CER": CER, "FA": FA, "MISS": MISS}
        for key, val in checks.items():
            assert_der(name_to_val[key], val)

    @pytest.mark.unit
    def test_ignore_overlap(self):
        """``ignore_overlap=True`` should suppress overlap-region scoring."""
        _, _, (DER_no, _, _, _) = _score_lhotse(
            [(0, 5, "A"), (3, 7, "B")],
            [(0, 5, "A"), (3, 7, "B")],
            ignore_overlap=False,
        )
        _, _, (DER_yes, _, _, _) = _score_lhotse(
            [(0, 5, "A"), (3, 7, "B")],
            [(0, 5, "A"), (3, 7, "B")],
            ignore_overlap=True,
        )
        assert_der(DER_no, 0.0)
        assert_der(DER_yes, 0.0)

    @pytest.mark.unit
    def test_accepts_supervision_set(self):
        """``score_labels`` should accept a ``SupervisionSet`` directly."""
        ref = SupervisionSet.from_segments(make_diar_annotation(["0 5 A", "5 10 B"], uniq_name="f1"))
        hyp = SupervisionSet.from_segments(make_diar_annotation(["0 5 A", "5 10 B"], uniq_name="f1"))
        result = score_labels(
            {"f1": {}},
            [("f1", ref)],
            [("f1", hyp)],
            collar=0.0,
            ignore_overlap=False,
            verbose=False,
        )
        assert result is not None
        _, _, (DER, _, _, _) = result
        assert_der(DER, 0.0)

    @pytest.mark.unit
    def test_multi_file_scoring(self):
        """Two files, one perfect and one with confusion → averaged DER."""
        f1_ref = make_diar_annotation(["0 5 A"], uniq_name="f1")
        f1_hyp = make_diar_annotation(["0 5 A"], uniq_name="f1")
        f2_ref = make_diar_annotation(["0 4 A", "4 8 B"], uniq_name="f2")
        f2_hyp = make_diar_annotation(["0 8 A"], uniq_name="f2")
        result = score_labels(
            {"f1": {}, "f2": {}},
            [("f1", f1_ref), ("f2", f2_ref)],
            [("f1", f1_hyp), ("f2", f2_hyp)],
            collar=0.0,
            ignore_overlap=False,
            verbose=False,
        )
        assert result is not None
        metric, _, (DER, _, _, _) = result
        assert_der(DER, 4.0 / 13.0)
        assert len(metric.results_) == 2


class TestLhotseStringEquivalence:
    """The lhotse path and the legacy label-string path must agree on every metric.

    Same reference + hypothesis fed through both ``score_labels`` (lhotse
    annotations) and ``score_labels_from_rttm_labels`` (label strings) must
    produce bit-identical (DER, CER, FA, MISS).
    """

    @staticmethod
    def _both(ref_segs, hyp_segs, **kw):
        string_path = _score(ref_segs, hyp_segs, **kw)
        lhotse_path = _score_lhotse(ref_segs, hyp_segs, **kw)
        return string_path[2], lhotse_path[2]  # itemized errors

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "ref_segs, hyp_segs, kwargs",
        [
            pytest.param(
                [(0, 5, "A"), (5, 10, "B")],
                [(0, 5, "A"), (5, 10, "B")],
                {},
                id="perfect",
            ),
            pytest.param(
                [(0, 5, "A"), (5, 10, "B")],
                [],
                {},
                id="complete_miss",
            ),
            pytest.param(
                [(0, 5, "A"), (5, 10, "B")],
                [(0, 5, "B"), (5, 10, "A")],
                {},
                id="speaker_swap",
            ),
            pytest.param(
                [(0, 5, "A"), (5, 10, "B")],
                [(0, 4.9, "A"), (5.1, 10, "B")],
                {"collar": 0.25},
                id="collar",
            ),
            pytest.param(
                [(0, 10, "A")],
                [(0, 5, "A"), (5, 10, "B")],
                {"uem_segs": [[2, 8]]},
                id="uem",
            ),
            pytest.param(
                [(0, 5, "A"), (3, 7, "B")],
                [(0, 4, "A"), (3, 7, "B")],
                {"ignore_overlap": True},
                id="ignore_overlap",
            ),
            pytest.param(
                [(0, 4, "A"), (4, 7, "B"), (7, 10, "C")],
                [(0, 4, "A"), (4, 7, "B"), (7, 10, "C")],
                {"collar": 0.5},
                id="three_speakers",
            ),
            pytest.param(
                [(0, 5, "A"), (5, 10, "B")],
                [(0, 5, "A"), (5, 8, "B"), (8, 10, "C")],
                {},
                id="extra_hyp_speaker",
            ),
        ],
    )
    def test_equivalence(self, ref_segs, hyp_segs, kwargs):
        string_path, lhotse_path = self._both(ref_segs, hyp_segs, **kwargs)
        assert string_path == pytest.approx(lhotse_path)


# ─── Tests: audio_end clipping ────────────────────────────────────────────


class TestAudioEndClipping:
    """Verify that the ``audio_end`` clipping mechanism prevents hypothesis
    segments from overshooting the manifest-declared audio duration.

    Mirrors the real-world scenario where the diarization model's last
    prediction frame extends past the audio boundary (e.g., DH_EVAL_0019:
    manifest duration=80.1785 s but the final hypothesis segment ends at
    80.23 s).
    """

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "labels, audio_end, expected_count, expected_last_end",
        [
            (["0.0 50.0 A", "75.280 80.230 B", "85.0 90.0 C"], 80.1785, 2, 80.1785),
            (["0.0 10.0 A", "10.0 20.5 B"], 20.0, 2, 20.0),
            (["0.0 5.0 A", "20.0 25.0 B"], 20.0, 1, 5.0),
            (["0.0 100.0 A"], None, 1, 100.0),
        ],
        ids=["overshoot+drop", "trim_last", "drop_past_boundary", "no_clip"],
    )
    def test_make_diar_annotation_audio_end(self, labels, audio_end, expected_count, expected_last_end):
        """Segments past audio_end are trimmed or dropped; None disables clipping."""
        ann = make_diar_annotation(labels, uniq_name="sess1", audio_end=audio_end)
        assert len(ann) == expected_count
        last = ann[-1]
        actual_end = last.start + last.duration
        assert abs(actual_end - expected_last_end) < 1e-9

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "manifest, expect_zero_fa",
        [
            ({"offset": 0, "duration": 80.0}, True),
            ({}, False),
        ],
        ids=["with_manifest_clamp", "no_manifest_counts_fa"],
    )
    def test_score_labels_manifest_uem_clamp(self, manifest, expect_zero_fa):
        """Manifest duration clamps the auto-UEM; without it, overshoot is FA."""
        file_id = "DH_EVAL_0019"
        ref_ann = make_diar_annotation(_labels((0, 80, "A")), uniq_name=file_id)
        hyp_ann = make_diar_annotation(_labels((0, 80.5, "A")), uniq_name=file_id)
        audio_rttm_map = {file_id: manifest}
        result = score_labels(
            audio_rttm_map,
            [(file_id, ref_ann)],
            [(file_id, hyp_ann)],
            collar=0.0,
            ignore_overlap=False,
            verbose=False,
        )
        assert result is not None
        _, _, (DER, CER, FA, MISS) = result
        if expect_zero_fa:
            assert_der(FA, 0.0)
        else:
            assert FA > 0.0, "Without manifest duration, overshoot should be counted as FA"
