# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
"""Unit tests for the TTS comparison report tool: context-type gating of metrics, the availability rule of
the ground-truth speaker-similarity metric (``pred_gt_ssim``), and the audio report, which shows the target
recording for every benchmark and the context prompt for audio-context benchmarks only."""

import json
import re
import warnings
from io import BytesIO
from pathlib import Path
from typing import Any, BinaryIO, Generator, Optional

import matplotlib
import pytest

from scripts.tts_comparison_report.generate_report import _validate_audio_report_benchmarks
from scripts.tts_comparison_report.reporting.components.audio_report import prepare_audio_pairs
from scripts.tts_comparison_report.reporting.components.boxplots import BoxPlotsConfig, prepare_boxplots
from scripts.tts_comparison_report.reporting.components.eval_report import prepare_eval_artifacts
from scripts.tts_comparison_report.reporting.components.metrics_table import (
    prepare_benchmark_metrics_table_rows,
    prepare_summary_metrics_table_rows,
)
from scripts.tts_comparison_report.reporting.components.stat_tests import run_stat_tests
from scripts.tts_comparison_report.reporting.constants import BENCHMARK_META, TEMPLATES_DIR, ContextType
from scripts.tts_comparison_report.reporting.metrics import (
    DistributionMetricSpec,
    DistributionMetricsRegistry,
    MetricsRegistry,
)
from scripts.tts_comparison_report.reporting.models import (
    BenchmarkData,
    BucketData,
    BucketStructure,
    ExpirationInfo,
    TaskInfo,
)
from scripts.tts_comparison_report.reporting.orchestrator import Orchestrator
from scripts.tts_comparison_report.reporting.renderer import Renderer
from scripts.tts_comparison_report.reporting.storage import BaseStorage

AUDIO_BENCHMARK = "de_qa"
TEXT_BENCHMARK = "de_qa_ct_text"
CONTEXT_SSIM_NAME = "SSIM (pred vs context)"
GT_SSIM_KEY = "pred_gt_ssim"
GT_SSIM_NAME = "SSIM (pred vs GT)"
NUM_SAMPLES = 8
# Number of utterances per benchmark in the in-memory buckets of the audio report tests.
NUM_AUDIO_SAMPLES = 2
# Audio layouts under ``audio/repeat_0`` that ``_bucket_files`` can write next to the generated audio.
AUDIO_LAYOUTS = ("audio", "text", "no-target", "none")
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
# Sentinel for the ground-truth SSIM fixture parameters: derive finite values from the bucket offset.
_FINITE: Any = object()


@pytest.fixture(autouse=True, scope="module")
def _headless_matplotlib():
    matplotlib.use("Agg")


def _values(start: float, step: float) -> list[float]:
    return [round(start + i * step, 4) for i in range(NUM_SAMPLES)]


def _make_benchmark(
    name: str,
    offset: float,
    context_ssim: Optional[list[float]],
    context_ssim_avg: Optional[float],
    gt_ssim: Optional[list[float]] = _FINITE,
    gt_ssim_avg: Optional[float] = _FINITE,
) -> BenchmarkData:
    """Build an in-memory benchmark with every metric the registries require.

    Args:
        name: Benchmark name, must be a key of ``BENCHMARK_META``.
        offset: Shift applied to the sample values so the two buckets differ.
        context_ssim: Filewise ``pred_context_ssim`` values, or ``None`` to omit the key.
        context_ssim_avg: Aggregated ``ssim_pred_context_avg`` value, or ``None`` to omit the key.
        gt_ssim: Filewise ``pred_gt_ssim`` values, or ``None`` to omit the key. By default finite values
            shifted by ``offset``, as the evaluator writes them when ground-truth audio exists.
        gt_ssim_avg: Aggregated ``ssim_pred_gt_avg`` value, or ``None`` to omit the key. By default a
            finite value shifted by ``offset``.
    """
    if gt_ssim is _FINITE:
        gt_ssim = _values(0.60 + offset, 0.01)
    if gt_ssim_avg is _FINITE:
        gt_ssim_avg = 0.635 + offset

    cer = _values(0.01 + offset, 0.005)
    utmos = _values(3.0 + offset, 0.05)
    filewise_metrics = []

    for i in range(NUM_SAMPLES):
        item = {"pred_audio_filepath": f"predicted_audio_{i}.wav", "cer": cer[i], "utmosv2": utmos[i]}
        if gt_ssim is not None:
            item["pred_gt_ssim"] = gt_ssim[i]
        if context_ssim is not None:
            item["pred_context_ssim"] = context_ssim[i]
        filewise_metrics.append(item)

    metrics = {
        "wer_cumulative": 0.05 + offset,
        "cer_cumulative": 0.02 + offset,
        "wer_filewise_avg": 0.05 + offset,
        "cer_filewise_avg": 0.02 + offset,
        "utmosv2_avg": 3.2 + offset,
        "eou_cutoff_rate": 0.0,
        "eou_silence_rate": 0.0,
        "eou_noise_rate": 0.0,
        "eou_error_rate": 0.0,
        "total_gen_audio_seconds": 100.0,
    }
    # The QA benchmarks share studio recordings as ground truth, so ``ssim_pred_gt_avg`` is finite for
    # audio- and text-context benchmarks alike; the evaluator writes NaN only without ground-truth audio.
    if gt_ssim_avg is not None:
        metrics["ssim_pred_gt_avg"] = gt_ssim_avg
    if context_ssim_avg is not None:
        metrics["ssim_pred_context_avg"] = context_ssim_avg

    return BenchmarkData(name=name, metrics=metrics, filewise_metrics=filewise_metrics)


def _make_bucket(name: str, benchmarks: list[BenchmarkData]) -> BucketData:
    return BucketData(
        name=name,
        path=Path(f"/buckets/{name}"),
        configuration_str="cfg",
        benchmarks={benchmark.name: benchmark for benchmark in benchmarks},
    )


def _audio_benchmark(
    offset: float,
    gt_ssim: Optional[list[float]] = _FINITE,
    gt_ssim_avg: Optional[float] = _FINITE,
) -> BenchmarkData:
    return _make_benchmark(
        AUDIO_BENCHMARK,
        offset,
        context_ssim=_values(0.70 + offset, 0.01),
        context_ssim_avg=0.735 + offset,
        gt_ssim=gt_ssim,
        gt_ssim_avg=gt_ssim_avg,
    )


def _text_benchmark(
    offset: float,
    gt_ssim: Optional[list[float]] = _FINITE,
    gt_ssim_avg: Optional[float] = _FINITE,
) -> BenchmarkData:
    """Text-context benchmark exactly as the evaluator writes it: context SSIM is NaN everywhere."""
    return _make_benchmark(
        TEXT_BENCHMARK,
        offset,
        context_ssim=[float("nan")] * NUM_SAMPLES,
        context_ssim_avg=float("nan"),
        gt_ssim=gt_ssim,
        gt_ssim_avg=gt_ssim_avg,
    )


def _nan_gt_ssim_benchmark(offset: float) -> BenchmarkData:
    """Audio-context benchmark with NaN ground-truth SSIM, i.e. a broken evaluation (every benchmark has GT audio)."""
    return _audio_benchmark(offset, gt_ssim=[float("nan")] * NUM_SAMPLES, gt_ssim_avg=float("nan"))


def _legacy_benchmark(offset: float) -> BenchmarkData:
    """Audio-context benchmark evaluated before ``pred_gt_ssim`` was saved filewise: the key is absent."""
    return _audio_benchmark(offset, gt_ssim=None)


def _make_buckets(*builders) -> tuple[BucketData, BucketData]:
    baseline = _make_bucket("baseline", [build(0.0) for build in builders])
    candidate = _make_bucket("candidate", [build(0.1) for build in builders])
    return baseline, candidate


def _row_names(rows: list[list[str]]) -> list[str]:
    return [row[0] for row in rows]


def _stat_metric_names(results) -> list[str]:
    return [result.metric_name for result in results]


def _skip_messages(record) -> list[str]:
    """Return the messages of recorded warnings about skipped statistical tests."""
    return [str(warning.message) for warning in record if str(warning.message).startswith("Skipping")]


class TestBenchmarkMeta:
    @pytest.mark.unit
    def test_registry_context_restrictions_and_optional_flags(self):
        aggregated = {metric.key: metric for metric in MetricsRegistry}
        distribution = {metric.key: metric for metric in DistributionMetricsRegistry}

        assert aggregated["ssim_pred_context_avg"].context_type == ContextType.audio
        assert distribution["pred_context_ssim"].context_type == ContextType.audio
        # Ground-truth SSIM depends on ground-truth audio, not on the context type; its per-file key may be
        # absent from older artifacts, so it is the only optional distribution metric. The whole spec is pinned:
        # a flipped `lower_is_better` would silently invert the reported winner.
        assert aggregated["ssim_pred_gt_avg"].context_type is None
        assert not aggregated["ssim_pred_gt_avg"].optional
        assert distribution[GT_SSIM_KEY] == DistributionMetricSpec(GT_SSIM_KEY, GT_SSIM_NAME, False, optional=True)
        assert [metric.key for metric in DistributionMetricsRegistry if metric.optional] == [GT_SSIM_KEY]

    @pytest.mark.unit
    def test_text_context_benchmarks_are_declared_by_name_suffix(self):
        for name, meta in BENCHMARK_META.items():
            expected = ContextType.text if name.endswith("_ct_text") else ContextType.audio
            assert meta.context_type == expected, name
            assert len(meta.lang) == 2, name


class TestBucketContextType:
    @pytest.mark.unit
    def test_benchmark_names_filtered_by_context_type(self):
        baseline, _ = _make_buckets(_audio_benchmark, _text_benchmark)

        assert baseline.get_benchmark_names() == [AUDIO_BENCHMARK, TEXT_BENCHMARK]
        assert baseline.get_benchmark_names(ContextType.audio) == [AUDIO_BENCHMARK]
        assert baseline.get_benchmark_names(ContextType.text) == [TEXT_BENCHMARK]
        assert baseline.get_benchmark_context_type(AUDIO_BENCHMARK) == ContextType.audio
        assert baseline.get_benchmark_context_type(TEXT_BENCHMARK) == ContextType.text

    @pytest.mark.unit
    def test_has_context_type(self):
        baseline, _ = _make_buckets(_audio_benchmark, _text_benchmark)
        text_only, _ = _make_buckets(_text_benchmark)

        assert baseline.has_context_type(None)
        assert baseline.has_context_type(None, TEXT_BENCHMARK)
        assert baseline.has_context_type(ContextType.audio)
        assert baseline.has_context_type(ContextType.audio, AUDIO_BENCHMARK)
        assert not baseline.has_context_type(ContextType.audio, TEXT_BENCHMARK)
        assert not text_only.has_context_type(ContextType.audio)

    @pytest.mark.unit
    def test_unknown_benchmark_raises(self):
        baseline, _ = _make_buckets(_audio_benchmark)

        with pytest.raises(ValueError, match="Unknown benchmark"):
            baseline.get_benchmark_context_type("libritts")

        # The unrestricted scope still validates an explicitly named benchmark.
        with pytest.raises(ValueError, match="Unknown benchmark"):
            baseline.has_context_type(None, "libritts")

    @pytest.mark.unit
    def test_metric_samples_pooled_over_requested_context_type_only(self):
        baseline, _ = _make_buckets(_audio_benchmark, _text_benchmark)

        pooled_all = baseline.get_metric_samples("cer")
        pooled_audio = baseline.get_metric_samples("cer", context_type=ContextType.audio)

        assert len(pooled_all) == 2 * NUM_SAMPLES
        assert pooled_audio == baseline.get_metric_samples("cer", AUDIO_BENCHMARK)

    @pytest.mark.unit
    def test_metric_samples_reject_mismatched_context_type(self):
        baseline, _ = _make_buckets(_audio_benchmark, _text_benchmark)
        text_only, _ = _make_buckets(_text_benchmark)

        with pytest.raises(ValueError, match="was not generated with 'audio' context"):
            baseline.get_metric_samples("cer", TEXT_BENCHMARK, ContextType.audio)

        with pytest.raises(ValueError, match="No benchmarks with 'audio' context"):
            text_only.get_metric_samples("cer", context_type=ContextType.audio)

    @pytest.mark.unit
    def test_empty_bucket_aggregation_raises_value_error(self):
        empty = _make_bucket("empty", [])

        with pytest.raises(ValueError, match="No benchmarks are available"):
            empty.get_metric_samples("cer")


class TestTextContextBenchmark:
    @pytest.mark.unit
    def test_context_ssim_omitted_even_when_values_are_numeric(self):
        """Gating is declared per benchmark, not inferred from NaN values."""

        def numeric_text_benchmark(offset: float) -> BenchmarkData:
            return _make_benchmark(
                TEXT_BENCHMARK,
                offset,
                context_ssim=_values(0.5 + offset, 0.01),
                context_ssim_avg=0.5 + offset,
            )

        baseline, candidate = _make_buckets(numeric_text_benchmark)

        rows = prepare_benchmark_metrics_table_rows(TEXT_BENCHMARK, baseline, candidate)
        results = run_stat_tests(baseline, candidate, TEXT_BENCHMARK)

        assert CONTEXT_SSIM_NAME not in _row_names(rows)
        assert CONTEXT_SSIM_NAME not in _stat_metric_names(results)

    @pytest.mark.unit
    def test_context_ssim_omitted_for_nan_values(self):
        baseline, candidate = _make_buckets(_text_benchmark)

        rows = prepare_benchmark_metrics_table_rows(TEXT_BENCHMARK, baseline, candidate)
        results = run_stat_tests(baseline, candidate, TEXT_BENCHMARK)

        assert CONTEXT_SSIM_NAME not in _row_names(rows)
        # Speaker similarity against the ground-truth recording does not depend on the context type.
        assert GT_SSIM_NAME in _row_names(rows)
        assert _stat_metric_names(results) == ["CER", "UTMOS v2", GT_SSIM_NAME]

    @pytest.mark.unit
    def test_text_only_bucket_summary_has_no_context_ssim(self):
        baseline, candidate = _make_buckets(_text_benchmark)

        rows = prepare_summary_metrics_table_rows(baseline, candidate)
        results = run_stat_tests(baseline, candidate)
        image = prepare_boxplots(baseline, candidate, results, BoxPlotsConfig())

        assert CONTEXT_SSIM_NAME not in _row_names(rows)
        assert GT_SSIM_NAME in _row_names(rows)
        assert _stat_metric_names(results) == ["CER", "UTMOS v2", GT_SSIM_NAME]
        assert image.getvalue().startswith(PNG_SIGNATURE)


class TestAudioContextBenchmark:
    @pytest.mark.unit
    def test_context_ssim_reported(self):
        baseline, candidate = _make_buckets(_audio_benchmark)

        rows = prepare_benchmark_metrics_table_rows(AUDIO_BENCHMARK, baseline, candidate)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            results = run_stat_tests(baseline, candidate, AUDIO_BENCHMARK)
        image = prepare_boxplots(baseline, candidate, results, BoxPlotsConfig(), AUDIO_BENCHMARK)

        assert CONTEXT_SSIM_NAME in _row_names(rows)
        assert _stat_metric_names(results) == ["CER", "UTMOS v2", GT_SSIM_NAME, CONTEXT_SSIM_NAME]
        assert _skip_messages(caught) == []
        assert image.getvalue().startswith(PNG_SIGNATURE)

    @pytest.mark.unit
    def test_nan_filewise_context_ssim_is_an_error(self):
        """A broken context-SSIM evaluation on an audio-context benchmark must not be silently dropped."""

        def broken_audio_benchmark(offset: float) -> BenchmarkData:
            return _make_benchmark(
                AUDIO_BENCHMARK,
                offset,
                context_ssim=[float("nan")] * NUM_SAMPLES,
                context_ssim_avg=0.7 + offset,
            )

        baseline, candidate = _make_buckets(broken_audio_benchmark)

        with pytest.raises(ValueError, match="pred_context_ssim.*contains NaN"):
            run_stat_tests(baseline, candidate, AUDIO_BENCHMARK)

        with pytest.raises(ValueError, match="pred_context_ssim.*contains NaN"):
            run_stat_tests(baseline, candidate)

    @pytest.mark.unit
    @pytest.mark.parametrize("context_ssim_avg", [None, float("nan")], ids=["missing", "nan"])
    def test_missing_aggregated_context_ssim_is_an_error(self, context_ssim_avg):
        def broken_audio_benchmark(offset: float) -> BenchmarkData:
            return _make_benchmark(
                AUDIO_BENCHMARK,
                offset,
                context_ssim=_values(0.7 + offset, 0.01),
                context_ssim_avg=context_ssim_avg,
            )

        baseline, candidate = _make_buckets(broken_audio_benchmark)

        with pytest.raises(ValueError, match="ssim_pred_context_avg"):
            prepare_benchmark_metrics_table_rows(AUDIO_BENCHMARK, baseline, candidate)

        with pytest.raises(ValueError, match="ssim_pred_context_avg"):
            prepare_summary_metrics_table_rows(baseline, candidate)

    @pytest.mark.unit
    def test_nan_aggregated_ground_truth_ssim_is_an_error(self):
        baseline, candidate = _make_buckets(_nan_gt_ssim_benchmark)

        with pytest.raises(ValueError, match="ssim_pred_gt_avg"):
            prepare_benchmark_metrics_table_rows(AUDIO_BENCHMARK, baseline, candidate)

        with pytest.raises(ValueError, match="ssim_pred_gt_avg"):
            prepare_summary_metrics_table_rows(baseline, candidate)


class TestMixedContextBuckets:
    @pytest.mark.unit
    def test_summary_context_ssim_averaged_over_audio_benchmarks_only(self):
        baseline, candidate = _make_buckets(_audio_benchmark, _text_benchmark)

        rows = prepare_summary_metrics_table_rows(baseline, candidate)
        row = next(row for row in rows if row[0] == CONTEXT_SSIM_NAME)

        # Only the audio-context benchmark contributes; its NaN text-context sibling is excluded.
        assert "0.735" in row[1]
        assert "0.835" in row[2]

    @pytest.mark.unit
    def test_eval_artifacts_gate_context_ssim_per_section(self):
        baseline, candidate = _make_buckets(_audio_benchmark, _text_benchmark)

        # Pooling the NaN text-context samples would raise, so a successful run proves they are excluded.
        artifacts = prepare_eval_artifacts(baseline, candidate, BoxPlotsConfig())

        assert CONTEXT_SSIM_NAME in _row_names(artifacts.summary.metrics_table_row)
        assert _row_names(artifacts.summary.stat_test_table_row) == [
            "CER",
            "UTMOS v2",
            GT_SSIM_NAME,
            CONTEXT_SSIM_NAME,
        ]

        audio_result = artifacts.benchmarks[AUDIO_BENCHMARK]
        assert CONTEXT_SSIM_NAME in _row_names(audio_result.metrics_table_row)
        assert CONTEXT_SSIM_NAME in _row_names(audio_result.stat_test_table_row)

        text_result = artifacts.benchmarks[TEXT_BENCHMARK]
        assert CONTEXT_SSIM_NAME not in _row_names(text_result.metrics_table_row)
        assert CONTEXT_SSIM_NAME not in _row_names(text_result.stat_test_table_row)

        for result in [artifacts.summary, audio_result, text_result]:
            # Ground-truth SSIM does not depend on the context type, so every section tests it.
            assert GT_SSIM_NAME in _row_names(result.stat_test_table_row)
            assert result.box_plots.getvalue().startswith(PNG_SIGNATURE)


class TestGroundTruthSsim:
    """Ground-truth SSIM is tested for every benchmark: NaN is an error, an absent key (older artifacts) is skipped."""

    @pytest.mark.unit
    def test_nan_filewise_ground_truth_ssim_is_an_error(self):
        def partial_nan_benchmark(offset: float) -> BenchmarkData:
            gt_ssim = _values(0.60 + offset, 0.01)
            gt_ssim[3] = float("nan")
            return _audio_benchmark(offset, gt_ssim=gt_ssim, gt_ssim_avg=float("nan"))

        # One NaN sample is enough to fail, and the error names that sample rather than the first one.
        baseline, candidate = _make_buckets(partial_nan_benchmark)

        with pytest.raises(ValueError, match="'pred_gt_ssim'.*NaN for sample 'predicted_audio_3.wav'"):
            run_stat_tests(baseline, candidate, AUDIO_BENCHMARK)

        with pytest.raises(ValueError, match="'pred_gt_ssim'.*NaN for sample 'predicted_audio_3.wav'"):
            run_stat_tests(baseline, candidate)

    @pytest.mark.unit
    def test_legacy_filewise_metrics_without_the_key_are_skipped(self):
        baseline, candidate = _make_buckets(_legacy_benchmark)

        with pytest.warns(UserWarning, match="'pred_gt_ssim' is absent from the filewise metrics"):
            results = run_stat_tests(baseline, candidate, AUDIO_BENCHMARK)
        with pytest.warns(UserWarning, match="pred_gt_ssim"):
            artifacts = prepare_eval_artifacts(baseline, candidate, BoxPlotsConfig())

        assert _stat_metric_names(results) == ["CER", "UTMOS v2", CONTEXT_SSIM_NAME]
        # Legacy artifacts still carry the aggregated average, so the metrics table keeps its row.
        audio_result = artifacts.benchmarks[AUDIO_BENCHMARK]
        assert GT_SSIM_NAME in _row_names(audio_result.metrics_table_row)
        assert GT_SSIM_NAME not in _row_names(audio_result.stat_test_table_row)
        assert GT_SSIM_NAME not in _row_names(artifacts.summary.stat_test_table_row)
        for result in [artifacts.summary, audio_result]:
            assert result.box_plots.getvalue().startswith(PNG_SIGNATURE)

    @pytest.mark.unit
    def test_pooled_summary_requires_every_benchmark(self):
        baseline, candidate = _make_buckets(_text_benchmark, _legacy_benchmark)

        with pytest.warns(UserWarning, match="pred_gt_ssim") as record:
            pooled = run_stat_tests(baseline, candidate)
        text_results = run_stat_tests(baseline, candidate, TEXT_BENCHMARK)
        with pytest.warns(UserWarning, match="on benchmark 'de_qa'"):
            audio_results = run_stat_tests(baseline, candidate, AUDIO_BENCHMARK)

        assert _stat_metric_names(pooled) == ["CER", "UTMOS v2", CONTEXT_SSIM_NAME]
        # The benchmark that has the metric still gets the test in its own section.
        assert _stat_metric_names(text_results) == ["CER", "UTMOS v2", GT_SSIM_NAME]
        assert _stat_metric_names(audio_results) == ["CER", "UTMOS v2", CONTEXT_SSIM_NAME]
        [message] = _skip_messages(record)
        assert "pooled statistical test for metric 'pred_gt_ssim'" in message
        assert "1 of 2 benchmarks (de_qa)" in message

    @pytest.mark.unit
    @pytest.mark.parametrize("lacking", ["baseline", "candidate"])
    def test_metric_must_be_available_in_both_buckets(self, lacking):
        baseline_builder = _legacy_benchmark if lacking == "baseline" else _audio_benchmark
        candidate_builder = _legacy_benchmark if lacking == "candidate" else _audio_benchmark
        baseline = _make_bucket("baseline", [baseline_builder(0.0)])
        candidate = _make_bucket("candidate", [candidate_builder(0.1)])
        available = "candidate" if lacking == "baseline" else "baseline"

        with pytest.warns(UserWarning, match="pred_gt_ssim") as record:
            results = run_stat_tests(baseline, candidate, AUDIO_BENCHMARK)
        with pytest.warns(UserWarning, match="unavailable for 1 of 1 benchmarks"):
            pooled = run_stat_tests(baseline, candidate)

        assert GT_SSIM_NAME not in _stat_metric_names(results)
        assert GT_SSIM_NAME not in _stat_metric_names(pooled)
        [message] = _skip_messages(record)
        assert f"in bucket '{lacking}'" in message
        assert f"in bucket '{available}'" not in message

    @pytest.mark.unit
    def test_key_present_in_some_samples_only_is_an_error(self):
        baseline, candidate = _make_buckets(_audio_benchmark)
        del candidate.benchmarks[AUDIO_BENCHMARK].filewise_metrics[3][GT_SSIM_KEY]

        with pytest.raises(ValueError, match="'pred_gt_ssim'.*missing for sample 'predicted_audio_3.wav'"):
            run_stat_tests(baseline, candidate, AUDIO_BENCHMARK)

    @pytest.mark.unit
    def test_required_metric_with_nan_or_missing_sample_fails(self):
        baseline, candidate = _make_buckets(_audio_benchmark)
        candidate.benchmarks[AUDIO_BENCHMARK].filewise_metrics[2]["cer"] = float("nan")

        with pytest.raises(ValueError, match="'cer'.*contains NaN for sample 'predicted_audio_2.wav'"):
            run_stat_tests(baseline, candidate, AUDIO_BENCHMARK)

        # A key missing from some samples is an error too; the test never runs on a reduced distribution.
        baseline, candidate = _make_buckets(_audio_benchmark)
        del candidate.benchmarks[AUDIO_BENCHMARK].filewise_metrics[1]["utmosv2"]
        with pytest.raises(ValueError, match="'utmosv2'.*missing for sample 'predicted_audio_1.wav'"):
            run_stat_tests(baseline, candidate, AUDIO_BENCHMARK)

    @pytest.mark.unit
    def test_has_metric_samples_scope(self):
        def no_gt_text_benchmark(offset: float) -> BenchmarkData:
            return _text_benchmark(offset, gt_ssim=None, gt_ssim_avg=None)

        mixed, _ = _make_buckets(_audio_benchmark, no_gt_text_benchmark)
        text_only, _ = _make_buckets(_text_benchmark)
        empty = _make_bucket("empty", [])

        assert mixed.has_metric_samples(GT_SSIM_KEY, AUDIO_BENCHMARK)
        assert not mixed.has_metric_samples(GT_SSIM_KEY, TEXT_BENCHMARK)
        assert not mixed.has_metric_samples(GT_SSIM_KEY)
        # The context filter restricts the pooled scope.
        assert mixed.has_metric_samples(GT_SSIM_KEY, context_type=ContextType.audio)
        assert not mixed.has_metric_samples(GT_SSIM_KEY, context_type=ContextType.text)
        # An empty scope cannot claim that samples exist.
        assert not text_only.has_metric_samples(GT_SSIM_KEY, context_type=ContextType.audio)
        assert not empty.has_metric_samples(GT_SSIM_KEY)
        assert not empty.has_metric_samples("cer")
        # An unloaded benchmark is an error, not an unavailable metric.
        not_loaded = _make_bucket("not_loaded", [BenchmarkData(name=AUDIO_BENCHMARK, filewise_metrics=None)])
        with pytest.raises(ValueError, match="Filewise metrics not loaded"):
            not_loaded.has_metric_samples(GT_SSIM_KEY)

    @pytest.mark.unit
    def test_describe_unavailable_metric(self):
        available, _ = _make_buckets(_audio_benchmark)
        legacy, _ = _make_buckets(_legacy_benchmark)
        broken, _ = _make_buckets(_audio_benchmark)
        items = broken.benchmarks[AUDIO_BENCHMARK].filewise_metrics
        items[5][GT_SSIM_KEY] = float("nan")
        not_loaded = _make_bucket("not_loaded", [BenchmarkData(name=AUDIO_BENCHMARK, filewise_metrics=None)])
        no_rows = _make_bucket("no_rows", [BenchmarkData(name=AUDIO_BENCHMARK, filewise_metrics=[])])

        assert available.describe_unavailable_metric(GT_SSIM_KEY, AUDIO_BENCHMARK) is None
        assert legacy.describe_unavailable_metric(GT_SSIM_KEY, AUDIO_BENCHMARK) == (
            "'pred_gt_ssim' is absent from the filewise metrics"
        )

        # The error names the offending sample, and a key missing from some rows is reported before a NaN.
        with pytest.raises(ValueError, match="NaN for sample 'predicted_audio_5.wav'"):
            broken.describe_unavailable_metric(GT_SSIM_KEY, AUDIO_BENCHMARK)
        del items[3][GT_SSIM_KEY]
        with pytest.raises(ValueError, match="missing for sample 'predicted_audio_3.wav'"):
            broken.describe_unavailable_metric(GT_SSIM_KEY, AUDIO_BENCHMARK)

        with pytest.raises(ValueError, match="Unknown benchmark"):
            available.describe_unavailable_metric(GT_SSIM_KEY, "libritts")

        for bucket in [not_loaded, no_rows]:
            with pytest.raises(ValueError, match="Filewise metrics not loaded"):
                bucket.describe_unavailable_metric(GT_SSIM_KEY, AUDIO_BENCHMARK)


class _InMemoryStorage(BaseStorage):
    """Minimal storage backend over an in-memory file tree."""

    def __init__(self, files: dict[str, bytes]):
        self._files = {Path(path): content for path, content in files.items()}
        self._dirs = {parent for path in self._files for parent in path.parents}

    def exists(self, path: Path) -> bool:
        return path in self._files or path in self._dirs

    def iter_dir(self, path: Path, only_dirs: bool = False) -> Generator[Path, None, None]:
        children = {p for p in self._files if p.parent == path} | {d for d in self._dirs if d.parent == path}
        for child in sorted(children):
            if only_dirs and child not in self._dirs:
                continue
            yield child

    def open_file(self, path: Path) -> BinaryIO:
        return BytesIO(self._files[path])

    def read_json(self, path: Path) -> Any:
        return json.loads(self._files[path])

    def read_bytes(self, path: Path) -> bytes:
        return self._files[path]


class _FakeS3Client:
    """Record uploaded keys and return deterministic URLs instead of talking to S3."""

    def __init__(self):
        self.keys: list[str] = []

    def upload_fileobj(self, fileobj: BinaryIO, key: str, expires_in: int, content_type: Optional[str] = None) -> str:
        assert fileobj.read() == b"RIFF"
        return self._store(key)

    def upload_bytes(self, data: bytes, key: str, expires_in: int, content_type: Optional[str] = None) -> str:
        return self._store(key)

    def _store(self, key: str) -> str:
        self.keys.append(key)
        return f"https://s3.test/{key}"


def _filewise_items(benchmark_name: str) -> list[dict[str, Any]]:
    """Filewise metrics rows carrying the keys the audio report pairs samples by.

    Text-context rows keep the ``context_audio_filepath`` key with a ``None`` value, as the evaluator writes them.
    """
    with_context = BENCHMARK_META[benchmark_name].context_type == ContextType.audio
    return [
        {
            "pred_audio_filepath": f"predicted_audio_{i}.wav",
            "gt_text": f"Utterance {i} of {benchmark_name}.",
            "gt_audio_filepath": f"gt/{benchmark_name}/{i}.wav",
            "context_audio_filepath": f"ctx/{benchmark_name}/{i}.wav" if with_context else None,
        }
        for i in range(NUM_AUDIO_SAMPLES)
    ]


def _bucket_files(root: str, benchmark_dirs: dict[str, str]) -> dict[str, bytes]:
    """Lay out a results bucket as ``magpietts_inference`` writes it.

    Args:
        root: Bucket root directory.
        benchmark_dirs: Mapping from results directory name (``<configuration>_<lang>_<benchmark>``) to the audio
            layout under ``audio/repeat_0``: ``"audio"`` (context, target and generated files), ``"text"`` (target
            and generated), ``"no-target"`` (context and generated) or ``"none"`` (no audio directory).
    """
    files = {}

    for dir_name, layout in benchmark_dirs.items():
        if layout not in AUDIO_LAYOUTS:
            raise ValueError(f"Unknown audio layout: '{layout}'.")

        benchmark_name = dir_name.split("_", 2)[2]
        base = f"{root}/results/{dir_name}"
        files[f"{base}/{benchmark_name}_metrics_0.json"] = b"{}"
        files[f"{base}/{benchmark_name}_filewise_metrics_0.json"] = json.dumps(
            _filewise_items(benchmark_name)
        ).encode()

        if layout == "none":
            continue

        prefixes = ["predicted_audio_"]
        if layout in ("audio", "no-target"):
            prefixes.append("context_audio_")
        if layout in ("audio", "text"):
            prefixes.append("target_audio_")

        for i in range(NUM_AUDIO_SAMPLES):
            for prefix in prefixes:
                files[f"{base}/audio/repeat_0/{prefix}{i}.wav"] = b"RIFF"

    return files


def _load_bucket(name: str, storage: BaseStorage, benchmark_names: tuple[str, ...]) -> BucketData:
    bucket = BucketData.from_storage(
        bucket_name=name,
        bucket_path=Path(f"/buckets/{name}"),
        bucket_structure=BucketStructure(),
        benchmark_names=benchmark_names,
        check_audio=True,
        storage=storage,
    )
    bucket.load_metrics(storage)
    return bucket


def _paired_buckets() -> tuple[BucketData, BucketData, _InMemoryStorage]:
    """Baseline and candidate buckets with one audio-context and one text-context benchmark in a shared storage."""
    layouts = {f"cfg_de_{AUDIO_BENCHMARK}": "audio", f"cfg_de_{TEXT_BENCHMARK}": "text"}
    files = {}

    for name in ("baseline", "candidate"):
        files.update(_bucket_files(f"/buckets/{name}", layouts))

    storage = _InMemoryStorage(files)
    names = (TEXT_BENCHMARK, AUDIO_BENCHMARK)
    return _load_bucket("baseline", storage, names), _load_bucket("candidate", storage, names), storage


def _sample_names(prefix: str) -> set[str]:
    return {f"{prefix}{i}" for i in range(NUM_AUDIO_SAMPLES)}


def _report_sections(report: str) -> dict[str, str]:
    """Split the rendered audio report into benchmark sections keyed by section id."""
    parts = re.split(r'<h2 id="([^"]+)">', report)
    return dict(zip(parts[1::2], parts[2::2]))


def _header_labels(section: str) -> list[str]:
    return re.findall(r'<div class="model-header">\s*(.*?)\s*</div>', section, flags=re.S)


class TestAudioReport:
    @pytest.mark.unit
    def test_bucket_loading_collects_target_audio_for_every_context_type(self):
        layouts = {f"cfg_de_{AUDIO_BENCHMARK}": "audio", f"cfg_de_{TEXT_BENCHMARK}": "text"}
        storage = _InMemoryStorage(_bucket_files("/buckets/a", layouts))

        bucket = _load_bucket("a", storage, (TEXT_BENCHMARK, AUDIO_BENCHMARK))
        audio_data = bucket.benchmarks[AUDIO_BENCHMARK]
        text_data = bucket.benchmarks[TEXT_BENCHMARK]

        assert set(bucket.benchmarks) == {AUDIO_BENCHMARK, TEXT_BENCHMARK}
        assert bucket.configuration_str == "cfg"
        assert set(audio_data.context_audio_paths) == _sample_names("context_audio_")
        assert set(audio_data.target_audio_paths) == _sample_names("target_audio_")
        assert set(audio_data.generated_audio_paths) == _sample_names("predicted_audio_")
        # Text-context benchmarks have no context prompt but do have the target recording.
        assert text_data.context_audio_paths == {}
        assert set(text_data.target_audio_paths) == _sample_names("target_audio_")
        assert set(text_data.generated_audio_paths) == _sample_names("predicted_audio_")

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "benchmark_name, layout, message",
        [
            (AUDIO_BENCHMARK, "text", "No context audio files"),
            (TEXT_BENCHMARK, "no-target", "No target audio files"),
            (AUDIO_BENCHMARK, "none", "Missing audio directory"),
        ],
    )
    def test_bucket_loading_requires_reference_audio(self, benchmark_name, layout, message):
        storage = _InMemoryStorage(_bucket_files("/buckets/a", {f"cfg_de_{benchmark_name}": layout}))

        with pytest.raises(FileNotFoundError, match=message):
            BucketData.from_storage(
                bucket_name="A",
                bucket_path=Path("/buckets/a"),
                bucket_structure=BucketStructure(),
                benchmark_names=(benchmark_name,),
                check_audio=True,
                storage=storage,
            )

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "benchmark_name, missing_prefix",
        [(TEXT_BENCHMARK, "target_audio_"), (AUDIO_BENCHMARK, "context_audio_")],
    )
    def test_bucket_loading_requires_a_reference_file_for_every_sample(self, benchmark_name, missing_prefix):
        layout = "audio" if benchmark_name == AUDIO_BENCHMARK else "text"
        files = _bucket_files("/buckets/a", {f"cfg_de_{benchmark_name}": layout})
        del files[f"/buckets/a/results/cfg_de_{benchmark_name}/audio/repeat_0/{missing_prefix}1.wav"]
        kind = missing_prefix.split("_")[0]
        message = f"Missing {kind} audio '{missing_prefix}1' for generated sample 'predicted_audio_1' in '/buckets/a/"

        with pytest.raises(ValueError, match=message):
            BucketData.from_storage(
                bucket_name="A",
                bucket_path=Path("/buckets/a"),
                bucket_structure=BucketStructure(),
                benchmark_names=(benchmark_name,),
                check_audio=True,
                storage=_InMemoryStorage(files),
            )

    @pytest.mark.unit
    def test_audio_report_benchmarks_admit_text_context(self):
        benchmarks = [AUDIO_BENCHMARK, TEXT_BENCHMARK]

        _validate_audio_report_benchmarks(benchmarks, [AUDIO_BENCHMARK, TEXT_BENCHMARK])

        with pytest.raises(ValueError, match="not included in evaluation benchmarks"):
            _validate_audio_report_benchmarks([TEXT_BENCHMARK], [AUDIO_BENCHMARK])

        with pytest.raises(ValueError, match="Empty list of benchmark names"):
            _validate_audio_report_benchmarks(benchmarks, [])

    @pytest.mark.unit
    def test_audio_pairs_carry_target_and_optional_context(self):
        baseline, candidate, _ = _paired_buckets()
        used = [AUDIO_BENCHMARK, TEXT_BENCHMARK]

        pairs = prepare_audio_pairs(baseline, candidate, BucketStructure(), used, NUM_AUDIO_SAMPLES)

        assert set(pairs) == set(used)
        for benchmark_name in used:
            assert len(pairs[benchmark_name]) == NUM_AUDIO_SAMPLES
            for pair in pairs[benchmark_name]:
                idx = pair.baseline_path.stem.removeprefix("predicted_audio_")
                assert pair.target_path.name == f"target_audio_{idx}.wav"
                assert pair.candidate_path.name == f"predicted_audio_{idx}.wav"
                assert pair.text == f"Utterance {idx} of {benchmark_name}."
        for pair in pairs[AUDIO_BENCHMARK]:
            assert pair.context_path is not None
            assert pair.context_path.name == pair.target_path.name.replace("target_", "context_")
        for pair in pairs[TEXT_BENCHMARK]:
            assert pair.context_path is None

        # A metrics row whose reference file was not discovered is an error that names the sample.
        del baseline.benchmarks[TEXT_BENCHMARK].target_audio_paths["target_audio_1"]
        with pytest.raises(ValueError, match="Missing target audio 'target_audio_1' for sample 'predicted_audio_1'"):
            baseline.get_benchmark_sample_meta(TEXT_BENCHMARK, BucketStructure())

    @pytest.mark.unit
    def test_text_context_sample_ids_match_across_buckets(self):
        """Text-context rows (null context path) get distinct sample ids that are equal across buckets."""
        baseline, candidate, _ = _paired_buckets()
        structure = BucketStructure()

        baseline_meta = baseline.get_benchmark_sample_meta(TEXT_BENCHMARK, structure)
        candidate_meta = candidate.get_benchmark_sample_meta(TEXT_BENCHMARK, structure)

        assert set(baseline_meta) == _sample_names("predicted_audio_")
        assert {n: m.sample_id for n, m in baseline_meta.items()} == {
            n: m.sample_id for n, m in candidate_meta.items()
        }
        assert len({m.sample_id for m in baseline_meta.values()}) == NUM_AUDIO_SAMPLES
        assert all(meta.context_path is None for meta in baseline_meta.values())
        assert all(meta.target_path.name.startswith("target_audio_") for meta in baseline_meta.values())

        # Buckets evaluated on different data are still detected.
        candidate.benchmarks[TEXT_BENCHMARK].filewise_metrics[0]["gt_audio_filepath"] = "gt/other.wav"
        with pytest.raises(ValueError, match="Sample id mismatch"):
            prepare_audio_pairs(baseline, candidate, structure, [TEXT_BENCHMARK], NUM_AUDIO_SAMPLES)

    @pytest.mark.unit
    def test_metrics_row_without_generated_audio_is_an_error(self):
        baseline, candidate, _ = _paired_buckets()
        for bucket in (baseline, candidate):
            del bucket.benchmarks[TEXT_BENCHMARK].generated_audio_paths["predicted_audio_1"]

        with pytest.raises(
            ValueError, match="Missing generated audio 'predicted_audio_1.wav' for sample 'predicted_audio_1'"
        ):
            prepare_audio_pairs(baseline, candidate, BucketStructure(), [TEXT_BENCHMARK], NUM_AUDIO_SAMPLES)

    @pytest.mark.unit
    def test_differing_sample_sets_name_the_missing_samples(self):
        baseline, candidate, _ = _paired_buckets()
        # The candidate was evaluated on one utterance fewer: its metrics and audio agree, but the buckets differ.
        del candidate.benchmarks[TEXT_BENCHMARK].generated_audio_paths["predicted_audio_1"]
        del candidate.benchmarks[TEXT_BENCHMARK].target_audio_paths["target_audio_1"]
        candidate.benchmarks[TEXT_BENCHMARK].filewise_metrics = [
            item
            for item in candidate.benchmarks[TEXT_BENCHMARK].filewise_metrics
            if item["pred_audio_filepath"] != "predicted_audio_1.wav"
        ]

        with pytest.raises(
            ValueError, match="differ for benchmark 'de_qa_ct_text': missing in bucket 'candidate': predicted_audio_1"
        ):
            prepare_audio_pairs(baseline, candidate, BucketStructure(), [TEXT_BENCHMARK], NUM_AUDIO_SAMPLES)

    @pytest.mark.unit
    def test_audio_report_renders_target_column_and_optional_context(self):
        baseline, candidate, storage = _paired_buckets()
        s3_client = _FakeS3Client()
        orchestrator = Orchestrator(BucketStructure(), storage, s3_client, Renderer(TEMPLATES_DIR))
        used = [AUDIO_BENCHMARK, TEXT_BENCHMARK]
        prefix = "reports/test"
        pairs = prepare_audio_pairs(baseline, candidate, BucketStructure(), used, NUM_AUDIO_SAMPLES)

        uploaded = orchestrator._upload_audio(used, pairs, prefix)
        report = orchestrator._render_audio_report(
            baseline_name="A",
            candidate_name="B",
            used_benchmarks=used,
            uploaded_audio_info=uploaded,
            task_info=TaskInfo(task_id="NEMOTTS-1", jira_id="NEMOTTS-1", jira_url="https://jira.test/NEMOTTS-1"),
            expiration_info=ExpirationInfo(timestamp=0, path_str="2027-01-01T00-00-00Z", user_str="2027-01-01"),
        )
        sections = _report_sections(report)
        audio_section = sections[AUDIO_BENCHMARK]
        text_section = sections[TEXT_BENCHMARK]

        assert _header_labels(audio_section) == ["Context", "Target", "A", "B"]
        assert audio_section.count("<audio") == 4 * NUM_AUDIO_SAMPLES
        assert "no-context" not in audio_section
        assert _header_labels(text_section) == ["Target", "A", "B"]
        assert text_section.count("<audio") == 3 * NUM_AUDIO_SAMPLES
        assert re.search(r'class="[^"]*\bno-context\b', text_section)

        for benchmark_name in used:
            assert f"{prefix}/audio/target_{benchmark_name}_0.wav" in s3_client.keys
            for i in range(NUM_AUDIO_SAMPLES):
                assert f"{prefix}/audio/context_{TEXT_BENCHMARK}_{i}.wav" not in s3_client.keys
        assert f"{prefix}/audio/context_{AUDIO_BENCHMARK}_0.wav" in s3_client.keys
        assert uploaded[TEXT_BENCHMARK][0].context_url is None
        assert (
            uploaded[TEXT_BENCHMARK][0].target_url == f"https://s3.test/{prefix}/audio/target_{TEXT_BENCHMARK}_0.wav"
        )
        assert uploaded[TEXT_BENCHMARK][0].target_url in text_section
        assert uploaded[AUDIO_BENCHMARK][0].context_url in audio_section
