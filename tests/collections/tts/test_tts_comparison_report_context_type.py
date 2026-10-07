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
"""Unit tests for the TTS comparison report tool: context-type gating of metrics and audio discovery, and
the availability rule of the ground-truth speaker-similarity metric (``pred_gt_ssim``)."""

import json
import warnings
from io import BytesIO
from pathlib import Path
from typing import Any, BinaryIO, Generator, Optional

import matplotlib
import pytest

from scripts.tts_comparison_report.generate_report import _validate_audio_report_benchmarks
from scripts.tts_comparison_report.reporting.components.boxplots import BoxPlotsConfig, prepare_boxplots
from scripts.tts_comparison_report.reporting.components.eval_report import prepare_eval_artifacts
from scripts.tts_comparison_report.reporting.components.metrics_table import (
    prepare_benchmark_metrics_table_rows,
    prepare_summary_metrics_table_rows,
)
from scripts.tts_comparison_report.reporting.components.stat_tests import run_stat_tests
from scripts.tts_comparison_report.reporting.constants import BENCHMARK_META, ContextType
from scripts.tts_comparison_report.reporting.metrics import (
    DistributionMetricSpec,
    DistributionMetricsRegistry,
    MetricsRegistry,
)
from scripts.tts_comparison_report.reporting.models import BenchmarkData, BucketData, BucketStructure
from scripts.tts_comparison_report.reporting.storage import BaseStorage

AUDIO_BENCHMARK = "de_qa"
TEXT_BENCHMARK = "de_qa_ct_text"
CONTEXT_SSIM_NAME = "SSIM (pred vs context)"
GT_SSIM_KEY = "pred_gt_ssim"
GT_SSIM_NAME = "SSIM (pred vs GT)"
NUM_SAMPLES = 8
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


def _bucket_files(root: str, benchmark_dirs: dict[str, bool]) -> dict[str, bytes]:
    """Lay out a results bucket; benchmarks mapped to True also get context and generated audio."""
    files = {}

    for dir_name, with_audio in benchmark_dirs.items():
        benchmark_name = dir_name.split("_", 2)[2]
        base = f"{root}/results/{dir_name}"
        files[f"{base}/{benchmark_name}_metrics_0.json"] = b"{}"
        files[f"{base}/{benchmark_name}_filewise_metrics_0.json"] = b"[]"
        if with_audio:
            files[f"{base}/audio/repeat_0/context_audio_0.wav"] = b"RIFF"
            files[f"{base}/audio/repeat_0/predicted_audio_0.wav"] = b"RIFF"

    return files


class TestAudioReportGating:
    @pytest.mark.unit
    def test_bucket_loading_skips_audio_discovery_for_text_context_benchmarks(self):
        storage = _InMemoryStorage(
            _bucket_files("/buckets/a", {f"cfg_de_{AUDIO_BENCHMARK}": True, f"cfg_de_{TEXT_BENCHMARK}": False})
        )

        bucket = BucketData.from_storage(
            bucket_name="A",
            bucket_path=Path("/buckets/a"),
            bucket_structure=BucketStructure(),
            benchmark_names=(TEXT_BENCHMARK, AUDIO_BENCHMARK),
            check_audio=True,
            storage=storage,
        )

        assert set(bucket.benchmarks) == {AUDIO_BENCHMARK, TEXT_BENCHMARK}
        assert bucket.configuration_str == "cfg"
        assert set(bucket.benchmarks[AUDIO_BENCHMARK].context_audio_paths) == {"context_audio_0"}
        assert set(bucket.benchmarks[AUDIO_BENCHMARK].generated_audio_paths) == {"predicted_audio_0"}
        assert bucket.benchmarks[TEXT_BENCHMARK].context_audio_paths == {}
        assert bucket.benchmarks[TEXT_BENCHMARK].generated_audio_paths == {}

    @pytest.mark.unit
    def test_bucket_loading_still_requires_audio_for_audio_context_benchmarks(self):
        storage = _InMemoryStorage(_bucket_files("/buckets/a", {f"cfg_de_{AUDIO_BENCHMARK}": False}))

        with pytest.raises(FileNotFoundError, match="Missing audio directory"):
            BucketData.from_storage(
                bucket_name="A",
                bucket_path=Path("/buckets/a"),
                bucket_structure=BucketStructure(),
                benchmark_names=(AUDIO_BENCHMARK,),
                check_audio=True,
                storage=storage,
            )

    @pytest.mark.unit
    def test_audio_report_rejects_text_context_benchmarks(self):
        benchmarks = [AUDIO_BENCHMARK, TEXT_BENCHMARK]

        _validate_audio_report_benchmarks(benchmarks, [AUDIO_BENCHMARK])

        with pytest.raises(ValueError, match="text context"):
            _validate_audio_report_benchmarks(benchmarks, [AUDIO_BENCHMARK, TEXT_BENCHMARK])

        with pytest.raises(ValueError, match="not included in evaluation benchmarks"):
            _validate_audio_report_benchmarks([TEXT_BENCHMARK], [AUDIO_BENCHMARK])
