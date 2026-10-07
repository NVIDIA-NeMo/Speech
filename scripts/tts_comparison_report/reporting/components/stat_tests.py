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
import html
import warnings
from enum import Enum
from typing import Optional

from scipy.stats import mannwhitneyu
from scripts.tts_comparison_report.reporting.constants import P_VAL_ROUND_DIGITS
from scripts.tts_comparison_report.reporting.metrics import DistributionMetricSpec, DistributionMetricsRegistry
from scripts.tts_comparison_report.reporting.models import BucketData, StatTestAnalysisInfo, StatTestResult, Winner


_SIGNIFICANCE_LEVEL: float = 0.05

# Appended to every warning about an optional distribution metric that is skipped as unavailable.
_UNAVAILABLE_METRIC_HINT: str = (
    "Filewise metrics written before the field was saved per file lack it; re-run the evaluation to include it."
)


class _Alternative(str, Enum):
    two_sided = "two-sided"
    greater = "greater"
    less = "less"


def _run_single_stat_test(
    baseline: list[float],
    candidate: list[float],
    lower_is_better: bool,
) -> tuple[Winner, _Alternative, float]:
    if not baseline:
        raise ValueError("Baseline sample is empty.")

    if not candidate:
        raise ValueError("Candidate sample is empty.")

    if len(baseline) != len(candidate):
        warnings.warn(
            "\nBaseline and candidate contain different numbers of samples. "
            "This may indicate missing filewise metrics or dataset mismatch.",
            stacklevel=2,
        )

    # First test whether distributions differ at all, then determine direction.
    p_val_two_sided = mannwhitneyu(baseline, candidate, alternative="two-sided", method="auto").pvalue

    if p_val_two_sided >= _SIGNIFICANCE_LEVEL:
        return Winner.tie, _Alternative.two_sided, round(p_val_two_sided, P_VAL_ROUND_DIGITS)

    p_val = mannwhitneyu(baseline, candidate, alternative="less", method="auto").pvalue

    if p_val < _SIGNIFICANCE_LEVEL:
        winner = Winner.baseline if lower_is_better else Winner.candidate
        p_val = round(p_val, P_VAL_ROUND_DIGITS)
        return winner, _Alternative.less, p_val

    p_val = mannwhitneyu(baseline, candidate, alternative="greater", method="auto").pvalue

    if p_val < _SIGNIFICANCE_LEVEL:
        winner = Winner.candidate if lower_is_better else Winner.baseline
        p_val = round(p_val, P_VAL_ROUND_DIGITS)
        return winner, _Alternative.greater, p_val

    return Winner.tie, _Alternative.two_sided, round(p_val_two_sided, P_VAL_ROUND_DIGITS)


def _map_winner_to_name(
    winner: Winner,
    baseline_name: str,
    candidate_name: str,
) -> str:
    if winner == Winner.baseline:
        return baseline_name
    if winner == Winner.candidate:
        return candidate_name
    return winner.value


def run_stat_tests(
    bucket_baseline: BucketData,
    bucket_candidate: BucketData,
    benchmark_name: Optional[str] = None,
) -> list[StatTestResult]:
    """Run statistical tests for all distribution metrics.

    Args:
        bucket_baseline: Baseline bucket data.
        bucket_candidate: Candidate bucket data.
        benchmark_name: Benchmark name. If omitted, metric samples are aggregated
            across all benchmarks.

    Returns:
        List of StatTestResult instances for configured distribution metrics. Metrics
        restricted to a context type are skipped when the benchmark scope contains no
        benchmark generated with that context type. Optional metrics are skipped with a
        warning when, in either bucket, the key is absent from the filewise metrics of a
        benchmark in the scope (see `BucketData.has_metric_samples`); the pooled scope
        therefore includes an optional metric only if every benchmark carries it in both buckets.

    Raises:
        ValueError: If a metric has a NaN sample or a key missing from some samples, no sample
            carries a required metric's key, or benchmark data is invalid.
        TypeError: If a metric value is not numeric.
    """
    results = []

    for metric in DistributionMetricsRegistry:
        if not bucket_baseline.has_context_type(metric.context_type, benchmark_name):
            continue

        if metric.optional and not (
            bucket_baseline.has_metric_samples(metric.key, benchmark_name, metric.context_type)
            and bucket_candidate.has_metric_samples(metric.key, benchmark_name, metric.context_type)
        ):
            message = _format_unavailable_metric_warning(metric, bucket_baseline, bucket_candidate, benchmark_name)
            warnings.warn(message, stacklevel=2)
            continue

        baseline = bucket_baseline.get_metric_samples(metric.key, benchmark_name, metric.context_type)
        candidate = bucket_candidate.get_metric_samples(metric.key, benchmark_name, metric.context_type)
        winner, alternative, p_value = _run_single_stat_test(
            baseline=baseline,
            candidate=candidate,
            lower_is_better=metric.lower_is_better,
        )
        result = StatTestResult(
            metric_name=metric.report_name,
            winner=winner,
            alternative=alternative.value,
            p_value=p_value,
        )
        results.append(result)

    return results


def prepare_stat_tests_table_rows(
    baseline_name: str,
    candidate_name: str,
    stat_test_results: list[StatTestResult],
) -> list[list[str]]:
    """Prepare formatted rows for a statistical test results table.

    Args:
        baseline_name: Name of the baseline model used in the reports.
        candidate_name: Name of the candidate model used in the reports.
        stat_test_results: Statistical test results to format.

    Returns:
        Table rows containing metric name, winner, alternative hypothesis,
        and p-value.
    """
    rows = []

    for res in stat_test_results:
        winner = _map_winner_to_name(res.winner, baseline_name, candidate_name)
        rows.append(
            [
                html.escape(res.metric_name),
                html.escape(winner),
                html.escape(res.alternative),
                html.escape(str(res.p_value)),
            ]
        )

    return rows


def prepare_stat_tests_analysis_info(
    baseline_name: str,
    candidate_name: str,
    stat_test_results: list[StatTestResult],
) -> StatTestAnalysisInfo:
    """Prepare summary information for the statistical test analysis section.

    Args:
        baseline_name: Name of the baseline model used in the reports.
        candidate_name: Name of the candidate model used in the reports.
        stat_test_results: Statistical test results to summarize.

    Returns:
        Instance of StatTestAnalysisInfo containing the overall winner and its
        key advantages. If no statistically significant wins are present, both
        fields are set to `None`.
    """
    a_wins, b_wins = [], []

    for res in stat_test_results:
        if res.winner == Winner.baseline:
            a_wins.append(res.metric_name)
        elif res.winner == Winner.candidate:
            b_wins.append(res.metric_name)

    if not a_wins and not b_wins:
        winner, advantages = None, None
    else:
        winner, wins = (baseline_name, a_wins) if len(a_wins) >= len(b_wins) else (candidate_name, b_wins)
        advantages = ", ".join(wins)

    return StatTestAnalysisInfo(
        winner=winner,
        advantages=advantages,
    )


def _format_unavailable_metric_warning(
    metric: DistributionMetricSpec,
    bucket_baseline: BucketData,
    bucket_candidate: BucketData,
    benchmark_name: Optional[str],
) -> str:
    buckets = (bucket_baseline, bucket_candidate)

    if benchmark_name is not None:
        reasons = []

        for bucket in buckets:
            reason = bucket.describe_unavailable_metric(metric.key, benchmark_name)

            if reason is not None:
                reasons.append(f"{reason} in bucket '{bucket.name}'")

        joined_reasons = "; ".join(reasons)
        return (
            f"Skipping the statistical test for metric '{metric.key}' on benchmark '{benchmark_name}': "
            f"{joined_reasons}. {_UNAVAILABLE_METRIC_HINT}"
        )

    # Benchmark names in discovery order, deduplicated across both buckets.
    in_scope: list[str] = []
    unavailable: list[str] = []

    for bucket in buckets:
        for name in bucket.get_benchmark_names(metric.context_type):
            if name not in in_scope:
                in_scope.append(name)

            if name not in unavailable and bucket.describe_unavailable_metric(metric.key, name) is not None:
                unavailable.append(name)

    joined_names = ", ".join(unavailable)
    return (
        f"Skipping the pooled statistical test for metric '{metric.key}': "
        f"unavailable for {len(unavailable)} of {len(in_scope)} benchmarks ({joined_names}); "
        f"see the per-benchmark warnings. {_UNAVAILABLE_METRIC_HINT}"
    )
