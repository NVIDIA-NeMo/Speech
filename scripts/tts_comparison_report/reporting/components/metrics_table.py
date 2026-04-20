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
from typing import Optional

import numpy as np
from scripts.tts_comparison_report.reporting.metrics import MetricSpec, MetricsRegistry
from scripts.tts_comparison_report.reporting.models import BucketData


def _metric_comparator(
    a: float,
    b: float,
    lower_is_better: Optional[bool],
) -> Optional[bool]:
    if lower_is_better is None:
        return None

    if lower_is_better:
        # If the values ​​are equal, the baseline wins.
        return a <= b

    return a >= b


def _format_metric_values(
    a: float,
    b: float,
    metric: MetricSpec,
) -> tuple[str, str]:
    a, b = metric.multiplier * a, metric.multiplier * b
    a_is_better = _metric_comparator(a, b, metric.lower_is_better)
    a, b = round(a, metric.round_digits), round(b, metric.round_digits)
    a_str, b_str = f"{a}{metric.units}", f"{b}{metric.units}"

    a_str = html.escape(a_str)
    b_str = html.escape(b_str)

    if metric.lower_is_better is not None:
        if a_is_better:
            a_str = f"<strong>{a_str}</strong>"
        else:
            b_str = f"<strong>{b_str}</strong>"

    return a_str, b_str


def prepare_benchmark_metrics_table_rows(
    benchmark_name: str,
    bucket_baseline: BucketData,
    bucket_candidate: BucketData,
) -> list[list[str]]:
    """Prepare formatted metric rows for one benchmark comparison table.

    Args:
        benchmark_name: Name of the benchmark to render.
        bucket_baseline: Baseline bucket data.
        bucket_candidate: Candidate bucket data.

    Returns:
        Table rows containing metric names and formatted baseline/candidate values.

    Raises:
        ValueError: If a required metric is missing for the benchmark.
    """
    rows = []

    for metric in MetricsRegistry:
        a = bucket_baseline.get_metric_avg_value(
            metric_name=metric.key,
            benchmark_name=benchmark_name,
        )
        b = bucket_candidate.get_metric_avg_value(
            metric_name=metric.key,
            benchmark_name=benchmark_name,
        )

        if a is None or b is None:
            if metric.optional:
                continue
            raise ValueError(f"Unknown metric '{metric.key}' for benchmark '{benchmark_name}'.")

        a_str, b_str = _format_metric_values(a, b, metric)

        rows.append([html.escape(metric.report_name), a_str, b_str])

    return rows


def prepare_summary_metrics_table_rows(
    bucket_baseline: BucketData,
    bucket_candidate: BucketData,
) -> list[list[str]]:
    """Prepare formatted metric rows for the summary comparison table.

    Args:
        bucket_baseline: Baseline bucket data.
        bucket_candidate: Candidate bucket data.

    Returns:
        Table rows containing metric names and formatted macro-averaged
        baseline/candidate values.

    Raises:
        ValueError: If a required metric is missing for any benchmark included
            in the summary.
    """
    rows = []

    for metric in MetricsRegistry:
        if not metric.include_in_summary:
            continue

        a_vals, b_vals = [], []
        skip = False

        for benchmark_name in bucket_baseline.benchmarks:
            a = bucket_baseline.get_metric_avg_value(
                metric_name=metric.key,
                benchmark_name=benchmark_name,
            )
            b = bucket_candidate.get_metric_avg_value(
                metric_name=metric.key,
                benchmark_name=benchmark_name,
            )

            if a is None or b is None:
                if metric.optional:
                    skip = True
                    break
                raise ValueError(f"Unknown metric '{metric.key}' for benchmark '{benchmark_name}'.")

            a_vals.append(a)
            b_vals.append(b)

        if skip:
            continue

        avg_a, avg_b = np.mean(a_vals), np.mean(b_vals)
        a_str, b_str = _format_metric_values(avg_a, avg_b, metric)
        rows.append([html.escape(metric.report_name), a_str, b_str])

    return rows
