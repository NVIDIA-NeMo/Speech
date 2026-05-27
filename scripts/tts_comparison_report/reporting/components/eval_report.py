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
from scripts.tts_comparison_report.reporting.components.boxplots import BoxPlotsConfig, prepare_boxplots
from scripts.tts_comparison_report.reporting.components.metrics_table import (
    prepare_benchmark_metrics_table_rows,
    prepare_summary_metrics_table_rows,
)
from scripts.tts_comparison_report.reporting.components.stat_tests import (
    prepare_stat_tests_analysis_info,
    prepare_stat_tests_table_rows,
    run_stat_tests,
)
from scripts.tts_comparison_report.reporting.models import BucketData, EvalArtifacts, EvalResult, ModelConfiguration


def prepare_eval_artifacts(
    bucket_baseline: BucketData,
    bucket_candidate: BucketData,
    box_plots_cfg: BoxPlotsConfig,
) -> EvalArtifacts:
    """Prepare summary and benchmark-level evaluation artifacts for report rendering.

    Args:
        bucket_baseline: Baseline bucket data.
        bucket_candidate: Candidate bucket data.
        box_plots_cfg: Configuration used to generate benchmark and summary box plots.

    Returns:
        Evaluation artifacts containing configuration metadata, summary results,
        and per-benchmark results.
    """
    baseline_name = bucket_baseline.name
    candidate_name = bucket_candidate.name
    is_self_comparison = bucket_baseline.path == bucket_candidate.path

    metrics_table_row = prepare_summary_metrics_table_rows(bucket_baseline, bucket_candidate)
    stat_test_results = run_stat_tests(bucket_baseline, bucket_candidate)
    stat_test_table_row = prepare_stat_tests_table_rows(baseline_name, candidate_name, stat_test_results)
    stat_tests_analysis_info = prepare_stat_tests_analysis_info(baseline_name, candidate_name, stat_test_results)

    box_plots = prepare_boxplots(
        bucket_baseline=bucket_baseline,
        bucket_candidate=bucket_candidate,
        stat_test_results=stat_test_results,
        cfg=box_plots_cfg,
    )

    configuration = ModelConfiguration(
        baseline=bucket_baseline.configuration_str,
        candidate=bucket_candidate.configuration_str,
    )
    summary = EvalResult(
        metrics_table_row=metrics_table_row,
        stat_test_table_row=stat_test_table_row,
        stat_tests_analysis_info=stat_tests_analysis_info,
        box_plots=box_plots,
    )
    benchmarks = {}

    for benchmark_name in bucket_baseline.benchmarks:
        metrics_table_row = prepare_benchmark_metrics_table_rows(benchmark_name, bucket_baseline, bucket_candidate)
        stat_test_results = run_stat_tests(bucket_baseline, bucket_candidate, benchmark_name)
        stat_test_table_row = prepare_stat_tests_table_rows(baseline_name, candidate_name, stat_test_results)
        stat_tests_analysis_info = prepare_stat_tests_analysis_info(baseline_name, candidate_name, stat_test_results)

        box_plots = prepare_boxplots(
            bucket_baseline=bucket_baseline,
            bucket_candidate=bucket_candidate,
            stat_test_results=stat_test_results,
            cfg=box_plots_cfg,
            benchmark_name=benchmark_name,
        )

        benchmarks[benchmark_name] = EvalResult(
            metrics_table_row=metrics_table_row,
            stat_test_table_row=stat_test_table_row,
            stat_tests_analysis_info=stat_tests_analysis_info,
            box_plots=box_plots,
        )

    return EvalArtifacts(
        configuration=configuration,
        summary=summary,
        benchmarks=benchmarks,
        is_self_comparison=is_self_comparison,
    )
