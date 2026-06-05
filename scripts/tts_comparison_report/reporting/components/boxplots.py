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
from dataclasses import dataclass, field
from io import BytesIO
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import PathPatch
from scripts.tts_comparison_report.reporting.metrics import DistributionMetricSpec, DistributionMetricsRegistry
from scripts.tts_comparison_report.reporting.models import BucketData, StatTestResult, Winner


@dataclass
class BoxPlotsConfig:
    """Styling and layout configuration for generated benchmark box plots."""

    font_family: str = "sans-serif"
    font_list: list[str] = field(default_factory=lambda: ["Arial", "Helvetica", "DejaVu Sans"])

    linewidth: float = 0.4
    default_model_color: str = "#36454F"
    winner_model_color: str = "#7393B3"
    box_alpha: float = 0.35
    grid_alpha: float = 0.4
    fontsize: int = 6
    fontsize_title: int = 8

    widths: float = 0.6
    mean_marker: str = "o"
    mean_marker_color: str = "#CD5C5C"
    mean_marker_size: float = 4.0
    median_color: str = "black"
    whisker_color: str = "#666666"
    cap_color: str = "#666666"
    outlier_color: str = "#708090"
    outlier_marker: str = "o"
    outlier_markersize: float = 3.0
    outlier_alpha: float = 0.5


def _style_boxplot(
    bp: dict[str, PathPatch],
    metric: DistributionMetricSpec,
    winner_lookup: dict[str, Winner],
    cfg: BoxPlotsConfig,
) -> None:
    for i, patch in enumerate(bp["boxes"]):
        winner = winner_lookup[metric.report_name]

        if (i == 0 and winner == Winner.baseline) or (i == 1 and winner == Winner.candidate):
            color = cfg.winner_model_color
        else:
            color = cfg.default_model_color

        patch.set_facecolor(color)
        patch.set_alpha(cfg.box_alpha)
        patch.set_edgecolor(color)
        patch.set_linewidth(cfg.linewidth)


def _add_mean_ci_labels(
    ax: Axes,
    baseline: np.ndarray,
    candidate: np.ndarray,
    metric: DistributionMetricSpec,
    cfg: BoxPlotsConfig,
) -> None:
    for x, values in [(1, baseline), (2, candidate)]:
        mean, median = values.mean(), np.median(values)
        sem = values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0
        ci95 = 1.96 * sem
        label = f"{mean:.3f} ± {ci95:.3f}"

        if metric.plot_range is not None:
            range_ = metric.plot_range[1] - metric.plot_range[0]
        else:
            range_ = values.max() - values.min()

        x_offset = 0.02
        y_offset = 0.03 * range_

        if median > mean and mean - y_offset > 0:
            y_offset = -y_offset

        ax.text(x + x_offset, mean + y_offset, label, ha="left", va="center", fontsize=cfg.fontsize)


def _configure_boxplot_axis(
    ax: Axes,
    metric: DistributionMetricSpec,
    baseline_name: str,
    candidate_name: str,
    cfg: BoxPlotsConfig,
) -> None:
    ax.set_title(metric.report_name, fontsize=cfg.fontsize_title)
    ax.set_xticks([1, 2])
    ax.set_xticklabels([baseline_name, candidate_name])
    ax.tick_params(axis="x", labelsize=cfg.fontsize)
    ax.tick_params(axis="y", labelsize=cfg.fontsize)
    ax.grid(True, axis="y", linestyle="dotted", alpha=cfg.grid_alpha)

    for spine in ax.spines.values():
        spine.set_linewidth(cfg.linewidth)

    ax.tick_params(axis="both", width=cfg.linewidth)

    if metric.plot_range is not None:
        ax.set_ylim(metric.plot_range[0], metric.plot_range[1])


def prepare_boxplots(
    bucket_baseline: BucketData,
    bucket_candidate: BucketData,
    stat_test_results: list[StatTestResult],
    cfg: BoxPlotsConfig,
    benchmark_name: Optional[str] = None,
) -> BytesIO:
    """Create an in-memory box plot figure for summary or benchmark-level metrics.

    Args:
        bucket_baseline: Baseline bucket data.
        bucket_candidate: Candidate bucket data.
        stat_test_results: Statistical test results used to highlight the winning model.
        cfg: Plot styling and layout configuration.
        benchmark_name: Benchmark name. If omitted, metric samples are aggregated
            across all benchmarks.

    Returns:
        PNG image stored in an in-memory bytes buffer.
    """
    baseline_name = bucket_baseline.name
    candidate_name = bucket_candidate.name
    winner_lookup = {res.metric_name: res.winner for res in stat_test_results}
    num_rows = sum(m.add_to_box_plot for m in DistributionMetricsRegistry)
    fig_height = max(2.0 * num_rows, 4.5)

    with plt.rc_context({"font.family": cfg.font_family, "font.sans-serif": cfg.font_list}):
        fig, axs = plt.subplots(num_rows, 1, figsize=(6, fig_height), squeeze=False)
        axs = axs.flatten()
        plot_idx = 0

        for metric in DistributionMetricsRegistry:
            if not metric.add_to_box_plot:
                continue

            baseline = bucket_baseline.get_metric_samples(
                metric_name=metric.key,
                benchmark_name=benchmark_name,
            )
            candidate = bucket_candidate.get_metric_samples(
                metric_name=metric.key,
                benchmark_name=benchmark_name,
            )
            baseline = np.asarray(baseline, dtype=float)
            candidate = np.asarray(candidate, dtype=float)

            ax = axs[plot_idx]
            plot_idx += 1

            bp = ax.boxplot(
                [baseline, candidate],
                positions=[1, 2],
                widths=cfg.widths,
                patch_artist=True,
                showmeans=True,
                meanline=False,
                meanprops={
                    "marker": cfg.mean_marker,
                    "markerfacecolor": cfg.mean_marker_color,
                    "markeredgecolor": cfg.mean_marker_color,
                    "markersize": cfg.mean_marker_size,
                },
                medianprops={
                    "color": cfg.median_color,
                    "linewidth": cfg.linewidth,
                },
                whiskerprops={
                    "color": cfg.whisker_color,
                    "linewidth": cfg.linewidth,
                },
                capprops={
                    "color": cfg.cap_color,
                    "linewidth": cfg.linewidth,
                },
                boxprops={
                    "linewidth": cfg.linewidth,
                },
                flierprops={
                    "marker": cfg.outlier_marker,
                    "markerfacecolor": cfg.outlier_color,
                    "markeredgecolor": cfg.outlier_color,
                    "markersize": cfg.outlier_markersize,
                    "alpha": cfg.outlier_alpha,
                },
            )

            _style_boxplot(bp, metric, winner_lookup, cfg)
            _add_mean_ci_labels(ax, baseline, candidate, metric, cfg)
            _configure_boxplot_axis(ax, metric, baseline_name, candidate_name, cfg)

        fig.tight_layout(rect=[0, 0, 1, 0.985])

    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)

    return buffer
