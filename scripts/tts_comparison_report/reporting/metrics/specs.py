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
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class MetricSpec:
    """Specification of a metric shown in evaluation report tables."""

    # Metric key expected in the aggregated metrics JSON.
    key: str
    # Metric name shown in the report tables.
    report_name: str
    # Whether smaller values are better; None means no winner highlighting.
    lower_is_better: Optional[bool]
    # Number of decimal digits used when formatting the metric value.
    round_digits: int
    # Optional unit suffix appended to the formatted metric value.
    units: str = ""
    # Scale factor applied before formatting, e.g. 100 for percentages.
    multiplier: float | int = 1
    # Whether this metric should appear in the cross-benchmark summary table.
    include_in_summary: bool = True
    # Whether this metric may be absent from bucket metrics without causing an error.
    optional: bool = False


@dataclass(frozen=True)
class DistributionMetricSpec:
    """Specification of a metric used in statistical tests and distribution plots."""

    # Metric key expected in the filewise metrics JSON used for statistical testing.
    key: str
    # Metric name shown in the statistical test tables.
    report_name: str
    # Whether smaller values indicate better quality for winner selection.
    lower_is_better: bool
    # Whether this metric should be included in the generated box plot figure.
    add_to_box_plot: bool = True
    # Optional y-axis range applied to the metric plot as (min, max).
    plot_range: Optional[tuple[float, float]] = None
