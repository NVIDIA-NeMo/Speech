# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

"""Common callback utilities for LLM training recipes.

This module provides factory functions for creating various callbacks used in LLM training,
including straggler detection and performance monitoring.
"""

from typing import Optional

from nemo_run import Config, cli

from nemo.utils.import_utils import safe_import

res_module, HAVE_RES = safe_import('nvidia_resiliency_ext.ptl_resiliency')


@cli.factory(is_target_default=True)
def straggler_det_callback(
    straggler_report_time_interval: Optional[int] = 300,
    stop_if_detected_straggler: Optional[bool] = True,
    gpu_relative_perf_threshold: Optional[float] = 0.7,
    gpu_individual_perf_threshold: Optional[float] = 0.7,
) -> Config[res_module.StragglerDetectionCallback]:
    """Creates a callback for detecting slower ranks in PyTorch distributed workloads.

    This callback from nvidia-resiliency-ext monitors rank performance using two metrics:
    1. Relative performance: Compared to the best-performing rank
    2. Individual performance: Compared to the rank's best historical performance

    Performance scores range from 0.0 (worst) to 1.0 (best). A rank is considered
    a straggler if its score falls below the configured threshold (default 0.7).
    The callback reports scores every 5 minutes by default.

    Args:
        straggler_report_time_interval: Performance score reporting frequency in seconds.
            Defaults to 300 seconds.
        stop_if_detected_straggler: Whether to stop training if a straggler is detected.
            Defaults to True.
        gpu_relative_perf_threshold: Relative performance threshold for straggler detection.
            Defaults to 0.7.
        gpu_individual_perf_threshold: Individual performance threshold for detection.
            Defaults to 0.7.
    """
    return Config(
        res_module.StragglerDetectionCallback,
        report_time_interval=straggler_report_time_interval,
        calc_relative_gpu_perf=True,
        calc_individual_gpu_perf=True,
        num_gpu_perf_scores_to_print=5,
        gpu_relative_perf_threshold=gpu_relative_perf_threshold,
        gpu_individual_perf_threshold=gpu_individual_perf_threshold,
        stop_if_detected=stop_if_detected_straggler,
        enable_ptl_logging=True,
    )
