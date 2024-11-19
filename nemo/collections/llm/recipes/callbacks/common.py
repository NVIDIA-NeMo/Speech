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

from typing import Optional

from nemo_run import Config, cli

from nemo.utils.import_utils import safe_import

res_module, HAVE_RES = safe_import('nvidia_resiliency_ext.ptl_resiliency')


@cli.factory(is_target_default=True)
def straggler_det_callback(
    straggler_report_time_interval: Optional[int] = 300, stop_if_detected_straggler: Optional[bool] = True
) -> Config[res_module.StragglerDetectionCallback]:
    """
    This callback is used to detect slower ranks participating in a PyTorch distributed workload.
    This callback is obtained from nvidia-resiliency-ext.
    Performance scores are scalar values from 0.0 (worst) to 1.0 (best), reflecting each rank's performance.
    A performance score can be interpreted as the ratio of current performance to reference performance.
    Depending on the reference used, there are two types of performance scores:
    Relative performance score: The best-performing rank in the workload is used as a reference.
    Individual performance score: The best historical performance of the rank is used as a reference.
    If the performance score drops below the threshold which is set to 0.7, it is deemed as a straggler.
    To detect the stragglers, users can enable this callback which reports the performance scores every 5mins.
    Args:
        straggler_report_time_interval (int): Performance score reporting frequency in seconds, Default is 300 seconds.
        stop_if_detected_straggler (bool): Whether to stop training if a straggler is detection. Default is True.
    """

    return Config(
        res_module.StragglerDetectionCallback,
        report_time_interval=straggler_report_time_interval,
        calc_relative_gpu_perf=True,
        calc_individual_gpu_perf=True,
        num_gpu_perf_scores_to_print=5,
        gpu_relative_perf_threshold=0.7,
        gpu_individual_perf_threshold=0.7,
        stop_if_detected=stop_if_detected_straggler,
        enable_ptl_logging=True,
    )
