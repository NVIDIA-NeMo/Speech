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
from scripts.tts_comparison_report.reporting.metrics.specs import DistributionMetricSpec, MetricSpec


MetricsRegistry: list[MetricSpec] = [
    MetricSpec("wer_cumulative", "WER (cumulative)", True, 2, "%", 100),
    MetricSpec("cer_cumulative", "CER (cumulative)", True, 2, "%", 100),
    MetricSpec("wer_filewise_avg", "WER (filewise avg)", True, 2, "%", 100),
    MetricSpec("cer_filewise_avg", "CER (filewise avg)", True, 2, "%", 100),
    MetricSpec("utmosv2_avg", "UTMOS v2", False, 3),
    MetricSpec("ssim_pred_gt_avg", "SSIM (pred vs GT)", False, 4),
    MetricSpec("ssim_pred_context_avg", "SSIM (pred vs context)", False, 4),
    MetricSpec("eou_cutoff_rate", "EoU cut-off rate", True, 3, "", 1, False, True),
    MetricSpec("eou_silence_rate", "EoU silence rate", True, 3, "", 1, False, True),
    MetricSpec("eou_noise_rate", "EoU noise rate", True, 3, "", 1, False, True),
    MetricSpec("eou_error_rate", "EoU error rate", True, 3, "", 1, False, True),
    MetricSpec("total_gen_audio_seconds", "Total audio (sec)", None, 1, "", 1, False),
]


DistributionMetricsRegistry: list[DistributionMetricSpec] = [
    DistributionMetricSpec("cer", "CER", True, True, (0.0, 0.3)),
    DistributionMetricSpec("utmosv2", "UTMOS v2", False),
    DistributionMetricSpec("pred_context_ssim", "SSIM (pred vs context)", False),
]
