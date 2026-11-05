# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
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
# Canary-1B-flash long-form chunked inference from an audio directory (with timestamps) and from a manifest.
set -e
coverage run -a --data-file=/workspace/.coverage --source=/workspace/nemo examples/asr/asr_chunked_inference/aed/speech_to_text_aed_chunked_infer.py \
    model_path=/home/TestData/asr/canary/models/canary-1b-flash_HF_20250318.nemo \
    audio_dir=/home/TestData/asr/longform/earnings22 \
    output_filename=preds.json \
    timestamps=True

coverage run -a --data-file=/workspace/.coverage --source=/workspace/nemo examples/asr/asr_chunked_inference/aed/speech_to_text_aed_chunked_infer.py \
    model_path=/home/TestData/asr/canary/models/canary-1b-flash_HF_20250318.nemo \
    dataset_manifest=/home/TestData/asr/longform/earnings22_manifest_1sample.json \
    output_filename=preds.json
