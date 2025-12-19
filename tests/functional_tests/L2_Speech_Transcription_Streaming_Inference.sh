# Copyright (c) 2020-2025, NVIDIA CORPORATION.
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
coverage run -a --data-file=/workspace/.coverage --source=/workspace/nemo examples/asr/asr_streaming_inference/asr_streaming_infer.py \
    --config-path="../conf/asr_streaming_inference/" \
    --config-name=buffered_ctc.yaml \
    audio_file="/home/TestData/an4_transcribe/test_subset/" \
    output_filename="/tmp/buffered_ctc_test_res.json" \
    output_dir="/tmp/buffered_ctc_test_dir" \
    lang=en \
    enable_pnc=False \
    enable_itn=False \
    enable_nmt=False \
    asr_output_granularity=segment

coverage run -a --data-file=/workspace/.coverage --source=/workspace/nemo examples/asr/asr_streaming_inference/asr_streaming_infer.py \
    --config-path="../conf/asr_streaming_inference/" \
    --config-name=buffered_rnnt.yaml \
    audio_file="/home/TestData/an4_transcribe/test_subset/" \
    output_filename="/tmp/buffered_rnnt_test_res.json" \
    output_dir="/tmp/buffered_rnnt_test_dir" \
    lang=en \
    enable_pnc=False \
    enable_itn=False \
    enable_nmt=False \
    asr_output_granularity=segment
