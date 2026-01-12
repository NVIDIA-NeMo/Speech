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

# Boosting ground truth - sanity check for per-utterance boosting
# RNN-T model
coverage run -a --data-file=/workspace/.coverage --source=/workspace/nemo \
    examples/asr/asr_streaming_inference/asr_streaming_infer.py \
    --config-path="../conf/asr_streaming_inference/" \
    --config-name=buffered_rnnt.yaml \
    audio_file="/home/TestData/asr/canary/dev-other-wav-10-boost-gt.json" \
    output_filename="/tmp/stt_inference_boost_gt_res_rnnt.json" \
    asr.model_name="stt_en_fastconformer_transducer_large" \
    streaming.batch_size=5 \
    lang=en \
    enable_pnc=False \
    enable_itn=False \
    enable_nmt=False \
    asr_output_granularity=segment

# TDT model
coverage run -a --data-file=/workspace/.coverage --source=/workspace/nemo \
    examples/asr/asr_streaming_inference/asr_streaming_infer.py \
    --config-path="../conf/asr_streaming_inference/" \
    --config-name=buffered_rnnt.yaml \
    audio_file="/home/TestData/asr/canary/dev-other-wav-10-boost-gt.json" \
    output_filename="/tmp/stt_inference_boost_gt_res_tdt.json" \
    asr.model_name="nvidia/stt_en_fastconformer_tdt_large" \
    streaming.batch_size=5 \
    lang=en \
    enable_pnc=False \
    enable_itn=False \
    enable_nmt=False \
    asr_output_granularity=segment
