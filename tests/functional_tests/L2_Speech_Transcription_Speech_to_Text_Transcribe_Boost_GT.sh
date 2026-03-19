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

# Test RNN-T
coverage run -a --data-file=/workspace/.coverage --source=/workspace/nemo \
    tests/functional_tests/asr_transcribe_boost_ground_truth.py \
    dataset_manifest="/home/TestData/asr/canary/dev-other-wav-10.json" \
    output_filename="/tmp/stt_transcribe_boost_gt_res_rnnt.json" \
    model_path="/home/TestData/asr/stt_en_fastconformer_transducer_large.nemo" \
    batch_size=5 \
    device='cuda:0'

# Test TDT
coverage run -a --data-file=/workspace/.coverage --source=/workspace/nemo \
    tests/functional_tests/asr_transcribe_boost_ground_truth.py \
    dataset_manifest="/home/TestData/asr/canary/dev-other-wav-10.json" \
    output_filename="/tmp/stt_transcribe_boost_gt_res_tdt.json" \
    model_path="/home/TestData/asr/stt_en_fastconformer_tdt_large.nemo" \
    batch_size=5 \
    device='cuda:0'
