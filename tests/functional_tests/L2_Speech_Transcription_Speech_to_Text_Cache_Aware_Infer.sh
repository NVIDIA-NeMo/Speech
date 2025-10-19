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

# TODO(vbataev): fix decoding with CUDA graphs on CI for this test
coverage run -a --data-file=/workspace/.coverage --source=/workspace/nemo \
    examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py \
    model_path="/home/TestData/asr/stt_en_fastconformer_hybrid_large_streaming_multi.nemo" \
    rnnt_decoding.greedy.use_cuda_graph_decoder=false \
    audio_dir="/home/TestData/an4_transcribe/test_subset/" \
    output_path="/tmp/stt_cache_aware_streaming_test_res"
