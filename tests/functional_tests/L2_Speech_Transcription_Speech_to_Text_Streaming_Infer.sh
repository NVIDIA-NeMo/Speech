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

coverage run -a --data-file=/workspace/.coverage --source=/workspace/nemo \
    examples/asr/asr_chunked_inference/rnnt/speech_to_text_streaming_infer_rnnt.py \
    model_path="/home/TestData/asr/stt_en_fastconformer_transducer_large.nemo" \
    audio_dir="/home/TestData/an4_transcribe/test_subset/" \
    chunk_secs=2.0 \
    left_context_secs=10.0 \
    right_context_secs=2.0 \
    timestamps=true \
    output_filename="/tmp/stt_streaming_test_res.json"

# Boosting ground truth - sanity check for per-utterance boosting
coverage run -a --data-file=/workspace/.coverage --source=/workspace/nemo \
    examples/asr/asr_chunked_inference/rnnt/speech_to_text_streaming_infer_rnnt.py \
    model_path="/home/TestData/asr/stt_en_fastconformer_transducer_large.nemo" \
    dataset_manifest="/home/TestData/asr/canary/dev-other-wav-10-boost-gt.json" \
    use_per_stream_biasing=true \
    chunk_secs=2.0 \
    left_context_secs=10.0 \
    right_context_secs=2.0 \
    batch_size=5 \
    timestamps=true \
    output_filename="/tmp/stt_streaming_test_res.json"
