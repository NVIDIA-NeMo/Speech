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



# Test with waitk policy
coverage run -a --data-file=/workspace/.coverage --source=/workspace/nemo examples/asr/asr_chunked_inference/aed/speech_to_text_aed_streaming_infer.py \
    model_path=/home/TestData/asr/canary/models/canary-1b-flash_HF_20250318.nemo \
    dataset_manifest=/home/TestData/asr/canary/dev-other-wav-10-canary-fields.json \
    output_filename=/tmp/streaming_preds.json \
    batch_size=10 \
    decoding.streaming_policy=waitk \
    +prompt.pnc=yes \
    +prompt.task=asr \
    +prompt.source_lang=en \
    +prompt.target_lang=en


# Test with alignatt policy
coverage run -a --data-file=/workspace/.coverage --source=/workspace/nemo examples/asr/asr_chunked_inference/aed/speech_to_text_aed_streaming_infer.py \
    model_path=/home/TestData/asr/canary/models/canary-1b-flash_HF_20250318.nemo \
    dataset_manifest=/home/TestData/asr/canary/dev-other-wav-10-canary-fields.json \
    output_filename=/tmp/streaming_preds.json \
    batch_size=10 \
    decoding.streaming_policy=alignatt \
    +prompt.pnc=yes \
    +prompt.task=asr \
    +prompt.source_lang=en \
    +prompt.target_lang=en
