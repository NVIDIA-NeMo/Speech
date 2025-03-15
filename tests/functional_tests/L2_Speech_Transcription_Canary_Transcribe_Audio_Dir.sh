# Copyright (c) 2020-2021, NVIDIA CORPORATION.
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
coverage run --branch -a --data-file=/workspace/.coverage --source=/workspace/nemo examples/asr/transcribe_speech.py \
    audio_dir=/home/TestData/asr/canary/dev-other-wav \
    output_filename=preds.json \
    batch_size=10 \
    pretrained_name=nvidia/canary-1b \
    num_workers=0 \
    amp=false \
    compute_dtype=bfloat16 \
    matmul_precision=medium
