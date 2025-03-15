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
CUDA_VISIBLE_DEVICES=0 coverage run --branch -a --data-file=/workspace/.coverage --source=/workspace/nemo examples/asr/speech_to_text_eval.py \
    pretrained_name=QuartzNet15x5Base-En \
    dataset_manifest=/home/TestData/librispeech/librivox-dev-other.json \
    batch_size=64 \
    tolerance=0.1012
