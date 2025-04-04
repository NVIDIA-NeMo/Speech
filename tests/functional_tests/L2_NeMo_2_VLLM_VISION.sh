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

TRANSFORMERS_OFFLINE=1 HF_HOME=/home/TestData/vlm/vision/hf/ coverage run -a --data-file=/workspace/.coverage --source=/workspace/nemo \
  tests/collections/vlm/vision/test_llava_next_InternVIT.py \
  --devices=1 \
  --max-steps=5

TRANSFORMERS_OFFLINE=1 HF_HOME=/home/TestData/vlm/vision/hf/ coverage run \
  -a --data-file=/workspace/.coverage --source=/workspace/nemo \
  scripts/vlm/import_hf.py --input_name_or_path="OpenGVLab/InternViT-300M-448px-V2_5"

TRANSFORMERS_OFFLINE=1 HF_HOME=/home/TestData/vlm/vision/hf/ coverage run \
  -a --data-file=/workspace/.coverage --source=/workspace/nemo \
  scripts/vlm/import_hf.py --input_name_or_path="openai/clip-vit-large-patch14"

TRANSFORMERS_OFFLINE=1 HF_HOME=/home/TestData/vlm/vision/hf/ coverage run \
  -a --data-file=/workspace/.coverage --source=/workspace/nemo \
  scripts/vlm/import_hf.py --input_name_or_path="google/siglip-base-patch16-224"