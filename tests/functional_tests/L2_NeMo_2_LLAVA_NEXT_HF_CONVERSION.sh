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

# Ensure output directory exists
mkdir -p /tmp/nemo2_llava_next_hf_conversion_results/$RUN_ID

# Run the HF conversion test with a smaller test model to save time
HF_HOME=/home/TestData/ykarnati/hf_data/ coverage run --branch -a --data-file=/workspace/.coverage --source=/workspace/nemo \
    tests/collections/vlm/llava_next/test_hf_conversion.py \
    --hf-path="llava-hf/llava-v1.6-vicuna-7b-hf" \
    --output-dir="/tmp/nemo2_llava_next_hf_conversion_results/$RUN_ID" 