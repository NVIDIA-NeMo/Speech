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

TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 TRANSFORMERS_OFFLINE=1 coverage run -a --data-file=/workspace/.coverage --source=/workspace/nemo tests/collections/llm/conversion/test_hyena_import_from_hf.py \
    --output-path=tests/collections/llm/hyena_conversion_results/$RUN_ID
