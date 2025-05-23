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
export HF_DATASETS_OFFLINE=1
export HF_HOME=/home/TestData/HF_HOME
export HF_DATASETS_CACHE=${HF_HOME}/datasets
coverage run -a --data-file=/workspace/.coverage --source=/workspace/nemo -m pytest tests/evaluation/eval_unittest.py -m "not pleasefixme"
