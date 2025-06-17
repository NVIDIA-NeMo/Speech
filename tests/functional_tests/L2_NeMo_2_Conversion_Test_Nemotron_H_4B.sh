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

export HF_LOCAL_MODEL_PATH=/home/TestData/llm/models/nemotronh-4B
NEMO_MODEL_TYPE=MambaModel
NEMO_MODEL_CONFIG=NemotronHConfig4B
NEMO_OUTPUT_PATH=/tmp/output_nemo2_ckpt
HF_OUTPUT_PATH=/tmp/output_hf_ckpt

coverage run -a --data-file=/workspace/.coverage --source=/workspace/nemo tests/collections/llm/conversion/test_nmh_conversion.py --conversion_type=HF_TO_NEMO2 --source_ckpt=${HF_LOCAL_MODEL_PATH} --target_ckpt=${NEMO_OUTPUT_PATH}
coverage run -a --data-file=/workspace/.coverage --source=/workspace/nemo tests/collections/llm/conversion/test_nmh_conversion.py --conversion_type=NEMO2_TO_HF --source_ckpt=${NEMO_OUTPUT_PATH} --target_ckpt=${HF_OUTPUT_PATH}
