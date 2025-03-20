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
# Parse passed arguments
for i in "$@"; do
  case $i in
    HF_ORI_PATH=*)
      HF_ORI_PATH="${i#*=}"
      ;;
    NEMO_MODEL_TYPE=*)
      NEMO_MODEL_TYPE="${i#*=}"
      ;;
    NEMO_MODEL_CONFIG=*)
      NEMO_MODEL_CONFIG="${i#*=}"
      ;;
    NEMO_OUTPUT_PATH=*)
      NEMO_OUTPUT_PATH="${i#*=}"
      ;;
    HF_OUTPUT_PATH=*)
      HF_OUTPUT_PATH="${i#*=}"
      ;;
    add-model-name)
      ADD_MODEL_NAME="--add-model-name"
      ;;
    *)
      # Handle other arguments or errors here
      echo "Unknown argument: $i"
      exit 1
      ;;
  esac
done

coverage run --branch -a --data-file=/workspace/.coverage --source=/workspace/nemo tests/collections/llm/conversion/test_import_from_hf.py --hf-path=${HF_ORI_PATH} --model-type=${NEMO_MODEL_TYPE} --model-config=${NEMO_MODEL_CONFIG} --output-path=${NEMO_OUTPUT_PATH}
coverage run --branch -a --data-file=/workspace/.coverage --source=/workspace/nemo tests/collections/llm/conversion/test_export_to_hf.py --nemo-path=${NEMO_OUTPUT_PATH} --original-hf-path=${HF_ORI_PATH} --output-path=${HF_OUTPUT_PATH} $ADD_MODEL_NAME
