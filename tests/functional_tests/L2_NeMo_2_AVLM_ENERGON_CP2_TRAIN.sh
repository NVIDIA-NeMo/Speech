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

TRANSFORMERS_OFFLINE=1 HF_HOME=/home/TestData/avlm/hf_home coverage run -a --data-file=/workspace/.coverage --source=/workspace/nemo \
    scripts/avlm/avlm_pretrain.py \
    --devices=2 \
    --max_steps=3 \
    --log_dir=/tmp/nemo2_avlm_energon_cp2_results/$RUN_ID \
    --data_type=energon \
    --data_path=/home/TestData/avlm/train/data/sample_data/wds \
    --seq_length=8192 \
    --cp_size=2 \
    --use_packed_sequence=false \
    --gbs=2 \
    --mbs=2 \
    --lr=0.001
