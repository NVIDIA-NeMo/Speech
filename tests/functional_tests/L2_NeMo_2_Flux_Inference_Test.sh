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

TRANSFORMER_OFFLINE=1 MASTER_PORT=29500 MASTER_ADDR=127.0.0.1 LOCAL_RANK=0 HF_HOME=/home/TestData/diffusion coverage run -a --data-file=/workspace/.coverage --source=/workspace/nemo \
  /workspace/scripts/flux/flux_controlnet_infer.py \
  --t5_version google/t5-v1_1-xxl \
  --t5_load_config_only \
  --clip_version openai/clip-vit-large-patch14 \
  --vae_ckpt null \
  --flux_ckpt /home/TestData/diffusion/ckpt/transformer \
  --controlnet_ckpt /home/TestData/diffusion/ckpt/controlnet_ckpt.safetensors \
  --do_convert_from_hf \
  --control_image /home/TestData/diffusion/example-control.jpg \
  --flux_num_joint_layers 1 \
  --flux_num_single_layers 1

TRANSFORMER_OFFLINE=1 MASTER_PORT=29500 MASTER_ADDR=127.0.0.1 LOCAL_RANK=0 HF_HOME=/home/TestData/diffusion coverage run -a --data-file=/workspace/.coverage --source=/workspace/nemo \
  /workspace/scripts/flux/flux_infer.py \
  --t5_version google/t5-v1_1-xxl \
  --t5_load_config_only \
  --clip_version openai/clip-vit-large-patch14 \
  --vae_ckpt /home/TestData/diffusion/ckpt/ae.safetensors \
  --num_joint_layers 1 \
  --num_single_layers 1


