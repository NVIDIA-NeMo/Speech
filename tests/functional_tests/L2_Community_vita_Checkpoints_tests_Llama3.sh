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
#!/bin/bash
UUID=$(cat /proc/sys/kernel/random/uuid)

mkdir /tmp/${UUID}
export PYTHONPATH=/home/TestData/multimodal/video_neva/LLaVA:$PYTHONPATH
CUDA_VISIBLE_DEVICES=0 coverage run --branch -a --data-file=/workspace/.coverage --source=/workspace/nemo examples/multimodal/multimodal_llm/neva/convert_llava_to_neva.py \
    --in-file /home/TestData/multimodal/video_neva/Llama-3-VILA1.5-8B/llm \
    --mm-projector-ckpt-dir /home/TestData/multimodal/video_neva/Llama-3-VILA1.5-8B/mm_projector \
    --mm-vision-tower /home/TestData/multimodal/video_neva/Llama-3-VILA1.5-8B/vision_tower \
    --tokenizer-model /home/TestData/multimodal/video_neva/vita-tokenizer/ \
    --config-file vita_config.yaml \
    --out-file=/tmp/${UUID}/llama3_ci.nemo \
    --model-type VITA \
    --conv-template llama_3
