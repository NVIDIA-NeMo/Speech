#!/bin/bash
UUID=$(cat /proc/sys/kernel/random/uuid)

mkdir /tmp/${UUID}
export PYTHONPATH=/home/TestData/multimodal/video_neva/LLaVA:$PYTHONPATH
CUDA_VISIBLE_DEVICES=0 python examples/multimodal/multimodal_llm/neva/convert_llava_to_neva.py \
    --in-file /home/TestData/multimodal/video_neva/Llama-3-VILA1.5-8B/llm \
    --mm-projector-ckpt-dir /home/TestData/multimodal/video_neva/Llama-3-VILA1.5-8B/mm_projector \
    --mm-vision-tower /home/TestData/multimodal/video_neva/Llama-3-VILA1.5-8B/vision_tower \
    --tokenizer-model /home/TestData/multimodal/video_neva/vita-tokenizer/ \
    --config-file vita_config.yaml \
    --out-file=/tmp/${UUID}/llama3_ci.nemo \
    --model-type VITA \
    --conv-template llama_3
