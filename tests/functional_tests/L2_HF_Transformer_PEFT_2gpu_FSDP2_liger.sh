export TRANSFORMERS_OFFLINE=1
export HF_HOME=/home/TestData/automodel/hf_home

coverage run -a --data-file=/workspace/.coverage --source=/workspace/nemo \
    examples/llm/finetune/automodel.py \
    --lora \
    --model /home/TestData/akoumparouli/hf_mixtral_2l/ \
    --max-steps 3 \
    --devices 2 \
    --strategy fsdp2 --liger
