TRANSFORMERS_OFFLINE=1 HF_HOME=/home/TestData/automodel/hf_home \
coverage run -a --data-file=/workspace/.coverage --source=/workspace/nemo \
    examples/llm/finetune/automodel.py \
    --model /home/TestData/nlp/hf_gemma/hf_gemma_2b \
    --max-steps 3 \
    --devices 2 \
    --strategy fsdp2 --fp8 --mock-dataset
