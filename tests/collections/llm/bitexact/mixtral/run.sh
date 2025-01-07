#!/bin/bash
set -ex

DATA_PATH="/home/TestData/nlp/megatron_t5/data/pile_val_small_bert_tokenizer_text_document"
VOCAB_FILE="/home/TestData/nlp/gpt2_tokenizer/vocab.json"
MERGE_FILE="/home/TestData/nlp/gpt2_tokenizer/merges.txt"
MCORE_OUTPUT_PATH="/tmp/bex_mixtral_mcore_output/"
NEMO_OUTPUT_PATH="/tmp/bex_mixtral_nemo_output/"

# Run Mcore
CUDA_DEVICE_MAX_CONNECTIONS=1 CUDA_LAUNCH_BLOCKING=1 TORCH_COMPILE_DISABLE=1 \
torchrun --nproc-per-node 1 --nnodes 1 /workspace/Megatron-LM/pretrain_gpt.py \
    --apply-layernorm-1p --rotary-percent 1.0 --rotary-base 1000000 \
    --no-position-embedding --position-embedding-type rope \
    --swiglu \
    --vocab-file "$VOCAB_FILE" \
    --merge-file "$MERGE_FILE" \
    --num-layers 2 \
    --hidden-size 32 --num-attention-heads 4 --seq-length 4096 --max-position-embeddings 4096 \
    --untie-embeddings-and-output-weights --disable-bias-linear --attention-dropout 0.0 \
    --hidden-dropout 0.0 --micro-batch-size 1 --global-batch-size 2 --clip-grad 0.0 \
    --lr 1e-2 --min-lr 1e-4 --lr-decay-style constant --weight-decay 0 \
    --adam-beta1 0.9 --adam-beta2 0.9 \
    --init-method-std 0.008 --bf16 --use-mcore-models --transformer-impl transformer_engine \
    --use-distributed-optimizer --train-iters=10 --dataloader-type single --use-dist-ckpt \
    --dist-ckpt-format=torch_dist \
    --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 \
    --no-gradient-accumulation-fusion \
    --data-path "$DATA_PATH" \
    --split 99,1,0 --log-interval 10 --save-interval 20000 --eval-interval 1000 --eval-iters 32 \
    --save "$MCORE_OUTPUT_PATH" \
    --log-num-zeros-in-grad --distributed-timeout-minutes 6000 --moe-router-topk 1 --num-experts 2 \
    --moe-router-pre-softmax --expert-model-parallel-size 1 --eval-iters=0 --attention-backend unfused

# Run NeMo
CUDA_LAUNCH_BLOCKING=1 TORCH_COMPILE_DISABLE=1 NVTE_FLASH_ATTN=0 NVTE_FUSED_ATTN=0 \
python3 /workspace/tests/collections/llm/bitexact/mixtral/pretrain_mini_mixtral.py \
    --devices=1 \
    --data-path="$DATA_PATH" \
    --vocab-path="$VOCAB_FILE" \
    --merges-path="$MERGE_FILE" \
    --exp-dir="$NEMO_OUTPUT_PATH"

# Compare outputs
python3 /workspace/tests/collections/llm/bitexact/mixtral/compare_ckpts.py \
  "$NEMO_OUTPUT_PATH/checkpoints/--None=0.0000-epoch=0-consumed_samples=20.0/weights" "$MCORE_OUTPUT_PATH/iter_0000010/"
