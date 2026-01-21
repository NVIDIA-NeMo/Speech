#!/bin/bash
#
# NeMo Streaming ASR Evaluation Script for HuggingFace Datasets
# For cache-aware streaming models
#
# Usage:
#   ./run_nemo_streaming.sh
#
# Environment variables (optional):
#   MODEL_PATH - Path to .nemo streaming model or pretrained model name
#   BATCH_SIZE - Batch size for evaluation
#   CUDA_DEVICE - CUDA device number
#   OUTPUT_DIR - Output directory for results
#   AMP - Use automatic mixed precision (true/false)
#   USE_CPU - Force CPU inference (true/false)
#   COMPARE_VS_OFFLINE - Compare streaming vs offline (true/false)

set -e  # Exit on error

# Configuration - Default to NVIDIA's pretrained streaming model
MODEL_PATH="${MODEL_PATH:-nvidia/nemotron-speech-streaming-en-0.6b}"
DATASET_PATH="${DATASET_PATH:-hf-audio/esb-datasets-test-only-sorted}"
BATCH_SIZE="${BATCH_SIZE:-1}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-./streaming_results}"
AMP="${AMP:-true}"
DATASET_STREAMING="${DATASET_STREAMING:-false}"
#MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:--1}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-50}"
COMPARE_VS_OFFLINE="${COMPARE_VS_OFFLINE:-false}"  # Changed from false to true
USE_CPU="${USE_CPU:-true}"

# Force CPU mode by hiding GPUs
CPU_FLAG=""
if [ "$USE_CPU" = "true" ]; then
    export CUDA_VISIBLE_DEVICES=""
    export TORCH_CUDA_ARCH_LIST=""
    export USE_CUDA=0
    export PYTORCH_JIT=0
    export TORCH_EXTENSIONS_DIR="/tmp/torch_cpu_extensions"
    # Disable oneDNN to avoid LSTM primitive errors
    export TORCH_USE_ONEDNN=0
    export MKLDNN_ENABLED=0
    unset CUDA_HOME
    unset CUDA_PATH
    CPU_FLAG="--cpu"
fi
CHUNK_SIZE="${CHUNK_SIZE:-2}"
SHIFT_SIZE="${SHIFT_SIZE:-2}"
LEFT_CHUNKS="${LEFT_CHUNKS:-35}"

# Convert boolean flags
AMP_FLAG=""
if [ "$AMP" = "true" ]; then
    AMP_FLAG="--amp"
fi

DATASET_STREAMING_FLAG=""
if [ "$DATASET_STREAMING" = "true" ]; then
    DATASET_STREAMING_FLAG="--streaming"
fi

COMPARE_FLAG=""
if [ "$COMPARE_VS_OFFLINE" = "true" ]; then
    COMPARE_FLAG="--compare_vs_offline"
fi

echo "========================================"
echo "NeMo Streaming ASR Evaluation"
echo "========================================"
echo "Model: ${MODEL_PATH}"
echo "Dataset Path: ${DATASET_PATH}"
echo "Batch Size: ${BATCH_SIZE}"
echo "CUDA Device: ${CUDA_DEVICE}"
echo "Output Dir: ${OUTPUT_DIR}"
echo "AMP: ${AMP}"
echo "USE_CPU: ${USE_CPU}"
echo "Compare vs Offline: ${COMPARE_VS_OFFLINE}"
echo "Chunk Size: ${CHUNK_SIZE}"
echo "Shift Size: ${SHIFT_SIZE}"
echo "Left Chunks: ${LEFT_CHUNKS}"
echo "========================================"

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Evaluate on LibriSpeech test-clean
echo "Evaluating on LibriSpeech test-clean..."
python run_nemo_streaming_eval.py \
    --model_path="${MODEL_PATH}" \
    --dataset_path="${DATASET_PATH}" \
    --dataset="librispeech" \
    --split="test.clean" \
    --batch_size=${BATCH_SIZE} \
    --cuda=${CUDA_DEVICE} \
    --output_dir="${OUTPUT_DIR}" \
    --max_eval_samples=${MAX_EVAL_SAMPLES} \
    --chunk_size=${CHUNK_SIZE} \
    --shift_size=${SHIFT_SIZE} \
    --left_chunks=${LEFT_CHUNKS} \
    ${AMP_FLAG} \
    ${DATASET_STREAMING_FLAG} \
    ${COMPARE_FLAG} \
    ${CPU_FLAG}

