#!/bin/bash
#
# NeMo ASR Evaluation Script for HuggingFace Datasets
# Similar to run_whisper.sh but for NeMo models
#
# Usage:
#   ./run_nemo_asr.sh
#
# Environment variables (optional):
#   MODEL_PATH - Path to .nemo model or pretrained model name
#   BATCH_SIZE - Batch size for evaluation
#   CUDA_DEVICE - CUDA device number
#   OUTPUT_DIR - Output directory for results
#   AMP - Use automatic mixed precision (true/false)
#   STREAMING - Use streaming mode (true/false)

set -e  # Exit on error

# Configuration
MODEL_PATH="${MODEL_PATH:-stt_en_conformer_ctc_large}"
DATASET_PATH="${DATASET_PATH:-hf-audio/esb-datasets-test-only-sorted}"
BATCH_SIZE="${BATCH_SIZE:-16}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-./results}"
AMP="${AMP:-true}"
STREAMING="${STREAMING:-false}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:--1}"

# Convert boolean flags
AMP_FLAG=""
if [ "$AMP" = "true" ]; then
    AMP_FLAG="--amp"
fi

STREAMING_FLAG=""
if [ "$STREAMING" = "true" ]; then
    STREAMING_FLAG="--streaming"
fi

echo "========================================"
echo "NeMo ASR Evaluation"
echo "========================================"
echo "Model: ${MODEL_PATH}"
echo "Dataset Path: ${DATASET_PATH}"
echo "Batch Size: ${BATCH_SIZE}"
echo "CUDA Device: ${CUDA_DEVICE}"
echo "Output Dir: ${OUTPUT_DIR}"
echo "AMP: ${AMP}"
echo "Streaming: ${STREAMING}"
echo "========================================"

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Evaluate on LibriSpeech test-clean
echo "Evaluating on LibriSpeech test-clean..."
python run_nemo_eval.py \
    --model_path="${MODEL_PATH}" \
    --dataset_path="${DATASET_PATH}" \
    --dataset="librispeech" \
    --split="test.clean" \
    --batch_size=${BATCH_SIZE} \
    --cuda=${CUDA_DEVICE} \
    --output_dir="${OUTPUT_DIR}" \
    --max_eval_samples=${MAX_EVAL_SAMPLES} \
    ${AMP_FLAG} \
    ${STREAMING_FLAG}

# Evaluate on LibriSpeech test-other
echo "Evaluating on LibriSpeech test-other..."
python run_nemo_eval.py \
    --model_path="${MODEL_PATH}" \
    --dataset_path="${DATASET_PATH}" \
    --dataset="librispeech" \
    --split="test.other" \
    --batch_size=${BATCH_SIZE} \
    --cuda=${CUDA_DEVICE} \
    --output_dir="${OUTPUT_DIR}" \
    --max_eval_samples=${MAX_EVAL_SAMPLES} \
    ${AMP_FLAG} \
    ${STREAMING_FLAG}

# Evaluate on TED-LIUM
echo "Evaluating on TED-LIUM..."
python run_nemo_eval.py \
    --model_path="${MODEL_PATH}" \
    --dataset_path="${DATASET_PATH}" \
    --dataset="tedlium" \
    --split="test" \
    --batch_size=${BATCH_SIZE} \
    --cuda=${CUDA_DEVICE} \
    --output_dir="${OUTPUT_DIR}" \
    --max_eval_samples=${MAX_EVAL_SAMPLES} \
    ${AMP_FLAG} \
    ${STREAMING_FLAG}

# Evaluate on GigaSpeech
echo "Evaluating on GigaSpeech..."
python run_nemo_eval.py \
    --model_path="${MODEL_PATH}" \
    --dataset_path="${DATASET_PATH}" \
    --dataset="gigaspeech" \
    --split="test" \
    --batch_size=${BATCH_SIZE} \
    --cuda=${CUDA_DEVICE} \
    --output_dir="${OUTPUT_DIR}" \
    --max_eval_samples=${MAX_EVAL_SAMPLES} \
    ${AMP_FLAG} \
    ${STREAMING_FLAG}

# Evaluate on AMI
echo "Evaluating on AMI..."
python run_nemo_eval.py \
    --model_path="${MODEL_PATH}" \
    --dataset_path="${DATASET_PATH}" \
    --dataset="ami" \
    --split="test" \
    --batch_size=${BATCH_SIZE} \
    --cuda=${CUDA_DEVICE} \
    --output_dir="${OUTPUT_DIR}" \
    --max_eval_samples=${MAX_EVAL_SAMPLES} \
    ${AMP_FLAG} \
    ${STREAMING_FLAG}

# Evaluate on Earnings22
echo "Evaluating on Earnings22..."
python run_nemo_eval.py \
    --model_path="${MODEL_PATH}" \
    --dataset_path="${DATASET_PATH}" \
    --dataset="earnings22" \
    --split="test" \
    --batch_size=${BATCH_SIZE} \
    --cuda=${CUDA_DEVICE} \
    --output_dir="${OUTPUT_DIR}" \
    --max_eval_samples=${MAX_EVAL_SAMPLES} \
    ${AMP_FLAG} \
    ${STREAMING_FLAG}

# Evaluate on VoxPopuli
echo "Evaluating on VoxPopuli..."
python run_nemo_eval.py \
    --model_path="${MODEL_PATH}" \
    --dataset_path="${DATASET_PATH}" \
    --dataset="voxpopuli" \
    --split="test" \
    --batch_size=${BATCH_SIZE} \
    --cuda=${CUDA_DEVICE} \
    --output_dir="${OUTPUT_DIR}" \
    --max_eval_samples=${MAX_EVAL_SAMPLES} \
    ${AMP_FLAG} \
    ${STREAMING_FLAG}

# Evaluate on SPGISpeech
echo "Evaluating on SPGISpeech..."
python run_nemo_eval.py \
    --model_path="${MODEL_PATH}" \
    --dataset_path="${DATASET_PATH}" \
    --dataset="spgispeech" \
    --split="test" \
    --batch_size=${BATCH_SIZE} \
    --cuda=${CUDA_DEVICE} \
    --output_dir="${OUTPUT_DIR}" \
    --max_eval_samples=${MAX_EVAL_SAMPLES} \
    ${AMP_FLAG} \
    ${STREAMING_FLAG}

echo "========================================"
echo "Evaluation Complete!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "========================================"

# Aggregate results
echo ""
echo "Summary of Results:"
echo "-------------------"
for summary_file in ${OUTPUT_DIR}/summary_*.json; do
    if [ -f "$summary_file" ]; then
        echo "File: $(basename $summary_file)"
        cat "$summary_file"
        echo ""
    fi
done
