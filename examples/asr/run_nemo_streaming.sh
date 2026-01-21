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
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-20}"
COMPARE_VS_OFFLINE="${COMPARE_VS_OFFLINE:-false}"  # Changed from false to true
USE_CPU="${USE_CPU:-false}"
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
    ${COMPARE_FLAG}
# Evaluate on LibriSpeech test-other
echo "Evaluating on LibriSpeech test-other..."
python run_nemo_streaming_eval.py \
    --model_path="${MODEL_PATH}" \
    --dataset_path="${DATASET_PATH}" \
    --dataset="librispeech" \
    --split="test.other" \
    --batch_size=${BATCH_SIZE} \
    --cuda=${CUDA_DEVICE} \
    --output_dir="${OUTPUT_DIR}" \
    --max_eval_samples=${MAX_EVAL_SAMPLES} \
    --chunk_size=${CHUNK_SIZE} \
    --shift_size=${SHIFT_SIZE} \
    --left_chunks=${LEFT_CHUNKS} \
    ${AMP_FLAG} \
    ${DATASET_STREAMING_FLAG} \
    ${COMPARE_FLAG}
# Evaluate on TED-LIUM
echo "Evaluating on TED-LIUM..."
python run_nemo_streaming_eval.py \
    --model_path="${MODEL_PATH}" \
    --dataset_path="${DATASET_PATH}" \
    --dataset="tedlium" \
    --split="test" \
    --batch_size=${BATCH_SIZE} \
    --cuda=${CUDA_DEVICE} \
    --output_dir="${OUTPUT_DIR}" \
    --max_eval_samples=${MAX_EVAL_SAMPLES} \
    --chunk_size=${CHUNK_SIZE} \
    --shift_size=${SHIFT_SIZE} \
    --left_chunks=${LEFT_CHUNKS} \
    ${AMP_FLAG} \
    ${DATASET_STREAMING_FLAG} \
    ${COMPARE_FLAG}
# Evaluate on GigaSpeech
echo "Evaluating on GigaSpeech..."
python run_nemo_streaming_eval.py \
    --model_path="${MODEL_PATH}" \
    --dataset_path="${DATASET_PATH}" \
    --dataset="gigaspeech" \
    --split="test" \
    --batch_size=${BATCH_SIZE} \
    --cuda=${CUDA_DEVICE} \
    --output_dir="${OUTPUT_DIR}" \
    --max_eval_samples=${MAX_EVAL_SAMPLES} \
    --chunk_size=${CHUNK_SIZE} \
    --shift_size=${SHIFT_SIZE} \
    --left_chunks=${LEFT_CHUNKS} \
    ${AMP_FLAG} \
    ${DATASET_STREAMING_FLAG} \
    ${COMPARE_FLAG}
# Evaluate on AMI
echo "Evaluating on AMI..."
python run_nemo_streaming_eval.py \
    --model_path="${MODEL_PATH}" \
    --dataset_path="${DATASET_PATH}" \
    --dataset="ami" \
    --split="test" \
    --batch_size=${BATCH_SIZE} \
    --cuda=${CUDA_DEVICE} \
    --output_dir="${OUTPUT_DIR}" \
    --max_eval_samples=${MAX_EVAL_SAMPLES} \
    --chunk_size=${CHUNK_SIZE} \
    --shift_size=${SHIFT_SIZE} \
    --left_chunks=${LEFT_CHUNKS} \
    ${AMP_FLAG} \
    ${DATASET_STREAMING_FLAG} \
    ${COMPARE_FLAG}
# Evaluate on Earnings22
echo "Evaluating on Earnings22..."
python run_nemo_streaming_eval.py \
    --model_path="${MODEL_PATH}" \
    --dataset_path="${DATASET_PATH}" \
    --dataset="earnings22" \
    --split="test" \
    --batch_size=${BATCH_SIZE} \
    --cuda=${CUDA_DEVICE} \
    --output_dir="${OUTPUT_DIR}" \
    --max_eval_samples=${MAX_EVAL_SAMPLES} \
    --chunk_size=${CHUNK_SIZE} \
    --shift_size=${SHIFT_SIZE} \
    --left_chunks=${LEFT_CHUNKS} \
    ${AMP_FLAG} \
    ${DATASET_STREAMING_FLAG} \
    ${COMPARE_FLAG}
# Evaluate on VoxPopuli
echo "Evaluating on VoxPopuli..."
python run_nemo_streaming_eval.py \
    --model_path="${MODEL_PATH}" \
    --dataset_path="${DATASET_PATH}" \
    --dataset="voxpopuli" \
    --split="test" \
    --batch_size=${BATCH_SIZE} \
    --cuda=${CUDA_DEVICE} \
    --output_dir="${OUTPUT_DIR}" \
    --max_eval_samples=${MAX_EVAL_SAMPLES} \
    --chunk_size=${CHUNK_SIZE} \
    --shift_size=${SHIFT_SIZE} \
    --left_chunks=${LEFT_CHUNKS} \
    ${AMP_FLAG} \
    ${DATASET_STREAMING_FLAG} \
    ${COMPARE_FLAG}
# Evaluate on SPGISpeech
echo "Evaluating on SPGISpeech..."
python run_nemo_streaming_eval.py \
    --model_path="${MODEL_PATH}" \
    --dataset_path="${DATASET_PATH}" \
    --dataset="spgispeech" \
    --split="test" \
    --batch_size=${BATCH_SIZE} \
    --cuda=${CUDA_DEVICE} \
    --output_dir="${OUTPUT_DIR}" \
    --max_eval_samples=${MAX_EVAL_SAMPLES} \
    --chunk_size=${CHUNK_SIZE} \
    --shift_size=${SHIFT_SIZE} \
    --left_chunks=${LEFT_CHUNKS} \
    ${AMP_FLAG} \
    ${DATASET_STREAMING_FLAG} \
    ${COMPARE_FLAG}
echo "========================================"
echo "Streaming Evaluation Complete!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "========================================"
# Aggregate results and calculate average WER
echo ""
echo "Calculating average WER across all datasets..."
echo ""
python calculate_average_wer.py "${OUTPUT_DIR}" --model_name "${MODEL_PATH}"
echo ""
echo "Individual Dataset Summaries:"
echo "-------------------"
for summary_file in ${OUTPUT_DIR}/streaming_summary_*.json; do
    if [ -f "$summary_file" ]; then
        echo "File: $(basename $summary_file)"
        cat "$summary_file"
        echo ""
    fi
done
