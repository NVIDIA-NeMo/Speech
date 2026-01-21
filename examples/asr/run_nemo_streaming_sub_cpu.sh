#!/bin/bash
#
# NeMo Streaming ASR Evaluation Script for HuggingFace Datasets (CPU Version)
# Uses run_nemo_streaming_eval_cpu.py with direct conformer_stream_step
#
# Usage:
#   ./run_nemo_streaming_sub_cpu.sh
#
# Environment variables (optional):
#   MODEL_PATH - Path to .nemo streaming model or pretrained model name
#   BATCH_SIZE - Batch size for evaluation
#   OUTPUT_DIR - Output directory for results

set -e  # Exit on error

# Configuration - Default to NVIDIA's pretrained streaming model
MODEL_PATH="${MODEL_PATH:-nvidia/nemotron-speech-streaming-en-0.6b}"
DATASET_PATH="${DATASET_PATH:-hf-audio/esb-datasets-test-only-sorted}"
BATCH_SIZE="${BATCH_SIZE:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-./streaming_results_cpu}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-1}"
CHUNK_SIZE="${CHUNK_SIZE:-28}"
SHIFT_SIZE="${SHIFT_SIZE:-28}"
LEFT_CHUNKS="${LEFT_CHUNKS:-2}"
MANIFEST_FILE="${MANIFEST_FILE:-${OUTPUT_DIR}/manifests/librispeech_test_clean.json}"

echo "========================================"
echo "NeMo Streaming ASR Evaluation (CPU)"
echo "========================================"
echo "Model: ${MODEL_PATH}"
echo "Dataset Path: ${DATASET_PATH}"
echo "Batch Size: ${BATCH_SIZE}"
echo "Device: CPU"
echo "Output Dir: ${OUTPUT_DIR}"
echo "Chunk Size: ${CHUNK_SIZE}"
echo "Shift Size: ${SHIFT_SIZE}"
echo "Left Chunks: ${LEFT_CHUNKS}"
echo "Manifest: ${MANIFEST_FILE}"
echo "========================================"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Evaluate on LibriSpeech test-clean
echo "Evaluating on LibriSpeech test-clean..."

if [ -f "${MANIFEST_FILE}" ]; then
    echo "Using cached manifest: ${MANIFEST_FILE}"
    python run_nemo_streaming_eval_cpu.py \
        --model_path="${MODEL_PATH}" \
        --manifest_file="${MANIFEST_FILE}" \
        --batch_size=${BATCH_SIZE} \
        --output_dir="${OUTPUT_DIR}" \
        --max_eval_samples=${MAX_EVAL_SAMPLES} \
        --chunk_size=${CHUNK_SIZE} \
        --shift_size=${SHIFT_SIZE} \
        --left_chunks=${LEFT_CHUNKS} \
        --cpu
else
    echo "No cached manifest found, downloading from HuggingFace..."
    python run_nemo_streaming_eval_cpu.py \
        --model_path="${MODEL_PATH}" \
        --dataset_path="${DATASET_PATH}" \
        --dataset="librispeech" \
        --split="test.clean" \
        --batch_size=${BATCH_SIZE} \
        --output_dir="${OUTPUT_DIR}" \
        --max_eval_samples=${MAX_EVAL_SAMPLES} \
        --chunk_size=${CHUNK_SIZE} \
        --shift_size=${SHIFT_SIZE} \
        --left_chunks=${LEFT_CHUNKS} \
        --dataset_streaming \
        --cpu
fi
