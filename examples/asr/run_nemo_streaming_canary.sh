#!/bin/bash
#
# NeMo Streaming AED (Canary) ASR Evaluation Script for HuggingFace Datasets
# Uses speech_to_text_aed_streaming_infer.py with overlapping chunks
# For short chunks (<10s) with left/right context padding
#
# Usage:
#   ./run_nemo_streaming_canary.sh
#
# Environment variables (optional):
#   CONFIG_NAME - Config file name (default: canary_1b_v2_streaming.yaml)
#   CHUNK_SECS - Chunk size in seconds (default: 1.0)
#   LEFT_CONTEXT_SECS - Left context padding (default: 10.0)
#   RIGHT_CONTEXT_SECS - Right context padding (default: 0.5)
#   STREAMING_POLICY - Streaming policy: alignatt or waitk (default: alignatt)
#   BATCH_SIZE - Batch size for evaluation (default: 32)
#   CUDA_DEVICE - CUDA device number (default: 0)
#   OUTPUT_DIR - Output directory for results

set -e  # Exit on error

# Use local NeMo installation
export PYTHONPATH="/datadisks/disk1/jiafa/accuracy/NeMo:${PYTHONPATH}"

# Configuration - streaming settings
CONFIG_NAME="${CONFIG_NAME:-canary_1b_streaming.yaml}"
STREAMING_POLICY="${STREAMING_POLICY:-alignatt}"

# Dataset and hardware settings
DATASET_PATH="${DATASET_PATH:-hf-audio/esb-datasets-test-only-sorted}"
BATCH_SIZE="${BATCH_SIZE:-16}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-./streaming_canary_results}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:--1}"
AUDIO_CACHE_DIR="${AUDIO_CACHE_DIR:-${OUTPUT_DIR}/manifests}"

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFER_SCRIPT="${SCRIPT_DIR}/asr_chunked_inference/aed/speech_to_text_aed_streaming_infer.py"
CONFIG_PATH="${SCRIPT_DIR}/conf/asr_streaming_inference/"

echo "========================================"
echo "NeMo Streaming AED (Canary) ASR Evaluation"
echo "========================================"
echo "Config: ${CONFIG_NAME}"
echo "Chunk Size: ${CHUNK_SECS}s"
echo "Left Context: ${LEFT_CONTEXT_SECS}s"
echo "Right Context: ${RIGHT_CONTEXT_SECS}s"
echo "Streaming Policy: ${STREAMING_POLICY}"
echo "Dataset Path: ${DATASET_PATH}"
echo "Batch Size: ${BATCH_SIZE}"
echo "CUDA Device: ${CUDA_DEVICE}"
echo "Output Dir: ${OUTPUT_DIR}"
echo "========================================"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Function to create manifest from HuggingFace dataset
create_manifest() {
    local dataset=$1
    local split=$2
    local manifest_dir="${AUDIO_CACHE_DIR}"
    local manifest_file="${manifest_dir}/${dataset}_${split//./_}.json"
    
    mkdir -p "${manifest_dir}"
    
    if [ -f "${manifest_file}" ]; then
        echo "Manifest already exists: ${manifest_file}" >&2
        echo "${manifest_file}"
        return
    fi
    
    echo "Creating manifest for ${dataset}/${split}..." >&2
    
    python -c "
import os
import sys
import json
import io
import soundfile as sf
import numpy as np
from datasets import load_dataset, Audio
from tqdm import tqdm

dataset_path = '${DATASET_PATH}'
dataset_name = '${dataset}'
split = '${split}'
manifest_file = '${manifest_file}'
cache_dir = '${manifest_dir}/audio_cache/${dataset}_${split//./_}'
max_samples = ${MAX_EVAL_SAMPLES}

os.makedirs(cache_dir, exist_ok=True)

print(f'Loading dataset: {dataset_path}/{dataset_name} split={split}', file=sys.stderr)
ds = load_dataset(dataset_path, dataset_name, split=split, streaming=True, token=True)
ds = ds.cast_column('audio', Audio(sampling_rate=16000))

manifest_entries = []
for idx, sample in enumerate(tqdm(ds, desc='Processing samples', file=sys.stderr)):
    if max_samples > 0 and idx >= max_samples:
        break
    
    audio = sample['audio']
    audio_array = audio['array']
    sr = audio['sampling_rate']
    
    # Get reference text
    text = sample.get('text', sample.get('sentence', sample.get('normalized_text', sample.get('transcript', ''))))
    
    # Save audio file
    audio_id = sample.get('id', str(idx)).replace('/', '_').removesuffix('.wav')
    audio_path = os.path.join(cache_dir, f'{audio_id}.wav')
    
    if not os.path.exists(audio_path):
        sf.write(audio_path, audio_array.astype(np.float32), sr)
    
    duration = len(audio_array) / sr
    
    # Canary models support source_lang/target_lang for ASR/translation
    manifest_entries.append({
        'audio_filepath': audio_path,
        'text': text,
        'duration': duration,
        'taskname': 'asr',
        'source_lang': 'en',
        'target_lang': 'en'
    })

# Write manifest
with open(manifest_file, 'w') as f:
    for entry in manifest_entries:
        f.write(json.dumps(entry) + '\n')

print(f'Manifest saved to: {manifest_file}', file=sys.stderr)
print(f'Total samples: {len(manifest_entries)}', file=sys.stderr)
"
    
    echo "${manifest_file}"
}

# Function to run evaluation on a dataset
run_eval() {
    local dataset=$1
    local split=$2
    local output_name="${dataset}_${split//./_}"
    local log_file="${OUTPUT_DIR}/log_${output_name}.txt"
    
    echo ""
    echo "========================================"
    echo "Evaluating on ${dataset}/${split}..."
    echo "========================================"
    
    # Create manifest
    manifest_file=$(create_manifest "${dataset}" "${split}")
    
    # Run inference with AED streaming script
    python "${INFER_SCRIPT}" \
        --config-path="${CONFIG_PATH}" \
        --config-name="${CONFIG_NAME}" \
        cuda=${CUDA_DEVICE} \
        batch_size=${BATCH_SIZE} \
        decoding.streaming_policy=${STREAMING_POLICY} \
        dataset_manifest="${manifest_file}" \
        output_filename="${OUTPUT_DIR}/output_${output_name}.json" 2>&1 | tee "${log_file}"
    
    # Extract WER and create summary JSON
    python -c "
import re
import json
import os

log_file = '${log_file}'
output_file = '${OUTPUT_DIR}/output_${output_name}.json'
manifest_file = '${manifest_file}'
summary_file = '${OUTPUT_DIR}/streaming_summary_${output_name}.json'
dataset = '${dataset}'
split = '${split}'

# Read log file
with open(log_file, 'r') as f:
    log_content = f.read()

# Extract WER - look for various formats
wer = None
# Try to find WER in percentage format
wer_match = re.search(r'WER[:\s]+(\d+\.?\d*)%', log_content)
if wer_match:
    wer = float(wer_match.group(1))

# Try to find WER as decimal
if wer is None:
    wer_match = re.search(r\"'wer':\s*(\d+\.?\d*)\", log_content)
    if wer_match:
        wer = float(wer_match.group(1)) * 100  # Convert to percentage

# Try another common format
if wer is None:
    wer_match = re.search(r'Word Error Rate[:\s]+(\d+\.?\d*)', log_content)
    if wer_match:
        wer = float(wer_match.group(1))

# Extract RTFX
rtfx = None
rtfx_match = re.search(r'RTFX[:\s]+(\d+\.?\d*)', log_content)
if rtfx_match:
    rtfx = float(rtfx_match.group(1))

# Count samples from output file first, then manifest as fallback
num_samples = 0
if os.path.exists(output_file):
    with open(output_file, 'r') as f:
        for line in f:
            if line.strip():
                num_samples += 1

# Fallback to manifest count
if num_samples == 0 and os.path.exists(manifest_file):
    with open(manifest_file, 'r') as f:
        for line in f:
            if line.strip():
                num_samples += 1

# Create summary
summary = {
    'dataset': dataset,
    'split': split,
    'num_samples': num_samples,
    'wer': wer,
    'rtfx': rtfx,
    'streaming_policy': '${STREAMING_POLICY}',
}

with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2)

# Print metrics prominently
print('')
print('=' * 50)
print(f'  {dataset}/{split} RESULTS')
print('=' * 50)
print(f'  Samples:  {num_samples}')
print(f'  WER:      {wer:.2f}%' if wer else '  WER:      N/A')
print(f'  RTFX:     {rtfx:.2f}x' if rtfx else '  RTFX:     N/A')
print(f'  Chunk:    ${CHUNK_SECS}s, Left: ${LEFT_CONTEXT_SECS}s, Right: ${RIGHT_CONTEXT_SECS}s')
print('=' * 50)
print('')
"
    
    echo "Results saved to: ${OUTPUT_DIR}/output_${output_name}.json"
}

# Evaluate on all datasets
run_eval "librispeech" "test.clean"
run_eval "librispeech" "test.other"
run_eval "tedlium" "test"
run_eval "gigaspeech" "test"
run_eval "ami" "test"
run_eval "earnings22" "test"
run_eval "voxpopuli" "test"
run_eval "spgispeech" "test"

echo ""
echo "========================================"
echo "Streaming AED (Canary) Evaluation Complete!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "========================================"

# Aggregate results and calculate average WER
echo ""
echo "Calculating average WER across all datasets..."
echo ""
if [ -f "${SCRIPT_DIR}/calculate_average_wer.py" ]; then
    python "${SCRIPT_DIR}/calculate_average_wer.py" "${OUTPUT_DIR}" --model_name "${CONFIG_NAME}"
fi

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
