#!/bin/bash
#
# NeMo Chunked/Buffered ASR Evaluation Script for HuggingFace Datasets
# Uses asr_streaming_infer.py with Hydra config for buffered streaming
#
# Usage:
#   ./run_nemo_chunked.sh
#
# Environment variables (optional):
#   CONFIG_NAME - Config file name (default: parakeet_tdt_0.6b_v3.yaml)
#   BATCH_SIZE - Batch size for evaluation (overrides yaml)
#   CUDA_DEVICE - CUDA device number (overrides yaml)
#   OUTPUT_DIR - Output directory for results

set -e  # Exit on error

# Configuration - all model/streaming settings come from yaml
CONFIG_NAME="${CONFIG_NAME:-parakeet_tdt_0.6b_v3.yaml}"
DATASET_PATH="${DATASET_PATH:-hf-audio/esb-datasets-test-only-sorted}"
BATCH_SIZE="${BATCH_SIZE:-1}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-./chunked_results}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-50}"
# Shared audio cache directory - reuse across runs
AUDIO_CACHE_DIR="${AUDIO_CACHE_DIR:-${OUTPUT_DIR}/manifests}"

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFER_SCRIPT="${SCRIPT_DIR}/asr_streaming_inference/asr_streaming_infer.py"
CONFIG_PATH="${SCRIPT_DIR}/conf/asr_streaming_inference/"

echo "========================================"
echo "NeMo Chunked/Buffered ASR Evaluation"
echo "========================================"
echo "Config: ${CONFIG_NAME}"
echo "Dataset Path: ${DATASET_PATH}"
echo "Batch Size: ${BATCH_SIZE}"
echo "CUDA Device: ${CUDA_DEVICE}"
echo "Output Dir: ${OUTPUT_DIR}"
echo "Audio Cache Dir: ${AUDIO_CACHE_DIR}"
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
    
    manifest_entries.append({
        'audio_filepath': audio_path,
        'text': text,
        'duration': duration
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
    
    # Run inference and capture output
    python "${INFER_SCRIPT}" \
        --config-path="${CONFIG_PATH}" \
        --config-name="${CONFIG_NAME}" \
        asr.device_id=${CUDA_DEVICE} \
        streaming.batch_size=${BATCH_SIZE} \
        audio_file="${manifest_file}" \
        output_filename="${OUTPUT_DIR}/output_${output_name}.json" \
        output_dir="${OUTPUT_DIR}" 2>&1 | tee "${log_file}"
    
    # Clean up individual segment JSON files (generated per-audio by NeMo)
    find "${OUTPUT_DIR}" -maxdepth 1 -name "*.json" ! -name "output_*.json" ! -name "streaming_summary_*.json" -type f -delete 2>/dev/null || true
    
    # Extract WER and RTFX from log and create summary JSON
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

# Extract WER - look for dict format like {'wer': 0.0287, ...} or 'wer': 0.0287
wer = None
# Try to find the total_res dict with wer
wer_match = re.search(r\"'wer':\s*(\d+\.?\d*)\", log_content)
if wer_match:
    wer = float(wer_match.group(1)) * 100  # Convert to percentage

# Also try WER: X.XX% format
if wer is None:
    wer_match = re.search(r'WER[:\s]+(\d+\.?\d*)%', log_content)
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
    'streaming_wer': wer,
    'streaming_rtfx': rtfx,
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
print('=' * 50)
print('')
"
    
    echo "Results saved to: ${OUTPUT_DIR}/output_${output_name}.json"
}

# Evaluate on all datasets
run_eval "librispeech" "test.clean"

echo ""
echo "========================================"
echo "Chunked/Buffered Evaluation Complete!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "========================================"

# Aggregate results and calculate average WER
echo ""
echo "Calculating average WER across all datasets..."
echo ""
python "${SCRIPT_DIR}/calculate_average_wer.py" "${OUTPUT_DIR}" --model_name "${CONFIG_NAME}"

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
