#!/bin/bash
#
# ASR Evaluation Script for NeMo Models
# Evaluates ASR models on multiple benchmark datasets
#
# Usage:
#   ./transcribe_eval.sh [model_path] [output_dir] [mode]
#
# Arguments:
#   model_path  - Path to .nemo model or pretrained model name (default: stt_en_conformer_ctc_large)
#   output_dir  - Output directory for results (default: ./results)
#   mode        - Evaluation mode: "standard", "streaming", or "parallel" (default: standard)
#
# Examples:
#   ./transcribe_eval.sh /path/to/model.nemo ./results standard
#   ./transcribe_eval.sh stt_en_conformer_ctc_large ./results parallel
#   ./transcribe_eval.sh /path/to/streaming_model.nemo ./results streaming

set -e  # Exit on error

# Default configuration
MODEL_PATH="${1:-stt_en_conformer_ctc_large}"
OUTPUT_DIR="${2:-./results}"
EVAL_MODE="${3:-standard}"  # standard, streaming, or parallel
BATCH_SIZE="${BATCH_SIZE:-32}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
AMP="${AMP:-true}"
NUM_WORKERS="${NUM_WORKERS:-4}"

# Dataset configurations
# You can customize these arrays to include/exclude datasets
DATASETS=(
    "librispeech"
    "tedlium"
    "gigaspeech"
    "ami"
    "earnings22"
    "voxpopuli"
    "spgispeech"
)

# Manifest files for each dataset (update these paths to your actual manifest locations)
declare -A MANIFEST_PATHS
MANIFEST_PATHS["librispeech_clean"]="/path/to/librispeech/test_clean.json"
MANIFEST_PATHS["librispeech_other"]="/path/to/librispeech/test_other.json"
MANIFEST_PATHS["tedlium"]="/path/to/tedlium/test.json"
MANIFEST_PATHS["gigaspeech"]="/path/to/gigaspeech/test.json"
MANIFEST_PATHS["ami"]="/path/to/ami/test.json"
MANIFEST_PATHS["earnings22"]="/path/to/earnings22/test.json"
MANIFEST_PATHS["voxpopuli"]="/path/to/voxpopuli/test.json"
MANIFEST_PATHS["spgispeech"]="/path/to/spgispeech/test.json"

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Log file
LOG_FILE="${OUTPUT_DIR}/evaluation_log_$(date +%Y%m%d_%H%M%S).txt"
echo "Starting ASR Evaluation" | tee ${LOG_FILE}
echo "Model: ${MODEL_PATH}" | tee -a ${LOG_FILE}
echo "Output Directory: ${OUTPUT_DIR}" | tee -a ${LOG_FILE}
echo "Evaluation Mode: ${EVAL_MODE}" | tee -a ${LOG_FILE}
echo "Batch Size: ${BATCH_SIZE}" | tee -a ${LOG_FILE}
echo "CUDA Device: ${CUDA_DEVICE}" | tee -a ${LOG_FILE}
echo "======================================" | tee -a ${LOG_FILE}

# Function to run standard transcription
run_standard_eval() {
    local dataset=$1
    local manifest=$2
    local output_path="${OUTPUT_DIR}/${dataset}"
    
    echo "Evaluating ${dataset} with transcribe_speech.py..." | tee -a ${LOG_FILE}
    
    python transcribe_speech.py \
        model_path="${MODEL_PATH}" \
        dataset_manifest="${manifest}" \
        output_filename="${output_path}_predictions.json" \
        batch_size=${BATCH_SIZE} \
        cuda=${CUDA_DEVICE} \
        amp=${AMP} \
        num_workers=${NUM_WORKERS} \
        calculate_wer=true \
        calculate_rtfx=true \
        overwrite_transcripts=true \
        2>&1 | tee -a ${LOG_FILE}
}

# Function to run streaming evaluation
run_streaming_eval() {
    local dataset=$1
    local manifest=$2
    local output_path="${OUTPUT_DIR}/${dataset}"
    
    echo "Evaluating ${dataset} with transcribe_streaming.py..." | tee -a ${LOG_FILE}
    
    python transcribe_streaming.py \
        model_path="${MODEL_PATH}" \
        dataset_manifest="${manifest}" \
        output_path="${output_path}" \
        batch_size=${BATCH_SIZE} \
        cuda=${CUDA_DEVICE} \
        amp=${AMP} \
        calculate_wer=true \
        calculate_rtfx=true \
        compare_vs_offline=false \
        debug_mode=false \
        2>&1 | tee -a ${LOG_FILE}
}

# Function to run parallel evaluation (multi-GPU)
run_parallel_eval() {
    local dataset=$1
    local manifest=$2
    local output_path="${OUTPUT_DIR}/${dataset}"
    
    echo "Evaluating ${dataset} with transcribe_speech_parallel.py..." | tee -a ${LOG_FILE}
    
    python transcribe_speech_parallel.py \
        model="${MODEL_PATH}" \
        predict_ds.manifest_filepath="${manifest}" \
        predict_ds.batch_size=${BATCH_SIZE} \
        output_path="${output_path}" \
        use_cer=false \
        trainer.devices=1 \
        trainer.accelerator=gpu \
        2>&1 | tee -a ${LOG_FILE}
}

# Main evaluation loop
echo "Starting evaluation on datasets..." | tee -a ${LOG_FILE}

# Evaluate LibriSpeech (both clean and other)
if [[ -f "${MANIFEST_PATHS[librispeech_clean]}" ]]; then
    case ${EVAL_MODE} in
        standard)
            run_standard_eval "librispeech_clean" "${MANIFEST_PATHS[librispeech_clean]}"
            ;;
        streaming)
            run_streaming_eval "librispeech_clean" "${MANIFEST_PATHS[librispeech_clean]}"
            ;;
        parallel)
            run_parallel_eval "librispeech_clean" "${MANIFEST_PATHS[librispeech_clean]}"
            ;;
    esac
fi

if [[ -f "${MANIFEST_PATHS[librispeech_other]}" ]]; then
    case ${EVAL_MODE} in
        standard)
            run_standard_eval "librispeech_other" "${MANIFEST_PATHS[librispeech_other]}"
            ;;
        streaming)
            run_streaming_eval "librispeech_other" "${MANIFEST_PATHS[librispeech_other]}"
            ;;
        parallel)
            run_parallel_eval "librispeech_other" "${MANIFEST_PATHS[librispeech_other]}"
            ;;
    esac
fi

# Evaluate other datasets
for dataset in "${DATASETS[@]}"; do
    manifest="${MANIFEST_PATHS[${dataset}]}"
    
    if [[ -f "${manifest}" ]]; then
        case ${EVAL_MODE} in
            standard)
                run_standard_eval "${dataset}" "${manifest}"
                ;;
            streaming)
                run_streaming_eval "${dataset}" "${manifest}"
                ;;
            parallel)
                run_parallel_eval "${dataset}" "${manifest}"
                ;;
        esac
    else
        echo "Warning: Manifest not found for ${dataset}: ${manifest}" | tee -a ${LOG_FILE}
    fi
done

# Aggregate results
echo "======================================" | tee -a ${LOG_FILE}
echo "Aggregating results..." | tee -a ${LOG_FILE}

SUMMARY_FILE="${OUTPUT_DIR}/summary_$(date +%Y%m%d_%H%M%S).txt"
echo "ASR Evaluation Summary" > ${SUMMARY_FILE}
echo "Model: ${MODEL_PATH}" >> ${SUMMARY_FILE}
echo "Evaluation Mode: ${EVAL_MODE}" >> ${SUMMARY_FILE}
echo "Date: $(date)" >> ${SUMMARY_FILE}
echo "======================================" >> ${SUMMARY_FILE}
echo "" >> ${SUMMARY_FILE}

# Extract WER/CER from log file
echo "Dataset Results:" >> ${SUMMARY_FILE}
grep -E "(WER|CER|RTFx)" ${LOG_FILE} >> ${SUMMARY_FILE} || echo "No metrics found in log" >> ${SUMMARY_FILE}

echo "" >> ${SUMMARY_FILE}
echo "Detailed logs: ${LOG_FILE}" >> ${SUMMARY_FILE}
echo "Output directory: ${OUTPUT_DIR}" >> ${SUMMARY_FILE}

cat ${SUMMARY_FILE} | tee -a ${LOG_FILE}

echo "======================================" | tee -a ${LOG_FILE}
echo "Evaluation complete!" | tee -a ${LOG_FILE}
echo "Results saved to: ${OUTPUT_DIR}" | tee -a ${LOG_FILE}
echo "Summary: ${SUMMARY_FILE}" | tee -a ${LOG_FILE}
echo "Log: ${LOG_FILE}" | tee -a ${LOG_FILE}
