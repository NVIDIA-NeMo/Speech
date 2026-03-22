#!/usr/bin/env bash
#
# Run simulstream latency, quality, and stats evaluation on existing inference output.
# Expects the output directory from run_simulstream_inference.sh (with segments-manifest)
# to contain audio_definitions.yaml, references.txt, transcripts.txt, simulstream_output.json.
#
# Usage:
#   ./run_simulstream_scores.sh \
#     output-dir=/path/to/run_output_dir \
#     [simulstream-config=/path/to/simulstream/config/nemo_cascade.yaml] \
#     [latency-unit=word] [sacrebleu-tokenizer=13a]
#
# If HF_TOKEN is set, also runs COMET quality scoring.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR=""
SIMULSTREAM_CONFIG=""
LATENCY_UNIT="word"
SACREBLEU_TOKENIZER="13a"
SEGMENTS_MANIFEST=""

usage() {
  echo "Usage: $0 output-dir=DIR [OPTIONS]"
  echo ""
  echo "Required:"
  echo "  output-dir=DIR           Run output directory (contains simulstream_output.json, etc.)"
  echo "Optional:"
  echo "  simulstream-config=YAML Simulstream eval config (auto-detected from output-dir if omitted)"
  echo "  latency-unit=UNIT         Latency unit (default: word)"
  echo "  sacrebleu-tokenizer=TOK  Tokenizer for sacrebleu (default: 13a)"
  echo "  segments-manifest=PATH   Segments manifest used to auto-create missing eval files"
  echo ""
  echo "COMET: set HF_TOKEN to run COMET quality evaluation in addition to sacrebleu."
  exit 1
}

for arg in "$@"; do
  case "$arg" in
    output-dir=*)          OUTPUT_DIR="${arg#*=}" ;;
    simulstream-config=*) SIMULSTREAM_CONFIG="${arg#*=}" ;;
    latency-unit=*)        LATENCY_UNIT="${arg#*=}" ;;
    sacrebleu-tokenizer=*) SACREBLEU_TOKENIZER="${arg#*=}" ;;
    segments-manifest=*)   SEGMENTS_MANIFEST="${arg#*=}" ;;
    -h|--help|help=true)  usage ;;
    *=*)                  echo "Unknown option: $arg"; usage ;;
    *)                    echo "Invalid argument format (expected key=value): $arg"; usage ;;
  esac
done

[[ -z "$OUTPUT_DIR" ]] && echo "Error: missing required argument: output-dir=DIR" && usage

OUTPUT_DIR_ABS="$(realpath "$OUTPUT_DIR")"
AUDIO_DEF="$OUTPUT_DIR_ABS/audio_definitions.yaml"
REFS="$OUTPUT_DIR_ABS/references.txt"
TRANSCRIPTS="$OUTPUT_DIR_ABS/transcripts.txt"
LOG_FILE="$OUTPUT_DIR_ABS/simulstream_output.json"

if [[ ! -f "$AUDIO_DEF" || ! -f "$REFS" || ! -f "$TRANSCRIPTS" ]]; then
  if [[ -z "$SEGMENTS_MANIFEST" ]]; then
    echo "Error: Missing eval files (audio_definitions.yaml/references.txt/transcripts.txt)."
    echo "Provide segments-manifest=PATH so they can be generated automatically."
    exit 1
  fi
  if [[ ! -f "$SEGMENTS_MANIFEST" ]]; then
    echo "Error: segments manifest not found: $SEGMENTS_MANIFEST"
    exit 1
  fi
  echo "Missing eval files detected. Generating from segments manifest..."
  python "$SCRIPT_DIR/create_audio_definitions_from_manifest.py" \
    --manifest "$(realpath "$SEGMENTS_MANIFEST")" \
    --output-dir "$OUTPUT_DIR_ABS"
fi

for f in "$AUDIO_DEF" "$REFS" "$TRANSCRIPTS" "$LOG_FILE"; do
  if [[ ! -f "$f" ]]; then
    echo "Error: required file not found: $f"
    echo "Run inference first to create simulstream_output.json."
    exit 1
  fi
done

if [[ -n "$SIMULSTREAM_CONFIG" ]]; then
  if [[ ! -f "$SIMULSTREAM_CONFIG" ]]; then
    echo "Error: simulstream config not found: $SIMULSTREAM_CONFIG"
    exit 1
  fi
  SIMULSTREAM_CONFIG="$(realpath "$SIMULSTREAM_CONFIG")"
else
  candidates=()
  shopt -s nullglob
  candidates=("${LOG_FILE%/*}"/*_simulstream.yaml)
  shopt -u nullglob
  if [[ ${#candidates[@]} -eq 0 ]]; then
    echo "Error: Could not auto-detect simulstream config in ${LOG_FILE%/*}"
    echo "Expected a file like *_simulstream.yaml or pass simulstream-config=PATH"
    exit 1
  fi
  if [[ ${#candidates[@]} -gt 1 ]]; then
    echo "Error: Multiple *_simulstream.yaml files found in ${LOG_FILE%/*}; pass simulstream-config=PATH explicitly."
    printf '  - %s\n' "${candidates[@]}"
    exit 1
  fi
  SIMULSTREAM_CONFIG="$(realpath "${candidates[0]}")"
fi
echo "Using simulstream config: $SIMULSTREAM_CONFIG"

SCORES_DIR="$OUTPUT_DIR_ABS/simulstream"
mkdir -p "$SCORES_DIR"

echo "========== Run simulstream eval (latency / quality / stats) =========="

echo "Running simulstream_score_latency..."
PYTHONUNBUFFERED=1  simulstream_score_latency \
  --scorer stream_laal \
  --eval-config "$SIMULSTREAM_CONFIG" \
  --log-file "$LOG_FILE" \
  --reference "$REFS" \
  --audio-definition "$AUDIO_DEF" \
  --latency-unit "$LATENCY_UNIT" > "$SCORES_DIR/simulstream_score_latency.log" 2>&1

echo "Running simulstream_score_quality (sacrebleu)..."
PYTHONUNBUFFERED=1 simulstream_score_quality \
  --scorer sacrebleu \
  --tokenizer "$SACREBLEU_TOKENIZER" \
  --eval-config "$SIMULSTREAM_CONFIG" \
  --log-file "$LOG_FILE" \
  --references "$REFS" \
  --transcripts "$TRANSCRIPTS" \
  --audio-definition "$AUDIO_DEF" \
  --latency-unit "$LATENCY_UNIT" \
  > "$SCORES_DIR/simulstream_score_quality.log" 2>&1

if [[ -n "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN detected. Logging into Hugging Face..."
  huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential

  echo "Running simulstream_score_quality (COMET Unbabel/XCOMET-XL)..."
  PYTHONUNBUFFERED=1 simulstream_score_quality \
    --scorer comet \
    --model "Unbabel/XCOMET-XL" \
    --batch-size "8" \
    --eval-config "$SIMULSTREAM_CONFIG" \
    --log-file "$LOG_FILE" \
    --references "$REFS" \
    --transcripts "$TRANSCRIPTS" \
    --audio-definition "$AUDIO_DEF" \
    --latency-unit "$LATENCY_UNIT" \
    > "$SCORES_DIR/simulstream_score_quality_comet_unbabel_xcomet_xl.log" 2>&1

  echo "Running simulstream_score_quality (COMET Unbabel/wmt22-comet-da)..."
  PYTHONUNBUFFERED=1 simulstream_score_quality \
    --scorer comet \
    --model "Unbabel/wmt22-comet-da" \
    --batch-size "8" \
    --eval-config "$SIMULSTREAM_CONFIG" \
    --log-file "$LOG_FILE" \
    --references "$REFS" \
    --transcripts "$TRANSCRIPTS" \
    --audio-definition "$AUDIO_DEF" \
    --latency-unit "$LATENCY_UNIT" \
    > "$SCORES_DIR/simulstream_score_quality_comet_unbabel_wmt22_comet_da.log" 2>&1
else
  echo "HF_TOKEN not set. Skipping COMET evaluation."
fi

echo "Running simulstream_stats..."
PYTHONUNBUFFERED=1 simulstream_stats \
  --eval-config "$SIMULSTREAM_CONFIG" \
  --log-file "$LOG_FILE" \
  --latency-unit "$LATENCY_UNIT" >> "$SCORES_DIR/simulstream_stats.log" 2>&1

echo "Simulstream scores in: $SCORES_DIR"
echo "Done."
