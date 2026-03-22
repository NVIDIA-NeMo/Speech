#!/usr/bin/env bash
#
# Run omnisteval longform evaluation on simulstream output.
# Expects the output directory from run_simulstream_inference.sh (with audio_definitions.yaml,
# references.txt, transcripts.txt, simulstream_output.json).
#
# Usage:
#   ./run_omnisteval_eval.sh \
#     output-dir=/path/to/run_output_dir \
#     tgt-lang=ru \
#     [simulstream-config=/path/to/simulstream/config/nemo_cascade.yaml] \
#     [comet=true]
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR=""
SIMULSTREAM_CONFIG=""
TGT_LANG=""
COMET=""
SEGMENTS_MANIFEST=""
LATENCY_UNIT="word"
BLEU_TOKENIZER="13a"
LATENCY_FLAG="--word_level"
if [[ "$LATENCY_UNIT" == "char" ]]; then
  LATENCY_FLAG="--char_level"
fi

usage() {
  echo "Usage: $0 output-dir=DIR tgt-lang=LANG [OPTIONS]"
  echo ""
  echo "Required:"
  echo "  output-dir=DIR           Run output directory (contains audio_definitions.yaml, references.txt, etc.)"
  echo "  tgt-lang=LANG           Target language code (e.g. ru, en)"
  echo ""
  echo "Optional:"
  echo "  simulstream-config=YAML Simulstream config for omnisteval (auto-detected from output-dir if omitted)"
  echo "  comet=true|false         Enable COMET in omnisteval (default: false)"
  echo "  segments-manifest=PATH   Segments manifest used to auto-create missing eval files"
  echo "  latency-unit=char|word   Latency unit for omnisteval (default: word)"
  echo "  bleu-tokenizer=TOKENIZER Tokenizer for SacreBLEU (default: 13a)"
  exit 1
}

for arg in "$@"; do
  case "$arg" in
    output-dir=*)          OUTPUT_DIR="${arg#*=}" ;;
    simulstream-config=*)  SIMULSTREAM_CONFIG="${arg#*=}" ;;
    tgt-lang=*)            TGT_LANG="${arg#*=}" ;;
    segments-manifest=*)   SEGMENTS_MANIFEST="${arg#*=}" ;;
    latency-unit=*)         LATENCY_UNIT="${arg#*=}" ;;
    bleu-tokenizer=*)       BLEU_TOKENIZER="${arg#*=}" ;;
    comet=*)
      COMET_VALUE="${arg#*=}"
      case "${COMET_VALUE,,}" in
        1|true|yes|on) COMET="--comet" ;;
        0|false|no|off|"") COMET="" ;;
        *) echo "Error: invalid comet value '$COMET_VALUE' (use true/false)"; usage ;;
      esac
      ;;
    -h|--help|help=true)   usage ;;
    *=*)                   echo "Unknown option: $arg"; usage ;;
    *)                     echo "Invalid argument format (expected key=value): $arg"; usage ;;
  esac
done

[[ -z "$OUTPUT_DIR" ]] && echo "Error: missing required argument: output-dir=DIR" && usage
[[ -z "$TGT_LANG" ]] && echo "Error: missing required argument: tgt-lang=LANG" && usage

OUTPUT_DIR_ABS="$(realpath "$OUTPUT_DIR")"
AUDIO_DEF="$OUTPUT_DIR_ABS/audio_definitions.yaml"
REFS="$OUTPUT_DIR_ABS/references.txt"
TRANSCRIPTS="$OUTPUT_DIR_ABS/transcripts.txt"
HYPOTHESIS_JSON="$OUTPUT_DIR_ABS/simulstream_output.json"

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

for f in "$AUDIO_DEF" "$REFS" "$TRANSCRIPTS" "$HYPOTHESIS_JSON"; do
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
  candidates=("${HYPOTHESIS_JSON%/*}"/*_simulstream.yaml)
  shopt -u nullglob
  if [[ ${#candidates[@]} -eq 0 ]]; then
    echo "Error: Could not auto-detect simulstream config in ${HYPOTHESIS_JSON%/*}"
    echo "Expected a file like *_simulstream.yaml or pass simulstream-config=PATH"
    exit 1
  fi
  if [[ ${#candidates[@]} -gt 1 ]]; then
    echo "Error: Multiple *_simulstream.yaml files found in ${HYPOTHESIS_JSON%/*}; pass simulstream-config=PATH explicitly."
    printf '  - %s\n' "${candidates[@]}"
    exit 1
  fi
  SIMULSTREAM_CONFIG="$(realpath "${candidates[0]}")"
fi
echo "Using simulstream config: $SIMULSTREAM_CONFIG"

OMNI_OUTPUT="$OUTPUT_DIR_ABS/omnisteval"
echo "========== Run omnisteval longform =========="
echo "Speech segmentation: $AUDIO_DEF"
python -m omnisteval.cli longform \
  --speech_segmentation "$AUDIO_DEF" \
  --ref_sentences_file "$REFS" \
  --hypothesis_file "$HYPOTHESIS_JSON" \
  --hypothesis_format=simulstream \
  --simulstream_config_file "$SIMULSTREAM_CONFIG" \
  --lang ${TGT_LANG} \
  --bleu_tokenizer ${BLEU_TOKENIZER} \
  --source_sentences_file "$TRANSCRIPTS" \
  --output_folder "$OMNI_OUTPUT" \
  $COMET \
  $LATENCY_FLAG \
  --comet_model Unbabel/XCOMET-XL

SUMMARY_FILE="$OMNI_OUTPUT/omnisteval_summary.txt"
{
  echo "Omnisteval evaluation summary"
  echo "output_dir: $OUTPUT_DIR_ABS"
  echo "hypothesis_file: $HYPOTHESIS_JSON"
  echo "lang: $TGT_LANG"
} > "$SUMMARY_FILE"

if [[ -f "$OMNI_OUTPUT/scores.tsv" ]]; then
  {
    echo ""
    echo "[scores.tsv]"
    cat "$OMNI_OUTPUT/scores.tsv"
  } >> "$SUMMARY_FILE"
fi

if [[ -f "$OMNI_OUTPUT/evaluation_report.txt" ]]; then
  {
    echo ""
    echo "[evaluation_report: selected metric lines]"
    awk 'BEGIN{IGNORECASE=1} /score|SacreBLEU|BLEU|COMET|LAAL|latency|quality|mean|avg|average|median|p50|p90|p95|p99|count|tokens|words|chars/' "$OMNI_OUTPUT/evaluation_report.txt"
  } >> "$SUMMARY_FILE"
fi

echo "Omnisteval results in: $OMNI_OUTPUT"
echo "Summary written to: $SUMMARY_FILE"
echo "Done."
