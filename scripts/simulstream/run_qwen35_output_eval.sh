#!/usr/bin/env bash
# Evaluate one simulstream output directory with omnisteval.
set -euo pipefail

OUTPUT_DIR=""
SRC_LANG=""
TGT_LANG=""
FORCE="${FORCE:-false}"
OMNISTEVAL_LOCAL_PATH="${OMNISTEVAL_LOCAL_PATH:-/lustre/fsw/portfolios/convai/users/lgrigoryan/iwslt26/omnisteval}"
SIMULSTREAM_LOCAL_PATH="${SIMULSTREAM_LOCAL_PATH:-/lustre/fsw/portfolios/convai/users/lgrigoryan/iwslt26/simulstream}"
INSTALL_OMNISTEVAL="${INSTALL_OMNISTEVAL:-false}"
INSTALL_SIMULSTREAM="${INSTALL_SIMULSTREAM:-false}"

usage() {
  echo "Usage: $0 output-dir=DIR src-lang=LANG tgt-lang=LANG [force=true|false]"
  exit 1
}

for arg in "$@"; do
  case "$arg" in
    output-dir=*) OUTPUT_DIR="${arg#*=}" ;;
    src-lang=*) SRC_LANG="${arg#*=}" ;;
    tgt-lang=*) TGT_LANG="${arg#*=}" ;;
    force=*) FORCE="${arg#*=}" ;;
    -h|--help|help=true) usage ;;
    *=*) echo "Unknown option: $arg"; usage ;;
    *) echo "Invalid argument format (expected key=value): $arg"; usage ;;
  esac
done

[[ -z "$OUTPUT_DIR" ]] && echo "Error: missing output-dir=DIR" && usage
[[ -z "$SRC_LANG" ]] && echo "Error: missing src-lang=LANG" && usage
[[ -z "$TGT_LANG" ]] && echo "Error: missing tgt-lang=LANG" && usage

if [[ ! -d "$OMNISTEVAL_LOCAL_PATH" ]]; then
  echo "Error: local omnisteval path not found: $OMNISTEVAL_LOCAL_PATH"
  exit 1
fi
if [[ ! -d "$SIMULSTREAM_LOCAL_PATH" ]]; then
  echo "Error: local simulstream path not found: $SIMULSTREAM_LOCAL_PATH"
  exit 1
fi

# Install local omnisteval + dependencies in the active container/venv.
if [[ "$INSTALL_OMNISTEVAL" == "true" ]]; then
  echo "Installing local omnisteval in current environment: $OMNISTEVAL_LOCAL_PATH"
  python -m pip install -e "${OMNISTEVAL_LOCAL_PATH}[comet]"
elif ! python -c "import omnisteval" >/dev/null 2>&1; then
  echo "Error: omnisteval is not installed and INSTALL_OMNISTEVAL=false"
  exit 1
fi

if [[ "$INSTALL_SIMULSTREAM" == "true" ]]; then
  echo "Installing local simulstream in current environment: $SIMULSTREAM_LOCAL_PATH"
  python -m pip install -e "$SIMULSTREAM_LOCAL_PATH"
elif ! python -c "import simulstream" >/dev/null 2>&1; then
  echo "Error: simulstream is not installed and INSTALL_SIMULSTREAM=false"
  exit 1
fi

OUTPUT_DIR_ABS="$(realpath "$OUTPUT_DIR")"
HYPOTHESIS_JSON="$OUTPUT_DIR_ABS/simulstream_output.json"
DONE_MARKER="$OUTPUT_DIR_ABS/.simulstream_eval_done"
OMNI_OUTPUT="$OUTPUT_DIR_ABS/omnisteval"
SUMMARY_FILE="$OMNI_OUTPUT/omnisteval_summary.txt"

SIMULSTREAM_CONFIG=""
shopt -s nullglob
cfg_candidates=("$OUTPUT_DIR_ABS"/*_simulstream.yaml)
shopt -u nullglob
if [[ ${#cfg_candidates[@]} -eq 0 ]]; then
  echo "Error: missing *_simulstream.yaml in $OUTPUT_DIR_ABS"
  exit 1
fi
if [[ ${#cfg_candidates[@]} -gt 1 ]]; then
  echo "Error: multiple *_simulstream.yaml files in $OUTPUT_DIR_ABS"
  printf '  - %s\n' "${cfg_candidates[@]}"
  exit 1
fi
SIMULSTREAM_CONFIG="${cfg_candidates[0]}"

if [[ ! -s "$HYPOTHESIS_JSON" ]]; then
  echo "Error: missing or empty hypothesis file: $HYPOTHESIS_JSON"
  exit 1
fi

if [[ "$FORCE" != "true" && -f "$DONE_MARKER" && -f "$SUMMARY_FILE" ]]; then
  echo "Skip existing eval: $OUTPUT_DIR_ABS"
  exit 0
fi

REFERENCE_FILE=""
TRANSCRIPT_FILE=""
AUDIO_DEFINITION=""
LATENCY_FLAG="--word_level"
BLEU_TOKENIZER="13a"

if [[ "$SRC_LANG" == "en" && "$TGT_LANG" == "zh" ]]; then
  LATENCY_FLAG="--char_level"
  BLEU_TOKENIZER="zh"
fi

if [[ "$SRC_LANG" == "cs" && "$TGT_LANG" == "en" ]]; then
  REFERENCE_FILE="/lustre/fsw/portfolios/convai/users/lgrigoryan/iwslt26/data/cs_en_dev/iwslt2026_cs_en_dev/raw/iwslt26-cs-dev-filtered.en"
  TRANSCRIPT_FILE="/lustre/fsw/portfolios/convai/users/lgrigoryan/iwslt26/data/cs_en_dev/iwslt2026_cs_en_dev/raw/iwslt26-cs-dev-filtered.cs"
  AUDIO_DEFINITION="/lustre/fsw/portfolios/convai/users/lgrigoryan/iwslt26/data/cs_en_dev/iwslt2026_cs_en_dev/raw/iwslt26-cs-dev-filtered.yaml"
else
  REFERENCE_FILE="/lustre/fsw/portfolios/convai/users/lgrigoryan/iwslt26/data/mcif/raw/ref/${TGT_LANG}.txt"
  TRANSCRIPT_FILE="/lustre/fsw/portfolios/convai/users/lgrigoryan/iwslt26/data/mcif/raw/ref/${SRC_LANG}.txt"
  AUDIO_DEFINITION="/lustre/fsw/portfolios/convai/users/lgrigoryan/iwslt26/data/mcif/raw/audio-segments.yaml"
fi

for f in "$REFERENCE_FILE" "$TRANSCRIPT_FILE" "$AUDIO_DEFINITION" "$SIMULSTREAM_CONFIG"; do
  if [[ ! -f "$f" ]]; then
    echo "Error: required file not found: $f"
    exit 1
  fi
done

mkdir -p "$OMNI_OUTPUT"
rm -f "$DONE_MARKER"

echo "Evaluating: $OUTPUT_DIR_ABS"
echo "  src:tgt = $SRC_LANG:$TGT_LANG"
echo "  tokenizer = $BLEU_TOKENIZER"
echo "  latency flag = $LATENCY_FLAG"

python -m omnisteval.cli longform \
  --speech_segmentation "$AUDIO_DEFINITION" \
  --source_sentences_file "$TRANSCRIPT_FILE" \
  --ref_sentences_file "$REFERENCE_FILE" \
  --hypothesis_file "$HYPOTHESIS_JSON" \
  --simulstream_config_file "$SIMULSTREAM_CONFIG" \
  --hypothesis_format simulstream \
  --comet \
  --comet_model Unbabel/XCOMET-XL \
  --lang "$TGT_LANG" \
  $LATENCY_FLAG \
  --bleu_tokenizer "$BLEU_TOKENIZER" \
  --output_folder "$OMNI_OUTPUT"

{
  echo "Omnisteval evaluation summary"
  echo "output_dir: $OUTPUT_DIR_ABS"
  echo "hypothesis_file: $HYPOTHESIS_JSON"
  echo "simulstream_config: $SIMULSTREAM_CONFIG"
  echo "src_lang: $SRC_LANG"
  echo "tgt_lang: $TGT_LANG"
  echo "bleu_tokenizer: $BLEU_TOKENIZER"
} > "$SUMMARY_FILE"

if [[ -f "$OMNI_OUTPUT/scores.tsv" ]]; then
  {
    echo ""
    echo "[scores.tsv]"
    awk '1' "$OMNI_OUTPUT/scores.tsv"
  } >> "$SUMMARY_FILE"
fi

touch "$DONE_MARKER"
echo "Done. Results: $OMNI_OUTPUT"
