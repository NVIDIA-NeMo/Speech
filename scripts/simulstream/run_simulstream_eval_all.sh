#!/usr/bin/env bash
#
# Run full simulstream pipeline: inference, then omnisteval evaluation, then simulstream scores.
# This script calls run_simulstream_inference.sh, run_omnisteval_eval.sh, and run_simulstream_scores.sh
# in sequence with the same output layout.
#
# Usage (same as legacy run_simulstream_eval.sh):
#   ./run_simulstream_eval_all.sh \
#     manifest=/path/to/longform_manifest.jsonl \
#     output-dir=/path/to/output_base \
#     src-lang=en tgt-lang=ru \
#     nemo-config=examples/asr/conf/asr_streaming_inference/cache_aware_rnnt.yaml \
#     [segments-manifest=/path/to/segments_manifest.jsonl] \
#     [llm-model="Qwen/Qwen3-4B-Instruct-2507"] \
#     [force=true|false] \
#     [latency-unit=word|char] [sacrebleu-tokenizer=13a|zh] \
#     [cache-att-context-size=INT] \
#     [buffered-chunk-size=FLOAT buffered-left-padding-size=FLOAT buffered-right-padding-size=FLOAT]
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEMO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

MANIFEST=""
SEGMENTS_MANIFEST=""
OUTPUT_DIR_BASE=""
SRC_LANG=""
TGT_LANG=""
NEMO_CONFIG=""
LLM_MODEL="Qwen/Qwen3-4B-Instruct-2507"

LATENCY_UNIT="word"
SACREBLEU_TOKENIZER="13a"
FORCE="false"
SKIP_OMNISTEVAL="false"

CACHE_ATT_CONTEXT_SIZE="13"

BUFFERED_CHUNK_SIZE="1.12"
BUFFERED_LEFT_PADDING_SIZE="5.6"
BUFFERED_RIGHT_PADDING_SIZE="0.56"

usage() {
  echo "Usage: $0 manifest=PATH output-dir=DIR src-lang=LANG tgt-lang=LANG nemo-config=YAML [OPTIONS]"
  echo ""
  echo "Required:"
  echo "  manifest=PATH         Longform NeMo manifest JSONL (for inference)"
  echo "  output-dir=DIR       Base directory for outputs (subdir will be created)"
  echo "  src-lang=LANG        Source language code (e.g. en, ru)"
  echo "  tgt-lang=LANG        Target language code (e.g. ru, en)"
  echo "  nemo-config=YAML     NeMo streaming config (e.g. cache_aware_rnnt.yaml)"
  echo ""
  echo "Optional:"
  echo "  segments-manifest=PATH   Segments manifest JSONL (needed for omnisteval/simulstream scoring)"
  echo "  llm-model=MODEL          LLM model for inference (default: Qwen/Qwen3-4B-Instruct-2507)"
  echo "  force=true|false         Re-run inference and overwrite existing output json (default: false)"
  echo "  skip-omnisteval=true|false  Skip omnisteval step (default: false)"
  echo "  latency-unit=UNIT        Latency unit for simulstream scoring (default: word)"
  echo "  sacrebleu-tokenizer=TOK  SacreBLEU tokenizer (default: 13a)"
  echo "  cache-att-context-size=INT   cache_aware_rnnt override used in output naming (default: 13)"
  echo "  buffered-chunk-size=FLOAT    buffered_rnnt override used in output naming (default: 1.12)"
  echo "  buffered-left-padding-size=FLOAT  buffered_rnnt override (default: 5.6)"
  echo "  buffered-right-padding-size=FLOAT buffered_rnnt override (default: 0.56)"
  exit 1
}

for arg in "$@"; do
  case "$arg" in
    manifest=*)            MANIFEST="${arg#*=}" ;;
    segments-manifest=*)   SEGMENTS_MANIFEST="${arg#*=}" ;;
    output-dir=*)          OUTPUT_DIR_BASE="${arg#*=}" ;;
    src-lang=*)            SRC_LANG="${arg#*=}" ;;
    tgt-lang=*)            TGT_LANG="${arg#*=}" ;;
    nemo-config=*)         NEMO_CONFIG="${arg#*=}" ;;
    llm-model=*)           LLM_MODEL="${arg#*=}" ;;
    latency-unit=*)        LATENCY_UNIT="${arg#*=}" ;;
    sacrebleu-tokenizer=*) SACREBLEU_TOKENIZER="${arg#*=}" ;;
    cache-att-context-size=*) CACHE_ATT_CONTEXT_SIZE="${arg#*=}" ;;
    buffered-chunk-size=*) BUFFERED_CHUNK_SIZE="${arg#*=}" ;;
    buffered-left-padding-size=*) BUFFERED_LEFT_PADDING_SIZE="${arg#*=}" ;;
    buffered-right-padding-size=*) BUFFERED_RIGHT_PADDING_SIZE="${arg#*=}" ;;
    skip-omnisteval=*)
      SKIP_OMNI_VALUE="${arg#*=}"
      case "${SKIP_OMNI_VALUE,,}" in
        1|true|yes|on) SKIP_OMNISTEVAL="true" ;;
        0|false|no|off|"") SKIP_OMNISTEVAL="false" ;;
        *) echo "Error: invalid skip-omnisteval value '$SKIP_OMNI_VALUE' (use true/false)"; usage ;;
      esac
      ;;
    force=*)
      FORCE_VALUE="${arg#*=}"
      case "${FORCE_VALUE,,}" in
        1|true|yes|on) FORCE="true" ;;
        0|false|no|off|"") FORCE="false" ;;
        *) echo "Error: invalid force value '$FORCE_VALUE' (use true/false)"; usage ;;
      esac
      ;;
    -h|--help|help=true)   usage ;;
    *=*)                   echo "Unknown option: $arg"; usage ;;
    *)                     echo "Invalid argument format (expected key=value): $arg"; usage ;;
  esac
done

[[ -z "$MANIFEST" ]] && echo "Error: missing required argument: manifest=PATH" && usage
[[ -z "$OUTPUT_DIR_BASE" ]] && echo "Error: missing required argument: output-dir=DIR" && usage
[[ -z "$SRC_LANG" ]] && echo "Error: missing required argument: src-lang=LANG" && usage
[[ -z "$TGT_LANG" ]] && echo "Error: missing required argument: tgt-lang=LANG" && usage
[[ -z "$NEMO_CONFIG" ]] && echo "Error: missing required argument: nemo-config=YAML" && usage

if [[ ! -f "$NEMO_CONFIG" ]] && [[ -f "$NEMO_ROOT/$NEMO_CONFIG" ]]; then
  NEMO_CONFIG="$NEMO_ROOT/$NEMO_CONFIG"
fi
if [[ ! -f "$NEMO_CONFIG" ]]; then
  echo "Error: nemo config not found: $NEMO_CONFIG"
  exit 1
fi
NEMO_CONFIG_ABS="$(realpath "$NEMO_CONFIG")"

CONFIG_NAME=$(basename "$NEMO_CONFIG" .yaml)
MANIFEST_NAME=$(basename "$MANIFEST")
MANIFEST_NAME="${MANIFEST_NAME%.jsonl}"
MANIFEST_NAME="${MANIFEST_NAME%.json}"
MANIFEST_NAME=${MANIFEST_NAME#manifest_}
LLM_MODEL_SAFE=${LLM_MODEL//\//_}
OUTPUT_DIR="$OUTPUT_DIR_BASE/${MANIFEST_NAME}/${CONFIG_NAME}/${LLM_MODEL_SAFE}"

if [[ "$CONFIG_NAME" == "cache_aware_rnnt" ]]; then
  if [[ -z "$CACHE_ATT_CONTEXT_SIZE" ]]; then
    echo "Error: cache-att-context-size=INT is required for cache_aware_rnnt."
    exit 1
  fi
  OUTPUT_DIR="$OUTPUT_DIR_BASE/${MANIFEST_NAME}/${CONFIG_NAME}_${CACHE_ATT_CONTEXT_SIZE}/${LLM_MODEL_SAFE}"
fi
if [[ "$CONFIG_NAME" == "buffered_rnnt" ]]; then
  if [[ -z "$BUFFERED_CHUNK_SIZE" || -z "$BUFFERED_LEFT_PADDING_SIZE" || -z "$BUFFERED_RIGHT_PADDING_SIZE" ]]; then
    echo "Error: buffered-chunk-size, buffered-left-padding-size, and buffered-right-padding-size are required for buffered_rnnt."
    exit 1
  fi
  OUTPUT_DIR="$OUTPUT_DIR_BASE/${MANIFEST_NAME}/${CONFIG_NAME}_c${BUFFERED_CHUNK_SIZE}_l${BUFFERED_LEFT_PADDING_SIZE}_r${BUFFERED_RIGHT_PADDING_SIZE}/${LLM_MODEL_SAFE}"
fi

mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR_ABS="$(realpath "$OUTPUT_DIR")"

echo "========== 1. Run NeMo simulstream inference =========="
INFERENCE_ARGS=(
  "manifest=$MANIFEST"
  "output-dir=$OUTPUT_DIR_BASE"
  "src-lang=$SRC_LANG"
  "tgt-lang=$TGT_LANG"
  "nemo-config=$NEMO_CONFIG_ABS"
  "llm-model=$LLM_MODEL"
  "force=$FORCE"
)
[[ -n "$CACHE_ATT_CONTEXT_SIZE" ]] && INFERENCE_ARGS+=("cache-att-context-size=$CACHE_ATT_CONTEXT_SIZE")
[[ -n "$BUFFERED_CHUNK_SIZE" ]] && INFERENCE_ARGS+=("buffered-chunk-size=$BUFFERED_CHUNK_SIZE")
[[ -n "$BUFFERED_LEFT_PADDING_SIZE" ]] && INFERENCE_ARGS+=("buffered-left-padding-size=$BUFFERED_LEFT_PADDING_SIZE")
[[ -n "$BUFFERED_RIGHT_PADDING_SIZE" ]] && INFERENCE_ARGS+=("buffered-right-padding-size=$BUFFERED_RIGHT_PADDING_SIZE")
"$SCRIPT_DIR/run_simulstream_inference.sh" "${INFERENCE_ARGS[@]}"

if [[ -z "$SEGMENTS_MANIFEST" ]]; then
  echo "Omnisteval and simulstream scores skipped (set segments-manifest=... to run)."
else
  if [[ "$SKIP_OMNISTEVAL" == "true" ]]; then
    echo ""
    echo "========== 2. Skip omnisteval (skip-omnisteval=true) =========="
  else
    echo ""
    echo "========== 2. Run omnisteval longform =========="
    OMNI_ARGS=(
      "output-dir=$OUTPUT_DIR_ABS"
      "tgt-lang=$TGT_LANG"
      "comet=true"
      "segments-manifest=$SEGMENTS_MANIFEST"
      "latency-unit=$LATENCY_UNIT"
      "bleu-tokenizer=$SACREBLEU_TOKENIZER"
    )
    if ! "$SCRIPT_DIR/run_omnisteval_eval.sh" "${OMNI_ARGS[@]}"; then
      echo "Warning: omnisteval step failed; continuing to simulstream scoring."
    fi
  fi

  echo ""
  echo "========== 3. Run simulstream scores (latency / quality / stats) =========="
  SCORES_ARGS=(
    "output-dir=$OUTPUT_DIR_ABS"
    "latency-unit=$LATENCY_UNIT"
    "sacrebleu-tokenizer=$SACREBLEU_TOKENIZER"
    "segments-manifest=$SEGMENTS_MANIFEST"
  )
  "$SCRIPT_DIR/run_simulstream_scores.sh" "${SCORES_ARGS[@]}"
fi

echo ""
echo "Done. Output directory: $OUTPUT_DIR_ABS"
