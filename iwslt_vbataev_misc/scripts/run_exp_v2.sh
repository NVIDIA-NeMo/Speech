DATA_DIR=/data/iwslt26/mcif
#NER_RESULTS_PATH=${DATA_DIR}/baseline/context/ner_llm_results.json
NEMO_CONFIG=examples/asr/conf/asr_streaming_inference/buffered_rnnt.yaml
NEMO_CONFIG="$(realpath "$NEMO_CONFIG")"

SRC_LANG_CODE=en
NMT_MODEL="Qwen/Qwen3-4B-Instruct-2507"
ASR_MODEL="/home/vbataev/code/checkpoints/asr/parakeet-unified-en-0.6b_cleaned.nemo"

TGT_LANG_CODES=(de it zh)

for TGT_LANG_CODE in "${TGT_LANG_CODES[@]}"; do
  RESULTS_DIR="_checks/pipeline_v2/asr-unified-096-096/nmt-q3-4b/en-${TGT_LANG_CODE}/baseline"
  mkdir -p $RESULTS_DIR
  RESULTS_DIR="$(realpath "$RESULTS_DIR")"

  EVAL_CONFIG="${RESULTS_DIR}/buffered_rnnt_simulstream.yaml"
  METRICS_LOG_FILE="${RESULTS_DIR}/simulstream_output.jsonl"
  DETAILED_LOG_FILE="${RESULTS_DIR}/detailed_log.jsonl"

  python nemo/collections/asr/inference/run_nemo_simulstream.py \
      --config "$NEMO_CONFIG" \
      --wav-list ${DATA_DIR}/wav_list.txt \
      --src-lang "$SRC_LANG_CODE" \
      --tgt-lang "$TGT_LANG_CODE" \
      --metrics-log "${METRICS_LOG_FILE}" \
      --use-adapter-v2 \
      streaming.left_padding_size=5.6 \
      streaming.chunk_size=0.96 \
      streaming.right_padding_size=0.96 \
      streaming.decode_temporary=true \
      endpointing.stop_history_eou=1200 \
      pipeline_v2.num_prev_sentences_for_translation=5 \
      detailed_log_path=${DETAILED_LOG_FILE} \
      asr.model_name=${ASR_MODEL} \
      nmt.model_name=${NMT_MODEL}


  #per_stream_boosting.phrases_file=${DATA_DIR}/boosting_phrases_abstract_v1.json \
  #per_stream_boosting.alpha=0.3 \

  . .evaluation/bin/activate

  REFERENCE_FILE=${DATA_DIR}/raw/ref/${TGT_LANG_CODE}.txt
  TRANSCRIPT_FILE=${DATA_DIR}/raw/ref/en.txt
  AUDIO_DEFINITION=${DATA_DIR}/raw/audio-segments.yaml

  if [ "$TGT_LANG_CODE" = "zh" ]; then
      SACREBLEU_TOKENIZER=zh
      MOSES_TOKENIZER=zh
      CHAR_LEVEL_FLAG="--char_level"
  else
      SACREBLEU_TOKENIZER=13a
      MOSES_TOKENIZER=13a
      CHAR_LEVEL_FLAG="--word_level"
  fi

  omnisteval longform \
    --speech_segmentation "$AUDIO_DEFINITION" \
    --source_sentences_file "$TRANSCRIPT_FILE" \
    --ref_sentences_file "$REFERENCE_FILE" \
    --hypothesis_file "${METRICS_LOG_FILE}" \
    --simulstream_config_file "$EVAL_CONFIG" \
    --hypothesis_format simulstream \
    --comet \
    --comet_model Unbabel/XCOMET-XL \
    --lang "${MOSES_TOKENIZER}" \
    $CHAR_LEVEL_FLAG \
    --bleu_tokenizer "${SACREBLEU_TOKENIZER}" \
    --output_folder "${RESULTS_DIR}/segmentation_output"
done