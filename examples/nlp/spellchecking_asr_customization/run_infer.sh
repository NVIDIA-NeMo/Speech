# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

## RUN INFERENCE ON NEMO MANIFEST AND CUSTOM VOCABULARY

## Path to NeMo repository
NEMO_PATH=NeMo

## Download model repo from Hugging Face (if clone doesn't work, run "git lfs install" and try again)
git clone https://huggingface.co/bene-ges/spellmapper_asr_customization_en
## Download repo with test data
git clone https://huggingface.co/datasets/bene-ges/spellmapper_en_evaluation

## Files in model repo
PRETRAINED_MODEL=spellmapper_asr_customization_en/training_10m_5ep.nemo
NGRAM_MAPPINGS=spellmapper_asr_customization_en/replacement_vocab_filt.txt
BIG_SAMPLE=spellmapper_asr_customization_en/big_sample.txt

## Override these two files if you want to test on your own data
## File with input nemo ASR manifest
INPUT_MANIFEST=spellmapper_en_evaluation/medical_manifest_ctc.json
## File containing custom words and phrases (plain text)
CUSTOM_VOCAB=spellmapper_en_evaluation/medical_custom_vocab.txt

## Other files will be created 
## File with index of custom vocabulary
INDEX="index.txt"
## File with short fragments and corresponding original sentences
SHORT2FULL="short2full.txt"
## File with input for SpellMapper inference
SPELLMAPPER_INPUT="spellmapper_input.txt"
## File with output of SpellMapper inference
SPELLMAPPER_OUTPUT="spellmapper_output.txt"
## File with output nemo ASR manifest
OUTPUT_MANIFEST="out_manifest.json"


# Create index of custom vocabulary
python ${NEMO_PATH}/examples/nlp/spellchecking_asr_customization/create_custom_vocab_index.py \
  --input_name ${CUSTOM_VOCAB} \
  --ngram_mappings ${NGRAM_MAPPINGS} \
  --output_name ${INDEX} \
  --min_log_prob -4.0 \
  --max_phrases_per_ngram 600

# Prepare input for SpellMapper inference
python ${NEMO_PATH}/examples/nlp/spellchecking_asr_customization/prepare_input_from_manifest.py \
  --manifest ${INPUT_MANIFEST} \
  --custom_vocab_index ${INDEX} \
  --big_sample ${BIG_SAMPLE} \
  --short2full_name ${SHORT2FULL} \
  --output_name ${SPELLMAPPER_INPUT} \
  --field_name "pred_text" \
  --len_in_words 16 \
  --step_in_words 8

# Run SpellMapper inference
python ${NEMO_PATH}/examples/nlp/spellchecking_asr_customization/spellchecking_asr_customization_infer.py \
  pretrained_model=${PRETRAINED_MODEL} \
  model.max_sequence_len=512 \
  inference.from_file=${SPELLMAPPER_INPUT} \
  inference.out_file=${SPELLMAPPER_OUTPUT} \
  inference.batch_size=16 \
  lang=en

# Postprocess and create output corrected manifest
python ${NEMO_PATH}/examples/nlp/spellchecking_asr_customization/postprocess_and_update_manifest.py \
  --input_manifest ${INPUT_MANIFEST} \
  --short2full_name ${SHORT2FULL} \
  --output_manifest ${OUTPUT_MANIFEST} \
  --spellmapper_result ${SPELLMAPPER_OUTPUT} \
  --replace_hyphen_to_space \
  --field_name "pred_text" \
  --use_dp \
  --ngram_mappings ${NGRAM_MAPPINGS} \
  --min_dp_score_per_symbol -1.5

# Check WER of initial manifest
python ${NEMO_PATH}/examples/asr/speech_to_text_eval.py \
  dataset_manifest=${INPUT_MANIFEST} \
  use_cer=False \
  only_score_manifest=True

# Check WER of corrected manifest
python ${NEMO_PATH}/examples/asr/speech_to_text_eval.py \
  dataset_manifest=${OUTPUT_MANIFEST} \
  use_cer=False \
  only_score_manifest=True
