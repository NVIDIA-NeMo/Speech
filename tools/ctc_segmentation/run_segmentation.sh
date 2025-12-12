#!/bin/bash

# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

# default values for optional arguments
MIN_SCORE=-2
CUT_PREFIX=0
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

SCRIPTS_DIR=$SCRIPT_DIR/"scripts" # /<PATH TO>/NeMo/tools/ctc_segmentation/tools/scripts/ directory
OFFSET=0
LANGUAGE='en' # 'en', 'es', 'ru'...
MAX_SEGMENT_LEN=30
ADDITIONAL_SPLIT_SYMBOLS=":|;"
USE_NEMO_NORMALIZATION='True'
NUM_JOBS=-2       # The maximum number of concurrently running jobs, `-2` - all CPUs but one are used
SAMPLE_RATE=16000 # Target sample rate (default for ASR data - 16000 Hz)
MAX_DURATION=20   # Maximum audio segment duration, in seconds. Samples that are longer will be dropped.

for ARG in "$@"; do
  key=$(echo $ARG | cut -f1 -d=)
  value=$(echo $ARG | cut -f2 -d=)

  if [[ $key == *"--"* ]]; then
    v="${key/--/}"
    declare $v="${value}"
  fi
done

echo "MODEL_NAME_OR_PATH = $MODEL_NAME_OR_PATH"
echo "DATA_DIR = $DATA_DIR"
echo "OUTPUT_DIR = $OUTPUT_DIR"
echo "MIN_SCORE = $MIN_SCORE"
echo "CUT_PREFIX = $CUT_PREFIX"
echo "SCRIPTS_DIR = $SCRIPTS_DIR"
echo "OFFSET = $OFFSET"
echo "LANGUAGE = $LANGUAGE"
echo "MIN_SEGMENT_LEN = $MIN_SEGMENT_LEN"
echo "MAX_SEGMENT_LEN = $MAX_SEGMENT_LEN"
echo "SAMPLE_RATE = $SAMPLE_RATE"
echo "ADDITIONAL_SPLIT_SYMBOLS = $ADDITIONAL_SPLIT_SYMBOLS"
echo "USE_NEMO_NORMALIZATION = $USE_NEMO_NORMALIZATION"

if [[ -z $MODEL_NAME_OR_PATH ]] || [[ -z $DATA_DIR ]] || [[ -z $OUTPUT_DIR ]]; then
  echo "Usage: $(basename "$0")
  --MODEL_NAME_OR_PATH=[model_name_or_path]
  --DATA_DIR=[data_dir]
  --OUTPUT_DIR=[output_dir]
  --LANGUAGE=[language (Optional)]
  --OFFSET=[offset value (Optional)]
  --CUT_PREFIX=[cut prefix in sec (Optional)]
  --SCRIPTS_DIR=[scripts_dir_path (Optional)]
  --MAX_SEGMENT_LEN=[max number of characters of the text segment for alignment (Optional)]
  --ADDITIONAL_SPLIT_SYMBOLS=[Additional symbols to use for
    sentence split if eos sentence split resulted in sequence longer than --max_length.
    Use '|' as a separator between symbols, for example: ';|:' (Optional)]
  --USE_NEMO_NORMALIZATION Set to 'True' to use NeMo Normalization tool to convert
    numbers from written to spoken format. By default num2words package will be used. (Optional)"
  exit 1
fi

# check if num2words and ctc_segmentation are installed
if ! command -v num2words &> /dev/null; then
  echo "num2words could not be found"
  echo "please install using tools/ctc_segmentation/requirements.txt"
  exit 1
fi

if ! command -v ctc_segmentation &> /dev/null; then
  echo "ctc_segmentation could not be found"
  echo "please install using tools/ctc_segmentation/requirements.txt"
  exit 1
fi

NEMO_NORMALIZATION=""
if [[ ${USE_NEMO_NORMALIZATION,,} == "true" ]]; then
  NEMO_NORMALIZATION="--use_nemo_normalization "
fi

# STEP #1
# Prepare text and audio data for segmentation
echo "TEXT AND AUDIO PREPROCESSING..."
python $SCRIPTS_DIR/prepare_data.py \
  --in_text=$DATA_DIR/text \
  --audio_dir=$DATA_DIR/audio \
  --output_dir=$OUTPUT_DIR/processed/ \
  --language=$LANGUAGE \
  --cut_prefix=$CUT_PREFIX \
  --model=$MODEL_NAME_OR_PATH \
  --max_length=$MAX_SEGMENT_LEN \
  --sample_rate=$SAMPLE_RATE \
  --additional_split_symbols=$ADDITIONAL_SPLIT_SYMBOLS $NEMO_NORMALIZATION || exit

# STEP #2
# Run CTC-segmentation. One might want to perform alignment with various window sizes
# Note, if the alignment with the initial window size isn't found, the window size will be double to re-attempt alignment
echo "SEGMENTATION STEP..."
for WINDOW in 8000 12000; do
  python $SCRIPTS_DIR/run_ctc_segmentation.py \
    --output_dir=$OUTPUT_DIR \
    --data=$OUTPUT_DIR/processed \
    --sample_rate=$SAMPLE_RATE \
    --model=$MODEL_NAME_OR_PATH \
    --window_len $WINDOW || exit
done

# STEP #3 (Optional)
# Verify aligned segments only if multiple WINDOWs used in the Step #2)
echo "VERIFYING SEGMENTS..."
python $SCRIPTS_DIR/verify_segments.py \
  --base_dir=$OUTPUT_DIR || exit

# STEP #4
# Cut the original audio files based on the alignment score. Only segments with alignment confidence score
# above the MIN_SCORE value will be saved to $OUTPUT_DIR/manifests/manifest.json
echo "CUTTING AUDIO..."
python $SCRIPTS_DIR/cut_audio.py \
  --output_dir=$OUTPUT_DIR \
  --alignment=$OUTPUT_DIR/verified_segments \
  --threshold=$MIN_SCORE \
  --offset=$OFFSET \
  --sample_rate=$SAMPLE_RATE \
  --max_duration=$MAX_DURATION || exit
