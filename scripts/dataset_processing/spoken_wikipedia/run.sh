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

## Download the Spoken Wikipedia corpus for English
## Note, that there are some other languages available
## @InProceedings{KHN16.518,
##  author = {Arne K{\"o}hn and Florian Stegen and Timo Baumann},
##  title = {Mining the Spoken Wikipedia for Speech Data and Beyond},
##  booktitle = {Proceedings of the Tenth International Conference on Language Resources and Evaluation (LREC 2016)},
##  year = {2016},
##  month = {may},
##  date = {23-28},
##  location = {Portorož, Slovenia},
##  editor = {Nicoletta Calzolari (Conference Chair) and Khalid Choukri and Thierry Declerck and Marko Grobelnik and Bente Maegaard and Joseph Mariani and Asuncion Moreno and Jan Odijk and Stelios Piperidis},
##  publisher = {European Language Resources Association (ELRA)},
##  address = {Paris, France},
##  isbn = {978-2-9517408-9-1},
##  islrn = {684-927-624-257-3/},
##  language = {english}
## }

wget https://corpora.uni-hamburg.de/hzsk/de/islandora/object/file:swc-2.0_en-with-audio/datastream/TAR/en-with-audio.tar .
tar -xvf en-with-audio.tar

##  We get a folder English with 1339 subfolders, each subfolder corresponds to a Wikipedia article. Example:
##  ├── Universal_suffrage
##  │   ├── aligned.swc
##  │   ├── audiometa.txt
##  │   ├── audio.ogg
##  │   ├── info.json
##  │   ├── wiki.html
##  │   ├── wiki.txt
##  │   └── wiki.xml

##  We will use two files: audio.ogg and wiki.txt

## Some folders have multiple .ogg files, this will be handled during preprocess.py. Example:
##  |── Universe
##  │   ├── aligned.swc
##  │   ├── audio1.ogg
##  │   ├── audio2.ogg
##  │   ├── audio3.ogg
##  │   ├── audio4.ogg
##  │   ├── audiometa.txt
##  │   ├── info.json
##  │   ├── wiki.html
##  │   ├── wiki.txt
##  │   └── wiki.xml

## Some rare folders are incomplete, these will be skipped during preprocessing.

## Rename some folders with special symbols because they cause problems to ffmpeg when concatening multiple .ogg files
mv "english/The_Hitchhiker%27s_Guide_to_the_Galaxy" "english/The_Hitchhikers_guide_to_the_Galaxy"
mv "english/SummerSlam_(2003)" "english/SummerSlam_2003"
mv "english/Over_the_Edge_(1999)" "english/Over_the_Edge_1999"
mv "english/Lost_(TV_series)" "english/Lost_TV_series"
mv "english/S._A._Andr%c3%a9e%27s_Arctic_Balloon_Expedition_of_1897" "english/S_A_Andres_Arctic_Balloon_Expedition_of_1897"

## path to NeMo repository, e.g. /home/user/NeMo
NEMO_PATH=

INPUT_DIR="english"
OUTPUT_DIR=${INPUT_DIR}_result

rm -rf $OUTPUT_DIR
rm -rf ${INPUT_DIR}_prepared
mkdir ${INPUT_DIR}_prepared
mkdir ${INPUT_DIR}_prepared/audio
mkdir ${INPUT_DIR}_prepared/text
python ${NEMO_PATH}/scripts/dataset_processing/spoken_wikipedia/preprocess.py --input_folder ${INPUT_DIR} --destination_folder ${INPUT_DIR}_prepared

## Now we have ${INPUT_DIR}_prepared folder with the following structure:
##  ├── audio
##  |   ├── 1.ogg
##  |   ├── 2.ogg
##  |   ...
##  └── text
##      ├── 1.txt     
##      ├── 2.txt     
##      ...

MODEL_FOR_SEGMENTATION="QuartzNet15x5Base-En" 
MODEL_FOR_RECOGNITION="stt_en_conformer_ctc_large"
## We set this threshold as very permissive, later we will use other metrics for filtering
THRESHOLD=-10

${NEMO_PATH}/tools/ctc_segmentation/run_segmentation.sh \
--SCRIPTS_DIR=${NEMO_PATH}/tools/ctc_segmentation/scripts \
--MODEL_NAME_OR_PATH=${MODEL_FOR_SEGMENTATION} \
--DATA_DIR=${INPUT_DIR}_prepared \
--OUTPUT_DIR=${OUTPUT_DIR} \
--MIN_SCORE=${MIN_SCORE}

# Thresholds for filtering
CER_THRESHOLD=20
WER_THRESHOLD=30
CER_EDGE_THRESHOLD=30
LEN_DIFF_RATIO_THRESHOLD=0.15
EDGE_LEN=25
BATCH_SIZE=1

${NEMO_PATH}/tools/ctc_segmentation/run_filter.sh \
--SCRIPTS_DIR=${NEMO_PATH}/tools/ctc_segmentation/scripts \
--MODEL_NAME_OR_PATH=${MODEL_FOR_RECOGNITION} \
--BATCH_SIZE=${BATCH_SIZE} \
--MANIFEST=$OUTPUT_DIR/manifests/manifest.json \
--INPUT_AUDIO_DIR=${INPUT_DIR}_prepared/audio/ \
--EDGE_LEN=${EDGE_LEN} \
--CER_THRESHOLD=${CER_THRESHOLD} \
--WER_THRESHOLD=${WER_THRESHOLD} \
--CER_EDGE_THRESHOLD=${CER_EDGE_THRESHOLD} \
--LEN_DIFF_RATIO_THRESHOLD=${LEN_DIFF_RATIO_THRESHOLD}

python ${NEMO_PATH}/examples/asr/speech_to_text_eval.py \
  dataset_manifest=${OUTPUT_DIR}/manifests/manifest_transcribed_metrics_filtered.json \
  use_cer=True \
  only_score_manifest=True

python ${NEMO_PATH}/examples/asr/speech_to_text_eval.py \
  dataset_manifest=${OUTPUT_DIR}/manifests/manifest_transcribed_metrics_filtered.json \
  use_cer=False \
  only_score_manifest=True
