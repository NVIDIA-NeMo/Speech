#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

clean_up() {
    kill -- -$$
}

depends_on () {
    HOST=$1
    PORT=$2
    STATUS=$(curl -X PUT http://$HOST:$PORT >/dev/null 2>/dev/null; echo $?)
    while [ $STATUS -ne 0 ]
    do
         echo "waiting for server ($HOST:$PORT) to be up"
         sleep 10
         STATUS=$(curl -X PUT http://$HOST:$PORT >/dev/null 2>/dev/null; echo $?)
    done
    echo "server ($HOST:$PORT) is up running"
}

load_variables() {
    PYTHONUNBUFFERED=TRUE 
    full_path=$(realpath $0)
    dir_path=$(dirname $full_path)
    source $dir_path/env_variables.sh
}

# load the environment variables
load_variables


# launch bert model service
python scripts/nlp_language_modeling/service_launch_scripts/start_bert_service.py \
          tokenizer.merge_file=$MERGE_FILE \
          tokenizer.vocab_file=$VOCAB_FILE \
          sentence_bert.sentence_bert=all-mpnet-base-v2 \
          sentence_bert.devices=$BERT_DEVICES \
          sentence_bert.port=${BERT_PORT} &

depends_on "0.0.0.0" ${BERT_PORT}

# launch static retrieval service
python scripts/nlp_language_modeling/service_launch_scripts/start_static_retrieval_service.py \
          tokenizer.merge_file=$MERGE_FILE \
          tokenizer.vocab_file=$VOCAB_FILE \
          service.faiss_devices=null \
          service.faiss_index=$STATIC_FAISS_INDEX \
          service.retrieval_index=$STATIC_RETRIVAL_DB \
          service.query_bert_port=${BERT_PORT} \
          service.port=${STATIC_RETRIEVAL_PORT} &

# launch dynamic retrieval service
python scripts/nlp_language_modeling/service_launch_scripts/start_dynamic_retrieval_service.py \
          tokenizer.merge_file=$MERGE_FILE \
          tokenizer.vocab_file=$VOCAB_FILE \
          service.faiss_devices=null \
          service.ctx_bert_port=${BERT_PORT} \
          service.query_bert_port=${BERT_PORT} \
          service.port=${DYNAMIC_RETRIEVAL_PORT} &

depends_on "0.0.0.0" ${STATIC_RETRIEVAL_PORT}
depends_on "0.0.0.0" ${DYNAMIC_RETRIEVAL_PORT}

# launch combo service
python scripts/nlp_language_modeling/service_launch_scripts/start_combo_retrieval_service.py \
          tokenizer.merge_file=$MERGE_FILE \
          tokenizer.vocab_file=$VOCAB_FILE \
          service.child_services.0.service_port=${STATIC_RETRIEVAL_PORT} \
          service.child_services.1.service_port=${DYNAMIC_RETRIEVAL_PORT} \
          service.port=${COMBO_RETRIEVAL_PORT} &

depends_on "0.0.0.0" ${COMBO_RETRIEVAL_PORT}

# launch text generation server
python scripts/nlp_language_modeling/service_launch_scripts/start_retro_model_service.py \
          trainer.devices=1 \
          trainer.num_nodes=1 \
          trainer.accelerator=gpu \
          trainer.precision=16 \
          retro_model_file=$RETRO_MODEL_PATH \
          retrieval_service.strategy=RetroModelTextGenerationStrategy \
          retrieval_service.neighbors=2 \
          retrieval_service.pad_tokens=True \
          retrieval_service.store_retrieved=True \
          retrieval_service.combo_service.service_port=${COMBO_RETRIEVAL_PORT} \
          port=${RETRO_MODEL_PORT} &

depends_on "0.0.0.0" $RETRO_MODEL_PORT

# launch the web server

python scripts/nlp_language_modeling/service_launch_scripts/start_web_service.py \
          text_service_port=${RETRO_MODEL_PORT} \
          combo_service_port=${COMBO_RETRIEVAL_PORT} \
          share=True \
          username=test \
          password=${PASSWORD} \
          port=${WEB_PORT}


echo "clean up dameons: $$"
clean_up
