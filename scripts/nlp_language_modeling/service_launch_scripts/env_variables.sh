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

MERGE_FILE=
VOCAB_FILE=

BERT_DEVICES=\'0,1,2,3\'
BERT_PORT=17190
CONTEXT_BERT_PORT=17191
QUERY_BERT_PORT=17192

STATIC_FAISS_INDEX=
STATIC_RETRIVAL_DB=
STATIC_RETRIEVAL_PORT=17179

DYNAMIC_RETRIEVAL_PORT=17180
COMBO_RETRIEVAL_PORT=17181

RETRO_MODEL_PATH=
RETRO_MODEL_PORT=5555
WEB_PORT=7777
PASSWORD=test2