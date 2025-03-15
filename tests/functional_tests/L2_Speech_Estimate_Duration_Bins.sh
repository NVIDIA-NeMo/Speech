# Copyright (c) 2020-2021, NVIDIA CORPORATION.
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
set -x
# 1D buckets [SSL, CTC]
coverage run --branch -a --data-file=/workspace/.coverage --source=/workspace/nemo scripts/speech_recognition/estimate_duration_bins.py \
  /home/TestData/an4_dataset/an4_train.json \
  --buckets 5
# 2D buckets [CTC, RNNT, TDT] / with tokenizer
coverage run --branch -a --data-file=/workspace/.coverage --source=/workspace/nemo scripts/speech_recognition/estimate_duration_bins_2d.py \
  /home/TestData/an4_dataset/an4_train_lang.json \
  --tokenizer /home/TestData/asr_tokenizers/canary/en/tokenizer_spe_bpe_v1024_max_4/tokenizer.model \
  --buckets 5 \
  --sub-buckets 2
# TODO(pzelasko): Figure out how to quote the value in the test properly for CI to accept it...
# 2D buckets with prompt [AED/Canary, SpeechLM] / with aggregate tokenizer + prompt format
# coverage run --branch -a --data-file=/workspace/.coverage --source=/workspace/nemo scripts/speech_recognition/estimate_duration_bins_2d.py \
#   /home/TestData/an4_dataset/an4_train_lang.json \
#   --tokenizer /home/TestData/asr_tokenizers/canary/canary_spl_tokenizer_v32/tokenizer.model \
#      /home/TestData/asr_tokenizers/canary/en/tokenizer_spe_bpe_v1024_max_4/tokenizer.model \
#      /home/TestData/asr_tokenizers/canary/es/tokenizer_spe_bpe_v1024_max_4/tokenizer.model \
#   --langs spl_tokens en es \
#   --prompt-format canary \
#   --prompt '[{"role":"user","slots":{"source_lang":"en","target_lang":"en","task":"asr","pnc":"yes"}}]' \
#   --buckets 5 \
#   --sub-buckets 2
