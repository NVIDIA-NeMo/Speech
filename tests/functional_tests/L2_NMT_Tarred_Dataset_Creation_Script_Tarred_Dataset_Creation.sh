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
cd examples/nlp/machine_translation &&
    coverage run --branch -a --data-file=/workspace/.coverage --source=/workspace/nemo create_tarred_parallel_dataset.py \
        --src_fname /home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
        --tgt_fname /home/TestData/nlp/nmt/toy_data/wmt14-de-en.ref \
        --out_dir $PWD/out_dir \
        --encoder_tokenizer_vocab_size=2000 \
        --decoder_tokenizer_vocab_size=2000 \
        --tokens_in_batch=1000 \
        --lines_per_dataset_fragment=500 \
        --num_batches_per_tarfile=10 \
        --n_preproc_jobs=2
