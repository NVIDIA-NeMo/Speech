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
    coverage run --branch -a --data-file=/workspace/.coverage --source=/workspace/nemo enc_dec_nmt.py \
        --config-path=conf \
        --config-name=aayn_base \
        do_training=false \
        model.preproc_out_dir=$PWD/preproc_out_dir \
        model.train_ds.use_tarred_dataset=true \
        model.train_ds.n_preproc_jobs=2 \
        model.train_ds.lines_per_dataset_fragment=500 \
        model.train_ds.num_batches_per_tarfile=10 \
        model.train_ds.src_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
        model.train_ds.tgt_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.ref \
        model.validation_ds.src_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
        model.validation_ds.tgt_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
        model.encoder_tokenizer.vocab_size=2000 \
        model.decoder_tokenizer.vocab_size=2000 \
        ~model.test_ds \
        trainer.devices=1 \
        trainer.accelerator="gpu" \
        +trainer.fast_dev_run=true \
        exp_manager=null
