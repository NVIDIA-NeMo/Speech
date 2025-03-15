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
    coverage run --branch -a --data-file=/workspace/.coverage --source=/workspace/nemo enc_dec_nmt_finetune.py \
        model_path=/home/TestData/nlp/nmt/toy_data/enes_v16k_s100k_6x6.nemo \
        trainer.devices=1 \
        ~trainer.max_epochs \
        model.train_ds.src_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
        model.train_ds.tgt_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.ref \
        model.validation_ds.src_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
        model.validation_ds.tgt_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
        model.test_ds.src_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
        model.test_ds.tgt_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
        +trainer.val_check_interval=10 \
        +trainer.limit_val_batches=1 \
        +trainer.limit_test_batches=1 \
        +trainer.max_steps=10 \
        +exp_manager.exp_dir=examples/nlp/machine_translation/nmt_finetune \
        +exp_manager.create_checkpoint_callback=True \
        +exp_manager.checkpoint_callback_params.monitor=val_sacreBLEU \
        +exp_manager.checkpoint_callback_params.mode=max \
        +exp_manager.checkpoint_callback_params.save_best_model=true
