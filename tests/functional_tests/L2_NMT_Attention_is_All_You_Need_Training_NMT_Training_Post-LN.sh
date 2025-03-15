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
coverage run --branch -a --data-file=/workspace/.coverage --source=/workspace/nemo examples/nlp/machine_translation/enc_dec_nmt.py \
  --config-path=conf \
  --config-name=aayn_base \
  do_testing=false \
  model.train_ds.src_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
  model.train_ds.tgt_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.ref \
  model.validation_ds.src_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
  model.validation_ds.tgt_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
  model.test_ds.src_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
  model.test_ds.tgt_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
  model.encoder_tokenizer.tokenizer_model=/home/TestData/nlp/nmt/toy_data/spm_4k_ende.model \
  model.decoder_tokenizer.tokenizer_model=/home/TestData/nlp/nmt/toy_data/spm_4k_ende.model \
  model.encoder.num_layers=1 \
  model.encoder.hidden_size=64 \
  model.encoder.inner_size=256 \
  model.decoder.num_layers=1 \
  model.decoder.hidden_size=64 \
  model.decoder.inner_size=256 \
  +model.optim.capturable=True \
  trainer.devices=1 \
  trainer.accelerator="gpu" \
  +trainer.val_check_interval=2 \
  +trainer.limit_val_batches=1 \
  +trainer.max_steps=2 \
  trainer.precision=16 \
  +exp_manager.explicit_log_dir=examples/nlp/machine_translation/nmt_results \
  +exp_manager.create_checkpoint_callback=true

coverage run --branch -a --data-file=/workspace/.coverage --source=/workspace/nemo examples/nlp/machine_translation/enc_dec_nmt.py \
  --config-path=conf \
  --config-name=aayn_base \
  do_testing=true \
  model.train_ds.src_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
  model.train_ds.tgt_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.ref \
  model.validation_ds.src_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
  model.validation_ds.tgt_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
  model.test_ds.src_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
  model.test_ds.tgt_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
  model.encoder_tokenizer.tokenizer_model=/home/TestData/nlp/nmt/toy_data/spm_4k_ende.model \
  model.decoder_tokenizer.tokenizer_model=/home/TestData/nlp/nmt/toy_data/spm_4k_ende.model \
  model.encoder.num_layers=1 \
  model.encoder.hidden_size=64 \
  model.encoder.inner_size=256 \
  model.decoder.num_layers=1 \
  model.decoder.hidden_size=64 \
  model.decoder.inner_size=256 \
  +model.optim.capturable=True \
  trainer.devices=1 \
  trainer.accelerator="gpu" \
  +trainer.val_check_interval=10 \
  +trainer.limit_val_batches=1 \
  +trainer.limit_test_batches=1 \
  +trainer.max_steps=10 \
  +exp_manager.explicit_log_dir=examples/nlp/machine_translation/nmt_results \
  +exp_manager.create_checkpoint_callback=true \
  +exp_manager.resume_if_exists=True
