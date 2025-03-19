# Copyright (c) 2020-2025, NVIDIA CORPORATION.
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
coverage run --branch -a --data-file=/workspace/.coverage --source=/workspace/nemo examples/asr/speech_multitask/speech_to_text_aed.py \
    model.prompt_format=canary \
    model.model_defaults.asr_enc_hidden=256 \
    model.model_defaults.lm_dec_hidden=256 \
    model.encoder.n_layers=12 \
    model.transf_encoder.num_layers=0 \
    model.transf_decoder.config_dict.num_layers=12 \
    model.train_ds.manifest_filepath=/home/TestData/asr/manifests/canary/an4_canary_train.json \
    model.train_ds.batch_duration=60 \
    model.train_ds.use_bucketing=false \
    model.train_ds.shuffle_buffer_size=100 \
    model.train_ds.num_workers=0 \
    ++model.train_ds.text_field="answer" \
    ++model.train_ds.lang_field="target_lang" \
    model.validation_ds.manifest_filepath=/home/TestData/asr/manifests/canary/an4_canary_val.json \
    ++model.validation_ds.text_field="answer" \
    ++model.validation_ds.lang_field="target_lang" \
    model.validation_ds.num_workers=0 \
    model.test_ds.manifest_filepath=/home/TestData/asr/manifests/canary/an4_canary_val.json \
    ++model.test_ds.text_field="answer" \
    ++model.test_ds.lang_field="target_lang" \
    model.test_ds.num_workers=0 \
    ++spl_tokens.model_dir=/home/TestData/asr_tokenizers/canary/canary_spl_tokenizer_v32 \
    model.tokenizer.langs.en.dir=/home/TestData/asr_tokenizers/canary/en/tokenizer_spe_bpe_v1024_max_4 \
    model.tokenizer.langs.en.type=bpe \
    ++model.tokenizer.langs.es.dir=/home/TestData/asr_tokenizers/canary/es/tokenizer_spe_bpe_v1024_max_4 \
    ++model.tokenizer.langs.es.type=bpe \
    trainer.devices=1 \
    trainer.accelerator="gpu" \
    +trainer.fast_dev_run=True \
    exp_manager.exp_dir=/tmp/speech_to_text_aed_results
