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
coverage run --branch -a --data-file=/workspace/.coverage --source=/workspace/nemo examples/asr/speech_to_text_finetune.py \
    --config-path="conf/asr_finetune" --config-name="speech_to_text_hf_finetune" \
    ~model.train_ds.hf_data_cfg \
    model.train_ds.num_workers=1 \
    model.train_ds.batch_size=2 model.validation_ds.batch_size=2 \
    model.train_ds.streaming=true \
    +model.train_ds.hf_data_cfg.path="librispeech_asr" \
    +model.train_ds.hf_data_cfg.name=null \
    +model.train_ds.hf_data_cfg.split="test.clean" \
    +model.train_ds.hf_data_cfg.streaming=true \
    +model.train_ds.hf_data_cfg.trust_remote_code=True \
    ++model.train_ds.hf_data_cfg.cache_dir=/home/TestData/HF_HOME \
    ~model.validation_ds.hf_data_cfg \
    model.validation_ds.streaming=true \
    +model.validation_ds.hf_data_cfg.path="librispeech_asr" \
    +model.validation_ds.hf_data_cfg.name=null \
    +model.validation_ds.hf_data_cfg.split="test.clean" \
    +model.validation_ds.hf_data_cfg.streaming=true \
    +model.validation_ds.hf_data_cfg.trust_remote_code=True \
    ++model.validation_ds.hf_data_cfg.cache_dir=/home/TestData/HF_HOME \
    ~model.test_ds \
    init_from_nemo_model=/home/TestData/asr/stt_en_fastconformer_transducer_large.nemo \
    model.tokenizer.update_tokenizer=False \
    model.optim.sched.warmup_steps=0 \
    +model.optim.sched.max_steps=3 \
    trainer.max_epochs=null \
    trainer.devices=1 \
    trainer.accelerator="gpu" \
    +trainer.fast_dev_run=True \
    exp_manager.exp_dir=/tmp/speech_finetuning_results
