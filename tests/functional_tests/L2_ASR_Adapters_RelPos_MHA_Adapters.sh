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
coverage run --branch -a --data-file=/workspace/.coverage --source=/workspace/nemo examples/asr/asr_adapters/train_asr_adapter.py \
    model.pretrained_model="stt_en_conformer_ctc_small" \
    model.adapter.adapter_name="encoder:an4" \
    model.adapter.adapter_type="tiny_attn" \
    model.adapter.tiny_attn.n_feat=176 \
    model.train_ds.manifest_filepath=/home/TestData/an4_dataset/an4_train.json \
    model.validation_ds.manifest_filepath=/home/TestData/an4_dataset/an4_val.json \
    trainer.max_steps=5 \
    trainer.devices=1 \
    trainer.accelerator="gpu" \
    +trainer.fast_dev_run=True \
    exp_manager.exp_dir=/tmp/speech_to_text_adapters_mha_results
