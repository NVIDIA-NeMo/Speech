# Copyright (c) 2026, NVIDIA CORPORATION.
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
HF_HUB_OFFLINE=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 coverage run -a --data-file=/workspace/.coverage --source=/workspace/nemo examples/tts/audio_codec.py \
    --config-name audio_codec_22050.yaml \
    +train_ds_meta.an4.manifest_path="/home/TestData/an4_dataset/an4_train_context_v1.json" \
    +train_ds_meta.an4.audio_dir="/" \
    +train_ds_meta.an4.sample_weight=1.0 \
    +val_ds_meta.an4.manifest_path="/home/TestData/an4_dataset/an4_val_context_v1.json" \
    +val_ds_meta.an4.audio_dir="/" \
    +log_ds_meta.an4.manifest_path="/home/TestData/an4_dataset/an4_val_context_v1.json" \
    +log_ds_meta.an4.audio_dir="/" \
    log_dir="/tmp/audio_codec_training_output" \
    max_epochs=1 \
    batch_size=4 \
    weighted_sampling_steps_per_epoch=10 \
    +trainer.limit_val_batches=1 \
    trainer.devices="[0]" \
    trainer.strategy=auto \
    model.train_ds.dataloader_params.num_workers=0 \
    model.validation_ds.dataloader_params.num_workers=0 \
    ~trainer.check_val_every_n_epoch
