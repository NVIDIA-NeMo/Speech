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
coverage run --branch -a --data-file=/workspace/.coverage --source=/workspace/nemo examples/tts/fastpitch.py \
    --config-name fastpitch_align_v1.05 \
    train_dataset=/home/TestData/an4_dataset/an4_train.json \
    validation_datasets=/home/TestData/an4_dataset/an4_val.json \
    sup_data_path=/home/TestData/an4_dataset/beta_priors \
    trainer.devices="[0]" \
    +trainer.limit_train_batches=1 \
    +trainer.limit_val_batches=1 \
    trainer.max_epochs=1 \
    trainer.strategy=auto \
    model.pitch_mean=212.35873413085938 \
    model.pitch_std=68.52806091308594 \
    model.train_ds.dataloader_params.batch_size=4 \
    model.train_ds.dataloader_params.num_workers=0 \
    model.validation_ds.dataloader_params.batch_size=4 \
    model.validation_ds.dataloader_params.num_workers=0 \
    model.symbols_embedding_dim=64 \
    model.input_fft.d_inner=384 \
    model.input_fft.n_layer=2 \
    model.output_fft.d_inner=384 \
    model.output_fft.n_layer=2 \
    ~trainer.check_val_every_n_epoch \
    ~model.text_normalizer \
    ~model.text_normalizer_call_kwargs
