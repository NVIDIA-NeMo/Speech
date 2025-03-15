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
coverage run --branch -a --data-file=/workspace/.coverage --source=/workspace/nemo examples/speaker_tasks/recognition/speaker_reco.py \
    model.train_ds.batch_size=10 \
    model.validation_ds.batch_size=2 \
    model.train_ds.manifest_filepath=/home/TestData/an4_speaker/train.json \
    model.validation_ds.manifest_filepath=/home/TestData/an4_speaker/dev.json \
    model.decoder.num_classes=2 \
    trainer.max_epochs=10 \
    trainer.devices=1 \
    trainer.accelerator="gpu" \
    +trainer.fast_dev_run=True \
    exp_manager.exp_dir=/tmp/speaker_recognition_results
