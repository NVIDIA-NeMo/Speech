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
coverage run -a --data-file=/workspace/.coverage --source=/workspace/nemo examples/speaker_tasks/diarization/neural_diarizer/streaming_sortformer_diar_train.py \
    trainer.devices="[0]" \
    batch_size=3 \
    model.train_ds.manifest_filepath=/home/TestData/an4_diarizer/simulated_train/eesd_train_tiny.json \
    model.test_ds.manifest_filepath=/home/TestData/an4_diarizer/simulated_valid/eesd_valid_tiny.json \
    model.validation_ds.manifest_filepath=/home/TestData/an4_diarizer/simulated_valid/eesd_valid_tiny.json \
    exp_manager.exp_dir=/tmp/speaker_diarization_results \
    +trainer.fast_dev_run=True