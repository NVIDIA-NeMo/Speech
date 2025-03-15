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
coverage run --branch -a --data-file=/workspace/.coverage --source=/workspace/nemo examples/speaker_tasks/diarization/neural_diarizer/multiscale_diar_decoder.py \
    model.diarizer.speaker_embeddings.model_path=titanet_large \
    model.train_ds.batch_size=5 \
    model.validation_ds.batch_size=5 \
    model.train_ds.emb_dir=examples/speaker_tasks/diarization/speaker_diarization_results \
    model.validation_ds.emb_dir=examples/speaker_tasks/diarization/speaker_diarization_results \
    model.train_ds.manifest_filepath=/home/TestData/an4_diarizer/simulated_train/msdd_data.50step.json \
    model.validation_ds.manifest_filepath=/home/TestData/an4_diarizer/simulated_valid/msdd_data.50step.json \
    trainer.devices=1 \
    trainer.accelerator="gpu" \
    +trainer.fast_dev_run=True \
    exp_manager.exp_dir=/tmp/speaker_diarization_results
