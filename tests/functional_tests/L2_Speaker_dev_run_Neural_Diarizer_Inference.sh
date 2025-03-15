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
coverage run --branch -a --data-file=/workspace/.coverage --source=/workspace/nemo examples/speaker_tasks/diarization/neural_diarizer/multiscale_diar_decoder_infer.py \
    diarizer.manifest_filepath=/home/TestData/an4_diarizer/an4_manifest.json \
    diarizer.msdd_model.model_path=/home/TestData/an4_diarizer/diar_msdd_telephonic.nemo \
    diarizer.speaker_embeddings.parameters.save_embeddings=True \
    diarizer.vad.model_path=/home/TestData/an4_diarizer/MatchboxNet_VAD_3x2.nemo \
    diarizer.out_dir=/tmp/neural_diarizer_results
