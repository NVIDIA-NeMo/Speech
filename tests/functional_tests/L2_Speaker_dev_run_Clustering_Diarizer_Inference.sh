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
coverage run --branch -a --data-file=/workspace/.coverage --source=/workspace/nemo examples/speaker_tasks/diarization/clustering_diarizer/offline_diar_infer.py \
    diarizer.manifest_filepath=/home/TestData/an4_diarizer/an4_manifest.json \
    diarizer.speaker_embeddings.model_path=/home/TestData/an4_diarizer/spkr.nemo \
    diarizer.speaker_embeddings.parameters.save_embeddings=True \
    diarizer.speaker_embeddings.parameters.window_length_in_sec=1.5 \
    diarizer.speaker_embeddings.parameters.shift_length_in_sec=0.75 \
    diarizer.speaker_embeddings.parameters.multiscale_weights=null \
    diarizer.vad.model_path=/home/TestData/an4_diarizer/MatchboxNet_VAD_3x2.nemo \
    diarizer.out_dir=/tmp/clustering_diarizer_results
