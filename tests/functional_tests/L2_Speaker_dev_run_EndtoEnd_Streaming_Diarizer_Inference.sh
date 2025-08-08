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
coverage run -a --data-file=/workspace/.coverage --source=/workspace/nemo examples/speaker_tasks/diarization/neural_diarizer/e2e_diarize_speech.py \
    model_path=/home/TestData/an4_diarizer/diar_streaming_sortformer_4spk-v2-tiny.nemo \
    dataset_manifest=/home/TestData/an4_diarizer/simulated_valid/eesd_valid_tiny.json \
    batch_size=1