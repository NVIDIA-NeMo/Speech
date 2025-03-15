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
coverage run --branch -a --data-file=/workspace/.coverage --source=/workspace/nemo tools/speech_data_simulator/multispeaker_simulator.py \
    --config-path=conf --config-name=data_simulator.yaml \
    data_simulator.random_seed=42 \
    data_simulator.manifest_filepath=/home/TestData/LibriSpeechShort/dev-clean-align-short.json \
    data_simulator.outputs.output_dir=/tmp/test_simulator \
    data_simulator.session_config.num_sessions=2 \
    data_simulator.session_config.session_length=60
