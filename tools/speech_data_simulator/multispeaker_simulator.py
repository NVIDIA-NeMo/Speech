# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

from multiprocessing import set_start_method

from nemo.collections.asr.data.data_simulation import MultiSpeakerSimulator, RIRMultiSpeakerSimulator
from nemo.core.config import hydra_runner


"""
This script creates a synthetic diarization session using the provided audio dataset with ctm files.
Usage:
  python <NEMO_ROOT>/tools/speech_data_simulator/multispeaker_simulator.py  \
    num_workers=10 \
    data_simulator.random_seed=42 \
    data_simulator.manifest_filepath=manifest_with_alignment_file.json \
    data_simulator.outputs.output_dir=./simulated_data \
    data_simulator.outputs.output_filename=sim_spk2_sess20 \
    data_simulator.session_config.num_sessions=1000 \
    data_simulator.session_config.num_speakers=2 \
    data_simulator.session_config.session_length=20 \
    data_simulator.background_noise.add_bg=False \
    data_simulator.background_noise.background_manifest=background_noise.json \
    data_simulator.background_noise.snr=40 \

Check out parameters in ./conf/data_simulator.yaml.
"""


@hydra_runner(config_path="conf", config_name="data_simulator.yaml")
def main(cfg):
    if cfg.data_simulator.rir_generation.use_rir:
        simulator = RIRMultiSpeakerSimulator(cfg=cfg)
    else:
        simulator = MultiSpeakerSimulator(cfg=cfg)

    set_start_method('spawn', force=True)
    simulator.generate_sessions()


if __name__ == "__main__":
    main()
