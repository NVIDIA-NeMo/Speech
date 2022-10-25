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

from nemo.collections.asr.data.data_simulation import RIRMixGenerator
from nemo.core.config import hydra_runner


"""
This script creates a corpus of signals using RIRs, speech and noise.

Usage:
    python rir_mix_generator.py --config-path PATH_TO_CONFIG_DIR --config-name CONFIG_NAME output_dir=OUTPUT_DIR

Details of the configuration can be found in the example configuration files in conf/*
"""


@hydra_runner(config_path="conf", config_name="rir_mix.yaml")
def main(cfg):
    rir_mix = RIRMixGenerator(cfg=cfg)
    rir_mix.generate()


if __name__ == "__main__":
    main()
