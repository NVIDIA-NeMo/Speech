# SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

from nemo.collections.audio.data.data_simulation import RIRCorpusGenerator
from nemo.core.config import hydra_runner


"""
This script creates a corpus of room impulse responses.

Usage:
    python rir_corpus_generator.py --config-path PATH_TO_CONFIG_DIR --config-name CONFIG_NAME output_dir=OUTPUT_DIR

Details of the configuration can be found in the example configuration files in conf/*
"""


@hydra_runner(config_path="conf", config_name="rir_corpus.yaml")
def main(cfg):
    room_corpus = RIRCorpusGenerator(cfg=cfg)
    room_corpus.generate()


if __name__ == "__main__":
    main()
