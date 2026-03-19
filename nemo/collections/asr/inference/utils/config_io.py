# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import os

from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig


def read_config(config_path: str, config_name: str) -> DictConfig:
    """
    Read configuration file
    Args:
        config_path: (str) Absolute path to the configuration file
        config_name: (str) Name of the configuration file
    Returns:
        (DictConfig) Configuration object
    """

    rel_to = os.path.dirname(os.path.realpath(__file__))

    # Reset the global Hydra instance if already initialized to prevent duplicate initialization errors
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with initialize(version_base=None, config_path=os.path.relpath(config_path, rel_to)):
        cfg = compose(config_name=config_name)
    return cfg
