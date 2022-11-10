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

import os
import tempfile
import uuid

import hydra
from omegaconf import OmegaConf

from nemo.utils import logging


def run_processors(cfg):
    logging.info(f"Hydra config: {OmegaConf.to_yaml(cfg)}")
    processors_to_run = cfg.get("processors_to_run", "all")

    if processors_to_run == "all":
        processors_to_run = ":"
    # converting processors_to_run into Python slice
    processors_to_run = slice(*map(lambda x: int(x.strip()) if x.strip() else None, processors_to_run.split(":")))
    processors_cfgs = cfg.processors[processors_to_run]
    logging.info(
        "Specified to run the following processors: %s ", [cfg["_target_"] for cfg in processors_cfgs],
    )

    processors = []
    # let's build all processors first to automatically check
    # for errors in parameters
    with tempfile.TemporaryDirectory() as tmp_dir:
        for idx, processor_cfg in enumerate(processors_cfgs):
            logging.info('=> Building processor "%s"', processor_cfg["_target_"])

            # we assume that each processor defines "output_manifest_file"
            # and "input_manifest_file" keys, which can be optional. In case they
            # are missing, we create tmp files here for them
            if "output_manifest_file" not in processor_cfg:
                tmp_file_path = os.path.join(tmp_dir, str(uuid.uuid4()))
                OmegaConf.set_struct(processor_cfg, False)
                processor_cfg["output_manifest_file"] = tmp_file_path
                OmegaConf.set_struct(processor_cfg, True)
                if idx != len(processors_cfgs) - 1 and "input_manifest_file" not in processors_cfgs[idx + 1]:
                    OmegaConf.set_struct(processors_cfgs[idx + 1], False)
                    processors_cfgs[idx + 1]["input_manifest_file"] = tmp_file_path
                    OmegaConf.set_struct(processors_cfgs[idx + 1], True)

            processor = hydra.utils.instantiate(processor_cfg)
            # running runtime tests to fail right-away if something is not
            # matching users expectations
            processor.test()
            processors.append(processor)

        for processor in processors:
            # TODO: add proper str method to all classes for good display
            logging.info('=> Running processor "%s"', processor)
            processor.process()
