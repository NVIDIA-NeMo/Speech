# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

"""
## Tasks
This script works with all GLUE Benchmark tasks, more details about the GLUE Benchmark could be found at
https://gluebenchmark.com/

More details on how to use this script could be found in tutorials/nlp/GLUE_Benchmark.ipynb

## Model Training

To train GLUEModel with the default config file, run:
    python glue_benchmark.py \
    model.dataset.data_dir=<PATH_TO_DATA_DIR>  \
    model.task_name=TASK_NAME \
    trainer.max_epochs=<NUM_EPOCHS> \
    trainer.devices="[<CHANGE_TO_GPU_YOU_WANT_TO_USE>]

Supported task names:
["cola", "sst-2", "mrpc", "sts-b", "qqp", "mnli", "qnli", "rte", "wnli"]
Note, MNLI task includes both matched and mismatched dev sets
"""

import os

import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

from nemo.collections.nlp.models import GLUEModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


@hydra_runner(config_name="glue_benchmark_config")
def main(cfg: DictConfig) -> None:
    # PTL 2.0 has find_unused_parameters as False by default, so its required to set it to True
    # when there are unused parameters like here
    if cfg.trainer.strategy == 'ddp':
        cfg.trainer.strategy = "ddp_find_unused_parameters_true"
    logging.info(f'Config: {OmegaConf.to_yaml(cfg)}')
    trainer = pl.Trainer(**cfg.trainer)
    exp_manager_cfg = cfg.get("exp_manager", None)

    if exp_manager_cfg:
        exp_manager_cfg.name = cfg.model.task_name
        logging.info(f'Setting task_name to {exp_manager_cfg.name} in exp_manager')
    exp_manager(trainer, exp_manager_cfg)

    if cfg.model.nemo_path and os.path.exists(cfg.model.nemo_path):
        model = GLUEModel.restore_from(cfg.model.nemo_path)
        logging.info(f'Restoring model from {cfg.model.nemo_path}')
        model.update_data_dir(data_dir=cfg.model.dataset.data_dir)
        model.setup_training_data()
        model.setup_multiple_validation_data()
        trainer.fit(model)
    else:
        model = GLUEModel(cfg.model, trainer=trainer)
        trainer.fit(model)
        if cfg.model.nemo_path:
            model.save_to(cfg.model.nemo_path)


if __name__ == '__main__':
    main()
