# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
This script contains an example on how to train SpellMapper (SpellcheckingAsrCustomizationModel).
It uses the `examples/nlp/spellchecking_asr_customization/conf/spellchecking_asr_customization_config.yaml`
config file by default. The other option is to set another config file via command
line arguments by `--config-name=CONFIG_FILE_PATH'. Probably it is worth looking
at the example config file to see the list of parameters used for training.

USAGE Example:
    See `examples/nlp/spellchecking_asr_customization/run_training.sh` for training on non-tarred data.
    and
    `examples/nlp/spellchecking_asr_customization/run_training_tarred.sh` for training on tarred data.

One (non-tarred) training example should consist of 4 tab-separated columns:
    1. text of ASR-hypothesis
    2. texts of 10 candidates separated by semicolon
    3. 1-based ids of correct candidates, or 0 if none
    4. start/end coordinates of correct candidates (correspond to ids in third column)
Example (in one line):
    a s t r o n o m e r s _ d i d i e _ s o m o n _ a n d _ t r i s t i a n _ g l l o
    d i d i e r _ s a u m o n;a s t r o n o m i e;t r i s t a n _ g u i l l o t;t r i s t e s s e;m o n a d e;c h r i s t i a n;a s t r o n o m e r;s o l o m o n;d i d i d i d i d i;m e r c y
    1 3
    CUSTOM 12 23;CUSTOM 28 41
"""

from helpers import MODEL, instantiate_model_and_trainer
from omegaconf import DictConfig, OmegaConf

from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


@hydra_runner(config_path="conf", config_name="spellchecking_asr_customization_config")
def main(cfg: DictConfig) -> None:
    # PTL 2.0 has find_unused_parameters as False by default, so its required to set it to True
    # when there are unused parameters like here
    if cfg.trainer.strategy == 'ddp':
        cfg.trainer.strategy = "ddp_find_unused_parameters_true"
    logging.info(f'Config Params: {OmegaConf.to_yaml(cfg)}')

    # Train the model
    if cfg.model.do_training:
        logging.info(
            "================================================================================================"
        )
        logging.info('Start training...')
        trainer, model = instantiate_model_and_trainer(cfg, MODEL, True)
        spellchecking_exp_manager = cfg.get('exp_manager', None)
        exp_manager(trainer, spellchecking_exp_manager)
        trainer.fit(model)
        logging.info('Training finished!')


if __name__ == '__main__':
    main()
