# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
This script contains an example of how to train and test dialogue models in NeMo.

***Setting the configs***
The model and the PT trainer are defined in a config file that declares multiple important sections.
The most important ones are:
    model: All arguments that are related to the Model - model, loss, optimizer,
            schedulers, and datasets/data loaders.
    trainer: Any argument to be passed to PyTorch Lightning including number of epochs, number of GPUs,
            precision level, etc.

This script uses the `/examples/nlp/dialogue_state_tracking/conf/dialog_config.yaml` config file
by default. You may update the config file from the file directly. The other option is to set another config file via command-line arguments by `--config-name=CONFIG_FILE_PATH'.


***Model Training***
    python dialogue.py
    do_training=True
    model.dataset.data_dir=<DATA_DIR_WITH_JSON_DATA>
    model.dataset.dialogues_example_dir=<DAT_DIR_FOR_CACHING_INTERMEDIATE_AND_SAVING_PREDICTIONS>
    model.dataset.task=<TASK - see conf/dialogue_config.yaml for full list> e.g. sgd
    model.language_model.pretrained_model_name=<TASK - see conf/dialogue_config.yaml for full list> e.g. gpt2
    trainer.devices=[<DEVICE_IDS_TO_USE>]

***Model Evaluation***
    command as above, change do_training=False
"""

import os

import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

from nemo.collections.nlp.models.dialogue.dialogue_gpt_classification_model import DialogueGPTClassificationModel
from nemo.collections.nlp.models.dialogue.dialogue_gpt_generation_model import DialogueGPTGenerationModel
from nemo.collections.nlp.models.dialogue.dialogue_nearest_neighbour_model import DialogueNearestNeighbourModel
from nemo.collections.nlp.models.dialogue.dialogue_s2s_generation_model import DialogueS2SGenerationModel
from nemo.collections.nlp.models.dialogue.dialogue_zero_shot_intent_model import DialogueZeroShotIntentModel
from nemo.collections.nlp.models.dialogue.intent_slot_classification_model import IntentSlotClassificationModel
from nemo.collections.nlp.models.dialogue.sgdqa_model import SGDQAModel
from nemo.collections.nlp.modules.common.megatron.megatron_utils import compute_model_parallel_rank
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.app_state import AppState
from nemo.utils.exp_manager import exp_manager


@hydra_runner(config_path="conf", config_name="dialogue_config")
def main(cfg: DictConfig) -> None:
    pl.seed_everything(42)
    logging.info(f'Config: {OmegaConf.to_yaml(cfg)}')

    try:
        strategy = NLPDDPStrategy(no_ddp_communication_hook=True, find_unused_parameters=True,)
    except (ImportError, ModuleNotFoundError):
        strategy = 'auto'

    trainer = pl.Trainer(**cfg.trainer, strategy=strategy)

    exp_manager(trainer, cfg.get("exp_manager", None))

    app_state = AppState()
    app_state.data_parallel_size = cfg.model.data_parallel_size
    if cfg.model.tensor_model_parallel_size > 1:
        app_state.model_parallel_size = cfg.model.tensor_model_parallel_size
        app_state.tensor_model_parallel_rank = compute_model_parallel_rank(
            trainer.local_rank, app_state.model_parallel_size
        )

    if 'bert' in cfg.model.language_model.pretrained_model_name:
        if cfg.model.dataset.task == 'sgd':
            if cfg.model.original_nemo_checkpoint is not None:
                model_class = DialogueZeroShotIntentModel
            else:
                model_class = SGDQAModel
        elif cfg.model.dataset.task in ['zero_shot', 'design']:
            model_class = DialogueZeroShotIntentModel
        else:
            model_class = IntentSlotClassificationModel
    elif 'gpt' in cfg.model.language_model.pretrained_model_name.lower():
        if cfg.model.dataset.task in ['ms_marco', 'mellon_qa']:
            model_class = DialogueGPTGenerationModel
        else:
            model_class = DialogueGPTClassificationModel
    elif (
        'bart' in cfg.model.language_model.pretrained_model_name.lower()
        or 't5' in cfg.model.language_model.pretrained_model_name.lower()
    ):
        # please use bf16/32 with t5-large and above
        # see https://github.com/huggingface/transformers/pull/10956
        model_class = DialogueS2SGenerationModel
    elif 'sentence-transformers' in cfg.model.language_model.pretrained_model_name.lower():
        model_class = DialogueNearestNeighbourModel

    if cfg.pretrained_model or (cfg.model.nemo_path and os.path.exists(cfg.model.nemo_path)):
        if cfg.pretrained_model:
            logging.info(f'Loading pretrained model {cfg.pretrained_model}')
            model = model_class.from_pretrained(cfg.pretrained_model)
        else:
            logging.info(f'Restoring model from {cfg.model.nemo_path}')
            model = model_class.restore_from(cfg.model.nemo_path, trainer=trainer)

        if cfg.do_training:
            model.setup_training_data(train_data_config=cfg.model.train_ds)
            model.setup_multiple_validation_data(val_data_config=cfg.model.validation_ds)
    else:
        logging.info(f'Config: {OmegaConf.to_yaml(cfg)}')
        model = model_class(cfg.model, trainer=trainer)

    if cfg.do_training:
        trainer.fit(model)
        if cfg.model.nemo_path:
            if not os.path.exists(cfg.model.nemo_path):
                model.save_to(cfg.model.nemo_path)
            else:
                updated_nemo_path = cfg.model.nemo_path.replace(".nemo", "_new.nemo")
                logging.warning("nemo path exists, saving at {} instead".format(updated_nemo_path))
                model.save_to(updated_nemo_path)

    else:
        data_dir = cfg.model.dataset.get('data_dir', None)
        dialogues_example_dir = cfg.model.dataset.get('dialogues_example_dir', None)

        if data_dir is None or dialogues_example_dir is None:
            raise ValueError('No dataset directory provided. Skipping evaluation. ')
        elif not os.path.exists(data_dir):
            raise ValueError(f'{data_dir} is not found, skipping evaluation on the test set.')
        else:
            if hasattr(model, "update_data_dirs"):
                model.update_data_dirs(data_dir=data_dir, dialogues_example_dir=dialogues_example_dir)
                model._cfg.dataset = cfg.model.dataset

    if hasattr(cfg.model, 'test_ds') and cfg.model.test_ds.ds_item is not None:
        model.setup_multiple_test_data(test_data_config=cfg.model.test_ds)
        trainer.test(model)


if __name__ == '__main__':
    main()
