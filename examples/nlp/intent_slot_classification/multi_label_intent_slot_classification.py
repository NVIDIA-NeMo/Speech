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

""" 
Sample command to run the script:

python multi_label_intent_slot_classification.py \
            model.data_dir=/home/user/multiatis \
            model.validation_ds.prefix=dev \
            model.test_ds.prefix=dev \
            trainer.devices=[0] \
            +trainer.fast_dev_run=true \
            exp_manager.exp_dir=checkpoints

fast_dev_run=false will save checkpoints for the model
"""


import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

from nemo.collections.nlp.models import MultiLabelIntentSlotClassificationModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


@hydra_runner(config_path="conf", config_name="multi_label_intent_slot_classification_config")
def main(cfg: DictConfig) -> None:
    logging.info(f'Config Params:\n {OmegaConf.to_yaml(cfg)}')
    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))

    # initialize the model using the config file
    model = MultiLabelIntentSlotClassificationModel(cfg.model, trainer=trainer)

    # training
    logging.info("================================================================================================")
    logging.info('Starting training...')
    trainer.fit(model)
    logging.info('Training finished!')

    # Stop further testing as fast_dev_run does not save checkpoints
    if trainer.fast_dev_run:
        return

    # after model training is done, you can load the model from the saved checkpoint
    # and evaluate it on a data file or on given queries.
    logging.info("================================================================================================")
    logging.info("Starting the testing of the trained model on test set...")
    logging.info("We will load the latest model saved checkpoint from the training...")

    # for evaluation and inference you can load the previously trained model saved in .nemo file
    # like this in your code, but we will just reuse the trained model here
    # eval_model = MultiLabelIntentSlotClassificationModel.restore_from(restore_path=checkpoint_path)
    eval_model = model

    # we will setup testing data reusing the same config (test section)
    eval_model.update_data_dir_for_testing(data_dir=cfg.model.data_dir)
    eval_model.setup_test_data(test_data_config=cfg.model.test_ds)

    trainer.test(model=eval_model, ckpt_path=None, verbose=False)
    logging.info("Testing finished!")

    # Optimize Threshold
    eval_model.optimize_threshold(cfg.model.test_ds, 'dev')

    # run an inference on a few examples
    logging.info("======================================================================================")
    logging.info("Evaluate the model on the given queries...")

    # this will work well if you train the model on ATIS dataset
    # for your own dataset change the examples appropriately
    queries = [
        'i would like to find a flight from charlotte to las vegas that makes a stop in st. louis',
        'on april first i need a ticket from tacoma to san jose departing before 7 am',
        'how much is the limousine service in boston',
    ]

    # We use the optimized threshold for predictions
    pred_intents, pred_slots, pred_list = eval_model.predict_from_examples(queries, cfg.model.test_ds)
    logging.info('The prediction results of some sample queries with the trained model:')

    for query, intent, slots in zip(queries, pred_intents, pred_slots):
        logging.info(f'Query : {query}')
        logging.info(f'Predicted Intents: {intent}')
        logging.info(f'Predicted Slots: {slots}')

    logging.info("Inference finished!")


if __name__ == '__main__':
    main()
