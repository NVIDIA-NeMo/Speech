# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

import lightning.pytorch as pl
from omegaconf import DictConfig

from nemo.collections.nlp.models import TokenClassificationModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


"""
This script shows how to perform evaluation and runs inference of a few examples.

More details on Token Classification model could be found in tutorials/nlp/Token_Classification_Named_Entity_Recognition.ipynb

*** Setting the configs ***

This script uses the `/examples/nlp/token_classification/conf/token_classification_config.yaml` config file
by default. You may update the config file from the file directly. 
The other option is to set another config file via command line arguments by `--config-name=CONFIG_FILE_PATH'.

For more details about the config files and different ways of model restoration, see tutorials/00_NeMo_Primer.ipynb

*** Model Evaluation ***

The script runs two types of evaluation: 
    * model.test() - this eval will use the config setting for evaluation such as model.dataset.max_seq_length
    * model.evaluate_from_file():
        * disregards model.dataset.max_seq_length and evaluates all the tokens, BERT max seq length - 512 tokens after tokenization
        * creates confusion matrix
        * saves predictions and labels (if provided)

To run the script:

    python token_classification_evaluate.py \
    model.dataset.data_dir=<PATH_TO_DATA_DIR>  \
    pretrained_model=ner_en_bert 

<PATH_TO_DATA_DIR> - a directory that contains test_ds.text_file and test_ds.labels_file (see the config)
pretrained_model   - pretrained TokenClassification model from list_available_models() or 
                     path to a .nemo file, for example: ner_en_bert or your_model.nemo

"""


@hydra_runner(config_path="conf", config_name="token_classification_config")
def main(cfg: DictConfig) -> None:
    logging.info(
        'During evaluation/testing, it is currently advisable to construct a new Trainer with single GPU and \
            no DDP to obtain accurate results'
    )

    if not hasattr(cfg.model, 'test_ds'):
        raise ValueError(f'model.test_ds was not found in the config, skipping evaluation')

    trainer = pl.Trainer(
        devices=1,
        precision=cfg.trainer.precision,
        logger=False,
        enable_checkpointing=False,
        accelerator=cfg.trainer.accelerator,
    )
    exp_dir = exp_manager(trainer, cfg.exp_manager)

    if not cfg.pretrained_model:
        raise ValueError(
            'To run evaluation and inference script a pre-trained model or .nemo file must be provided.'
            f'Choose from {TokenClassificationModel.list_available_models()} or "pretrained_model"="your_model.nemo"'
        )

    if os.path.exists(cfg.pretrained_model):
        model = TokenClassificationModel.restore_from(cfg.pretrained_model)
    elif cfg.pretrained_model in TokenClassificationModel.get_available_model_names():
        model = TokenClassificationModel.from_pretrained(cfg.pretrained_model)
    else:
        raise ValueError(
            f'Provide path to the pre-trained .nemo checkpoint or choose from {TokenClassificationModel.list_available_models()}'
        )

    data_dir = cfg.model.dataset.get('data_dir', None)
    if data_dir is None:
        logging.error(
            'No dataset directory provided. Skipping evaluation. '
            'To run evaluation on a file, specify path to the directory that contains test_ds.text_file and test_ds.labels_file with "model.dataset.data_dir" argument.'
        )
    elif not os.path.exists(data_dir):
        logging.error(f'{data_dir} is not found, skipping evaluation on the test set.')
    else:
        model.update_data_dir(data_dir=data_dir)
        model._cfg.dataset = cfg.model.dataset

        if not hasattr(cfg.model, 'test_ds'):
            logging.error(f'model.test_ds was not found in the config, skipping evaluation')
        elif model.prepare_test(trainer):
            model.setup_test_data(cfg.model.test_ds)
            trainer.test(model)

            model.evaluate_from_file(
                text_file=os.path.join(data_dir, cfg.model.test_ds.text_file),
                labels_file=os.path.join(data_dir, cfg.model.test_ds.labels_file),
                output_dir=exp_dir,
                add_confusion_matrix=True,
                normalize_confusion_matrix=True,
            )
        else:
            logging.error('Skipping the evaluation. The trainer is not setup properly.')

    # run an inference on a few examples
    queries = ['we bought four shirts from the nvidia gear store in santa clara.', 'Nvidia is a company.']
    results = model.add_predictions(queries, output_file='predictions.txt')

    for query, result in zip(queries, results):
        logging.info(f'Query : {query}')
        logging.info(f'Result: {result.strip()}\n')

    logging.info(f'Results are saved at {exp_dir}')


if __name__ == '__main__':
    main()
