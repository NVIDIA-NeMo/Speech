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

import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

from nemo.collections.nlp.models import TokenClassificationModel
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


"""
This scripts shows how to train a Token Classification model.

The Token Classification model supports Named Entity Recognition task and other token level classification tasks,
as long as the data follows the format specified below.

More details on how to use this script could be found in 
tutorials/nlp/Token_Classification_Named_Entity_Recognition.ipynb

*** Data Format *** 
Token Classification Model requires the data to be split into 2 files: text.txt and labels.txt.
Each line of the text.txt file contains text sequences, where words are separated with spaces, i.e.:
[WORD] [SPACE] [WORD] [SPACE] [WORD].
The labels.txt file contains corresponding labels for each word in text.txt, the labels are separated with spaces, i.e.:
[LABEL] [SPACE] [LABEL] [SPACE] [LABEL].

Example of a text.txt file:
Jennifer is from New York City .
She likes ...
...

Corresponding labels.txt file:
B-PER O O B-LOC I-LOC I-LOC O
O O ...
...

*** Preparing the dataset ***

To convert an IOB format data to the format required for training, run
examples/nlp/token_classification/data/import_from_iob_format.py on your train and dev files, as follows:

python examples/nlp/token_classification/data/import_from_iob_format.py --data_file PATH_TO_IOB_FORMAT_DATAFILE

*** Setting the configs ***

The model and the PT trainer are defined in a config file which declares multiple important sections.
The most important ones are:
    model: All arguments that are related to the Model - language model, tokenizer, token classifier, optimizer,
            schedulers, and datasets/data loaders.
    trainer: Any argument to be passed to PyTorch Lightning including number of epochs, number of GPUs,
            precision level, etc.
This script uses the `/examples/nlp/token_classification/conf/token_classification_config.yaml` config file
by default. You may update the config file from the file directly. 
The other option is to set another config file via command line arguments by `--config-name=CONFIG_FILE_PATH'.

For more details about the config files and different ways of model restoration, see tutorials/00_NeMo_Primer.ipynb

*** Model Training ***

To train TokenClassification model from scratch with the default config file, run:

    python token_classification_train.py \
           model.dataset.data_dir=<PATH_TO_DATA_DIR>  \
           trainer.max_epochs=<NUM_EPOCHS> \
           trainer.devices=[<CHANGE_TO_GPU(s)_YOU_WANT_TO_USE>]

To use one of the pretrained versions of the model specify a `pretrained_model` arg with either 
TokenClassification model from list_available_models() or path to a .nemo file, for example: 
ner_en_bert or model.nemo, run:

    python token_classification_train.py pretrained_model=ner_en_bert

To use one of the pretrained versions of the model and fine-tune it, run:

    python token_classification_train.py \
           model.dataset.data_dir=<PATH_TO_DATA_DIR>  \
           pretrained_model=ner_en_bert

<PATH_TO_DATA_DIR> - a directory that contains test_ds.text_file and test_ds.labels_file (see the config)
pretrained_model   - pretrained TokenClassification model from list_available_models() or 
                     path to a .nemo file, for example: ner_en_bert or model.nemo
                     
For more ways of restoring a pre-trained model, see tutorials/00_NeMo_Primer.ipynb
"""


@hydra_runner(config_path="conf", config_name="token_classification_config")
def main(cfg: DictConfig) -> None:
    try:
        strategy = NLPDDPStrategy(find_unused_parameters=True)
    except (ImportError, ModuleNotFoundError):
        strategy = 'auto'

    trainer = pl.Trainer(strategy=strategy, **cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))

    if not cfg.pretrained_model:
        logging.info(f'Config: {OmegaConf.to_yaml(cfg)}')
        model = TokenClassificationModel(cfg.model, trainer=trainer)
    else:
        if os.path.exists(cfg.pretrained_model):
            # TODO: can we drop strict=False?
            model = TokenClassificationModel.restore_from(cfg.pretrained_model, trainer=trainer, strict=False)
        elif cfg.pretrained_model in TokenClassificationModel.get_available_model_names():
            model = TokenClassificationModel.from_pretrained(cfg.pretrained_model)
        else:
            raise ValueError(
                f'Provide path to the pre-trained .nemo file or choose from {TokenClassificationModel.list_available_models()}'
            )

        data_dir = cfg.model.dataset.get('data_dir', None)
        if data_dir:
            if not os.path.exists(data_dir):
                raise ValueError(f'{data_dir} is not found at')

            # we can also do finetuning of the pretrained model but it will require
            # setup the data dir to get class weights statistics
            model.update_data_dir(data_dir=data_dir)
            # finally, setup train and validation Pytorch DataLoaders
            model.setup_training_data()
            model.setup_validation_data()
            # then we're setting up loss, use model.dataset.class_balancing,
            # if you want to add class weights to the CrossEntropyLoss
            model.setup_loss(class_balancing=cfg.model.dataset.class_balancing)
            logging.info(f'Using config file of the pretrained model')
        else:
            raise ValueError(
                'Specify a valid dataset directory that contains test_ds.text_file and test_ds.labels_file \
                with "model.dataset.data_dir" argument'
            )

    trainer.fit(model)


if __name__ == '__main__':
    main()
