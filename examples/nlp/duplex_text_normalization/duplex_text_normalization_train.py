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

"""
This script contains an example on how to train a DuplexTextNormalizationModel.
Note that DuplexTextNormalizationModel is essentially a wrapper class around
two other classes:

(1) DuplexTaggerModel is a model for identifying spans in the input that need to
be normalized. Usually, such spans belong to semiotic classes (e.g., DATE, NUMBERS, ...).

(2) DuplexDecoderModel is a model for normalizing the spans identified by the tagger.
For example, in the text normalization (TN) problem, each span will be converted to its
spoken form. In the inverse text normalization (ITN) problem, each span will be converted
to its written form.

Therefore, this script consists of two parts, one is for training the tagger model
and the other is for training the decoder.

This script uses the `/examples/nlp/duplex_text_normalization/conf/duplex_tn_config.yaml`
config file by default. The other option is to set another config file via command
line arguments by `--config-name=CONFIG_FILE_PATH'. Probably it is worth looking
at the example config file to see the list of parameters used for training.

USAGE Example:
1. Obtain a processed dataset (refer to the `text_normalization doc <https://github.com/NVIDIA/NeMo/blob/main/docs/source/nlp/text_normalization/nn_text_normalization.rst>`)
2. Run:
# python duplex_text_normalization_train.py \
        data.validation_ds.data_path=PATH_TO_VALIDATION_FILE \
        data.train_ds.data_path=PATH_TO_TRAIN_FILE \
        mode={tn,itn,joint} \
        lang={en,ru,de}

There are 3 different modes. `tn` mode is for training a system for TN only.
`itn` mode is for training a system for ITN. `joint` is for training a system
that can do both TN and ITN at the same time. Note that the above command will
first train a tagger and then train a decoder sequentially.

You can also train only a tagger (without training a decoder) by running the
following command:
# python duplex_text_normalization_train.py
        data.validation_ds.data_path=PATH_TO_VALIDATION_FILE \
        data.train_ds.data_path=PATH_TO_TRAIN_FILE \
        data.test_ds.data_path=PATH_TO_TEST_FILE \
        mode={tn,itn,joint}
        lang={en,ru,de}
        decoder_model.do_training=false

Or you can also train only a decoder (without training a tagger):
# python duplex_text_normalization_train.py \
        data.validation_ds.data_path=PATH_TO_VALIDATION_FILE \
        data.train_ds.data_path=PATH_TO_TRAIN_FILE \
        data.test_ds.data_path=PATH_TO_TEST_FILE \
        mode={tn,itn,joint} \
        lang={en,ru,de} \
        tagger_model.do_training=false

To use tarred dataset for decoder training set:

    data.train_ds.use_tarred_dataset=True \
    data.train_ds.tar_metadata_file=\PATH_TO\<TARRED_DATA_OUTPUT_DIR>\metadata.json

Information on the arguments:

Most arguments in the example config file are quite self-explanatory (e.g.,
`decoder_model.optim.lr` refers to the learning rate for training the decoder).
Some arguments we want to mention are:

+ lang: The language of the dataset.

+ tagger_model.nemo_path: This is the path where the final trained tagger model
will be saved to.

+ decoder_model.nemo_path: This is the path where the final trained decoder model
will be saved to.
"""


from helpers import DECODER_MODEL, TAGGER_MODEL, instantiate_model_and_trainer
from omegaconf import DictConfig, OmegaConf

from nemo.collections.nlp.data.text_normalization import TextNormalizationTestDataset
from nemo.collections.nlp.models import DuplexTextNormalizationModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


@hydra_runner(config_path="conf", config_name="duplex_tn_config")
def main(cfg: DictConfig) -> None:
    logging.info(f'Config Params: {OmegaConf.to_yaml(cfg)}')

    # Train the tagger
    if cfg.tagger_model.do_training:
        logging.info(
            "================================================================================================"
        )
        logging.info('Starting training tagger...')
        tagger_trainer, tagger_model = instantiate_model_and_trainer(cfg, TAGGER_MODEL, True)
        tagger_exp_manager = cfg.get('tagger_exp_manager', None)
        exp_manager(tagger_trainer, tagger_exp_manager)
        tagger_trainer.fit(tagger_model)
        logging.info('Training finished!')

    # Train the decoder
    if cfg.decoder_model.do_training:
        logging.info(
            "================================================================================================"
        )
        logging.info('Starting training decoder...')
        decoder_trainer, decoder_model = instantiate_model_and_trainer(cfg, DECODER_MODEL, True)
        decoder_exp_manager = cfg.get('decoder_exp_manager', None)
        exp_manager(decoder_trainer, decoder_exp_manager)
        decoder_trainer.fit(decoder_model)
        logging.info('Training finished!')

    # Evaluation after training
    if (
        hasattr(cfg.data, 'test_ds')
        and cfg.data.test_ds.data_path is not None
        and cfg.tagger_model.do_training
        and cfg.decoder_model.do_training
    ):
        tn_model = DuplexTextNormalizationModel(tagger_model, decoder_model, cfg.lang)
        test_dataset = TextNormalizationTestDataset(cfg.data.test_ds.data_path, cfg.mode, cfg.lang)
        results = tn_model.evaluate(test_dataset, cfg.data.test_ds.batch_size, cfg.data.test_ds.errors_log_fp)
        print(f'\nTest results: {results}')


if __name__ == '__main__':
    main()
