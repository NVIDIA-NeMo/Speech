# Copyright (c) 2020, MeetKai Inc.  All rights reserved.
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
This script contains an example on how to train and save a Text2SparqlModel.
Text2SparqlModel in NeMo supports sequence to sequence problems such as language translation
and text summarization, provided the data follows the format specified below.


***Data format***
Text2SparqlModel requires the data to be stored in TAB separated files (.tsv) with two columns of
sentence and label, where the first line is a header of format:
    sentence[TAB]label
And each line is of the format:
    [SENTENCE][TAB][LABEL]

If your dataset is stored in another format, you need to convert it to this format to use a
Text2SparqlModel.


***Setting the configs***
This script uses the `/examples/nlp/text2sparql/conf/text2sparql_config.yaml` config file by default.
You may update the config file from the file directly or by using the command line arguments.
Another other option is to set another config file via command line arguments by `--config-name=CONFIG_FILE_PATH'.

A Text2SparqlModel's config file declares multiple import sections. They are:
    - trainer: Arguments to be passed to PyTorch Lightning.
    - model: All arguments that relate to the Model - language_model, tokenizers, datasets, optimizer, generate.
    - exp_manager: Arguments to be passed to NeMo's experiment manager.
    - hydra: Arguments to be passed to Hydra.

If using text2sparql_config.yaml, you must first update the following fields in the config:
    - model.nemo_path: Model save path. Eg. [PATH]/bart.nemo
    - model.data_dir: Path to data directory. Alternatively, you can adjust the file paths directly:
        - model.train_ds.filepath
        - model.validation_ds.filepath
        - model.test_ds.filepath
    - exp_manager.exp_dir: Directory to log results from the experiment.

It is highly recommended to also adjust these parameters as necessary:
    - trainer.devices: Set to 0 to use CPU. Otherwise the number denotes the number of GPUs.
    - trainer.max_epochs: Maximum number of epochs to train for.
    - model.batch_size: 8 is sufficient to train a decent Bart model for Text2Sparql.
    - model.max_seq_length: Maximum (tokenized) sequence length. 150 is sufficient for Text2Sparql.
    - model.language_model.pretrained_model_name: End2end pretrained model name from huggingface.
        - model.encoder_tokenizer.tokenizer_name: Pretrained tokenizer name from huggingface.
        - model.decoder_tokenizer.tokenizer_name: The same as above, as the tokenizer will handle encoding and decoding.
    - model.optim.lr: Learning rate.

You can also specify an encoder and decoder rather than using an end2end model like Bart by defining these parameters:
    - model.language_model.pretrained_encoder_model_name: Pretrained huggingface encoder model name.
        - model.encoder_tokenizer.tokenizer_name: Pretrained huggingface encoder tokenizer name.
    - model.language_model.pretrained_decoder_model_name: Pretrained huggingface decoder model name.
        - model.decoder_tokenizer.tokenizer_name: Pretrained huggingface decoder tokenizer name.
    - model.language_model.pretrained_model_name: Set this to null.


***How to run the script?***
- First, download the data to TGT_DATA_DIR (see ./data/import_datasets.py):

SRC_DATA_DIR=./data/text2sparql_src
TGT_DATA_DIR=./data/text2sparql_tgt

python ./data/import_datasets.py \
    --source_data_dir $SRC_DATA_DIR \
    --target_data_dir $TGT_DATA_DIR

- And run the following to train and save the model:

python text2sparql.py \
    model.train_ds.filepath="$TGT_DATA_DIR"/train.tsv \
    model.validation_ds.filepath="$TGT_DATA_DIR"/test_easy.tsv \
    model.test_ds.filepath="$TGT_DATA_DIR"/test_hard.tsv \
    model.batch_size=16 \
    model.nemo_path=./NeMo_logs/bart.nemo \
    exp_manager.exp_dir=./NeMo_logs
"""

import lightning.pytorch as pl
from omegaconf import DictConfig, OmegaConf

from nemo.collections.nlp.models.text2sparql import Text2SparqlModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


@hydra_runner(config_path="conf", config_name="text2sparql_config")
def main(cfg: DictConfig) -> None:
    logging.info(f"Config:\n {OmegaConf.to_yaml(cfg)}")
    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))
    nmt_model = Text2SparqlModel(cfg.model, trainer=trainer)
    trainer.fit(nmt_model)
    if cfg.model.nemo_path:
        nmt_model.save_to(cfg.model.nemo_path)


if __name__ == "__main__":
    main()
