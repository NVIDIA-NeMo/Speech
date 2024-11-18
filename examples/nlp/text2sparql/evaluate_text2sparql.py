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
This script contains an example on how to evaluate a NeuralMachineTranslationModel.
To load the example Text2Sparql dataset, please refer to ./data/import_datasets.py.
To train a model, please refer to text2sparql.py.


***Setting the configs***
This script uses the `/examples/nlp/text2sparql/conf/text2sparql_config.yaml` config file by default.
You may update the config file from the file directly or by using the command line arguments.
Another other option is to set another config file via command line arguments by `--config-name=CONFIG_FILE_PATH'.

Please refer to text2sparql.py for detailed instructions on setting the configuration.


***How to run the script?***
- To reload and evaluate the model, run:

python evaluate_text2sparql.py \
    model.test_ds.filepath="$TGT_DATA_DIR"/test_easy.tsv \
    model.batch_size=16 \
    model.nemo_path=./NeMo_logs/bart.nemo \
    exp_manager.exp_dir=./NeMo_logs
"""

import os

import lightning.pytorch as pl
from omegaconf import DictConfig, OmegaConf

from nemo.collections.nlp.models.text2sparql import Text2SparqlModel
from nemo.core.config import hydra_runner
from nemo.utils import logging


@hydra_runner(config_path="conf", config_name="text2sparql_config")
def main(cfg: DictConfig) -> None:
    logging.info(f"Config:\n {OmegaConf.to_yaml(cfg)}")
    trainer = pl.Trainer(devices=cfg.trainer.devices, accelerator=cfg.trainer.accelerator)
    nmt_model = Text2SparqlModel.restore_from(restore_path=cfg.model.nemo_path)
    nmt_model.setup_test_data(cfg.model.test_ds)
    results = trainer.test(nmt_model)

    with open(cfg.model.test_ds.filepath, "r", encoding='utf-8') as f:
        lines = f.readlines()

    lines[0] = lines[0].strip() + f"\tpredictions\n"
    for i, res in enumerate(results[0]["texts"]):
        lines[i + 1] = lines[i + 1].strip() + f"\t{res}\n"

    savepath = os.path.join(cfg.exp_manager.exp_dir, os.path.basename(cfg.model.test_ds.filepath))
    with open(savepath, "w", encoding='utf-8') as f:
        f.writelines(lines)
        logging.info(f"Predictions saved to {savepath}")


if __name__ == "__main__":
    main()
