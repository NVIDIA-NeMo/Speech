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

import os

import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

from nemo.collections.nlp.models.question_answering.qa_bert_model import BERTQAModel
from nemo.collections.nlp.models.question_answering.qa_gpt_model import GPTQAModel
from nemo.collections.nlp.models.question_answering.qa_s2s_model import S2SQAModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


@hydra_runner(config_path="conf", config_name="qa_conf")
def main(cfg: DictConfig) -> None:
    pl.seed_everything(42)
    # PTL 2.0 has find_unused_parameters as False by default, so its required to set it to True
    # when there are unused parameters like here
    if cfg.trainer.strategy == 'ddp':
        cfg.trainer.strategy = "ddp_find_unused_parameters_true"
    logging.info(f'Config: {OmegaConf.to_yaml(cfg)}')
    trainer = pl.Trainer(**cfg.trainer)
    exp_dir = exp_manager(trainer, cfg.get("exp_manager", None))

    if "bert" in cfg.model.language_model.pretrained_model_name.lower():
        model_class = BERTQAModel
    elif "gpt" in cfg.model.language_model.pretrained_model_name.lower():
        model_class = GPTQAModel
    elif (
        "bart" in cfg.model.language_model.pretrained_model_name.lower()
        or "t5" in cfg.model.language_model.pretrained_model_name.lower()
    ):
        model_class = S2SQAModel

    if cfg.pretrained_model or (cfg.model.nemo_path and os.path.exists(cfg.model.nemo_path)):
        if cfg.pretrained_model:
            logging.info(f'Loading pretrained model {cfg.pretrained_model}')
            model = model_class.from_pretrained(cfg.pretrained_model)
        else:
            logging.info(f'Restoring model from {cfg.model.nemo_path}')
            model = model_class.restore_from(cfg.model.nemo_path)

        if cfg.do_training:
            model.setup_training_data(train_data_config=cfg.model.train_ds)
            model.setup_multiple_validation_data(val_data_config=cfg.model.validation_ds)
    else:
        logging.info(f'Config: {OmegaConf.to_yaml(cfg)}')
        model = model_class(cfg.model, trainer=trainer)

    if cfg.do_training:
        trainer.fit(model)
        if cfg.model.nemo_path:
            model.save_to(cfg.model.nemo_path)

    if hasattr(cfg.model, 'test_ds') and cfg.model.test_ds.file is not None:
        eval_device = [cfg.trainer.devices[0]] if isinstance(cfg.trainer.devices, list) else 1
        trainer = pl.Trainer(devices=eval_device, accelerator=cfg.trainer.accelerator, precision=16)
        model.setup_test_data(test_data_config=cfg.model.test_ds)
        trainer.test(model)

        # specifiy .json file to dump predictions. e.g. os.path.join(exp_dir, "output_nbest_file.json")
        output_nbest_file = None
        # specifiy .json file to dump predictions. e.g. os.path.join(exp_dir, "output_prediction_file.json")
        output_prediction_file = None
        inference_samples = 5  # for test purposes. To use entire inference dataset set to -1
        all_preds, all_nbest = model.inference(
            cfg.model.test_ds.file,
            output_prediction_file=output_prediction_file,
            output_nbest_file=output_nbest_file,
            num_samples=inference_samples,
        )

        for question_id in all_preds:
            print(all_preds[question_id])


if __name__ == "__main__":
    main()
