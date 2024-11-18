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

import lightning.pytorch as pl
from lightning.pytorch import seed_everything
from omegaconf import OmegaConf

from nemo.collections.asr.models import EncDecDiarLabelModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager

"""
Example training session (single GPU training on telephonic datasets)

python ./multiscale_diar_decoder.py --config-path='../conf/neural_diarizer' --config-name='msdd_5scl_15_05_50Povl_256x3x32x2.yaml' \
    trainer.devices=1 \
    model.base.diarizer.speaker_embeddings.model_path="titanet_large" \
    model.train_ds.manifest_filepath="<train_manifest_path>" \
    model.validation_ds.manifest_filepath="<dev_manifest_path>" \
    model.train_ds.emb_dir="<train_temp_dir>" \
    model.validation_ds.emb_dir="<dev_temp_dir>" \
    exp_manager.name='sample_train' \
    exp_manager.exp_dir='./msdd_exp'
"""

seed_everything(42)


@hydra_runner(config_path="../conf/neural_diarizer", config_name="msdd_5scl_15_05_50Povl_256x3x32x2.yaml")
def main(cfg):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')
    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))
    msdd_model = EncDecDiarLabelModel(cfg=cfg.model, trainer=trainer)
    trainer.fit(msdd_model)


if __name__ == '__main__':
    main()
