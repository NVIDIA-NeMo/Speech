# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from collections import OrderedDict

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf

from nemo.collections.asr.models import EncDecSpeakerLabelModel, SpeechEncDecSelfSupervisedModel
from nemo.core.classes.common import typecheck
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager

typecheck.set_typecheck_enabled(enabled=False)

"""
Example script for training a speech classification model with a self-supervised pre-trained encoder, and 
use the SSL encoder for multi-layer feature extraction.

# Example of training a speaker classification model with a self-supervised pre-trained encoder
```sh
python speech_classification_mfa_train.py \
    # (Optional: --config-path=<path to dir of configs> --config-name=<name of config without .yaml>) \
    ++init_from_nemo_model=<path to pre-trained SSL .nemo file> \
    # or use ++init_from_pretrained_model=<model_name> \
    model.train_ds.manifest_filepath=<path to train manifest> \
    model.validation_ds.manifest_filepath=<path to val manifest> \
    trainer.devices=-1 \
    trainer.accelerator="gpu" \
    strategy="ddp"  \
    trainer.max_epochs=100 \
    model.optim.name="adamw" \
    model.optim.lr=0.001 \
    model.optim.betas=[0.9,0.999] \
    model.optim.weight_decay=0.0001 \
    model.optim.sched.warmup_steps=2000
    exp_manager.create_wandb_logger=True \
    exp_manager.wandb_logger_kwargs.name="<Name of experiment>" \
    exp_manager.wandb_logger_kwargs.project="<Namex of project>"
```
    
"""


def load_ssl_encoder(model, cfg):
    if cfg.get("init_from_ptl_ckpt", None) is not None:
        state_dict = torch.load(cfg.init_from_ptl_ckpt, map_location='cpu')['state_dict']
        logging.info(f"Loading encoder from PyTorch Lightning checkpoint: {cfg.init_from_ptl_ckpt}")
    elif cfg.get("init_from_nemo_model", None) is not None:
        ssl_model = SpeechEncDecSelfSupervisedModel.restore_from(cfg.init_from_nemo_model, map_location='cpu')
        state_dict = ssl_model.state_dict()
        logging.info(f"Loading encoder from NeMo model: {cfg.init_from_nemo_model}")
    elif cfg.get("init_from_pretrained_model", None) is not None:
        ssl_model = SpeechEncDecSelfSupervisedModel.from_pretrained(cfg.init_from_pretrained_model, map_location='cpu')
        state_dict = ssl_model.state_dict()
        logging.info(f"Loading encoder from pretrained model: {cfg.init_from_pretrained_model}")
    else:
        logging.info("No model checkpoint or pretrained model specified for encoder initialization.")
        return model

    encoder_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith('encoder.'):
            encoder_state_dict[f'preprocessor.feature_extractor.{key}'] = value

    model.load_state_dict(encoder_state_dict, strict=False)
    logging.info("Loaded ssl encoder state dict.")

    return model


@hydra_runner(config_path="../conf/ssl/nest/multi_layer_feat", config_name="nest_ecapa_tdnn_small")
def main(cfg):

    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')
    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))

    speaker_model = EncDecSpeakerLabelModel(cfg=cfg.model, trainer=trainer)

    if cfg.model.preprocessor.get("encoder", None) is not None:
        # multi-layer feature extractor
        speaker_model = load_ssl_encoder(speaker_model, cfg)
    else:
        speaker_model.maybe_init_from_pretrained_checkpoint(cfg)

    trainer.fit(speaker_model)

    if hasattr(cfg.model, 'test_ds') and cfg.model.test_ds.manifest_filepath is not None:
        if speaker_model.prepare_test(trainer):
            trainer.test(speaker_model)


if __name__ == '__main__':
    main()
