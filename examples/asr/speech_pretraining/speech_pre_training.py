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


import pytorch_lightning as pl
from omegaconf import OmegaConf

from nemo.collections.asr.models.ssl_models import SpeechEncDecSelfSupervisedModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from nemo.utils.trainer_utils import resolve_trainer_cfg

"""
# Example of unsupervised pre-training of a model
```sh
python speech_pre_training.py \
    # (Optional: --config-path=<path to dir of configs> --config-name=<name of config without .yaml>) \
    model.train_ds.manifest_filepath=<path to train manifest> \
    model.validation_ds.manifest_filepath=<path to val/test manifest> \
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

For documentation on fine-tuning, please visit -
https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/configs.html#fine-tuning-configurations
When doing supervised fine-tuning from unsupervised pre-trained encoder, set flag init_strict to False

"""


@hydra_runner(config_path="../conf/ssl/citrinet/", config_name="citrinet_ssl_1024")
def main(cfg):
    logging.info(f"Hydra config: {OmegaConf.to_yaml(cfg)}")

    trainer = pl.Trainer(**resolve_trainer_cfg(cfg.trainer))
    exp_manager(trainer, cfg.get("exp_manager", None))
    asr_model = SpeechEncDecSelfSupervisedModel(cfg=cfg.model, trainer=trainer)

    # Initialize the weights of the model from another model, if provided via config
    asr_model.maybe_init_from_pretrained_checkpoint(cfg)

    trainer.fit(asr_model)


if __name__ == "__main__":
    main()
