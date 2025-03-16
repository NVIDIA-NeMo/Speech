# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
from omegaconf import OmegaConf, open_dict

from nemo.collections.tts.models import MagpieTTS_Model, MagpieTTS_ModelDPO, MagpieTTS_ModelInference
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


@hydra_runner(config_path="conf/magpietts", config_name="magpietts_en")
def main(cfg):
    logging.info('\nConfig Params:\n%s', OmegaConf.to_yaml(cfg, resolve=True))
    if not cfg.model.get('use_lthose', False):
        import torch.multiprocessing as mp

        mp.set_start_method("spawn", force=True)

    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))

    if cfg.get('mode', 'train') == 'train':
        model = MagpieTTS_Model(cfg=cfg.model, trainer=trainer)
    elif cfg.get('mode', 'dpo_train') == 'dpo_train':
        model_cfg = cfg.model
        with open_dict(model_cfg):
            model_cfg.reference_model_ckpt_path = cfg.init_from_ptl_ckpt
        model = MagpieTTS_ModelDPO(cfg=model_cfg, trainer=trainer)
    elif cfg.get('mode', 'train') == 'test':
        model = MagpieTTS_ModelInference(cfg=cfg.model, trainer=trainer)
    else:
        raise NotImplementedError(f"Only train, dpo_train and test modes are supported. Got {cfg.mode}")

    model.maybe_init_from_pretrained_checkpoint(cfg=cfg)

    if cfg.get('mode', 'train') in ['train', 'dpo_train']:
        trainer.fit(model)
    elif cfg.get('mode', 'train') == 'test':
        trainer.test(model)
    else:
        raise NotImplementedError(f"Only train and test modes are supported. Got {cfg.mode}")


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
