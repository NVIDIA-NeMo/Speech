# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
import torch
from omegaconf.omegaconf import OmegaConf, open_dict
from torch._dynamo import disable
from torch._inductor import config as inductor_config

from nemo.collections.multimodal.models.text_to_image.imagen.imagen import MegatronImagen
from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronTrainerBuilder
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


@hydra_runner(config_path='conf', config_name='base64-500m')
def main(cfg) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    trainer = MegatronTrainerBuilder(cfg).create_trainer()
    exp_manager(trainer, cfg.exp_manager)

    # hydra interpolation does not work here as the interpolation key is lost when PTL saves hparams
    with open_dict(cfg):
        cfg.model.precision = cfg.trainer.precision

    model = MegatronImagen(cfg.model, trainer)

    if cfg.model.get("inductor", False):
        # Temporary hack to get rid of TorchDynamo issue with DDP
        # TODO: remove these if https://github.com/pytorch/pytorch/issues/94574 fixed
        torch.arange = disable(torch.arange)
        torch.ones = disable(torch.ones)
        torch.zeros = disable(torch.zeros)

        # TODO: remove this if latest TorchDynamo fixed `t.uniform_(0, 1)` failure
        torch.Tensor.uniform_ = disable(torch.Tensor.uniform_)

        # Disable TorchDynamo for unsupported function
        pl.core.LightningModule.log = disable(pl.core.LightningModule.log)

        # TorchInductor with CUDA graph can lead to OOM
        inductor_config.triton.cudagraphs = cfg.model.inductor_cudagraphs
        model.model.model.unet = torch.compile(model.model.model.unet)

    trainer.fit(model)


if __name__ == '__main__':
    main()
