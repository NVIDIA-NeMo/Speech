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
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.plugins.environments import TorchElasticEnvironment
from omegaconf import open_dict

from nemo.collections.multimodal.models.text_to_image.stable_diffusion.ldm.ddpm import MegatronLatentDiffusion
from nemo.collections.multimodal.parts.stable_diffusion.pipeline import pipeline
from nemo.collections.multimodal.parts.utils import setup_trainer_and_model_for_inference
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy, NLPSaveRestoreConnector
from nemo.collections.nlp.parts.peft_config import PEFT_CONFIG_MAP
from nemo.core.config import hydra_runner


@hydra_runner(config_path='conf', config_name='dreambooth_lora_infer')
def main(cfg):
    def model_cfg_modifier(model_cfg):
        model_cfg.precision = cfg.trainer.precision
        model_cfg.ckpt_path = None
        model_cfg.inductor = False
        model_cfg.target = (
            'nemo.collections.multimodal.models.text_to_image.stable_diffusion.ldm.ddpm.MegatronLatentDiffusion'
        )
        if cfg.model.unet_config.from_pretrained:
            model_cfg.unet_config.from_pretrained = cfg.model.unet_config.from_pretrained

    model_cfg = MegatronLatentDiffusion.restore_from(
        restore_path=cfg.model.peft.restore_from_path,
        trainer=None,
        save_restore_connector=NLPSaveRestoreConnector(),
        return_config=True,
    )

    with open_dict(model_cfg):
        model_cfg_modifier(model_cfg)

    plugins = []
    plugins.append(TorchElasticEnvironment())
    strategy = NLPDDPStrategy(
        no_ddp_communication_hook=True,
        find_unused_parameters=False,
    )
    trainer = Trainer(plugins=plugins, strategy=strategy, **cfg.trainer)

    model = MegatronLatentDiffusion(model_cfg, trainer=trainer)
    model.setup_complete = True

    peft_cfg_cls = PEFT_CONFIG_MAP[model_cfg.peft.peft_scheme]

    model.load_adapters(cfg.model.peft.restore_from_path, peft_cfg_cls(model_cfg))
    rng = torch.Generator().manual_seed(cfg.infer.seed)

    model = model.model.cuda().eval()
    pipeline(model, cfg, rng=rng)


if __name__ == "__main__":
    main()
