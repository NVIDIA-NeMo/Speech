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

import os

import torch
from omegaconf.omegaconf import OmegaConf

from nemo.collections.multimodal.models.text_to_image.stable_diffusion.ldm.ddpm import MegatronLatentDiffusion
from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronTrainerBuilder
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy
from nemo.collections.nlp.parts.peft_config import PEFT_CONFIG_MAP
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.callbacks import CUDAGraphCallback
from nemo.utils.exp_manager import exp_manager


class MegatronStableDiffusionTrainerBuilder(MegatronTrainerBuilder):
    """Builder for SD model Trainer with overrides."""

    def _training_strategy(self) -> NLPDDPStrategy:
        """
        Returns a ddp strategy passed to Trainer.strategy.
        """
        ddp_overlap = self.cfg.model.get('ddp_overlap', True)
        if ddp_overlap:
            return NLPDDPStrategy(
                no_ddp_communication_hook=False,
                gradient_as_bucket_view=self.cfg.model.gradient_as_bucket_view,
                find_unused_parameters=True,
                bucket_cap_mb=256,
            )
        else:
            return NLPDDPStrategy(
                no_ddp_communication_hook=True,
                gradient_as_bucket_view=self.cfg.model.gradient_as_bucket_view,
                find_unused_parameters=False,
            )


@hydra_runner(config_path='conf', config_name='sd_train')
def main(cfg) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    torch.backends.cuda.matmul.allow_tf32 = True

    callbacks = (
        None
        if cfg.model.capture_cudagraph_iters < 0
        else [CUDAGraphCallback(capture_iteration=cfg.model.capture_cudagraph_iters)]
    )
    trainer = MegatronStableDiffusionTrainerBuilder(cfg).create_trainer(callbacks)

    exp_manager(trainer, cfg.exp_manager)

    model = MegatronLatentDiffusion(cfg.model, trainer)

    if cfg.model.capture_cudagraph_iters >= 0:
        # Warmup the model with random data
        with torch.cuda.stream(torch.cuda.Stream()):
            n, c, h = cfg.model.micro_batch_size, cfg.model.channels, cfg.model.image_size
            x = torch.randn((n, c, h, h), dtype=torch.float32, device="cuda")
            t = torch.randint(77, (n,), device="cuda")
            cc = torch.randn(
                (n, 77, cfg.model.unet_config.context_dim),
                dtype=torch.float32,
                device="cuda",
            )
            if cfg.model.precision in [16, '16']:
                x = x.type(torch.float16)
                cc = cc.type(torch.float16)
                autocast_enabled = False
                dgrad_dtype = torch.float16
            else:
                autocast_enabled = True
                dgrad_dtype = torch.float16

            model = model.cuda()
            for _ in range(5):
                with torch.autocast(device_type="cuda", enabled=autocast_enabled, dtype=torch.float16):
                    out = model.model.model.diffusion_model(x, t, context=cc)
                grad = torch.randn_like(out, dtype=dgrad_dtype)
                out.backward(grad)
                model.zero_grad()

    if cfg.model.get('peft', None):
        peft_cfg_cls = PEFT_CONFIG_MAP[cfg.model.peft.peft_scheme]
        if cfg.model.peft.restore_from_path is not None:
            # initialize peft weights from a checkpoint instead of randomly
            # This is not the same as resume training because optimizer states are not restored.
            logging.info("PEFT Weights will be loaded from", cfg.model.peft.restore_from_path)
            model.load_adapters(cfg.model.peft.restore_from_path, peft_cfg_cls(model_cfg))
        elif peft_cfg_cls is not None:
            logging.info("Adding adapter weights to the model for PEFT")
            model.add_adapter(peft_cfg_cls(cfg.model))
        else:
            logging.info(f"Running full finetuning since no peft scheme is given.\n{model.summarize()}")

    trainer.fit(model)


if __name__ == '__main__':
    main()
