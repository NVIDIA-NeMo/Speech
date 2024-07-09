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

import sys
import torch
import torch._dynamo.config as dynamo_config
from omegaconf.omegaconf import OmegaConf
from pytorch_lightning import Trainer

from nemo.collections.multimodal.models.text_to_image.stable_diffusion.diffusion_engine import MegatronDiffusionEngine
from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronTrainerBuilder
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy, NLPFSDPStrategy
from nemo.collections.nlp.parts.peft_config import PEFT_CONFIG_MAP
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


class MegatronStableDiffusionTrainerBuilder(MegatronTrainerBuilder):
    """Builder for SD model Trainer with overrides."""

    def _training_strategy(self) -> NLPDDPStrategy:
        """
        Returns a ddp strategy passed to Trainer.strategy.
        """
        """
                Returns a DDP or a FSDP strategy passed to Trainer.strategy.
                """
        # check interactive environment
        _IS_INTERACTIVE = hasattr(sys, "ps1") or bool(sys.flags.interactive)
        if _IS_INTERACTIVE and self.cfg.trainer.devices == 1:
            logging.info("Detected interactive environment, using NLPDDPStrategyNotebook")
            return NLPDDPStrategyNotebook(
                no_ddp_communication_hook=True,
                find_unused_parameters=False,
            )

        if self.cfg.model.get('fsdp', False):
            assert (
                not self.cfg.model.optim.get('name') == 'distributed_fused_adam'
            ), 'Distributed optimizer cannot be used with FSDP.'
            if self.cfg.model.get('megatron_amp_O2', False):
                logging.info('Torch FSDP is not compatible with O2 precision recipe. Setting O2 `False`.')
                self.cfg.model.megatron_amp_O2 = False
            return NLPFSDPStrategy(
                limit_all_gathers=self.cfg.model.get('fsdp_limit_all_gathers', True),
                sharding_strategy=self.cfg.model.get('fsdp_sharding_strategy', 'full'),
                cpu_offload=self.cfg.model.get('fsdp_cpu_offload', False),
                grad_reduce_dtype=self.cfg.model.get('fsdp_grad_reduce_dtype', 32),
                precision=self.cfg.trainer.precision,
                use_orig_params=self.cfg.model.inductor,
                set_buffer_dtype=self.cfg.get('fsdp_set_buffer_dtype', None),
            )

        return NLPDDPStrategy(
            no_ddp_communication_hook=(not self.cfg.model.get('ddp_overlap')),
            gradient_as_bucket_view=self.cfg.model.gradient_as_bucket_view,
            find_unused_parameters=False,
        )


@hydra_runner(config_path='conf', config_name='sd_xl_base_train')
def main(cfg) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    torch.backends.cuda.matmul.allow_tf32 = True

    trainer = MegatronStableDiffusionTrainerBuilder(cfg).create_trainer()

    exp_manager(trainer, cfg.exp_manager)

    model = MegatronDiffusionEngine(cfg.model, trainer)

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
            logging.info(f"Running full finetuning since no peft scheme is given.")

    trainer.fit(model)


if __name__ == '__main__':
    main()
