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


import torch.multiprocessing as mp
from omegaconf.omegaconf import OmegaConf, open_dict
from pytorch_lightning import Trainer
from pytorch_lightning.plugins.environments import TorchElasticEnvironment

from nemo.collections.nlp.models.language_modeling.megatron_t5_adapter_model import MegatronT5LoraModel
from nemo.collections.nlp.parts.nlp_overrides import (
    CustomProgressBar,
    GradScaler,
    MegatronHalfPrecisionPlugin,
    NLPDDPStrategy,
    NLPSaveRestoreConnector,
    PipelineMixedPrecisionPlugin,
)
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.decorators import deprecated
from nemo.utils.exp_manager import exp_manager

mp.set_start_method("spawn", force=True)

"""
This is the script to train an Adapter infused GPT Model for text generation.
A base GPT Model is required as a starting point. This script will then insert
Adapters into each Transformer layer and will train/update only these adapters
during training. The base GPT Model weights will remain frozen.

During training this script will only save the newly trained Adapter weights
in checkpoints. At the end of training a .nemo file of Adapter weights will 
be saved.

Usage:
    Assuming the base model is a 125m GPT Model, with TP=1, PP=1:
    a. run a training run for a base gpt nemo file:
        python megatron_gpt_adapter_tuning.py \
            "model.data.train_ds=[PATH TO TRAINING JSONL FILE]",
            "model.data.validation_ds=[PATH TO VALIDATION JSONL FILE]",
            model.language_model_path="PATH TO BASE GPT MODEL .nemo FILE"
            name="NAME OF TRAINING RUN"
            exp_manager.exp_dir="DIR TO SAVE CHECKPOINTS and .nemo FILE",
            trainer.max_epochs=2
"""


@deprecated(
    explanation=f"{__file__} is deprecated. Please use MegatronT5SFTModel.add_adapter() for PEFT features."
    "See updated scripts `megatron_t5_peft_tuning.py` and `megatron_t5_peft_eval.py` for examples."
)
@hydra_runner(config_path="conf", config_name="megatron_t5_lora_tuning_config")
def main(cfg) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    megatron_amp_O2 = cfg.model.get('megatron_amp_O2', False)
    with_distributed_adam = cfg.model.optim.get('name') == 'distributed_fused_adam'

    plugins = []
    strategy = NLPDDPStrategy(
        no_ddp_communication_hook=True,  # we don't use DDP for async grad allreduce
        gradient_as_bucket_view=cfg.model.gradient_as_bucket_view,
        find_unused_parameters=False,
    )
    if cfg.trainer.precision in [16, '16', 'bf16', '16-mixed', 'bf16-mixed']:
        scaler = None
        if cfg.trainer.precision in [16, '16', '16-mixed']:
            scaler = GradScaler(
                init_scale=cfg.model.get('native_amp_init_scale', 2 ** 32),
                growth_interval=cfg.model.get('native_amp_growth_interval', 1000),
                hysteresis=cfg.model.get('hysteresis', 2),
            )
            # MixedPrecisionPlugin in PTL >= 2.0 requires precision to be 16-mixed or bf16-mixed
            plugin_precision = '16-mixed'
        else:
            plugin_precision = 'bf16-mixed'
        if megatron_amp_O2 and not with_distributed_adam:
            plugins.append(MegatronHalfPrecisionPlugin(precision=plugin_precision, device='cuda', scaler=scaler))
        else:
            plugins.append(PipelineMixedPrecisionPlugin(precision=plugin_precision, device='cuda', scaler=scaler))

    if cfg.get('cluster_type', None) == 'BCP':
        plugins.append(TorchElasticEnvironment())

    trainer = Trainer(plugins=plugins, strategy=strategy, **cfg.trainer, callbacks=[CustomProgressBar()])
    exp_manager(trainer, cfg.exp_manager)

    # load existing or init new soft prompt GPT model
    if cfg.model.get("restore_path", None):
        model = MegatronT5LoraModel.restore_from(
            cfg.model.restore_path, cfg.model, trainer=trainer, save_restore_connector=NLPSaveRestoreConnector()
        )
    else:
        model = MegatronT5LoraModel(cfg.model, trainer=trainer)

    trainer.fit(model)


if __name__ == '__main__':
    main()
