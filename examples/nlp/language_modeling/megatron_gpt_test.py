# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

from omegaconf.omegaconf import OmegaConf
from pytorch_lightning import Trainer

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.modules.common.megatron.megatron_utils import compute_model_parallel_rank
from nemo.collections.nlp.parts.nlp_overrides import (
    NLPDDPStrategy,
    NLPMixedPrecisionPlugin,
    NLPPrecisionPlugin,
    NLPSaveRestoreConnector,
)
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.app_state import AppState


@hydra_runner(config_path="conf", config_name="megatron_gpt_config")
def main(cfg) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    trainer = None
    if cfg.trainer.precision in [16, '16', '16-mixed']:
        trainer = Trainer(
            plugins=[
                NLPMixedPrecisionPlugin(
                    init_scale=cfg.model.get('native_amp_init_scale', 2 ** 32),
                    growth_interval=cfg.model.get('native_amp_growth_interval', 1000),
                ),
            ],
            strategy=NLPDDPStrategy(),
            **cfg.trainer,
        )
    elif cfg.trainer.precision in ['bf16', 'bf16-mixed']:
        trainer = Trainer(plugins=[NLPNativeBfloat16PrecisionPlugin(),], strategy=NLPDDPStrategy(), **cfg.trainer,)
    else:
        trainer = Trainer(plugins=[NLPPrecisionPlugin()], strategy=NLPDDPStrategy(), **cfg.trainer)

    app_state = AppState()
    app_state.model_parallel_size = cfg.model.tensor_model_parallel_size
    app_state.model_parallel_rank = compute_model_parallel_rank(trainer.local_rank, app_state.model_parallel_size)

    model = MegatronGPTModel.restore_from(
        cfg.restore_from_path, trainer=trainer, save_restore_connector=NLPSaveRestoreConnector(),
    )

    # Note: most nemo models must have the data paths configured before instantiating the model
    # MegatronGPTMOdel sets up the data in the PTL method .setup which happens after DDP spawns.
    model.cfg.data.splits_string = cfg.model.data.splits_string

    trainer.test(model)


if __name__ == '__main__':
    main()
