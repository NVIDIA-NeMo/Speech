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

from hydra.utils import get_class, instantiate
from lightning.pytorch import Trainer, seed_everything
from omegaconf.omegaconf import DictConfig, OmegaConf

from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


@hydra_runner(config_path='config', config_name='config')
def main(cfg: DictConfig) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    seed_everything(cfg.seed, workers=True)

    mode = cfg.mode
    logging.info(f"{mode=}")

    model = None
    model_cls = get_class(cfg.model._target_)
    if cfg.model.resume_from_checkpoint is None:
        model = model_cls(cfg=cfg.model)
    else:
        logging.info(f"Loading model from checkpoint: {cfg.model.resume_from_checkpoint}")
        model = model_cls.load_from_checkpoint(cfg.model.resume_from_checkpoint, strict=False, cfg=cfg.model)

    if mode == "export-mesh":
        mesh = model.mesh(resolution=cfg.mesh_resolution)
        mesh.export(cfg.mesh_fname)
        return

    # Prepare callbacks
    callbacks = []
    if cfg.enable_benchmark:
        callbacks.append(instantiate(cfg.benchmark_callback))

    # Setup trainer
    trainer = Trainer(callbacks=callbacks, **cfg.trainer)
    exp_manager(trainer, cfg.exp_manager)

    # Setup datamodule
    dm = instantiate(cfg.model.data)

    if mode == "fit":
        trainer.fit(model, datamodule=dm)
    elif mode == "validate":
        trainer.validate(model, datamodule=dm)
    elif mode == "test":
        trainer.test(model, datamodule=dm)
    else:
        raise ValueError(f"Invalid mode: {mode}")


if __name__ == '__main__':
    main()
