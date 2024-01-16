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

from pytorch_lightning import Trainer

from nemo.collections.multimodal.models.text_to_image.controlnet.controlnet import MegatronControlNet
from nemo.collections.multimodal.models.text_to_image.controlnet.util import ImageLogger
from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronTrainerBuilder
from nemo.core.config import hydra_runner
from nemo.utils.exp_manager import exp_manager


class MegatronControlNetTrainerBuilder(MegatronTrainerBuilder):
    """Builder for T5 model Trainer with overrides."""

    def create_trainer(self, callbacks=[]) -> Trainer:
        strategy = self._training_strategy()
        plugins = self._plugins()
        return Trainer(plugins=plugins, strategy=strategy, **self.cfg.trainer, callbacks=callbacks)


@hydra_runner(config_path='conf', config_name='controlnet_v1-5.yaml')
def main(cfg):
    callbacks = []

    if cfg.model.get('image_logger', None):
        callbacks.append(ImageLogger(**cfg.model.image_logger))

    trainer = MegatronControlNetTrainerBuilder(cfg).create_trainer(callbacks=callbacks)

    exp_manager(trainer, cfg.get("exp_manager", None))

    model = MegatronControlNet(cfg.model, trainer)

    trainer.fit(model)


if __name__ == '__main__':
    main()
