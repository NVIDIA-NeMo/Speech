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

import lightning.pytorch as pl

from nemo.collections.asr.models import EncDecCTCModel, configs
from nemo.core.config import hydra_runner
from nemo.utils.config_utils import update_model_config
from nemo.utils.exp_manager import exp_manager


"""
python speech_to_text_hybrid.py \
    --config-path="conf/quartznet" \
    --config-name="quartznet_15x5" \
    model.train_ds.manifest_filepath="/home/smajumdar/PycharmProjects/NeMo-som/examples/asr/an4/train_manifest.json" \
    model.validation_ds.manifest_filepath="/home/smajumdar/PycharmProjects/NeMo-som/examples/asr/an4/test_manifest.json" \
    trainer.devices=1
"""


@hydra_runner(config_path="conf/quartznet", config_name="quartznet_15x5")
def main(cfg):
    # Generate default asr model config
    asr_model_config = configs.EncDecCTCModelConfig()

    # Merge hydra updates with model config
    # `drop_missing_subconfig=True` is necessary here. Without it, while the data class will instantiate and be added
    # to the config, it contains test_ds.sample_rate = MISSING and test_ds.labels = MISSING.
    # This will raise a OmegaConf MissingMandatoryValue error when processing the dataloaders inside
    # model_utils.resolve_test_dataloaders(model=self) (used for multi data loader support).
    # In general, any operation that tries to use a DictConfig with MISSING in it will fail,
    # other than explicit update operations to change MISSING to some actual value.
    asr_model_config = update_model_config(asr_model_config, cfg, drop_missing_subconfigs=True)

    # From here on out, its a general OmegaConf DictConfig, directly usable by our code.
    trainer = pl.Trainer(**asr_model_config.trainer)
    exp_manager(trainer, asr_model_config.get("exp_manager", None))
    asr_model = EncDecCTCModel(cfg=asr_model_config.model, trainer=trainer)

    trainer.fit(asr_model)


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
