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

"""
# Evaluate an adapted model

python eval_asr_adapter.py \
    --config-path="../conf/asr_adapters" \
    --config-name="asr_adaptation.yaml" \
    model.pretrained_model=null \
    model.nemo_model=null \
    model.adapter.adapter_name=<name of the adapter to evaluate> \
    model.test_ds.manifest_filepath="<Path to validation/test manifest>" \
    model.test_ds.batch_size=16 \
    model.train_ds.manifest_filepath=null \
    model.validation_ds.manifest_filepath=null \
    model.adapter.in_features=null \
    trainer.devices=1 \
    trainer.precision=32

# Pretrained Models

For documentation on existing pretrained models, please visit -
https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/results.html

"""

import lightning.pytorch as pl
from omegaconf import OmegaConf, open_dict

from nemo.collections.asr.models import ASRModel
from nemo.core import adapter_mixins
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


def update_encoder_config_to_support_adapter(model_cfg):
    with open_dict(model_cfg):
        adapter_metadata = adapter_mixins.get_registered_adapter(model_cfg.encoder._target_)
        if adapter_metadata is not None:
            model_cfg.encoder._target_ = adapter_metadata.adapter_class_path


def update_model_cfg(original_cfg, new_cfg):
    with open_dict(new_cfg):
        # drop keys which dont exist in old config
        new_keys = list(new_cfg.keys())
        for key in new_keys:
            if key not in original_cfg:
                new_cfg.pop(key)
                print("Removing unavailable key from config :", key)

        new_cfg = OmegaConf.merge(original_cfg, new_cfg)
    return new_cfg


@hydra_runner(config_path="../conf/asr_adapters", config_name="asr_adaptation.yaml")
def main(cfg):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    if cfg.model.pretrained_model is None and cfg.model.nemo_model is None:
        raise ValueError("Either set `cfg.model.nemo_model` or `cfg.model.pretrained_model`")
    if cfg.model.pretrained_model is not None and cfg.model.nemo_model is not None:
        raise ValueError("Cannot set `cfg.model.nemo_model` and `cfg.model.pretrained_model`. Select one only.")

    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))

    if cfg.model.pretrained_model is not None:
        model_cfg = ASRModel.from_pretrained(cfg.model.pretrained_model, return_config=True)
        update_encoder_config_to_support_adapter(model_cfg)
        model = ASRModel.from_pretrained(cfg.model.pretrained_model, override_config_path=model_cfg, trainer=trainer)

    else:
        model_cfg = ASRModel.restore_from(cfg.model.nemo_model, return_config=True)
        update_encoder_config_to_support_adapter(model_cfg)
        model = ASRModel.restore_from(cfg.model.nemo_model, override_config_path=model_cfg, trainer=trainer)

    # Setup model for finetuning (train and validation only)
    cfg.model.test_ds = update_model_cfg(model.cfg.test_ds, cfg.model.test_ds)

    # Call the dataloaders and optimizer + scheduler
    model.setup_multiple_test_data(cfg.model.test_ds)

    # Setup adapters
    with open_dict(cfg.model.adapter):
        adapter_name = cfg.model.adapter.pop("adapter_name", None)

    # Disable all other adapters, enable just the current adapter.
    model.set_enabled_adapters(enabled=False)  # disable all adapters prior to training

    if adapter_name is not None:
        model.set_enabled_adapters(adapter_name, enabled=True)  # enable just one adapter by name if provided

    # First, Freeze all the weights of the model (not just encoder, everything)
    model.freeze()

    # Finally, train model
    trainer.test(model)


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
