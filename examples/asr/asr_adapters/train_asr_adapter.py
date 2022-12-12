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
# Adapting the model

python train_asr_adapter.py \
    --config-path="../conf/asr_adapters" \
    --config-name="asr_adaptation.yaml" \
    model.pretrained_model=null \
    model.nemo_model=null \
    model.adapter.adapter_name=<Unique adapter name> \
    model.adapter.adapter_type="<linear, tiny_attn, or others from config sub-sections of `adapter`>" \
    model.adapter.adapter_module_name=<null, or str module. Type: encoder, decoder, joint, or multiple with + between them> \
    model.adapter.linear.in_features=<dimension of the layer outputs of the model> \
    model.adapter.linear.dim=32 \
    model.adapter.linear.dropout=0.0 \
    model.train_ds.manifest_filepath=<Path to manifest> \
    model.train_ds.batch_size=16 \
    model.validation_ds.manifest_filepath=<Path to manifest> \
    model.validation_ds.batch_size=16 \
    model.optim.lr=0.001 \
    model.optim.weight_decay=0.0 \
    model.optim.sched.warmup_steps=100 \
    trainer.max_steps=300 \
    trainer.devices=1 \
    trainer.precision=32 \
    exp_manager.exp_dir=<Some directory for experiment manager>

# Hyper Parmaeter Search

python train_asr_adapter.py \
    --config-path="../conf/asr_adapters" \
    --config-name="asr_adaptation_hp.yaml" \
    -m \
    model.pretrained_model=null \
    model.nemo_model=null \
    model.adapter.adapter_name=<Unique adapter name> \
    model.adapter.adapter_type="<linear, tiny_attn, or others from config sub-sections of `adapter`>" \
    model.adapter.adapter_module_name=<null, or str module. Type: encoder, decoder, joint, or multiple with + between them> \
    model.adapter.linear.in_features=<dimension of the layer outputs of the model> \
    model.train_ds.manifest_filepath=<Path to manifest> \
    model.train_ds.batch_size=16 \
    model.validation_ds.manifest_filepath=<Path to manifest> \
    model.validation_ds.batch_size=16 \
    exp_manager.exp_dir="<some directory>" \
    exp_manager.create_wandb_logger=true \
    exp_manager.wandb_logger_kwargs.project="<Project Name>" \
    ++delete_ckpt_after_train=True

# Fine-tune a model

While adaptation is very efficient for low-resource datasets, it imposes several restrictions -

- The vocabulary of the new dataset must be supported by the pre-existing vocabulary or tokenizer.
    If tokens exist outside this scope, the adapter will have to learn UNK tokens (or fail entirely
    for character based models).

- As a consequence of the above, the language of the new dataset must be the same as the original model.
    There is ongoing research to enable more sophisticated adapters for other languages.

When adapters cannot be readily used due to the above limitations, fine-tuning may be a better alternative.

For documentation on fine-tuning a model, please visit -
https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/configs.html#fine-tuning-configurations

# Pretrained Models

For documentation on existing pretrained models, please visit -
https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/results.html

"""
import os
from dataclasses import is_dataclass

import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf, open_dict

from nemo.collections.asr.models import ASRModel
from nemo.core import adapter_mixins
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import clean_exp_ckpt, exp_manager


def update_model_config_to_support_adapter(model_cfg, current_cfg):
    with open_dict(model_cfg):
        # Override prediction logging in config
        model_cfg.log_prediction = current_cfg.model.get('log_prediction', False)

        # Update encoder adapter compatible config
        adapter_metadata = adapter_mixins.get_registered_adapter(model_cfg.encoder._target_)
        if adapter_metadata is not None:
            model_cfg.encoder._target_ = adapter_metadata.adapter_class_path


def update_model_cfg(original_cfg, new_cfg):
    with open_dict(original_cfg), open_dict(new_cfg):
        # force inject some keys into the config
        whitelist_keys = ['num_workers', 'pin_memory']
        for wkey in whitelist_keys:
            if wkey in new_cfg:
                original_cfg[wkey] = new_cfg[wkey]
                print(f"Injecting white listed key `{wkey}` into config")

        # drop keys which don't exist in old config and are not whitelisted
        new_keys = list(new_cfg.keys())
        for key in new_keys:
            if key not in original_cfg:
                new_cfg.pop(key)
                print("Removing unavailable key from config :", key)

        new_cfg = OmegaConf.merge(original_cfg, new_cfg)
    return new_cfg


def add_global_adapter_cfg(model, global_adapter_cfg):
    # Convert to DictConfig from dict or Dataclass
    if is_dataclass(global_adapter_cfg):
        global_adapter_cfg = OmegaConf.structured(global_adapter_cfg)

    if not isinstance(global_adapter_cfg, DictConfig):
        global_adapter_cfg = DictConfig(global_adapter_cfg)

    # Update the model.cfg with information about the new adapter global cfg
    with open_dict(global_adapter_cfg), open_dict(model.cfg):
        if 'adapters' not in model.cfg:
            model.cfg.adapters = OmegaConf.create({})

        # Add the global config for adapters to the model's internal config
        model.cfg.adapters[model.adapter_global_cfg_key] = global_adapter_cfg

        # Update all adapter modules (that already exist) with this global adapter config
        model.update_adapter_cfg(model.cfg.adapters)


@hydra_runner(config_path="../conf/asr_adapters", config_name="asr_adaptation.yaml")
def main(cfg):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    if cfg.model.pretrained_model is None and cfg.model.nemo_model is None:
        raise ValueError("Either set `cfg.model.nemo_model` or `cfg.model.pretrained_model`")
    if cfg.model.pretrained_model is not None and cfg.model.nemo_model is not None:
        raise ValueError("Cannot set both `cfg.model.nemo_model` and `cfg.model.pretrained_model`. Select one only.")

    trainer = pl.Trainer(**cfg.trainer)
    exp_log_dir = exp_manager(trainer, cfg.get("exp_manager", None))

    if cfg.model.pretrained_model is not None:
        model_cfg = ASRModel.from_pretrained(cfg.model.pretrained_model, return_config=True)
        update_model_config_to_support_adapter(model_cfg, cfg)
        model = ASRModel.from_pretrained(cfg.model.pretrained_model, override_config_path=model_cfg, trainer=trainer)

    else:
        model_cfg = ASRModel.restore_from(cfg.model.nemo_model, return_config=True)
        update_model_config_to_support_adapter(model_cfg, cfg)
        model = ASRModel.restore_from(cfg.model.nemo_model, override_config_path=model_cfg, trainer=trainer)

    # Setup model for finetuning (train and validation only)
    cfg.model.train_ds = update_model_cfg(model.cfg.train_ds, cfg.model.train_ds)
    model.setup_training_data(cfg.model.train_ds)

    if 'validation_ds' in cfg.model:
        cfg.model.validation_ds = update_model_cfg(model.cfg.validation_ds, cfg.model.validation_ds)
        model.setup_multiple_validation_data(cfg.model.validation_ds)

    # Setup optimizer
    model.setup_optimization(cfg.model.optim)

    # Setup spec augmentation
    if 'spec_augment' in cfg.model:
        model.spec_augmentation = model.from_config_dict(cfg.model.spec_augment)
    else:
        model.spec_augmentation = None
        del model.cfg.spec_augment

    # Setup adapters
    with open_dict(cfg.model.adapter):
        # Extract the name of the adapter (must be give for training)
        adapter_name = cfg.model.adapter.pop("adapter_name")
        adapter_type = cfg.model.adapter.pop("adapter_type")
        adapter_module_name = cfg.model.adapter.pop("adapter_module_name", None)
        adapter_state_dict_name = cfg.model.adapter.pop("adapter_state_dict_name", None)

        # Resolve the config of the specified `adapter_type`
        if adapter_type not in cfg.model.adapter.keys():
            raise ValueError(
                f"Adapter type ({adapter_type}) config could not be found. Adapter setup config - \n"
                f"{OmegaConf.to_yaml(cfg.model.adapter)}"
            )

        adapter_type_cfg = cfg.model.adapter[adapter_type]
        print(f"Found `{adapter_type}` config :\n" f"{OmegaConf.to_yaml(adapter_type_cfg)}")

        # Augment adapter name with module name, if not provided by user
        if adapter_module_name is not None and ':' not in adapter_name:
            adapter_name = f'{adapter_module_name}:{adapter_name}'

        # Extract the global adapter config, if provided
        adapter_global_cfg = cfg.model.adapter.pop(model.adapter_global_cfg_key, None)
        if adapter_global_cfg is not None:
            add_global_adapter_cfg(model, adapter_global_cfg)

    model.add_adapter(adapter_name, cfg=adapter_type_cfg)
    assert model.is_adapter_available()

    # Disable all other adapters, enable just the current adapter.
    model.set_enabled_adapters(enabled=False)  # disable all adapters prior to training
    model.set_enabled_adapters(adapter_name, enabled=True)  # enable just one adapter by name

    # First, Freeze all the weights of the model (not just encoder, everything)
    model.freeze()
    # Activate dropout() and other modules that depend on train mode.
    model = model.train()
    # Then, Unfreeze just the adapter weights that were enabled above (no part of encoder/decoder/joint/etc)
    model.unfreeze_enabled_adapters()

    # Update model config prior to training
    model.cfg = model.cfg

    # Finally, train model
    trainer.fit(model)

    # Save the adapter state dict
    if adapter_state_dict_name is not None:
        state_path = exp_log_dir if exp_log_dir is not None else os.getcwd()
        ckpt_path = os.path.join(state_path, "checkpoints")
        if os.path.exists(ckpt_path):
            state_path = ckpt_path
        state_path = os.path.join(state_path, adapter_state_dict_name)

        # Save the adapter modules in a seperate file
        model.save_adapters(str(state_path))

    if 'delete_ckpt_after_train' in cfg:
        delete_ckpt_after_train = cfg.delete_ckpt_after_train
        if delete_ckpt_after_train:
            # Remove PTL ckpt file, and potentially also remove .nemo file to conserve storage space.
            clean_exp_ckpt(exp_log_dir, remove_ckpt=True, remove_nemo=False)


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
