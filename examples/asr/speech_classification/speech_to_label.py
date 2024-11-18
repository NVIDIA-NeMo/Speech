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
# Task 1: Speech Command Recognition

## Preparing the dataset
Use the `process_speech_commands_data.py` script under <NEMO_ROOT>/scripts/dataset_processing in order to prepare the dataset.

```sh
python <NEMO_ROOT>/scripts/dataset_processing/process_speech_commands_data.py \
    --data_root=<absolute path to where the data should be stored> \
    --data_version=<either 1 or 2, indicating version of the dataset> \
    --class_split=<either "all" or "sub", indicates whether all 30/35 classes should be used, or the 10+2 split should be used> \
    --rebalance \
    --log
```

## Train to convergence
```sh
python speech_to_label.py \
    # (Optional: --config-path=<path to dir of configs> --config-name=<name of config without .yaml>) \
    model.train_ds.manifest_filepath="<path to train manifest>" \
    model.validation_ds.manifest_filepath=["<path to val manifest>","<path to test manifest>"] \
    trainer.devices=2 \
    trainer.accelerator="gpu" \
    strategy="ddp" \
    trainer.max_epochs=200 \
    exp_manager.create_wandb_logger=True \
    exp_manager.wandb_logger_kwargs.name="MatchboxNet-3x1x64-v1" \
    exp_manager.wandb_logger_kwargs.project="MatchboxNet-v1" \
    +trainer.precision=16 \
    +trainer.amp_level=O1  # needed if using PyTorch < 1.6
```


# Task 2: Voice Activity Detection

## Preparing the dataset
Use the `process_vad_data.py` script under <NEMO_ROOT>/scripts/dataset_processing in order to prepare the dataset.

```sh
python process_vad_data.py \
    --out_dir=<output path to where the generated manifest should be stored> \
    --speech_data_root=<path where the speech data are stored> \
    --background_data_root=<path where the background data are stored> \
    --rebalance_method=<'under' or 'over' of 'fixed'> \
    --log
    (Optional --demo (for demonstration in tutorial). If you want to use your own background noise data, make sure to delete --demo)
```

## Train to convergence
```sh
python speech_to_label.py \
    --config-path=<path to dir of configs e.g. "conf">
    --config-name=<name of config without .yaml e.g. "matchboxnet_3x1x64_vad"> \
    model.train_ds.manifest_filepath="<path to train manifest>" \
    model.validation_ds.manifest_filepath=["<path to val manifest>","<path to test manifest>"] \
    trainer.devices=2 \
    trainer.accelerator="gpu" \
    strategy="ddp" \
    trainer.max_epochs=200 \
    exp_manager.create_wandb_logger=True \
    exp_manager.wandb_logger_kwargs.name="MatchboxNet-3x1x64-vad" \
    exp_manager.wandb_logger_kwargs.project="MatchboxNet-vad" \
    +trainer.precision=16 \
    +trainer.amp_level=O1  # needed if using PyTorch < 1.6
```

# Task 3: Language Identification

## Preparing the dataset
Use the `filelist_to_manifest.py` script under <NEMO_ROOT>/scripts/speaker_tasks in order to prepare the dataset.
```

## Train to convergence
```sh
python speech_to_label.py \
    --config-path=<path to dir of configs e.g. "../conf/lang_id">
    --config-name=<name of config without .yaml e.g. "titanet_large"> \
    model.train_ds.manifest_filepath="<path to train manifest>" \
    model.validation_ds.manifest_filepath="<path to val manifest>" \
    model.train_ds.augmentor.noise.manifest_path="<path to noise manifest>" \
    model.train_ds.augmentor.impulse.manifest_path="<path to impulse manifest>" \
    model.decoder.num_classes=<num of languages> \
    trainer.devices=2 \
    trainer.max_epochs=40 \
    exp_manager.create_wandb_logger=True \
    exp_manager.wandb_logger_kwargs.name="titanet" \
    exp_manager.wandb_logger_kwargs.project="langid" \
    +exp_manager.checkpoint_callback_params.monitor="val_acc_macro" \
    +exp_manager.checkpoint_callback_params.mode="max" \
    +trainer.precision=16 \
```


# Optional: Use tarred dataset to speed up data loading. Apply to both tasks.
## Prepare tarred dataset. 
   Prepare ONE manifest that contains all training data you would like to include. Validation should use non-tarred dataset.
   Note that it's possible that tarred datasets impacts validation scores because it drop values in order to have same amount of files per tarfile; 
   Scores might be off since some data is missing. 

   Use the `convert_to_tarred_audio_dataset.py` script under <NEMO_ROOT>/scripts/speech_recognition in order to prepare tarred audio dataset.
   For details, please see TarredAudioToClassificationLabelDataset in <NEMO_ROOT>/nemo/collections/asr/data/audio_to_label.py

python speech_to_label.py \
    --config-path=<path to dir of configs e.g. "conf">
    --config-name=<name of config without .yaml e.g. "matchboxnet_3x1x64_vad"> \
    model.train_ds.manifest_filepath=<path to train tarred_audio_manifest.json> \
    model.train_ds.is_tarred=True \
    model.train_ds.tarred_audio_filepaths=<path to train tarred audio dataset e.g. audio_{0..2}.tar> \
    +model.train_ds.num_worker=<num_shards used generating tarred dataset> \
    model.validation_ds.manifest_filepath=<path to validation audio_manifest.json>\
    trainer.devices=2 \
    trainer.accelerator="gpu" \
    strategy="ddp" \ \
    trainer.max_epochs=200 \
    exp_manager.create_wandb_logger=True \
    exp_manager.wandb_logger_kwargs.name="MatchboxNet-3x1x64-vad" \
    exp_manager.wandb_logger_kwargs.project="MatchboxNet-vad" \
    +trainer.precision=16 \
    +trainer.amp_level=O1  # needed if using PyTorch < 1.6

# Fine-tune a model

For documentation on fine-tuning this model, please visit -
https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/configs.html#fine-tuning-configurations

# Pretrained Models

For documentation on existing pretrained models, please visit -
https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/speech_classification/results.html#

"""
import lightning.pytorch as pl
import torch
from omegaconf import OmegaConf

from nemo.collections.asr.models import EncDecClassificationModel, EncDecSpeakerLabelModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


@hydra_runner(config_path="../conf/matchboxnet", config_name="matchboxnet_3x1x64_v1")
def main(cfg):

    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))

    if 'titanet' in cfg.name.lower():
        model = EncDecSpeakerLabelModel(cfg=cfg.model, trainer=trainer)
    else:
        model = EncDecClassificationModel(cfg=cfg.model, trainer=trainer)

    # Initialize the weights of the model from another model, if provided via config
    model.maybe_init_from_pretrained_checkpoint(cfg)
    trainer.fit(model)
    torch.distributed.destroy_process_group()

    if hasattr(cfg.model, 'test_ds') and cfg.model.test_ds.manifest_filepath is not None:
        if trainer.is_global_zero:
            trainer = pl.Trainer(devices=1, accelerator=cfg.trainer.accelerator, strategy=cfg.trainer.strategy)
            if model.prepare_test(trainer):
                trainer.test(model)


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
