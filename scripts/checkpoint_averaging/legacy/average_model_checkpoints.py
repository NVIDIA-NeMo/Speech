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

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
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
# Changes to script
Change the script to import the NeMo model class you would like to load a checkpoint for,
then update the model constructor to use this model class. This can be found by the line:

<<< Change model class here ! >>>

By default, this script imports and creates the `EncDecCTCModelBPE` class but it can be
changed to any NeMo Model.

# Run the script

## Saving a .nemo model file (loaded with ModelPT.restore_from(...))
HYDRA_FULL_ERROR=1 python average_model_checkpoints.py \
    --config-path="<path to config directory>" \
    --config-name="<config name>" \
    name=<name of the averaged checkpoint> \
    +checkpoint_dir=<OPTIONAL: directory of checkpoint> \
    +checkpoint_paths=\"[/path/to/ptl_1.ckpt,/path/to/ptl_2.ckpt,/path/to/ptl_3.ckpt,...]\"


## Saving an averaged pytorch checkpoint (loaded with torch.load(...))
HYDRA_FULL_ERROR=1 python average_model_checkpoints.py \
    --config-path="<path to config directory>" \
    --config-name="<config name>" \
    name=<name of the averaged checkpoint> \
     +checkpoint_dir=<OPTIONAL: directory of checkpoint> \
    +checkpoint_paths=\"[/path/to/ptl_1.ckpt,/path/to/ptl_2.ckpt,/path/to/ptl_3.ckpt,...]\" \
    +save_ckpt_only=true

"""

import os

import lightning.pytorch as pl
import torch
from omegaconf import OmegaConf, open_dict

# Change this import to the model you would like to average
from nemo.collections.asr.models import EncDecCTCModelBPE
from nemo.core.config import hydra_runner
from nemo.utils import logging


def process_config(cfg: OmegaConf):
    """
    Process config
    """
    if 'name' not in cfg or cfg.name is None:
        raise ValueError("`cfg.name` must be provided to save a model checkpoint")

    if 'checkpoint_paths' not in cfg or cfg.checkpoint_paths is None:
        raise ValueError(
            "`cfg.checkpoint_paths` must be provided as a list of one or more str paths to "
            "pytorch lightning checkpoints"
        )

    save_ckpt_only = False

    with open_dict(cfg):
        name_prefix = cfg.name
        checkpoint_paths = cfg.pop('checkpoint_paths')

        if 'checkpoint_dir' in cfg:
            checkpoint_dir = cfg.pop('checkpoint_dir')
        else:
            checkpoint_dir = None

        if 'save_ckpt_only' in cfg:
            save_ckpt_only = cfg.pop('save_ckpt_only')

    if type(checkpoint_paths) not in (list, tuple):
        checkpoint_paths = str(checkpoint_paths).replace("[", "").replace("]", "")
        checkpoint_paths = checkpoint_paths.split(",")
        checkpoint_paths = [ckpt_path.strip() for ckpt_path in checkpoint_paths]

    if checkpoint_dir is not None:
        checkpoint_paths = [os.path.join(checkpoint_dir, path) for path in checkpoint_paths]

    return name_prefix, checkpoint_paths, save_ckpt_only


@hydra_runner(config_path=None, config_name=None)
def main(cfg):
    """
    Main function
    """

    logging.info("This script is deprecated and will be removed in the 25.01 release.")

    name_prefix, checkpoint_paths, save_ckpt_only = process_config(cfg)

    if not save_ckpt_only:
        trainer = pl.Trainer(**cfg.trainer)

        # <<< Change model class here ! >>>
        # Model architecture which will contain the averaged checkpoints
        # Change the model constructor to the one you would like (if needed)
        model = EncDecCTCModelBPE(cfg=cfg.model, trainer=trainer)

    """ < Checkpoint Averaging Logic > """
    # load state dicts
    n = len(checkpoint_paths)
    avg_state = None

    logging.info(f"Averaging {n} checkpoints ...")

    for ix, path in enumerate(checkpoint_paths):
        checkpoint = torch.load(path, map_location='cpu')

        if 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']

        if ix == 0:
            # Initial state
            avg_state = checkpoint

            logging.info(f"Initialized average state dict with checkpoint : {path}")
        else:
            # Accumulated state
            for k in avg_state:
                avg_state[k] = avg_state[k] + checkpoint[k]

            logging.info(f"Updated average state dict with state from checkpoint : {path}")

    for k in avg_state:
        if str(avg_state[k].dtype).startswith("torch.int"):
            # For int type, not averaged, but only accumulated.
            # e.g. BatchNorm.num_batches_tracked
            pass
        else:
            avg_state[k] = avg_state[k] / n

    # Save model
    if save_ckpt_only:
        ckpt_name = name_prefix + '-averaged.ckpt'
        torch.save(avg_state, ckpt_name)

        logging.info(f"Averaged pytorch checkpoint saved as : {ckpt_name}")
    else:
        # Set model state
        logging.info("Loading averaged state dict in provided model")
        model.load_state_dict(avg_state, strict=True)

        ckpt_name = name_prefix + '-averaged.nemo'
        model.save_to(ckpt_name)

        logging.info(f"Averaged model saved as : {ckpt_name}")


if __name__ == '__main__':
    main()
