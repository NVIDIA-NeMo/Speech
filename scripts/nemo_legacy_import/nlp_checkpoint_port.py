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

import argparse
import logging
import sys

import pytorch_lightning as pl
from omegaconf import OmegaConf, open_dict

from nemo.collections.nlp.parts.nlp_overrides import NLPSaveRestoreConnector
from nemo.core import ModelPT
from nemo.core.config import TrainerConfig


def get_args(argv):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=f"Update NLP models trained on previous versions to current version ",
    )
    parser.add_argument("source", help="Source .nemo file")
    parser.add_argument("out", help="Location to write result to")
    parser.add_argument("--megatron-legacy", help="If the source model is megatron-bert trained on NeMo < 1.5")
    parser.add_argument(
        "--megatron-checkpoint",
        type=str,
        help="Path of the MegatronBert nemo checkpoint converted from MegatronLM using megatron_lm_ckpt_to_nemo.py file (Not NLP model checkpoint)",
    )
    parser.add_argument("--verbose", default=None, help="Verbose level for logging, numeric")
    args = parser.parse_args(argv)
    return args


def nemo_convert(argv):
    args = get_args(argv)
    loglevel = logging.INFO
    # assuming loglevel is bound to the string value obtained from the
    # command line argument. Convert to upper case to allow the user to
    # specify --log=DEBUG or --log=debug
    if args.verbose is not None:
        numeric_level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError('Invalid log level: %s' % numeric_level)
        loglevel = numeric_level

    logger = logging.getLogger(__name__)
    if logger.handlers:
        for handler in logger.handlers:
            logger.removeHandler(handler)
    logging.basicConfig(level=loglevel, format='%(asctime)s [%(levelname)s] %(message)s')
    logging.info("Logging level set to {}".format(loglevel))

    """Convert a .nemo saved model trained on previous versions of nemo into a nemo fie with current version."""
    nemo_in = args.source
    out = args.out

    # Create a PL trainer object which is required for restoring Megatron models
    cfg_trainer = TrainerConfig(
        devices=1,
        accelerator='auto',
        num_nodes=1,
        # Need to set the following two to False as ExpManager will take care of them differently.
        logger=False,
        enable_checkpointing=False,
    )
    cfg_trainer = OmegaConf.to_container(OmegaConf.create(cfg_trainer))
    trainer = pl.Trainer(**cfg_trainer)

    logging.info("Restoring NeMo model from '{}'".format(nemo_in))
    try:
        # If the megatron based NLP model was trained on NeMo < 1.5, then we need to update the lm_checkpoint on the model config
        if args.megatron_legacy:
            if args.megatron_checkpoint:
                connector = NLPSaveRestoreConnector()
                model_cfg = ModelPT.restore_from(
                    restore_path=nemo_in, save_restore_connector=connector, trainer=trainer, return_config=True
                )
                OmegaConf.set_struct(model_cfg, True)
                with open_dict(model_cfg):
                    model_cfg.language_model.lm_checkpoint = args.megatron_checkpoint
                    model_cfg['megatron_legacy'] = True
                    model_cfg['masked_softmax_fusion'] = False
                    model_cfg['bias_gelu_fusion'] = False
                model = ModelPT.restore_from(
                    restore_path=nemo_in,
                    save_restore_connector=connector,
                    trainer=trainer,
                    override_config_path=model_cfg,
                )
            else:
                logging.error("Megatron Checkpoint must be provided if Megatron legacy is chosen")
        else:
            model = ModelPT.restore_from(restore_path=nemo_in, trainer=trainer)
        logging.info("Model {} restored from '{}'".format(model.cfg.target, nemo_in))

        # Save the model
        model.save_to(out)
        logging.info("Successfully converted to {}".format(out))

        del model
    except Exception as e:
        logging.error(
            "Failed to restore model from NeMo file : {}. Please make sure you have the latest NeMo package installed with [all] dependencies.".format(
                nemo_in
            )
        )
        raise e


if __name__ == '__main__':
    nemo_convert(sys.argv[1:])
