# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""This script converts old Jasper/QuartzNet models from NeMo 0.11.* to NeMo v1.0.0*
"""

import argparse

import torch
from omegaconf import DictConfig
from ruamel.yaml import YAML

import nemo.collections.asr as nemo_asr
from nemo.utils import logging


def get_parser():
    parser = argparse.ArgumentParser(description="Converts old Jasper/QuartzNet models to NeMo v1.0beta")
    parser.add_argument("--config_path", default=None, required=True, help="Path to model config (NeMo v1.0beta)")
    parser.add_argument("--encoder_ckpt", default=None, required=True, help="Encoder checkpoint path")
    parser.add_argument("--decoder_ckpt", default=None, required=True, help="Decoder checkpoint path")
    parser.add_argument("--output_path", default=None, required=True, help="Output checkpoint path (should be .nemo)")
    parser.add_argument(
        "--model_type",
        default='asr',
        type=str,
        choices=['asr', 'speech_label', 'speaker'],
        help="Type of decoder used by the model.",
    )

    return parser


def main(config_path, encoder_ckpt, decoder_ckpt, output_path, model_type):

    yaml = YAML(typ='safe')
    with open(config_path) as f:
        params = yaml.load(f)

    model = None
    if model_type == 'asr':
        logging.info("Creating ASR NeMo 1.0 model")
        model = nemo_asr.models.EncDecCTCModel(cfg=DictConfig(params['model']))
    elif model_type == 'speech_label':
        logging.info("Creating speech label NeMo 1.0 model")
        model = nemo_asr.models.EncDecClassificationModel(cfg=DictConfig(params['model']))
    else:
        logging.info("Creating Speaker Recognition NeMo 1.0 model")
        model = nemo_asr.models.EncDecSpeakerLabelModel(cfg=DictConfig(params['model']))

    model.encoder.load_state_dict(torch.load(encoder_ckpt))
    model.decoder.load_state_dict(torch.load(decoder_ckpt))
    logging.info("Succesfully ported old checkpoint")

    model.save_to(output_path)
    logging.info("new model saved at {}".format(output_path))


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args.config_path, args.encoder_ckpt, args.decoder_ckpt, args.output_path, args.model_type)
