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


"""
A script to convert a Nemo ASR Hybrid model file (.nemo) to a Nemo ASR CTC or RNNT model file (.nemo)

This allows you to train a RNNT-CTC Hybrid model, but then convert it into a pure CTC or pure RNNT model for use
in NeMo. The resulting .nemo file will be a pure CTC or RNNT model, and can be used like any other .nemo model
including in nemo2riva.

Usage: python convert_nemo_asr_hybrid_to_ctc.py -i /path/to/hybrid.nemo -o /path/to/saved_ctc_model.nemo -t ctc|rnnt

"""


import argparse
import os
from copy import deepcopy

import torch
from omegaconf import OmegaConf

from nemo.collections.asr.models import (
    ASRModel,
    EncDecCTCModel,
    EncDecCTCModelBPE,
    EncDecRNNTBPEModel,
    EncDecRNNTModel,
)
from nemo.utils import logging


def extract_model_ctc(args, hybrid_model):
    """
    A function which converts a hybrid model to a pure ctc model.
    Args:
        args (argparse): the args collection from ArgumentParser created by running this script
        hybrid_model (ASRModel): the loaded hybrid RNNT-CTC Nemo model
    """
    BPE = False
    ctc_class = EncDecCTCModel
    if 'tokenizer' in hybrid_model.cfg.keys():
        BPE = True
        ctc_class = EncDecCTCModelBPE

    hybrid_model_cfg = OmegaConf.to_container(hybrid_model.cfg)

    new_cfg = deepcopy(hybrid_model_cfg)
    new_cfg['ctc_reduction'] = hybrid_model_cfg['aux_ctc']['ctc_reduction']
    new_cfg['decoder'] = hybrid_model_cfg['aux_ctc']['decoder']
    del new_cfg['compute_eval_loss']
    del new_cfg['model_defaults']
    del new_cfg['joint']
    del new_cfg['decoding']
    del new_cfg['aux_ctc']
    del new_cfg['loss']
    if BPE and 'labels' in new_cfg:
        del new_cfg['labels']
    elif (not BPE) and 'tokenizer' in new_cfg:
        del new_cfg['tokenizer']
    del new_cfg['target']
    del new_cfg['nemo_version']

    new_cfg_oc = OmegaConf.create(new_cfg)

    # we call restore_from with strict=False because the .nemo file we're restoring from is a hybrid model, which will have named
    # tensors in the state_dict that do not exist in the pure CTC model class, which would result in an exception with strict=True
    ctc_model = ctc_class.restore_from(
        args.input, map_location=torch.device('cpu'), override_config_path=new_cfg_oc, strict=False
    )

    assert all(
        [
            torch.allclose(hybrid_model.state_dict()[x], ctc_model.state_dict()[x])
            for x in hybrid_model.state_dict().keys()
            if x.split('.')[0] in ['preprocessor', 'encoder']
        ]
    ), "Encoder and preprocessor state dicts don't match!"

    ctc_model.decoder.load_state_dict(hybrid_model.ctc_decoder.state_dict())

    assert all(
        [
            torch.allclose(hybrid_model.ctc_decoder.state_dict()[x], ctc_model.decoder.state_dict()[x])
            for x in hybrid_model.ctc_decoder.state_dict().keys()
        ]
    ), "Decoder state_dict load failed!"

    assert isinstance(ctc_model, ctc_class), "Extracted CTC model is of the wrong expected class!"

    return ctc_model


def extract_model_rnnt(args, hybrid_model):
    """
    A function which converts a hybrid model to a pure rnnt model.
    Args:
        args (argparse): the args collection from ArgumentParser created by running this script
        hybrid_model (ASRModel): the loaded hybrid RNNT-CTC Nemo model
    """
    BPE = False
    rnnt_class = EncDecRNNTModel
    if 'tokenizer' in hybrid_model.cfg.keys():
        BPE = True
        rnnt_class = EncDecRNNTBPEModel

    hybrid_model_cfg = OmegaConf.to_container(hybrid_model.cfg)

    new_cfg = deepcopy(hybrid_model_cfg)
    del new_cfg['aux_ctc']
    if BPE and 'labels' in new_cfg:
        del new_cfg['labels']
    elif (not BPE) and 'tokenizer' in new_cfg:
        del new_cfg['tokenizer']
    del new_cfg['target']
    del new_cfg['nemo_version']

    new_cfg_oc = OmegaConf.create(new_cfg)

    # we call restore_from with strict=False because the .nemo file we're restoring from is a hybrid model, which will have named
    # tensors in the state_dict that do not exist in the pure RNNT model class, which would result in an exception with strict=True
    rnnt_model = rnnt_class.restore_from(
        args.input, map_location=torch.device('cpu'), override_config_path=new_cfg_oc, strict=False
    )

    assert all(
        [
            torch.allclose(hybrid_model.state_dict()[x], rnnt_model.state_dict()[x])
            for x in hybrid_model.state_dict().keys()
            if x.split('.')[0] in ['preprocessor', 'encoder', 'decoder', 'joint']
        ]
    ), "State dict values mismatch, something went wrong!"

    assert isinstance(rnnt_model, rnnt_class), "Extracted RNNT model is of the wrong expected class!"

    return rnnt_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True, type=str, help='path to Nemo Hybrid model .nemo file')
    parser.add_argument('-o', '--output', required=True, type=str, help='path and name of output .nemo file')
    parser.add_argument(
        '-t',
        '--model_type',
        required=False,
        type=str,
        default='ctc',
        choices=['ctc', 'rnnt'],
        help='whether to output a ctc or rnnt model from the hybrid',
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        logging.critical(f'Input file [ {args.input} ] does not exist or cannot be found. Aborting.')
        exit(255)

    hybrid_model = ASRModel.restore_from(args.input, map_location=torch.device('cpu'))

    if args.model_type == 'ctc':
        output_model = extract_model_ctc(args, hybrid_model)
    elif args.model_type == 'rnnt':
        output_model = extract_model_rnnt(args, hybrid_model)
    else:
        logging.critical(
            f"the model_type arg must be one of 'ctc' or 'rnnt', received unknown value: '{args.model_type}'. Aborting."
        )
        exit(255)

    output_model.save_to(args.output)
    logging.info(f'Converted {args.model_type.upper()} model was successfully saved to {args.output}')
