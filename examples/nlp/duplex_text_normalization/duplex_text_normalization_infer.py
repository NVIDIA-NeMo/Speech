# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
This script contains an example on how to run inference with the DuplexTextNormalizationModel.
DuplexTextNormalizationModel is essentially a wrapper class around DuplexTaggerModel and DuplexDecoderModel.
Therefore, two trained NeMo models should be specified to run the joint evaluation
(one is a trained DuplexTaggerModel and the other is a trained DuplexDecoderModel).

This script can perform inference for 2 settings:
1. inference from a raw file (no labels required). Each line of the file represents a single example for inference.
    Specify in inference.from_file and inference.batch_size parameters.

    python duplex_text_normalization_infer.py \
        tagger_pretrained_model=PATH_TO_TRAINED_TAGGER \
        decoder_pretrained_model=PATH_TO_TRAINED_DECODER \
        mode={tn,itn,joint} \
        lang={en,ru,de} \
        inference.from_file=PATH_TO_RAW_TEXT_FILE

    The predictions will be saved at "_norm" and "_denorm" files.

2. Interactive inference (one query at a time), set inference.interactive to True to enter the interactive mode
    python duplex_text_normalization_infer.py \
        tagger_pretrained_model=PATH_TO_TRAINED_TAGGER \
        decoder_pretrained_model=PATH_TO_TRAINED_DECODER \
        mode={tn,itn,joint} \
        lang={en,ru,de} \
        inference.interactive=true

This script uses the `/examples/nlp/duplex_text_normalization/conf/duplex_tn_config.yaml`
config file by default. The other option is to set another config file via command
line arguments by `--config-name=CONFIG_FILE_PATH'.
"""


import os
from typing import List

from helpers import DECODER_MODEL, TAGGER_MODEL, instantiate_model_and_trainer
from nn_wfst.en.electronic.normalize import ElectronicNormalizer
from nn_wfst.en.whitelist.normalize import WhitelistNormalizer
from omegaconf import DictConfig, OmegaConf

from nemo.collections.nlp.data.text_normalization import constants
from nemo.collections.nlp.models import DuplexTextNormalizationModel
from nemo.core.config import hydra_runner
from nemo.utils import logging

try:
    from nemo_text_processing.text_normalization.data_loader_utils import post_process_punct
except (ImportError, ModuleNotFoundError):
    raise ModuleNotFoundError(
        "The package `nemo_text_processing` was not installed in this environment. Please refer to"
        " https://github.com/NVIDIA/NeMo-text-processing and install this package before using "
        "this script"
    )


@hydra_runner(config_path="conf", config_name="duplex_tn_config")
def main(cfg: DictConfig) -> None:
    logging.debug(f'Config Params: {OmegaConf.to_yaml(cfg)}')
    lang = cfg.lang

    if cfg.decoder_pretrained_model is None or cfg.tagger_pretrained_model is None:
        raise ValueError("Both pre-trained models (DuplexTaggerModel and DuplexDecoderModel) should be provided.")
    tagger_trainer, tagger_model = instantiate_model_and_trainer(cfg, TAGGER_MODEL, False)
    decoder_trainer, decoder_model = instantiate_model_and_trainer(cfg, DECODER_MODEL, False)
    decoder_model.max_sequence_len = 512
    tagger_model.max_sequence_len = 512
    tn_model = DuplexTextNormalizationModel(tagger_model, decoder_model, lang)

    if lang == constants.ENGLISH:
        normalizer_electronic = ElectronicNormalizer(input_case="cased", lang=lang, deterministic=True)
        normalizer_whitelist = WhitelistNormalizer(input_case="cased", lang=lang, deterministic=True)

    if cfg.inference.get("from_file", False):
        text_file = cfg.inference.from_file
        logging.info(f'Running inference on {text_file}...')
        if not os.path.exists(text_file):
            raise ValueError(f'{text_file} not found.')

        with open(text_file, 'r') as f:
            lines = f.readlines()

        if lang == constants.ENGLISH:
            new_lines = normalizer_electronic.normalize_list(lines)
            lines = [
                post_process_punct(input=input_, normalized_text=norm_) for input_, norm_ in zip(lines, new_lines)
            ]
            new_lines = normalizer_whitelist.normalize_list(lines)
            lines = [
                post_process_punct(input=input_, normalized_text=norm_) for input_, norm_ in zip(lines, new_lines)
            ]

        def _get_predictions(lines: List[str], mode: str, batch_size: int, text_file: str):
            """ Runs inference on a batch data without labels and saved predictions to a file. """
            assert mode in ['tn', 'itn']
            file_name, extension = os.path.splitext(text_file)
            batch, all_preds = [], []
            for i, line in enumerate(lines):
                batch.append(line.strip())
                if len(batch) == batch_size or i == len(lines) - 1:
                    outputs = tn_model._infer(batch, [constants.DIRECTIONS_TO_MODE[mode]] * len(batch),)
                    all_preds.extend([x for x in outputs[-1]])
                    batch = []
            assert len(all_preds) == len(lines)
            out_file = f'{file_name}_{mode}{extension}'
            with open(f'{out_file}', 'w') as f_out:
                f_out.write("\n".join(all_preds))
            logging.info(f'Predictions for {mode} save to {out_file}.')

        batch_size = cfg.inference.get("batch_size", 8)
        if cfg.mode in ['tn', 'joint']:
            # TN mode
            _get_predictions(lines, 'tn', batch_size, text_file)
        if cfg.mode in ['itn', 'joint']:
            # ITN mode
            _get_predictions(lines, 'itn', batch_size, text_file)

    else:
        print('Entering interactive mode.')
        done = False
        while not done:
            print('Type "STOP" to exit.')
            test_input = input('Input a test input:')
            if test_input == "STOP":
                done = True
            if not done:
                if lang == constants.ENGLISH:
                    new_input = normalizer_electronic.normalize(test_input, verbose=False)
                    test_input = post_process_punct(input=test_input, normalized_text=new_input)
                    new_input = normalizer_whitelist.normalize(test_input, verbose=False)
                    test_input = post_process_punct(input=test_input, normalized_text=new_input)
                directions = []
                inputs = []
                if cfg.mode in ['itn', 'joint']:
                    directions.append(constants.DIRECTIONS_TO_MODE[constants.ITN_MODE])
                    inputs.append(test_input)
                if cfg.mode in ['tn', 'joint']:
                    directions.append(constants.DIRECTIONS_TO_MODE[constants.TN_MODE])
                    inputs.append(test_input)
                outputs = tn_model._infer(inputs, directions)[-1]
                if cfg.mode in ['joint', 'itn']:
                    print(f'Prediction (ITN): {outputs[0]}')
                if cfg.mode in ['joint', 'tn']:
                    print(f'Prediction (TN): {outputs[-1]}')


if __name__ == '__main__':
    main()
