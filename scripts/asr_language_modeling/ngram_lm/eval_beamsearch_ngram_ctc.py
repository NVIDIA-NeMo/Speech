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
#

"""
# This script would evaluate an N-gram language model trained with KenLM library (https://github.com/kpu/kenlm) in
# fusion with beam search decoders on top of a trained ASR model with CTC decoder. To evaluate a model with 
# Transducer (RNN-T) decoder use another script 'scripts/asr_language_modeling/ngram_lm/eval_beamsearch_ngram_transducer.py'. 
# NeMo's beam search decoders are capable of using the KenLM's N-gram models
# to find the best candidates. This script supports both character level and BPE level
# encodings and models which is detected automatically from the type of the model.
# You may train the LM model with 'scripts/asr_language_modeling/ngram_lm/train_kenlm.py'.

# Config Help

To discover all arguments of the script, please run :
python eval_beamsearch_ngram_ctc.py --help
python eval_beamsearch_ngram_ctc.py --cfg job

# USAGE

python eval_beamsearch_ngram_ctc.py nemo_model_file=<path to the .nemo file of the model> \
           input_manifest=<path to the evaluation JSON manifest file> \
           kenlm_model_file=<path to the binary KenLM model> \
           beam_width=[<list of the beam widths, separated with commas>] \
           beam_alpha=[<list of the beam alphas, separated with commas>] \
           beam_beta=[<list of the beam betas, separated with commas>] \
           preds_output_folder=<optional folder to store the predictions> \
           probs_cache_file=null \
           decoding_mode=beamsearch_ngram
           ...


# Grid Search for Hyper parameters

For grid search, you can provide a list of arguments as follows -

           beam_width=[4,8,16,....] \
           beam_alpha=[-2.0,-1.0,...,1.0,2.0] \
           beam_beta=[-1.0,-0.5,0.0,...,1.0] \

# You may find more info on how to use this script at:
# https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/asr_language_modeling.html

"""


import contextlib
import json
import os
import pickle
from dataclasses import dataclass, field, is_dataclass
from pathlib import Path
from typing import List, Optional

import editdistance
import numpy as np
import torch
from omegaconf import MISSING, OmegaConf
from sklearn.model_selection import ParameterGrid
from tqdm.auto import tqdm

import nemo.collections.asr as nemo_asr
from nemo.collections.asr.models import EncDecHybridRNNTCTCModel
from nemo.collections.asr.parts.submodules import ctc_beam_decoding
from nemo.collections.asr.parts.utils.transcribe_utils import PunctuationCapitalization, TextProcessingConfig
from nemo.core.config import hydra_runner
from nemo.utils import logging


# fmt: off


@dataclass
class EvalBeamSearchNGramConfig:
    """
    Evaluate an ASR model with beam search decoding and n-gram KenLM language model.
    """
    # # The path of the '.nemo' file of the ASR model or the name of a pretrained model (ngc / huggingface)
    nemo_model_file: str = MISSING

    # File paths
    input_manifest: str = MISSING  # The manifest file of the evaluation set
    kenlm_model_file: Optional[str] = None  # The path of the KenLM binary model file
    preds_output_folder: Optional[str] = None  # The optional folder where the predictions are stored
    hyps_cache_file: Optional[str] = None  # The cache file for storing the logprobs of the model

    # Parameters for inference
    batch_size: int = 16  # The batch size
    device: str = "cuda"  # The device to load the model onto to calculate log probabilities
    use_amp: bool = False  # Whether to use AMP if available to calculate log probabilities

    # Beam Search hyperparameters

    # The decoding scheme to be used for evaluation.
    # Can be one of ["greedy", "beamsearch", "beamsearch_ngram"]
    decoding_mode: str = "beamsearch_ngram"

    beam_width: List[int] = field(default_factory=lambda: [128])  # The width or list of the widths for the beam search decoding
    beam_alpha: List[float] = field(default_factory=lambda: [1.0])  # The alpha parameter or list of the alphas for the beam search decoding
    beam_beta: List[float] = field(default_factory=lambda: [0.0])  # The beta parameter or list of the betas for the beam search decoding

    decoding_strategy: str = "beam"
    decoding: ctc_beam_decoding.BeamCTCInferConfig = field(default_factory=lambda: ctc_beam_decoding.BeamCTCInferConfig(beam_size=128))
    
    text_processing: Optional[TextProcessingConfig] = field(default_factory=lambda: TextProcessingConfig(
        punctuation_marks = ".,?",
        separate_punctuation = False,
        do_lowercase = False,
        rm_punctuation = False,
    ))
# fmt: on


def apply_text_processing(
    punctuation_capitalization: PunctuationCapitalization, cfg: EvalBeamSearchNGramConfig, text: List[str] | str
) -> List[str] | str:
    is_list = isinstance(text, list)
    text_arr = text if is_list else [text]
    if cfg.text_processing.do_lowercase:
        text_arr = punctuation_capitalization.do_lowercase(text_arr)
    if cfg.text_processing.rm_punctuation:
        text_arr = punctuation_capitalization.rm_punctuation(text_arr)
    if cfg.text_processing.separate_punctuation:
        text_arr = punctuation_capitalization.separate_punctuation(text_arr)

    return text_arr if is_list else text_arr[0]


def beam_search_eval(
    audio_filepaths,
    model: nemo_asr.models.ASRModel,
    cfg: EvalBeamSearchNGramConfig,
    target_transcripts: List[str],
    preds_output_file: str = None,
    lm_path: str = None,
    beam_alpha: float = 1.0,
    beam_beta: float = 0.0,
    beam_width: int = 128,
    punctuation_capitalization: PunctuationCapitalization = None,
):
    level = logging.getEffectiveLevel()
    logging.setLevel(logging.CRITICAL)
    # Reset config
    if isinstance(model, EncDecHybridRNNTCTCModel):
        model.change_decoding_strategy(decoding_cfg=None, decoder_type="ctc")
    else:
        model.change_decoding_strategy(None)

    # Override the beam search config with current search candidate configuration
    cfg.decoding.beam_size = beam_width
    cfg.decoding.ngram_lm_alpha = beam_alpha
    cfg.decoding.beam_beta = beam_beta
    cfg.decoding.return_best_hypothesis = False
    cfg.decoding.ngram_lm_model = cfg.kenlm_model_file

    # Update model's decoding strategy config
    model.cfg.decoding.strategy = cfg.decoding_strategy
    model.cfg.decoding.beam = cfg.decoding

    # Update model's decoding strategy
    if isinstance(model, EncDecHybridRNNTCTCModel):
        model.change_decoding_strategy(model.cfg.decoding, decoder_type='ctc')
    else:
        model.change_decoding_strategy(model.cfg.decoding)
    logging.setLevel(level)

    all_hyps = model.transcribe(audio_filepaths, cfg.batch_size)

    wer_dist_first = cer_dist_first = 0
    wer_dist_best = cer_dist_best = 0
    words_count = 0
    chars_count = 0
    if preds_output_file:
        out_file = open(preds_output_file, 'w', encoding='utf_8', newline='\n')

    for batch_idx, nbest_hyp in enumerate(all_hyps):
        target = target_transcripts[batch_idx]
        target_split_w = target.split()
        target_split_c = list(target)
        words_count += len(target_split_w)
        chars_count += len(target_split_c)
        wer_dist_min = cer_dist_min = float("inf")
        for candidate_idx, candidate in enumerate(nbest_hyp):
            pred_text = apply_text_processing(punctuation_capitalization, cfg, candidate.text)

            pred_split_w = pred_text.split()
            wer_dist = editdistance.eval(target_split_w, pred_split_w)
            pred_split_c = list(pred_text)
            cer_dist = editdistance.eval(target_split_c, pred_split_c)

            wer_dist_min = min(wer_dist_min, wer_dist)
            cer_dist_min = min(cer_dist_min, cer_dist)

            if candidate_idx == 0:
                # first candidate
                wer_dist_first += wer_dist
                cer_dist_first += cer_dist

            score = candidate.score
            if preds_output_file:
                out_file.write('{}\t{}\n'.format(pred_text, score))
        wer_dist_best += wer_dist_min
        cer_dist_best += cer_dist_min

    if preds_output_file:
        out_file.close()
        logging.info(f"Stored the predictions of beam search decoding at '{preds_output_file}'.")

    if lm_path:
        logging.info(
            'WER/CER with beam search decoding and N-gram model = {:.2%}/{:.2%}'.format(
                wer_dist_first / words_count, cer_dist_first / chars_count
            )
        )
    else:
        logging.info(
            'WER/CER with beam search decoding = {:.2%}/{:.2%}'.format(
                wer_dist_first / words_count, cer_dist_first / chars_count
            )
        )
    logging.info(
        'Oracle WER/CER in candidates with perfect LM= {:.2%}/{:.2%}'.format(
            wer_dist_best / words_count, cer_dist_best / chars_count
        )
    )

    logging.info(f"=================================================================================")

    return wer_dist_first / words_count, cer_dist_first / chars_count


@hydra_runner(config_path=None, config_name='EvalBeamSearchNGramConfig', schema=EvalBeamSearchNGramConfig)
def main(cfg: EvalBeamSearchNGramConfig):
    if is_dataclass(cfg):
        cfg = OmegaConf.structured(cfg)  # type: EvalBeamSearchNGramConfig

    valid_decoding_modes = ["greedy", "beamsearch", "beamsearch_ngram"]
    if cfg.decoding_mode not in valid_decoding_modes:
        raise ValueError(
            f"Given decoding_mode={cfg.decoding_mode} is invalid. Available options are :\n" f"{valid_decoding_modes}"
        )

    if cfg.nemo_model_file.endswith('.nemo'):
        asr_model = nemo_asr.models.ASRModel.restore_from(cfg.nemo_model_file, map_location=torch.device(cfg.device))
    else:
        logging.warning(
            "nemo_model_file does not end with .nemo, therefore trying to load a pretrained model with this name."
        )
        asr_model = nemo_asr.models.ASRModel.from_pretrained(
            cfg.nemo_model_file, map_location=torch.device(cfg.device)
        )

    target_transcripts = []
    manifest_dir = Path(cfg.input_manifest).parent
    with open(cfg.input_manifest, 'r', encoding='utf_8') as manifest_file:
        audio_file_paths = []
        for line in tqdm(manifest_file, desc=f"Reading Manifest {cfg.input_manifest} ...", ncols=120):
            data = json.loads(line)
            audio_file = Path(data['audio_filepath'])
            if not audio_file.is_file() and not audio_file.is_absolute():
                audio_file = manifest_dir / audio_file
            target_transcripts.append(data['text'])
            audio_file_paths.append(str(audio_file.absolute()))

    punctuation_capitalization = PunctuationCapitalization(cfg.text_processing.punctuation_marks)
    target_transcripts = apply_text_processing(punctuation_capitalization, cfg, target_transcripts)

    if cfg.hyps_cache_file and os.path.exists(cfg.hyps_cache_file):
        logging.info(f"Found a pickle file of hypotheses at '{cfg.hyps_cache_file}'.")
        logging.info(f"Loading the cached pickle file of hypotheses from '{cfg.hyps_cache_file}' ...")
        with open(cfg.hyps_cache_file, 'rb') as probs_file:
            all_hyps = pickle.load(probs_file)

        if len(all_hyps) != len(audio_file_paths):
            raise ValueError(
                f"The number of samples in the hypotheses file '{cfg.hyps_cache_file}' does not "
                f"match the manifest file. You may need to delete the hypotheses cached file."
            )
    else:

        with torch.amp.autocast(asr_model.device.type, enabled=cfg.use_amp):
            with torch.no_grad():
                if isinstance(asr_model, EncDecHybridRNNTCTCModel):
                    asr_model.cur_decoder = 'ctc'
                all_hyps = asr_model.transcribe(audio_file_paths, batch_size=cfg.batch_size)

        if cfg.hyps_cache_file:
            os.makedirs(os.path.split(cfg.hyps_cache_file)[0], exist_ok=True)
            logging.info(f"Writing pickle files of hypotheses at '{cfg.hyps_cache_file}'...")
            with open(cfg.hyps_cache_file, 'wb') as f_dump:
                pickle.dump(all_hyps, f_dump)

    wer_dist_greedy = 0
    cer_dist_greedy = 0
    words_count = 0
    chars_count = 0
    for batch_idx, hyp in enumerate(all_hyps):
        pred_text = apply_text_processing(punctuation_capitalization, cfg, hyp.text)

        pred_split_w = pred_text.split()
        target_split_w = target_transcripts[batch_idx].split()
        pred_split_c = list(pred_text)
        target_split_c = list(target_transcripts[batch_idx])

        wer_dist = editdistance.eval(target_split_w, pred_split_w)
        cer_dist = editdistance.eval(target_split_c, pred_split_c)

        wer_dist_greedy += wer_dist
        cer_dist_greedy += cer_dist
        words_count += len(target_split_w)
        chars_count += len(target_split_c)

    logging.info('Greedy WER/CER = {:.2%}/{:.2%}'.format(wer_dist_greedy / words_count, cer_dist_greedy / chars_count))

    asr_model = asr_model.to('cpu')

    if cfg.decoding_mode == "beamsearch_ngram":
        if not os.path.exists(cfg.kenlm_model_file):
            raise FileNotFoundError(f"Could not find the KenLM model file '{cfg.kenlm_model_file}'.")
        lm_path = cfg.kenlm_model_file
    else:
        lm_path = None

    # 'greedy' decoding_mode would skip the beam search decoding
    if cfg.decoding_mode in ["beamsearch_ngram", "beamsearch"]:
        if cfg.beam_width is None or cfg.beam_alpha is None or cfg.beam_beta is None:
            raise ValueError("beam_width, beam_alpha and beam_beta are needed to perform beam search decoding.")
        params = {'beam_width': cfg.beam_width, 'beam_alpha': cfg.beam_alpha, 'beam_beta': cfg.beam_beta}
        hp_grid = ParameterGrid(params)
        hp_grid = list(hp_grid)

        best_wer_beam_size, best_cer_beam_size = None, None
        best_wer_alpha, best_cer_alpha = None, None
        best_wer_beta, best_cer_beta = None, None
        best_wer, best_cer = float("inf"), float("inf")

        logging.info(f"==============================Starting the beam search decoding===============================")
        logging.info(f"Grid search size: {len(hp_grid)}")
        logging.info(f"It may take some time...")
        logging.info(f"==============================================================================================")

        if cfg.preds_output_folder and not os.path.exists(cfg.preds_output_folder):
            os.mkdir(cfg.preds_output_folder)
        for hp in hp_grid:
            if cfg.preds_output_folder:
                preds_output_file = os.path.join(
                    cfg.preds_output_folder,
                    f"preds_out_width{hp['beam_width']}_alpha{hp['beam_alpha']}_beta{hp['beam_beta']}.tsv",
                )
            else:
                preds_output_file = None

            candidate_wer, candidate_cer = beam_search_eval(
                audio_file_paths,
                asr_model,
                cfg,
                target_transcripts=target_transcripts,
                preds_output_file=preds_output_file,
                lm_path=lm_path,
                beam_width=hp["beam_width"],
                beam_alpha=hp["beam_alpha"],
                beam_beta=hp["beam_beta"],
                punctuation_capitalization=punctuation_capitalization,
            )

            if candidate_cer < best_cer:
                best_cer_beam_size, best_cer_alpha, best_cer_beta, best_cer = (
                    hp["beam_width"],
                    hp["beam_alpha"],
                    hp["beam_beta"],
                    candidate_cer,
                )

            if candidate_wer < best_wer:
                best_wer_beam_size, best_wer_alpha, best_wer_beta, best_wer = (
                    hp["beam_width"],
                    hp["beam_alpha"],
                    hp["beam_beta"],
                    candidate_wer,
                )

        logging.info(
            f'Best WER Candidate = {best_wer:.2%} :: Beam size = {best_wer_beam_size}, '
            f'Beam alpha = {best_wer_alpha}, Beam beta = {best_wer_beta}'
        )

        logging.info(
            f'Best CER Candidate = {best_cer:.2%} :: Beam size = {best_cer_beam_size}, '
            f'Beam alpha = {best_cer_alpha}, Beam beta = {best_cer_beta}'
        )
        logging.info(f"=================================================================================")


if __name__ == '__main__':
    main()
