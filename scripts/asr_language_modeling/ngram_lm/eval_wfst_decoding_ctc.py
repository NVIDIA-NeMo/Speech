# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
# This script would evaluate an N-gram language model in ARPA format in
# fusion with WFST decoders on top of a trained ASR model with CTC decoder.
# NeMo's WFST decoders use WFST decoding graphs made from ARPA LMs
# to find the best candidates. This script supports BPE level encodings only
# and models which is detected automatically from the type of the model.
# You may train the LM model with e.g. SRILM.

# Config Help

To discover all arguments of the script, please run :
python eval_wfst_decoding_ctc.py --help
python eval_wfst_decoding_ctc.py --cfg job

# USAGE

python eval_wfst_decoding_ctc.py nemo_model_file=<path to the .nemo file of the model> \
           input_manifest=<path to the evaluation JSON manifest file> \
           arpa_model_file=<path to the ARPA LM model> \
           decoding_wfst_file=<path to the decoding WFST file> \
           beam_width=[<list of the beam widths, separated with commas>] \
           lm_weight=[<list of the LM weight multipliers, separated with commas>] \
           decoding_mode=<decoding mode, affects output. Usually "nbest"> \
           decoding_search_type=<a.k.a. decoding backend. Usually "riva"> \
           open_vocabulary_decoding=<whether to use open vocabulary mode for WFST decoding> \
           preds_output_folder=<optional folder to store the predictions> \
           probs_cache_file=null
           ...


# Grid Search for Hyper parameters

For grid search, you can provide a list of arguments as follows -

           beam_width=[5.0,10.0,15.0,20.0] \
           lm_weight=[0.1,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,2.0] \

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
class EvalWFSTNGramConfig:
    """
    Evaluate an ASR model with WFST decoding and n-gram ARPA language model.
    """
    # # The path of the '.nemo' file of the ASR model or the name of a pretrained model (ngc / huggingface)
    nemo_model_file: str = MISSING

    # File paths
    input_manifest: str = MISSING  # The manifest file of the evaluation set
    arpa_model_file: Optional[str] = None  # The path of the ARPA model file
    decoding_wfst_file: Optional[str] = None  # The path of the decoding WFST file
    preds_output_folder: Optional[str] = None  # The optional folder where the predictions are stored
    probs_cache_file: Optional[str] = None  # The cache file for storing the logprobs of the model

    # Parameters for inference
    acoustic_batch_size: int = 16  # The batch size to calculate log probabilities
    beam_batch_size: int = 512  # The batch size to be used for beam search decoding
    device: str = "cuda"  # The device to load the model onto to calculate log probabilities and run WFST decoding
    use_amp: bool = False  # Whether to use AMP if available to calculate log probabilities

    # WFST decoding hyperparameters

    beam_width: List[float] = field(default_factory=lambda: [10])  # The width or list of the beam widths for the WFST decoding
    lm_weight: List[float] = field(default_factory=lambda: [1.0])  # The language model weight parameter or list of parameters for the WFST decoding

    open_vocabulary_decoding: bool = False  # Whether to use open vocabulary mode for WFST decoding
    decoding_mode: str = "nbest"
    decoding_search_type: str = "riva"
    decoding: ctc_beam_decoding.WfstCTCInferConfig = field(
        default_factory=lambda: ctc_beam_decoding.WfstCTCInferConfig(beam_size=1)
    )
    
    text_processing: Optional[TextProcessingConfig] = field(default_factory=lambda: TextProcessingConfig(
        punctuation_marks = ".,?",
        separate_punctuation = False,
        do_lowercase = False,
        rm_punctuation = False,
    ))
# fmt: on


def beam_search_eval(
    model: nemo_asr.models.ASRModel,
    cfg: EvalWFSTNGramConfig,
    all_probs: List[torch.Tensor],
    target_transcripts: List[str],
    preds_output_file: str = None,
    lm_weight: float = 1.0,
    beam_width: float = 10.0,
    beam_batch_size: int = 512,
    progress_bar: bool = True,
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
    cfg.decoding.beam_width = beam_width
    cfg.decoding.lm_weight = lm_weight
    cfg.decoding.open_vocabulary_decoding = cfg.open_vocabulary_decoding
    cfg.decoding.return_best_hypothesis = False
    cfg.decoding.arpa_lm_path = cfg.arpa_model_file
    cfg.decoding.wfst_lm_path = cfg.decoding_wfst_file
    cfg.decoding.device = cfg.device
    cfg.decoding.decoding_mode = cfg.decoding_mode
    cfg.decoding.search_type = cfg.decoding_search_type

    # Update model's decoding strategy config
    model.cfg.decoding.strategy = "wfst"
    model.cfg.decoding.wfst = cfg.decoding

    # Update model's decoding strategy
    if isinstance(model, EncDecHybridRNNTCTCModel):
        model.change_decoding_strategy(model.cfg.decoding, decoder_type='ctc')
        decoding = model.ctc_decoding
    else:
        model.change_decoding_strategy(model.cfg.decoding)
        decoding = model.decoding
    logging.setLevel(level)

    wer_dist_first = cer_dist_first = 0
    wer_dist_best = cer_dist_best = 0
    words_count = 0
    chars_count = 0
    sample_idx = 0
    if preds_output_file:
        out_file = open(preds_output_file, 'w', encoding='utf_8', newline='\n')

    if progress_bar:
        it = tqdm(
            range(int(np.ceil(len(all_probs) / beam_batch_size))),
            desc=f"Beam search decoding with width={beam_width}, lm_weight={lm_weight}",
            ncols=120,
        )
    else:
        it = range(int(np.ceil(len(all_probs) / beam_batch_size)))
    for batch_idx in it:
        # disabling type checking
        probs_batch = all_probs[batch_idx * beam_batch_size : (batch_idx + 1) * beam_batch_size]
        probs_lens = torch.tensor([prob.shape[0] for prob in probs_batch])
        with torch.no_grad():
            packed_batch = torch.zeros(len(probs_batch), max(probs_lens), probs_batch[0].shape[-1], device='cpu')

            for prob_index in range(len(probs_batch)):
                packed_batch[prob_index, : probs_lens[prob_index], :] = probs_batch[prob_index].to(
                    device=packed_batch.device, dtype=packed_batch.dtype
                )

            _, beams_batch = decoding.ctc_decoder_predictions_tensor(
                packed_batch,
                decoder_lengths=probs_lens,
                return_hypotheses=True,
            )

        for beams_idx, beams in enumerate(beams_batch):
            target = target_transcripts[sample_idx + beams_idx]
            target_split_w = target.split()
            target_split_c = list(target)
            words_count += len(target_split_w)
            chars_count += len(target_split_c)
            wer_dist_min = cer_dist_min = 10000
            for candidate_idx, candidate in enumerate(beams):  # type: (int, ctc_beam_decoding.rnnt_utils.Hypothesis)
                pred_text = candidate.text
                if cfg.text_processing.do_lowercase:
                    pred_text = punctuation_capitalization.do_lowercase([pred_text])[0]
                if cfg.text_processing.rm_punctuation:
                    pred_text = punctuation_capitalization.rm_punctuation([pred_text])[0]
                if cfg.text_processing.separate_punctuation:
                    pred_text = punctuation_capitalization.separate_punctuation([pred_text])[0]
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
                    out_file.write(f'{pred_text}\t{score}\n')
            wer_dist_best += wer_dist_min
            cer_dist_best += cer_dist_min
        sample_idx += len(probs_batch)

    if preds_output_file:
        out_file.close()
        logging.info(f"Stored the predictions of beam search decoding at '{preds_output_file}'.")

    logging.info(
        'WER/CER with beam search decoding and N-gram model = {:.2%}/{:.2%}'.format(
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


@hydra_runner(config_path=None, config_name='EvalWFSTNGramConfig', schema=EvalWFSTNGramConfig)
def main(cfg: EvalWFSTNGramConfig):
    if is_dataclass(cfg):
        cfg = OmegaConf.structured(cfg)  # type: EvalWFSTNGramConfig

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
    if cfg.text_processing.do_lowercase:
        target_transcripts = punctuation_capitalization.do_lowercase(target_transcripts)
    if cfg.text_processing.rm_punctuation:
        target_transcripts = punctuation_capitalization.rm_punctuation(target_transcripts)
    if cfg.text_processing.separate_punctuation:
        target_transcripts = punctuation_capitalization.separate_punctuation(target_transcripts)

    if cfg.probs_cache_file and os.path.exists(cfg.probs_cache_file):
        logging.info(f"Found a pickle file of probabilities at '{cfg.probs_cache_file}'.")
        logging.info(f"Loading the cached pickle file of probabilities from '{cfg.probs_cache_file}' ...")
        with open(cfg.probs_cache_file, 'rb') as probs_file:
            all_probs = pickle.load(probs_file)

        if len(all_probs) != len(audio_file_paths):
            raise ValueError(
                f"The number of samples in the probabilities file '{cfg.probs_cache_file}' does not "
                f"match the manifest file. You may need to delete the probabilities cached file."
            )
    else:

        with torch.amp.autocast(asr_model.device.type, enabled=cfg.use_amp):
            with torch.no_grad():
                if isinstance(asr_model, EncDecHybridRNNTCTCModel):
                    asr_model.cur_decoder = 'ctc'
                all_hyps = asr_model.transcribe(
                    audio_file_paths, batch_size=cfg.acoustic_batch_size, return_hypotheses=True
                )
                all_logits = [h.y_sequence for h in all_hyps]

        all_probs = all_logits
        if cfg.probs_cache_file:
            os.makedirs(os.path.split(cfg.probs_cache_file)[0], exist_ok=True)
            logging.info(f"Writing pickle files of probabilities at '{cfg.probs_cache_file}'...")
            with open(cfg.probs_cache_file, 'wb') as f_dump:
                pickle.dump(all_probs, f_dump)

    wer_dist_greedy = 0
    cer_dist_greedy = 0
    words_count = 0
    chars_count = 0
    for batch_idx, probs in enumerate(all_probs):
        preds = np.argmax(probs, axis=1)
        preds_tensor = preds.to(device='cpu').unsqueeze(0)
        preds_lens = torch.tensor([preds_tensor.shape[1]], device='cpu')
        if isinstance(asr_model, EncDecHybridRNNTCTCModel):
            pred_text = asr_model.ctc_decoding.ctc_decoder_predictions_tensor(preds_tensor, preds_lens)[0]
        else:
            pred_text = asr_model.decoding.ctc_decoder_predictions_tensor(preds_tensor, preds_lens)[0]

        if cfg.text_processing.do_lowercase:
            pred_text = punctuation_capitalization.do_lowercase([pred_text])[0]
        if cfg.text_processing.rm_punctuation:
            pred_text = punctuation_capitalization.rm_punctuation([pred_text])[0]
        if cfg.text_processing.separate_punctuation:
            pred_text = punctuation_capitalization.separate_punctuation([pred_text])[0]

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

    if (cfg.arpa_model_file is None or not os.path.exists(cfg.arpa_model_file)) and (
        cfg.decoding_wfst_file is None or not os.path.exists(cfg.decoding_wfst_file)
    ):
        raise FileNotFoundError(
            f"Could not find both the ARPA model file `{cfg.arpa_model_file}` "
            f"and the decoding WFST file `{cfg.decoding_wfst_file}`."
        )

    if cfg.beam_width is None or cfg.lm_weight is None:
        raise ValueError("beam_width and lm_weight are needed to perform WFST decoding.")
    params = {'beam_width': cfg.beam_width, 'lm_weight': cfg.lm_weight}
    hp_grid = ParameterGrid(params)
    hp_grid = list(hp_grid)

    best_wer_beam_width, best_cer_beam_width = None, None
    best_wer_lm_weight, best_cer_lm_weight = None, None
    best_wer, best_cer = 1e6, 1e6

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
                f"preds_out_beam_width{hp['beam_width']}_lm_weight{hp['lm_weight']}.tsv",
            )
        else:
            preds_output_file = None

        candidate_wer, candidate_cer = beam_search_eval(
            asr_model,
            cfg,
            all_probs=all_probs,
            target_transcripts=target_transcripts,
            preds_output_file=preds_output_file,
            beam_width=hp["beam_width"],
            lm_weight=hp["lm_weight"],
            beam_batch_size=cfg.beam_batch_size,
            progress_bar=True,
            punctuation_capitalization=punctuation_capitalization,
        )

        if candidate_cer < best_cer:
            best_cer_beam_width = hp["beam_width"]
            best_cer_lm_weight = hp["lm_weight"]
            best_cer = candidate_cer

        if candidate_wer < best_wer:
            best_wer_beam_width = hp["beam_width"]
            best_wer_lm_weight = hp["lm_weight"]
            best_wer = candidate_wer

    logging.info(
        f'Best WER Candidate = {best_wer:.2%} :: Beam size = {best_wer_beam_width}, LM weight = {best_wer_lm_weight}'
    )

    logging.info(
        f'Best CER Candidate = {best_cer:.2%} :: Beam size = {best_cer_beam_width}, LM weight = {best_cer_lm_weight}'
    )
    logging.info(f"=================================================================================")


if __name__ == '__main__':
    main()
