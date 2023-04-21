# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
# fusion with beam search decoders on top of a trained ASR Transducer model. NeMo's beam search decoders are capable of using the
# KenLM's N-gram models to find the best candidates. This script supports both character level and BPE level
# encodings and models which is detected automatically from the type of the model.
# You may train the LM model with 'scripts/ngram_lm/train_kenlm.py'.

# Config Help

To discover all arguments of the script, please run :
python eval_beamsearch_ngram.py --help
python eval_beamsearch_ngram.py --cfg job

# USAGE

python eval_beamsearch_ngram_transducer.py nemo_model_file=<path to the .nemo file of the model> \
           input_manifest=<path to the evaluation JSON manifest file \
           kenlm_model_file=<path to the binary KenLM model> \
           beam_width=[<list of the beam widths, separated with commas>] \
           beam_alpha=[<list of the beam alphas, separated with commas>] \
           preds_output_folder=<optional folder to store the predictions> \
           probs_cache_file=null \
           decoding_strategy=<greedy_batch or maes decoding>
           maes_prefix_alpha=[<list of the maes prefix alphas, separated with commas>] \
           maes_expansion_gamma=[<list of the maes expansion gammas, separated with commas>] \
           hat_subtract_ilm=<in case of HAT model: subtract internal LM or not> \
           hat_ilm_weight=[<in case of HAT model: list of the HAT internal LM weights, separated with commas>] \
           ...


# Grid Search for Hyper parameters

For grid search, you can provide a list of arguments as follows -

           beam_width=[4,8,16,....] \
           beam_alpha=[-2.0,-1.0,...,1.0,2.0] \

# You may find more info on how to use this script at:
# https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/asr_language_modeling.html

"""


import contextlib
import json
import os
import pickle
import tempfile
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
from nemo.collections.asr.parts.submodules import rnnt_beam_decoding
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
    probs_cache_file: Optional[str] = None  # The cache file for storing the logprobs of the model

    # Parameters for inference
    acoustic_batch_size: int = 128  # The batch size to calculate log probabilities
    beam_batch_size: int = 128  # The batch size to be used for beam search decoding
    device: str = "cuda"  # The device to load the model onto to calculate log probabilities
    use_amp: bool = False  # Whether to use AMP if available to calculate log probabilities
    num_workers: int = 1  # Number of workers for DataLoader

    # The decoding scheme to be used for evaluation
    decoding_strategy: str = "greedy_batch" # ["greedy_batch", "beam", "tsd", "alsd", "maes"]

    # Beam Search hyperparameters
    beam_width: List[int] = field(default_factory=lambda: [8])  # The width or list of the widths for the beam search decoding
    beam_alpha: List[float] = field(default_factory=lambda: [0.2])  # The alpha parameter or list of the alphas for the beam search decoding

    maes_prefix_alpha: List[int] = field(default_factory=lambda: [2])  # The maes_prefix_alpha or list of the maes_prefix_alpha for the maes decoding
    maes_expansion_gamma: List[float] = field(default_factory=lambda: [2.3])  # The maes_expansion_gamma or list of the maes_expansion_gamma for the maes decoding

    # HAT related parameters (only for internal lm subtraction)
    hat_subtract_ilm: bool = False
    hat_ilm_weight: List[float] = field(default_factory=lambda: [0.0])

    decoding: rnnt_beam_decoding.BeamRNNTInferConfig = rnnt_beam_decoding.BeamRNNTInferConfig(beam_size=128)


# fmt: on


def decoding_step(
    model: nemo_asr.models.ASRModel,
    cfg: EvalBeamSearchNGramConfig,
    all_probs: List[torch.Tensor],
    target_transcripts: List[str],
    preds_output_file: str = None,
    beam_batch_size: int = 128,
    progress_bar: bool = True,
):
    level = logging.getEffectiveLevel()
    logging.setLevel(logging.CRITICAL)
    # Reset config
    model.change_decoding_strategy(None)

    cfg.decoding.hat_ilm_weight = cfg.decoding.hat_ilm_weight * cfg.hat_subtract_ilm
    # Override the beam search config with current search candidate configuration
    cfg.decoding.return_best_hypothesis = False
    cfg.decoding.ngram_lm_model = cfg.kenlm_model_file
    cfg.decoding.hat_subtract_ilm = cfg.hat_subtract_ilm

    # Update model's decoding strategy config
    model.cfg.decoding.strategy = cfg.decoding_strategy
    model.cfg.decoding.beam = cfg.decoding

    # Update model's decoding strategy
    model.change_decoding_strategy(model.cfg.decoding)
    logging.setLevel(level)

    wer_dist_first = cer_dist_first = 0
    wer_dist_best = cer_dist_best = 0
    words_count = 0
    chars_count = 0
    sample_idx = 0
    if preds_output_file:
        out_file = open(preds_output_file, 'w', encoding='utf_8', newline='\n')

    if progress_bar:
        if cfg.decoding_strategy == "greedy_batch":
            description = "Greedy_batch decoding.."
        else:
            description = f"{cfg.decoding_strategy} decoding with bw={cfg.decoding.beam_size}, ba={cfg.decoding.ngram_lm_alpha}, ma={cfg.decoding.maes_prefix_alpha}, mg={cfg.decoding.maes_expansion_gamma}, hat_ilmw={cfg.decoding.hat_ilm_weight}"
        it = tqdm(range(int(np.ceil(len(all_probs) / beam_batch_size))), desc=description, ncols=120)
    else:
        it = range(int(np.ceil(len(all_probs) / beam_batch_size)))
    for batch_idx in it:
        # disabling type checking
        probs_batch = all_probs[batch_idx * beam_batch_size : (batch_idx + 1) * beam_batch_size]
        probs_lens = torch.tensor([prob.shape[-1] for prob in probs_batch])
        with torch.no_grad():
            packed_batch = torch.zeros(len(probs_batch), probs_batch[0].shape[0], max(probs_lens), device='cpu')

            for prob_index in range(len(probs_batch)):
                packed_batch[prob_index, :, : probs_lens[prob_index]] = torch.tensor(
                    probs_batch[prob_index].unsqueeze(0), device=packed_batch.device, dtype=packed_batch.dtype
                )
            best_hyp_batch, beams_batch = model.decoding.rnnt_decoder_predictions_tensor(
                packed_batch, probs_lens, return_hypotheses=True,
            )
        if cfg.decoding_strategy == "greedy_batch":
            beams_batch = [[x] for x in best_hyp_batch]

        for beams_idx, beams in enumerate(beams_batch):
            target = target_transcripts[sample_idx + beams_idx]
            target_split_w = target.split()
            target_split_c = list(target)
            words_count += len(target_split_w)
            chars_count += len(target_split_c)
            wer_dist_min = cer_dist_min = 10000
            for candidate_idx, candidate in enumerate(beams):  # type: (int, rnnt_beam_decoding.rnnt_utils.Hypothesis)
                pred_text = candidate.text
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
        sample_idx += len(probs_batch)

    if cfg.decoding_strategy == "greedy_batch":
        return wer_dist_first / words_count, cer_dist_first / chars_count

    if preds_output_file:
        out_file.close()
        logging.info(f"Stored the predictions of {cfg.decoding_strategy} decoding at '{preds_output_file}'.")

    if cfg.decoding.ngram_lm_model:
        logging.info(
            f"WER/CER with {cfg.decoding_strategy} decoding and N-gram model = {wer_dist_first / words_count:.2%}/{cer_dist_first / chars_count:.2%}"
        )
    else:
        logging.info(
            f"WER/CER with {cfg.decoding_strategy} decoding = {wer_dist_first / words_count:.2%}/{cer_dist_first / chars_count:.2%}"
        )
    logging.info(
        f"Oracle WER/CER in candidates with perfect LM= {wer_dist_best / words_count:.2%}/{cer_dist_best / chars_count:.2%}"
    )
    logging.info(f"=================================================================================")

    return wer_dist_first / words_count, cer_dist_first / chars_count


@hydra_runner(config_path=None, config_name='EvalBeamSearchNGramConfig', schema=EvalBeamSearchNGramConfig)
def main(cfg: EvalBeamSearchNGramConfig):
    if is_dataclass(cfg):
        cfg = OmegaConf.structured(cfg)  # type: EvalBeamSearchNGramConfig

    valid_decoding_strategis = ["greedy_batch", "beam", "tsd", "alsd", "maes"]
    if cfg.decoding_strategy not in valid_decoding_strategis:
        raise ValueError(
            f"Given decoding_strategy={cfg.decoding_strategy} is invalid. Available options are :\n"
            f"{valid_decoding_strategis}"
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

    if cfg.kenlm_model_file:
        if not os.path.exists(cfg.kenlm_model_file):
            raise FileNotFoundError(f"Could not find the KenLM model file '{cfg.kenlm_model_file}'.")
        if cfg.decoding_strategy != "maes":
            raise ValueError(f"Decoding with kenlm model is supported only for maes decoding algorithm.")
        lm_path = cfg.kenlm_model_file
    else:
        lm_path = None
        cfg.beam_alpha = [0.0]
    if cfg.hat_subtract_ilm:
        assert lm_path, "kenlm must be set for hat internal lm subtraction"

    if cfg.decoding_strategy != "maes":
        cfg.maes_prefix_alpha, cfg.maes_expansion_gamma, cfg.hat_ilm_weight = [0], [0], [0]

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

        @contextlib.contextmanager
        def default_autocast():
            yield

        if cfg.use_amp:
            if torch.cuda.is_available() and hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'autocast'):
                logging.info("AMP is enabled!\n")
                autocast = torch.cuda.amp.autocast

            else:
                autocast = default_autocast
        else:

            autocast = default_autocast

        # manual calculation of encoder_embeddings
        with autocast():
            with torch.no_grad():
                asr_model.eval()
                asr_model.encoder.freeze()
                device = next(asr_model.parameters()).device
                all_probs = []
                with tempfile.TemporaryDirectory() as tmpdir:
                    with open(os.path.join(tmpdir, 'manifest.json'), 'w', encoding='utf-8') as fp:
                        for audio_file in audio_file_paths:
                            entry = {'audio_filepath': audio_file, 'duration': 100000, 'text': ''}
                            fp.write(json.dumps(entry) + '\n')
                    config = {
                        'paths2audio_files': audio_file_paths,
                        'batch_size': cfg.acoustic_batch_size,
                        'temp_dir': tmpdir,
                        'num_workers': cfg.num_workers,
                        'channel_selector': None,
                        'augmentor': None,
                    }
                    temporary_datalayer = asr_model._setup_transcribe_dataloader(config)
                    for test_batch in tqdm(temporary_datalayer, desc="Transcribing", disable=True):
                        encoded, encoded_len = asr_model.forward(
                            input_signal=test_batch[0].to(device), input_signal_length=test_batch[1].to(device)
                        )
                        # dump encoder embeddings per file
                        for idx in range(encoded.shape[0]):
                            encoded_no_pad = encoded[idx, :, : encoded_len[idx]]
                            all_probs.append(encoded_no_pad)

        if cfg.probs_cache_file:
            logging.info(f"Writing pickle files of probabilities at '{cfg.probs_cache_file}'...")
            with open(cfg.probs_cache_file, 'wb') as f_dump:
                pickle.dump(all_probs, f_dump)

    if cfg.decoding_strategy == "greedy_batch":
        asr_model = asr_model.to('cpu')
        candidate_wer, candidate_cer = decoding_step(
            asr_model,
            cfg,
            all_probs=all_probs,
            target_transcripts=target_transcripts,
            beam_batch_size=cfg.beam_batch_size,
            progress_bar=True,
        )
        logging.info(f"Greedy batch WER/CER = {candidate_wer:.2%}/{candidate_cer:.2%}")

    asr_model = asr_model.to('cpu')

    # 'greedy_batch' decoding_strategy would skip the beam search decoding
    if cfg.decoding_strategy in ["beam", "tsd", "alsd", "maes"]:
        if cfg.beam_width is None or cfg.beam_alpha is None:
            raise ValueError("beam_width and beam_alpha are needed to perform beam search decoding.")
        params = {
            'beam_width': cfg.beam_width,
            'beam_alpha': cfg.beam_alpha,
            'maes_prefix_alpha': cfg.maes_prefix_alpha,
            'maes_expansion_gamma': cfg.maes_expansion_gamma,
            'hat_ilm_weight': cfg.hat_ilm_weight,
        }
        hp_grid = ParameterGrid(params)
        hp_grid = list(hp_grid)

        best_wer_beam_size, best_cer_beam_size = None, None
        best_wer_alpha, best_cer_alpha = None, None
        best_wer, best_cer = 1e6, 1e6

        logging.info(
            f"==============================Starting the {cfg.decoding_strategy} decoding==============================="
        )
        logging.info(f"Grid search size: {len(hp_grid)}")
        logging.info(f"It may take some time...")
        logging.info(f"==============================================================================================")

        if cfg.preds_output_folder and not os.path.exists(cfg.preds_output_folder):
            os.mkdir(cfg.preds_output_folder)
        for hp in hp_grid:
            if cfg.preds_output_folder:
                results_file = f"preds_out_{cfg.decoding_strategy}_bw{hp['beam_width']}"
                if cfg.decoding_strategy == "maes":
                    results_file = f"{results_file}_ma{hp['maes_prefix_alpha']}_mg{hp['maes_expansion_gamma']}"
                    if cfg.kenlm_model_file:
                        results_file = f"{results_file}_ba{hp['beam_alpha']}"
                        if cfg.hat_subtract_ilm:
                            results_file = f"{results_file}_hat_ilmw{hp['hat_ilm_weight']}"
                preds_output_file = os.path.join(cfg.preds_output_folder, f"{results_file}.tsv")
            else:
                preds_output_file = None

            cfg.decoding.beam_size = hp["beam_width"]
            cfg.decoding.ngram_lm_alpha = hp["beam_alpha"]
            cfg.decoding.maes_prefix_alpha = hp["maes_prefix_alpha"]
            cfg.decoding.maes_expansion_gamma = hp["maes_expansion_gamma"]
            cfg.decoding.hat_ilm_weight = hp["hat_ilm_weight"]

            candidate_wer, candidate_cer = decoding_step(
                asr_model,
                cfg,
                all_probs=all_probs,
                target_transcripts=target_transcripts,
                preds_output_file=preds_output_file,
                beam_batch_size=cfg.beam_batch_size,
                progress_bar=True,
            )

            if candidate_cer < best_cer:
                best_cer_beam_size = hp["beam_width"]
                best_cer_alpha = hp["beam_alpha"]
                best_cer_ma = hp["maes_prefix_alpha"]
                best_cer_mg = hp["maes_expansion_gamma"]
                best_cer_hat_ilm_weight = hp["hat_ilm_weight"]
                best_cer = candidate_cer

            if candidate_wer < best_wer:
                best_wer_beam_size = hp["beam_width"]
                best_wer_alpha = hp["beam_alpha"]
                best_wer_ma = hp["maes_prefix_alpha"]
                best_wer_ga = hp["maes_expansion_gamma"]
                best_wer_hat_ilm_weight = hp["hat_ilm_weight"]
                best_wer = candidate_wer

        wer_hat_parameter = ""
        if cfg.hat_subtract_ilm:
            wer_hat_parameter = f"HAT ilm weight = {best_wer_hat_ilm_weight}, "
        logging.info(
            f'Best WER Candidate = {best_wer:.2%} :: Beam size = {best_wer_beam_size}, '
            f'Beam alpha = {best_wer_alpha}, {wer_hat_parameter}'
            f'maes_prefix_alpha = {best_wer_ma}, maes_expansion_gamma = {best_wer_ga} '
        )

        cer_hat_parameter = ""
        if cfg.hat_subtract_ilm:
            cer_hat_parameter = f"HAT ilm weight = {best_cer_hat_ilm_weight}"
        logging.info(
            f'Best CER Candidate = {best_cer:.2%} :: Beam size = {best_cer_beam_size}, '
            f'Beam alpha = {best_cer_alpha}, {cer_hat_parameter} '
            f'maes_prefix_alpha = {best_cer_ma}, maes_expansion_gamma = {best_cer_mg}'
        )
        logging.info(f"=================================================================================")


if __name__ == '__main__':
    main()
