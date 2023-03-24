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

# This script would evaluate an N-gram language model trained with KenLM library (https://github.com/kpu/kenlm) in
# fusion with beam search decoders on top of a trained ASR model. NeMo's beam search decoders are capable of using the
# KenLM's N-gram models to find the best candidates. This script supports both character level and BPE level
# encodings and models which is detected automatically from the type of the model.
# You may train the LM model with 'scripts/ngram_lm/train_kenlm.py'.
#
# USAGE: python eval_beamsearch_ngram.py  --nemo_model_file=<path to the .nemo file of the model> \
#                                         --input_manifest=<path to the evaluation JSON manifest file \
#                                         --kenlm_model_file=<path to the binary KenLM model> \
#                                         --beam_width=<list of the beam widths> \
#                                         --beam_alpha=<list of the beam alphas> \
#                                         --preds_output_folder=<optional folder to store the predictions> \
#                                         --decoding_mode=maes
#                                         ...
#
# You may find more info on how to use this script at:
# https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/asr_language_modeling.html


import argparse
import contextlib
import json
import os
import pickle

import editdistance
import torch
from omegaconf import OmegaConf
from tqdm.auto import tqdm

import nemo.collections.asr as nemo_asr
from nemo.collections.asr.metrics.rnnt_wer_bpe import RNNTBPEDecodingConfig
from nemo.utils import logging


def beam_search_eval(all_hypotheses, target_transcripts, preds_output_file=None):
    wer_dist_first = cer_dist_first = 0
    wer_dist_best = cer_dist_best = 0
    words_count = 0
    chars_count = 0
    if preds_output_file:
        out_file = open(preds_output_file, 'w')

    it = tqdm(range(len(all_hypotheses)), desc=f"Beam search decoding...", ncols=120,)

    for sample_idx in it:
        hypotheses = all_hypotheses[sample_idx]
        target = target_transcripts[sample_idx]
        target_split_w = target.split()
        target_split_c = list(target)
        words_count += len(target_split_w)
        chars_count += len(target_split_c)
        wer_dist_min = cer_dist_min = 10000
        if not isinstance(hypotheses, list):
            hypotheses = [hypotheses]
        for candidate_idx, candidate in enumerate(hypotheses):
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

            if preds_output_file:
                out_file.write('{}\t{}\n'.format(pred_text, candidate.score))
        wer_dist_best += wer_dist_min
        cer_dist_best += cer_dist_min

    if preds_output_file:
        out_file.close()
        logging.info(f"Stored the predictions of beam search decoding at '{preds_output_file}'.")

    logging.info(
        'WER/CER with the provided decoding strategy = {:.2%}/{:.2%}'.format(
            wer_dist_first / words_count, cer_dist_first / chars_count
        )
    )

    logging.info(
        'Oracle WER/CER in candidates = {:.2%}/{:.2%}'.format(wer_dist_best / words_count, cer_dist_best / chars_count)
    )
    logging.info(f"=================================================================================")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate an ASR model with beam search decoding and n-gram KenLM language model.'
    )
    parser.add_argument(
        "--nemo_model_file", required=True, type=str, help="The path of the '.nemo' file of the ASR model"
    )
    parser.add_argument(
        "--kenlm_model_file", required=False, default=None, type=str, help="The path of the KenLM binary model file"
    )
    parser.add_argument("--input_manifest", required=True, type=str, help="The manifest file of the evaluation set")
    parser.add_argument(
        "--preds_output_folder", default=None, type=str, help="The optional folder where the predictions are stored"
    )
    parser.add_argument(
        "--probs_cache_file", default=None, type=str, help="The cache file for storing the outputs of the model"
    )
    parser.add_argument(
        "--acoustic_batch_size", default=16, type=int, help="The batch size to calculate log probabilities"
    )
    parser.add_argument(
        "--device", default="cuda", type=str, help="The device to load the model onto to calculate log probabilities"
    )
    parser.add_argument(
        "--use_amp", action="store_true", help="Whether to use AMP if available to calculate log probabilities"
    )
    parser.add_argument(
        "--decoding_mode",
        choices=["greedy", "greedy_batch", "beam", "tsd", "alsd", "maes"],
        default="beam",
        type=str,
        help="The decoding scheme to be used for evaluation.",
    )
    parser.add_argument(
        "--beam_width", required=True, type=int, help="The width for the beam search decoding",
    )
    parser.add_argument(
        "--beam_alpha", required=True, type=float, help="The alpha parameter for the beam search decoding",
    )
    parser.add_argument(
        "--beam_batch_size", default=128, type=int, help="The batch size to be used for beam search decoding"
    )
    parser.add_argument(
        "--maes_prefix_alpha",
        default=1,
        type=int,
        help="Float pruning threshold used in the prune-by-value step when computing the expansions.",
    )
    parser.add_argument(
        "--maes_expansion_gamma", default=2.3, type=float, help="Maximum prefix length in prefix search"
    )
    parser.add_argument(
        "--hat_subtract_ilm", action="store_true", help="Subtract internal LM from the final HAT logprobs"
    )
    parser.add_argument("--hat_ilm_weight", default=0.0, type=float, help="lamda2 weight for HAT ILM subsrtact")

    args = parser.parse_args()

    if args.kenlm_model_file and args.decoding_mode != "maes":
        raise ValueError("External n-gram LM fusion is available only for 'maes' decoding mode.")

    if args.nemo_model_file.endswith('.nemo'):
        asr_model = nemo_asr.models.ASRModel.restore_from(args.nemo_model_file, map_location=torch.device(args.device))
    else:
        logging.warning(
            "nemo_model_file does not end with .nemo, therefore trying to load a pretrained model with this name."
        )
        asr_model = nemo_asr.models.ASRModel.from_pretrained(
            args.nemo_model_file, map_location=torch.device(args.device)
        )

    target_transcripts = []
    with open(args.input_manifest, 'r') as manifest_file:
        audio_file_paths = []
        durations = []
        for line in tqdm(manifest_file, desc=f"Reading Manifest {args.input_manifest} ...", ncols=120):
            data = json.loads(line)
            target_transcripts.append(data['text'])
            audio_file_paths.append(data['audio_filepath'])
            durations.append(data['duration'])

    if args.probs_cache_file and os.path.exists(args.probs_cache_file):
        logging.info(f"Found a pickle file of probabilities at '{args.probs_cache_file}'.")
        logging.info(f"Loading the cached pickle file of probabilities from '{args.probs_cache_file}' ...")
        with open(args.probs_cache_file, 'rb') as probs_file:
            all_probs = pickle.load(probs_file)

        if len(all_probs) != len(audio_file_paths):
            raise ValueError(
                f"The number of samples in the probabilities file '{args.probs_cache_file}' does not "
                f"match the manifest file. You may need to delete the probabilities cached file."
            )
    else:
        asr_model = asr_model.eval()
        rnnt_cfg = RNNTBPEDecodingConfig()
        rnnt_cfg.strategy = args.decoding_mode  # beam greedy
        rnnt_cfg.beam.beam_size = args.beam_width
        rnnt_cfg.beam.ngram_lm_model = args.kenlm_model_file
        rnnt_cfg.beam.ngram_lm_alpha = args.beam_alpha  # 0.2, 0.3
        rnnt_cfg.compute_hypothesis_token_set = False
        rnnt_cfg.beam.return_best_hypothesis = False
        rnnt_cfg.beam.maes_prefix_alpha = args.maes_prefix_alpha
        rnnt_cfg.beam.maes_expansion_gamma = args.maes_expansion_gamma
        rnnt_cfg.beam.hat_subtract_ilm = args.hat_subtract_ilm
        rnnt_cfg.beam.hat_ilm_weight = args.hat_ilm_weight
        asr_model.change_decoding_strategy(OmegaConf.structured(rnnt_cfg))

        @contextlib.contextmanager
        def default_autocast():
            yield

        if args.use_amp:
            if torch.cuda.is_available() and hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'autocast'):
                logging.info("AMP is enabled!\n")
                autocast = torch.cuda.amp.autocast
            else:
                autocast = default_autocast
        else:
            autocast = default_autocast

        params = {'beam_width': args.beam_width, 'beam_alpha': args.beam_alpha}

        logging.info(f"==============================Starting the beam search decoding===============================")
        logging.info(f"Beam search params: {params}")
        logging.info(f"It may take some time...")
        logging.info(f"==============================================================================================")

        with autocast():
            with torch.no_grad():
                hypotheses, all_hypotheses = asr_model.transcribe(
                    audio_file_paths, batch_size=args.acoustic_batch_size, return_hypotheses=True
                )

    # delete the model to free the memory
    del asr_model

    if args.preds_output_folder and not os.path.exists(args.preds_output_folder):
        os.mkdir(args.preds_output_folder)

    if args.preds_output_folder:
        preds_output_file = os.path.join(
            args.preds_output_folder, f"preds_out_width{args.beam_width}_alpha{args.beam_alpha}.tsv",
        )
        preds_output_manifest = os.path.join(args.preds_output_folder, f"preds_manifest.json",)
        with open(preds_output_manifest, 'w') as fn:
            for i, file_name in enumerate(audio_file_paths):
                item = {
                    'audio_filepath': file_name,
                    'duration': durations[i],
                    'text': target_transcripts[i],
                    'pred_text': hypotheses[i].text,
                }
                fn.write(json.dumps(item) + "\n")

    else:
        preds_output_file = None

    beam_search_eval(
        all_hypotheses=all_hypotheses, target_transcripts=target_transcripts, preds_output_file=preds_output_file,
    )


if __name__ == '__main__':
    main()
