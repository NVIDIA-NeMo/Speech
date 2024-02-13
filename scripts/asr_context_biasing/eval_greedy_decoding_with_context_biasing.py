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
# This script evaluates CTC and Transducer (RNNT) models (only Hybrid Transducer-CTC in case of Transducer) in context biasing mode
# by applying CTC-based Word Spotter (paper link) 

# Config Help

To discover all arguments of the script, please run :
python eval_greedy_decoding_with_context_biasing.py --help
python eval_greedy_decoding_with_context_biasing.py --cfg job

# USAGE

python eval_greedy_decoding_with_context_biasing.py \
            nemo_model_file=<path to the .nemo file of the model> \
            input_manifest=<path to the evaluation JSON manifest file \
            preds_output_folder=<folder to store the predictions> \
            decoder_type=<type of model decoder [ctc or rnnt]> \
            acoustic_batch_size=<batch size to calculate log probabilities> \
            apply_context_biasing=<True or False to apply context biasing> \
            context_file=<path to the context biasing file with key words/phrases> \
            beam_threshold=[<list of the beam thresholds, separated with commas>] \
            context_score=[<list of the context scores, separated with commas>] \
            ctc_ali_token_weight=[<list of the ctc alignment token weights, separated with commas>] \
            ...

# Description of context biasing graph:
Context biasing file contains words/phrases with their spellings
(one word/phrase per line, spellings are separated from word/phrase by underscore symbol):
WORD1_SPELLING1
WORD2_SPELLING1_SPELLING2
...
nvidia_nvidia
gpu_gpu_g p u
nvlink_nvlink_nv link
...
alternative spellings help to improve the recognition accuracy of abbreviations and complicated words,
which are often recognized as separate words (gpu -> g p u, nvlink -> nv link, tensorrt -> tensor rt, and so on).


# Grid Search for Hyper parameters

For grid search, you can provide a list of arguments as follows -

            beam_threshold=[4.0,5.0,6.0,....] \
            context_score=[1.0,1.5,...,4.0,4.5] \
            ctc_ali_token_weight=[0.1,0.2,...,0.7,0.8] \

"""


import contextlib
import json
import os
import tempfile
from dataclasses import dataclass, field, is_dataclass
from pathlib import Path
from typing import Dict, Optional

import editdistance
import numpy as np
import torch
from omegaconf import MISSING, OmegaConf
from sklearn.model_selection import ParameterGrid
from tqdm.auto import tqdm

import nemo.collections.asr as nemo_asr
from nemo.collections.asr.models import EncDecCTCModelBPE, EncDecHybridRNNTCTCModel
from nemo.collections.asr.parts import context_biasing
from nemo.core.config import hydra_runner
from nemo.utils import logging


@dataclass
class EvalContextBiasingConfig:
    """
    Evaluate CTC and Transducer (RNNT) ASR models in greedy decoding with context biasing.
    """

    # # The path of the '.nemo' file of the ASR model or the name of a pretrained model (ngc / huggingface)
    nemo_model_file: str = MISSING

    # File paths
    input_manifest: str = MISSING  # The manifest file of the evaluation set
    preds_output_folder: str = MISSING  # The folder where the predictions are stored

    # Parameters for inference
    acoustic_batch_size: int = 128  # The batch size to calculate log probabilities
    beam_batch_size: int = 128  # The batch size to be used for beam search decoding
    device: str = "cuda"  # The device to load the model onto to calculate log probabilities
    use_amp: bool = False  # Whether to use AMP if available to calculate log probabilities
    num_workers: int = 1  # Number of workers for DataLoader
    decoder_type: Optional[str] = None  # [ctc, rnnt] decoder type for asr model

    # Context-Biasing params
    apply_context_biasing: bool = False  # True in case of context biasing
    context_file: str = MISSING  # text file with context biasing words and their spellings
    spelling_separator: str = "_"  # separator between word and its spellings in context biasing file
    beam_threshold: list[float] = field(default_factory=lambda: [5.0])  # beam pruning threshold for ctc-ws decoding
    context_score: list[float] = field(default_factory=lambda: [3.0])  # per token weight for context biasing words
    ctc_ali_token_weight: list[float] = field(
        default_factory=lambda: [0.6]
    )  # weight of CTC tokens to prevent false accept errors
    print_cb_stats: bool = False  # print context biasing stats (mostly for debugging)

    # Auxiliary parameters
    sort_logits: bool = True  # do logits sorting before decoding - it reduces computation on puddings
    softmax_temperature: float = 1.00
    preserve_alignments: bool = False


def decoding_step(
    asr_model: nemo_asr.models.ASRModel,
    cfg: EvalContextBiasingConfig,
    encoder_outputs: list[torch.Tensor],
    ctc_logprobs: list[np.ndarray],
    target_transcripts: list[str],
    audio_file_paths: list[str],
    durations: list[str],
    preds_output_manifest: str,
    beam_batch_size: int = 128,
    progress_bar: bool = True,
    context_graph: context_biasing.ContextGraphCTC = None,
    blank_idx: int = 0,
    hp: Optional[Dict] = None,
) -> tuple[float, float]:

    # run CTC-based Word Spotter:
    if cfg.apply_context_biasing:
        ws_results = {}
        for idx, logits in tqdm(
            enumerate(ctc_logprobs), desc=f"Eval CTC-based Word Spotter...", ncols=120, total=len(ctc_logprobs)
        ):
            ws_results[audio_file_paths[idx]] = context_biasing.run_word_spotter(
                logits,
                context_graph,
                asr_model,
                blank_idx=blank_idx,
                beam_threshold=hp['beam_threshold'],
                cb_weight=hp['context_score'],
                ctc_ali_token_weight=hp['ctc_ali_token_weight'],
            )

    level = logging.getEffectiveLevel()
    logging.setLevel(logging.CRITICAL)
    # reset config
    asr_model.change_decoding_strategy(None)

    # preserve alignment:
    asr_model.cfg.decoding.preserve_alignments = cfg.preserve_alignments

    # update model's decoding strategy config
    if isinstance(asr_model, EncDecCTCModelBPE):
        # in case of ctc
        asr_model.cfg.decoding.strategy = "greedy"
    else:
        # in case of rnnt
        asr_model.cfg.decoding.strategy = "greedy_batch"
        # fast greedy batch decoding:
        asr_model.cfg.decoding.greedy.loop_labels = True

    # update model's decoding strategy
    asr_model.change_decoding_strategy(asr_model.cfg.decoding)
    logging.setLevel(level)

    wer_dist_first = cer_dist_first = 0
    words_count = chars_count = sample_idx = 0

    out_manifest = open(preds_output_manifest, 'w', encoding='utf_8', newline='\n')

    # ctc part for both EncDecCTCModelBPE and EncDecHybridRNNTCTCModel
    if cfg.decoder_type == "ctc":
        for batch_idx, probs in enumerate(ctc_logprobs):
            preds = np.argmax(probs, axis=1)
            if cfg.apply_context_biasing and ws_results[audio_file_paths[batch_idx]]:
                # make new text by mearging alignment with ctc-ws predictions:
                if cfg.print_cb_stats:
                    logging.info("\n" + "********" * 10)
                    logging.info(f"File name: {audio_file_paths[batch_idx]}")
                pred_text, raw_text = context_biasing.merge_alignment_with_ws_hyps(
                    preds,
                    asr_model,
                    ws_results[audio_file_paths[batch_idx]],
                    decoder_type="ctc",
                    blank_idx=blank_idx,
                    print_stats=cfg.print_cb_stats,
                )
                if cfg.print_cb_stats:
                    logging.info(f"raw text: {raw_text}")
                    logging.info(f"hyp text: {pred_text}")
                    logging.info(f"ref text: {target_transcripts[batch_idx]}")
            else:
                preds_tensor = torch.tensor(preds, device='cpu').unsqueeze(0)
                if isinstance(asr_model, EncDecHybridRNNTCTCModel):
                    pred_text = asr_model.ctc_decoding.ctc_decoder_predictions_tensor(preds_tensor)[0][0]
                else:
                    pred_text = asr_model.wer.decoding.ctc_decoder_predictions_tensor(preds_tensor)[0][0]

            pred_split_w = pred_text.split()
            target_split_w = target_transcripts[batch_idx].split()
            pred_split_c = list(pred_text)
            target_split_c = list(target_transcripts[batch_idx])

            wer_dist = editdistance.eval(target_split_w, pred_split_w)
            cer_dist = editdistance.eval(target_split_c, pred_split_c)

            wer_dist_first += wer_dist
            cer_dist_first += cer_dist
            words_count += len(target_split_w)
            chars_count += len(target_split_c)

            if preds_output_manifest:
                item = {
                    'audio_filepath': audio_file_paths[batch_idx],
                    'duration': durations[batch_idx],
                    'text': target_transcripts[batch_idx],
                    'pred_text': pred_text,
                    'wer': f"{wer_dist/len(target_split_w):.4f}",
                }
                print(json.dumps(item), file=out_manifest)
        out_manifest.close()

        return wer_dist_first / words_count, cer_dist_first / chars_count

    # rnnt part for EncDecHybridRNNTCTCModel
    else:
        if progress_bar:
            description = "Greedy_batch decoding.."
            it = tqdm(range(int(np.ceil(len(encoder_outputs) / beam_batch_size))), desc=description, ncols=120)
        else:
            it = range(int(np.ceil(len(encoder_outputs) / beam_batch_size)))
        for batch_idx in it:
            probs_batch = encoder_outputs[batch_idx * beam_batch_size : (batch_idx + 1) * beam_batch_size]
            probs_lens = torch.tensor([prob.shape[-1] for prob in probs_batch])
            with torch.no_grad():
                packed_batch = torch.zeros(len(probs_batch), probs_batch[0].shape[0], max(probs_lens), device='cpu')

                for prob_index in range(len(probs_batch)):
                    packed_batch[prob_index, :, : probs_lens[prob_index]] = torch.tensor(
                        probs_batch[prob_index].unsqueeze(0), device=packed_batch.device, dtype=packed_batch.dtype
                    )
                best_hyp_batch, beams_batch = asr_model.decoding.rnnt_decoder_predictions_tensor(
                    packed_batch, probs_lens, return_hypotheses=True,
                )
            beams_batch = [[x] for x in best_hyp_batch]

            for beams_idx, beams in enumerate(beams_batch):
                target = target_transcripts[sample_idx + beams_idx]
                target_split_w = target.split()
                target_split_c = list(target)
                words_count += len(target_split_w)
                chars_count += len(target_split_c)
                for candidate_idx, candidate in enumerate(beams):
                    if cfg.apply_context_biasing and ws_results[audio_file_paths[sample_idx + beams_idx]]:
                        # make new text by mearging alignment with ctc-ws predictions:
                        if cfg.print_cb_stats:
                            logging.info("\n" + "********" * 10)
                            logging.info(f"File name: {audio_file_paths[batch_idx]}")
                        pred_text, raw_text = context_biasing.merge_alignment_with_ws_hyps(
                            candidate,
                            asr_model,
                            ws_results[audio_file_paths[sample_idx + beams_idx]],
                            decoder_type="rnnt",
                            blank_idx=blank_idx,
                            print_stats=cfg.print_cb_stats,
                        )
                        if cfg.print_cb_stats:
                            logging.info(f"raw text: {raw_text}")
                            logging.info(f"hyp text: {pred_text}")
                            logging.info(f"ref text: {target_transcripts[sample_idx + beams_idx]}")
                    else:
                        pred_text = candidate.text

                    pred_split_w = pred_text.split()
                    wer_dist = editdistance.eval(target_split_w, pred_split_w)
                    pred_split_c = list(pred_text)
                    cer_dist = editdistance.eval(target_split_c, pred_split_c)

                    if candidate_idx == 0:
                        # first candidate
                        wer_dist_tosave = wer_dist
                        wer_dist_first += wer_dist
                        cer_dist_first += cer_dist

                # write manifest with prediction results
                alignment = []

                if preds_output_manifest:
                    item = {
                        'audio_filepath': audio_file_paths[sample_idx + beams_idx],
                        'duration': durations[sample_idx + beams_idx],
                        'text': target_transcripts[sample_idx + beams_idx],
                        'pred_text': pred_text,
                        'wer': f"{wer_dist_tosave/len(target_split_w):.3f}",
                        'alignment': f"{alignment}",
                    }
                    print(json.dumps(item), file=out_manifest)

            sample_idx += len(probs_batch)
        out_manifest.close()

        return wer_dist_first / words_count, cer_dist_first / chars_count


@hydra_runner(config_path=None, config_name='EvalContextBiasingConfig', schema=EvalContextBiasingConfig)
def main(cfg: EvalContextBiasingConfig):
    if is_dataclass(cfg):
        cfg = OmegaConf.structured(cfg)

    assert os.path.isfile(cfg.input_manifest), f"input_manifest {cfg.input_manifest} does not exist"
    assert cfg.context_file, "context_file must be provided for f-score computation"
    assert os.path.isfile(cfg.context_file), f"context_file {cfg.context_file} does not exist"
    assert cfg.decoder_type in ["ctc", "rnnt"], "decoder_type must be ctc or rnnt"
    assert cfg.preds_output_folder, "preds_output_folder must be provided"
    assert os.path.isdir(cfg.preds_output_folder), f"preds_output_folder {cfg.preds_output_folder} does not exist"

    # load nemo asr model
    if cfg.nemo_model_file.endswith('.nemo'):
        asr_model = nemo_asr.models.ASRModel.restore_from(cfg.nemo_model_file, map_location=torch.device(cfg.device))
    else:
        logging.warning(
            "nemo_model_file does not end with .nemo, therefore trying to load a pretrained model with this name."
        )
        asr_model = nemo_asr.models.ASRModel.from_pretrained(
            cfg.nemo_model_file, map_location=torch.device(cfg.device)
        )
    if not isinstance(asr_model, (EncDecCTCModelBPE, EncDecHybridRNNTCTCModel)):
        raise ValueError("ASR model must be CTC BPE or Hybrid Transducer-CTC")

    # load nemo manifest
    target_transcripts = []
    durations = []
    manifest_dir = Path(cfg.input_manifest).parent
    with open(cfg.input_manifest, 'r', encoding='utf_8') as manifest_file:
        audio_file_paths = []
        for line in tqdm(manifest_file, desc=f"Reading Manifest {cfg.input_manifest} ...", ncols=120):
            data = json.loads(line)
            audio_file = Path(data['audio_filepath'])
            if not audio_file.is_file() and not audio_file.is_absolute():
                audio_file = manifest_dir / audio_file
            target_transcripts.append(data['text'])
            durations.append(data['duration'])
            audio_file_paths.append(str(audio_file.absolute()))

    if cfg.use_amp:
        if torch.cuda.is_available() and hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'autocast'):
            logging.info("AMP is enabled!\n")
            autocast = torch.cuda.amp.autocast
        else:
            autocast = contextlib.nullcontext
    else:
        autocast = contextlib.nullcontext

    # manual calculation of encoder_embeddings
    with autocast():
        with torch.no_grad():
            asr_model.eval()
            asr_model.encoder.freeze()
            device = next(asr_model.parameters()).device
            encoder_outputs = []
            ctc_logprobs = []
            if isinstance(asr_model, EncDecCTCModelBPE):
                # in case of EncDecCTCModelBPE
                hyp_results = asr_model.transcribe(
                    audio_file_paths, batch_size=cfg.acoustic_batch_size, return_hypotheses=True
                )
                ctc_logprobs = [hyp.alignments.cpu().numpy() for hyp in hyp_results]
                blank_idx = asr_model.decoding.blank_id
            else:
                # in case of EncDecHybridRNNTCTCModel
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

                    for test_batch in tqdm(
                        temporary_datalayer, desc="Getting encoder and CTC decoder outputs...", disable=False
                    ):
                        encoded, encoded_len = asr_model.forward(
                            input_signal=test_batch[0].to(device), input_signal_length=test_batch[1].to(device)
                        )
                        ctc_dec_outputs = asr_model.ctc_decoder(encoder_output=encoded).cpu()
                        # dump encoder embeddings per file
                        for idx in range(encoded.shape[0]):
                            encoded_no_pad = encoded[idx, :, : encoded_len[idx]]
                            ctc_dec_outputs_no_pad = ctc_dec_outputs[idx, : encoded_len[idx]]
                            encoder_outputs.append(encoded_no_pad)
                            ctc_logprobs.append(ctc_dec_outputs_no_pad.cpu().numpy())
                    blank_idx = asr_model.decoder.blank_idx

    # load context biasing words
    context_transcripts = []
    for line in open(cfg.context_file).readlines():
        item = line.strip().lower().split(cfg.spelling_separator)
        word = item[0]
        word_tokenization = [asr_model.tokenizer.text_to_ids(x) for x in item[1:]]
        context_transcripts.append([word, word_tokenization])
    context_words = [item[0] for item in context_transcripts]
    # build context graph:
    if cfg.apply_context_biasing:
        context_graph = context_biasing.ContextGraphCTC(blank_id=blank_idx)
        context_graph.add_to_graph(context_transcripts)
    else:
        context_graph = None

    # sort encoder_outputs according to length:
    if cfg.decoder_type == "rnnt" and cfg.sort_logits:
        encoder_outputs_with_indeces = sorted(enumerate(encoder_outputs), key=lambda x: x[1].size()[1], reverse=True)
        encoder_outputs_sorted = []
        target_transcripts_sorted = []
        audio_file_paths_sorted = []
        durations_sorted = []
        ctc_logprobs_sorted = []
        for pair in encoder_outputs_with_indeces:
            encoder_outputs_sorted.append(pair[1])
            target_transcripts_sorted.append(target_transcripts[pair[0]])
            audio_file_paths_sorted.append(audio_file_paths[pair[0]])
            durations_sorted.append(durations[pair[0]])
            ctc_logprobs_sorted.append(ctc_logprobs[pair[0]])
        encoder_outputs = encoder_outputs_sorted
        target_transcripts = target_transcripts_sorted
        audio_file_paths = audio_file_paths_sorted
        durations = durations_sorted
        ctc_logprobs = ctc_logprobs_sorted

    # setup search parameters grid
    params = {
        'beam_threshold': cfg.beam_threshold,
        'context_score': cfg.context_score,
        'ctc_ali_token_weight': cfg.ctc_ali_token_weight,
    }
    hp_grid = ParameterGrid(params)
    hp_grid = list(hp_grid)

    logging.info(f"=========================Starting the decoding========================")
    logging.info(f"Grid search size: {len(hp_grid)}")
    logging.info(f"It may take some time...")
    logging.info(f"======================================================================")

    asr_model = asr_model.to('cpu')
    best_wer = 1e6

    # run decoding step for each hyper parameter set
    for hp in hp_grid:
        results_file = f"preds_out_manifest_bthr-{hp['beam_threshold']}_cs-{hp['context_score']}ctcw-{hp['ctc_ali_token_weight']}.json"
        preds_output_manifest = os.path.join(cfg.preds_output_folder, results_file)
        candidate_wer, candidate_cer = decoding_step(
            asr_model,
            cfg,
            encoder_outputs=encoder_outputs,
            target_transcripts=target_transcripts,
            audio_file_paths=audio_file_paths,
            durations=durations,
            beam_batch_size=cfg.beam_batch_size,
            progress_bar=True,
            preds_output_manifest=preds_output_manifest,
            context_graph=context_graph,
            ctc_logprobs=ctc_logprobs,
            blank_idx=blank_idx,
            hp=hp,
        )

        # compute fscore
        fscore_stats = context_biasing.compute_fscore(preds_output_manifest, context_words)

        # find the best wer value
        if candidate_wer < best_wer:
            best_beam_threshold = hp["beam_threshold"]
            best_context_score = hp["context_score"]
            best_ctc_ali_token_weight = hp["ctc_ali_token_weight"]
            best_wer = candidate_wer
            best_fscore_stats = fscore_stats

        logging.info(f"======================================================================")
        logging.info(f"Greedy WER/CER = {candidate_wer:.2%}/{candidate_cer:.2%}")
        logging.info(f"Precision/Recall/Fscore = {fscore_stats[0]:.4f}/{fscore_stats[1]:.4f}/{fscore_stats[2]:.4f}")
        logging.info(
            f"Params: b_thr = {hp['beam_threshold']}, cs = {hp['context_score']}, ctc_ali_weight = {hp['ctc_ali_token_weight']}"
        )
        logging.info(f"======================================================================")

    if len(hp_grid) > 1:
        logging.info(f"=========================Best Results=================================")
        logging.info(f"Best WER = {best_wer:.2%}")
        logging.info(
            f"Best Precision/Recall/Fscore = {best_fscore_stats[0]:.4f}/{best_fscore_stats[1]:.4f}/{best_fscore_stats[2]:.4f}"
        )
        logging.info(
            f"Best beam_threshold = {best_beam_threshold}, context_score = {best_context_score}, ctc_ali_token_weight = {best_ctc_ali_token_weight}"
        )


if __name__ == '__main__':
    main()
