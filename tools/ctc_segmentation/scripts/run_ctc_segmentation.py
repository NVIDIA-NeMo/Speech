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

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wav
import torch
from joblib import Parallel, delayed
from tqdm import tqdm
from utils import get_segments

import nemo.collections.asr as nemo_asr
from nemo.collections.asr.models.ctc_models import EncDecCTCModel
from nemo.collections.asr.models.hybrid_rnnt_ctc_models import EncDecHybridRNNTCTCModel

parser = argparse.ArgumentParser(description="CTC Segmentation")
parser.add_argument("--output_dir", default="output", type=str, help="Path to output directory")
parser.add_argument(
    "--data",
    type=str,
    required=True,
    help="Path to directory with audio files and associated transcripts (same respective names only formats are "
    "different or path to wav file (transcript should have the same base name and be located in the same folder"
    "as the wav file.",
)
parser.add_argument("--window_len", type=int, default=8000, help="Window size for ctc segmentation algorithm")
parser.add_argument("--sample_rate", type=int, default=16000, help="Sampling rate, Hz")
parser.add_argument(
    "--model", type=str, default="QuartzNet15x5Base-En", help="Path to model checkpoint or pre-trained model name",
)
parser.add_argument("--debug", action="store_true", help="Flag to enable debugging messages")
parser.add_argument(
    "--num_jobs",
    default=-2,
    type=int,
    help="The maximum number of concurrently running jobs, `-2` - all CPUs but one are used",
)

logger = logging.getLogger("ctc_segmentation")  # use module name

if __name__ == "__main__":

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    # setup logger
    log_dir = os.path.join(args.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"ctc_segmentation_{args.window_len}.log")
    if os.path.exists(log_file):
        os.remove(log_file)
    level = "DEBUG" if args.debug else "INFO"

    logger = logging.getLogger("CTC")
    file_handler = logging.FileHandler(filename=log_file)
    stdout_handler = logging.StreamHandler(sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(handlers=handlers, level=level)

    if os.path.exists(args.model):
        asr_model = nemo_asr.models.ASRModel.restore_from(args.model)
    else:
        asr_model = nemo_asr.models.ASRModel.from_pretrained(args.model, strict=False)

    if not (isinstance(asr_model, EncDecCTCModel) or isinstance(asr_model, EncDecHybridRNNTCTCModel)):
        raise NotImplementedError(
            f"Model is not an instance of NeMo EncDecCTCModel or ENCDecHybridRNNTCTCModel."
            " Currently only instances of these models are supported"
        )

    bpe_model = isinstance(asr_model, nemo_asr.models.EncDecCTCModelBPE) or isinstance(
        asr_model, nemo_asr.models.EncDecHybridRNNTCTCBPEModel
    )

    # get tokenizer used during training, None for char based models
    if bpe_model:
        tokenizer = asr_model.tokenizer
    else:
        tokenizer = None

    if isinstance(asr_model, EncDecHybridRNNTCTCModel):
        asr_model.change_decoding_strategy(decoder_type="ctc")

    # extract ASR vocabulary and add blank symbol
    if hasattr(asr_model, 'tokenizer'):  # i.e. tokenization is BPE-based
        vocabulary = asr_model.tokenizer.vocab
    elif hasattr(asr_model.decoder, "vocabulary"):  # i.e. tokenization is character-based
        vocabulary = asr_model.cfg.decoder.vocabulary
    else:
        raise ValueError("Unexpected model type. Vocabulary list not found.")

    vocabulary = ["ε"] + list(vocabulary)
    logging.debug(f"ASR Model vocabulary: {vocabulary}")

    data = Path(args.data)
    output_dir = Path(args.output_dir)

    if os.path.isdir(data):
        audio_paths = data.glob("*.wav")
        data_dir = data
    else:
        audio_paths = [Path(data)]
        data_dir = Path(os.path.dirname(data))

    all_log_probs = []
    all_transcript_file = []
    all_segment_file = []
    all_wav_paths = []
    segments_dir = os.path.join(args.output_dir, "segments")
    os.makedirs(segments_dir, exist_ok=True)

    index_duration = None
    for path_audio in audio_paths:
        logging.info(f"Processing {path_audio.name}...")
        transcript_file = os.path.join(data_dir, path_audio.name.replace(".wav", ".txt"))
        segment_file = os.path.join(
            segments_dir, f"{args.window_len}_" + path_audio.name.replace(".wav", "_segments.txt")
        )
        if not os.path.exists(transcript_file):
            logging.info(f"{transcript_file} not found. Skipping {path_audio.name}")
            continue
        try:
            sample_rate, signal = wav.read(path_audio)
            if len(signal) == 0:
                logging.error(f"Skipping {path_audio.name}")
                continue

            assert (
                sample_rate == args.sample_rate
            ), f"Sampling rate of the audio file {path_audio} doesn't match --sample_rate={args.sample_rate}"

            original_duration = len(signal) / sample_rate
            logging.debug(f"len(signal): {len(signal)}, sr: {sample_rate}")
            logging.debug(f"Duration: {original_duration}s, file_name: {path_audio}")

            hypotheses = asr_model.transcribe([str(path_audio)], batch_size=1, return_hypotheses=True)
            # if hypotheses form a tuple (from Hybrid model), extract just "best" hypothesis
            if type(hypotheses) == tuple and len(hypotheses) == 2:
                hypotheses = hypotheses[0]
            log_probs = hypotheses[
                0
            ].alignments  # note: "[0]" is for batch dimension unpacking (and here batch size=1)

            # move blank values to the first column (ctc-package compatibility)
            blank_col = log_probs[:, -1].reshape((log_probs.shape[0], 1))
            log_probs = np.concatenate((blank_col, log_probs[:, :-1]), axis=1)

            all_log_probs.append(log_probs)
            all_segment_file.append(str(segment_file))
            all_transcript_file.append(str(transcript_file))
            all_wav_paths.append(path_audio)

            if index_duration is None:
                index_duration = len(signal) / log_probs.shape[0] / sample_rate

        except Exception as e:
            logging.error(e)
            logging.error(f"Skipping {path_audio.name}")
            continue

    asr_model_type = type(asr_model)
    del asr_model
    torch.cuda.empty_cache()

    if len(all_log_probs) > 0:
        start_time = time.time()

        normalized_lines = Parallel(n_jobs=args.num_jobs)(
            delayed(get_segments)(
                all_log_probs[i],
                all_wav_paths[i],
                all_transcript_file[i],
                all_segment_file[i],
                vocabulary,
                tokenizer,
                bpe_model,
                index_duration,
                args.window_len,
                log_file=log_file,
                debug=args.debug,
            )
            for i in tqdm(range(len(all_log_probs)))
        )

        total_time = time.time() - start_time
        logger.info(f"Total execution time: ~{round(total_time/60)}min")
        logger.info(f"Saving logs to {log_file}")

    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            lines = f.readlines()
