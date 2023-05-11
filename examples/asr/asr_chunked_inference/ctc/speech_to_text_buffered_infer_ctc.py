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

"""
This script serves three goals:
    (1) Demonstrate how to use NeMo Models outside of PytorchLightning
    (2) Shows example of batch ASR inference
    (3) Serves as CI test for pre-trained checkpoint

python speech_to_text_buffered_infer_ctc.py \
    model_path=null \
    pretrained_name=null \
    audio_dir="<remove or path to folder of audio files>" \
    dataset_manifest="<remove or path to manifest>" \
    output_filename="<remove or specify output filename>" \
    total_buffer_in_secs=4.0 \
    chunk_len_in_secs=1.6 \
    model_stride=4 \
    batch_size=32 \
    clean_groundtruth_text=True \
    langid='en'

# NOTE:
    You can use `DEBUG=1 python speech_to_text_buffered_infer_ctc.py ...` to print out the
    predictions of the model, and ground-truth text if presents in manifest.
"""
import contextlib
import copy
import glob
import math
import os
from dataclasses import dataclass, is_dataclass
from typing import Optional

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf

from nemo.collections.asr.metrics.wer import CTCDecodingConfig
from nemo.collections.asr.models import EncDecCTCModel, EncDecHybridRNNTCTCModel
from nemo.collections.asr.parts.utils.eval_utils import cal_write_wer
from nemo.collections.asr.parts.utils.streaming_utils import FrameBatchASR
from nemo.collections.asr.parts.utils.transcribe_utils import (
    compute_output_filename,
    get_buffered_pred_feat,
    setup_model,
    write_transcription,
)
from nemo.core.config import hydra_runner
from nemo.utils import logging

can_gpu = torch.cuda.is_available()


@dataclass
class TranscriptionConfig:
    # Required configs
    model_path: Optional[str] = None  # Path to a .nemo file
    pretrained_name: Optional[str] = None  # Name of a pretrained model
    audio_dir: Optional[str] = None  # Path to a directory which contains audio files
    dataset_manifest: Optional[str] = None  # Path to dataset's JSON manifest

    # General configs
    output_filename: Optional[str] = None
    batch_size: int = 32
    num_workers: int = 0
    append_pred: bool = False  # Sets mode of work, if True it will add new field transcriptions.
    pred_name_postfix: Optional[str] = None  # If you need to use another model name, rather than standard one.
    random_seed: Optional[int] = None  # seed number going to be used in seed_everything()

    # Set to True to output greedy timestamp information (only supported models)
    compute_timestamps: bool = False

    # Set to True to output language ID information
    compute_langs: bool = False

    # Chunked configs
    chunk_len_in_secs: float = 1.6  # Chunk length in seconds
    total_buffer_in_secs: float = 4.0  # Length of buffer (chunk + left and right padding) in seconds
    model_stride: int = 8  # Model downsampling factor, 8 for Citrinet and FasConformer models and 4 for Conformer models.

    # Decoding strategy for CTC models
    decoding: CTCDecodingConfig = CTCDecodingConfig()

    # Set `cuda` to int to define CUDA device. If 'None', will look for CUDA
    # device anyway, and do inference on CPU only if CUDA device is not found.
    # If `cuda` is a negative number, inference will be on CPU only.
    cuda: Optional[int] = None
    amp: bool = False
    audio_type: str = "wav"

    # Recompute model transcription, even if the output folder exists with scores.
    overwrite_transcripts: bool = True

    # Config for word / character error rate calculation
    calculate_wer: bool = True
    clean_groundtruth_text: bool = False
    langid: str = "en"  # specify this for convert_num_to_words step in groundtruth cleaning
    use_cer: bool = False


@hydra_runner(config_name="TranscriptionConfig", schema=TranscriptionConfig)
def main(cfg: TranscriptionConfig) -> TranscriptionConfig:
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')
    torch.set_grad_enabled(False)

    for key in cfg:
        cfg[key] = None if cfg[key] == 'None' else cfg[key]

    if is_dataclass(cfg):
        cfg = OmegaConf.structured(cfg)

    if cfg.random_seed:
        pl.seed_everything(cfg.random_seed)

    if cfg.model_path is None and cfg.pretrained_name is None:
        raise ValueError("Both cfg.model_path and cfg.pretrained_name cannot be None!")
    if cfg.audio_dir is None and cfg.dataset_manifest is None:
        raise ValueError("Both cfg.audio_dir and cfg.dataset_manifest cannot be None!")

    filepaths = None
    manifest = cfg.dataset_manifest
    if cfg.audio_dir is not None:
        filepaths = list(glob.glob(os.path.join(cfg.audio_dir, f"**/*.{cfg.audio_type}"), recursive=True))
        manifest = None  # ignore dataset_manifest if audio_dir and dataset_manifest both presents

    # setup GPU
    if cfg.cuda is None:
        if torch.cuda.is_available():
            device = [0]  # use 0th CUDA device
            accelerator = 'gpu'
        else:
            device = 1
            accelerator = 'cpu'
    else:
        device = [cfg.cuda]
        accelerator = 'gpu'
    map_location = torch.device('cuda:{}'.format(device[0]) if accelerator == 'gpu' else 'cpu')
    logging.info(f"Inference will be done on device : {device}")

    asr_model, model_name = setup_model(cfg, map_location)

    model_cfg = copy.deepcopy(asr_model._cfg)
    OmegaConf.set_struct(model_cfg.preprocessor, False)
    # some changes for streaming scenario
    model_cfg.preprocessor.dither = 0.0
    model_cfg.preprocessor.pad_to = 0

    if model_cfg.preprocessor.normalize != "per_feature":
        logging.error("Only EncDecCTCModelBPE models trained with per_feature normalization are supported currently")

    # Disable config overwriting
    OmegaConf.set_struct(model_cfg.preprocessor, True)

    # setup AMP (optional)
    if cfg.amp and torch.cuda.is_available() and hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'autocast'):
        logging.info("AMP enabled!\n")
        autocast = torch.cuda.amp.autocast
    else:

        @contextlib.contextmanager
        def autocast():
            yield

    # Compute output filename
    cfg = compute_output_filename(cfg, model_name)

    # if transcripts should not be overwritten, and already exists, skip re-transcription step and return
    if not cfg.overwrite_transcripts and os.path.exists(cfg.output_filename):
        logging.info(
            f"Previous transcripts found at {cfg.output_filename}, and flag `overwrite_transcripts`"
            f"is {cfg.overwrite_transcripts}. Returning without re-transcribing text."
        )
        return cfg

    # Setup decoding strategy
    if hasattr(asr_model, 'change_decoding_strategy'):
        if not isinstance(asr_model, EncDecCTCModel) and not isinstance(asr_model, EncDecHybridRNNTCTCModel):
            raise ValueError("The script supports ctc model and hybrid model with ctc decodng!")

        else:
            if cfg.compute_langs:
                raise ValueError("CTC models do not support `compute_langs` at the moment.")

            if hasattr(
                asr_model, 'cur_decoder'
            ):  # hybrid model with ctc decoding or potential other models containing decoding switch feature
                asr_model.change_decoding_strategy(cfg.decoding, decoder_type='ctc')

            else:  # ctc model
                asr_model.change_decoding_strategy(cfg.decoding)

    asr_model.eval()
    asr_model = asr_model.to(asr_model.device)

    feature_stride = model_cfg.preprocessor['window_stride']
    model_stride_in_secs = feature_stride * cfg.model_stride
    total_buffer = cfg.total_buffer_in_secs
    chunk_len = float(cfg.chunk_len_in_secs)

    tokens_per_chunk = math.ceil(chunk_len / model_stride_in_secs)
    mid_delay = math.ceil((chunk_len + (total_buffer - chunk_len) / 2) / model_stride_in_secs)
    logging.info(f"tokens_per_chunk is {tokens_per_chunk}, mid_delay is {mid_delay}")

    frame_asr = FrameBatchASR(
        asr_model=asr_model, frame_len=chunk_len, total_buffer=cfg.total_buffer_in_secs, batch_size=cfg.batch_size,
    )

    hyps = get_buffered_pred_feat(
        frame_asr,
        chunk_len,
        tokens_per_chunk,
        mid_delay,
        model_cfg.preprocessor,
        model_stride_in_secs,
        asr_model.device,
        manifest,
        filepaths,
    )
    output_filename, pred_text_attr_name = write_transcription(
        hyps, cfg, model_name, filepaths=filepaths, compute_langs=False, compute_timestamps=False
    )
    logging.info(f"Finished writing predictions to {output_filename}!")

    if cfg.calculate_wer:
        output_manifest_w_wer, total_res, _ = cal_write_wer(
            pred_manifest=output_filename,
            pred_text_attr_name=pred_text_attr_name,
            clean_groundtruth_text=cfg.clean_groundtruth_text,
            langid=cfg.langid,
            use_cer=cfg.use_cer,
            output_filename=None,
        )
        if output_manifest_w_wer:
            logging.info(f"Writing prediction and error rate of each sample to {output_manifest_w_wer}!")
            logging.info(f"{total_res}")

    return cfg


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
