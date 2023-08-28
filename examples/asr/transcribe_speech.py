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

import contextlib
import os
from dataclasses import dataclass, is_dataclass
from typing import List, Optional, Union

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf, open_dict

from nemo.collections.asr.metrics.rnnt_wer import RNNTDecodingConfig
from nemo.collections.asr.metrics.wer import CTCDecodingConfig
from nemo.collections.asr.models import EncDecCTCModel, EncDecHybridRNNTCTCModel
from nemo.collections.asr.modules.conformer_encoder import ConformerChangeConfig
from nemo.collections.asr.parts.utils.eval_utils import cal_write_wer
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from nemo.collections.asr.parts.utils.transcribe_utils import (
    compute_output_filename,
    prepare_audio_data,
    setup_model,
    transcribe_partial_audio,
    write_transcription,
)
from nemo.core.config import hydra_runner
from nemo.utils import logging

"""
Transcribe audio file on a single CPU/GPU. Useful for transcription of moderate amounts of audio data.

# Arguments
  model_path: path to .nemo ASR checkpoint
  pretrained_name: name of pretrained ASR model (from NGC registry)
  audio_dir: path to directory with audio files
  dataset_manifest: path to dataset JSON manifest file (in NeMo format)

  compute_timestamps: Bool to request greedy time stamp information (if the model supports it)
  compute_langs: Bool to request language ID information (if the model supports it)

  (Optionally: You can limit the type of timestamp computations using below overrides)
  ctc_decoding.ctc_timestamp_type="all"  # (default all, can be [all, char, word])
  rnnt_decoding.rnnt_timestamp_type="all"  # (default all, can be [all, char, word])

  (Optionally: You can limit the type of timestamp computations using below overrides)
  ctc_decoding.ctc_timestamp_type="all"  # (default all, can be [all, char, word])
  rnnt_decoding.rnnt_timestamp_type="all"  # (default all, can be [all, char, word])

  output_filename: Output filename where the transcriptions will be written
  batch_size: batch size during inference

  cuda: Optional int to enable or disable execution of model on certain CUDA device.
  allow_mps: Bool to allow using MPS (Apple Silicon M-series GPU) device if available
  amp: Bool to decide if Automatic Mixed Precision should be used during inference
  audio_type: Str filetype of the audio. Supported = wav, flac, mp3

  overwrite_transcripts: Bool which when set allows repeated transcriptions to overwrite previous results.

  ctc_decoding: Decoding sub-config for CTC. Refer to documentation for specific values.
  rnnt_decoding: Decoding sub-config for RNNT. Refer to documentation for specific values.

  calculate_wer: Bool to decide whether to calculate wer/cer at end of this script
  clean_groundtruth_text: Bool to clean groundtruth text
  langid: Str used for convert_num_to_words during groundtruth cleaning
  use_cer: Bool to use Character Error Rate (CER)  or Word Error Rate (WER)

# Usage
ASR model can be specified by either "model_path" or "pretrained_name".
Data for transcription can be defined with either "audio_dir" or "dataset_manifest".
append_pred - optional. Allows you to add more than one prediction to an existing .json
pred_name_postfix - optional. The name you want to be written for the current model
Results are returned in a JSON manifest file.

python transcribe_speech.py \
    model_path=null \
    pretrained_name=null \
    audio_dir="<remove or path to folder of audio files>" \
    dataset_manifest="<remove or path to manifest>" \
    output_filename="<remove or specify output filename>" \
    clean_groundtruth_text=True \
    langid='en' \
    batch_size=32 \
    compute_timestamps=False \
    compute_langs=False \
    cuda=0 \
    amp=True \
    append_pred=False \
    pred_name_postfix="<remove or use another model name for output filename>"
"""


@dataclass
class ModelChangeConfig:

    # Sub-config for changes specific to the Conformer Encoder
    conformer: ConformerChangeConfig = ConformerChangeConfig()


@dataclass
class TranscriptionConfig:
    # Required configs
    model_path: Optional[str] = None  # Path to a .nemo file
    pretrained_name: Optional[str] = None  # Name of a pretrained model
    audio_dir: Optional[str] = None  # Path to a directory which contains audio files
    dataset_manifest: Optional[str] = None  # Path to dataset's JSON manifest
    channel_selector: Optional[
        Union[int, str]
    ] = None  # Used to select a single channel from multichannel audio, or use average across channels
    audio_key: str = 'audio_filepath'  # Used to override the default audio key in dataset_manifest
    eval_config_yaml: Optional[str] = None  # Path to a yaml file of config of evaluation

    # General configs
    output_filename: Optional[str] = None
    batch_size: int = 32
    num_workers: int = 0
    append_pred: bool = False  # Sets mode of work, if True it will add new field transcriptions.
    pred_name_postfix: Optional[str] = None  # If you need to use another model name, rather than standard one.
    random_seed: Optional[int] = None  # seed number going to be used in seed_everything()

    # Set to True to output greedy timestamp information (only supported models)
    compute_timestamps: bool = False
    # set to True if need to return full alignment information
    preserve_alignment: bool = False

    # Set to True to output language ID information
    compute_langs: bool = False

    # Set `cuda` to int to define CUDA device. If 'None', will look for CUDA
    # device anyway, and do inference on CPU only if CUDA device is not found.
    # If `cuda` is a negative number, inference will be on CPU only.
    cuda: Optional[int] = None
    allow_mps: bool = False  # allow to select MPS device (Apple Silicon M-series GPU)
    amp: bool = False
    amp_dtype: str = "float16"  # can be set to "float16" or "bfloat16" when using amp
    audio_type: str = "wav"

    # Recompute model transcription, even if the output folder exists with scores.
    overwrite_transcripts: bool = True

    # Decoding strategy for CTC models
    ctc_decoding: CTCDecodingConfig = CTCDecodingConfig()

    # Decoding strategy for RNNT models
    rnnt_decoding: RNNTDecodingConfig = RNNTDecodingConfig(fused_batch_size=-1)

    # decoder type: ctc or rnnt, can be used to switch between CTC and RNNT decoder for Hybrid RNNT/CTC models
    decoder_type: Optional[str] = None
    # att_context_size can be set for cache-aware streaming models with multiple look-aheads
    att_context_size: Optional[list] = None

    # Use this for model-specific changes before transcription
    model_change: ModelChangeConfig = ModelChangeConfig()

    # Config for word / character error rate calculation
    calculate_wer: bool = True
    clean_groundtruth_text: bool = False
    langid: str = "en"  # specify this for convert_num_to_words step in groundtruth cleaning
    use_cer: bool = False

    # can be set to True to return list of transcriptions instead of the config
    # if True, will also skip writing anything to the output file
    return_transcriptions: bool = False

    # Set to False to return text instead of hypotheses from the transcribe function, so as to save memory
    return_hypotheses: bool = True


@hydra_runner(config_name="TranscriptionConfig", schema=TranscriptionConfig)
def main(cfg: TranscriptionConfig) -> Union[TranscriptionConfig, List[Hypothesis]]:
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

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

    # Load augmentor from exteranl yaml file which contains eval info, could be extend to other feature such VAD, P&C
    augmentor = None
    if cfg.eval_config_yaml:
        eval_config = OmegaConf.load(cfg.eval_config_yaml)
        augmentor = eval_config.test_ds.get("augmentor")
        logging.info(f"Will apply on-the-fly augmentation on samples during transcription: {augmentor} ")

    # setup GPU
    if cfg.cuda is None:
        if torch.cuda.is_available():
            device = [0]  # use 0th CUDA device
            accelerator = 'gpu'
            map_location = torch.device('cuda:0')
        elif cfg.allow_mps and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logging.warning(
                "MPS device (Apple Silicon M-series GPU) support is experimental."
                " Env variable `PYTORCH_ENABLE_MPS_FALLBACK=1` should be set in most cases to avoid failures."
            )
            device = [0]
            accelerator = 'mps'
            map_location = torch.device('mps')
        else:
            device = 1
            accelerator = 'cpu'
            map_location = torch.device('cpu')
    else:
        device = [cfg.cuda]
        accelerator = 'gpu'
        map_location = torch.device(f'cuda:{cfg.cuda}')

    logging.info(f"Inference will be done on device: {map_location}")

    asr_model, model_name = setup_model(cfg, map_location)

    trainer = pl.Trainer(devices=device, accelerator=accelerator)
    asr_model.set_trainer(trainer)
    asr_model = asr_model.eval()

    # we will adjust this flag if the model does not support it
    compute_timestamps = cfg.compute_timestamps
    compute_langs = cfg.compute_langs
    # has to be True if timestamps are required
    preserve_alignment = True if cfg.compute_timestamps else cfg.preserve_alignment

    # Check whether model and decoder type match
    if isinstance(asr_model, EncDecCTCModel):
        if cfg.decoder_type and cfg.decoder_type != 'ctc':
            raise ValueError('CTC model only support ctc decoding!')
    elif isinstance(asr_model, EncDecHybridRNNTCTCModel):
        if cfg.decoder_type and cfg.decoder_type not in ['ctc', 'rnnt']:
            raise ValueError('Hybrid model only support ctc or rnnt decoding!')
    else:  # rnnt model, there could be other models needs to be addressed.
        if cfg.decoder_type and cfg.decoder_type != 'rnnt':
            raise ValueError('RNNT model only support rnnt decoding!')

    if cfg.decoder_type and hasattr(asr_model.encoder, 'set_default_att_context_size'):
        asr_model.encoder.set_default_att_context_size(cfg.att_context_size)

    # Setup decoding strategy
    if hasattr(asr_model, 'change_decoding_strategy'):
        if cfg.decoder_type is not None:
            # TODO: Support compute_langs in CTC eventually
            if cfg.compute_langs and cfg.decoder_type == 'ctc':
                raise ValueError("CTC models do not support `compute_langs` at the moment")

            decoding_cfg = cfg.rnnt_decoding if cfg.decoder_type == 'rnnt' else cfg.ctc_decoding
            decoding_cfg.compute_timestamps = cfg.compute_timestamps  # both ctc and rnnt support it
            if 'preserve_alignments' in decoding_cfg:
                decoding_cfg.preserve_alignments = preserve_alignment
            if 'compute_langs' in decoding_cfg:
                decoding_cfg.compute_langs = cfg.compute_langs
            if hasattr(asr_model, 'cur_decoder'):
                asr_model.change_decoding_strategy(decoding_cfg, decoder_type=cfg.decoder_type)
            else:
                asr_model.change_decoding_strategy(decoding_cfg)

        # Check if ctc or rnnt model
        elif hasattr(asr_model, 'joint'):  # RNNT model
            cfg.rnnt_decoding.fused_batch_size = -1
            cfg.rnnt_decoding.compute_timestamps = cfg.compute_timestamps
            cfg.rnnt_decoding.compute_langs = cfg.compute_langs
            if 'preserve_alignments' in cfg.rnnt_decoding:
                cfg.rnnt_decoding.preserve_alignments = preserve_alignment

            asr_model.change_decoding_strategy(cfg.rnnt_decoding)
        else:
            if cfg.compute_langs:
                raise ValueError("CTC models do not support `compute_langs` at the moment.")
            cfg.ctc_decoding.compute_timestamps = cfg.compute_timestamps

            asr_model.change_decoding_strategy(cfg.ctc_decoding)

    # Setup decoding config based on model type and decoder_type
    with open_dict(cfg):
        if isinstance(asr_model, EncDecCTCModel) or (
            isinstance(asr_model, EncDecHybridRNNTCTCModel) and cfg.decoder_type == "ctc"
        ):
            cfg.decoding = cfg.ctc_decoding
        else:
            cfg.decoding = cfg.rnnt_decoding

    # prepare audio filepaths and decide wether it's partial audio
    filepaths, partial_audio = prepare_audio_data(cfg)

    # setup AMP (optional)
    if cfg.amp and torch.cuda.is_available() and hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'autocast'):
        logging.info("AMP enabled!\n")
        autocast = torch.cuda.amp.autocast
    else:

        @contextlib.contextmanager
        def autocast(dtype=None):
            yield

    # Compute output filename
    cfg = compute_output_filename(cfg, model_name)

    # if transcripts should not be overwritten, and already exists, skip re-transcription step and return
    if not cfg.return_transcriptions and not cfg.overwrite_transcripts and os.path.exists(cfg.output_filename):
        logging.info(
            f"Previous transcripts found at {cfg.output_filename}, and flag `overwrite_transcripts`"
            f"is {cfg.overwrite_transcripts}. Returning without re-transcribing text."
        )
        return cfg

    # transcribe audio

    amp_dtype = torch.float16 if cfg.amp_dtype == "float16" else torch.bfloat16

    with autocast(dtype=amp_dtype):
        with torch.no_grad():
            if partial_audio:
                transcriptions = transcribe_partial_audio(
                    asr_model=asr_model,
                    path2manifest=cfg.dataset_manifest,
                    batch_size=cfg.batch_size,
                    num_workers=cfg.num_workers,
                    return_hypotheses=cfg.return_hypotheses,
                    channel_selector=cfg.channel_selector,
                    augmentor=augmentor,
                    decoder_type=cfg.decoder_type,
                )
            else:
                transcriptions = asr_model.transcribe(
                    paths2audio_files=filepaths,
                    batch_size=cfg.batch_size,
                    num_workers=cfg.num_workers,
                    return_hypotheses=cfg.return_hypotheses,
                    channel_selector=cfg.channel_selector,
                    augmentor=augmentor,
                )

    logging.info(f"Finished transcribing {len(filepaths)} files !")
    logging.info(f"Writing transcriptions into file: {cfg.output_filename}")

    # if transcriptions form a tuple (from RNNT), extract just "best" hypothesis
    if type(transcriptions) == tuple and len(transcriptions) == 2:
        transcriptions = transcriptions[0]

    if cfg.return_transcriptions:
        return transcriptions

    # write audio transcriptions
    output_filename, pred_text_attr_name = write_transcription(
        transcriptions,
        cfg,
        model_name,
        filepaths=filepaths,
        compute_langs=compute_langs,
        compute_timestamps=compute_timestamps,
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
