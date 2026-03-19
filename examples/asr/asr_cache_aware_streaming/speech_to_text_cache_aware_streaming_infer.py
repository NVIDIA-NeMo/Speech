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

"""
This script can be used to simulate cache-aware streaming for ASR models. The ASR model to be used with this script need to get trained in streaming mode. Currently only Conformer models supports this streaming mode.
You may find examples of streaming models under 'NeMo/example/asr/conf/conformer/streaming/'.

It works both on a manifest of audio files or a single audio file. It can perform streaming for a single stream (audio) or perform the evalution in multi-stream model (batch_size>1).
The manifest file must conform to standard ASR definition - containing `audio_filepath` and `text` as the ground truth.

# Usage

## To evaluate a model in cache-aware streaming mode on a single audio file:

python speech_to_text_cache_aware_streaming_infer.py \
    model_path=asr_model.nemo \
    audio_file=audio_file.wav \
    compare_vs_offline=true \
    amp=true \
    debug_mode=true

## To evaluate a model in cache-aware streaming mode on a manifest file:

python speech_to_text_cache_aware_streaming_infer.py \
    model_path=asr_model.nemo \
    dataset_manifest=manifest_file.json \
    batch_size=16 \
    compare_vs_offline=true \
    amp=true \
    debug_mode=true

## It is also possible to use phrase boosting or external LM with cache-aware models:

python speech_to_text_cache_aware_streaming_infer.py \
    model_path=asr_model.nemo \
    dataset_manifest=manifest_file.json \
    batch_size=16 \
    rnnt_decoding.greedy.boosting_tree.key_phrases_file=key_words_list.txt \
    rnnt_decoding.greedy.boosting_tree_alpha=1.0 \
    rnnt_decoding.greedy.ngram_lm_model=lm_model.nemo \
    rnnt_decoding.greedy.ngram_lm_model=0.5 \
    compare_vs_offline=true \
    amp=true \
    debug_mode=true

You may drop the 'debug_mode' and 'compare_vs_offline' to speedup the streaming evaluation.
If compare_vs_offline is not used, then significantly larger batch_size can be used.
Setting `pad_and_drop_preencoded` would perform the caching for all steps including the first step.
It may result in slightly different outputs from the sub-sampling module compared to offline mode for some techniques like striding and sw_striding.
Enabling it would make it easier to export the model to ONNX.

For customization details (phrases list, n-gram LM) see details in the documentation:
https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/asr_language_modeling_and_customization.html

## Hybrid ASR models
For Hybrid ASR models which have two decoders, you may select the decoder by decoder_type DECODER_TYPE, where DECODER_TYPE can be "ctc" or "rnnt".
If decoder is not set, then the default decoder would be used which is the RNNT decoder for Hybrid ASR models.

## Multi-lookahead models
For models which support multiple lookaheads, the default is the first one in the list of model.encoder.att_context_size. To change it, you may use att_context_size, for example att_context_size=§[70,1].


## Evaluate a model trained with full context for offline mode

You may try the cache-aware streaming with a model trained with full context in offline mode.
But the accuracy would not be very good with small chunks as there is inconsistency between how the model is trained and how the streaming inference is done.
The accuracy of the model on the borders of chunks would not be very good.

To use a model trained with full context, you need to pass the chunk_size and shift_size arguments.
If shift_size is not passed, chunk_size would be used as the shift_size too.
Also argument online_normalization should be enabled to simulate a realistic streaming.
The following command would simulate cache-aware streaming on a pretrained model from NGC with chunk_size of 100, shift_size of 50 and 2 left chunks as left context.
The chunk_size of 100 would be 100*4*10=4000ms for a model with 4x downsampling and 10ms shift in feature extraction.

python speech_to_text_cache_aware_streaming_infer.py \
    pretrained_name=stt_en_conformer_ctc_large \
    chunk_size=100 \
    shift_size=50 \
    left_chunks=2 \
    online_normalization=true \
    dataset_manifest=manifest_file.json \
    batch_size=16 \
    compare_vs_offline=true \
    debug_mode=true

"""


import glob
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import lightning.pytorch as pl
import torch
from omegaconf import OmegaConf

from nemo.collections.asr.metrics.wer import word_error_rate
from nemo.collections.asr.parts.submodules.ctc_decoding import CTCDecodingConfig
from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTDecodingConfig
from nemo.collections.asr.parts.utils.manifest_utils import read_manifest
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer
from nemo.collections.asr.parts.utils.transcribe_utils import get_inference_device, get_inference_dtype, setup_model
from nemo.core.config import hydra_runner
from nemo.utils import logging


@dataclass
class TranscriptionConfig:
    """
    Transcription Configuration for cache-aware inference.
    """

    # Required configs
    model_path: Optional[str] = None  # Path to a .nemo file
    pretrained_name: Optional[str] = None  # Name of a pretrained model
    audio_dir: Optional[str] = None  # Path to a directory which contains audio files
    audio_type: str = "wav"  # type of audio file if audio_dir passed
    audio_file: Optional[str] = None  # Path to an audio file to perform streaming
    dataset_manifest: Optional[str] = None  # Path to dataset's JSON manifest
    output_path: Optional[str] = None  # Path to output file when manifest is used as input

    # General configs
    batch_size: int = 32
    # num_workers: int = 0
    # append_pred: bool = False  # Sets mode of work, if True it will add new field transcriptions.
    # pred_name_postfix: Optional[str] = None  # If you need to use another model name, rather than standard one.
    random_seed: Optional[int] = None  # seed number going to be used in seed_everything()

    # Chunked configs
    chunk_size: int = -1  # The chunk_size to be used for models trained with full context and offline models
    shift_size: int = -1  # The shift_size to be used for models trained with full context and offline models
    left_chunks: Optional[int] = (
        2  # The number of left chunks to be used as left context via caching for offline models
    )
    online_normalization: bool = False  # Perform normalization on the run per chunk.
    # `pad_and_drop_preencoded` enables padding the audio input and then dropping the extra steps after
    # the pre-encoding for all the steps including the the first step. It may make the outputs of the downsampling
    # slightly different from offline mode for some techniques like striding or sw_striding.
    pad_and_drop_preencoded: bool = False
    att_context_size: Optional[list] = (
        None  # Sets the att_context_size for the models which support multiple lookaheads
    )

    compare_vs_offline: bool = False  #  Whether to compare the output of the model with the offline mode.

    # Set `cuda` to int to define CUDA device. If 'None', will look for CUDA
    # device anyway, and do inference on CPU only if CUDA device is not found.
    # If `cuda` is a negative number, inference will be on CPU only.
    cuda: Optional[int] = None
    allow_mps: bool = False  # allow to select MPS device (Apple Silicon M-series GPU)
    amp: bool = False
    amp_dtype: str = "float16"  # can be set to "float16" or "bfloat16" when using amp
    # NB: default compute_dtype is float32 since currently cache-aware models do not work with different dtype
    compute_dtype: Optional[str] = (
        "float32"  # "float32" (default), "bfloat16" or "float16"; if None: bfloat16 if available else float32
    )
    matmul_precision: str = "high"  # Literal["highest", "high", "medium"]

    # Decoding strategy for CTC models
    ctc_decoding: CTCDecodingConfig = field(default_factory=CTCDecodingConfig)
    # Decoding strategy for RNNT models
    rnnt_decoding: RNNTDecodingConfig = field(default_factory=lambda: RNNTDecodingConfig(fused_batch_size=-1))
    # Selects the decoder for Hybrid ASR models which has both the CTC and RNNT decoder.
    decoder_type: Optional[str] = None  # Literal["ctc", "rnnt"]

    # Config for word / character error rate calculation
    # calculate_wer: bool = True
    # clean_groundtruth_text: bool = False
    # langid: str = "en"  # specify this for convert_num_to_words step in groundtruth cleaning
    # use_cer: bool = False
    debug_mode: bool = False  # Whether to print more detail in the output.


def extract_transcriptions(hyps):
    """
    The transcribed_texts returned by CTC and RNNT models are different.
    This method would extract and return the text section of the hypothesis.
    """
    if isinstance(hyps[0], Hypothesis):
        transcriptions = []
        for hyp in hyps:
            transcriptions.append(hyp.text)
    else:
        transcriptions = hyps
    return transcriptions


def calc_drop_extra_pre_encoded(asr_model, step_num, pad_and_drop_preencoded):
    # for the first step there is no need to drop any tokens after the downsampling as no caching is being used
    if step_num == 0 and not pad_and_drop_preencoded:
        return 0
    else:
        return asr_model.encoder.streaming_cfg.drop_extra_pre_encoded


def perform_streaming(
    asr_model,
    streaming_buffer,
    compute_dtype: torch.dtype,
    compare_vs_offline=False,
    debug_mode=False,
    pad_and_drop_preencoded=False,
):
    batch_size = len(streaming_buffer.streams_length)
    if compare_vs_offline:
        # would pass the whole audio at once through the model like offline mode in order to compare the results with the stremaing mode
        # the output of the model in the offline and streaming mode should be exactly the same
        with torch.inference_mode():
            processed_signal, processed_signal_length = streaming_buffer.get_all_audios()
            processed_signal = processed_signal.to(compute_dtype)
            with torch.no_grad():
                (
                    pred_out_offline,
                    transcribed_texts,
                    cache_last_channel_next,
                    cache_last_time_next,
                    cache_last_channel_len,
                    best_hyp,
                ) = asr_model.conformer_stream_step(
                    processed_signal=processed_signal,
                    processed_signal_length=processed_signal_length,
                    return_transcription=True,
                )
        final_offline_tran = extract_transcriptions(transcribed_texts)
        logging.info(f" Final offline transcriptions:   {final_offline_tran}")
    else:
        final_offline_tran = None

    cache_last_channel, cache_last_time, cache_last_channel_len = asr_model.encoder.get_initial_cache_state(
        batch_size=batch_size
    )

    previous_hypotheses = None
    streaming_buffer_iter = iter(streaming_buffer)
    pred_out_stream = None
    for step_num, (chunk_audio, chunk_lengths) in enumerate(streaming_buffer_iter):
        with torch.inference_mode():
            # keep_all_outputs needs to be True for the last step of streaming when model is trained with att_context_style=regular
            # otherwise the last outputs would get dropped
            chunk_audio = chunk_audio.to(compute_dtype)
            with torch.no_grad():
                (
                    pred_out_stream,
                    transcribed_texts,
                    cache_last_channel,
                    cache_last_time,
                    cache_last_channel_len,
                    previous_hypotheses,
                ) = asr_model.conformer_stream_step(
                    processed_signal=chunk_audio,
                    processed_signal_length=chunk_lengths,
                    cache_last_channel=cache_last_channel,
                    cache_last_time=cache_last_time,
                    cache_last_channel_len=cache_last_channel_len,
                    keep_all_outputs=streaming_buffer.is_buffer_empty(),
                    previous_hypotheses=previous_hypotheses,
                    previous_pred_out=pred_out_stream,
                    drop_extra_pre_encoded=calc_drop_extra_pre_encoded(asr_model, step_num, pad_and_drop_preencoded),
                    return_transcription=True,
                )

        if debug_mode:
            logging.info(f"Streaming transcriptions: {extract_transcriptions(transcribed_texts)}")

    final_streaming_tran = extract_transcriptions(transcribed_texts)
    logging.info(f"Final streaming transcriptions: {final_streaming_tran}")

    if compare_vs_offline:
        # calculates and report the differences between the predictions of the model in offline mode vs streaming mode
        # Normally they should be exactly the same predictions for streaming models
        pred_out_stream_cat = torch.cat(pred_out_stream)
        pred_out_offline_cat = torch.cat(pred_out_offline)
        if pred_out_stream_cat.size() == pred_out_offline_cat.size():
            diff_num = torch.sum(pred_out_stream_cat != pred_out_offline_cat).cpu().numpy()
            logging.info(
                f"Found {diff_num} differences in the outputs of the model in streaming mode vs offline mode."
            )
        else:
            logging.info(
                f"The shape of the outputs of the model in streaming mode ({pred_out_stream_cat.size()}) is different from offline mode ({pred_out_offline_cat.size()})."
            )

    return final_streaming_tran, final_offline_tran


@hydra_runner(config_name="TranscriptionConfig", schema=TranscriptionConfig)
def main(cfg: TranscriptionConfig):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')
    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision(cfg.matmul_precision)
    cfg = OmegaConf.structured(cfg)
    if cfg.random_seed:
        pl.seed_everything(cfg.random_seed)

    # setup device
    device = get_inference_device(cuda=cfg.cuda, allow_mps=cfg.allow_mps)

    if (cfg.compute_dtype is not None and cfg.compute_dtype != "float32") and cfg.amp:
        raise ValueError("amp=true is mutually exclusive with a compute_dtype other than float32")

    amp_dtype = torch.float16 if cfg.amp_dtype == "float16" else torch.bfloat16

    compute_dtype: torch.dtype
    if cfg.amp:
        # with amp model weights required to be in float32
        compute_dtype = torch.float32
    else:
        compute_dtype = get_inference_dtype(compute_dtype=cfg.compute_dtype, device=device)

    if compute_dtype != torch.float32:
        # NB: cache-aware models do not currently work with compute_dtype != float32
        # since in some layers output is force-casted to float32
        # TODO(vbataev): implement support in future; set `compute_dtype` in config to None by default
        raise NotImplementedError(
            f"Compute dtype {compute_dtype} is not yet supported for cache-aware models, use float32 instead"
        )

    if sum((cfg.audio_file is not None, cfg.dataset_manifest is not None, cfg.audio_dir is not None)) != 1:
        raise ValueError("Exactly one of the `audio_file`, `dataset_manifest` or `audio_dir` should be non-empty!")

    asr_model, model_name = setup_model(cfg=cfg, map_location=device)

    logging.info(asr_model.encoder.streaming_cfg)
    if cfg.att_context_size is not None:
        if hasattr(asr_model.encoder, "set_default_att_context_size"):
            asr_model.encoder.set_default_att_context_size(att_context_size=cfg.att_context_size)
        else:
            raise ValueError("Model does not support multiple lookaheads.")

    # Setup decoding strategy
    if hasattr(asr_model, 'change_decoding_strategy') and hasattr(asr_model, 'decoding'):
        if cfg.decoder_type is not None:
            decoding_cfg = cfg.rnnt_decoding if cfg.decoder_type == 'rnnt' else cfg.ctc_decoding

            if hasattr(asr_model, 'cur_decoder'):
                asr_model.change_decoding_strategy(decoding_cfg, decoder_type=cfg.decoder_type)
            else:
                asr_model.change_decoding_strategy(decoding_cfg)

        # Check if ctc or rnnt model
        elif hasattr(asr_model, 'joint'):  # RNNT model
            cfg.rnnt_decoding.fused_batch_size = -1
            if hasattr(asr_model, 'cur_decoder'):
                asr_model.change_decoding_strategy(cfg.rnnt_decoding, decoder_type=cfg.decoder_type)
            else:
                asr_model.change_decoding_strategy(cfg.rnnt_decoding)
        else:
            asr_model.change_decoding_strategy(cfg.ctc_decoding)

    asr_model = asr_model.to(device=device, dtype=compute_dtype)
    asr_model.eval()

    # chunk_size is set automatically for models trained for streaming. For models trained for offline mode with full context, we need to pass the chunk_size explicitly.
    if cfg.chunk_size > 0:
        if cfg.shift_size < 0:
            shift_size = cfg.chunk_size
        else:
            shift_size = cfg.shift_size
        asr_model.encoder.setup_streaming_params(
            chunk_size=cfg.chunk_size, left_chunks=cfg.left_chunks, shift_size=shift_size
        )

    # In streaming, offline normalization is not feasible as we don't have access to the whole audio at the beginning
    # When online_normalization is enabled, the normalization of the input features (mel-spectrograms) are done per step
    # It is suggested to train the streaming models without any normalization in the input features.
    if cfg.online_normalization:
        if asr_model.cfg.preprocessor.normalize not in ["per_feature", "all_feature"]:
            logging.warning(
                "online_normalization is enabled but the model has no normalization in the feature extration part, so it is ignored."
            )
            online_normalization = False
        else:
            online_normalization = True

    else:
        online_normalization = False

    streaming_buffer = CacheAwareStreamingAudioBuffer(
        model=asr_model,
        online_normalization=online_normalization,
        pad_and_drop_preencoded=cfg.pad_and_drop_preencoded,
    )
    with torch.amp.autocast('cuda' if device.type == "cuda" else "cpu", dtype=amp_dtype, enabled=cfg.amp):
        if cfg.audio_file is not None:
            # stream a single audio file
            _ = streaming_buffer.append_audio_file(cfg.audio_file, stream_id=-1)
            perform_streaming(
                asr_model=asr_model,
                streaming_buffer=streaming_buffer,
                compute_dtype=compute_dtype,
                compare_vs_offline=cfg.compare_vs_offline,
                pad_and_drop_preencoded=cfg.pad_and_drop_preencoded,
            )
        else:
            # stream audio files in a manifest file in batched mode
            all_streaming_tran = []
            all_offline_tran = []
            all_refs_text = []
            batch_size = cfg.batch_size

            if cfg.dataset_manifest is not None:
                manifest_dir = Path(cfg.dataset_manifest).parent
                samples = read_manifest(cfg.dataset_manifest)
                # fix relative paths
                for item in samples:
                    audio_filepath = Path(item["audio_filepath"])
                    if not audio_filepath.is_absolute():
                        item["audio_filepath"] = str(manifest_dir / audio_filepath)

                logging.info(f"Loaded {len(samples)} from the manifest at {cfg.dataset_manifest}.")
                dataset_title = os.path.splitext(os.path.basename(cfg.dataset_manifest))[0]
            else:
                assert cfg.audio_dir is not None
                samples = [
                    {"audio_filepath": audio_filepath}
                    for audio_filepath in (
                        glob.glob(os.path.join(cfg.audio_dir, f"**/*.{cfg.audio_type}"), recursive=True)
                    )
                ]
                dataset_title = os.path.basename(cfg.audio_dir)

            start_time = time.time()
            for sample_idx, sample in enumerate(samples):
                _ = streaming_buffer.append_audio_file(sample['audio_filepath'], stream_id=-1)
                if "text" in sample:
                    all_refs_text.append(sample["text"])
                logging.info(f'Added this sample to the buffer: {sample["audio_filepath"]}')

                if (sample_idx + 1) % batch_size == 0 or sample_idx == len(samples) - 1:
                    logging.info(
                        f"Starting to stream samples {sample_idx - len(streaming_buffer) + 1} to {sample_idx}..."
                    )
                    streaming_tran, offline_tran = perform_streaming(
                        asr_model=asr_model,
                        streaming_buffer=streaming_buffer,
                        compute_dtype=compute_dtype,
                        compare_vs_offline=cfg.compare_vs_offline,
                        debug_mode=cfg.debug_mode,
                        pad_and_drop_preencoded=cfg.pad_and_drop_preencoded,
                    )
                    all_streaming_tran.extend(streaming_tran)
                    if cfg.compare_vs_offline:
                        all_offline_tran.extend(offline_tran)
                    streaming_buffer.reset_buffer()

        if cfg.compare_vs_offline and len(all_refs_text) == len(all_offline_tran):
            offline_wer = word_error_rate(hypotheses=all_offline_tran, references=all_refs_text)
            logging.info(f"WER% of offline mode: {round(offline_wer * 100, 2)}")
        if len(all_refs_text) == len(all_streaming_tran):
            streaming_wer = word_error_rate(hypotheses=all_streaming_tran, references=all_refs_text)
            logging.info(f"WER% of streaming mode: {round(streaming_wer*100, 2)}")

        end_time = time.time()
        logging.info(f"The whole streaming process took: {round(end_time - start_time, 2)}s")

        # stores the results including the transcriptions of the streaming inference in a json file
        if cfg.output_path is not None and len(all_refs_text) == len(all_streaming_tran):
            fname = "streaming_out_" + os.path.splitext(os.path.basename(model_name))[0] + f"_{dataset_title}.json"

            hyp_json = os.path.join(cfg.output_path, fname)
            os.makedirs(cfg.output_path, exist_ok=True)
            with open(hyp_json, "w") as out_f:
                for i, hyp in enumerate(all_streaming_tran):
                    record = {
                        "pred_text": hyp,
                        "text": all_refs_text[i],
                        "wer": round(word_error_rate(hypotheses=[hyp], references=[all_refs_text[i]]) * 100, 2),
                    }
                    out_f.write(json.dumps(record) + '\n')


if __name__ == '__main__':
    main()
