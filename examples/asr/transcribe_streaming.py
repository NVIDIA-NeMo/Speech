# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
Evaluate cache-aware streaming ASR models with comprehensive metrics (WER, CER, RTFx).

This script evaluates streaming ASR models and computes metrics similar to transcribe_speech_parallel.py
but specifically for cache-aware streaming models.

# Usage

## Basic evaluation on a manifest file:

python transcribe_streaming.py \
    model_path=asr_model.nemo \
    dataset_manifest=manifest_file.json \
    batch_size=16 \
    output_path=/output/dir/

## With WER/CER and RTFx calculation:

python transcribe_streaming.py \
    model_path=asr_model.nemo \
    dataset_manifest=manifest_file.json \
    batch_size=16 \
    output_path=/output/dir/ \
    calculate_wer=true \
    use_cer=false \
    calculate_rtfx=true \
    amp=true

## Compare streaming vs offline mode:

python transcribe_streaming.py \
    model_path=asr_model.nemo \
    dataset_manifest=manifest_file.json \
    batch_size=16 \
    compare_vs_offline=true \
    output_path=/output/dir/

## Evaluate a single audio file:

python transcribe_streaming.py \
    model_path=asr_model.nemo \
    audio_file=audio.wav \
    output_path=/output/dir/

## For models trained with full context (offline models):

python transcribe_streaming.py \
    pretrained_name=stt_en_conformer_ctc_large \
    chunk_size=100 \
    shift_size=50 \
    left_chunks=2 \
    online_normalization=true \
    dataset_manifest=manifest_file.json \
    batch_size=16 \
    output_path=/output/dir/

"""

import glob
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List

import lightning.pytorch as pl
import torch
import soundfile as sf
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
class StreamingTranscriptionConfig:
    """
    Configuration for cache-aware streaming transcription with evaluation metrics.
    """

    # Required configs
    model_path: Optional[str] = None  # Path to a .nemo file
    pretrained_name: Optional[str] = None  # Name of a pretrained model
    audio_dir: Optional[str] = None  # Path to a directory which contains audio files
    audio_type: str = "wav"  # type of audio file if audio_dir passed
    audio_file: Optional[str] = None  # Path to an audio file to perform streaming
    dataset_manifest: Optional[str] = None  # Path to dataset's JSON manifest
    output_path: Optional[str] = None  # Path to output directory

    # General configs
    batch_size: int = 32
    random_seed: Optional[int] = None

    # Chunked configs for offline models
    chunk_size: int = -1
    shift_size: int = -1
    left_chunks: Optional[int] = 2

    # Device configs
    cuda: Optional[int] = None
    allow_mps: bool = False
    amp: bool = False
    amp_dtype: str = "float16"  # "float16" or "bfloat16"
    compute_dtype: Optional[str] = None
    matmul_precision: str = "high"

    # Streaming specific configs
    compare_vs_offline: bool = False
    debug_mode: bool = False
    online_normalization: bool = False
    pad_and_drop_preencoded: bool = False

    # Decoding configs
    rnnt_decoding: RNNTDecodingConfig = field(default_factory=RNNTDecodingConfig)
    ctc_decoding: CTCDecodingConfig = field(default_factory=CTCDecodingConfig)
    decoder_type: Optional[str] = None

    # Evaluation metrics
    calculate_wer: bool = True
    use_cer: bool = False
    calculate_rtfx: bool = False
    clean_groundtruth_text: bool = False
    langid: str = "en"

    # Multi-lookahead support
    att_context_size: Optional[list] = None


def extract_transcriptions(transcribed_texts):
    """Extract text from transcription hypotheses."""
    if isinstance(transcribed_texts[0], Hypothesis):
        transcriptions = [transcribed_texts[i].text for i in range(len(transcribed_texts))]
    else:
        transcriptions = transcribed_texts
    return transcriptions


def calc_drop_extra_pre_encoded(asr_model, step_num, pad_and_drop_preencoded):
    """Calculate tokens to drop after downsampling."""
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
    """Perform streaming inference on buffered audio."""
    batch_size = len(streaming_buffer.streams_length)
    
    if compare_vs_offline:
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
        if debug_mode:
            logging.info(f"Final offline transcriptions: {final_offline_tran}")
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
    if debug_mode:
        logging.info(f"Final streaming transcriptions: {final_streaming_tran}")

    if compare_vs_offline:
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


@hydra_runner(config_name="StreamingTranscriptionConfig", schema=StreamingTranscriptionConfig)
def main(cfg: StreamingTranscriptionConfig):
    """
    Main function for streaming ASR evaluation with comprehensive metrics.
    """
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    if cfg.random_seed:
        pl.seed_everything(cfg.random_seed)

    if cfg.model_path is None and cfg.pretrained_name is None:
        raise ValueError("Either model_path or pretrained_name must be specified!")
    if cfg.audio_file is None and cfg.dataset_manifest is None and cfg.audio_dir is None:
        raise ValueError("One of audio_file, dataset_manifest, or audio_dir must be specified!")

    # Setup device
    torch.set_float32_matmul_precision(cfg.matmul_precision)
    device, map_location = get_inference_device(
        cuda=cfg.cuda,
        allow_mps=cfg.allow_mps,
    )

    logging.info(f"Inference will be done on device: {map_location}")

    # Load model
    asr_model, model_name = setup_model(cfg, map_location)
    asr_model = asr_model.eval()

    # Setup amp and compute dtype
    if (cfg.compute_dtype is not None and cfg.compute_dtype != "float32") and cfg.amp:
        raise ValueError("amp=true is mutually exclusive with compute_dtype other than float32")

    amp_dtype = torch.float16 if cfg.amp_dtype == "float16" else torch.bfloat16
    
    if cfg.amp:
        compute_dtype = torch.float32
    else:
        compute_dtype = get_inference_dtype(compute_dtype=cfg.compute_dtype, device=map_location)

    asr_model.to(compute_dtype)

    # Setup decoding strategy
    if hasattr(asr_model, 'change_decoding_strategy'):
        if cfg.decoder_type == 'rnnt':
            asr_model.change_decoding_strategy(cfg.rnnt_decoding)
        elif cfg.decoder_type == 'ctc':
            asr_model.change_decoding_strategy(cfg.ctc_decoding)
        elif hasattr(asr_model, 'joint'):  # RNNT model
            asr_model.change_decoding_strategy(cfg.rnnt_decoding)
        else:  # CTC model
            asr_model.change_decoding_strategy(cfg.ctc_decoding)

    # Setup att_context_size for multi-lookahead models
    if cfg.att_context_size and hasattr(asr_model.encoder, 'set_default_att_context_size'):
        asr_model.encoder.set_default_att_context_size(cfg.att_context_size)

    # Determine chunk_size and shift_size
    if hasattr(asr_model.encoder, 'streaming_cfg'):
        chunk_size = (
            cfg.chunk_size
            if cfg.chunk_size >= 0
            else asr_model.encoder.streaming_cfg.last_channel_cache_size
        )
        shift_size = cfg.shift_size if cfg.shift_size >= 0 else chunk_size
        if shift_size > chunk_size:
            raise ValueError(f"shift_size ({shift_size}) should be <= chunk_size ({chunk_size})")
        logging.info(f"Using chunk_size={chunk_size} and shift_size={shift_size}")
    else:
        if cfg.chunk_size < 0:
            raise ValueError("chunk_size must be specified for models without streaming_cfg")
        chunk_size = cfg.chunk_size
        shift_size = cfg.shift_size if cfg.shift_size >= 0 else chunk_size

    # Setup online normalization
    online_normalization = cfg.online_normalization
    if hasattr(asr_model.encoder, 'streaming_cfg'):
        if hasattr(asr_model.encoder.streaming_cfg, 'norm_before_caching'):
            if asr_model.encoder.streaming_cfg.norm_before_caching:
                online_normalization = True

    # Create streaming buffer
    model_stride_in_secs = asr_model.encoder.conv_context_size()[0]
    streaming_buffer = CacheAwareStreamingAudioBuffer(
        asr_model=asr_model,
        model_stride_in_secs=model_stride_in_secs,
        device=device,
        chunk_size=chunk_size,
        shift_size=shift_size,
        batch_size=cfg.batch_size,
        online_normalization=online_normalization,
        pad_and_drop_preencoded=cfg.pad_and_drop_preencoded,
    )

    # Process audio
    with torch.amp.autocast('cuda' if device.type == "cuda" else "cpu", dtype=amp_dtype, enabled=cfg.amp):
        if cfg.audio_file is not None:
            # Single audio file mode
            logging.info(f"Processing single audio file: {cfg.audio_file}")
            _ = streaming_buffer.append_audio_file(cfg.audio_file, stream_id=-1)
            streaming_tran, offline_tran = perform_streaming(
                asr_model=asr_model,
                streaming_buffer=streaming_buffer,
                compute_dtype=compute_dtype,
                compare_vs_offline=cfg.compare_vs_offline,
                pad_and_drop_preencoded=cfg.pad_and_drop_preencoded,
                debug_mode=cfg.debug_mode,
            )
            logging.info(f"Transcription: {streaming_tran[0]}")
        else:
            # Batch processing mode
            all_streaming_tran = []
            all_offline_tran = []
            all_refs_text = []
            all_audio_durations = []
            batch_size = cfg.batch_size

            # Load samples
            if cfg.dataset_manifest is not None:
                manifest_dir = Path(cfg.dataset_manifest).parent
                samples = read_manifest(cfg.dataset_manifest)
                # Fix relative paths
                for item in samples:
                    audio_filepath = Path(item["audio_filepath"])
                    if not audio_filepath.is_absolute():
                        item["audio_filepath"] = str(manifest_dir / audio_filepath)
                logging.info(f"Loaded {len(samples)} samples from {cfg.dataset_manifest}")
                dataset_title = os.path.splitext(os.path.basename(cfg.dataset_manifest))[0]
            else:
                samples = [
                    {"audio_filepath": audio_filepath}
                    for audio_filepath in (
                        glob.glob(os.path.join(cfg.audio_dir, f"**/*.{cfg.audio_type}"), recursive=True)
                    )
                ]
                dataset_title = os.path.basename(cfg.audio_dir)
                logging.info(f"Found {len(samples)} audio files in {cfg.audio_dir}")

            start_time = time.time()
            total_audio_duration = 0.0

            for sample_idx, sample in enumerate(samples):
                _ = streaming_buffer.append_audio_file(sample['audio_filepath'], stream_id=-1)
                
                # Get audio duration for RTFx calculation
                if cfg.calculate_rtfx or "duration" not in sample:
                    try:
                        audio_info = sf.info(sample['audio_filepath'])
                        duration = audio_info.duration
                    except Exception as e:
                        logging.warning(f"Could not get duration for {sample['audio_filepath']}: {e}")
                        duration = 0.0
                else:
                    duration = sample.get("duration", 0.0)
                
                all_audio_durations.append(duration)
                total_audio_duration += duration

                if "text" in sample:
                    all_refs_text.append(sample["text"])
                
                if cfg.debug_mode:
                    logging.info(f'Added sample to buffer: {sample["audio_filepath"]}')

                if (sample_idx + 1) % batch_size == 0 or sample_idx == len(samples) - 1:
                    logging.info(
                        f"Streaming samples {sample_idx - len(streaming_buffer) + 1} to {sample_idx}..."
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

            end_time = time.time()
            total_processing_time = end_time - start_time
            logging.info(f"Total processing time: {round(total_processing_time, 2)}s")

            # Calculate metrics
            if cfg.compare_vs_offline and len(all_refs_text) == len(all_offline_tran):
                offline_wer = word_error_rate(hypotheses=all_offline_tran, references=all_refs_text, use_cer=cfg.use_cer)
                metric_name = "CER" if cfg.use_cer else "WER"
                logging.info(f"{metric_name}% of offline mode: {round(offline_wer * 100, 2)}")

            if len(all_refs_text) == len(all_streaming_tran):
                streaming_wer = word_error_rate(hypotheses=all_streaming_tran, references=all_refs_text, use_cer=cfg.use_cer)
                metric_name = "CER" if cfg.use_cer else "WER"
                logging.info(f"{metric_name}% of streaming mode: {round(streaming_wer * 100, 2)}")
            elif cfg.calculate_wer:
                logging.warning(f"Cannot calculate {metric_name}: number of predictions ({len(all_streaming_tran)}) != number of references ({len(all_refs_text)})")

            # Calculate RTFx
            if cfg.calculate_rtfx and total_audio_duration > 0:
                # RTFx = Real-Time Factor = audio_duration / processing_time
                # RTFx > 1 means faster than real-time (good!)
                # RTFx < 1 means slower than real-time (bad)
                rtfx = total_audio_duration / total_processing_time
                logging.info(f"RTFx (Real-Time Factor): {round(rtfx, 4)}")
                logging.info(f"Throughput: {round(rtfx, 2)}x faster than real-time")

            # Save results
            if cfg.output_path is not None and len(all_streaming_tran) > 0:
                os.makedirs(cfg.output_path, exist_ok=True)
                fname = "streaming_" + os.path.splitext(os.path.basename(model_name))[0] + f"_{dataset_title}.json"
                output_file = os.path.join(cfg.output_path, fname)
                
                with open(output_file, "w") as out_f:
                    for i, hyp in enumerate(all_streaming_tran):
                        record = {
                            "pred_text": hyp,
                            "audio_filepath": samples[i]["audio_filepath"],
                        }
                        if i < len(all_refs_text):
                            record["text"] = all_refs_text[i]
                            record["wer"] = round(
                                word_error_rate(hypotheses=[hyp], references=[all_refs_text[i]], use_cer=cfg.use_cer) * 100, 2
                            )
                        if i < len(all_audio_durations):
                            record["duration"] = all_audio_durations[i]
                        if cfg.compare_vs_offline and i < len(all_offline_tran):
                            record["offline_pred_text"] = all_offline_tran[i]
                        
                        out_f.write(json.dumps(record) + '\n')
                
                logging.info(f"Results saved to: {output_file}")

                # Save summary
                summary_file = os.path.join(cfg.output_path, f"summary_{dataset_title}.json")
                summary = {
                    "model": model_name,
                    "dataset": dataset_title,
                    "num_samples": len(all_streaming_tran),
                    "total_audio_duration": round(total_audio_duration, 2),
                    "total_processing_time": round(total_processing_time, 2),
                }
                
                if len(all_refs_text) == len(all_streaming_tran):
                    metric_name = "cer" if cfg.use_cer else "wer"
                    summary[f"streaming_{metric_name}"] = round(streaming_wer * 100, 2)
                    
                if cfg.compare_vs_offline and len(all_refs_text) == len(all_offline_tran):
                    summary[f"offline_{metric_name}"] = round(offline_wer * 100, 2)
                    
                if cfg.calculate_rtfx and total_audio_duration > 0:
                    summary["rtfx"] = round(rtfx, 4)
                    summary["throughput_vs_realtime"] = round(1.0 / rtfx, 2)
                
                with open(summary_file, "w") as f:
                    json.dump(summary, f, indent=2)
                
                logging.info(f"Summary saved to: {summary_file}")


if __name__ == '__main__':
    main()
