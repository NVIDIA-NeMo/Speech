# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
Used in inference and evaluation scripts to obtain metrics such as ASR_WER and UTMOSV2 scores.
"""
import argparse
import json
import os
import pprint
import string
import tempfile
import time
from functools import partial

import librosa
import numpy as np
import soundfile as sf
import torch
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector, WhisperForConditionalGeneration, WhisperProcessor

import nemo.collections.asr as nemo_asr
from nemo.collections.asr.metrics.wer import word_error_rate_detail
from nemo.collections.tts.metrics.frechet_codec_distance import FrechetCodecDistance
from nemo.utils import logging

# Optional import for UTMOSv2 (audio quality metric)
try:
    from nemo.collections.tts.modules.utmosv2 import UTMOSv2Calculator

    UTMOSV2_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as e:
    UTMOSV2_AVAILABLE = False
    logging.warning(
        f"UTMOSv2Calculator not available: {e}. "
        "UTMOSv2 metrics will be disabled. Install required dependencies to enable."
        "To install utmosv2 run `pip install git+https://github.com/sarulab-speech/UTMOSv2.git@v1.2.1`."
    )


def load_evalset_config(config_path: str = None) -> dict:
    """Load dataset meta info from JSON config file."""
    if config_path is None or not os.path.exists(config_path):
        raise ValueError("No dataset_json_path provided, please provide a valid path to the evalset config file.")
    logging.info(f"Loading evalset config from {config_path}")
    with open(config_path, 'r') as f:
        return json.load(f)


def find_generated_files(audio_dir, prefix, extension):
    file_list = []
    for f in os.listdir(audio_dir):
        if prefix in f and f.endswith(extension):
            audio_number = int(f.split("_")[-1].split(extension)[0])
            file_list.append((audio_number, os.path.join(audio_dir, f)))
    file_list.sort()
    file_list = [t[1] for t in file_list]
    return file_list


def find_generated_audio_files(audio_dir):
    return find_generated_files(audio_dir=audio_dir, prefix="predicted_audio", extension=".wav")


def find_generated_codec_files(audio_dir):
    return find_generated_files(audio_dir=audio_dir, prefix="predicted_codes", extension=".pt")


def get_wav_file_duration(audio_path: str) -> float:
    """
    Get the duration of an WAV file in seconds.
    """
    # get extension of the file
    extension = os.path.splitext(audio_path)[1]
    if extension.lower() != ".wav":
        raise ValueError(f"Audio path {audio_path} is not a WAV file")
    info = sf.info(audio_path)
    seconds = info.frames / info.samplerate
    return seconds


def read_manifest(manifest_path):
    records = []
    with open(manifest_path, 'r') as f:
        all_lines = f.readlines()
        for line in all_lines:
            line = line.strip()
            records.append(json.loads(line))
    return records


def process_text(input_text):
    # Convert text to lowercase
    lower_case_text = input_text.lower()

    # Remove commas from text
    no_comma_text = lower_case_text.replace(",", "")

    # Replace "-" with spaces
    no_dash_text = no_comma_text.replace("-", " ")

    # Replace double spaces with single space
    single_space_text = " ".join(no_dash_text.split())

    single_space_text = single_space_text.translate(str.maketrans('', '', string.punctuation))

    return single_space_text


def transcribe_with_whisper(whisper_model, whisper_processor, audio_path, language, device):
    speech_array, sampling_rate = librosa.load(audio_path, sr=16000)
    # Set the language task (optional, improves performance for specific languages)
    forced_decoder_ids = (
        whisper_processor.get_decoder_prompt_ids(language=language, task="transcribe") if language else None
    )
    inputs = whisper_processor(speech_array, sampling_rate=sampling_rate, return_tensors="pt").input_features
    inputs = inputs.to(device)
    # Generate transcription
    with torch.inference_mode():
        predicted_ids = whisper_model.generate(inputs, forced_decoder_ids=forced_decoder_ids)

    # Decode transcription
    transcription = whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)
    result = transcription[0]
    return result


def transcribe_with_whisper_batch(whisper_model, whisper_processor, audio_paths, language, device, batch_size=8):
    """Transcribe multiple audio files with Whisper in batches. Returns list of transcriptions (one per path)."""
    forced_decoder_ids = (
        whisper_processor.get_decoder_prompt_ids(language=language, task="transcribe") if language else None
    )
    all_transcriptions = []
    for start in range(0, len(audio_paths), batch_size):
        batch_paths = audio_paths[start : start + batch_size]
        speech_arrays = [librosa.load(p, sr=16000)[0] for p in batch_paths]
        inputs = whisper_processor(
            speech_arrays, sampling_rate=16000, return_tensors="pt", padding=True
        ).input_features
        inputs = inputs.to(device)
        with torch.inference_mode():
            predicted_ids = whisper_model.generate(inputs, forced_decoder_ids=forced_decoder_ids)
        transcriptions = whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)
        all_transcriptions.extend(transcriptions)
    return all_transcriptions


def pad_audio_to_min_length(audio_np: np.ndarray, sampling_rate: int, min_seconds: float) -> np.ndarray:
    """
    Pad audio to make it at least `min_seconds` long by adding silence at the end if needed.
    """
    if audio_np.ndim != 1:
        raise ValueError("Audio array must be 1D")

    n_samples = len(audio_np)
    min_samples = round(min_seconds * sampling_rate)

    if n_samples < min_samples:
        logging.info(f"Padding audio from {n_samples/sampling_rate} seconds to {min_samples/sampling_rate} seconds")
        padding_needed = min_samples - n_samples
        audio_np = np.pad(audio_np, (0, padding_needed), mode='constant', constant_values=0)
    return audio_np


def extract_embedding(model, extractor, audio_path, device, sv_model_type):
    speech_array, sampling_rate = librosa.load(audio_path, sr=16000)
    # pad to 0.5 seconds as the extractor may not be able to handle very short signals
    speech_array = pad_audio_to_min_length(speech_array, int(sampling_rate), min_seconds=0.5)
    if sv_model_type == "wavlm":
        inputs = extractor(speech_array, sampling_rate=sampling_rate, return_tensors="pt").input_values.to(device)
        with torch.inference_mode():
            embeddings = model(inputs).embeddings
    else:  # Titanet
        with tempfile.NamedTemporaryFile(suffix=".wav") as temp_file:
            # the embedding model doesn't accept NumPy arrays, so we write to a temporary file
            sf.write(temp_file.name, speech_array, samplerate=16000)
            with torch.inference_mode():
                embeddings = model.get_embedding(temp_file.name).squeeze()

    return embeddings.squeeze()


def compute_utmosv2_scores(audio_dir, device):
    if not UTMOSV2_AVAILABLE:
        logging.warning("UTMOSv2Calculator not available. Skipping UTMOSv2 score computation.")
        return {}

    logging.info(f"\nComputing UTMOSv2 scores for files in {audio_dir}...")
    start_time = time.time()
    utmosv2_calculator = UTMOSv2Calculator(device=device)
    utmosv2_scores = utmosv2_calculator.process_directory(audio_dir)
    # convert to to a dictionary indexed by file path
    utmosv2_scores_dict = {os.path.normpath(item['file_path']): item['predicted_mos'] for item in utmosv2_scores}
    end_time = time.time()
    logging.info(f"UTMOSv2 scores computed for {len(utmosv2_scores)} files in {end_time - start_time:.2f} seconds\n")
    return utmosv2_scores_dict


def evaluate(
    manifest_path,
    audio_dir,
    generated_audio_dir,
    language="en",
    sv_model_type="titanet",
    asr_model_name="stt_en_conformer_transducer_large",
    with_utmosv2=True,
    with_fcd=True,
    codec_model_path=None,
    asr_batch_size=32,
):
    logging.info(f"Evaluating generated audio in {generated_audio_dir}...")

    # Timing collection for profiling
    from collections import defaultdict

    timing_stats = defaultdict(lambda: {'total': 0.0, 'count': 0})

    def record_time(section_name, elapsed):
        timing_stats[section_name]['total'] += elapsed
        timing_stats[section_name]['count'] += 1

    eval_start_time = time.time()

    # Time file discovery and manifest reading
    t0 = time.time()
    audio_file_lists = find_generated_audio_files(generated_audio_dir)
    records = read_manifest(manifest_path)
    assert len(audio_file_lists) == len(records)
    if with_fcd:
        if codec_model_path is None:
            raise ValueError("codec_model_path is required when with_fcd is True")
        codes_file_lists = find_generated_codec_files(generated_audio_dir)
        assert len(codes_file_lists) == len(records)
    record_time('file_discovery', time.time() - t0)

    device = "cuda"

    # Time model loading
    t0 = time.time()
    whisper_processor = None  # Address CodeQL issue even though this variable is only used when language != "en"
    utmosv2_scores = None  # Address CodeQL issue even though this variable is only used when with_utmosv2 is true
    if language == "en":
        if asr_model_name.startswith("nvidia/") or asr_model_name in ["stt_en_conformer_transducer_large"]:
            asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=asr_model_name)
        else:
            raise ValueError(f"ASR model {asr_model_name} not supported")
        asr_model = asr_model.to(device)
        asr_model.eval()
    else:
        whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
        whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")
        whisper_model = whisper_model.to(device)
        whisper_model.eval()
    record_time('load_asr_model', time.time() - t0)

    t0 = time.time()
    if sv_model_type == "wavlm":
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('microsoft/wavlm-base-plus-sv')
        speaker_verification_model = WavLMForXVector.from_pretrained('microsoft/wavlm-base-plus-sv').to(device).eval()
    else:
        feature_extractor = None
        speaker_verification_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
            model_name='titanet_large'
        )
        speaker_verification_model = speaker_verification_model.to(device)
        speaker_verification_model.eval()

    logging.info("Loading `titanet_small` model...")
    # The model `titanet_small` prints thousands of lines during initialization, so suppress logs temporarily
    with logging.temp_verbosity(logging.ERROR):
        speaker_verification_model_alternate = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
            model_name='titanet_small'
        )
    speaker_verification_model_alternate = speaker_verification_model_alternate.to(device)
    speaker_verification_model_alternate.eval()
    record_time('load_speaker_models', time.time() - t0)

    t0 = time.time()
    if with_fcd:
        fcd_metric = FrechetCodecDistance(codec_name=codec_model_path).to(device)
    else:
        fcd_metric = None
    record_time('load_fcd_model', time.time() - t0)

    t0 = time.time()
    if with_utmosv2:
        if not UTMOSV2_AVAILABLE:
            logging.warning(
                "UTMOSv2 was requested (with_utmosv2=True) but the UTMOSv2 library is not available. "
                "UTMOSv2 scores will be set to NaN for all files."
            )
        utmosv2_scores = compute_utmosv2_scores(generated_audio_dir, device)
    record_time('compute_utmosv2', time.time() - t0)

    # Resolve ground-truth audio paths for all records (for batched ASR)
    resolved_gt_audio_paths = []
    for record in records:
        p = record.get('audio_filepath', None)
        if audio_dir is not None and p is not None:
            p = os.path.join(audio_dir, p)
        resolved_gt_audio_paths.append(p)

    # Batched ASR stage: transcribe all predicted and all gt audios in batches
    t0 = time.time()
    pred_texts = []
    if language == "en":
        for start in range(0, len(audio_file_lists), asr_batch_size):
            batch_paths = audio_file_lists[start : start + asr_batch_size]
            try:
                with torch.inference_mode():
                    batch_results = asr_model.transcribe(batch_paths, batch_size=len(batch_paths), use_lhotse=False)
                for r in batch_results:
                    pred_texts.append(process_text(r.text))
            except Exception as e:
                logging.info("Error during batched ASR (predicted): {}".format(e))
                pred_texts.extend([""] * len(batch_paths))
    else:
        try:
            raw_pred = transcribe_with_whisper_batch(
                whisper_model, whisper_processor, audio_file_lists, language, device, batch_size=asr_batch_size
            )
            pred_texts = [process_text(t) for t in raw_pred]
        except Exception as e:
            logging.info("Error during batched ASR (predicted): {}".format(e))
            pred_texts = [""] * len(audio_file_lists)

    gt_audio_texts = [None] * len(records)
    gt_indices_and_paths = [(i, p) for i, p in enumerate(resolved_gt_audio_paths) if p is not None]
    if gt_indices_and_paths:
        indices, gt_paths = zip(*gt_indices_and_paths)
        indices, gt_paths = list(indices), list(gt_paths)
        if language == "en":
            for start in range(0, len(gt_paths), asr_batch_size):
                batch_paths = gt_paths[start : start + asr_batch_size]
                batch_indices = indices[start : start + asr_batch_size]
                try:
                    with torch.inference_mode():
                        batch_results = asr_model.transcribe(
                            batch_paths, batch_size=len(batch_paths), use_lhotse=False
                        )
                    for idx, r in zip(batch_indices, batch_results):
                        gt_audio_texts[idx] = process_text(r.text)
                except Exception as e:
                    logging.info("Error during batched ASR (gt audio): {}".format(e))
                    for idx in batch_indices:
                        gt_audio_texts[idx] = ""
        else:
            try:
                raw_gt = transcribe_with_whisper_batch(
                    whisper_model, whisper_processor, gt_paths, language, device, batch_size=asr_batch_size
                )
                for idx, t in zip(indices, raw_gt):
                    gt_audio_texts[idx] = process_text(t)
            except Exception as e:
                logging.info("Error during batched ASR (gt audio): {}".format(e))
                for idx in indices:
                    gt_audio_texts[idx] = ""
    record_time('asr_transcription', time.time() - t0)

    filewise_metrics = []
    gt_texts = []
    total_generated_audio_seconds = 0.0
    for ridx, record in enumerate(records):
        gt_audio_filepath = record.get('audio_filepath', None)
        context_audio_filepath = record.get('context_audio_filepath', None)
        if audio_dir is not None and gt_audio_filepath is not None:
            gt_audio_filepath = os.path.join(audio_dir, gt_audio_filepath)
            if context_audio_filepath is not None:
                context_audio_filepath = os.path.join(audio_dir, context_audio_filepath)

        # Update the FCD metric with real (ground truth) codes
        t0 = time.time()
        if fcd_metric is not None:
            fcd_metric.update_from_audio_file(gt_audio_filepath, True)
        record_time('fcd_update_gt', time.time() - t0)

        pred_audio_filepath = audio_file_lists[ridx]
        pred_text = pred_texts[ridx]
        gt_audio_text = gt_audio_texts[ridx]

        if with_utmosv2 and UTMOSV2_AVAILABLE:
            utmosv2_score = utmosv2_scores[os.path.normpath(pred_audio_filepath)]
        else:
            utmosv2_score = float('nan')

        if "original_text" in record:
            gt_text = process_text(record['original_text'])
        elif 'normalized_text' in record:
            gt_text = process_text(record['normalized_text'])
        else:
            gt_text = process_text(record['text'])

        t0 = time.time()
        detailed_cer = word_error_rate_detail(hypotheses=[pred_text], references=[gt_text], use_cer=True)
        detailed_wer = word_error_rate_detail(hypotheses=[pred_text], references=[gt_text], use_cer=False)
        record_time('wer_cer_computation', time.time() - t0)

        logging.info(f"{ridx} GT Text: {gt_text}")
        logging.info(f"{ridx} Pr Text: {pred_text}")
        # Format cer and wer to 2 decimal places
        logging.info(f"CER: {detailed_cer[0]:.4f} | WER: {detailed_wer[0]:.4f}")

        gt_texts.append(gt_text)

        # Update FCD metric with generated codes
        t0 = time.time()
        if fcd_metric is not None:
            predicted_codes = torch.load(codes_file_lists[ridx]).unsqueeze(0).to(device)  # B, C, T
            predicted_codes_lens = torch.tensor([predicted_codes.size(-1)], dtype=torch.int, device=device)
            fcd_metric.update(predicted_codes, predicted_codes_lens, False)
        record_time('fcd_update_pred', time.time() - t0)

        pred_context_ssim = 0.0
        gt_context_ssim = 0.0
        t0 = time.time()
        with torch.inference_mode():
            extract_embedding_fn = partial(
                extract_embedding,
                model=speaker_verification_model,
                extractor=feature_extractor,
                device=device,
                sv_model_type=sv_model_type,
            )
            extract_embedding_fn_alternate = partial(
                extract_embedding,
                model=speaker_verification_model_alternate,
                extractor=feature_extractor,
                device=device,
                sv_model_type=sv_model_type,
            )

            # Initialize SSIMs with a default since the context or ground truth audio
            # may be unavailable.
            pred_context_ssim = float('NaN')
            gt_context_ssim = float('NaN')
            pred_context_ssim_alternate = float('NaN')
            gt_context_ssim_alternate = float('NaN')
            pred_gt_ssim = float('NaN')
            pred_gt_ssim_alternate = float('NaN')

            if gt_audio_filepath is not None:
                # Ground truth vs. predicted
                gt_speaker_embedding = extract_embedding_fn(audio_path=gt_audio_filepath)
                pred_speaker_embedding = extract_embedding_fn(audio_path=pred_audio_filepath)
                pred_gt_ssim = torch.nn.functional.cosine_similarity(
                    gt_speaker_embedding, pred_speaker_embedding, dim=0
                ).item()

                # Ground truth vs. predicted (alternate model)
                gt_speaker_embedding_alternate = extract_embedding_fn_alternate(audio_path=gt_audio_filepath)
                pred_speaker_embedding_alternate = extract_embedding_fn_alternate(audio_path=pred_audio_filepath)
                pred_gt_ssim_alternate = torch.nn.functional.cosine_similarity(
                    gt_speaker_embedding_alternate, pred_speaker_embedding_alternate, dim=0
                ).item()

            if context_audio_filepath is not None:
                context_speaker_embedding = extract_embedding_fn(audio_path=context_audio_filepath)
                context_speaker_embedding_alternate = extract_embedding_fn_alternate(audio_path=context_audio_filepath)

                # Predicted vs. context
                pred_context_ssim = torch.nn.functional.cosine_similarity(
                    pred_speaker_embedding, context_speaker_embedding, dim=0
                ).item()
                # Ground truth vs. context
                if gt_audio_filepath is not None:
                    gt_context_ssim = torch.nn.functional.cosine_similarity(
                        gt_speaker_embedding, context_speaker_embedding, dim=0
                    ).item()

                # Predicted vs. context (alternate model)
                pred_context_ssim_alternate = torch.nn.functional.cosine_similarity(
                    pred_speaker_embedding_alternate, context_speaker_embedding_alternate, dim=0
                ).item()
                # Ground truth vs. context (alternate model)
                if gt_audio_filepath is not None:
                    gt_context_ssim_alternate = torch.nn.functional.cosine_similarity(
                        gt_speaker_embedding_alternate, context_speaker_embedding_alternate, dim=0
                    ).item()
            file_duration = get_wav_file_duration(pred_audio_filepath)
            total_generated_audio_seconds += file_duration
        record_time('speaker_embedding', time.time() - t0)

        filewise_metrics.append(
            {
                'gt_text': gt_text,
                'pred_text': pred_text,
                'gt_audio_text': gt_audio_text,
                'detailed_cer': detailed_cer,
                'detailed_wer': detailed_wer,
                'cer': detailed_cer[0],
                'wer': detailed_wer[0],
                'pred_gt_ssim': pred_gt_ssim,
                'pred_context_ssim': pred_context_ssim,
                'gt_context_ssim': gt_context_ssim,
                'pred_gt_ssim_alternate': pred_gt_ssim_alternate,
                'pred_context_ssim_alternate': pred_context_ssim_alternate,
                'gt_context_ssim_alternate': gt_context_ssim_alternate,
                'gt_audio_filepath': gt_audio_filepath,
                'pred_audio_filepath': pred_audio_filepath,
                'context_audio_filepath': context_audio_filepath,
                'utmosv2': utmosv2_score,
                'total_gen_audio_seconds': file_duration,
            }
        )

    # compute frechet distance for the whole dataset
    t0 = time.time()
    if fcd_metric is not None:
        fcd = fcd_metric.compute().cpu().item()
        fcd_metric.reset()
    else:
        fcd = float('nan')
    record_time('fcd_compute_final', time.time() - t0)

    # Compute global/aggregate metrics using the shared helper (also used by
    # chunked scoring aggregation). FCD was already computed above, so we pass
    # it in directly rather than recomputing.
    t0 = time.time()
    avg_metrics = compute_global_metrics(filewise_metrics=filewise_metrics)
    # Override FCD with the value computed inline (more efficient than recomputing)
    avg_metrics["frechet_codec_distance"] = fcd
    record_time('metrics_aggregation', time.time() - t0)

    total_eval_time = time.time() - eval_start_time

    # Print timing breakdown
    logging.info("\n" + "=" * 60)
    logging.info("EVALUATION TIMING BREAKDOWN")
    logging.info("=" * 60)
    logging.info(f"{'Section':<30} {'Total (s)':<12} {'Count':<8} {'Avg (s)':<12} {'% of Total':<10}")
    logging.info("-" * 60)

    # Sort by total time descending
    sorted_stats = sorted(timing_stats.items(), key=lambda x: x[1]['total'], reverse=True)
    for section, stats in sorted_stats:
        total_time = stats['total']
        count = stats['count']
        avg_time = total_time / count if count > 0 else 0
        pct = (total_time / total_eval_time * 100) if total_eval_time > 0 else 0
        logging.info(f"{section:<30} {total_time:<12.2f} {count:<8} {avg_time:<12.4f} {pct:<10.1f}%")

    logging.info("-" * 60)
    logging.info(f"{'TOTAL EVALUATION TIME':<30} {total_eval_time:<12.2f}")
    logging.info("=" * 60 + "\n")

    # filewise_metrics is in original manifest order so callers can safely
    # map filewise_metrics[i] back to input record[i]. Callers that want
    # sorted output for human readability should sort themselves.
    return avg_metrics, filewise_metrics


def compute_fcd(gt_audio_paths, predicted_codes_paths, codec_model_path):
    """Compute Frechet Codec Distance from ground-truth audio paths and predicted codec codes paths.

    Args:
        gt_audio_paths: List of paths to ground-truth audio files.
        predicted_codes_paths: List of paths to predicted codec codes (.pt) files.
        codec_model_path: Path or name of the codec model for FCD computation.

    Returns:
        FCD score (float).
    """
    device = "cuda"
    fcd_metric = FrechetCodecDistance(codec_name=codec_model_path).to(device)
    for gt_path, codes_path in zip(gt_audio_paths, predicted_codes_paths):
        fcd_metric.update_from_audio_file(gt_path, True)
        predicted_codes = torch.load(codes_path).unsqueeze(0).to(device)  # B, C, T
        predicted_codes_lens = torch.tensor([predicted_codes.size(-1)], dtype=torch.int, device=device)
        fcd_metric.update(predicted_codes, predicted_codes_lens, False)
    fcd = fcd_metric.compute().cpu().item()
    fcd_metric.reset()
    return fcd


def compute_global_metrics(
    filewise_metrics,
    gt_audio_paths=None,
    predicted_codes_paths=None,
    codec_model_path=None,
):
    """Recompute global/aggregate metrics from per-file results.

    Used by the aggregation step after chunked scoring to produce correct
    global metrics (cumulative WER/CER, FCD) from per-file data collected
    across all chunks.

    Args:
        filewise_metrics: List of per-file metric dicts. Each must contain at least
            'pred_text', 'gt_text', 'cer', 'wer', 'pred_gt_ssim', 'pred_context_ssim',
            'gt_context_ssim', 'utmosv2'.
        gt_audio_paths: Optional list of ground-truth audio paths for FCD computation.
        predicted_codes_paths: Optional list of predicted codec codes paths for FCD computation.
        codec_model_path: Optional codec model path/name for FCD computation.

    Returns:
        dict of global metrics (same keys as evaluate() avg_metrics).
    """
    n = len(filewise_metrics)
    pred_texts = [m['pred_text'] for m in filewise_metrics]
    gt_texts = [m['gt_text'] for m in filewise_metrics]

    avg_metrics = {}
    avg_metrics['cer_filewise_avg'] = sum(m['cer'] for m in filewise_metrics) / n
    avg_metrics['wer_filewise_avg'] = sum(m['wer'] for m in filewise_metrics) / n
    avg_metrics['cer_cumulative'] = word_error_rate_detail(hypotheses=pred_texts, references=gt_texts, use_cer=True)[0]
    avg_metrics['wer_cumulative'] = word_error_rate_detail(hypotheses=pred_texts, references=gt_texts, use_cer=False)[
        0
    ]
    avg_metrics['ssim_pred_gt_avg'] = sum(m.get('pred_gt_ssim', float('nan')) for m in filewise_metrics) / n
    avg_metrics['ssim_pred_context_avg'] = sum(m.get('pred_context_ssim', float('nan')) for m in filewise_metrics) / n
    avg_metrics['ssim_gt_context_avg'] = sum(m.get('gt_context_ssim', float('nan')) for m in filewise_metrics) / n
    avg_metrics['ssim_pred_gt_avg_alternate'] = (
        sum(m.get('pred_gt_ssim_alternate', float('nan')) for m in filewise_metrics) / n
    )
    avg_metrics['ssim_pred_context_avg_alternate'] = (
        sum(m.get('pred_context_ssim_alternate', float('nan')) for m in filewise_metrics) / n
    )
    avg_metrics['ssim_gt_context_avg_alternate'] = (
        sum(m.get('gt_context_ssim_alternate', float('nan')) for m in filewise_metrics) / n
    )

    # Cumulative WER/CER on ground-truth audio transcriptions (if available)
    gt_audio_texts = [m.get('gt_audio_text') for m in filewise_metrics]
    if None not in gt_audio_texts:
        avg_metrics['cer_gt_audio_cumulative'] = word_error_rate_detail(
            hypotheses=gt_audio_texts, references=gt_texts, use_cer=True
        )[0]
        avg_metrics['wer_gt_audio_cumulative'] = word_error_rate_detail(
            hypotheses=gt_audio_texts, references=gt_texts, use_cer=False
        )[0]
    else:
        avg_metrics['cer_gt_audio_cumulative'] = float('NaN')
        avg_metrics['wer_gt_audio_cumulative'] = float('NaN')

    avg_metrics['utmosv2_avg'] = sum(m.get('utmosv2', float('nan')) for m in filewise_metrics) / n
    avg_metrics['total_gen_audio_seconds'] = sum(m.get('total_gen_audio_seconds', 0.0) for m in filewise_metrics)

    # FCD: compute only if all required paths are provided
    if gt_audio_paths and predicted_codes_paths and codec_model_path:
        avg_metrics['frechet_codec_distance'] = compute_fcd(gt_audio_paths, predicted_codes_paths, codec_model_path)
    else:
        avg_metrics['frechet_codec_distance'] = float('nan')

    pprint.pprint(avg_metrics)
    return avg_metrics


def main():
    # audio_dir="/datap/misc/Datasets/riva" \
    parser = argparse.ArgumentParser(description='Evaluate Generated Audio')
    parser.add_argument('--manifest_path', type=str, default=None)
    parser.add_argument('--audio_dir', type=str, default=None)
    parser.add_argument('--generated_audio_dir', type=str, default=None)
    parser.add_argument('--whisper_language', type=str, default="en")
    parser.add_argument('--evalset', type=str, default=None)
    args = parser.parse_args()

    if args.evalset is not None:
        dataset_meta_info = load_evalset_config()
        assert args.evalset in dataset_meta_info, f"Dataset '{args.evalset}' not found in evalset_config.json"
        args.manifest_path = dataset_meta_info[args.evalset]['manifest_path']
        args.audio_dir = dataset_meta_info[args.evalset]['audio_dir']

    evaluate(
        args.manifest_path,
        args.audio_dir,
        args.generated_audio_dir,
        args.whisper_language,
        sv_model_type="wavlm",
        asr_model_name="nvidia/parakeet-ctc-0.6b",
    )


if __name__ == "__main__":
    main()
