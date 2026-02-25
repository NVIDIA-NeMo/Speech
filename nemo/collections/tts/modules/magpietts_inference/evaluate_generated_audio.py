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


FILEWISE_METRICS_TO_SAVE = [
    'cer',
    'wer',
    'pred_context_ssim',
    'pred_text',
    'gt_text',
    'gt_audio_filepath',
    'pred_audio_filepath',
    'context_audio_filepath',
    'utmosv2',
]


def load_evalset_config(config_path: str = None) -> dict:
    """Load dataset meta info from JSON config file."""
    if config_path is None or not os.path.exists(config_path):
        raise ValueError("No dataset_json_path provided, please provide a valid path to the evalset config file.")
    logging.info(f"Loading evalset config from {config_path}")
    with open(config_path, 'r') as f:
        return json.load(f)


def _resolve_path(audio_dir, path):
    """Resolve a relative path against audio_dir if both are provided."""
    if audio_dir is not None and path is not None:
        return os.path.join(audio_dir, path)
    return path


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


def transcribe_with_nemo_asr_batched(asr_model, audio_paths, batch_size=8, label=""):
    """Transcribe multiple audio files with a NeMo ASR model in batches. Returns list of transcriptions (one per path)."""
    all_transcriptions = []
    for start in range(0, len(audio_paths), batch_size):
        batch_paths = audio_paths[start : start + batch_size]
        try:
            with torch.inference_mode():
                batch_results = asr_model.transcribe(batch_paths, batch_size=len(batch_paths), use_lhotse=False)
            for r in batch_results:
                all_transcriptions.append(process_text(r.text))
        except Exception as e:
            logging.info("Error during batched ASR ({} audio): {}".format(label, e))
            all_transcriptions.extend([""] * len(batch_paths))
    return all_transcriptions


def transcribe_with_whisper_batched(
    whisper_model, whisper_processor, audio_paths, language, device, batch_size=8, label=""
):
    """Transcribe multiple audio files with Whisper in batches. Returns list of transcriptions (one per path)."""
    forced_decoder_ids = (
        whisper_processor.get_decoder_prompt_ids(language=language, task="transcribe") if language else None
    )
    all_transcriptions = []
    for start in range(0, len(audio_paths), batch_size):
        batch_paths = audio_paths[start : start + batch_size]
        try:
            speech_arrays = [librosa.load(p, sr=16000)[0] for p in batch_paths]
            inputs = whisper_processor(
                speech_arrays, sampling_rate=16000, return_tensors="pt", padding=True
            ).input_features
            inputs = inputs.to(device)
            with torch.inference_mode():
                predicted_ids = whisper_model.generate(inputs, forced_decoder_ids=forced_decoder_ids)
            transcriptions = whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)
            all_transcriptions.extend(process_text(t) for t in transcriptions)
        except Exception as e:
            logging.info("Error during batched Whisper ASR ({} audio): {}".format(label, e))
            all_transcriptions.extend([""] * len(batch_paths))
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


def batch_transcribe(
    audio_paths,
    language,
    asr_model,
    whisper_model,
    whisper_processor,
    device,
    asr_batch_size,
    label="",
):
    """Transcribe a list of audio files using NeMo ASR (English) or Whisper (other languages)."""
    if language == "en":
        texts = transcribe_with_nemo_asr_batched(asr_model, audio_paths, batch_size=asr_batch_size, label=label)
    else:
        texts = transcribe_with_whisper_batched(
            whisper_model, whisper_processor, audio_paths, language, device, batch_size=asr_batch_size, label=label
        )
    return texts


def load_evaluation_models(
    language="en", sv_model_type="titanet", asr_model_name="stt_en_conformer_transducer_large", device="cuda"
):
    """Load ASR and speaker verification models used for evaluation.

    Args:
        language: Language code. "en" uses a NeMo ASR model; other languages use Whisper.
        sv_model_type: Speaker verification model type ("wavlm" or "titanet").
        asr_model_name: Name of the NeMo ASR model (used only when language is "en").
        device: Device to place models on.

    Returns:
        Dict with keys: asr_model, whisper_model, whisper_processor, feature_extractor,
        sv_model, sv_model_alternate.
    """
    models = {
        'asr_model': None,
        'whisper_model': None,
        'whisper_processor': None,
        'feature_extractor': None,
    }

    if language == "en":
        if asr_model_name.startswith("nvidia/") or asr_model_name in ["stt_en_conformer_transducer_large"]:
            models['asr_model'] = nemo_asr.models.ASRModel.from_pretrained(model_name=asr_model_name).to(device).eval()
        else:
            raise ValueError(f"ASR model {asr_model_name} not supported")
    else:
        models['whisper_processor'] = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
        models['whisper_model'] = (
            WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3").to(device).eval()
        )

    if sv_model_type == "wavlm":
        models['feature_extractor'] = Wav2Vec2FeatureExtractor.from_pretrained('microsoft/wavlm-base-plus-sv')
        models['sv_model'] = WavLMForXVector.from_pretrained('microsoft/wavlm-base-plus-sv').to(device).eval()
    else:
        models['sv_model'] = (
            nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_name='titanet_large').to(device).eval()
        )

    logging.info("Loading `titanet_small` model...")
    with logging.temp_verbosity(logging.ERROR):
        models['sv_model_alternate'] = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
            model_name='titanet_small'
        )
    models['sv_model_alternate'] = models['sv_model_alternate'].to(device).eval()

    return models


def evaluate_dir(
    manifest_path,
    audio_dir,
    generated_audio_dir,
    language="en",
    sv_model_type="titanet",
    asr_model_name="stt_en_conformer_transducer_large",
    with_utmosv2=True,
    asr_batch_size=32,
    device="cuda",
):
    """Compute per-file evaluation metrics for a directory of generated audio.

    Evaluates individual files (ASR WER/CER, speaker similarity, UTMOSv2) but
    does not compute global/aggregate metrics (cumulative WER/CER, FCD). Use
    compute_global_metrics() to aggregate results from one or more directories.

    External callers (e.g. NeMo Skills) can run evaluate_dir() in parallel across
    multiple directories (on different workers/GPUs), concatenate the resulting
    filewise_metrics lists, and call compute_global_metrics() once at the end to
    accumulate the global metrics.

    Returns:
        filewise_metrics: List of per-file metric dictionaries.
    """
    logging.info(f"Evaluating generated audio in {generated_audio_dir}...")

    # 1. Collect files to evaluate
    audio_file_lists = find_generated_audio_files(generated_audio_dir)
    records = read_manifest(manifest_path)
    assert len(audio_file_lists) == len(records)
    codes_file_lists = find_generated_codec_files(generated_audio_dir)
    has_codes = len(codes_file_lists) == len(records)
    # Resolve ground-truth and context audio paths for all records
    gt_audio_paths = [_resolve_path(audio_dir, r.get('audio_filepath')) for r in records]
    context_audio_paths = [_resolve_path(audio_dir, r.get('context_audio_filepath')) for r in records]

    # 2. Load models
    models = load_evaluation_models(language, sv_model_type, asr_model_name, device)

    asr_model = models['asr_model']
    whisper_model = models['whisper_model']
    whisper_processor = models['whisper_processor']
    feature_extractor = models['feature_extractor']
    speaker_verification_model = models['sv_model']
    speaker_verification_model_alternate = models['sv_model_alternate']

    # 3. Compute UTMOSv2 scores
    utmosv2_scores = None
    if with_utmosv2:
        if not UTMOSV2_AVAILABLE:
            logging.warning(
                "UTMOSv2 was requested (with_utmosv2=True) but the UTMOSv2 library is not available. "
                "UTMOSv2 scores will be set to NaN for all files."
            )
        utmosv2_scores = compute_utmosv2_scores(generated_audio_dir, device)

    # 4. ASR transcription in batches
    logging.info(f"Doing batched ASR transcription with batch size {asr_batch_size}...")
    # Transcribe predicted audios
    pred_texts = batch_transcribe(
        audio_file_lists,
        language,
        asr_model,
        whisper_model,
        whisper_processor,
        device,
        asr_batch_size,
        label="predicted",
    )
    # Transcribe ground truth audios
    if len(gt_audio_paths) > 0:
        gt_audio_texts = batch_transcribe(
            gt_audio_paths,
            language,
            asr_model,
            whisper_model,
            whisper_processor,
            device,
            asr_batch_size,
            label="ground truth",
        )
    else:
        gt_audio_texts = [None] * len(records)

    # 5. Compute metrics for each utterance (sequential)
    filewise_metrics = []
    total_generated_audio_seconds = 0.0
    for ridx, record in enumerate(records):
        gt_audio_filepath = gt_audio_paths[ridx]
        context_audio_filepath = context_audio_paths[ridx]

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

        detailed_cer = word_error_rate_detail(hypotheses=[pred_text], references=[gt_text], use_cer=True)
        detailed_wer = word_error_rate_detail(hypotheses=[pred_text], references=[gt_text], use_cer=False)

        logging.info(f"{ridx} GT Text: {gt_text}")
        logging.info(f"{ridx} Pr Text: {pred_text}")
        # Format cer and wer to 2 decimal places
        logging.info(f"CER: {detailed_cer[0]:.4f} | WER: {detailed_wer[0]:.4f}")

        pred_context_ssim = 0.0
        gt_context_ssim = 0.0
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
                'predicted_codes_path': codes_file_lists[ridx] if has_codes else None,
            }
        )

    return filewise_metrics


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
    device="cuda",
):
    """Evaluate generated audio, computing both per-file and global metrics.

    Convenience wrapper that calls evaluate_dir() for per-file scoring and
    compute_global_metrics() for aggregation of metrics.

    External callers (e.g. NeMo Skills) can run evaluate_dir() in parallel across
    multiple directories (on different workers/GPUs), concatenate the resulting
    filewise_metrics lists, and call compute_global_metrics() once at the end to
    aggregate the global metrics.

    Note the FCD computation is deferred to compute_global_metrics() because the
    metric implementation is stateful and therefore cannot be easily merged across
    subsets (directories).

    Returns:
        Tuple of (avg_metrics dict, filewise_metrics list).
    """
    start_time = time.time()

    if with_fcd and codec_model_path is None:
        raise ValueError("codec_model_path is required when with_fcd is True")

    filewise_metrics = evaluate_dir(
        manifest_path=manifest_path,
        audio_dir=audio_dir,
        generated_audio_dir=generated_audio_dir,
        language=language,
        sv_model_type=sv_model_type,
        asr_model_name=asr_model_name,
        with_utmosv2=with_utmosv2,
        asr_batch_size=asr_batch_size,
        device=device,
    )

    gt_audio_paths = None
    predicted_codes_paths = None
    if with_fcd:
        gt_audio_paths = [m['gt_audio_filepath'] for m in filewise_metrics]
        predicted_codes_paths = [m['predicted_codes_path'] for m in filewise_metrics]

    avg_metrics = compute_global_metrics(
        filewise_metrics=filewise_metrics,
        gt_audio_paths=gt_audio_paths,
        predicted_codes_paths=predicted_codes_paths,
        codec_model_path=codec_model_path,
        device=device,
    )

    elapsed = time.time() - start_time
    logging.info(f"evaluate() completed in {elapsed:.1f}s ({elapsed / 60:.1f} min)")

    filtered_filewise = [{k: m[k] for k in FILEWISE_METRICS_TO_SAVE if k in m} for m in filewise_metrics]
    return avg_metrics, filtered_filewise


def compute_fcd(gt_audio_paths, predicted_codes_paths, codec_model_path, device="cuda"):
    """Compute Frechet Codec Distance from ground-truth audio paths and predicted codec codes paths.

    Args:
        gt_audio_paths: List of paths to ground-truth audio files.
        predicted_codes_paths: List of paths to predicted codec codes (.pt) files.
        codec_model_path: Path or name of the codec model for FCD computation.
        device: Device to use for computation.

    Returns:
        FCD score (float).
    """
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
    device="cuda",
):
    """Aggregate metrics from per-file results.

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
    avg_metrics['ssim_pred_gt_avg'] = sum(m['pred_gt_ssim'] for m in filewise_metrics) / n
    avg_metrics['ssim_pred_context_avg'] = sum(m['pred_context_ssim'] for m in filewise_metrics) / n
    avg_metrics['ssim_gt_context_avg'] = sum(m['gt_context_ssim'] for m in filewise_metrics) / n
    avg_metrics['ssim_pred_gt_avg_alternate'] = sum(m['pred_gt_ssim_alternate'] for m in filewise_metrics) / n
    avg_metrics['ssim_pred_context_avg_alternate'] = (
        sum(m['pred_context_ssim_alternate'] for m in filewise_metrics) / n
    )
    avg_metrics['ssim_gt_context_avg_alternate'] = sum(m['gt_context_ssim_alternate'] for m in filewise_metrics) / n

    # Cumulative WER/CER on ground-truth audio transcriptions (if available)
    gt_audio_texts = [m['gt_audio_text'] for m in filewise_metrics]
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
        logging.warning(
            "Ground truth audio files are missing. Setting cumulative CER and WER for ground truth audio to NaN."
        )

    avg_metrics['utmosv2_avg'] = sum(m['utmosv2'] for m in filewise_metrics) / n
    avg_metrics['total_gen_audio_seconds'] = sum(m['total_gen_audio_seconds'] for m in filewise_metrics)

    # FCD: compute only if all required paths are provided
    if gt_audio_paths and predicted_codes_paths and codec_model_path:
        avg_metrics['frechet_codec_distance'] = compute_fcd(
            gt_audio_paths, predicted_codes_paths, codec_model_path, device=device
        )
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
