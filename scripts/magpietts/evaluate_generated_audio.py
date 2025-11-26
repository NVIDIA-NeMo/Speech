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
Used in infer_and_evaluate.py to obtain metrics such as ASR_WER and UTMOSV2 scores.
"""
import argparse
import json
import logging
import os
import pprint
import string
import tempfile
import time
from contextlib import contextmanager
from functools import partial

import librosa
import numpy as np
import scripts.magpietts.evalset_config as evalset_config
import soundfile as sf
import torch
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector, WhisperForConditionalGeneration, WhisperProcessor

import nemo.collections.asr as nemo_asr
from nemo.collections.asr.metrics.wer import word_error_rate_detail
from nemo.collections.tts.modules.utmosv2 import UTMOSv2Calculator


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


def pad_audio_to_min_length(audio_np: np.ndarray, sampling_rate: int, min_seconds: float) -> np.ndarray:
    """
    Pad audio to make it at least `min_seconds` long by adding silence at the end if needed.
    """
    if audio_np.ndim != 1:
        raise ValueError("Audio array must be 1D")

    n_samples = len(audio_np)
    min_samples = round(min_seconds * sampling_rate)

    if n_samples < min_samples:
        print(f"Padding audio from {n_samples/sampling_rate} seconds to {min_samples/sampling_rate} seconds")
        padding_needed = min_samples - n_samples
        audio_np = np.pad(audio_np, (0, padding_needed), mode='constant', constant_values=0)
    return audio_np


@contextmanager
def nemo_log_level(level):
    """
    A context manager that temporarily sets the logging level for the NeMo logger
    and restores the original level when the context manager is exited.

    Args:
        level (int): The logging level to set.
    """
    logger = logging.getLogger("nemo_logger")
    original_level = logger.level
    logger.setLevel(level)
    try:
        yield
    finally:
        # restore the original level when the context manager is exited (even if an exception was raised)
        logger.setLevel(original_level)


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
    print(f"\nComputing UTMOSv2 scores for files in {audio_dir}...")
    start_time = time.time()
    utmosv2_calculator = UTMOSv2Calculator(device=device)
    utmosv2_scores = utmosv2_calculator.process_directory(audio_dir)
    # convert to to a dictionary indexed by file path
    utmosv2_scores_dict = {os.path.normpath(item['file_path']): item['predicted_mos'] for item in utmosv2_scores}
    end_time = time.time()
    print(f"UTMOSv2 scores computed for {len(utmosv2_scores)} files in {end_time - start_time:.2f} seconds\n")
    return utmosv2_scores_dict


def evaluate(
    manifest_path,
    audio_dir,
    generated_audio_dir,
    language="en",
    sv_model_type="titanet",
    asr_model_name="stt_en_conformer_transducer_large",
    with_utmosv2=True,
):
    audio_file_lists = find_generated_audio_files(generated_audio_dir)
    records = read_manifest(manifest_path)
    assert len(audio_file_lists) == len(records)

    device = "cuda"

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
    with nemo_log_level(logging.ERROR):
        # The model `titanet_small` prints thousands of lines during initialization, so suppress logs temporarily
        print("Loading `titanet_small` model...")
        speaker_verification_model_alternate = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
            model_name='titanet_small'
        )
    speaker_verification_model_alternate = speaker_verification_model_alternate.to(device)
    speaker_verification_model_alternate.eval()

    if with_utmosv2:
        utmosv2_scores = compute_utmosv2_scores(generated_audio_dir, device)
    filewise_metrics = []
    pred_texts = []
    gt_texts = []
    gt_audio_texts = []
    total_generated_audio_seconds = 0.0
    for ridx, record in enumerate(records):
        gt_audio_filepath = record['audio_filepath']
        context_audio_filepath = record.get('context_audio_filepath', None)
        if audio_dir is not None:
            gt_audio_filepath = os.path.join(audio_dir, gt_audio_filepath)
            if context_audio_filepath is not None:
                context_audio_filepath = os.path.join(audio_dir, context_audio_filepath)

        pred_audio_filepath = audio_file_lists[ridx]

        if with_utmosv2:
            utmosv2_score = utmosv2_scores[os.path.normpath(pred_audio_filepath)]
        else:
            utmosv2_score = 0.0

        try:
            if language == "en":
                with torch.inference_mode():
                    pred_text = asr_model.transcribe([pred_audio_filepath], batch_size=1, use_lhotse=False)[0].text
                    pred_text = process_text(pred_text)
                    gt_audio_text = asr_model.transcribe([gt_audio_filepath], batch_size=1, use_lhotse=False)[0].text
                    gt_audio_text = process_text(gt_audio_text)
            else:
                pred_text = transcribe_with_whisper(
                    whisper_model, whisper_processor, pred_audio_filepath, language, device
                )
                pred_text = process_text(pred_text)
                gt_audio_text = transcribe_with_whisper(
                    whisper_model, whisper_processor, gt_audio_filepath, language, device
                )
                gt_audio_text = process_text(gt_audio_text)
        except Exception as e:
            print("Error during ASR: {}".format(e))
            pred_text = ""
            gt_audio_text = ""

        if "original_text" in record:
            gt_text = process_text(record['original_text'])
        elif 'normalized_text' in record:
            gt_text = process_text(record['normalized_text'])
        else:
            gt_text = process_text(record['text'])

        detailed_cer = word_error_rate_detail(hypotheses=[pred_text], references=[gt_text], use_cer=True)
        detailed_wer = word_error_rate_detail(hypotheses=[pred_text], references=[gt_text], use_cer=False)

        print("{} GT Text:".format(ridx), gt_text)
        print("{} Pr Text:".format(ridx), pred_text)
        # Format cer and wer to 2 decimal places
        print("CER:", "{:.4f} | WER: {:.4f}".format(detailed_cer[0], detailed_wer[0]))

        pred_texts.append(pred_text)
        gt_texts.append(gt_text)
        gt_audio_texts.append(gt_audio_text)

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
                gt_context_ssim = torch.nn.functional.cosine_similarity(
                    gt_speaker_embedding, context_speaker_embedding, dim=0
                ).item()

                # Predicted vs. context (alternate model)
                pred_context_ssim_alternate = torch.nn.functional.cosine_similarity(
                    pred_speaker_embedding_alternate, context_speaker_embedding_alternate, dim=0
                ).item()
                # Ground truth vs. context (alternate model)
                gt_context_ssim_alternate = torch.nn.functional.cosine_similarity(
                    gt_speaker_embedding_alternate, context_speaker_embedding_alternate, dim=0
                ).item()
            total_generated_audio_seconds += get_wav_file_duration(pred_audio_filepath)

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
            }
        )

    filewise_metrics_keys_to_save = [
        'cer',
        'wer',
        'pred_context_ssim',
        'pred_text',
        'gt_text',
        'gt_audio_filepath',
        'pred_audio_filepath',
        'context_audio_filepath',
    ]
    filtered_filewise_metrics = []
    for m in filewise_metrics:
        filtered_filewise_metrics.append({k: m[k] for k in filewise_metrics_keys_to_save})

    # Sort filewise metrics by cer in reverse
    filewise_metrics.sort(key=lambda x: x['cer'], reverse=True)

    avg_metrics = {}
    avg_metrics['cer_filewise_avg'] = sum([m['detailed_cer'][0] for m in filewise_metrics]) / len(filewise_metrics)
    avg_metrics['wer_filewise_avg'] = sum([m['detailed_wer'][0] for m in filewise_metrics]) / len(filewise_metrics)
    avg_metrics['cer_cumulative'] = word_error_rate_detail(hypotheses=pred_texts, references=gt_texts, use_cer=True)[0]
    avg_metrics['wer_cumulative'] = word_error_rate_detail(hypotheses=pred_texts, references=gt_texts, use_cer=False)[
        0
    ]
    avg_metrics['ssim_pred_gt_avg'] = sum([m['pred_gt_ssim'] for m in filewise_metrics]) / len(filewise_metrics)
    avg_metrics['ssim_pred_context_avg'] = sum([m['pred_context_ssim'] for m in filewise_metrics]) / len(
        filewise_metrics
    )
    avg_metrics['ssim_gt_context_avg'] = sum([m['gt_context_ssim'] for m in filewise_metrics]) / len(filewise_metrics)
    avg_metrics['ssim_pred_gt_avg_alternate'] = sum([m['pred_gt_ssim_alternate'] for m in filewise_metrics]) / len(
        filewise_metrics
    )
    avg_metrics['ssim_pred_context_avg_alternate'] = sum(
        [m['pred_context_ssim_alternate'] for m in filewise_metrics]
    ) / len(filewise_metrics)
    avg_metrics['ssim_gt_context_avg_alternate'] = sum(
        [m['gt_context_ssim_alternate'] for m in filewise_metrics]
    ) / len(filewise_metrics)
    avg_metrics["cer_gt_audio_cumulative"] = word_error_rate_detail(
        hypotheses=gt_audio_texts, references=gt_texts, use_cer=True
    )[0]
    avg_metrics["wer_gt_audio_cumulative"] = word_error_rate_detail(
        hypotheses=gt_audio_texts, references=gt_texts, use_cer=False
    )[0]
    avg_metrics["utmosv2_avg"] = sum([m['utmosv2'] for m in filewise_metrics]) / len(filewise_metrics)
    avg_metrics["total_gen_audio_seconds"] = total_generated_audio_seconds
    pprint.pprint(avg_metrics)

    return avg_metrics, filewise_metrics


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
        dataset_meta_info = evalset_config.dataset_meta_info
        assert args.evalset in dataset_meta_info
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
