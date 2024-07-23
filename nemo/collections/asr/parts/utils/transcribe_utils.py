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
import glob
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
from omegaconf import DictConfig
from tqdm.auto import tqdm

import nemo.collections.asr as nemo_asr
from nemo.collections.asr.metrics.wer import word_error_rate
from nemo.collections.asr.models import ASRModel, EncDecHybridRNNTCTCModel, EncDecMultiTaskModel
from nemo.collections.asr.parts.utils import manifest_utils, rnnt_utils
from nemo.collections.asr.parts.utils.streaming_utils import FrameBatchASR, FrameBatchMultiTaskAED
from nemo.collections.common.metrics.punct_er import OccurancePunctuationErrorRate
from nemo.collections.common.parts.preprocessing.manifest import get_full_path
from nemo.utils import logging, model_utils


def get_buffered_pred_feat_rnnt(
    asr: FrameBatchASR,
    tokens_per_chunk: int,
    delay: int,
    model_stride_in_secs: int,
    batch_size: int,
    manifest: str = None,
    filepaths: List[list] = None,
) -> List[rnnt_utils.Hypothesis]:
    """
    Moved from examples/asr/asr_chunked_inference/rnnt/speech_to_text_buffered_infer_rnnt.py
    Write all information presented in input manifest to output manifest and removed WER calculation.
    """
    hyps = []
    refs = []

    if filepaths and manifest:
        raise ValueError("Please select either filepaths or manifest")
    if filepaths is None and manifest is None:
        raise ValueError("Either filepaths or manifest shoud not be None")

    if manifest:
        filepaths = []
        with open(manifest, "r", encoding='utf_8') as mfst_f:
            print("Parsing manifest files...")
            for l in mfst_f:
                row = json.loads(l.strip())
                audio_file = get_full_path(audio_file=row['audio_filepath'], manifest_file=manifest)
                filepaths.append(audio_file)
                if 'text' in row:
                    refs.append(row['text'])

    with torch.inference_mode():
        with torch.cuda.amp.autocast():
            batch = []
            asr.sample_offset = 0
            for idx in tqdm(range(len(filepaths)), desc='Sample:', total=len(filepaths)):
                batch.append((filepaths[idx]))

                if len(batch) == batch_size:
                    audio_files = [sample for sample in batch]

                    asr.reset()
                    asr.read_audio_file(audio_files, delay, model_stride_in_secs)
                    hyp_list = asr.transcribe(tokens_per_chunk, delay)
                    hyps.extend(hyp_list)

                    batch.clear()
                    asr.sample_offset += batch_size

            if len(batch) > 0:
                asr.batch_size = len(batch)
                asr.frame_bufferer.batch_size = len(batch)
                asr.reset()

                audio_files = [sample for sample in batch]
                asr.read_audio_file(audio_files, delay, model_stride_in_secs)
                hyp_list = asr.transcribe(tokens_per_chunk, delay)
                hyps.extend(hyp_list)

                batch.clear()
                asr.sample_offset += len(batch)

    if os.environ.get('DEBUG', '0') in ('1', 'y', 't'):
        if len(refs) == 0:
            print("ground-truth text does not present!")
            for hyp in hyps:
                print("hyp:", hyp)
        else:
            for hyp, ref in zip(hyps, refs):
                print("hyp:", hyp)
                print("ref:", ref)

    wrapped_hyps = wrap_transcription(hyps)
    return wrapped_hyps


def get_buffered_pred_feat(
    asr: FrameBatchASR,
    frame_len: float,
    tokens_per_chunk: int,
    delay: int,
    preprocessor_cfg: DictConfig,
    model_stride_in_secs: int,
    device: Union[List[int], int],
    manifest: str = None,
    filepaths: List[list] = None,
) -> List[rnnt_utils.Hypothesis]:
    """
    Moved from examples/asr/asr_chunked_inference/ctc/speech_to_text_buffered_infer_ctc.py
    Write all information presented in input manifest to output manifest and removed WER calculation.
    """
    # Create a preprocessor to convert audio samples into raw features,
    # Normalization will be done per buffer in frame_bufferer
    # Do not normalize whatever the model's preprocessor setting is
    preprocessor_cfg.normalize = "None"
    preprocessor = nemo_asr.models.EncDecCTCModelBPE.from_config_dict(preprocessor_cfg)
    preprocessor.to(device)
    hyps = []
    refs = []

    if filepaths and manifest:
        raise ValueError("Please select either filepaths or manifest")
    if filepaths is None and manifest is None:
        raise ValueError("Either filepaths or manifest shoud not be None")

    if filepaths:
        for l in tqdm(filepaths, desc="Sample:"):
            asr.reset()
            asr.read_audio_file(l, delay, model_stride_in_secs)
            hyp = asr.transcribe(tokens_per_chunk, delay)
            hyps.append(hyp)
    else:
        with open(manifest, "r", encoding='utf_8') as mfst_f:
            for l in tqdm(mfst_f, desc="Sample:"):
                asr.reset()
                row = json.loads(l.strip())
                if 'text' in row:
                    refs.append(row['text'])
                audio_file = get_full_path(audio_file=row['audio_filepath'], manifest_file=manifest)
                # do not support partial audio
                asr.read_audio_file(audio_file, delay, model_stride_in_secs)
                hyp = asr.transcribe(tokens_per_chunk, delay)
                hyps.append(hyp)

    if os.environ.get('DEBUG', '0') in ('1', 'y', 't'):
        if len(refs) == 0:
            print("ground-truth text does not present!")
            for hyp in hyps:
                print("hyp:", hyp)
        else:
            for hyp, ref in zip(hyps, refs):
                print("hyp:", hyp)
                print("ref:", ref)

    wrapped_hyps = wrap_transcription(hyps)
    return wrapped_hyps


def get_buffered_pred_feat_multitaskAED(
    asr: FrameBatchMultiTaskAED,
    preprocessor_cfg: DictConfig,
    model_stride_in_secs: int,
    device: Union[List[int], int],
    manifest: str = None,
    filepaths: List[list] = None,
    delay: float = 0.0,
) -> List[rnnt_utils.Hypothesis]:
    # Create a preprocessor to convert audio samples into raw features,
    # Normalization will be done per buffer in frame_bufferer
    # Do not normalize whatever the model's preprocessor setting is
    preprocessor_cfg.normalize = "None"
    preprocessor = EncDecMultiTaskModel.from_config_dict(preprocessor_cfg)
    preprocessor.to(device)
    hyps = []
    refs = []

    if filepaths and manifest:
        raise ValueError("Please select either filepaths or manifest")
    if filepaths is None and manifest is None:
        raise ValueError("Either filepaths or manifest shoud not be None")

    if filepaths:
        logging.info(
            "Deteced audio files as input, default to English ASR with Punctuation and Capitalization output. Please use manifest input for other options."
        )
        for audio_file in tqdm(filepaths, desc="Transcribing:", total=len(filepaths), ncols=80):
            meta = {
                'audio_filepath': audio_file,
                'duration': 100000,
                'source_lang': 'en',
                'taskname': 'asr',
                'target_lang': 'en',
                'pnc': 'yes',
                'answer': 'nothing',
            }
            asr.reset()
            asr.read_audio_file(audio_file, delay, model_stride_in_secs, meta_data=meta)
            hyp = asr.transcribe()
            hyps.append(hyp)
    else:
        with open(manifest, "r", encoding='utf_8') as fin:
            lines = list(fin.readlines())
            for line in tqdm(lines, desc="Transcribing:", total=len(lines), ncols=80):
                asr.reset()
                sample = json.loads(line.strip())
                if 'text' in sample:
                    refs.append(sample['text'])
                audio_file = get_full_path(audio_file=sample['audio_filepath'], manifest_file=manifest)
                # do not support partial audio
                asr.read_audio_file(audio_file, delay, model_stride_in_secs, meta_data=sample)
                hyp = asr.transcribe()
                hyps.append(hyp)

    wrapped_hyps = wrap_transcription(hyps)
    return wrapped_hyps


def wrap_transcription(hyps: List[str]) -> List[rnnt_utils.Hypothesis]:
    """Wrap transcription to the expected format in func write_transcription"""
    wrapped_hyps = []
    for hyp in hyps:
        hypothesis = rnnt_utils.Hypothesis(score=0.0, y_sequence=[], text=hyp)
        wrapped_hyps.append(hypothesis)
    return wrapped_hyps


def setup_model(cfg: DictConfig, map_location: torch.device) -> Tuple[ASRModel, str]:
    """Setup model from cfg and return model and model name for next step"""
    if cfg.model_path is not None and cfg.model_path != "None":
        # restore model from .nemo file path
        model_cfg = ASRModel.restore_from(restore_path=cfg.model_path, return_config=True)
        classpath = model_cfg.target  # original class path
        imported_class = model_utils.import_class_by_path(classpath)  # type: ASRModel
        logging.info(f"Restoring model : {imported_class.__name__}")
        asr_model = imported_class.restore_from(
            restore_path=cfg.model_path,
            map_location=map_location,
        )  # type: ASRModel
        model_name = os.path.splitext(os.path.basename(cfg.model_path))[0]
    else:
        # restore model by name
        asr_model = ASRModel.from_pretrained(
            model_name=cfg.pretrained_name,
            map_location=map_location,
        )  # type: ASRModel
        model_name = cfg.pretrained_name

    if hasattr(cfg, "model_change") and hasattr(asr_model, "change_attention_model"):
        asr_model.change_attention_model(
            self_attention_model=cfg.model_change.conformer.get("self_attention_model", None),
            att_context_size=cfg.model_change.conformer.get("att_context_size", None),
        )

    return asr_model, model_name


def prepare_audio_data(cfg: DictConfig) -> Tuple[List[str], bool]:
    """Prepare audio data and decide whether it's partial_audio condition."""
    # this part may need refactor alongsides with refactor of transcribe
    partial_audio = False

    if cfg.audio_dir is not None and not cfg.append_pred:
        filepaths = list(glob.glob(os.path.join(cfg.audio_dir, f"**/*.{cfg.audio_type}"), recursive=True))
    else:
        # get filenames from manifest
        filepaths = []
        if os.stat(cfg.dataset_manifest).st_size == 0:
            logging.error(f"The input dataset_manifest {cfg.dataset_manifest} is empty. Exiting!")
            return None

        audio_key = cfg.get('audio_key', 'audio_filepath')

        with open(cfg.dataset_manifest, "rt") as fh:
            for line in fh:
                item = json.loads(line)
                item[audio_key] = get_full_path(item[audio_key], cfg.dataset_manifest)
                if item.get("duration") is None and cfg.presort_manifest:
                    raise ValueError(
                        f"Requested presort_manifest=True, but line {line} in manifest {cfg.dataset_manifest} lacks a 'duration' field."
                    )
        all_entries_have_offset_and_duration = True
        for item in read_and_maybe_sort_manifest(cfg.dataset_manifest, try_sort=cfg.presort_manifest):
            if not ("offset" in item and "duration" in item):
                all_entries_have_offset_and_duration = False
            audio_file = get_full_path(audio_file=item[audio_key], manifest_file=cfg.dataset_manifest)
            filepaths.append(audio_file)
        partial_audio = all_entries_have_offset_and_duration
    logging.info(f"\nTranscribing {len(filepaths)} files...\n")

    return filepaths, partial_audio


def read_and_maybe_sort_manifest(path: str, try_sort: bool = False) -> List[dict]:
    """Sorts the manifest if duration key is available for every utterance."""
    items = manifest_utils.read_manifest(path)
    if try_sort and all("duration" in item and item["duration"] is not None for item in items):
        items = sorted(items, reverse=True, key=lambda item: item["duration"])
    return items


def restore_transcription_order(manifest_path: str, transcriptions: list) -> list:
    with open(manifest_path) as f:
        items = [(idx, json.loads(l)) for idx, l in enumerate(f)]
    if not all("duration" in item[1] and item[1]["duration"] is not None for item in items):
        return transcriptions
    new2old = [item[0] for item in sorted(items, reverse=True, key=lambda it: it[1]["duration"])]
    del items  # free up some memory
    is_list = isinstance(transcriptions[0], list)
    if is_list:
        transcriptions = list(zip(*transcriptions))
    reordered = [None] * len(transcriptions)
    for new, old in enumerate(new2old):
        reordered[old] = transcriptions[new]
    if is_list:
        reordered = tuple(map(list, zip(*reordered)))
    return reordered


def compute_output_filename(cfg: DictConfig, model_name: str) -> DictConfig:
    """Compute filename of output manifest and update cfg"""
    if cfg.output_filename is None:
        # create default output filename
        if cfg.audio_dir is not None:
            cfg.output_filename = os.path.dirname(os.path.join(cfg.audio_dir, '.')) + '.json'
        elif cfg.pred_name_postfix is not None:
            cfg.output_filename = cfg.dataset_manifest.replace('.json', f'_{cfg.pred_name_postfix}.json')
        else:
            cfg.output_filename = cfg.dataset_manifest.replace('.json', f'_{model_name}.json')
    return cfg


def normalize_timestamp_output(timestamps: dict):
    """
    Normalize the dictionary of timestamp values to JSON serializable values.
    Expects the following keys to exist -
        "start_offset": int-like object that represents the starting index of the token
            in the full audio after downsampling.
        "end_offset": int-like object that represents the ending index of the token
            in the full audio after downsampling.

    Args:
        timestamps: Nested dict.

    Returns:
        Normalized `timestamps` dictionary (in-place normalized)
    """
    for val_idx in range(len(timestamps)):
        timestamps[val_idx]['start_offset'] = int(timestamps[val_idx]['start_offset'])
        timestamps[val_idx]['end_offset'] = int(timestamps[val_idx]['end_offset'])
    return timestamps


def write_transcription(
    transcriptions: Union[List[rnnt_utils.Hypothesis], List[List[rnnt_utils.Hypothesis]], List[str]],
    cfg: DictConfig,
    model_name: str,
    filepaths: List[str] = None,
    compute_langs: bool = False,
    compute_timestamps: bool = False,
) -> Tuple[str, str]:
    """Write generated transcription to output file."""
    if cfg.append_pred:
        logging.info(f'Transcripts will be written in "{cfg.output_filename}" file')
        if cfg.pred_name_postfix is not None:
            pred_by_model_name = cfg.pred_name_postfix
        else:
            pred_by_model_name = model_name
        pred_text_attr_name = 'pred_text_' + pred_by_model_name
    else:
        pred_text_attr_name = 'pred_text'

    return_hypotheses = True
    if isinstance(transcriptions[0], str):  # List[str]:
        best_hyps = transcriptions
        return_hypotheses = False
    elif isinstance(transcriptions[0], rnnt_utils.Hypothesis):  # List[rnnt_utils.Hypothesis]
        best_hyps = transcriptions
        assert cfg.decoding.beam.return_best_hypothesis, "Works only with return_best_hypothesis=true"
    elif isinstance(transcriptions[0], list) and isinstance(
        transcriptions[0][0], rnnt_utils.Hypothesis
    ):  # List[List[rnnt_utils.Hypothesis]] NBestHypothesis
        best_hyps, beams = [], []
        for hyps in transcriptions:
            best_hyps.append(hyps[0])
            if not cfg.decoding.beam.return_best_hypothesis:
                beam = []
                for hyp in hyps:
                    score = hyp.score.numpy().item() if isinstance(hyp.score, torch.Tensor) else hyp.score
                    beam.append((hyp.text, score))
                beams.append(beam)
    else:
        raise TypeError

    # create output dir if not exists
    Path(cfg.output_filename).parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.output_filename, 'w', encoding='utf-8', newline='\n') as f:
        if cfg.audio_dir is not None:
            for idx, transcription in enumerate(best_hyps):  # type: rnnt_utils.Hypothesis or str
                if not return_hypotheses:  # transcription is str
                    item = {'audio_filepath': filepaths[idx], pred_text_attr_name: transcription}
                else:  # transcription is Hypothesis
                    item = {'audio_filepath': filepaths[idx], pred_text_attr_name: transcription.text}

                    if compute_timestamps:
                        timestamps = transcription.timestep
                        if timestamps is not None and isinstance(timestamps, dict):
                            timestamps.pop(
                                'timestep', None
                            )  # Pytorch tensor calculating index of each token, not needed.
                            for key in timestamps.keys():
                                values = normalize_timestamp_output(timestamps[key])
                                item[f'timestamps_{key}'] = values

                    if compute_langs:
                        item['pred_lang'] = transcription.langs
                        item['pred_lang_chars'] = transcription.langs_chars
                    if not cfg.decoding.beam.return_best_hypothesis:
                        item['beams'] = beams[idx]
                f.write(json.dumps(item) + "\n")
        else:
            with open(cfg.dataset_manifest, 'r', encoding='utf-8') as fr:
                for idx, line in enumerate(fr):
                    item = json.loads(line)
                    if not return_hypotheses:  # transcription is str
                        item[pred_text_attr_name] = best_hyps[idx]
                    else:  # transcription is Hypothesis
                        item[pred_text_attr_name] = best_hyps[idx].text

                        if compute_timestamps:
                            timestamps = best_hyps[idx].timestep
                            if timestamps is not None and isinstance(timestamps, dict):
                                timestamps.pop(
                                    'timestep', None
                                )  # Pytorch tensor calculating index of each token, not needed.
                                for key in timestamps.keys():
                                    values = normalize_timestamp_output(timestamps[key])
                                    item[f'timestamps_{key}'] = values

                        if compute_langs:
                            item['pred_lang'] = best_hyps[idx].langs
                            item['pred_lang_chars'] = best_hyps[idx].langs_chars

                        if not cfg.decoding.beam.return_best_hypothesis:
                            item['beams'] = beams[idx]
                    f.write(json.dumps(item) + "\n")

    return cfg.output_filename, pred_text_attr_name


def transcribe_partial_audio(
    asr_model,
    path2manifest: str = None,
    batch_size: int = 4,
    logprobs: bool = False,
    return_hypotheses: bool = False,
    num_workers: int = 0,
    channel_selector: Optional[int] = None,
    augmentor: DictConfig = None,
    decoder_type: Optional[str] = None,
) -> List[str]:
    """
    See description of this function in trancribe() in nemo/collections/asr/models/ctc_models.py and nemo/collections/asr/models/rnnt_models.py
    """

    if return_hypotheses and logprobs:
        raise ValueError(
            "Either `return_hypotheses` or `logprobs` can be True at any given time."
            "Returned hypotheses will contain the logprobs."
        )
    if num_workers is None:
        num_workers = min(batch_size, os.cpu_count() - 1)

    # We will store transcriptions here
    hypotheses = []
    # Model's mode and device
    mode = asr_model.training
    device = next(asr_model.parameters()).device
    dither_value = asr_model.preprocessor.featurizer.dither
    pad_to_value = asr_model.preprocessor.featurizer.pad_to

    if decoder_type is not None:  # Hybrid model
        decode_function = (
            asr_model.decoding.rnnt_decoder_predictions_tensor
            if decoder_type == 'rnnt'
            else asr_model.ctc_decoding.ctc_decoder_predictions_tensor
        )
    elif hasattr(asr_model, 'joint'):  # RNNT model
        decode_function = asr_model.decoding.rnnt_decoder_predictions_tensor
    else:  # CTC model
        decode_function = asr_model.decoding.ctc_decoder_predictions_tensor

    try:
        asr_model.preprocessor.featurizer.dither = 0.0
        asr_model.preprocessor.featurizer.pad_to = 0
        # Switch model to evaluation mode
        asr_model.eval()
        # Freeze the encoder and decoder modules
        asr_model.encoder.freeze()
        asr_model.decoder.freeze()
        logging_level = logging.get_verbosity()
        logging.set_verbosity(logging.WARNING)

        config = {
            'manifest_filepath': path2manifest,
            'batch_size': batch_size,
            'num_workers': num_workers,
            'channel_selector': channel_selector,
        }
        if augmentor:
            config['augmentor'] = augmentor

        temporary_datalayer = asr_model._setup_transcribe_dataloader(config)
        for test_batch in tqdm(temporary_datalayer, desc="Transcribing"):
            outputs = asr_model.forward(
                input_signal=test_batch[0].to(device), input_signal_length=test_batch[1].to(device)
            )
            logits, logits_len = outputs[0], outputs[1]

            if isinstance(asr_model, EncDecHybridRNNTCTCModel) and decoder_type == "ctc":
                logits = asr_model.ctc_decoder(encoder_output=logits)

            logits = logits.cpu()

            if logprobs:
                logits = logits.numpy()
                # dump log probs per file
                for idx in range(logits.shape[0]):
                    lg = logits[idx][: logits_len[idx]]
                    hypotheses.append(lg)
            else:
                current_hypotheses, _ = decode_function(
                    logits,
                    logits_len,
                    return_hypotheses=return_hypotheses,
                )

                if return_hypotheses:
                    # dump log probs per file
                    for idx in range(logits.shape[0]):
                        current_hypotheses[idx].y_sequence = logits[idx][: logits_len[idx]]
                        if current_hypotheses[idx].alignments is None:
                            current_hypotheses[idx].alignments = current_hypotheses[idx].y_sequence

                hypotheses += current_hypotheses

            del logits
            del test_batch

    finally:
        # set mode back to its original value
        asr_model.train(mode=mode)
        asr_model.preprocessor.featurizer.dither = dither_value
        asr_model.preprocessor.featurizer.pad_to = pad_to_value
        if mode is True:
            asr_model.encoder.unfreeze()
            asr_model.decoder.unfreeze()
        logging.set_verbosity(logging_level)
    return hypotheses


def compute_metrics_per_sample(
    manifest_path: str,
    reference_field: str = "text",
    hypothesis_field: str = "pred_text",
    metrics: List[str] = ["wer"],
    punctuation_marks: List[str] = [".", ",", "?"],
    output_manifest_path: str = None,
) -> dict:
    '''
    Computes metrics per sample for given manifest

    Args:
        manifest_path: str, Required - path to dataset JSON manifest file (in NeMo format)
        reference_field: str, Optional - name of field in .json manifest with the reference text ("text" by default).
        hypothesis_field: str, Optional - name of field in .json manifest with the hypothesis text ("pred_text" by default).
        metrics: list[str], Optional - list of metrics to be computed (currently supported "wer", "cer", "punct_er")
        punctuation_marks: list[str], Optional - list of punctuation marks for computing punctuation error rate ([".", ",", "?"] by default).
        output_manifest_path: str, Optional - path where .json manifest with calculated metrics will be saved.

    Returns:
        samples: dict - Dict of samples with calculated metrics
    '''

    supported_metrics = ["wer", "cer", "punct_er"]

    if len(metrics) == 0:
        raise AssertionError(
            f"'metrics' list is empty. \
            Select the metrics from the supported: {supported_metrics}."
        )

    for metric in metrics:
        if metric not in supported_metrics:
            raise AssertionError(
                f"'{metric}' metric is not supported. \
                Currently supported metrics are {supported_metrics}."
            )

    if "punct_er" in metrics:
        if len(punctuation_marks) == 0:
            raise AssertionError("punctuation_marks list can't be empty when 'punct_er' metric is enabled.")
        else:
            oper_obj = OccurancePunctuationErrorRate(punctuation_marks=punctuation_marks)

    use_wer = "wer" in metrics
    use_cer = "cer" in metrics
    use_punct_er = "punct_er" in metrics

    with open(manifest_path, 'r') as manifest:
        lines = manifest.readlines()
        samples = [json.loads(line) for line in lines]
        samples_with_metrics = []

        logging.info(f"Computing {', '.join(metrics)} per sample")

        for sample in tqdm(samples):
            reference = sample[reference_field]
            hypothesis = sample[hypothesis_field]

            if use_wer:
                sample_wer = word_error_rate(hypotheses=[hypothesis], references=[reference], use_cer=False)
                sample["wer"] = round(100 * sample_wer, 2)

            if use_cer:
                sample_cer = word_error_rate(hypotheses=[hypothesis], references=[reference], use_cer=True)
                sample["cer"] = round(100 * sample_cer, 2)

            if use_punct_er:
                operation_amounts, substitution_amounts, punctuation_rates = oper_obj.compute(
                    reference=reference, hypothesis=hypothesis
                )
                sample["punct_correct_rate"] = round(100 * punctuation_rates.correct_rate, 2)
                sample["punct_deletions_rate"] = round(100 * punctuation_rates.deletions_rate, 2)
                sample["punct_insertions_rate"] = round(100 * punctuation_rates.insertions_rate, 2)
                sample["punct_substitutions_rate"] = round(100 * punctuation_rates.substitutions_rate, 2)
                sample["punct_error_rate"] = round(100 * punctuation_rates.punct_er, 2)

            samples_with_metrics.append(sample)

    if output_manifest_path is not None:
        with open(output_manifest_path, 'w') as output:
            for sample in samples_with_metrics:
                line = json.dumps(sample)
                output.writelines(f'{line}\n')
        logging.info(f'Output manifest saved: {output_manifest_path}')

    return samples_with_metrics


class PunctuationCapitalization:
    def __init__(self, punctuation_marks: str):
        """
        Class for text processing with punctuation and capitalization. Can be used with class TextProcessingConfig.

        Args:
            punctuation_marks (str): String with punctuation marks to process.
        Example: punctuation_marks = '.,?'
        """
        if punctuation_marks:
            self.regex_punctuation = re.compile(fr"([{''.join(punctuation_marks)}])")
            self.regex_extra_space = re.compile('\s{2,}')
        else:
            self.regex_punctuation = None

    def separate_punctuation(self, lines: List[str]) -> List[str]:
        if self.regex_punctuation is not None:
            return [
                self.regex_extra_space.sub(' ', self.regex_punctuation.sub(r' \1 ', line)).strip() for line in lines
            ]
        else:
            return lines

    def do_lowercase(self, lines: List[str]) -> List[str]:
        return [line.lower() for line in lines]

    def rm_punctuation(self, lines: List[str]) -> List[str]:
        if self.regex_punctuation is not None:
            return [self.regex_extra_space.sub(' ', self.regex_punctuation.sub(' ', line)).strip() for line in lines]
        else:
            return lines


@dataclass
class TextProcessingConfig:
    # Punctuation marks to process. Example: ".,?"
    punctuation_marks: str = ""

    # Whether to apply lower case conversion on the training text.
    do_lowercase: bool = False

    # Whether to remove punctuation marks from text.
    rm_punctuation: bool = False

    # Whether to separate punctuation with the previouse word by space.
    separate_punctuation: bool = True
