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

from omegaconf import DictConfig

from nemo.collections.asr.parts.utils.eval_utils import cal_write_text_metric, cal_write_wer, compute_laal
from nemo.utils import logging


def evaluate_pipeline(output_path: str, cfg: DictConfig) -> None:
    """
    Evaluate pipeline output and overwrite the output file with the metrics.
    Args:
        output_path: Path to the output file.
        cfg: Configuration object.
    """

    if cfg.calculate_wer:
        try:
            asr_metrics_cfg = cfg.metrics.asr
            output_manifest_w_wer, total_res, _ = cal_write_wer(
                pred_manifest=output_path,
                gt_text_attr_name=asr_metrics_cfg.gt_text_attr_name,
                pred_text_attr_name="pred_text",
                output_filename=None,
                clean_groundtruth_text=asr_metrics_cfg.clean_groundtruth_text,
                langid=asr_metrics_cfg.langid,
                use_cer=asr_metrics_cfg.use_cer,
                ignore_capitalization=asr_metrics_cfg.ignore_capitalization,
                ignore_punctuation=asr_metrics_cfg.ignore_punctuation,
            )
            if output_manifest_w_wer:
                logging.info(f"Writing prediction and error rate of each sample to {output_manifest_w_wer}!")
                logging.info(f"{total_res}")
            else:
                logging.warning(
                    "WER calculation is skipped because the output manifest does not contain ground truth text."
                )
        except Exception as e:
            logging.error(f"Error calculating WER: {e}")

    if cfg.calculate_bleu:
        if cfg.enable_nmt:
            try:
                nmt_metrics_cfg = cfg.metrics.nmt
                output_manifest_w_bleu, total_res, _ = cal_write_text_metric(
                    pred_manifest=output_path,
                    pred_text_attr_name="pred_translation",
                    gt_text_attr_name=nmt_metrics_cfg.gt_text_attr_name,
                    output_filename=None,
                    ignore_capitalization=nmt_metrics_cfg.ignore_capitalization,
                    ignore_punctuation=nmt_metrics_cfg.ignore_punctuation,
                    strip_punc_space=nmt_metrics_cfg.strip_punc_space,
                )
                if output_manifest_w_bleu:
                    logging.info(f"Writing prediction and BLEU score of each sample to {output_manifest_w_bleu}!")
                    logging.info(f"{total_res}")
                else:
                    logging.warning(
                        "BLEU calculation is skipped because the output manifest does not contain ground truth translation."
                    )
            except Exception as e:
                logging.error(f"Error calculating BLEU score: {e}")
        else:
            logging.warning("BLEU calculation is skipped because NMT is not enabled.")


def _compute_pipeline_laal(
    output: dict, durations: dict[str, float], manifest: list[dict], gt_text_attr_name: str, segments_key: str
) -> float | None:
    """
    Shared LAAL core over per-step ``(text, delay)`` segments, averaged over streams with a reference.
    Each word inherits its segment's delay (capped at the audio duration).

    Args:
        output: Pipeline output; each stream has a `segments_key` list of ``(text, delay_in_seconds)``.
        durations: Duration (seconds) of each audio file.
        manifest: Ground-truth entries (reference word count via `gt_text_attr_name`).
        gt_text_attr_name: Manifest attribute holding the reference text/translation.
        segments_key: Key of the per-step ``(text, delay)`` list in each stream output.
    Returns:
        float | None: Length-Adaptive Average Lagging (ms), or None if no stream had a reference.
    """
    ref_texts = {item["audio_filepath"]: item[gt_text_attr_name] for item in manifest}

    laal_list = []
    for stream_output in output.values():
        audio_filepath = stream_output["audio_filepath"]
        if audio_filepath not in ref_texts:
            continue
        duration = durations[audio_filepath] * 1000
        num_words_in_ref = len(ref_texts[audio_filepath].split())

        lagging = []
        for text, delay in stream_output.get(segments_key, []):
            text = text.strip()
            if not text:
                continue
            cur_words = text.split()
            lag = min(delay * 1000, duration)
            lagging.extend([lag] * len(cur_words))

        if len(lagging) == 0:
            lagging.append(0)

        laal_list.append(compute_laal(lagging, duration, num_words_in_ref))

    if not laal_list:
        return None

    return sum(laal_list) / len(laal_list)


def calculate_translation_laal(
    output: dict, durations: dict[str, float], manifest: list[dict], cfg: DictConfig
) -> float | None:
    """
    Translation LAAL of the pipeline output.

    Args:
        output: Dictionary containing the pipeline output.
        durations: Dictionary containing the duration of each audio file.
        manifest: List of dictionaries containing the ground truth translation for each audio file.
        cfg: Configuration object.
    Returns:
        float | None: Length-Adaptive Average Lagging (ms), or None if NMT is off or no manifest is given.
    """

    if not cfg.enable_nmt:
        logging.warning("LAAL calculation is skipped because NMT is not enabled.")
        return None

    if manifest is None:
        logging.warning("LAAL calculation is skipped because manifest is not provided.")
        return None

    return _compute_pipeline_laal(
        output, durations, manifest, cfg.metrics.nmt.gt_text_attr_name, "translation_segments"
    )


def calculate_asr_laal(
    output: dict, durations: dict[str, float], manifest: list[dict], cfg: DictConfig
) -> float | None:
    """
    ASR LAAL of the pipeline output: how far behind the audio the transcription is committed -- a proxy
    for end-of-utterance latency.

    Args:
        output: Dictionary containing the pipeline output (each stream has an `asr_segments` list of
            ``(text, delay_in_seconds)`` pairs).
        durations: Dictionary containing the duration of each audio file.
        manifest: List of dictionaries containing the ground truth text for each audio file.
        cfg: Configuration object.
    Returns:
        float | None: Length-Adaptive Average Lagging (ms), or None if EoU is disabled or no manifest is given.
    """

    # EoU disabled (stop_history_eou < 0): one segment finalized at stream end, so no latency signal.
    if cfg.get("endpointing", {}).get("stop_history_eou", -1) < 0:
        logging.warning("ASR LAAL calculation is skipped because end-of-utterance detection is disabled.")
        return None

    if manifest is None:
        logging.warning("ASR LAAL calculation is skipped because manifest is not provided.")
        return None

    return _compute_pipeline_laal(output, durations, manifest, cfg.metrics.asr.gt_text_attr_name, "asr_segments")
