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


def calculate_pipeline_laal(
    output: dict, durations: dict[str, float], manifest: list[dict], cfg: DictConfig
) -> float | None:
    """
    Calculate the LAAL of the pipeline output.
    Args:
        output: Dictionary containing the pipeline output.
        durations: Dictionary containing the duration of each audio file.
        manifest: List of dictionaries containing the ground truth translation for each audio file.
        cfg: Configuration object.
    Returns:
        float | None: Length-Adaptive Average Lagging (LAAL) for Simultaneous Speech Translation in milliseconds
    """

    if not cfg.enable_nmt:
        logging.warning("LAAL calculation is skipped because NMT is not enabled.")
        return None

    if manifest is None:
        logging.warning("LAAL calculation is skipped because manifest is not provided.")
        return None

    gt_text_attr_name = cfg.metrics.nmt.gt_text_attr_name
    ref_translations = {item["audio_filepath"]: item[gt_text_attr_name] for item in manifest}

    laal_list = []
    for stream_id, stream_output in output.items():
        audio_filepath = stream_output["audio_filepath"]
        duration = durations[audio_filepath] * 1000  # ms
        num_words_in_ref_translation = len(ref_translations[audio_filepath].split())
        translation_segments = stream_output["translation_segments"]

        lagging = []
        for translation, delay in translation_segments:
            translation = translation.strip()
            if not translation:
                continue
            cur_words = translation.split()
            lag = min(delay * 1000, duration)
            lagging.extend([lag] * len(cur_words))

        if len(lagging) == 0:
            lagging.append(0)

        laal = compute_laal(lagging, duration, num_words_in_ref_translation)
        laal_list.append(laal)

    return sum(laal_list) / len(laal_list)
