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

from nemo.collections.asr.parts.utils.eval_utils import cal_write_text_metric, cal_write_wer
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
            wer_config = cfg.metrics.wer
            output_manifest_w_wer, total_res, _ = cal_write_wer(
                pred_manifest=output_path,
                gt_text_attr_name=wer_config.gt_text_attr_name,
                pred_text_attr_name="pred_text",
                output_filename=None,
                clean_groundtruth_text=wer_config.clean_groundtruth_text,
                langid=wer_config.langid,
                use_cer=wer_config.use_cer,
                ignore_capitalization=wer_config.ignore_capitalization,
                ignore_punctuation=wer_config.ignore_punctuation,
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
                bleu_config = cfg.metrics.bleu
                output_manifest_w_bleu, total_res, _ = cal_write_text_metric(
                    pred_manifest=output_path,
                    pred_text_attr_name="pred_translation",
                    gt_text_attr_name=bleu_config.gt_text_attr_name,
                    output_filename=None,
                    ignore_capitalization=bleu_config.ignore_capitalization,
                    ignore_punctuation=bleu_config.ignore_punctuation,
                    strip_punc_space=bleu_config.strip_punc_space,
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
