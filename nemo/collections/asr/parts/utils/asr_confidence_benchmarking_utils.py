# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
import copy
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import texterrors
import torch
from omegaconf import open_dict

from nemo.collections.asr.models import ASRModel, EncDecRNNTModel
from nemo.collections.asr.parts.utils.confidence_metrics import (
    auc_nt,
    auc_pr,
    auc_roc,
    auc_yc,
    ece,
    nce,
    save_confidence_hist,
    save_custom_confidence_curve,
    save_nt_curve,
    save_pr_curve,
    save_roc_curve,
)
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis


def get_correct_marks(r: Union[List[int], List[str]], h: Union[List[int], List[str]]) -> List[bool]:
    """Get correct marks by aligning the reference text with a hypothesis.

    This method considers only insertions and substitutions as incorrect marks.
    """
    return [
        a == b
        for a, b in zip(*(texterrors.align_texts([str(rr) for rr in r], [str(hh) for hh in h], False)[:-1]))
        if b != "<eps>"
    ]


def get_token_targets_with_confidence(hyp: Hypothesis) -> List[Tuple[str, float]]:
    return [(y, c) for y, c in zip(hyp.y_sequence, hyp.token_confidence)]


def get_word_targets_with_confidence(hyp: Hypothesis) -> List[Tuple[str, float]]:
    return [(y, c) for y, c in zip(hyp.words, hyp.word_confidence)]


def run_confidence_benchmark(
    model: ASRModel,
    target_level: str,
    filepaths: List[str],
    reference_texts: List[str],
    batch_size: int = 8,
    num_workers: int = 4,
    plot_dir: Optional[Union[str, Path]] = None,
    autocast: Optional = None,
):
    """Run benchmark and plot histograms and curves, if plot_dir is provided.

    Returns:
        Dictionary with benchmark results of the following scheme:
        `level: (auc_roc, auc_pr, auc_nt, nce, ece, auc_yc, std_yc, max_yc)` with `level` being 'token' or 'word'.
    """
    draw_plot = plot_dir is not None
    if isinstance(plot_dir, str):
        plot_dir = Path(plot_dir)
    is_rnnt = isinstance(model, EncDecRNNTModel)

    # setup autocast if necessary
    if autocast is None:

        @contextlib.contextmanager
        def autocast():
            yield

    # transcribe audio
    with autocast():
        with torch.no_grad():
            transcriptions = model.transcribe(
                paths2audio_files=filepaths, batch_size=batch_size, return_hypotheses=True, num_workers=num_workers
            )
    if is_rnnt:
        transcriptions = transcriptions[0]

    levels = []
    if target_level != "word":
        levels.append("token")
    if target_level != "token":
        levels.append("word")
    results = {}
    for level in levels:
        if level == "token":
            targets_with_confidence = [get_token_targets_with_confidence(tran) for tran in transcriptions]
            correct_marks = [
                get_correct_marks(model.tokenizer.text_to_ids(r), model.tokenizer.text_to_ids(h.text))
                for r, h in zip(reference_texts, transcriptions)
            ]
        else:  # "word"
            targets_with_confidence = [get_word_targets_with_confidence(tran) for tran in transcriptions]
            correct_marks = [get_correct_marks(r.split(), h.words) for r, h in zip(reference_texts, transcriptions)]

        y_true, y_score = np.array(
            [[f, p[1]] for cm, twc in zip(correct_marks, targets_with_confidence) for f, p in zip(cm, twc)]
        ).T
        # output scheme: yc.mean(), yc.max(), yc.std() or yc.mean(), yc.max(), yc.std(), (thresholds, yc)
        result_yc = auc_yc(y_true, y_score, return_std_maximum=True, return_curve=draw_plot)
        # output scheme: ece or ece, (thresholds, ece_curve)
        results_ece = ece(y_true, y_score, return_curve=draw_plot)
        results[level] = [
            auc_roc(y_true, y_score),
            auc_pr(y_true, y_score),
            auc_nt(y_true, y_score),
            nce(y_true, y_score),
            results_ece if isinstance(results_ece, float) else results_ece[0],
        ] + list(result_yc[:3])

        if draw_plot:
            os.makedirs(plot_dir, exist_ok=True)

            mask_correct = y_true == 1
            y_score_correct = y_score[mask_correct]
            y_score_incorrect = y_score[~mask_correct]
            # histogram of the correct distribution
            save_confidence_hist(y_score_correct, plot_dir, level + "_" + "hist_correct")
            # histogram of the incorrect distribution
            save_confidence_hist(y_score_incorrect, plot_dir, level + "_" + "hist_incorrect")
            # AUC-ROC curve
            save_roc_curve(y_true, y_score, plot_dir, level + "_" + "roc")
            # AUC-PR curve
            save_pr_curve(y_true, y_score, plot_dir, level + "_" + "pr")
            # AUC-NT curve
            save_nt_curve(y_true, y_score, plot_dir, level + "_" + "nt")
            # AUC-YC curve
            yc_thresholds, yc_values = result_yc[-1]
            save_custom_confidence_curve(
                yc_thresholds,
                yc_values,
                plot_dir,
                level + "_" + "yc",
                "Threshold",
                "True positive rate − False Positive Rate",
            )
            # ECE curve
            ece_thresholds, ece_values = results_ece[-1]
            ece_values /= max(ece_values)
            save_custom_confidence_curve(
                ece_thresholds, ece_values, plot_dir, level + "_" + "ece", "Threshold", "|Accuracy − Confidence score|"
            )

    return results


def apply_confidence_parameters(decoding_cfg, hp):
    """Apply parameters from a parameter grid to a decoding config.

    Returns:
        Updated decoding config.
    """
    new_decoding_cfg = copy.deepcopy(decoding_cfg)
    confidence_cfg_fields = ("aggregation", "exclude_blank")
    confidence_method_cfg_fields = ("name", "alpha", "entropy_type", "entropy_norm")
    with open_dict(new_decoding_cfg):
        for p, v in hp.items():
            if p in confidence_cfg_fields:
                new_decoding_cfg.confidence_cfg[p] = v
            elif p in confidence_method_cfg_fields:
                new_decoding_cfg.confidence_cfg.method_cfg[p] = v
    return new_decoding_cfg
