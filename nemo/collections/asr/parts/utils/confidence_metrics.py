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

import math
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def auc_roc(y_true: Union[List[int], np.ndarray], y_score: Union[List[float], np.ndarray]) -> float:
    """Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.

    Note: If only one class is present in y_true, 0.5 is returned.
    """
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    assert len(y_true) == len(y_score)
    assert np.all(y_true >= 0) and np.all(y_true <= 1)
    if np.all(y_true == 0) or np.all(y_true == 1):
        return 0.5
    return roc_auc_score(y_true, y_score)


def auc_pr(y_true: Union[List[int], np.ndarray], y_score: Union[List[float], np.ndarray]) -> float:
    """Compute Area Under the Precision-Recall Curve (PR AUC) from prediction scores.

    Note: If only regatives are present in y_true, 0.0 is returned.
    """
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    assert len(y_true) == len(y_score)
    assert np.all(y_true >= 0) and np.all(y_true <= 1)
    if np.all(y_true == 0):
        return 0.0
    return average_precision_score(y_true, y_score)


def auc_nt(y_true: Union[List[int], np.ndarray], y_score: Union[List[float], np.ndarray]) -> float:
    """Compute Area Under the Negative Predictive Value vs. True Negative Rate Curve (NT AUC) from prediction scores.

    This metric can be thought of as a PR AUC in which errors are treated as positives.

    Note: If only positives are present in y_true, 0.0 is returned.
    """
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    assert len(y_true) == len(y_score)
    assert np.all(y_true >= 0) and np.all(y_true <= 1)
    if np.all(y_true == 1):
        return 0.0
    return average_precision_score(1 - y_true, 1 - y_score)


def nce(y_true: Union[List[int], np.ndarray], y_score: Union[List[float], np.ndarray]) -> float:
    """Compute Normalized Cross Entropy (NCE) from prediction scores. Also known as the Normalized Mutual Information.

    NCE measures how close the correct prediction scores are to one and the incorrect prediction scores are to zero.
    Negative NCE values indicate that the classifier performs worse than the setting all prediction scores
    as the proportion of correct predictions.

    Note: If only one class is present in y_true, 0.5 is returned.
    """
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    assert len(y_true) == len(y_score)
    assert np.all(y_true >= 0) and np.all(y_true <= 1)
    if np.all(y_true == 0) or np.all(y_true == 1):
        return -math.inf
    p = y_true.mean()
    eps = 1e-15
    Hp = -(math.log(p + eps) * p + math.log(1 - p + eps) * (1 - p))
    return (Hp - log_loss(y_true, y_score)) / Hp


def ece(
    y_true: Union[List[int], np.ndarray],
    y_score: Union[List[float], np.ndarray],
    n_bins: int = 100,
    return_curve: bool = False,
) -> Union[float, Tuple[float, Tuple[List[int], List[float]]]]:
    """Compute Expected Calibration Error (ECE) from prediction scores.

    ECE measures how close the correct prediction scores are to one and the incorrect prediction scores are to zero.
    ECE ranges from zero to one with the best value zero (the lower the value, the better).
    """
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    assert len(y_true) == len(y_score)
    assert np.all(y_true >= 0) and np.all(y_true <= 1)
    py = np.array([1 - y_score, y_score]).T
    acc, conf = np.zeros(n_bins), np.zeros(n_bins)
    Bm = np.zeros(n_bins)
    ece_curve = []
    thresholds = []
    for m in range(n_bins):
        a, b = m / n_bins, (m + 1) / n_bins
        threshold = (a + b) / 2
        thresholds.append(threshold)
        py_index = (py.T[1] >= threshold).astype(int)
        py_value = py[np.arange(len(py_index)), py_index]
        bin_range = ((py_value > a) & (py_value <= b)).nonzero()[0]
        Bm[m] = len(bin_range)
        if Bm[m] > 0:
            acc[m] = (py_index[bin_range] == y_true[bin_range]).sum() / Bm[m]
            conf[m] = py_value[bin_range].sum() / Bm[m]
        ece_curve.append(Bm[m] * np.abs(acc[m] - conf[m]))
    ece = sum(ece_curve) / sum(Bm)
    if return_curve:
        return ece, (thresholds, ece_curve)
    else:
        return ece


def auc_yc(
    y_true: Union[List[int], np.ndarray],
    y_score: Union[List[float], np.ndarray],
    n_bins: int = 100,
    return_std_maximum: bool = False,
    return_curve: bool = False,
) -> Union[
    float,
    Tuple[float, Tuple[List[int], List[float]]],
    Tuple[float, float, float],
    Tuple[float, float, float, Tuple[List[int], List[float]]],
]:
    """Compute Area Under the Youden's Curve (YC AUC) from prediction scores.

    YC AUC represents the rate of the effective threshold range.

    If return_std_maximum is set to True, std and maximum values of the Youden's Curve are returned with the AUC.

    Note: If only one class is present in y_true, zeroes are returned for every entity.
    """
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    thresholds = np.linspace(0, 1, n_bins + 1)
    assert len(y_true) == len(y_score)
    assert np.all(y_true >= 0) and np.all(y_true <= 1)
    if np.all(y_true == 0) or np.all(y_true == 1):
        if return_std_maximum and return_curve:
            return 0.0, 0.0, 0.0, (thresholds, np.zeros(len(thresholds)))
        elif return_std_maximum:
            return 0.0, 0.0, 0.0
        elif return_curve:
            return 0.0, (thresholds, np.zeros(len(thresholds)))
        else:
            return 0.0
    mask_correct = y_true == 1
    count_correct = max(len(mask_correct.nonzero()[0]), 1)
    count_incorrect = max(len(y_true) - count_correct, 1)
    y_score_correct = y_score[mask_correct]
    y_score_incorrect = y_score[~mask_correct]
    yc = []
    for threshold in thresholds:
        tnr = len((y_score_incorrect < threshold).nonzero()[0]) / count_incorrect
        fnr = len((y_score_correct < threshold).nonzero()[0]) / count_correct
        yc.append(abs(tnr - fnr))
    yc = np.array(yc)
    if return_std_maximum and return_curve:
        return yc.mean(), yc.std(), yc.max(), (thresholds, yc)
    elif return_std_maximum:
        return yc.mean(), yc.std(), yc.max()
    elif return_curve:
        return yc.mean(), (thresholds, yc)
    else:
        return yc.mean()


def save_confidence_hist(y_score: Union[List[float], np.ndarray], plot_dir: Union[str, Path], name: str = "hist"):
    os.makedirs(plot_dir, exist_ok=True)
    plt.hist(np.array(y_score), 50, range=(0, 1))
    plt.title(name)
    plt.xlabel("Confidence score")
    plt.ylabel("Count")
    plt.savefig(Path(plot_dir) / Path(name + ".png"), dpi=300)
    plt.clf()


def save_roc_curve(
    y_true: Union[List[int], np.ndarray],
    y_score: Union[List[float], np.ndarray],
    plot_dir: Union[str, Path],
    name: str = "roc",
):
    assert len(y_true) == len(y_score)
    os.makedirs(plot_dir, exist_ok=True)
    fpr, tpr, _ = roc_curve(1 - np.array(y_true), 1 - np.array(y_score))
    RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
    plt.title(name)
    plt.savefig(Path(plot_dir) / Path(name + ".png"), dpi=300)
    plt.clf()


def save_pr_curve(
    y_true: Union[List[int], np.ndarray],
    y_score: Union[List[float], np.ndarray],
    plot_dir: Union[str, Path],
    name: str = "pr",
):
    assert len(y_true) == len(y_score)
    os.makedirs(plot_dir, exist_ok=True)
    precision, recall, _ = precision_recall_curve(np.array(y_true), np.array(y_score))
    PrecisionRecallDisplay(precision=precision, recall=recall).plot()
    plt.title(name)
    plt.savefig(Path(plot_dir) / Path(name + ".png"), dpi=300)
    plt.clf()


def save_nt_curve(
    y_true: Union[List[int], np.ndarray],
    y_score: Union[List[float], np.ndarray],
    plot_dir: Union[str, Path],
    name: str = "nt",
):
    assert len(y_true) == len(y_score)
    os.makedirs(plot_dir, exist_ok=True)
    precision, recall, _ = precision_recall_curve(1 - np.array(y_true), 1 - np.array(y_score))
    PrecisionRecallDisplay(precision=precision, recall=recall).plot()
    plt.title(name)
    plt.savefig(Path(plot_dir) / Path(name + ".png"), dpi=300)
    plt.clf()


def save_custom_confidence_curve(
    thresholds: Union[List[float], np.ndarray],
    values: Union[List[float], np.ndarray],
    plot_dir: Union[str, Path],
    name: str = "my_awesome_curve",
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
):
    assert len(thresholds) == len(values)
    os.makedirs(plot_dir, exist_ok=True)
    plt.plot(thresholds, values)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.title(name)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    plt.savefig(Path(plot_dir) / Path(name + ".png"), dpi=300)
    plt.clf()
