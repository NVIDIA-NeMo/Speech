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

__all__ = ['list2str', 'tensor2list', 'plot_confusion_matrix', 'get_classification_report']

import os
import time
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from torch import Tensor

from nemo.collections.nlp.modules.common.megatron.utils import erf_gelu
from nemo.collections.nlp.modules.common.megatron.utils import openai_gelu as openai_gelu_func
from nemo.collections.nlp.modules.common.megatron.utils import squared_relu
from nemo.utils import logging


def torch_dtype_from_precision(precision: Union[int, str], megatron_amp_O2: Optional[bool] = None) -> torch.dtype:
    """ Mapping from PTL precision types to corresponding PyTorch parameter datatype."""
    if megatron_amp_O2 is not None and megatron_amp_O2 is False:
        return torch.float32

    if precision in ['bf16', 'bf16-mixed']:
        return torch.bfloat16
    elif precision in [16, '16', '16-mixed']:
        return torch.float16
    elif precision in [32, '32', '32-true']:
        return torch.float32
    else:
        raise ValueError(f"Could not parse the precision of `{precision}` to a valid torch.dtype")


def list2str(l: List[int]) -> str:
    """ Converts list to a string"""
    return ' '.join([str(x) for x in l])


def tensor2list(tensor: Tensor) -> List[Union[int, float]]:
    """ Converts tensor to a list """
    return tensor.detach().cpu().tolist()


def plot_confusion_matrix(
    labels: List[int],
    preds: List[int],
    graph_fold: str,
    label_ids: Dict[str, int] = None,
    normalize: bool = False,
    prefix: str = '',
):
    '''
    Plot confusion matrix.
    Args:
      labels: ground truth labels
      preds: model predictions
      graph_fold: path to a folder to store confusion matrix plot
      label_ids: str label to id map, for example: {'O': 0, 'LOC': 1}
      normalize: whether to normalize confusion matrix
      prefix: prefix for the plot name
    '''
    if label_ids is None:
        _plot_confusion_matrix(labels, preds, graph_fold)

    else:
        # remove labels from label_ids that don't appear in the dev set
        used_labels = set(labels) | set(preds)
        label_ids = {k: label_ids[k] for k, v in label_ids.items() if v in used_labels}

        ids_to_labels = {label_ids[k]: k for k in label_ids}
        classes = [ids_to_labels[id] for id in sorted(label_ids.values())]

        title = 'Confusion_matrix'
        cm = confusion_matrix(labels, preds)
        if normalize:
            sums = cm.sum(axis=1)[:, np.newaxis]
            sums = np.where(sums == 0, 1, sums)
            cm = cm.astype('float') / sums
            title = 'Normalized_' + title

        fig = plt.figure()
        ax = fig.add_subplot(111)

        cax = ax.matshow(cm)

        ax.set_xticks(np.arange(-1, len(classes)))
        ax.set_yticks(np.arange(-1, len(classes)))
        ax.set_xticklabels([''] + classes, rotation=90)
        ax.set_yticklabels([''] + classes)
        ax.set_ylabel('True')
        ax.set_xlabel('Predicted')

        os.makedirs(graph_fold, exist_ok=True)
        fig.colorbar(cax)

        title = (prefix + title).strip()
        fig_name = os.path.join(graph_fold, title + '_' + time.strftime('%Y%m%d-%H%M%S'))
        plt.savefig(fig_name)
        logging.info(f'Confusion matrix saved to {fig_name}')


def _plot_confusion_matrix(labels: List[int], preds: List[int], graph_fold: str):
    """
    Plot confusion matrix
    Args:
        labels: ground truth labels
        preds: model predictions
        graph_fold: path to a folder to store confusion matrix plot
    """
    cm = confusion_matrix(labels, preds)
    logging.info(f'Confusion matrix:\n{cm}')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    plt.title('Confusion matrix of the classifier')
    fig.colorbar(cax)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    os.makedirs(graph_fold, exist_ok=True)
    plt.savefig(os.path.join(graph_fold, time.strftime('%Y%m%d-%H%M%S')))


def get_classification_report(labels, preds, label_ids, output_dict=False):
    """
    Returns classification report
    """
    # remove labels from label_ids that don't appear in predictions or ground truths
    used_labels = set(labels) | set(preds)
    labels_names = [
        k + ' (label id: ' + str(v) + ')'
        for k, v in sorted(label_ids.items(), key=lambda item: item[1])
        if v in used_labels
    ]

    return classification_report(labels, preds, target_names=labels_names, digits=4, output_dict=output_dict)


def is_last_rank():
    return torch.distributed.get_rank() == (torch.distributed.get_world_size() - 1)


def get_last_rank():
    return torch.distributed.get_world_size() - 1


def activation_to_func(activation: str, openai_gelu: bool = False, onnx_safe: bool = False) -> Callable:
    """ Converts an activation function represented as a string to a function.

    Args:
        activation (str): string representation of an activation function, typically gotten from the model config.
        openai_gelu (bool): whether to use the OpenAI GELU implementation. Used with HF compatibility.
        onnx_safe (bool): whether to use the ONNX-compatible implementation of GELU.
    
    Returns:
        Callable: the activation function.
    """

    supported_activations = [
        'gelu',
        'geglu',
        'reglu',
        'swiglu',
        'squared-relu',
        'fast-geglu',
        'fast-swiglu',
        'fast-reglu',
    ]

    if activation not in supported_activations:
        raise ValueError(f"Unsupported activation {activation}. Supported activations: {supported_activations} ")

    # Give openai_gelu precedence over other activations if set, for HF compatibility.
    # Normally this is off and shouldn't affect regular model training.
    if openai_gelu:
        activation_func = openai_gelu_func
    elif activation in ["gelu", "geglu", "fast-geglu"]:
        activation_func = F.gelu
    elif onnx_safe:
        activation_func = erf_gelu
    elif activation in ["reglu", "fast-reglu"]:
        activation_func = F.relu
    elif activation in ["swiglu", "fast-swiglu"]:
        # SiLU or sigmoid linear unit is the same as swish with beta = 1 (which is what https://arxiv.org/pdf/2002.05202.pdf uses.)
        activation_func = F.silu
    elif activation == 'squared-relu':
        activation_func = squared_relu

    return activation_func
