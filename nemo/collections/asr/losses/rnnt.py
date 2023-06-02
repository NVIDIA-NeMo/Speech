# ! /usr/bin/python
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

# Copyright 2018-2019, Mingkun Huang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import operator
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set

import torch
from omegaconf import DictConfig, OmegaConf

from nemo.collections.asr.losses.rnnt_pytorch import MultiblankRNNTLossPytorch, RNNTLossPytorch, TDTLossPytorch
from nemo.core.classes import Loss, typecheck
from nemo.core.neural_types import LabelsType, LengthsType, LogprobsType, LossType, NeuralType
from nemo.core.utils.k2_utils import K2_INSTALLATION_MESSAGE
from nemo.core.utils.numba_utils import NUMBA_INSTALLATION_MESSAGE
from nemo.utils import logging, model_utils

try:
    import warprnnt_pytorch as warprnnt

    WARP_RNNT_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    WARP_RNNT_AVAILABLE = False

try:
    from nemo.collections.asr.parts.numba.rnnt_loss import MultiblankRNNTLossNumba, RNNTLossNumba, TDTLossNumba

    NUMBA_RNNT_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    NUMBA_RNNT_AVAILABLE = False

try:
    from nemo.collections.asr.parts.k2.graph_transducer import GraphRnntLoss
    from nemo.collections.asr.parts.k2.w_transducer import GraphWTransducerLoss

    K2_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    K2_AVAILABLE = False

WARP_RNNT_INSTALLATION_MESSAGE = (
    "Could not import `warprnnt_pytorch`.\n"
    "Please visit https://github.com/HawkAaron/warp-transducer "
    "and follow the steps in the readme to build and install the "
    "pytorch bindings for RNNT Loss, or use the provided docker "
    "container that supports RNN-T loss."
)


@dataclass
class RNNTLossConfig:
    loss_name: str
    lib_name: str
    is_available: bool = False
    installation_msg: str = ""
    min_version: Optional[str] = None
    force_float32: bool = True  # default True for now for all losses except graph-based


# Resolved list of available RNNT losses
RNNT_LOSS_RESOLVER = {
    "warprnnt": RNNTLossConfig(
        loss_name="warprnnt",
        lib_name="warprnnt_pytorch",
        is_available=WARP_RNNT_AVAILABLE,
        installation_msg=WARP_RNNT_INSTALLATION_MESSAGE,
        force_float32=True,
    ),
    "warprnnt_numba": RNNTLossConfig(
        loss_name="warprnnt_numba",
        lib_name="numba",
        min_version='0.53.0',
        is_available=NUMBA_RNNT_AVAILABLE,
        installation_msg=NUMBA_INSTALLATION_MESSAGE,
        force_float32=True,
    ),
    "pytorch": RNNTLossConfig(
        loss_name="pytorch",
        lib_name="torch",
        min_version='0.0',
        is_available=True,
        installation_msg="Pure Pytorch implementation of RNN-T loss. Slow and for debugging purposes only.",
        force_float32=True,
    ),
    "multiblank_rnnt": RNNTLossConfig(
        loss_name="multiblank_rnnt",
        lib_name="numba",
        min_version='0.53.0',
        is_available=NUMBA_RNNT_AVAILABLE,
        installation_msg=NUMBA_INSTALLATION_MESSAGE,
        force_float32=True,
    ),
    "multiblank_rnnt_pytorch": RNNTLossConfig(
        loss_name="pytorch",
        lib_name="torch",
        min_version='0.0',
        is_available=True,
        installation_msg="Pure Pytorch implementation of Multiblank RNN-T loss. Slow and for debugging purposes only.",
        force_float32=True,
    ),
    "graph_w_transducer": RNNTLossConfig(
        loss_name="graph_w_transducer",
        lib_name="k2",
        is_available=K2_AVAILABLE,
        installation_msg=K2_INSTALLATION_MESSAGE,
        force_float32=False,
    ),
    "graph_rnnt": RNNTLossConfig(
        loss_name="graph_rnnt",
        lib_name="k2",
        is_available=K2_AVAILABLE,
        installation_msg=K2_INSTALLATION_MESSAGE,
        force_float32=False,
    ),
    "tdt": RNNTLossConfig(
        loss_name="tdt",
        lib_name="numba",
        min_version='0.53.0',
        is_available=NUMBA_RNNT_AVAILABLE,
        installation_msg=NUMBA_INSTALLATION_MESSAGE,
    ),
    "tdt_pytorch": RNNTLossConfig(
        loss_name="tdt_pytorch",
        lib_name="torch",
        min_version='0.0',
        is_available=True,
        installation_msg="Pure Pytorch implementation of TDT loss. Slow and for debugging purposes only.",
    ),
}

RNNT_LOSS_RESOLVER['default'] = RNNT_LOSS_RESOLVER['warprnnt_numba']


def _warn_unused_additional_kwargs(loss_name, kwargs):
    if len(kwargs) > 0:
        logging.warning(
            f"Loss function `{loss_name}` was provided with following additional kwargs,\n"
            f"however they were ignored as it is unused.\n"
            f"{kwargs}"
        )


def _clean_kwargs(
    loss_name: str, kwargs: Optional[Dict[str, Any]], init_method: Callable, ignore_params: Optional[Set[str]] = None
) -> Dict[str, Any]:
    """
    Cleans kwargs for the given loss function. Warn if there are unused kwargs.

    Args:
        loss_name: name of the loss function
        kwargs: kwargs to clean
        init_method: LossClass.__init__ method
        ignore_params: set of argument names for init_method to ignore

    Returns:
        only used kwargs for the given `init_method`
    """
    if not kwargs:
        return {}
    init_params = set(inspect.signature(init_method).parameters.keys()) - {"self"}
    if ignore_params is not None:
        init_params -= ignore_params
    unused_kwargs = dict()
    used_kwargs = dict()
    for key, value in kwargs.items():
        if key not in init_params:
            unused_kwargs[key] = value
        else:
            used_kwargs[key] = value
    if len(unused_kwargs) > 0:
        _warn_unused_additional_kwargs(loss_name, unused_kwargs)
    return used_kwargs


def resolve_rnnt_default_loss_name() -> str:
    return RNNT_LOSS_RESOLVER['default'].loss_name


def resolve_rnnt_loss(loss_name: str, blank_idx: int, loss_kwargs: dict = None) -> torch.nn.Module:
    loss_function_names = list(RNNT_LOSS_RESOLVER.keys())

    if loss_name not in loss_function_names:
        raise ValueError(
            f"Provided `loss_name` {loss_name} not in list of available RNNT losses \n" f"{loss_function_names}"
        )

    all_available_losses = {name: config for name, config in RNNT_LOSS_RESOLVER.items() if config.is_available}

    loss_config = RNNT_LOSS_RESOLVER[loss_name]  # type: RNNTLossConfig

    # Re-raise import error with installation message
    if not loss_config.is_available:
        msg = (
            f"Installed RNNT losses are : {list(all_available_losses.keys())}.\n"
            f"****************************************************************\n"
            f"To install the selected loss function, please follow the steps below:\n"
            f"{loss_config.installation_msg}"
        )
        raise ImportError(msg)

    # Library version check
    if loss_config.min_version is not None:
        ver_matched, msg = model_utils.check_lib_version(
            loss_config.lib_name, checked_version=loss_config.min_version, operator=operator.ge
        )

        if ver_matched is False:
            msg = (
                f"{msg}\n"
                f"****************************************************************\n"
                f"To update the selected loss function, please follow the steps below:\n"
                f"{loss_config.installation_msg}"
            )
            raise RuntimeError(msg)

    # Resolve loss functions sequentially
    loss_kwargs = {} if loss_kwargs is None else loss_kwargs

    if isinstance(loss_kwargs, DictConfig):
        loss_kwargs = OmegaConf.to_container(loss_kwargs, resolve=True)

    # Get actual loss name for `default`
    if loss_name == 'default':
        loss_name = loss_config.loss_name

    """
    Resolve RNNT loss functions
    """
    if loss_name == 'warprnnt':
        loss_func = warprnnt.RNNTLoss(blank=blank_idx, reduction='none')
        _warn_unused_additional_kwargs(loss_name, loss_kwargs)

    elif loss_name == 'warprnnt_numba':
        fastemit_lambda = loss_kwargs.pop('fastemit_lambda', 0.0)
        clamp = loss_kwargs.pop('clamp', -1.0)
        loss_func = RNNTLossNumba(blank=blank_idx, reduction='none', fastemit_lambda=fastemit_lambda, clamp=clamp)
        _warn_unused_additional_kwargs(loss_name, loss_kwargs)

    elif loss_name == 'pytorch':
        loss_func = RNNTLossPytorch(blank=blank_idx, reduction='none')
        _warn_unused_additional_kwargs(loss_name, loss_kwargs)

    elif loss_name == 'multiblank_rnnt':
        fastemit_lambda = loss_kwargs.pop('fastemit_lambda', 0.0)
        clamp = loss_kwargs.pop('clamp', -1.0)
        big_blank_durations = loss_kwargs.pop('big_blank_durations', None)
        sigma = loss_kwargs.pop('sigma', 0.0)
        loss_func = MultiblankRNNTLossNumba(
            blank=blank_idx,
            big_blank_durations=big_blank_durations,
            reduction='none',
            fastemit_lambda=fastemit_lambda,
            clamp=clamp,
            sigma=sigma,
        )
        _warn_unused_additional_kwargs(loss_name, loss_kwargs)

    elif loss_name == 'multiblank_rnnt_pytorch':
        big_blank_durations = loss_kwargs.pop('big_blank_durations', None)
        sigma = loss_kwargs.pop('sigma', 0.0)
        loss_func = MultiblankRNNTLossPytorch(
            blank=blank_idx, big_blank_durations=big_blank_durations, reduction='none', sigma=sigma
        )
        _warn_unused_additional_kwargs(loss_name, loss_kwargs)

    elif loss_name == 'tdt':
        fastemit_lambda = loss_kwargs.pop('fastemit_lambda', 0.0)
        clamp = loss_kwargs.pop('clamp', -1.0)
        durations = loss_kwargs.pop('durations', None)
        sigma = loss_kwargs.pop('sigma', 0.0)
        omega = loss_kwargs.pop('omega', 0.0)
        loss_func = TDTLossNumba(
            blank=blank_idx,
            durations=durations,
            reduction='none',
            fastemit_lambda=fastemit_lambda,
            clamp=clamp,
            sigma=sigma,
            omega=omega,
        )
        _warn_unused_additional_kwargs(loss_name, loss_kwargs)

    elif loss_name == 'tdt_pytorch':
        durations = loss_kwargs.pop('durations', None)
        sigma = loss_kwargs.pop('sigma', 0.0)
        loss_func = TDTLossPytorch(blank=blank_idx, durations=durations, reduction='none', sigma=sigma)
        _warn_unused_additional_kwargs(loss_name, loss_kwargs)

    elif loss_name == "graph_rnnt":
        loss_kwargs = _clean_kwargs(loss_name, loss_kwargs, GraphRnntLoss.__init__, ignore_params={"blank"})
        loss_func = GraphRnntLoss(blank=blank_idx, **loss_kwargs)
    elif loss_name == "graph_w_transducer":
        loss_kwargs = _clean_kwargs(loss_name, loss_kwargs, GraphWTransducerLoss.__init__, ignore_params={"blank"})
        loss_func = GraphWTransducerLoss(blank=blank_idx, **loss_kwargs)
    else:
        raise ValueError(
            f"Invalid value of `loss_name`: {loss_name}. Allowed loss names are :" f"{loss_function_names}"
        )

    return loss_func


class RNNTLoss(Loss):
    @property
    def input_types(self):
        """Input types definitions for CTCLoss.
        """
        return {
            "log_probs": NeuralType(('B', 'T', 'T', 'D'), LogprobsType()),
            "targets": NeuralType(('B', 'T'), LabelsType()),
            "input_lengths": NeuralType(tuple('B'), LengthsType()),
            "target_lengths": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        """Output types definitions for CTCLoss.
        loss:
            NeuralType(None)
        """
        return {"loss": NeuralType(elements_type=LossType())}

    def __init__(self, num_classes, reduction: str = 'mean_batch', loss_name: str = "default", loss_kwargs=None):
        """
        RNN-T Loss function based on https://github.com/HawkAaron/warp-transducer.
        Optionally, can utilize a numba implementation of the same loss without having to compile the loss,
        albiet there is a small speed penalty for JIT numba compile.

        Note:
            Requires Numba 0.53.0 or later to be installed to use this loss function.

        Losses can be selected via the config, and optionally be passed keyword arguments as follows.

        Examples:
            .. code-block:: yaml

                model:  # RNNT Model config
                    ...
                    loss:
                        loss_name: "warprnnt_numba"
                        warprnnt_numba_kwargs:
                            fastemit_lambda: 0.0

        Warning:
            In the case that GPU memory is exhausted in order to compute RNNTLoss, it might cause
            a core dump at the cuda level with the following error message.

            ```
                ...
                costs = costs.to(acts.device)
            RuntimeError: CUDA error: an illegal memory access was encountered
            terminate called after throwing an instance of 'c10::Error'
            ```

            Please kill all remaining python processes after this point, and use a smaller batch size
            for train, validation and test sets so that CUDA memory is not exhausted.

        Args:
            num_classes: Number of target classes for the joint network to predict.
                In all cases (conventional RNNT, multi-blank RNNT, and TDT model), this equals the token-id
                for the standard "blank" symbol. In particular, say V is the number of non-blank tokens in
                the vocabulary, then in the case of,
                standard RNNT: num_classes = V
                multiblank RNNT: num_classes = V + number-big-blanks (since we store big-blanks before 
                                 standard blank, and the standard blank is the last symbol in the vocab)
                TDT: num_classes = V. Note, V here does not include any of the "duration outputs".

            reduction: Type of reduction to perform on loss. Possible values are 
                `mean_batch`, 'mean_volume`, `mean`, `sum` or None.
                `None` will return a torch vector comprising the individual loss values of the batch.
                `mean_batch` will average the losses in the batch
                `mean` will divide each loss by the target length and then average
                `mean_volume` will add up all the losses and divide by sum of target lengths

            loss_name: String that is resolved into an RNNT loss function. Available list of losses
                is ininitialized in `RNNT_LOSS_RESOLVER` dictionary.

            loss_kwargs: Optional Dict of (str, value) pairs that are passed to the instantiated loss
                function.
        """
        super(RNNTLoss, self).__init__()

        if reduction not in [None, 'mean', 'sum', 'mean_batch', 'mean_volume']:
            raise ValueError('`reduction` must be one of [mean, sum, mean_batch, mean_volume]')

        self._blank = num_classes
        self.reduction = reduction
        self._loss = resolve_rnnt_loss(loss_name, blank_idx=self._blank, loss_kwargs=loss_kwargs)
        self._force_float32 = RNNT_LOSS_RESOLVER[loss_name].force_float32

    def reduce(self, losses, target_lengths):

        if isinstance(losses, List):
            losses = torch.cat(losses, 0)
            target_lengths = torch.cat(target_lengths, 0)

        if self.reduction == 'mean_batch':
            losses = losses.mean()  # global batch size average
        elif self.reduction == 'mean':
            losses = torch.div(losses, target_lengths).mean()
        elif self.reduction == 'sum':
            losses = losses.sum()
        elif self.reduction == 'mean_volume':
            losses = losses.sum() / target_lengths.sum()  # same as above but longer samples weigh more

        return losses

    @typecheck()
    def forward(self, log_probs, targets, input_lengths, target_lengths):
        # Cast to int 64
        targets = targets.long()
        input_lengths = input_lengths.long()
        target_lengths = target_lengths.long()

        max_logit_len = input_lengths.max()
        max_targets_len = target_lengths.max()

        # Force cast joint to float32
        # TODO: Remove once Numba supports FP16
        if self._force_float32 and log_probs.dtype != torch.float32:
            logits_orig = log_probs
            log_probs = log_probs.float()
            del logits_orig  # save memory *before* computing the loss

        # Ensure that shape mismatch does not occur due to padding
        # Due to padding and subsequent downsampling, it may be possible that
        # max sequence length computed does not match the actual max sequence length
        # of the log_probs tensor, therefore we increment the input_lengths by the difference.
        # This difference is generally small.
        if log_probs.shape[1] != max_logit_len:
            log_probs = log_probs.narrow(dim=1, start=0, length=max_logit_len).contiguous()

        # Reduce transcript length to correct alignment if additional padding was applied.
        # Transcript: [B, L] -> [B, L']; If L' < L
        if not targets.is_contiguous():
            targets = targets.contiguous()

        if targets.shape[1] != max_targets_len:
            targets = targets.narrow(dim=1, start=0, length=max_targets_len).contiguous()

        # Temporarily override loss reduction
        loss_reduction = self._loss.reduction
        self._loss.reduction = None

        # Compute RNNT loss
        loss = self._loss(acts=log_probs, labels=targets, act_lens=input_lengths, label_lens=target_lengths)

        # Loss reduction can be dynamic, so reset it after call
        self._loss.reduction = loss_reduction

        # reduce here using our own reduction function
        if self.reduction is not None:
            loss = self.reduce(loss, target_lengths)

        # del new variables that may have been created
        del (
            log_probs,
            targets,
            input_lengths,
            target_lengths,
        )

        return loss
