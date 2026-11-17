# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

import torch
import torch.nn as nn

from nemo.collections.asr.parts.turbo_transducer.backend import TurboTransducerBackend
from nemo.core.utils.optional_libs import TRITON_AVAILABLE, triton_required
from nemo.utils import logging

if TRITON_AVAILABLE:
    from nemo.collections.asr.parts.turbo_transducer.rnnt_loss_triton import (
        rnnt_loss_from_logprobs_triton,
        rnnt_loss_triton,
    )


class TurboRnntLoss(nn.Module):
    """
    Memory-efficient RNN-T loss (Turbo-RNNT).

    The Joint output (raw logits) is first reduced to log-probabilities of the target and blank labels
    (`[B, T, U+1]`, no vocabulary dimension) by a fused log-softmax kernel; the loss itself is computed from these
    small tensors. The gradient for the logits is computed in the precision of the logits without
    upcasting, which saves memory for bfloat16/float16 inputs. All computations are performed in float32
    (float64 for float64 inputs).

    The loss uses lengths to skip padding: the Joint output and targets can be padded beyond the maximum lengths.
    Lengths and targets are not validated to avoid host-device synchronization; instead, the per-utterance loss is

    - NaN, if the lengths exceed the size of the Joint output or targets, or a target is outside the vocabulary;
    - infinite, if there is no alignment with non-zero probability (e.g., empty source or `-inf` logits);
      the gradient is zero for such utterance.

    Use `mask_non_finite` to exclude such utterances from training, `warn_non_finite` to report them.

    Gradient clamping (`clamp`) is applied to the gradient of each per-utterance loss for the logits before it is
    multiplied by the upstream gradient (as in `warprnnt_numba`), so it does not depend on the reduction of the loss
    and the batch size. Clamping requires the logits and is not available in `forward_from_logprobs`.
    """

    def __init__(
        self,
        blank: int,
        fastemit_lambda: float = 0.0,
        clamp: float = -1.0,
        mask_non_finite: bool = False,
        warn_non_finite: bool = False,
        backend: str | TurboTransducerBackend = TurboTransducerBackend.TRITON,
    ):
        """
        Init method

        Args:
            blank: blank label index
            fastemit_lambda: Float scaling factor for FastEmit regularization. Default 0.0 (disabled).
            clamp: Float value. When set to a positive value, the gradient for the logits (Joint output)
                is clamped to [-clamp, clamp]. Default -1.0 (disabled).
            mask_non_finite: replace non-finite per-sample losses with 0.0 (the gradient for these samples is zero)
            warn_non_finite: log the loss tensor when non-finite values are detected
            backend: implementation backend; only `triton` is currently supported.
        """
        super().__init__()
        self.blank = blank
        self.fastemit_lambda = fastemit_lambda
        self.clamp = float(clamp) if clamp > 0 else 0.0  # 0.0: no clamping
        self.reduction = None  # NB: `RNNTLoss` wrapper reads/overrides this attribute and applies reduction itself
        self.mask_non_finite = mask_non_finite
        self.warn_non_finite = warn_non_finite
        self.backend = TurboTransducerBackend(backend)
        if self.backend is not TurboTransducerBackend.TRITON:
            raise NotImplementedError(f"Backend {self.backend} is not supported, only `triton` backend is available")
        if not TRITON_AVAILABLE:
            # allow instantiation (e.g., to restore a model for inference), but the loss can't be computed
            logging.warning("Triton is not available, Turbo-RNNT loss can't be computed. Install Triton to use it.")

    @triton_required
    def forward(
        self,
        acts: torch.Tensor,
        labels: torch.Tensor,
        act_lens: torch.Tensor,
        label_lens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute forward method for RNN-T.

        Args:
            acts: activations (joint tensor). NB: raw logits, not after log-softmax
            labels: target labels
            act_lens: lengths of activations
            label_lens: length of labels sequences

        Returns:
            batch of RNN-T scores (loss)
        """
        # argument names are consistent with NeMo, see RNNTLoss.forward:
        # self._loss(acts=log_probs, labels=targets, act_lens=input_lengths, label_lens=target_lengths)
        if acts.device.type != "cuda":
            raise NotImplementedError("Turbo-RNNT loss supports only CUDA inputs")
        loss_batch = rnnt_loss_triton(
            blank_id=self.blank,
            logits=acts,
            targets=labels,
            src_lengths=act_lens,
            tgt_lengths=label_lens,
            fastemit_lambda=self.fastemit_lambda,
            clamp=self.clamp,
        )
        return self._postprocess_loss(loss_batch)

    @triton_required
    def forward_from_logprobs(
        self,
        target_logprobs: torch.Tensor,
        blank_logprobs: torch.Tensor,
        act_lens: torch.Tensor,
        label_lens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute RNN-T loss from target and blank log-probabilities.

        Args:
            target_logprobs: log-probabilities for target labels of size [B, T, U+1]
            blank_logprobs: log-probabilities for the blank label of size [B, T, U+1]
            act_lens: lengths of activations (source lengths) of size [B]
            label_lens: lengths of label sequences of size [B]

        Returns:
            batch of RNN-T scores (loss) of size [B]
        """
        if self.clamp > 0.0:
            raise NotImplementedError(
                "Gradient clamping is applied to the gradient for the logits "
                "and is not supported when computing the loss from log-probabilities"
            )
        if target_logprobs.device.type != "cuda" or blank_logprobs.device.type != "cuda":
            raise NotImplementedError("Turbo-RNNT loss supports only CUDA inputs")

        loss_batch = rnnt_loss_from_logprobs_triton(
            target_logprobs=target_logprobs,
            blank_logprobs=blank_logprobs,
            src_lengths=act_lens,
            tgt_lengths=label_lens,
            fastemit_lambda=self.fastemit_lambda,
        )
        return self._postprocess_loss(loss_batch)

    def _postprocess_loss(self, loss_batch: torch.Tensor) -> torch.Tensor:
        if self.warn_non_finite or self.mask_non_finite:
            loss_finite_mask = torch.isfinite(loss_batch)
            if self.warn_non_finite and not loss_finite_mask.all():
                logging.warning(f"Loss contains NaN/inf values: {loss_batch}")
            if self.mask_non_finite:
                loss_batch = torch.where(loss_finite_mask, loss_batch, torch.zeros_like(loss_batch))

        return loss_batch
