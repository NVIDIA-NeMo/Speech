# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
#
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


import numpy as np
import torch
from torch.autograd import Function, Variable
from torch.nn import Module


def check_type(var, t, name):
    if var.dtype is not t:
        raise TypeError("{} must be {}".format(name, t))


def check_contiguous(var, name):
    if not var.is_contiguous():
        raise ValueError("{} must be contiguous".format(name))


def check_dim(var, dim, name):
    if len(var.shape) != dim:
        raise ValueError("{} must be {}D".format(name, dim))


def certify_inputs(log_probs, labels, lengths, label_lengths):
    # check_type(log_probs, torch.float32, "log_probs")
    check_type(labels, torch.int64, "labels")
    check_type(label_lengths, torch.int64, "label_lengths")
    check_type(lengths, torch.int64, "lengths")
    check_contiguous(log_probs, "log_probs")
    check_contiguous(labels, "labels")
    check_contiguous(label_lengths, "label_lengths")
    check_contiguous(lengths, "lengths")

    if lengths.shape[0] != log_probs.shape[0]:
        raise ValueError(
            f"Must have a length per example. "
            f"Given lengths dim: {lengths.shape[0]}, "
            f"Log probs dim : {log_probs.shape[0]}"
        )
    if label_lengths.shape[0] != log_probs.shape[0]:
        raise ValueError(
            "Must have a label length per example. "
            f"Given label lengths dim : {label_lengths.shape[0]}, "
            f"Log probs dim : {log_probs.shape[0]}"
        )

    check_dim(log_probs, 4, "log_probs")
    check_dim(labels, 2, "labels")
    check_dim(lengths, 1, "lenghts")
    check_dim(label_lengths, 1, "label_lenghts")
    max_T = torch.max(lengths)
    max_U = torch.max(label_lengths)
    T, U = log_probs.shape[1:3]
    if T != max_T:
        raise ValueError(f"Input length mismatch! Given T: {T}, Expected max T from input lengths: {max_T}")
    if U != max_U + 1:
        raise ValueError(f"Output length mismatch! Given U: {U}, Expected max U from target lengths: {max_U} + 1")


def _assert_no_grad(tensor):
    assert not tensor.requires_grad, (
        "gradients only computed for log_probs - please " "mark other tensors as not requiring gradients"
    )


class LogSoftmaxGradModification(Function):
    @staticmethod
    def forward(ctx, acts, clamp):
        if clamp < 0:
            raise ValueError("`clamp` must be 0.0 or positive float.")

        res = acts.new(acts)
        ctx.clamp = clamp
        return res

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = torch.clamp(grad_output, -ctx.clamp, ctx.clamp)
        return (
            grad_output,
            None,
        )


def forward_pass(log_probs, labels, blank):
    """
    Computes probability of the forward variable alpha.

    Args:
        log_probs: Tensor of shape [T, U, V+1]
        labels: Labels of shape [B, U]
        blank: Index of the blank token.

    Returns:
        A tuple of the forward variable probabilities - alpha of shape [T, U]
        and the log likelihood of this forward step.
    """
    T, U, _ = log_probs.shape
    alphas = np.zeros((T, U), dtype='f')

    for t in range(1, T):
        alphas[t, 0] = alphas[t - 1, 0] + log_probs[t - 1, 0, blank]

    for u in range(1, U):
        alphas[0, u] = alphas[0, u - 1] + log_probs[0, u - 1, labels[u - 1]]
    for t in range(1, T):
        for u in range(1, U):
            no_emit = alphas[t - 1, u] + log_probs[t - 1, u, blank]
            emit = alphas[t, u - 1] + log_probs[t, u - 1, labels[u - 1]]
            alphas[t, u] = np.logaddexp(emit, no_emit)

    loglike = alphas[T - 1, U - 1] + log_probs[T - 1, U - 1, blank]
    return alphas, loglike


def backward_pass(log_probs, labels, blank):
    """
    Computes probability of the backward variable beta.

    Args:
        log_probs: Tensor of shape [T, U, V+1]
        labels: Labels of shape [B, U]
        blank: Index of the blank token.

    Returns:
        A tuple of the backward variable probabilities - beta of shape [T, U]
        and the log likelihood of this backward step.
    """
    T, U, _ = log_probs.shape
    betas = np.zeros((T, U), dtype='f')
    betas[T - 1, U - 1] = log_probs[T - 1, U - 1, blank]

    for t in reversed(range(T - 1)):
        betas[t, U - 1] = betas[t + 1, U - 1] + log_probs[t, U - 1, blank]

    for u in reversed(range(U - 1)):
        betas[T - 1, u] = betas[T - 1, u + 1] + log_probs[T - 1, u, labels[u]]

    for t in reversed(range(T - 1)):
        for u in reversed(range(U - 1)):
            no_emit = betas[t + 1, u] + log_probs[t, u, blank]
            emit = betas[t, u + 1] + log_probs[t, u, labels[u]]
            betas[t, u] = np.logaddexp(emit, no_emit)

    return betas, betas[0, 0]


def compute_gradient(log_probs, alphas, betas, labels, blank, fastemit_lambda):
    """
    Computes the gradients of the log_probs with respect to the log probability of this step occuring.

    Args:
    Args:
        log_probs: Tensor of shape [T, U, V+1]
        alphas: Tensor of shape [T, U] which represents the forward variable.
        betas: Tensor of shape [T, U] which represents the backward variable.
        labels: Labels of shape [B, U]
        blank: Index of the blank token.

    Returns:
        Gradients of shape [T, U, V+1] with respect to the forward log probability
    """
    T, U, _ = log_probs.shape
    grads = np.full(log_probs.shape, -float("inf"))
    log_like = betas[0, 0]  # == alphas[T - 1, U - 1] + betas[T - 1, U - 1]

    # // grad to last blank transition
    grads[T - 1, U - 1, blank] = alphas[T - 1, U - 1]
    grads[: T - 1, :, blank] = alphas[: T - 1, :] + betas[1:, :]

    # // grad to label transition
    for u, l in enumerate(labels):
        grads[:, u, l] = alphas[:, u] + betas[:, u + 1]

    grads = -np.exp(grads + log_probs - log_like)

    if fastemit_lambda > 0.0:
        for u, l in enumerate(labels):
            grads[:, u, l] = (1.0 + fastemit_lambda) * grads[:, u, l]

    return grads


def fastemit_regularization(log_probs, labels, alphas, betas, blank, fastemit_lambda):
    """
    Describes the computation of FastEmit regularization from the paper -
    [FastEmit: Low-latency Streaming ASR with Sequence-level Emission Regularization](https://arxiv.org/abs/2010.11148)

    Args:
        log_probs: Tensor of shape [T, U, V+1]
        labels: Unused. Labels of shape [B, U]
        alphas: Tensor of shape [T, U] which represents the forward variable.
        betas: Unused. Tensor of shape [T, U] which represents the backward variable.
        blank: Index of the blank token.
        fastemit_lambda: Float scaling factor for FastEmit regularization.

    Returns:
        The regularized negative log likelihood - lambda * P˜(At, u|x)
    """
    # General calculation of the fastemit regularization alignments
    T, U, _ = log_probs.shape
    # alignment = np.zeros((T, U), dtype='float32')
    #
    # for t in range(0, T):
    #     alignment[t, U - 1] = alphas[t, U - 1] + betas[t, U - 1]
    #
    # for t in range(0, T):
    #     for u in range(0, U - 1):
    #         emit = alphas[t, u] + log_probs[t, u, labels[u]] + betas[t, u + 1]
    #         alignment[t, u] = emit
    # reg = fastemit_lambda * (alignment[T - 1, U - 1])

    # The above is equivalent to below, without need of computing above
    # reg = fastemit_lambda * (alphas[T - 1, U - 1] + betas[T - 1, U - 1])

    # The above is also equivalent to below, without need of computing the betas alignment matrix
    reg = fastemit_lambda * (alphas[T - 1, U - 1] + log_probs[T - 1, U - 1, blank])
    return -reg


def transduce(log_probs, labels, blank=0, fastemit_lambda=0.0):
    """
    Args:
        log_probs: 3D array with shape
              [input len, output len + 1, vocab size]
        labels: 1D array with shape [output time steps]
        blank: Index of the blank token.
        fastemit_lambda: Float scaling factor for FastEmit regularization.

    Returns:
        float: The negative log-likelihood
        3D array: Gradients with respect to the
                    unnormalized input actications
        2d arrays: Alphas matrix (TxU)
        2d array: Betas matrix (TxU)
    """
    alphas, ll_forward = forward_pass(log_probs, labels, blank)
    betas, ll_backward = backward_pass(log_probs, labels, blank)
    grads = compute_gradient(log_probs, alphas, betas, labels, blank, fastemit_lambda)
    return -ll_forward, grads, alphas, betas


def transduce_batch(log_probs, labels, flen, glen, blank=0, fastemit_lambda=0.0):
    """
    Compute the transducer loss of the batch.

    Args:
        log_probs: [B, T, U, V+1]. Activation matrix normalized with log-softmax.
        labels: [B, U+1] - ground truth labels with <SOS> padded as blank token in the beginning.
        flen: Length vector of the acoustic sequence.
        glen: Length vector of the target sequence.
        blank: Id of the blank token.
        fastemit_lambda: Float scaling factor for FastEmit regularization.

    Returns:
        Batch of transducer forward log probabilities (loss) and the gradients of the activation matrix.
    """
    grads = np.zeros_like(log_probs)
    costs = []
    for b in range(log_probs.shape[0]):
        t = int(flen[b])
        u = int(glen[b]) + 1

        ll, g, alphas, betas = transduce(log_probs[b, :t, :u, :], labels[b, : u - 1], blank, fastemit_lambda)
        grads[b, :t, :u, :] = g

        reg = fastemit_regularization(
            log_probs[b, :t, :u, :], labels[b, : u - 1], alphas, betas, blank, fastemit_lambda
        )
        ll += reg
        costs.append(ll)
    return costs, grads


class _RNNT(Function):
    @staticmethod
    def forward(ctx, acts, labels, act_lens, label_lens, blank, fastemit_lambda):
        costs, grads = transduce_batch(
            acts.detach().cpu().numpy(),
            labels.cpu().numpy(),
            act_lens.cpu().numpy(),
            label_lens.cpu().numpy(),
            blank,
            fastemit_lambda,
        )

        costs = torch.FloatTensor([sum(costs)])
        grads = torch.Tensor(grads).to(acts)

        ctx.grads = grads
        return costs

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.view(-1, 1, 1, 1).to(ctx.grads)
        return ctx.grads.mul(grad_output), None, None, None, None, None


class RNNTLoss(Module):
    """
    Parameters:
        `blank_label` (int): default 0 - label index of blank token
        fastemit_lambda: Float scaling factor for FastEmit regularization.
    """

    def __init__(self, blank: int = 0, fastemit_lambda: float = 0.0, clamp: float = -1.0):
        super(RNNTLoss, self).__init__()
        self.blank = blank
        self.fastemit_lambda = fastemit_lambda
        self.clamp = float(clamp) if clamp > 0 else 0.0
        self.rnnt = _RNNT.apply

    def forward(self, acts, labels, act_lens, label_lens):
        assert len(labels.size()) == 2
        _assert_no_grad(labels)
        _assert_no_grad(act_lens)
        _assert_no_grad(label_lens)
        certify_inputs(acts, labels, act_lens, label_lens)

        # CPU Patch for fp16 - force cast to fp32
        if not acts.is_cuda and acts.dtype == torch.float16:
            acts = acts.float()

        if self.clamp > 0.0:
            acts = LogSoftmaxGradModification.apply(acts, self.clamp)

        acts = torch.nn.functional.log_softmax(acts, -1)

        return self.rnnt(acts, labels, act_lens, label_lens, self.blank, self.fastemit_lambda)


if __name__ == '__main__':
    loss = RNNTLoss(fastemit_lambda=0.01)

    torch.manual_seed(0)

    acts = torch.randn(1, 2, 5, 3)
    labels = torch.tensor([[0, 2, 1, 2]], dtype=torch.int64)
    act_lens = torch.tensor([2], dtype=torch.int64)
    label_lens = torch.tensor([len(labels[0])], dtype=torch.int64)

    loss_val = loss(acts, labels, act_lens, label_lens)
