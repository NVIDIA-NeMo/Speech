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

import torch
from torchmetrics import Metric

__all__ = ['SequencePerplexity']


class SequencePerplexity(Metric):
    """
    This class computes mean perplexity across the batches of sequences.

    You have to provide ``log_probs`` (float tensor of shape [batch_size x seq_length x vocab_size]) and
    ``labels`` (int tensor of shape [batch_size x seq_length] with values from the range [0, vocab_size-1])
    to the :meth:`update` method. If some of the sequences are shorter than seq_length, you can also provide
    an optional argument ``mask`` (bool tensor of shape [batch_size x seq_length]) which masks out tokens
    not participating in perplexity computation.

    See :doc:`PyTorch Lightning Metrics<pytorch-lightning:metrics>` for the metric usage instructions.

    Args:
        dist_sync_on_step:
            Synchronize metric state across processes at each ``forward()`` before returning the value at the step.
        process_group:
            Specify the process group on which synchronization is called. default: ``None`` (which selects the entire
                world)
        dist_sync_fn:
            Callback that performs the allgather operation on the metric state. When ``None``, DDP will be used
                to perform the allgather.
    """

    def __init__(self, dist_sync_on_step=False, process_group=None, dist_sync_fn=None):
        super().__init__(
            dist_sync_on_step=dist_sync_on_step, process_group=process_group, dist_sync_fn=dist_sync_fn,
        )

        # Total sum of exponentiated average negative log likelihoods
        self.add_state('perplexities_sum', default=torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx='sum')
        # Total number of sequences in all batches
        self.add_state('num_sequences', default=torch.tensor(0, dtype=torch.int64), dist_reduce_fx='sum')

    def update(self, log_probs: torch.Tensor, labels: torch.Tensor, mask=None):

        if mask is None:
            mask = torch.ones_like(labels)
        if mask.dtype is not log_probs.dtype:
            mask = mask.to(log_probs.dtype)

        target_log_probs = log_probs.gather(2, labels.unsqueeze(2)).squeeze(2)
        avg_neg_ll = -(target_log_probs * mask).sum(dim=-1) / mask.sum(dim=-1)
        ppl = avg_neg_ll.exp()
        self.num_sequences += ppl.numel()
        self.perplexities_sum += ppl.sum()

    def compute(self):
        """
        Returns perplexity across all workers and resets to 0 :attr:`perplexities_sum` and :attr:`num_sequences`.
        """
        if self.num_sequences.eq(0):
            return None
        return self.perplexities_sum / self.num_sequences
