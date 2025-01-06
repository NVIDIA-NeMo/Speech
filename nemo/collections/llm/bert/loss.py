# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from typing import Dict, List, Literal, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.distributed import all_gather as all_gather_no_backprop
from torch.distributed.nn.functional import all_gather as all_gather_with_backprop

from nemo.collections.nlp.modules.common.megatron.utils import average_losses_across_data_parallel_group
from nemo.lightning.megatron_parallel import MaskedTokenLossReduction, MegatronLossReduction


class BERTLossReduction(MegatronLossReduction):
    """Bert Loss Function.
    when add_sop_loss = False, only calculate Masked token loss.
    """

    def __init__(self, validation_step: bool = False, val_drop_last: bool = True, add_sop_loss: bool = True) -> None:
        super().__init__()
        self.validation_step = validation_step
        self.val_drop_last = val_drop_last
        self.add_sop_loss = add_sop_loss
        if not add_sop_loss:
            # BERTLoss would act like MaskedTokenLossReduction when only use MLM loss
            self.mlm = MaskedTokenLossReduction(validation_step, val_drop_last)

    def forward(
        self, batch: Dict[str, torch.Tensor], forward_out: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Perform Loss calculation on batch.
        Currently, Context parallelism is not supported for SOP loss.
        """

        # Update loss_mask to batch.
        # Model forward did no update to loss_mask, but for unknown reason loss_mask can get lost (to None)
        # in 'batch' during update. We use the original loss_mask in the dataloader as the ground truth.
        batch['loss_mask'] = forward_out['loss_mask']
        if not self.add_sop_loss:
            return self.mlm.forward(batch, forward_out['lm_loss'])

        from megatron.core import parallel_state

        from nemo.collections.nlp.modules.common.megatron.utils import average_losses_across_data_parallel_group

        lm_loss_, sop_logits = forward_out['lm_loss'], forward_out['binary_logits']
        assert sop_logits is not None, (
            'Attempting to calculate Sentence Order Prediction Loss but SOP logits '
            'are not provideds, Please Make sure you have added binary head.'
        )

        cp_size = parallel_state.get_context_parallel_world_size()
        if cp_size == 1:
            sop_loss_for_ub = sentence_order_prediction_loss(sop_logits, batch["is_random"])
            lm_loss_for_ub = masked_token_with_zero(lm_loss_, batch["loss_mask"])
        else:
            raise NotImplementedError('CP is not supported for SOP loss yet')

        loss_for_ub = sop_loss_for_ub + lm_loss_for_ub
        reduced_loss = average_losses_across_data_parallel_group([loss_for_ub])
        return loss_for_ub * cp_size, {"avg": reduced_loss}

    def reduce(self, losses_reduced_per_micro_batch) -> torch.Tensor:
        """Taken from: https://github.com/NVIDIA/NeMo/blob/main
        /nemo/collections/nlp/models/language_modeling/megatron_gpt_model.py#L535-L552 ."""
        if losses_reduced_per_micro_batch:
            if "avg" in losses_reduced_per_micro_batch[0]:
                loss_tensors_list = [loss_reduced["avg"] for loss_reduced in losses_reduced_per_micro_batch]
                loss_tensor = torch.concat(loss_tensors_list)

                return loss_tensor.mean()

            # Get the total loss since micro batches sizes are not uniform
            loss_sum_tensors_list: List[torch.Tensor] = [
                loss_sum["loss_sum_and_ub_size"]
                for loss_sum in losses_reduced_per_micro_batch
                if loss_sum["loss_sum_and_ub_size"][1] > 0
            ]
            loss_sum = (
                torch.vstack(loss_sum_tensors_list).sum(dim=0)
                if len(loss_sum_tensors_list) > 0
                else torch.tensor([0.0, 0.0], device=torch.cuda.current_device())
            )
            return loss_sum

        return torch.tensor(0.0, device=torch.cuda.current_device())


class BERTInBatchExclusiveHardNegativesRankingLoss(MegatronLossReduction):
    """
    This loss uses in-batch negative samples + hard-negative samples.
    The difference of this loss to the default MultipleNegativesRankingLoss
    from Sentence Transformers is that the latter shares the hard negatives
    as negatives for all examples, whereas this loss uses hard negatives
    exclusively for the example they are associated.

    This loss is also capable of using in-batch negatives from all ranks during training.
    """

    def __init__(
        self,
        validation_step: bool = False,
        val_drop_last: bool = True,
        num_hard_negatives: int = 1,
        scale: float = 20,
        label_smoothing: float = 0.0,
        global_in_batch_negatives: bool = False,
        backprop_type: Literal["local", "global"] = 'local',
    ) -> None:
        super().__init__()
        self.validation_step = validation_step
        self.val_drop_last = val_drop_last
        self.num_hard_negatives = num_hard_negatives
        self.scale = scale
        self.cross_entropy_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.global_in_batch_negatives = global_in_batch_negatives
        self.backprop_type = backprop_type

    def _gather_global_in_batch_representations(self, local_tensor):
        from megatron.core import parallel_state

        local_tensor = local_tensor.contiguous()
        if self.backprop_type == 'local':
            global_tensors = [
                torch.zeros_like(local_tensor) for _ in range(parallel_state.get_data_parallel_world_size())
            ]
            all_gather_no_backprop(global_tensors, local_tensor, group=parallel_state.get_data_parallel_group())
            global_tensors[parallel_state.get_data_parallel_rank()] = local_tensor
            global_tensors = torch.cat(global_tensors, dim=0)

        else:
            global_tensors = all_gather_with_backprop(local_tensor)
            global_tensors = torch.cat(global_tensors, dim=0)

        return global_tensors

    def forward(
        self, batch: Dict[str, torch.Tensor], forward_out: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        from megatron.core import parallel_state

        cp_size = parallel_state.get_context_parallel_world_size()
        if cp_size != 1:
            raise NotImplementedError(f'CP is not supported for {self.__class__} yet.')

        if self.global_in_batch_negatives and not self.validation_step:
            forward_out = self._gather_global_in_batch_representations(forward_out)

        num_tensors_per_example = 2 + self.num_hard_negatives
        batch_size = forward_out.shape[0] // num_tensors_per_example
        chunks = forward_out.chunk(batch_size)
        # Get Queries, Positives, Negatives
        queries = torch.stack([item[0] for item in chunks])
        positives = torch.stack([item[1] for item in chunks])
        hard_negs = [
            torch.stack([item[i + 2] for item in chunks]) for i in range(self.num_hard_negatives)
        ]  # List of length "num_negatives", each tensor of shape (bs, embedding_dim)

        # Calculate scores
        pos_in_batch_negs_scores = torch.mm(
            queries, positives.transpose(0, 1)  # shape (bs, bs); each positive is negative for other queries.
        )
        hard_negs_scores = (
            torch.multiply(
                queries.unsqueeze(0).repeat(len(hard_negs), 1, 1),
                torch.stack(hard_negs),
            )
            .sum(axis=-1)
            .T
        )  # shape = (bs, num_negatives); Hard negatives are not shared between queries.
        scores = torch.cat([pos_in_batch_negs_scores, hard_negs_scores], axis=1)

        scores = scores.clamp(-1.0, 1.0)
        scores *= self.scale
        labels = torch.tensor(
            range(len(scores)), dtype=torch.long, device=scores.device
        )  # Indices of the (query, positive) pairs
        ce_loss = self.cross_entropy_loss(scores, labels)
        reduced_loss = average_losses_across_data_parallel_group([ce_loss])
        return ce_loss, {"avg": reduced_loss}

    def reduce(self, losses_reduced_per_micro_batch) -> torch.Tensor:
        """Taken from: https://github.com/NVIDIA/NeMo/blob/main
        /nemo/collections/nlp/models/language_modeling/megatron_gpt_model.py#L535-L552 ."""
        if losses_reduced_per_micro_batch:
            if "avg" in losses_reduced_per_micro_batch[0]:
                loss_tensors_list = [loss_reduced["avg"] for loss_reduced in losses_reduced_per_micro_batch]
                loss_tensor = torch.concat(loss_tensors_list)

                return loss_tensor.mean()

            # Get the total loss since micro batches sizes are not uniform
            loss_sum_tensors_list: List[torch.Tensor] = [
                loss_sum["loss_sum_and_ub_size"]
                for loss_sum in losses_reduced_per_micro_batch
                if loss_sum["loss_sum_and_ub_size"][1] > 0
            ]
            loss_sum = (
                torch.vstack(loss_sum_tensors_list).sum(dim=0)
                if len(loss_sum_tensors_list) > 0
                else torch.tensor([0.0, 0.0], device=torch.cuda.current_device())
            )
            return loss_sum

        return torch.tensor(0.0, device=torch.cuda.current_device())


def masked_token_with_zero(tensor: Tensor, mask: Tensor):
    """Calculate masked token loss with consideration of possible NaN.
    Sometimes when the number of tokens is very small, none of the tokens get masked for prediction.
    In that case loss mask is all zeros i.e Happens when the entire batch is masked out
    (Practically when MBS=1 or 2, and the number of tokens in each batch is < 7 )
    """
    losses = tensor.float()
    loss_mask = mask.float()
    if loss_mask.sum() == 0:
        loss = torch.sum(losses.view(-1)) * 0.0
    else:
        loss = torch.sum(losses.view(-1) * loss_mask.reshape(-1)) / loss_mask.sum()

    return loss


def sentence_order_prediction_loss(tensor: Tensor, sentence_order: Tensor):
    """Calculate sentence order prediction loss."""
    losses = tensor.view(-1, 2).float()
    sentence_order = sentence_order.view(-1)
    loss = F.cross_entropy(losses, sentence_order, ignore_index=-1)

    return loss
