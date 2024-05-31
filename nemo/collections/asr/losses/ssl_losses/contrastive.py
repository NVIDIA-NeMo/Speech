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

from math import ceil

import torch
import torch.nn.functional as F
from torch import nn

from nemo.core import Loss, typecheck
from nemo.core.neural_types import AcousticEncodedRepresentation, LengthsType, LossType, NeuralType, SpectrogramType

__all__ = ["ContrastiveLoss"]


class ContrastiveLoss(Loss):
    @property
    def input_types(self):
        """Input types definitions for Contrastive."""
        return {
            "spectrograms": NeuralType(("B", "D", "T"), SpectrogramType()),
            "spec_masks": NeuralType(("B", "D", "T"), SpectrogramType()),
            "decoder_outputs": NeuralType(("B", "T", "D"), AcousticEncodedRepresentation()),
            "decoder_lengths": NeuralType(tuple('B'), LengthsType(), optional=True),
        }

    @property
    def output_types(self):
        """Output types definitions for Contrastive.
        loss:
            NeuralType(None)
        """
        return {"loss": NeuralType(elements_type=LossType())}

    @property
    def needs_labels(self):
        return False

    def __init__(
        self,
        in_dim: int,
        proj_dim: int = 128,
        combine_time_steps: int = 1,
        num_negatives: int = 100,
        quantized_targets: bool = False,
        codebook_size: int = 320,
        prob_ppl_weight: float = 0.1,
        logit_temp: float = 0.1,
        reduce: str = "sum",
        sample_from_same_utterance_only: bool = True,
        sample_from_non_masked: bool = False,
        sample_from_codebook: bool = False,
        group_loss: bool = False,
        num_groups: int = 2,
        quantizer_temp_start: float = 2,
        quantizer_temp_min: float = 0.5,
        quantizer_temp_decay: float = 0.999995,
        mask_threshold: float = 0.8,
        store_ids: bool = True,
        reduce_ids: bool = False,
        multiplier: float = 16.0,
    ):
        """
        Loss function representing the contrastive task of identifying the true latent speech representation of
        the masked spectrogram steps from a set of sampled distractors.

        Args:
            in_dim: Number of spectrogram channels.
            proj_dim: Number of channels in the model outputs.
            combine_time_steps: How many time steps should be combined into a single representation.
            num_negatives: Number of sampled negatives for each target.
            quantized_targets: Bool that determines if the targets should be quantized.
            codebook_size: Number of vectors in the codebook per group.
            prob_ppl_weight: Float multiplier on the perplexity loss for target quantization.
            logit_temp: Float temperature for normalizing logits.
            reduce: String representing the type of reduction used for cross entropy.
            sample_from_same_utterance_only: Bool that determines if negatives should be sampled only from same utterance.
            sample_from_non_masked: Bool that determines if negatives should be sampled from non-masked steps of the spectrogram.
            sample_from_codebook: Bool that determines if negatives should be sampled from entire codebook.
            group_loss: Bool that determines if loss should be computed separately for each group in the quantizer codebook.
            num_groups: Number of groups in the quantizer codebook.
            quantizer_temp_start: Starting temperature in quantizer.
            quantizer_temp_min: Minimum temperature in quantizer.
            quantizer_temp_decay: Decay rate of quantizer temperature per global step.
            mask_threshold: Float threshold for determining if a time step of the spectrogram is masked based on percent of masked channels.
            store_ids: Bool that determines if the quantizer ids will be stored to be potentially used by other losses.
            reduce_ids: Bool that determines if we convert any sequence of consecutive equivalent ids to a single occurence of that id.
            multiplier: Float multipler on final loss
        """

        super().__init__()
        quantizer_temp = (quantizer_temp_start, quantizer_temp_min, quantizer_temp_decay)
        self.quantized_targets = quantized_targets
        self.num_negatives = num_negatives
        self.prob_ppl_weight = prob_ppl_weight
        if self.quantized_targets:
            quantizer_cfg = {
                "_target_": "nemo.collections.asr.parts.submodules.ssl_quantizers.GumbelVectorQuantizer",
                "dim": in_dim * combine_time_steps,
                "vq_dim": proj_dim,
                "num_vars": codebook_size,
                "groups": num_groups,
                "temp": quantizer_temp,
                "combine_groups": True,
                "time_first": True,
            }
            self.quantizer = ContrastiveLoss.from_config_dict(quantizer_cfg)
        self.prob_ppl_weight = prob_ppl_weight
        self.logit_temp = logit_temp
        self.reduce = reduce
        self.combine_time_steps = combine_time_steps
        self.sample_from_same_utterance_only = sample_from_same_utterance_only
        self.sample_from_non_masked = sample_from_non_masked
        self.sample_from_codebook = sample_from_codebook
        self.group_loss = group_loss
        self.mask_threshold = mask_threshold
        self.multiplier = multiplier

        self.store_ids = store_ids
        self.reduce_ids = reduce_ids

        if not self.quantized_targets:
            self.target_proj = nn.Linear(in_dim * combine_time_steps, proj_dim)

    def sample_negatives(self, y, num):
        # y - T'xBxC or T'xC

        high = y.shape[0]
        neg_idxs = torch.multinomial(torch.ones((num, high), device=y.device), self.num_negatives)

        negs = y[neg_idxs.view(-1)]
        negs = negs.view((num, self.num_negatives) + y.shape[1:])
        negs = negs.transpose(0, 1)
        # negs - NxT'xBxC or NxT'xC

        return negs, neg_idxs

    @typecheck()
    def forward(self, spectrograms, spec_masks, decoder_outputs, decoder_lengths=None):
        targets = spectrograms.transpose(-2, -1)
        masks = spec_masks.transpose(-2, -1)
        # BxTxC
        diff = int(ceil(targets.shape[1] / decoder_outputs.shape[1]) * decoder_outputs.shape[1]) - targets.shape[1]

        if diff > 0:
            targets = F.pad(targets, (0, 0, 0, diff))
            masks = F.pad(masks, (0, 0, 0, diff))

        targets = targets.reshape(targets.shape[0], decoder_outputs.shape[1], -1)
        masks = masks.reshape(targets.shape[0], decoder_outputs.shape[1], -1)

        if self.quantized_targets:
            if self.store_ids:
                # store ids for use by other losses
                targets, prob_ppl_loss, cur_codebook_temp, self.target_ids = self.quantizer(targets, return_ids=True)

                if self.reduce_ids:
                    # reduce consecutive equivalent ids to a single occurence
                    _, indices = torch.unique_consecutive(self.target_ids, return_inverse=True)
                    indices -= indices.min(dim=1, keepdims=True)[0]
                    reduced_ids = torch.zeros_like(self.target_ids)
                    reduced_ids = reduced_ids.scatter_(1, indices, self.target_ids)
                    reduced_lens = indices.max(dim=-1)[0] + 1

                    self.target_ids = reduced_ids.narrow(1, 0, reduced_lens.max())
                    self.target_lengths = reduced_lens

                else:
                    self.target_lengths = None

            else:
                targets, prob_ppl_loss, cur_codebook_temp = self.quantizer(targets)
        else:
            targets = self.target_proj(targets)

        if self.sample_from_same_utterance_only:
            bs = decoder_outputs.shape[0]
            masks = masks.mean(-1) > self.mask_threshold
            out_masked_only = decoder_outputs[masks]
            targets_masked_only = targets[masks]
            out_masked_only = out_masked_only.reshape(bs, -1, out_masked_only.shape[-1])
            targets_masked_only = targets_masked_only.reshape(bs, -1, targets_masked_only.shape[-1])

            # BxT'xC
            # number of masked time steps to predict (T')
            # -> T'xBxC

            out_masked_only = out_masked_only.transpose(0, 1)
            targets_masked_only = targets_masked_only.transpose(0, 1)
            # -> T'xBxC

            if self.sample_from_non_masked:
                # sample from all steps in utterance
                negatives, _ = self.sample_negatives(
                    targets.transpose(0, 1),
                    targets_masked_only.size(0),  # TxBxC  # T'
                )
            else:
                # only sample from masked steps in utterance
                negatives, _ = self.sample_negatives(targets_masked_only, targets_masked_only.size(0))  # T'xBxC  # T'
            # NxT'xBxC

            out_masked_only = out_masked_only.reshape(-1, out_masked_only.shape[-1])
            targets_masked_only = targets_masked_only.reshape(-1, targets_masked_only.shape[-1])
            negatives = negatives.reshape(self.num_negatives, -1, negatives.shape[-1])

            # T'BxC and NxT'BxC

        else:
            masks = masks.mean(-1) > self.mask_threshold
            out_masked_only = decoder_outputs[masks]
            targets_masked_only = targets[masks]

            # T'xC
            # number of masked time steps to predict (T')

            if self.group_loss:
                num_groups = self.quantizer.groups
                negatives = self.quantizer.vars.reshape(num_groups, self.quantizer.num_vars, -1)
                # GxNx(C//G)
                negatives = negatives.transpose(0, 1)
                # NxGx(C//G)
                negatives = negatives.unsqueeze(1).expand(-1, out_masked_only.shape[0], -1, -1)
                # NxT'xGx(C//G)
                negatives = negatives.reshape(negatives.shape[0], -1, negatives.shape[-1])
                # NxT'Gx(C//G)

                out_masked_only = out_masked_only.reshape(-1, out_masked_only.shape[-1] // num_groups)
                targets_masked_only = targets_masked_only.reshape(-1, targets_masked_only.shape[-1] // num_groups)
                # T'Gx(C//G)
            elif self.sample_from_codebook:
                # sample from the full codebook
                negatives = self.quantizer.sample_from_codebook(self.num_negatives, targets_masked_only.size(0))
            elif self.sample_from_non_masked:
                # sample from all steps in batch
                negatives, _ = self.sample_negatives(
                    targets.reshape(targets.shape[0] * targets.shape[1], -1),
                    targets_masked_only.size(0),  # BTxC
                )  # T'
            else:
                # only sample from masked steps
                negatives, _ = self.sample_negatives(targets_masked_only, targets_masked_only.size(0))  # T'xC  # T'
                # NxT'xC

        # Calculate similarity between outputs and all targets
        similarity_scores = self._calculate_similarity(out_masked_only, negatives, targets_masked_only)
        # (1+N)xT'
        # cosine similarity of outs with targets + N negatives

        # Create targets of size T
        similarity_targets = decoder_outputs.new_zeros(similarity_scores.size(1), dtype=torch.long)
        # T'
        # targets are 0, since it's the first, followed by N sampled negatives

        # Transpose similarity scores to TxF for loss
        similarity_scores = similarity_scores.transpose(0, 1)
        # T'x(1+N)

        loss = F.cross_entropy(similarity_scores, similarity_targets, reduction=self.reduce)

        sample_size = similarity_targets.numel()

        if self.prob_ppl_weight != 0 and self.quantized_targets:
            prob_ppl_loss = self.prob_ppl_weight * prob_ppl_loss * sample_size
            loss += prob_ppl_loss

        if not isinstance(loss, torch.Tensor):
            loss = torch.Tensor([0]).to(device=decoder_outputs.device)

        batch_size = spectrograms.shape[0]
        loss *= self.multiplier / batch_size

        return loss

    def _calculate_similarity(self, logits, negatives, targets):
        neg_is_pos = (targets == negatives).all(-1)
        # NxT' - true where the negative is actually the positive
        targets = targets.unsqueeze(0)
        # 1xT'xC
        targets = torch.cat([targets, negatives], dim=0)
        # (1+N)xT'XC
        logits = torch.cosine_similarity(
            logits.float().unsqueeze(0).expand(targets.shape[0], -1, -1), targets.float(), dim=-1
        ).type_as(logits)
        # (1+N)xT'
        logits /= self.logit_temp
        if neg_is_pos.any():
            logits[1:][neg_is_pos] = float("-inf")
        return logits

    def set_num_updates(self, num_updates):
        if self.quantized_targets:
            self.quantizer.set_num_updates(num_updates)
