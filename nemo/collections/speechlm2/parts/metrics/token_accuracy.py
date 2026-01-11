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
from collections import defaultdict
import torch


def compute_token_accuracy_with_tolerance(target, pred, token, tolerance=1):
    """
    Computes the accuracy of `token` in `target` vs `pred` within a ±`tolerance` window.

    Args:
        target (torch.Tensor): Batch of target sequences (batch_size, seq_len)
        pred (torch.Tensor): Batch of predicted sequences (batch_size, seq_len)
        token (int): The token to compute accuracy for
        tolerance (int): Allowed index difference (window) for correct predictions

    Returns:
        float: Accuracy as correct / total occurrences of `token` in target
    """
    batch_size, seq_len = target.shape

    # Mask of positions where token appears
    target_mask = target == token
    pred_mask = pred == token

    correct = 0
    total = 0

    # For each sequence in the batch
    for b in range(batch_size):
        # Get indices of token in target and pred
        target_indices = target_mask[b].nonzero(as_tuple=True)[0]
        pred_indices = pred_mask[b].nonzero(as_tuple=True)[0]

        total += len(target_indices)
        if len(pred_indices) == 0:
            continue  # No token in pred, none can be correct

        # Compute pairwise distances
        distances = torch.abs(target_indices[:, None] - pred_indices[None, :])  # shape (n_target, n_pred)
        min_distances, _ = torch.min(distances, dim=1)  # min distance for each target occurrence

        # Count how many target tokens have a pred within tolerance
        correct += torch.sum(min_distances <= tolerance).item()

    accuracy = correct / total if total > 0 else 0.0
    return accuracy


class TokenAccuracy:
    """
    Computes Token Accuracy scores.
    """

    def __init__(self, token_name: str, token_id: int, tolerance: int = 1, verbose: bool = True):
        self.token_name = token_name
        self.token_id = token_id
        self.tolerance = tolerance
        self.verbose = verbose
        self.scores = defaultdict(list)

    def reset(self):
        return self

    def update(self, name: str, refs: torch.Tensor, hyps: torch.Tensor) -> None:
        token_acc = compute_token_accuracy_with_tolerance(refs, hyps, token=self.token_id, tolerance=self.tolerance)
        self.scores[name].append(token_acc)

    def compute(self) -> dict[str, torch.Tensor]:
        corpus_metric = {}
        for name in self.scores.keys():
            metric = torch.tensor(self.scores[name]).mean()
            corpus_metric[f"token_acc_{self.token_name}_{name}"] = metric
        self.scores.clear()
        return corpus_metric
