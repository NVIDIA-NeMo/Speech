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


def compute_turn_taking_metrics(
    source_tokens, pred_tokens, eos_token_id, bos_token_id, tolerance=13, latency_multiplier=0.08
):
    """
    Computes turn taking accuracy and latency.

    Args:
        source_tokens (torch.Tensor): Batch of source sequences (batch_size, seq_len) - user speech
        pred_tokens (torch.Tensor): Batch of predicted sequences (batch_size, seq_len) - agent speech
        eos_token_id (int): End of speech token ID for user speech
        bos_token_id (int): Beginning of speech token ID for agent speech
        tolerance (int): Allowed index difference for successful turn taking
        latency_multiplier (float): Multiplier to convert index difference to latency (default: 0.08)

    Returns:
        tuple: (accuracy, average_latency) - both as Python floats
    """
    # Convert to CPU and Python lists to avoid any symbolic tensor issues
    source_tokens = source_tokens.cpu().numpy()
    pred_tokens = pred_tokens.cpu().numpy()

    batch_size = source_tokens.shape[0]

    successful_turns = 0
    total_turns = 0
    successful_latencies = []

    for b in range(batch_size):
        # Find first EOS in source tokens (user speech end) using numpy operations
        eos_positions = (source_tokens[b] == eos_token_id).nonzero()[0]
        if len(eos_positions) == 0:
            continue  # No user speech end found, skip

        # Find first BOS in predicted tokens (agent speech start) using numpy operations
        bos_positions = (pred_tokens[b] == bos_token_id).nonzero()[0]
        if len(bos_positions) == 0:
            total_turns += 1
            continue  # No agent speech start found, failed turn taking

        # Calculate difference between user speech end and agent speech start
        user_eos_pos = int(eos_positions[0])
        agent_bos_pos = int(bos_positions[0])
        diff = agent_bos_pos - user_eos_pos

        total_turns += 1

        # Check if within tolerance
        if abs(diff) <= tolerance:
            successful_turns += 1
            latency = diff * latency_multiplier
            successful_latencies.append(latency)

    # Calculate metrics as Python floats
    accuracy = successful_turns / total_turns if total_turns > 0 else 0.0
    avg_latency = sum(successful_latencies) / len(successful_latencies) if successful_latencies else 0.0

    return float(accuracy), float(avg_latency)


class TurnTakingMetrics:
    """
    Computes turn taking accuracy and latency metrics.
    Following the same pattern as BLEU metrics to ensure multi-node compatibility.
    """

    def __init__(self, eos_token_id: int, bos_token_id: int, tolerance: int = 13, latency_multiplier: float = 0.08):
        self.eos_token_id = eos_token_id
        self.bos_token_id = bos_token_id
        self.tolerance = tolerance
        self.latency_multiplier = latency_multiplier
        # Store Python lists like BLEU does, not tensor values
        self.accuracies = defaultdict(list)
        self.latencies = defaultdict(list)

    def reset(self):
        self.accuracies.clear()
        self.latencies.clear()
        return self

    def update(self, name: str, source_tokens: torch.Tensor, pred_tokens: torch.Tensor) -> None:
        """
        Update metrics with a batch of samples.

        Args:
            name (str): Dataset name
            source_tokens (torch.Tensor): User speech tokens (batch_size, seq_len)
            pred_tokens (torch.Tensor): Agent speech tokens (batch_size, seq_len)
        """
        accuracy, avg_latency = compute_turn_taking_metrics(
            source_tokens, pred_tokens, self.eos_token_id, self.bos_token_id, self.tolerance, self.latency_multiplier
        )

        # Store Python floats, just like BLEU stores Python strings
        self.accuracies[name].append(accuracy)
        if avg_latency > 0:  # Only add latency if there were successful cases
            self.latencies[name].append(avg_latency)

    def compute(self) -> dict[str, torch.Tensor]:
        """
        Compute final metrics across all updates.
        Following the same pattern as BLEU.compute()

        Returns:
            dict: Dictionary with turn_taking_acc and turn_taking_latency metrics
        """
        corpus_metrics = {}

        # Get all dataset names to ensure consistent metric structure across all ranks
        all_names = set(self.accuracies.keys()) | set(self.latencies.keys())

        # Compute accuracy metrics - same pattern as BLEU
        for name in all_names:
            if self.accuracies[name]:
                # Calculate mean accuracy and create tensor from Python float (like BLEU does)
                acc_mean = sum(self.accuracies[name]) / len(self.accuracies[name])
                corpus_metrics[f"turn_taking_acc_{name}"] = torch.tensor(acc_mean)
            else:
                # Ensure consistent metric structure even if no data
                corpus_metrics[f"turn_taking_acc_{name}"] = torch.tensor(0.0)

        # Compute latency metrics - CRITICAL: always include latency metrics for all datasets
        # to ensure consistent metric structure across all ranks in distributed training
        for name in all_names:
            if self.latencies[name]:
                # Calculate mean latency and create tensor from Python float (like BLEU does)
                latency_mean = sum(self.latencies[name]) / len(self.latencies[name])
                corpus_metrics[f"turn_taking_latency_{name}"] = torch.tensor(latency_mean)
            else:
                # CRITICAL: Even if no successful latencies, return 0.0 to maintain structure consistency
                corpus_metrics[f"turn_taking_latency_{name}"] = torch.tensor(0.0)

        # Clear stored values
        self.accuracies.clear()
        self.latencies.clear()

        return corpus_metrics
