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
import torch.nn.functional as F
from nemo.utils import logging


class Perplexity:
    """
    Computes perplexity for language model evaluation.
    Perplexity is calculated as exp(cross_entropy_loss).
    """

    def __init__(self, ignore_index: int = -100, verbose: bool = True):
        """
        Args:
            ignore_index: Index to ignore when computing loss (e.g., padding token)
            verbose: Whether to log detailed information
        """
        self.ignore_index = ignore_index
        self.verbose = verbose
        self.losses = defaultdict(list)
        self.token_counts = defaultdict(list)

    def reset(self):
        """Reset accumulated losses and token counts."""
        self.losses.clear()
        self.token_counts.clear()
        return self

    def update(self, name: str, logits: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Update perplexity with new batch.

        Args:
            name: Dataset name
            logits: Model logits of shape (batch_size, seq_len, vocab_size)
            targets: Target tokens of shape (batch_size, seq_len)

        Returns:
            float: Perplexity for this batch
        """
        # Reshape for loss calculation
        logits_flat = logits.view(-1, logits.size(-1))  # (batch_size * seq_len, vocab_size)
        targets_flat = targets.view(-1)  # (batch_size * seq_len,)

        # Calculate cross entropy loss
        loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=self.ignore_index, reduction='sum')

        # Count valid tokens (not ignored)
        valid_tokens = (targets_flat != self.ignore_index).sum().item()

        if valid_tokens > 0:
            # Average loss per token
            avg_loss = loss.item() / valid_tokens
            batch_ppl = torch.exp(torch.tensor(avg_loss))

            self.losses[name].append(loss.item())
            self.token_counts[name].append(valid_tokens)

            if self.verbose:
                logging.debug(f"Batch perplexity for {name}: {batch_ppl:.4f}")

            return batch_ppl.item()
        else:
            logging.warning(f"No valid tokens found in batch for {name}")
            return float('inf')

    def compute(self) -> dict[str, torch.Tensor]:
        """
        Compute final perplexity across all batches.

        Returns:
            dict: Dictionary with perplexity scores for each dataset
        """
        corpus_metric = {}

        for name in self.losses.keys():
            total_loss = sum(self.losses[name])
            total_tokens = sum(self.token_counts[name])

            if total_tokens > 0:
                avg_loss = total_loss / total_tokens
                ppl = torch.exp(torch.tensor(avg_loss))
                corpus_metric[f"perplexity_{name}"] = ppl

                if self.verbose:
                    logging.info(
                        f"Final perplexity for {name}: {ppl:.4f} (avg_loss: {avg_loss:.4f}, tokens: {total_tokens})"
                    )
            else:
                logging.warning(f"No tokens accumulated for {name}")
                corpus_metric[f"perplexity_{name}"] = torch.tensor(float('inf'))

        # Clear accumulated data
        self.losses.clear()
        self.token_counts.clear()

        return corpus_metric


class ValidationLoss:
    """
    Computes validation loss for model evaluation.
    """

    def __init__(self, ignore_index: int = -100):
        """
        Args:
            ignore_index: Index to ignore when computing loss (e.g., padding token)
        """
        self.ignore_index = ignore_index
        self.losses = defaultdict(list)
        self.token_counts = defaultdict(list)

    def reset(self):
        """Reset accumulated losses and token counts."""
        self.losses.clear()
        self.token_counts.clear()
        return self

    def update(self, name: str, logits: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Update validation loss with new batch.

        Args:
            name: Dataset name
            logits: Model logits of shape (batch_size, seq_len, vocab_size)
            targets: Target tokens of shape (batch_size, seq_len)

        Returns:
            float: Loss for this batch
        """
        # Reshape for loss calculation
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = targets.view(-1)

        # Calculate cross entropy loss
        loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=self.ignore_index, reduction='sum')

        # Count valid tokens
        valid_tokens = (targets_flat != self.ignore_index).sum().item()

        if valid_tokens > 0:
            self.losses[name].append(loss.item())
            self.token_counts[name].append(valid_tokens)
            return loss.item() / valid_tokens
        else:
            return 0.0

    def compute(self) -> dict[str, torch.Tensor]:
        """
        Compute final validation loss across all batches.

        Returns:
            dict: Dictionary with validation loss for each dataset
        """
        corpus_metric = {}

        for name in self.losses.keys():
            total_loss = sum(self.losses[name])
            total_tokens = sum(self.token_counts[name])

            if total_tokens > 0:
                avg_loss = total_loss / total_tokens
                corpus_metric[f"val_loss_{name}"] = torch.tensor(avg_loss)
            else:
                corpus_metric[f"val_loss_{name}"] = torch.tensor(0.0)

        # Clear accumulated data
        self.losses.clear()
        self.token_counts.clear()

        return corpus_metric
