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
from typing import List

import torch
from nemo.utils import logging


class EmptyTextMetric:
    """
    Computes empty text metrics on text predictions.
    Counts how many hypotheses are empty or contain only whitespace.
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self._hyps = defaultdict(list)
        self._empty_counts = defaultdict(int)
        self._total_counts = defaultdict(int)

    def reset(self):
        self._hyps.clear()
        self._empty_counts.clear()
        self._total_counts.clear()
        return self

    def update(self, name: str, hyps: List[str]) -> None:
        """
        Update the metric with new hypotheses.

        Args:
            name: Name identifier for this set of hypotheses
            hyps: List of hypothesis strings to evaluate
        """
        for hyp in hyps:
            # Check if hypothesis is empty or only whitespace
            is_empty = not hyp.strip()

            self._hyps[name].append(hyp)
            self._total_counts[name] += 1

            if is_empty:
                self._empty_counts[name] += 1

                if self.verbose:
                    logging.info(f"[EMPTY_HYP]\t'{hyp}' (length: {len(hyp)})")

        if self.verbose:
            empty_rate = self._empty_counts[name] / self._total_counts[name] if self._total_counts[name] > 0 else 0.0
            logging.info(
                f"Batch {name}: {self._empty_counts[name]}/{self._total_counts[name]} empty hypotheses ({empty_rate:.2%})"
            )

    def compute(self) -> dict[str, torch.Tensor]:
        """
        Compute the final empty text metrics.

        Returns:
            Dictionary containing empty text metrics for each name and overall average
        """
        corpus_metric = {}

        for name in self._hyps.keys():
            total_count = self._total_counts[name]
            empty_count = self._empty_counts[name]

            # Empty rate as percentage
            empty_rate = empty_count / total_count if total_count > 0 else 0.0

            # Store metrics
            corpus_metric[f"empty_text_rate_{name}"] = torch.tensor(empty_rate)
            corpus_metric[f"empty_text_count_{name}"] = torch.tensor(empty_count)
            corpus_metric[f"total_text_count_{name}"] = torch.tensor(total_count)

        # Overall average empty rate across all names
        if corpus_metric:
            empty_rate_keys = [k for k in corpus_metric.keys() if k.startswith("empty_text_rate_")]
            if empty_rate_keys:
                overall_empty_rate = torch.stack([corpus_metric[k] for k in empty_rate_keys]).mean()
                corpus_metric["empty_text_rate"] = overall_empty_rate

                # Overall counts
                total_empty = sum(corpus_metric[f"empty_text_count_{name}"] for name in self._hyps.keys())
                total_texts = sum(corpus_metric[f"total_text_count_{name}"] for name in self._hyps.keys())
                corpus_metric["empty_text_count"] = torch.tensor(total_empty)
                corpus_metric["total_text_count"] = torch.tensor(total_texts)

        # Clear stored data after computing
        self._hyps.clear()
        self._empty_counts.clear()
        self._total_counts.clear()

        return corpus_metric


def count_empty_texts(hyps: List[str], verbose: bool = True) -> dict:
    """
    Convenience function to quickly count empty texts in a list.

    Args:
        hyps: List of hypothesis strings
        verbose: Whether to print detailed information

    Returns:
        Dictionary with empty text statistics
    """
    metric = EmptyTextMetric(verbose=verbose)
    metric.update("single_batch", hyps)
    return metric.compute()
