# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
"""
Visualization utilities for MagpieTTS evaluation metrics.

This module provides functions for creating:
- create_violin_plot: Violin plots for per-file metrics
- create_combined_box_plot:Combined box plots comparing multiple datasets
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nemo.utils import logging


def create_violin_plot(
    metrics: List[dict],
    metric_keys: List[str],
    output_path: Union[str, Path],
) -> None:
    """Create violin plots for the specified metrics.

    Generates a side-by-side violin plot for each metric key, showing the
    distribution of values along with mean and 95% confidence interval.

    Args:
        metrics: List of metric dictionaries (one per file/sample).
        metric_keys: List of metric names to plot.
        output_path: Path to save the output PNG file.
    """
    df = pd.DataFrame(metrics)
    num_columns = len(metric_keys)
    width = num_columns * 5
    fig, axs = plt.subplots(1, num_columns, figsize=(width, 4))

    # Handle single metric case (axs won't be an array)
    if num_columns == 1:
        axs = [axs]

    for i, column in enumerate(metric_keys):
        if column not in df.columns:
            logging.warning(f"Metric '{column}' not found in data, skipping.")
            continue

        axs[i].violinplot(df[column], showmedians=True, positions=[i], widths=0.5)
        axs[i].set_title(column)
        axs[i].set_xticks([i])
        axs[i].set_xticklabels([column])
        axs[i].grid(True, linestyle="dotted")

        # Calculate and display mean with 95% CI
        mean = df[column].mean()
        sem = df[column].sem()
        axs[i].plot(i, mean, "o", color="red", markersize=4, label="Mean (95%CI)")

        label_numeric = f"{mean:.2f}±{1.96 * sem:.2f}"
        axs[i].text(i + 0.06, mean, label_numeric, ha="center", va="top")

    # Create a single legend for all subplots
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left")

    plt.tight_layout()
    plt.savefig(output_path, format="png", bbox_inches="tight")
    plt.close()
    logging.info(f"Violin plot saved to: {output_path}")


def create_combined_box_plot(
    dataset_metrics: Dict[str, List[dict]],
    metric_keys: List[str],
    output_path: Union[str, Path],
) -> None:
    """Create box plots comparing multiple datasets for each metric.

    Generates a figure with one subplot per metric, showing box plots for
    each dataset side by side for easy comparison.

    Args:
        dataset_metrics: Dictionary mapping dataset names to lists of metric dicts.
        metric_keys: List of metric names to plot.
        output_path: Path to save the output PNG file.
    """
    datasets = list(dataset_metrics.keys())
    num_datasets = len(datasets)
    num_metrics = len(metric_keys)

    if num_datasets < 2:
        logging.warning("Combined plot requires at least 2 datasets, skipping.")
        return

    fig, axs = plt.subplots(1, num_metrics, figsize=(num_metrics * 6, 6))

    # Handle single metric case
    if num_metrics == 1:
        axs = [axs]

    # Define colors for different datasets
    colors = plt.cm.Set3(np.linspace(0, 1, num_datasets))

    for metric_idx, metric in enumerate(metric_keys):
        ax = axs[metric_idx]

        # Collect data for all datasets for this metric
        all_data = []
        positions = []
        dataset_labels = []

        for dataset_idx, dataset in enumerate(datasets):
            df = pd.DataFrame(dataset_metrics[dataset])
            if metric in df.columns:
                data = df[metric].dropna()
                all_data.append(data)
                positions.append(dataset_idx + 1)
                dataset_labels.append(dataset)

        if not all_data:
            logging.warning(f"No data for metric '{metric}', skipping subplot.")
            continue

        # Create box plots
        bp = ax.boxplot(
            all_data,
            positions=positions,
            widths=0.6,
            patch_artist=True,
            showmeans=True,
            meanline=False,
            meanprops={
                'marker': 'o',
                'markerfacecolor': 'red',
                'markeredgecolor': 'red',
                'markersize': 6,
            },
        )

        # Color the box plots
        for i, patch in enumerate(bp['boxes']):
            patch.set_facecolor(colors[i])
            patch.set_alpha(0.7)

        # Add mean labels
        for i, (data, pos) in enumerate(zip(all_data, positions)):
            mean = data.mean()
            sem = data.sem()
            label_numeric = f"{mean:.3f}±{1.96 * sem:.3f}"
            ax.text(pos + 0.1, mean, label_numeric, ha="left", va="center", fontsize=8)

        # Set labels and title
        ax.set_title(f"{metric.upper()}", fontsize=12, fontweight='bold')
        ax.set_xticks(positions)
        ax.set_xticklabels(dataset_labels, rotation=45, ha='right')
        ax.grid(True, linestyle="dotted", alpha=0.7)
        ax.set_xlabel("Dataset")
        ax.set_ylabel(metric)

        # Set y-axis limit for CER metrics
        if 'cer' in metric.lower():
            ax.set_ylim(0, 0.3)

    fig.suptitle("Performance Comparison Across Datasets", fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, format="png", bbox_inches="tight", dpi=300)
    plt.close()
    logging.info(f"Combined box plot saved to: {output_path}")
