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

"""
This script is used to analyze the results of the experiments from a CSV file.

Basic Usage:
    To perform analysis on the adapters experiment results::

        python scoring_and_analysis.py \
            --csv <path to cleaned result csv file> \
            --dataset_type_column <column name in csv with the dataset types>

    To perform analysis on the finetuning experiment results::

        $ python scoring_and_analysis.py \
            --csv <path to csv> \
            --dataset_type_column <column name in csv with the dataset types> \
            -ft

Advanced Usage:
    The script by default shows only the best hyperparameters for each crietria.
    To see a ranking of all the hyperparameters for each criteria in order to visualize
    how the results were selected use the `--show_analysis` flag. Moreover, instead of
    displaying only the best hyperparameters, you can use the `--topk` flag to show the
    top *k* hyperparameters::

        $ python scoring_and_analysis.py \
            --csv <path to csv> \
            --dataset_type_column <dataset_group_column_name> \
            --show_analysis \
            --topk 3

    Instead of doing the analysis over all possible combinations of all the hyperparameters,
    you can restrict the search space only to a subset of experiments. This can be achieved
    by the `-uargs` and the `-cargs` flag for the unconstrained and the constrained
    experiments respectively::

        $ python scoring_and_analysis.py \
            --csv <path to csv> \
            --dataset_type_column <dataset_group_column_name> \
            -cargs 'Adapter Position' encoder \
            -cargs 'Adapter Dropout' 0.5 \
            -uargs 'Train Steps' 5000
"""

import argparse
from typing import Tuple

import numpy as np
import pandas as pd

# CHANGE: Specify the column names and their attributes to consider for the selection
# of the best results
UNCONSTRAINED_EXP_KEY = {'name': 'WER: Test', 'attribute': min}
CONSTRAINED_EXP_KEY = {'name': 'Score', 'attribute': max}

# CHANGE: Hyperparamters of the best run to display in the output
ADAPTER_HYPERPARAMTER_COLUMNS = ['Adapter Dimensions', 'Adapter Dropout', 'Stochastic Depth', 'Train Steps']
FINETUNING_HYPERPARAMETER_COLUMNS = ['Train Steps', 'Learning Rate']

# CHANGE: Column name for the test set WER on the new domain
TEST_WER_COLUMN = 'WER: Test'

# CHANGE: Column name for the test set WER on the original domain
ORIGINAL_TEST_WER_COLUMN = 'WER: Librispeech Test Other'

# CHANGE: Based on the experiment type, get the column name for categorizing the results
EXP_CATEGORY_KEY = {'adapters': 'Adapter Position', 'finetuning': 'Frozen Module'}

# CHANGE: Maximum absolute WER degradation allowed in the original domain
MAX_DEGRADATION_PERCENTAGE = 3

# CHANGE: Baseline WER in the original domain
BASELINE_ORIGINAL_WER = 5.118

# CHANGE: Baseline WER in the domain to be adapted
# The keys of this dictionary should cover all values of the `dataset_type_column`
BASELINE_ADAPTED_WER = {
    'irish_english_male': 20.690,
    'midlands_english_female': 9.612,
    'midlands_english_male': 11.253,
    'northern_english_female': 11.108,
    'northern_english_male': 10.180,
    'scottish_english_female': 12.309,
    'scottish_english_male': 11.942,
    'southern_english_female': 9.701,
    'southern_english_male': 10.215,
    'welsh_english_female': 8.514,
    'welsh_english_male': 11.463,
}


def calculate_original_scale(original_wer):
    wer_do = abs(original_wer - BASELINE_ORIGINAL_WER)
    return (MAX_DEGRADATION_PERCENTAGE - min(MAX_DEGRADATION_PERCENTAGE, wer_do)) / MAX_DEGRADATION_PERCENTAGE


def calculate_adapt_werr(adapted_wer, group):
    return max(BASELINE_ADAPTED_WER[group] - adapted_wer, 0) / BASELINE_ADAPTED_WER[group]


def parse_results(filepath: str, dataset_type_col: str, exp_type: str) -> Tuple[pd.DataFrame]:
    """Calculate the scoring metric for each experiment

    Args:
        filepath: Path to the csv file containing the results
        dataset_type_col: Name of the column containing the dataset types
        exp_type: Type of experiments in the csv file

    Returns:
        Dataframes of all the experiments with scores
    """
    global UNCONSTRAINED_EXP_KEY, TEST_WER_COLUMN

    df = pd.read_csv(filepath)
    df.drop(columns=['Model', 'Model Size'], errors='ignore', inplace=True)  # Drop columns if exists

    if exp_type == 'finetuning':
        df['Frozen Module'] = df['Frozen Module'].replace('-', 'null')

    if 'Score' not in df:
        # Calculate the selection scoring metric
        df['Original Scale'] = df.apply(lambda x: calculate_original_scale(x[ORIGINAL_TEST_WER_COLUMN]), axis=1)
        df['Adapt WERR'] = df.apply(lambda x: calculate_adapt_werr(x[TEST_WER_COLUMN], x[dataset_type_col]), axis=1)
        df['Score'] = df['Original Scale'] * df['Adapt WERR']

        # Round off the values to 4 decimal places
        df = df.round({'Original Scale': 4, 'Adapt WERR': 4, 'Score': 4})

        # Save the updated csv with scores
        df.to_csv(filepath, index=False)

    return df


def display_analysis_table(df_analysis: pd.DataFrame, key_info: dict):
    """Display the analysis table used to select the best hyperparameter configuration

    Args:
        df_analysis: Dataframe of the analysis table
        key_info: Dictionary containing the name of the column and the attribute to use for analysis
    """
    # Calculate each column length for the table
    column_lengths = {x: max(len(x), df_analysis[x].map(str).apply(len).max()) for x in df_analysis.columns}

    print(' | '.join([f'{x:^{column_lengths[x]}}' for x in df_analysis.columns]))
    print('-' * sum([column_lengths[x] + 3 for x in df_analysis.columns]))

    for idx in range(len(df_analysis)):
        row_str = []
        for column in df_analysis.columns:
            row_str.append(f'{df_analysis.iloc[idx][column]:^{column_lengths[column]}}')
        print(' | '.join(row_str))


def display_results(df_all: pd.DataFrame, category: str, best_config: pd.Series, dataset_type_col: str, exp_type: str):
    """Display the Test and the Librispeech Test Other WER for the best configuration.

    Args:
        df_all: Dataframe of all the experiments
        category: Adapter position or frozen module in case of finetuning
        best_config: Best hyperparameter configurations
        dataset_type_col: Name of the column containing the dataset types
        exp_type: Type of experiments in the dataframe
    """
    test_wer_values, ls_test_other_wer_values = [], []

    print(f'{dataset_type_col:^25} | {TEST_WER_COLUMN:<20} | {ORIGINAL_TEST_WER_COLUMN:<20}')
    print('-' * 70)
    for dtype in df_all[dataset_type_col].unique():
        df_filtered = df_all[(df_all[dataset_type_col] == dtype) & (df_all[EXP_CATEGORY_KEY[exp_type]] == category)]
        for col in ADAPTER_HYPERPARAMTER_COLUMNS if exp_type == 'adapters' else FINETUNING_HYPERPARAMETER_COLUMNS:
            df_filtered = df_filtered[df_filtered[col] == best_config[col]]

        if len(df_filtered) == 0:
            continue

        if len(df_filtered) > 1:
            raise ValueError(f'More than one row found for dtype: {dataset_type_col} and category: {category}')

        dtype_data = df_filtered.iloc[0]
        test_wer_values.append(dtype_data[TEST_WER_COLUMN])
        ls_test_other_wer_values.append(dtype_data[ORIGINAL_TEST_WER_COLUMN])
        print(
            f'{dtype_data[dataset_type_col]:^25} | {dtype_data[TEST_WER_COLUMN]:^20} | {dtype_data[ORIGINAL_TEST_WER_COLUMN]:^20}'
        )
    print('-' * 70)
    print(f'{"Average":^25} | {np.mean(test_wer_values):^20} | {np.mean(ls_test_other_wer_values):^20}')
    print('\n')


def get_best_config(
    df_exp: pd.DataFrame,
    dataset_type_col: str,
    key_info: dict,
    topk: int,
    show_analysis: bool,
    exp_type: str,
):
    """Get the best hyperparameter configuration for a given subset of experiments.

    Args:
        df_exp: Dataframe of all experiments
        dataset_type_col: Name of the column containing the dataset types
        key_info: Dictionary containing the name of the column and the attribute to use for analysis
        topk: Number of top-k results to display
        show_analysis: Whether to display the analysis table
        exp_type: Type of experiments in the dataframe
    """
    # Columns to consider for hyperparameter combinations
    hyperparamter_cols = ADAPTER_HYPERPARAMTER_COLUMNS if exp_type == 'adapters' else FINETUNING_HYPERPARAMETER_COLUMNS

    # Columns to display in the analysis table
    analysis_columns = list(set([key_info['name'], TEST_WER_COLUMN, ORIGINAL_TEST_WER_COLUMN]))

    df_analyze = df_exp.drop(
        columns=[
            x
            for x in df_exp.columns
            if x not in set(hyperparamter_cols + [EXP_CATEGORY_KEY[exp_type]] + analysis_columns)
        ]
    )

    for category in df_exp[EXP_CATEGORY_KEY[exp_type]].unique():
        # Group all hyperparameter configurations and do mean across all speakers
        df_category_mean = (
            df_analyze[df_analyze[EXP_CATEGORY_KEY[exp_type]] == category]
            .groupby(hyperparamter_cols, as_index=False)[analysis_columns]
            .mean()
        )

        # Sort the values by the key in order to get the top-k results
        df_category_mean.sort_values(
            by=key_info['name'], ascending=True if key_info['attribute'].__qualname__ == 'min' else False, inplace=True
        )

        print('=' * len(category))
        print(category.upper())
        print('=' * len(category) + '\n')

        if show_analysis:
            display_analysis_table(df_category_mean, key_info)
            print('\n')

        for idx in range(min(topk, len(df_category_mean))):
            print('-----')
            print(f'Top-{idx + 1}')
            print('-----')

            df_category_best = df_category_mean.iloc[idx]

            print(f'\nHyperparamters')
            print('---------------\n')
            for hyperparamter in hyperparamter_cols + [key_info['name']]:
                print(f'{hyperparamter:<20}: {df_category_best[hyperparamter]}')
            print()

            print('\nResults')
            print('-------\n')
            display_results(df_exp, category, df_category_best, dataset_type_col, exp_type)


def analyze_results(
    df_exp: pd.DataFrame,
    fixed_hyperparameters: list,
    title: str,
    dataset_type_col: str,
    key_info: dict,
    topk: int,
    show_analysis: bool,
    exp_type: str,
):
    """Perform analysis on a given subset of experiments

    Args:
        df_exp: Dataframe of all experiments
        fixed_hyperparameters: List of pair of hyperparamters and their values to fix in the analysis
        title: Title of the analysis (for logging)
        dataset_type_col: Name of the column containing the dataset types
        key_info: Dictionary containing the name of the column and the attribute to use for analysis
        topk: Number of top-k results to display
        show_analysis: Whether to display the analysis table
        exp_type: Type of experiments in the dataframe
    """
    # Filter experiments based on the fixed hyperparameters
    for hyperparameter_name, hyperparameter_value in fixed_hyperparameters:
        df_exp = df_exp[df_exp[hyperparameter_name] == hyperparameter_value]

    # Perform analysis
    print('+' * len(title))
    print(title)
    print('+' * len(title) + '\n')
    get_best_config(df_exp, dataset_type_col, key_info, topk, show_analysis, exp_type)
    print()


def __validate_arg_type(arg):
    """Validate the type of the command line argument value."""
    dtype = float if '.' in arg else int
    try:
        return dtype(arg)
    except ValueError:
        return arg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--csv', required=True, help='Path to the cleaned results CSV file')
    parser.add_argument(
        '-dtype',
        '--dataset_type_column',
        required=True,
        help='Name of the column containing the dataset type. Example: For SLR83 it is "Group", for GSC it is "Dataset Size"',
    )
    parser.add_argument(
        '-cargs',
        '--constrained_args',
        nargs=2,
        action='append',
        default=[],
        type=__validate_arg_type,
        help='Hyperparameters to fix for the constrained experiments',
    )
    parser.add_argument(
        '-uargs',
        '--unconstrained_args',
        nargs=2,
        action='append',
        default=[],
        type=__validate_arg_type,
        help='Hyperparameters to fix for the unconstrained experiments',
    )
    parser.add_argument('-k', '--topk', type=int, default=1, help='Number of top-k results to display')
    parser.add_argument(
        '-ft', '--finetuning', action='store_true', help='True if the CSV contains Finetuning experiments'
    )
    parser.add_argument(
        '-s', '--show_analysis', action='store_true', help='Show the key values of all the dataset types'
    )
    args = parser.parse_args()

    # Get the experiment type
    exp_type = 'finetuning' if args.finetuning else 'adapters'

    # Parse CSV file
    df = parse_results(args.csv, args.dataset_type_column, exp_type)

    # Perform analysis - Constrained Adaptation
    analyze_results(
        df,
        args.constrained_args,
        'Constrained Experiment Results',
        args.dataset_type_column,
        CONSTRAINED_EXP_KEY,
        args.topk,
        args.show_analysis,
        exp_type,
    )

    # Perform analysis - Unconstrained Adaptation
    analyze_results(
        df,
        args.unconstrained_args,
        'Unconstrained Experiment Results',
        args.dataset_type_column,
        UNCONSTRAINED_EXP_KEY,
        args.topk,
        args.show_analysis,
        exp_type,
    )
