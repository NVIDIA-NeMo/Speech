# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

NeMo LLM Customization service requires data to be in the form of .jsonl file with each line having only two fields (namely prompt and completion).

However, you might not have your data readily in this format (or even filetype).

This script will help you to convert from what you have to what you will need quickly and easily.

You will need your datafile (in the form of a .jsonl, .json, .csv, .tsv or .xlsx). 
Each row should contain one sample. 
Make sure that the directory your file is in is readable and writeable.
Otherwise, please change it using chmod. Don't worry, we will not overwrite your existing file.

With close to a dozen consideration factors that makes training optimal, there might just be something you overlook (we all do!). 
To check if dataset has been prepared correctly

!python customization_dataset_preparation.py --filename <filename> 

To format dataset from an alternative jsonl/json/csv/tsv/xlsx column structure (example here for Question Answering task)

For instances, if you are working on a Question Answering Task, you would typically have the columns `context`, `question` and `answer`

!python customization_dataset_preparation.py --filename <filename> --prompt_template "Context: {context} Question: {question} Answer:" --completion_template "{answer}"

Other flags that can be set

1.   `--drop_duplicates` : Use this flag to drop rows that are exactly the same for both prompt and completion
2.   `--split_train_validation` : Use this flag to split one file into separate train and validation files.
3.   `--val_proportion 0.1`: Use a float (default 0.1) between 0 and 1 to control how much of the dataset to allocate to the validation set and the remaining for the train dataset.
4.   `--short_context_model`: Use this flag to prepare data for use with models that have shorter context length of 2048 tokens (e.g. 5B and 20B models)

What to expect

After running this code, you see a list of suggestions to use under ACTIONABLE MESSAGES as well as some insights into your dataset under INFORMATIONAL MESSAGES.

We suggest you prioritize changes suggested under ACTIONABLE MESSAGES but also have a look at the INFORMATIONAL MESSAGES to ensure that changes are done in an expected manner.

"""
import argparse
import math
import os
import pathlib
from collections import Counter

import numpy as np
import pandas as pd


def load_file_into_df(filename):
    message = None
    if not os.path.isfile(filename):
        raise ValueError(f"File {filename} does not exist")
    if filename.lower().endswith(".jsonl"):
        df = pd.read_json(filename, lines=True, dtype=str).fillna("")
    elif filename.lower().endswith(".json"):
        df = pd.read_json(filename, dtype=str).fillna("")
    elif filename.lower().endswith(".xlsx"):
        df = pd.read_excel(filename, dtype=str).fillna("")
        message = "Note only the first sheet in your Excel file will be read."
    elif filename.lower().endswith(".csv"):
        df = pd.read_csv(filename, sep=",", dtype=str).fillna("")
    elif filename.lower().endswith(".tsv"):
        df = pd.read_csv(filename, sep="\t", dtype=str).fillna("")
    else:
        raise ValueError(
            f"Filename {filename} does not have the acceptable extension of .jsonl, .json, .xlsx, .csv or .tsv"
        )
    return df, message


def recommend_hyperparameters_human_readable(recommended_hyperparameters):
    message = 'TODO: Recommended hyperparameters\n'
    for param, param_value in recommended_hyperparameters.items():
        message += f'{param}: {param_value}\n'
    return message


def recommend_hyperparameters(df, model=None):
    """
    Makes recommendations on the batch_size to use for training, based on the dataset size
    
    """
    potential_batch_sizes = [2, 4, 8, 12, 16, 32, 64, 128]

    max_bs = 128
    if len(df) < 128:
        max_bs = 2
        for potential_bs in potential_batch_sizes:
            if potential_bs < len(df) * 0.9:
                max_bs = potential_bs

    bs = min(max_bs, 32)

    df_char_length = df.apply(lambda x: len(x.prompt) + len(x.completion), axis=1)
    length_by_chars = sorted(list(df_char_length))
    n_samples_under_99p5_limit = math.ceil(len(df_char_length) * 0.995)
    char_length_99p5 = length_by_chars[n_samples_under_99p5_limit - 1]
    mean_char_length = np.mean(length_by_chars)
    std_char_length = np.std(length_by_chars)

    # filter out only outliers that are >2 std above mean
    max_char_length = max(min(mean_char_length + 2 * std_char_length, length_by_chars[-1]), char_length_99p5)

    # every token is around 4 chars + 100 for extra capacity
    max_seq_length = max_char_length // 4 + 100

    if len(df) <= 100:
        encoder_hidden_size = 1024
    elif len(df) <= 1000:
        encoder_hidden_size = 2048
    else:
        encoder_hidden_size = 4096

    if len(df) <= 100:
        lr = 5e-3
    elif len(df) <= 1000:
        lr = 1e-3
    elif len(df) <= 10000:
        lr = 5e-4
    else:
        lr = 1e-4

    return {
        'batch_size': bs,
        'max_batch_size': max_bs,
        'num_virtual_tokens': 10,
        'lr': lr,
        'epochs': 10,
        'max_seq_length': max_seq_length,
        'encoder_hidden_size': encoder_hidden_size,
    }


def estimating_customization_job_time(df, recommended_hyperparameters):
    recommended_batch_size = recommended_hyperparameters['batch_size']

    size = df.memory_usage(index=True, deep=True).sum()
    time_in_seconds_per_epoch = size / recommended_batch_size * 0.0025

    if time_in_seconds_per_epoch < 60:
        time_per_epoch = f"{round(time_in_seconds_per_epoch, 2)} seconds"
    elif time_in_seconds_per_epoch < 3600:
        time_per_epoch = f"{round(time_in_seconds_per_epoch/60, 2)} minutes"
    else:
        time_per_epoch = f"{round(time_in_seconds_per_epoch/3600, 2)} hours"

    message = f"TODO: Training will take around {time_per_epoch} for each epoch for gpt20b model and around half of that for gpt5b. Please set no. of epochs accordingly to ensure that the limit of 8h total is not exceeded."
    return message


def warn_completion_is_not_empty(df):
    message = None
    field = "completion"
    empty_rows = (df[field] == "") | (df[field].isnull())
    empty_indexes = df.reset_index().index[empty_rows].tolist()
    if len(empty_indexes) == len(df):
        message = (
            "TODO: Note all completion fields are empty. This is possibly expected for inference but not for training"
        )
    elif len(empty_indexes) != 0:
        message = f"""TODO: completion contains {len(empty_indexes)} empty values at rows ({empty_indexes})
                Please check the original file that the fields for prompt template are 
                not empty and rerun dataset validation"""
    return message


def warn_imbalanced_completion(df):
    completions = df["completion"].tolist()
    completions_counter = Counter(completions)
    message = None
    # low variety of unique completions relative to completions
    # suggesting it is a classification set up
    if len(completions_counter) < len(completions) / 3:
        message = f"There are {len(completions_counter)} unique completions over {len(completions)} samples.\nThe five most common completions are:"
        for completion, n in completions_counter.most_common(5):
            message += f"\n {n} samples ({round(100*n/len(completions),0)}%) with completion: {completion}"
    return message


def get_common_suffix(series):
    common_suffix = ""
    while True:
        candidate_common_suffixes = series.str[-(len(common_suffix) + 1) :]
        if candidate_common_suffixes.nunique() != 1:
            # candidate_common_suffixes contains more than one value
            # therefore, it is no longer a common suffix
            break
        elif common_suffix == candidate_common_suffixes.values[0]:
            # candidate is the same as previous common_suffix
            # therefore values in series are too short to move back by one char
            break
        else:
            common_suffix = candidate_common_suffixes.values[0]
    return common_suffix


def warn_missing_suffix(df):
    message = ''
    for field in ["prompt", "completion"]:
        if not get_common_suffix(df[field]):
            message += f"TODO: {field} does not have common suffix, please add one (e.g. \\n) at the end of {field}_template\n"
    return message if message else None


def validate_template(template):
    template_with_only_brackets = [i for i in template if i in ["{", "}"]]
    error_msg = (
        "Your template ("
        + template
        + ") is not in the correct format.\
                Template must be in the format contains zero or more fields, \
                each field specified by {field}\
                For instance, it can be 'Context: {context} Question: {question}:"
    )
    if len(template_with_only_brackets) % 2 != 0:
        raise ValueError(error_msg)
    for i in range(0, len(template_with_only_brackets), 2):
        if not (template_with_only_brackets[i] == "{" and template_with_only_brackets[i + 1] == "}"):
            raise ValueError(error_msg)
    return None


def parse_template(template):
    field_names = []
    i = 0
    in_field = False
    while i < len(template):
        if template[i] == "{":
            field_names.append("")
            in_field = True
        elif template[i] == "}":
            in_field = False
        elif in_field:
            field_names[-1] += template[i]
        else:
            pass
        i += 1
    return field_names


def warn_duplicated_rows(df):
    message = None
    duplicated_rows = df.duplicated()
    duplicated_indices = df.reset_index().index[duplicated_rows].tolist()
    if len(duplicated_indices) > 0:
        message = f"TODO: There are {len(duplicated_indices)} duplicated rows "
        message += f"at rows ({duplicated_indices}) \n"
        message += "Please check the original file to make sure that is expected\n"
        message += "If it is not, please add the argument --drop_duplicate"
    return message


def drop_duplicated_rows(df):
    duplicated_rows = df.duplicated()
    duplicated_indices = df.reset_index().index[duplicated_rows].tolist()
    message = None
    if len(duplicated_indices) > 0:
        df = df.drop_duplicates()
        message = f"There are {len(duplicated_indices)} duplicated rows\n"
        message += f"Removed {len(duplicated_indices)} duplicate rows"
    return df, message


def template_mapper(row, field_names, template):
    for field_name in field_names:
        template = template.replace("{" + field_name + "}", row[field_name])
    return template


def drop_unrequired_fields(df, required_fields=["prompt", "completion"]):
    for column in df.columns:
        if column not in required_fields:
            df = df.drop(column, axis=1)
    return df


def convert_into_template(df, template, prompt_or_completion="prompt"):
    validate_template(template)
    template = template.replace("\\n", "\n")
    field_names = parse_template(template)
    for field_name in field_names:
        if field_name not in df.columns:
            raise ValueError(
                f"Field {field_name} requested in {prompt_or_completion}_template ({template}) but not found in file columns, which contains {list(df.columns)}"
            )
    df[prompt_or_completion] = df.apply(lambda row: template_mapper(row, field_names, template), axis=1)
    return df


def convert_into_prompt_completion_only(df, prompt_template="{prompt}", completion_template="{completion}"):
    df = convert_into_template(df, prompt_template, prompt_or_completion="prompt")
    df = convert_into_template(df, completion_template, prompt_or_completion="completion")
    df = drop_unrequired_fields(df)
    return df


def warn_and_drop_long_samples(df, max_total_char_length):
    long_examples = df.apply(lambda x: len(x.prompt) + len(x.completion) > max_total_char_length, axis=1)
    indices_of_long_examples = df.reset_index().index[long_examples].tolist()
    message = None
    if len(indices_of_long_examples) > 0:
        message = f"""TODO: There are {len(indices_of_long_examples)} / {len(df)} 
        samples that have its prompt and completion too long 
        (over {max_total_char_length} chars), which have been dropped."""
        df = df.drop(indices_of_long_examples).reset_index()
        df = df.drop('index', axis=1)
    return df, message


def warn_low_n_samples(df, min_samples=64):
    if len(df) < min_samples:
        return f"""TODO: We would recommend having more samples (>{min_samples}) if possible but current_file only contains {len(df)} samples. """
    return None


def show_first_example_in_df(df):
    message = ''
    for column in df.columns:
        # prints \n instead of an a newline
        column_value = df[column][0].replace('\n', '\\n')
        message += f"-->Column {column}:\n{column_value}\n"
    return message


def get_prepared_filename(filename, split_train_validation=False):
    message = ""

    file_extension = pathlib.Path(filename).suffix
    if not split_train_validation:
        new_filename = filename.replace(file_extension, "_prepared.jsonl")
        retry = 0
        while os.path.isfile(new_filename):
            message += f"File {new_filename} exists. Trying next available filename increment\n"
            retry += 1
            new_filename = filename.replace(file_extension, f"_prepared{retry}.jsonl")
        return new_filename, message if message else None
    else:
        train_filename = filename.replace(file_extension, "_prepared_train.jsonl")
        val_filename = filename.replace(file_extension, "_prepared_val.jsonl")
        retry = 0
        while os.path.isfile(train_filename) or os.path.isfile(val_filename):
            message += f"File {train_filename} or {val_filename} exists. Trying next available filename increment\n"
            retry += 1
            train_filename = filename.replace(file_extension, f"_prepared_train{retry}.jsonl")
            val_filename = filename.replace(file_extension, f"_prepared_val{retry}.jsonl")
        return [train_filename, val_filename], message if message else None


def split_into_train_validation(df, val_proportion=0.1):
    n_val = int(val_proportion * len(df))
    df_val = df.sample(n=n_val, random_state=42)
    df_train = df.drop(df_val.index)
    return df_train, df_val


def write_df_to_jsonl(df, filename):
    df.to_json(filename, lines=True, orient="records", force_ascii=False)
    return f"File {filename} written"


def print_select_messages(title, select_messages):
    print("*" * 40)
    print(title)
    print("*" * 40)
    for idx, message in enumerate(select_messages):
        print(f"{idx+1}.")
        print(message)


def print_all_messages(messages):
    messages = [message for message in messages if message]
    info_messages = [message for message in messages if not message.startswith("TODO")]
    to_do_messages = [message for message in messages if message.startswith("TODO")]

    print_select_messages("ACTIONABLE MESSAGES", to_do_messages)
    print_select_messages("INFORMATIONAL MESSAGES", info_messages)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepares data for NeMoLLM Customization Service")
    parser.add_argument("--filename", "-f", required=True)
    parser.add_argument("--prompt_template", "-pt", default="{prompt}")
    parser.add_argument("--completion_template", "-ct", default="{completion}")
    parser.add_argument("--drop_duplicates", "-dd", action="store_true")
    parser.add_argument("--split_train_validation", "-stv", action="store_true")
    parser.add_argument(
        "--short_context_model",
        "-scm",
        action="store_true",
        help="Specifies if using models with shorter context length of 2048 tokens e.g. 5B and 20B models",
    )
    parser.add_argument(
        "--val_proportion",
        "-vp",
        default=0.1,
        type=float,
        help="Give a number between 0 to 1, \
                        representing proportion of samples to go into the validation set\
                        only use when --split_train_validation is set",
    )
    args = parser.parse_args()
    messages = []
    messages.append(str(args))

    if args.short_context_model:
        MAX_TOKEN_LENGTH = 2048
    else:
        MAX_TOKEN_LENGTH = 4096

    # every token is around 4 chars
    MAX_TOTAL_CHAR_LENGTH = 4 * MAX_TOKEN_LENGTH

    df, message = load_file_into_df(args.filename)
    messages.append(message)

    messages.append("-------Before converting into prompt and completion template------ \n")
    messages[-1] += show_first_example_in_df(df)

    df = convert_into_prompt_completion_only(
        df, prompt_template=args.prompt_template, completion_template=args.completion_template
    )

    messages.append("-------After converting into prompt and completion template------ \n")
    messages[-1] += show_first_example_in_df(df)

    if args.drop_duplicates:
        df, message = drop_duplicated_rows(df)
        messages.append(message)
    else:
        messages.append(warn_duplicated_rows(df))

    messages.append(warn_missing_suffix(df))

    messages.append(warn_completion_is_not_empty(df))
    messages.append(warn_imbalanced_completion(df))
    messages.append(warn_low_n_samples(df))

    df, message = warn_and_drop_long_samples(df, MAX_TOTAL_CHAR_LENGTH)
    messages.append(message)

    recommended_hyperparameters = recommend_hyperparameters(df)
    recommend_hyperparameters_message = recommend_hyperparameters_human_readable(recommended_hyperparameters)
    messages.append(recommend_hyperparameters_message)

    messages.append(estimating_customization_job_time(df, recommended_hyperparameters))

    prepared_filename, message = get_prepared_filename(
        args.filename, split_train_validation=args.split_train_validation
    )
    messages.append(message)
    if args.split_train_validation:
        df_train, df_val = split_into_train_validation(df, val_proportion=args.val_proportion)
        messages.append(write_df_to_jsonl(df_train, prepared_filename[0]))
        messages.append(write_df_to_jsonl(df_val, prepared_filename[1]))
    else:
        messages.append(write_df_to_jsonl(df, prepared_filename))

    print_all_messages(messages)
