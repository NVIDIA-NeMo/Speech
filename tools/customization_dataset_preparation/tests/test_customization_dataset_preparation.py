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

import pandas as pd
import pytest
from ..customization_dataset_preparation import (
    convert_into_prompt_completion_only,
    convert_into_template,
    drop_duplicated_rows,
    drop_unrequired_fields,
    get_common_suffix,
    get_prepared_filename,
    parse_template,
    recommend_hyperparameters,
    show_first_example_in_df,
    split_into_train_validation,
    template_mapper,
    validate_template,
    warn_and_drop_long_samples,
    warn_completion_is_not_empty,
    warn_duplicated_rows,
    warn_imbalanced_completion,
    warn_low_n_samples,
    warn_missing_suffix,
)


def test_recommend_hyperparameters():
    df_100 = pd.DataFrame({'prompt': ['prompt'] * 100, 'completion': ['completion'] * 100})
    assert recommend_hyperparameters(df_100) == {
        'batch_size': 32,
        'max_batch_size': 64,
        'num_virtual_tokens': 10,
        'encoder_hidden_size': 1024,
        'lr': 0.005,
        'epochs': 10,
        'max_seq_length': 104,
    }

    df_1000 = pd.DataFrame({'prompt': ['prompt'] * 1000, 'completion': ['completion'] * 1000})
    assert recommend_hyperparameters(df_1000) == {
        'batch_size': 32,
        'max_batch_size': 128,
        'num_virtual_tokens': 10,
        'encoder_hidden_size': 2048,
        'lr': 0.001,
        'epochs': 10,
        'max_seq_length': 104,
    }
    df_10000 = pd.DataFrame({'prompt': ['prompt'] * 10000, 'completion': ['completion'] * 10000})
    assert recommend_hyperparameters(df_10000) == {
        'batch_size': 32,
        'max_batch_size': 128,
        'num_virtual_tokens': 10,
        'encoder_hidden_size': 4096,
        'lr': 0.0005,
        'epochs': 10,
        'max_seq_length': 104,
    }
    df_100000 = pd.DataFrame({'prompt': ['prompt'] * 100000, 'completion': ['completion'] * 100000})
    assert recommend_hyperparameters(df_100000) == {
        'batch_size': 32,
        'max_batch_size': 128,
        'num_virtual_tokens': 10,
        'encoder_hidden_size': 4096,
        'lr': 0.0001,
        'epochs': 10,
        'max_seq_length': 104,
    }


def test_warn_completion_is_not_empty():

    df_all_empty = pd.DataFrame({'prompt': ['prompt'] * 2, 'completion': [''] * 2})

    msg_all_empty = (
        "TODO: Note all completion fields are empty. This is possibly expected for inference but not for training"
    )

    assert warn_completion_is_not_empty(df_all_empty) == msg_all_empty

    df_some_empty = pd.DataFrame({'prompt': ['prompt'] * 2, 'completion': ['', 'completion']})

    msg_some_empty = f"""TODO: completion contains {1} empty values at rows ({[0]})
                Please check the original file that the fields for prompt template are 
                not empty and rerun dataset validation"""

    assert warn_completion_is_not_empty(df_some_empty) == msg_some_empty

    df_no_empty = pd.DataFrame({'prompt': ['prompt'] * 2, 'completion': ['completion'] * 2})

    assert warn_completion_is_not_empty(df_no_empty) is None


def test_warn_imbalanced_completion():
    df_generation = pd.DataFrame(
        {'prompt': [f'prompt{i}' for i in range(100)], 'completion': [f'completion{i}' for i in range(100)]}
    )
    assert warn_imbalanced_completion(df_generation) is None

    df_classification_balanced = pd.DataFrame(
        {'prompt': [f'prompt{i}' for i in range(100)], 'completion': [f'completion{i}' for i in range(5)] * 20}
    )

    msg_classification_balanced = (
        f"There are {5} unique completions over {100} samples.\nThe five most common completions are:"
    )
    for i in range(5):
        msg_classification_balanced += f"\n {20} samples ({20.0}%) with completion: completion{i}"

    assert warn_imbalanced_completion(df_classification_balanced) == msg_classification_balanced

    df_classification_imbalanced = pd.DataFrame(
        {
            'prompt': [f'prompt{i}' for i in range(100)],
            'completion': ['completion0'] * 95 + [f'completion{i}' for i in range(5)],
        }
    )

    msg_classification_imbalanced = (
        f"There are {5} unique completions over {100} samples.\nThe five most common completions are:"
    )
    msg_classification_imbalanced += f"\n {96} samples ({96.0}%) with completion: completion0"
    for i in range(1, 5):
        msg_classification_imbalanced += f"\n {1} samples ({1.0}%) with completion: completion{i}"

    assert warn_imbalanced_completion(df_classification_imbalanced) == msg_classification_imbalanced


def test_get_common_suffix():
    df = pd.DataFrame(
        {
            'prompt': [f'prompt{i} answer:' for i in range(100)],
            'completion': [f'completion{i}' for i in range(100)],
            'empty_completion': [''] * 100,
            'some_empty_completion': ['', 'completion'] * 50,
        }
    )
    assert get_common_suffix(df.prompt) == " answer:"
    assert get_common_suffix(df.completion) == ""
    assert get_common_suffix(df.empty_completion) == ""
    assert get_common_suffix(df.some_empty_completion) == ""


def test_warn_missing_suffix():
    df_no_common = pd.DataFrame(
        {'prompt': [f'prompt{i}' for i in range(100)], 'completion': [f'completion{i}' for i in range(100)],}
    )
    message = f"TODO: prompt does not have common suffix, please add one (e.g. \\n) at the end of prompt_template\n"
    message += (
        f"TODO: completion does not have common suffix, please add one (e.g. \\n) at the end of completion_template\n"
    )

    assert warn_missing_suffix(df_no_common) == message
    df_common = pd.DataFrame(
        {'prompt': [f'prompt{i} answer:' for i in range(100)], 'completion': [f'completion{i}\n' for i in range(100)],}
    )
    assert warn_missing_suffix(df_common) is None


def test_parse_template():
    template_qa_prompt = "Context: {context}, Question: {question} Answer:"
    template_qa_completion = "{answer}"
    template_prompt = "{prompt}"
    template_completion = "{completion}"
    assert parse_template(template_qa_prompt) == ['context', 'question']
    assert parse_template(template_qa_completion) == ['answer']
    assert parse_template(template_prompt) == ['prompt']
    assert parse_template(template_completion) == ['completion']


def test_validate_template():
    template = "{prompt}"
    template_missing_left = "prompt}"
    template_missing_right = "{prompt"
    template_twice = "{{prompt}}"
    template_enclosed = "{prompt{enclosed}}"
    assert validate_template(template) is None
    with pytest.raises(ValueError):
        validate_template(template_missing_left)
    with pytest.raises(ValueError):
        validate_template(template_missing_right)
    with pytest.raises(ValueError):
        validate_template(template_twice)
    with pytest.raises(ValueError):
        validate_template(template_enclosed)


def test_warn_duplicated_rows():
    df_duplicated = pd.DataFrame({'prompt': ['prompt'] * 2, 'completion': ['completion'] * 2})

    message_duplicated = f"TODO: There are {1} duplicated rows "
    message_duplicated += f"at rows ([1]) \n"
    message_duplicated += "Please check the original file to make sure that is expected\n"
    message_duplicated += "If it is not, please add the argument --drop_duplicate"

    assert warn_duplicated_rows(df_duplicated) == message_duplicated

    df_unique = pd.DataFrame({'prompt': ['prompt', 'prompt1'], 'completion': ['completion', 'completion1']})
    assert warn_duplicated_rows(df_unique) is None

    df_only_prompt_duplicated = pd.DataFrame({'prompt': ['prompt'] * 2, 'completion': ['completion', 'completion1']})
    assert warn_duplicated_rows(df_only_prompt_duplicated) is None


def test_drop_duplicated_rows():
    df_deduplicated = pd.DataFrame({'prompt': ['prompt'], 'completion': ['completion']})

    df_duplicated = pd.DataFrame({'prompt': ['prompt'] * 2, 'completion': ['completion'] * 2})
    message_duplicated = "There are 1 duplicated rows\n"
    message_duplicated += "Removed 1 duplicate rows"

    assert drop_duplicated_rows(df_duplicated)[0].equals(df_deduplicated)
    assert drop_duplicated_rows(df_duplicated)[1] == message_duplicated

    df_unique = pd.DataFrame({'prompt': ['prompt', 'prompt1'], 'completion': ['completion', 'completion1']})
    assert drop_duplicated_rows(df_unique) == (df_unique, None)

    df_only_prompt_duplicated = pd.DataFrame({'prompt': ['prompt'] * 2, 'completion': ['completion', 'completion1']})
    assert drop_duplicated_rows(df_only_prompt_duplicated) == (df_only_prompt_duplicated, None)


def test_template_mapper():

    df = pd.DataFrame({'prompt': ['prompt sample'],})

    template = "{prompt}"
    field_names = ['prompt']
    assert template_mapper(df.iloc[0], field_names, template) == 'prompt sample'

    df_qa = pd.DataFrame({'question': ['question sample'], 'context': ['context sample']})

    template_qa = "Context: {context} Question: {question} Answer:"
    field_names_qa = ['context', 'question']
    assert (
        template_mapper(df_qa.iloc[0], field_names_qa, template_qa)
        == "Context: context sample Question: question sample Answer:"
    )


def test_drop_unrequired_fields():
    df = pd.DataFrame(
        {'question': ['question'], 'context': ['context'], 'prompt': ['prompt'], 'completion': ['completion']}
    )

    df_dropped_unnecessary_fields = pd.DataFrame({'prompt': ['prompt'], 'completion': ['completion']})
    assert df_dropped_unnecessary_fields.equals(drop_unrequired_fields(df))


def test_convert_into_template():
    df_non_existant_field_name = pd.DataFrame({'question': ['question']})

    template = "Context: {context} Question: {question} Answer:"
    with pytest.raises(ValueError):
        convert_into_template(df_non_existant_field_name, template)

    df = pd.DataFrame({'question': ['question sample'], 'context': ['context sample'],})

    df_prompt = pd.DataFrame(
        {
            'question': ['question sample'],
            'context': ['context sample'],
            'prompt': ["Context: context sample Question: question sample Answer:"],
        }
    )
    assert convert_into_template(df, template).equals(df_prompt)


def test_convert_into_prompt_completion_only():
    df = pd.DataFrame({'question': ['question sample'], 'context': ['context sample'], 'answer': ['answer sample']})

    df_prompt = pd.DataFrame(
        {'prompt': ["Context: context sample Question: question sample Answer:"], 'completion': ["answer sample"]}
    )

    prompt_template = "Context: {context} Question: {question} Answer:"
    completion_template = "{answer}"

    assert df_prompt.equals(
        convert_into_prompt_completion_only(
            df, prompt_template=prompt_template, completion_template=completion_template
        )
    )
    assert df_prompt.equals(convert_into_prompt_completion_only(df_prompt))


def get_indexes_of_long_examples(df, max_total_char_length):
    long_examples = df.apply(lambda x: len(x.prompt) + len(x.completion) > max_total_char_length, axis=1)
    return df.reset_index().index[long_examples].tolist()


def test_warn_and_drop_long_samples():
    df = pd.DataFrame({'prompt': ['a' * 12000, 'a' * 9000, 'a'], 'completion': ['b' * 12000, 'b' * 2000, 'b']})

    expected_df = pd.DataFrame({'prompt': ['a'], 'completion': ['b']})
    message = f"""TODO: There are {2} / {3} 
        samples that have its prompt and completion too long 
        (over {10000} chars), which have been dropped."""

    assert expected_df.equals(warn_and_drop_long_samples(df, 10000)[0])
    assert warn_and_drop_long_samples(df, 10000)[1] == message

    df_short = pd.DataFrame({'prompt': ['a'] * 2, 'completion': ['b'] * 2})

    assert warn_and_drop_long_samples(df_short, 10000) == (df_short, None)


def test_warn_low_n_samples():
    df_low = pd.DataFrame({'prompt': ['a'] * 10, 'completion': ['b'] * 10})

    df_high = pd.DataFrame({'prompt': ['a'] * 100, 'completion': ['b'] * 100})

    message = (
        "TODO: We would recommend having more samples (>64) if possible but current_file only contains 10 samples. "
    )
    assert warn_low_n_samples(df_low) == message
    assert warn_low_n_samples(df_high) is None


def test_show_first_example_in_df():
    df = pd.DataFrame({'question': ['question sample'], 'context': ['context sample'], 'answer': ['answer sample']})

    message = f"-->Column question:\nquestion sample\n"
    message += f"-->Column context:\ncontext sample\n"
    message += f"-->Column answer:\nanswer sample\n"

    assert message == show_first_example_in_df(df)


def test_get_prepared_filename():
    filename = "tmp/sample.jsonl"
    prepared_filename = "tmp/sample_prepared.jsonl"
    prepared_train_filename = "tmp/sample_prepared_train.jsonl"
    prepared_val_filename = "tmp/sample_prepared_val.jsonl"
    assert get_prepared_filename(filename) == (prepared_filename, None)
    assert get_prepared_filename(filename, split_train_validation=True) == (
        [prepared_train_filename, prepared_val_filename,],
        None,
    )
    csv_filename = "tmp/sample.csv"
    prepared_filename = "tmp/sample_prepared.jsonl"
    assert get_prepared_filename(csv_filename) == (prepared_filename, None)


def test_split_into_train_validation():
    df = pd.DataFrame({'prompt': ['a'] * 10, 'completion': ['b'] * 10})
    df_train, df_val = split_into_train_validation(df, val_proportion=0.1)
    assert len(df_train) == 9
    assert len(df_val) == 1

    df_train, df_val = split_into_train_validation(df, val_proportion=0.2)
    assert len(df_train) == 8
    assert len(df_val) == 2
