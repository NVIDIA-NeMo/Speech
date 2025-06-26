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

import argparse
import csv
import io
import json
import os
import random
import tarfile
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Tuple

from datasets import load_dataset

random.seed(42)

subcategories = {
    "abstract_algebra": ["math"],
    "anatomy": ["health"],
    "astronomy": ["physics"],
    "business_ethics": ["business"],
    "clinical_knowledge": ["health"],
    "college_biology": ["biology"],
    "college_chemistry": ["chemistry"],
    "college_computer_science": ["computer science"],
    "college_mathematics": ["math"],
    "college_medicine": ["health"],
    "college_physics": ["physics"],
    "computer_security": ["computer science"],
    "conceptual_physics": ["physics"],
    "econometrics": ["economics"],
    "electrical_engineering": ["engineering"],
    "elementary_mathematics": ["math"],
    "formal_logic": ["philosophy"],
    "global_facts": ["other"],
    "high_school_biology": ["biology"],
    "high_school_chemistry": ["chemistry"],
    "high_school_computer_science": ["computer science"],
    "high_school_european_history": ["history"],
    "high_school_geography": ["geography"],
    "high_school_government_and_politics": ["politics"],
    "high_school_macroeconomics": ["economics"],
    "high_school_mathematics": ["math"],
    "high_school_microeconomics": ["economics"],
    "high_school_physics": ["physics"],
    "high_school_psychology": ["psychology"],
    "high_school_statistics": ["math"],
    "high_school_us_history": ["history"],
    "high_school_world_history": ["history"],
    "human_aging": ["health"],
    "human_sexuality": ["culture"],
    "international_law": ["law"],
    "jurisprudence": ["law"],
    "logical_fallacies": ["philosophy"],
    "machine_learning": ["computer science"],
    "management": ["business"],
    "marketing": ["business"],
    "medical_genetics": ["health"],
    "miscellaneous": ["other"],
    "moral_disputes": ["philosophy"],
    "moral_scenarios": ["philosophy"],
    "nutrition": ["health"],
    "philosophy": ["philosophy"],
    "prehistory": ["history"],
    "professional_accounting": ["other"],
    "professional_law": ["law"],
    "professional_medicine": ["health"],
    "professional_psychology": ["psychology"],
    "public_relations": ["politics"],
    "security_studies": ["politics"],
    "sociology": ["culture"],
    "us_foreign_policy": ["politics"],
    "virology": ["health"],
    "world_religions": ["philosophy"],
}


def process_mmlu_dataset(split) -> List[Dict[str, Any]]:
    """Process the MMLU dataset and return a list of formatted entries."""
    data_url = "https://people.eecs.berkeley.edu/~hendrycks/data.tar"
    data_dir = Path(__file__).absolute().parent
    data_file = str(data_dir / f"data.tar")
    data_dir.mkdir(exist_ok=True)
    output_file = str(data_dir / f"{split}.jsonl")

    urllib.request.urlretrieve(data_url, data_file)
    result = {}

    column_names = ["question", "A", "B", "C", "D", "expected_answer"]

    with tarfile.open(data_file, 'r') as tar:
        # List all members of the tar file
        members = tar.getmembers()

        # Filter for CSV files in the 'data/test' directory
        csv_files = [
            member for member in members if member.name.startswith(f'data/{split}/') and member.name.endswith('.csv')
        ]

        for csv_file in csv_files:
            # Extract the file name without the path
            file_name = os.path.basename(csv_file.name)

            # Read the CSV file content
            file_content = tar.extractfile(csv_file)
            if file_content is not None:
                # Decode bytes to string
                content_str = io.TextIOWrapper(file_content, encoding='utf-8')

                # Use csv to read the CSV content without a header
                csv_reader = csv.reader(content_str)

                # Convert CSV data to list of dictionaries with specified column names
                csv_data = []
                for row in csv_reader:
                    if len(row) == len(column_names):
                        csv_data.append(dict(zip(column_names, row)))
                    else:
                        print(f"Warning: Skipping row in {file_name} due to incorrect number of columns")

                # Add to result dictionary
                result[file_name.rsplit('_', 1)[0]] = csv_data

    # Define the allowed supercategories
    chosen_subcategories = {"math", "health", "physics", "biology", "chemistry", "computer science", "engineering"}

    # Filter the result dictionary
    filtered_result = {
        k: v
        for k, v in result.items()
        if any(supercat in chosen_subcategories for supercat in subcategories.get(k, []))
    }

    output_data = []
    skipped = 0

    for category, question_list in filtered_result.items():
        for item in question_list:
            question = item['question']
            choice_1 = item['A']
            choice_2 = item['B']
            choice_3 = item['C']
            choice_4 = item['D']
            answer = item['expected_answer']

            if not question or not answer or not choice_1 or not choice_2 or not choice_3 or not choice_4:
                skipped += 1
                continue

            entry = {
                'Question': question,
                'Choice 1': choice_1,
                'Choice 2': choice_2,
                'Choice 3': choice_3,
                'Choice 4': choice_4,
                'Answer': answer,
                'Subject': category,
            }

            output_data.append(entry)

    return output_data, skipped


def process_gpqa_dataset() -> List[Dict[str, Any]]:
    """Process the GPQA dataset and return a list of formatted entries."""
    dataset = load_dataset("Idavidrein/gpqa", "gpqa_main")
    output_data = []
    skipped = 0

    for split in dataset.keys():
        for item in dataset[split]:
            question = item['Question']
            correct_answer = item['Correct Answer']
            incorrect_1 = item['Incorrect Answer 1']
            incorrect_2 = item['Incorrect Answer 2']
            incorrect_3 = item['Incorrect Answer 3']

            if not question or not correct_answer or not incorrect_1 or not incorrect_2 or not incorrect_3:
                skipped += 1
                continue

            answers = [(correct_answer, True), (incorrect_1, False), (incorrect_2, False), (incorrect_3, False)]

            random.shuffle(answers)

            choices = {}
            correct_letter = None

            for i, (answer_text, is_correct) in enumerate(answers):
                letter = chr(65 + i)
                choices[f'Choice {i+1}'] = answer_text
                if is_correct:
                    correct_letter = letter

            entry = {
                'Question': question,
                'Choice 1': choices['Choice 1'],
                'Choice 2': choices['Choice 2'],
                'Choice 3': choices['Choice 3'],
                'Choice 4': choices['Choice 4'],
                'Answer': correct_letter,
            }

            output_data.append(entry)

    return output_data, skipped


def process_gpqa_diamond_dataset() -> List[Dict[str, Any]]:
    """Process the GPQA Diamond dataset and return a list of formatted entries."""
    dataset = load_dataset("Idavidrein/gpqa", "gpqa_diamond")
    output_data = []
    skipped = 0

    for split in dataset.keys():
        for item in dataset[split]:
            question = item['Question']
            correct_answer = item['Correct Answer']
            incorrect_1 = item['Incorrect Answer 1']
            incorrect_2 = item['Incorrect Answer 2']
            incorrect_3 = item['Incorrect Answer 3']

            if not question or not correct_answer or not incorrect_1 or not incorrect_2 or not incorrect_3:
                skipped += 1
                continue

            answers = [(correct_answer, True), (incorrect_1, False), (incorrect_2, False), (incorrect_3, False)]

            random.shuffle(answers)

            choices = {}
            correct_letter = None

            for i, (answer_text, is_correct) in enumerate(answers):
                letter = chr(65 + i)
                choices[f'Choice {i+1}'] = answer_text
                if is_correct:
                    correct_letter = letter

            entry = {
                'Question': question,
                'Choice 1': choices['Choice 1'],
                'Choice 2': choices['Choice 2'],
                'Choice 3': choices['Choice 3'],
                'Choice 4': choices['Choice 4'],
                'Answer': correct_letter,
            }

            output_data.append(entry)

    return output_data, skipped


def write_to_jsonl(data: List[Dict[str, Any]], filename: str) -> None:
    """Write the processed data to a JSONL file."""
    with open(filename, 'w') as f:
        for entry in data:
            f.write(json.dumps(entry) + '\n')


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Convert datasets to JSONL format')
    parser.add_argument(
        '--datasets',
        nargs='+',
        choices=['mmlu', 'gpqa', 'gpqa_diamond', 'all'],
        default=['all'],
        help='Datasets to process (default: all)',
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # If 'all' is selected, process all datasets
    if 'all' in args.datasets:
        args.datasets = ['mmlu', 'gpqa', 'gpqa_diamond']

    # Process selected datasets
    if 'mmlu' in args.datasets:
        for split in ["test", "val"]:
            mmlu_data, mmlu_skipped = process_mmlu_dataset(split)
            write_to_jsonl(mmlu_data, f'mmlu_dataset_{split}.jsonl')
            print(f"Converted {len(mmlu_data)} {split} questions to mmlu_dataset_{split}.jsonl")
            if mmlu_skipped > 0:
                print(f"Skipped {mmlu_skipped} MMLU entries due to missing data")
            print("\nMMLU Sample entry:")
            print(json.dumps(mmlu_data[0], indent=2))

    if 'gpqa' in args.datasets:
        gpqa_data, gpqa_skipped = process_gpqa_dataset()
        write_to_jsonl(gpqa_data, 'gpqa_dataset.jsonl')
        print(f"\nConverted {len(gpqa_data)} questions to gpqa_dataset.jsonl")
        if gpqa_skipped > 0:
            print(f"Skipped {gpqa_skipped} GPQA entries due to missing data")
        print("\nGPQA Sample entry:")
        print(json.dumps(gpqa_data[0], indent=2))

    if 'gpqa_diamond' in args.datasets:
        gpqa_diamond_data, gpqa_diamond_skipped = process_gpqa_diamond_dataset()
        write_to_jsonl(gpqa_diamond_data, 'gpqa_diamond_dataset.jsonl')
        print(f"\nConverted {len(gpqa_diamond_data)} questions to gpqa_diamond_dataset.jsonl")
        if gpqa_diamond_skipped > 0:
            print(f"Skipped {gpqa_diamond_skipped} GPQA Diamond entries due to missing data")
        print("\nGPQA Diamond Sample entry:")
        print(json.dumps(gpqa_diamond_data[0], indent=2))


if __name__ == "__main__":
    main()
