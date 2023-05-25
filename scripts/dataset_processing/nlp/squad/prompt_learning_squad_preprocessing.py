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

import argparse
import json

from tqdm import tqdm


"""
Dataset preprocessing script for the SQuAD dataset: https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json
Converts the dataset into a jsonl format that can be used for p-tuning/prompt tuning in NeMo. 

Inputs:
    data-dir: (str) The directory where the squad dataset was downloaded, files will be saved here
    train-file: (str) Name of train set file, either train-v1.1.json or train-v2.0.json
    dev-file: (str) Name of dev set file, either dev-v1.1.json or dev-v2.0.json
    save-name-base: (str) The base name for each of the train, val, and test files. If save-name-base were 'squad' for
                    example, the files would be saved as squad_train.jsonl, squad_val.jsonl, and squad_test.jsonl
    include-topic-name: Whether to include the topic name for the paragraph in the data json. See the squad explaination
                        below for more context on what is ment by 'topic name'.
    random-seed: (int) Random seed for repeatable shuffling of train/val/test splits. 

Saves train, val, and test files for the SQuAD dataset. The val and test splits are the same data, because the given test
split lacks ground truth answers. 

An example of the processed output written to file:
    
    {
        "taskname": "squad", 
        "context": "Red is the traditional color of warning and danger. In the Middle Ages, a red flag announced that the defenders of a town or castle would fight to defend it, and a red flag hoisted by a warship meant they would show no mercy to their enemy. In Britain, in the early days of motoring, motor cars had to follow a man with a red flag who would warn horse-drawn vehicles, before the Locomotives on Highways Act 1896 abolished this law. In automobile races, the red flag is raised if there is danger to the drivers. In international football, a player who has made a serious violation of the rules is shown a red penalty card and ejected from the game.", 
        "question": "What did a red flag signal in the Middle Ages?", 
        "answer": " defenders of a town or castle would fight to defend it"
    },


"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=".")
    parser.add_argument("--train-file", type=str, default="train-v1.1.json")
    parser.add_argument("--dev-file", type=str, default="dev-v1.1.json")
    parser.add_argument("--save-name-base", type=str, default="squad")
    parser.add_argument("--include-topic-name", action='store_true')
    parser.add_argument("--random-seed", type=int, default=1234)
    parser.add_argument("--sft-format", action='store_true')
    args = parser.parse_args()

    train_data_dict = json.load(open(f"{args.data_dir}/{args.train_file}"))
    dev_data_dict = json.load(open(f"{args.data_dir}/{args.dev_file}"))
    train_data = train_data_dict['data']
    val_data = dev_data_dict['data']

    save_name_base = f"{args.data_dir}/{args.save_name_base}"

    process_data(train_data, val_data, save_name_base, args.include_topic_name, args.sft_format)


def process_data(train_data, val_data, save_name_base, include_topic, sft_format):
    train_set = extract_questions(train_data, include_topic, sft_format, split="train")
    val_set = extract_questions(val_data, include_topic, sft_format, split="val")
    test_set = extract_questions(val_data, include_topic, sft_format, split="test")

    gen_file(train_set, save_name_base, 'train', sft_format)
    gen_file(val_set, save_name_base, 'val', sft_format)
    gen_file(test_set, save_name_base, 'test', sft_format, make_ground_truth=True)
    gen_file(test_set, save_name_base, 'test', sft_format, make_ground_truth=False)


def extract_questions(data, include_topic, sft_format, split):
    processed_data = []

    # Iterate over topics, want to keep them seprate in train/val/test splits
    for question_group in data:
        processed_topic_data = []
        topic = question_group['title']
        questions = question_group['paragraphs']

        # Iterate over paragraphs related to topics
        for qa_group in questions:
            context = qa_group['context']
            qas = qa_group['qas']

            # Iterate over questions about paragraph
            for qa in qas:
                question = qa['question']

                try:
                    # Dev set has multiple right answers. Want all possible answers in test split ground truth
                    if split == "test":
                        answers = [qa['answers'][i]['text'] for i in range(len(qa['answers']))]

                    # Choose one anser from dev set if making validation split, train set only has one answer
                    else:
                        answers = qa['answers'][0]["text"]

                except IndexError:
                    continue

                if sft_format:
                    example_json = {
                        "input": f"User: Context:{context} Question:{question}\n\nAssistant:",
                        "output": answers,
                    }
                else:
                    example_json = {"taskname": "squad", "context": context, "question": question, "answer": answers}

                if include_topic:
                    example_json["topic"] = topic

                processed_topic_data.append(example_json)
        processed_data.extend(processed_topic_data)

    return processed_data


def gen_file(data, save_name_base, split_type, sft_format, make_ground_truth=False):
    save_path = f"{save_name_base}_{split_type}.jsonl"

    if make_ground_truth:
        save_path = f"{save_name_base}_{split_type}_ground_truth.jsonl"

    print(f"Saving {split_type} split to {save_path}")

    with open(save_path, 'w') as save_file:
        for example_json in tqdm(data):

            # Dont want labels in the test set
            if split_type == "test" and not make_ground_truth:
                if sft_format:
                    example_json["output"] = ""
                else:
                    del example_json["answer"]

            save_file.write(json.dumps(example_json) + '\n')


if __name__ == "__main__":
    main()
