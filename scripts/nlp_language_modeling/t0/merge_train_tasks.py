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
This script will merge prompt-specific train files into a single file per task.
"""
import json
import os
from argparse import ArgumentParser

tasks = [
    'adversarial_qa',
    'ag_news',
    'ai2_arc_ARC_Challenge',
    'ai2_arc_ARC_Easy',
    'amazon_polarity',
    'anli',
    'app_reviews',
    'cnn_dailymail_3.0.0',
    'common_gen',
    'cos_e_v1.11',
    'cosmos_qa',
    'dbpedia_14',
    'dream',
    'duorc_ParaphraseRC',
    'duorc_SelfRC',
    'gigaword',
    'glue_mrpc',
    'glue_qqp',
    'hellaswag',
    'imdb',
    'kilt_tasks_hotpotqa',
    'multi_news',
    'openbookqa_main',
    'paws_labeled_final',
    'piqa',
    'qasc',
    'quail',
    'quarel',
    'quartz',
    'quoref',
    'race_high',
    'race_middle',
    'ropes',
    'rotten_tomatoes',
    'samsum',
    'sciq',
    'social_i_qa',
    'squad_v2',
    'super_glue_boolq',
    'super_glue_cb',
    'super_glue_copa',
    'super_glue_multirc',
    'super_glue_record',
    'super_glue_rte',
    'super_glue_wic',
    'super_glue_wsc',
    'trec',
    'trivia_qa',
    'web_questions',
    'wiki_bio',
    'wiki_hop',
    'wiki_qa',
    'winogrande_winogrande',
    'wiqa',
    'xsum',
    'yelp_review_full',
]


def merge_train_folder(train_data_folder, merged_train_data_folder):
    if not os.path.exists(merged_train_data_folder):
        os.makedirs(merged_train_data_folder)
    task_counter = {task: 0 for task in tasks}
    fptrs = {task: open(os.path.join(merged_train_data_folder, task + '.jsonl'), 'w') for task in tasks}
    for idx, fname in enumerate(os.listdir(train_data_folder)):
        if idx % 10 == 0:
            print(f'Processed {idx + 1}/{len(os.listdir(train_data_folder))} files ...')
        if fname.endswith('.jsonl') and '_score_eval' not in fname:
            found = False
            for task in tasks:
                if fname.startswith(task):
                    task_counter[task] += 1
                    found = True
                    with open(os.path.join(train_data_folder, fname), 'r') as f:
                        for line in f:
                            line = json.loads(line)
                            line['task_name_with_prompt'] = fname
                            if line['input'].strip() == '':
                                print(f'WARNING: Empty input for {fname}')
                                continue
                            if line['output'].strip() == '':
                                print(f'WARNING: Empty output for {fname}')
                                continue
                            fptrs[task].write(json.dumps(line) + '\n')
            if not found:
                print(f'WARNING: Could not find task for {fname}')

    for _, v in fptrs.items():
        v.close()
        if task_counter[task] == 0:
            print('WARNING: No files found for task: ', task)

    for k, v in task_counter.items():
        print(f'Task {k} had {v} prompt templates.')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        "--p3_processed_train_dataset_path",
        type=str,
        required=True,
        help="Path to the processed P3 train dataset. This is the output of the t0_dataset_preproc.py script.",
    )
    parser.add_argument(
        "--p3_processed_merged_train_dataset_path",
        type=str,
        required=True,
        help="Path to output folder where merged JSONL files will be written.",
    )
    args = parser.parse_args()
    merge_train_folder(args.p3_processed_train_dataset_path, args.p3_processed_merged_train_dataset_path)
