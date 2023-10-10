#!/usr/bin/env python3
# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

"""Processing data for megatron pretraining.

Example to create dataset used for training attribute prediction model:
    python preprocessing.py --input_file dataset/2023-04-12_oasst_all.trees.jsonl output_file_prefix=oasst_output mask_role=User type=TEXT_TO_VALUE split_ratio=0.95, seed=10

Example to create dataset used for attribute conditioned SFT model:
    python preprocessing.py --input_file dataset/2023-04-12_oasst_all.trees.jsonl output_file_prefix=oasst_output mask_role=User type=VALUE_TO_TEXT split_ratio=0.95, seed=10

"""

import json
import random

import fire

# All the keys ['spam', 'lang_mismatch', 'pii', 'not_appropriate', 'hate_speech', 'sexual_content', 'quality', 'toxicity', 'humor', 'creativity', 'violence', 'fails_task', 'helpfulness', 'political_content', 'moral_judgement']
selected_keys = [
    'quality',
    'toxicity',
    'humor',
    'creativity',
    'violence',
    'helpfulness',
    'not_appropriate',
    'hate_speech',
    'sexual_content',
    'fails_task',
    'political_content',
    'moral_judgement',
]
label_values = {}
likert_scale = 5
system_prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n\n"


def encode_labels(labels):
    items = []
    for key in selected_keys:
        if key in labels:
            value = labels[key]['value']
            items.append(f'{key}:{round(value*(likert_scale-1))}')
    return ','.join(items)


def parse_conversations(tree_obj):
    """ recusive function that returns all the sub converstaions in a list starting from node tree_obj

    Args:
        tree_obj (obj): current conversation node

    Returns:
        a list of sub conversation threads including the current conversation node
    """
    if 'prompt' in tree_obj:
        prompt_obj = tree_obj['prompt']
    elif 'text' in tree_obj and 'role' in tree_obj:
        prompt_obj = tree_obj
    else:
        return [[]]
    if prompt_obj['role'] == 'prompter':
        role = 'User'
    elif prompt_obj['role'] == 'assistant':
        role = 'Assistant'
    else:
        raise ValueError(f'unknown role {prompt_obj["role"]}')
    turn = {'value': prompt_obj['text'], 'from': role}
    if 'labels' in prompt_obj:
        # remove human labels
        # turn['human_labels'] = prompt_obj['labels']
        # for key in turn['human_labels']:
        #     value_set = label_values.get(key, set())
        #     value_set.add(turn['human_labels'][key]['value'])
        #     label_values[key] = value_set
        turn['label'] = encode_labels(prompt_obj['labels'])
    if 'lang' in prompt_obj:
        turn['lang'] = prompt_obj['lang'].split('-')[0]
        if turn['label'] == '':
            turn['label'] = f'lang:{turn["lang"]}'
        else:
            turn['label'] = turn['label'] + f',lang:{turn["lang"]}'
        value_set = label_values.get('lang', set())
        value_set.add(turn['lang'])
        label_values['lang'] = value_set
    all_conversations = []
    multiple_sub_threads = []
    for next_obj in prompt_obj['replies']:
        multiple_threads = parse_conversations(next_obj)
        multiple_sub_threads.extend(multiple_threads)
    if len(multiple_sub_threads) != 0:
        for sub_thread in multiple_sub_threads:
            all_conversations.append([turn] + sub_thread)
    else:
        all_conversations.append([turn])
    return all_conversations


def get_data_records(objs, mask_role, type):
    output = []
    for obj in objs:
        multi_conversations = parse_conversations(obj)
        for conversations in multi_conversations:
            if len(conversations) <= 1:
                # remove single turn conversations
                continue
            conversation_obj = {}
            conversation_obj['conversations'] = []
            conversation_obj['tree_id'] = obj['message_tree_id']
            conversation_obj['conversations'] = conversations
            conversation_obj['system'] = system_prompt
            conversation_obj['mask'] = mask_role
            conversation_obj['type'] = type
            output.append(conversation_obj)
    return output


def main(
    input_file='2023-04-12_oasst_all.trees.jsonl',
    output_file_prefix='oasst_output',
    mask_role='User',
    type='TEXT_TO_VALUE',
    split_ratio=0.95,
    seed=10,
):
    all_objs = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            all_objs.append(obj)
    random.seed(seed)
    random.shuffle(all_objs)
    train_num = int(len(all_objs) * split_ratio)
    train_objs = all_objs[:train_num]
    val_objs = all_objs[train_num:]
    train_records = get_data_records(train_objs, mask_role, type)
    val_records = get_data_records(val_objs, mask_role, type)

    with open(f'{output_file_prefix}_train.jsonl', 'w', encoding='utf-8') as f:
        for record in train_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    with open(f'{output_file_prefix}_val.jsonl', 'w', encoding='utf-8') as f:
        for record in val_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    for label in label_values:
        values = sorted(list(label_values[label]))
        print(f'{label} values: {values}')


if __name__ == "__main__":
    fire.Fire(main)
