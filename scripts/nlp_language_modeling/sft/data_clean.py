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

"""script to clean the data for sft chat dataset. It will remove the records if all the tokens are masked due to truncation by sequence length
Example usage:

MPT-7B:
    python data_clean.py --dataset_file /dataset/INPUT.jsonl --output_file /dataset/OUTPUT.jsonl --library huggingface --model_name EleutherAI/gpt-neox-20b --seq_len 4096
NeMo GPT:
    python data_clean.py --dataset_file /dataset/INPUT.jsonl --output_file /dataset/OUTPUT.jsonl --library sentencepiece --model_file sentencepiece.model --seq_len 4096
"""


import argparse
import json
from nemo.collections.nlp.data.language_modeling.megatron.gpt_sft_chat_dataset import GPTSFTChatDataset
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer


def data_clean(
    dataset_file,
    output_file,
    seq_len=4096,
    library='huggingface',
    model_name='EleutherAI/gpt-neox-20b',
    tokenizer_model=None,
):
    tokenizer = get_nmt_tokenizer(
        library=library, model_name=model_name, tokenizer_model=tokenizer_model, use_fast=True
    )
    if library == 'huggingface':
        tokenizer.add_special_tokens({'additional_special_tokens': ['<extra_id_0>', '<extra_id_1>', '<extra_id_2>']})
    d = GPTSFTChatDataset(dataset_file, tokenizer, seq_len, 1, hf_dataset=True)
    total_records = len(d)
    removed_ids = set()
    for i in range(total_records):
        if i % 1000 == 0:
            print(i)
        try:
            if d[i]['mask'][: seq_len + 1].sum().item() == 0:
                removed_ids.add(i)
                print(f'removed {i}')
                continue
        except:
            removed_ids.add(i)
            print(f'Exception removed {i}')
    with open(dataset_file, 'r', encoding='utf-8') as f:
        with open(output_file, 'w', encoding='utf-8') as o:
            for i, line in enumerate(f):
                if i in removed_ids:
                    continue
                obj = json.loads(line)
                o.write(json.dumps(obj, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_file", type=str, required=True, default='/dataset/input.jsonl')
    parser.add_argument(
        "--model_file", type=str, required=False, default=None, help="Path to the sentence piece model file."
    )
    parser.add_argument(
        "--library",
        type=str,
        required=False,
        default='huggingface',
        help="tokenizer library, huggingface or sentencepiece",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=False,
        default='EleutherAI/gpt-neox-20b',
        help="huggingface tokenizer model name",
    )
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--seq_len", type=int, required=False, default=4096)
    args = parser.parse_args()
    data_clean(
        dataset_file=args.dataset_file,
        output_file=args.output_file,
        seq_len=args.seq_len,
        library=args.library,
        model_name=args.model_name,
        tokenizer_model=args.model_file,
    )
