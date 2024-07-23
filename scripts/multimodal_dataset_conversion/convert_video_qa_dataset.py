# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
#

"""
The script is used to convert the QA dataset to the format required by the model training script.

qa dataset:

[
  {
    "video_id": "-4RXOT_UfpM_1",
    "conversations": [
      {
        "role": "user",
        "content": "Provide a detailed description of the makeup tutorial video."
      },
      {
        "role": "assistant",
        "content": "<0s> <34.00066733400067s> The video does not describe any specific event."
      },
      {
        "role": "user",
        "content": "Is there any event described in the video between <0s> and <34.00066733400067s>?"
      },
      {
        "role": "assistant",
        "content": "No, there is no event described in the video between <0s> and <34.00066733400067s>."
      }
    ],
    "duration": 34.00066733400067,
    "source": "video"
  },
  ....
]

to nemo video training dataset format and convert the timestamps to the format required by the model training script.

[
        "id": "-4RXOT_UfpM_11",
        "video": "-4RXOT_UfpM_11.mp4",
        "conversations": [
            {
                "from": "human",
                "value": "<video> \nWhat is the first task performed in the video, and when does it start and end?"
            },
            {
                "from": "gpt",
                "value": "The first task is to apply highlight on the cheekbone, nose, and mouth peak with a brush, which starts at <t5> and ends at <t33>."
            },
            {
                "from": "human",
                "value": "Write a step-by-step guide of the makeup tutorial video."
            },
            {
                "from": "gpt",
                "value": "Between <t5> and <t33>, apply highlight on the cheekbone, nose, and mouth peak with a brush. From <t33> to <t38>, use a pencil to draw the eyebrows. Between <t76> and <t82>, use a lipliner to draw the lip line. From <t82> to <t96>, apply lipstick on the lips."
            },
            {
                "from": "human",
                "value": "What is the next task performed after applying highlight, and when does it start and end?"
            },
            {
                "from": "gpt",
                "value": "The next task is to use a pencil to draw the eyebrows, which starts at <t33> and ends at <t38>."
            },
            {
                "from": "human",
                "value": "When does the task of drawing the lip line with lipliner start and end?"
            },
            {
                "from": "gpt",
                "value": "The task of drawing the lip line with lipliner starts at <t76> and ends at <t82>."
            },
            {
                "from": "human",
                "value": "What is the task performed immediately after drawing the lip line, and when does it start and end?"
            },
            {
                "from": "gpt",
                "value": "The task performed immediately after drawing the lip line is to apply lipstick on the lips, which starts at <t82> and ends at <t96>."
            }
        ],
        "duration": 103.003003003003
    },
    ....
]

"""

import argparse
import json
import re
import numpy as np

# from nemo.collections.multimodal.data.neva.conversation import TIME_TOKEN_TEMPLATE
TIME_TOKEN_TEMPLATE = "<t{t}>"


def process(value, duration: float, num_time_tokens: int = 100):
    def time_to_string(time):
        # time is normalized in [0, 1]
        max_offset = float(num_time_tokens - 1)
        time = int(np.round(max_offset * time))
        return TIME_TOKEN_TEMPLATE.format(t=time)

    def repl(match):
        value = float(match.group(1)) / duration
        return time_to_string(value) + f"<!|t{value}t|!>"

    value = re.sub(r"<([\d.]{1,20})s>", repl, value)
    value = re.sub(r"\s([\d.]{1,20})s[\s\.,>]", repl, value)
    value = re.sub(r"\s([\d.]{1,20}) seconds", repl, value)
    value = re.sub(r"\s([\d.]{1,20}) second", repl, value)

    # This is to remove the timestamps from the text
    value = re.sub(r"<!\|t([\d.]+)t\|!>", "", value)
    return value.strip()


def convert(qa_dataset, output_file, num_time_tokens, ext=".mp4"):
    with open(qa_dataset, 'r') as f:
        qa_data = json.load(f)

    role_mapping = {
        "user": "human",
        "assistant": "gpt",
        "system": "system",
    }

    list_data_dict = []
    for sample in qa_data:
        new_sample = {}
        id = sample['video_id']
        video = id + ext
        conversations = []
        for idx, conversation in enumerate(sample['conversations']):
            if 'role' in conversation and 'content' in conversation:
                new_role = role_mapping[conversation['role']]
                new_content = conversation['content']
            elif 'from' in conversation and 'value' in conversation:
                new_role = role_mapping[conversation["from"]]
                new_content = conversation["value"]
            else:
                raise ValueError("Invalid conversation format")

            new_content = process(new_content, sample["duration"], num_time_tokens)
            if idx == 0 and new_role == "human":
                new_content = "<video> \n" + new_content
            conversations.append({"from": new_role, "value": new_content})

        new_sample['id'] = id
        new_sample['video'] = video
        new_sample['conversations'] = conversations
        new_sample["duration"] = sample["duration"]
        list_data_dict.append(new_sample)

    with open(output_file, 'w') as f:
        json.dump(list_data_dict, f, indent=4)


def main():
    parser = argparse.ArgumentParser(description='Convert QA dataset to NeMo video training dataset format')
    parser.add_argument('--qa_dataset', type=str, required=True, help='QA dataset in json format')
    parser.add_argument('--output_file', type=str, required=True, help='Output file in json format')
    parser.add_argument('--num_time_tokens', type=int, default=100, help='Number of time tokens')
    args = parser.parse_args()

    convert(args.qa_dataset, args.output_file, args.num_time_tokens)


if __name__ == "__main__":
    main()
