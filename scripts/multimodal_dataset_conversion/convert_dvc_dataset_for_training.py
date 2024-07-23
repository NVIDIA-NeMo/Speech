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
This script is used to convert the DVC dataset to the format required by the model training script.
The DVC dataset should have the below structure:
{
    "1043215450": {          # video_name is the unique video file name (the extension should be .mp4)
        "duration": 125.0,
        "timestamps": [
            [0, 5], 
            [3, 9]
        ],
        "sentences": [                  # For custom caption or event localization task
            "Here is your caption 1",
            "Here is your caption 2",
        ],
        "events": [                   # For custom event task
            "Event 1",
            "Event 2",
        ]
    },
    ...
}

The converted dataset format is as follows:
[
    # 1st example: dense video captioning  (custom event or custom caption task)
    {
        "id": "xxxx",
        "video: "xxxx.mp4",
        "conversations":
        [
            {"from": "human", "value": "<video> \n"Provide a detailed description of the given video.Prepend each sentence with its start and end timestamps."},
            {"from": "gpt", "value": "<t1> <t2> Apply eyeshadow on the crease with brush <t3> <t4> Apply eyeshadow on the outer corner of eyes with brush"}
        ],
        "duration": 125.0
    },
    # 2nd example: event classification
    {
        "id": "xxxx",
        "video: "xxxx.mp4",
        "conversations":
        [
            {"from": "human", "value": "<video> \n"What is the action performed in this video?"},
            {"from": "gpt", "value": "brush hair"}
        ],
        "duration": 34.0
    },
    # 3rd example: event localization
    {
        "id": "xxxx",
        "video: "xxxx.mp4",
        "conversations":
        [
            {"from": "human", "value": "<video> \nWhen does brush hair happen in the video? Answer the question only using start and end timestamps."},
            {"from": "gpt", "value": "<t2> <t10>"}
        ],
        "duration": 34.0
    },
    ...
]

event_prompts.json and caption_prompts.json are optional.
Example:
event_prompts.json:
[
    "What is the action performed in this video?",
    "Can you highlight the action performed in this video?",
    ...
]

caption_prompts.json:
[
    "Provide a detailed description of the given video.",
    "Write a informative summary of the video.",
    ...
]

If the subtask is custom_caption, then the "events" field is not required.
If the subtask is custom_event, then the "sentences" field is not required.
If the subtask is event_localization, then the "events" field is not required.
If you want to do event classification, please set "disable_dvc_time_tokens" to true.

## Usage:
python convert_DVC_dataset.py \
    --input_dvc_dataset /path/to/dvc_dataset.json \
    --output_file /path/to/output_dataset.json \
    --video_path_prefix /path/to/video/folder/ \
    --subtask custom_caption \   # or custom_event
    --event_prompts /path/to/event_prompts.json \
    --caption_prompts /path/to/caption_prompts.json \
    --num_time_tokens 100 \
    --data_multiplier 1 \

"""

import argparse
import json
import os
import random

import numpy as np

# from nemo.collections.multimodal.data.neva.conversation import TIME_TOKEN_TEMPLATE
TIME_TOKEN_TEMPLATE = "<t{t}>"
caption_prompts = [
    "Provide a detailed description of the given video.",
    "Describe the provided video in detail.",
    "Summarize the visual content of the video.",
    "Write a informative summary of the video.",
]

event_prompts = [
    "What is the action performed in this video?",
    "Can you highlight the action performed in this video?",
    "What is the main event or action captured in this video?",
    "Could you summarize the sequence of events depicted in this video?",
]

time_prompts = [
    "Each sentence should begin with the start and end timestamps.",
    "At the beginning of each sentence, include the start and end timestamps.",
    "Prepend each sentence with its start and end timestamps.",
]

event_loc_prompts = [
    "When does \"%s\" happen in the video?",
    "At what point in the video does \"%s\" happen?",
    "When is \"%s\" depicted in the video?",
    "At what time in the video does \"%s\" take place?",
]

event_loc_time_prompts = [
    "Answer the question only using start and end timestamps.",
    "Provide a response using only start and end timestamps.",
    "Convey your answer using start and end timestamps exclusively.",
]


def convert(
    input_dvc_dataset,
    output_dataset,
    video_path_prefix,
    num_time_tokens,
    disable_dvc_time_tokens,
    prompts,
    time_prompts,
    field,
    ext=".mp4",
    subtask="custom_caption",
    data_multiplier=1,
):

    def time_to_string(time):
        # time is normalized in [0, 1]
        max_offset = float(num_time_tokens - 1)
        time = int(np.round(max_offset * time))
        return TIME_TOKEN_TEMPLATE.format(t=time)

    def get_prompt(subtask, prompts, time_prompts, sentence=None):
        if subtask == "event_localization":
            desc_prompt = random.choice(prompts)
            time_prompt = random.choice(time_prompts)
            sentence = sentence.strip().rstrip('.')
            task_prompt = (desc_prompt % sentence) + ' ' + time_prompt
        else:
            if disable_dvc_time_tokens:
                task_prompt = random.choice(prompts)
            else:
                task_prompt = random.choice(prompts) + ' ' + random.choice(time_prompts)

        return '<video>' + ' \n' + task_prompt

    dvc_dataset = {}
    with open(input_dvc_dataset, "r") as f:
        dvc_dataset = json.load(f)

    list_data_dict = []
    for i in range(data_multiplier):
        for video_name, video_info in dvc_dataset.items():
            out = {}
            video_file = video_name + ext
            if video_path_prefix is not None:
                # do a sanity check to see if the video file exists
                video_path = os.path.join(video_path_prefix, video_file)
                if not os.path.exists(video_path):
                    continue
            vid = video_name.split(".")[0]
            video = video_file
            texts = video_info[field]
            duration = video_info["duration"]
            timestamps = video_info["timestamps"]
            if len(texts) == 0:
                continue
            if subtask == "event_localization":
                # only pick one sentence and timestamps
                idx = random.choice(range(len(texts)))
                # rng = np.random.RandomState()
                # idx = rng.choice(list(range(len(timestamps))))
                texts = [texts[idx]]
                timestamps = [timestamps[idx]]
            gpt_value = ""
            for i, text in enumerate(texts):
                start, end = float(timestamps[i][0]), float(timestamps[i][1])
                start, end = start / duration, end / duration
                start_str = time_to_string(start)
                end_str = time_to_string(end)
                seg_caption = text.strip()
                if subtask == "event_localization":
                    gpt_value = f"{start_str} {end_str}"
                else:
                    if disable_dvc_time_tokens:
                        gpt_value += f"{seg_caption} "
                    else:
                        gpt_value += f"{start_str} {end_str} {seg_caption} "

            convo = []
            if gpt_value == "":
                continue
            if subtask == "event_localization":
                convo.append({"from": "human", "value": get_prompt(subtask, prompts, time_prompts, texts[0])})
            else:
                convo.append({"from": "human", "value": get_prompt(subtask, prompts, time_prompts)})
            convo.append({"from": "gpt", "value": gpt_value.strip()})
            out["id"] = vid
            out["video"] = video
            out["conversations"] = convo
            out["durations"] = duration
            list_data_dict.append(out)

    with open(output_dataset, "w") as f:
        json.dump(list_data_dict, f, indent=4)


def load_prompts(prompts_path):
    prompts = []
    with open(prompts_path, "r") as f:
        prompts = json.load(f)
    assert len(prompts) > 0, "Event prompts should not be empty"
    return prompts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dvc_dataset", type=str, required=True)
    parser.add_argument("--output_file", default="dvc_train.json", type=str, required=True)
    parser.add_argument(
        "--subtask", choices=["custom_event", "custom_caption", "event_localization"], type=str, required=True
    )
    parser.add_argument("--video_path_prefix", default=None, type=str, required=False)
    parser.add_argument(
        "--event_prompts", type=str, default=None, required=False, help="Path to the event prompt json file; Optional"
    )
    parser.add_argument(
        "--caption_prompts",
        type=str,
        default=None,
        required=False,
        help="Path to the caption prompt json file; Optional",
    )
    parser.add_argument(
        "--num_time_tokens", type=int, default=100, help="Number of time tokens to use for time tokens"
    )
    parser.add_argument("--disable_dvc_time_tokens", action="store_true")
    parser.add_argument("--data_multiplier", type=int, default=1, help="Number of times to repeat the dataset")
    args = parser.parse_args()

    # load event_prompts and caption_prompts
    custom_event_prompts = []
    if args.event_prompts:
        custom_event_prompts = load_prompts(args.event_prompts)
    else:
        custom_event_prompts = event_prompts

    custom_caption_prompts = []
    if args.caption_prompts:
        custom_caption_prompts = load_prompts(args.caption_prompts)
    else:
        custom_caption_prompts = caption_prompts

    t_prompts = time_prompts
    prompts = []
    if args.subtask == "custom_event":
        prompts = custom_event_prompts
    elif args.subtask == "custom_caption":
        prompts = custom_caption_prompts
    elif args.subtask == "event_localization":
        prompts = event_loc_prompts
        t_prompts = event_loc_time_prompts

    field = "events" if args.subtask == "custom_event" else "sentences"

    convert(
        args.input_dvc_dataset,
        args.output_file,
        args.video_path_prefix,
        args.num_time_tokens,
        args.disable_dvc_time_tokens,
        prompts,
        t_prompts,
        field,
        ext=".mp4",
        subtask=args.subtask,
        data_multiplier=args.data_multiplier,
    )


if __name__ == "__main__":
    main()
