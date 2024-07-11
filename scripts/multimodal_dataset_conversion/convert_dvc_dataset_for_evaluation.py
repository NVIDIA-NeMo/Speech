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
This script is used to convert the DVC dataset to the format required by the model evaluation for RTL task.
The DVC dataset should have the below structure:
{
    "-4RXOT_UfpM_3": {          # video_name is the unique video file name, extention is .mp4
        "duration": 118.01801801801803,
        "timestamps": [
            [5, 58], 
            [66, 82],
            [82, 96]
        ],
        "sentences": [
            "Apply eyeshadow on the lower area then crease with brush",
            "Apply eyeshadow on the outer corner of eyes with brush",
            "Apply eyeshadow on the outer half of eyes with brush",
        ]
    },
    ...
}

The converted format will be as follows:
[
    {
        "video": "-4RXOT_UfpM_3.mp4",
        "question_id": "-4RXOT_UfpM_3_0",
        "question": "When does \"Apply eyeshadow on the lower area then crease with brush\" happen in the video? Provide a response using only start and end timestamps.",
        "ref_answer": "<5> <58> Apply eyeshadow on the lower area then crease with brush",
        "duration": 118.01801801801803
    },
    {
        "video": "-4RXOT_UfpM_3.mp4",
        "question_id": "-4RXOT_UfpM_3_1",
        "question": "When is \"Apply eyeshadow on the outer corner of eyes with brush\" depicted in the video? Convey your answer using start and end timestamps exclusively.",
        "ref_answer": "<66> <82> Apply eyeshadow on the outer corner of eyes with brush",
        "duration": 118.01801801801803
    },
    {
        "video": "-4RXOT_UfpM_3.mp4",
        "question_id": "-4RXOT_UfpM_3_2",
        "question": "When does \"Apply eyeshadow on the outer half of eyes with brush\" happen in the video? Provide a response using only start and end timestamps.",
        "ref_answer": "<82> <96> Apply eyeshadow on the outer half of eyes with brush",
        "duration": 118.01801801801803
    },
    .....
]

For each sentence in the sentences list, we will generate one question for it and the answer will be the sentence itself with the timestamps.
USAGE:
python convert_dvc_dataset_for_evaluation.py --input <input_file> --output_file <output_file> --ratio <sampling_ratio>

"""

import argparse
import json
import os
import random


class RTLConverter:
    def __init__(self, input_file, output_file, sample_ratio, ext):
        self.input_file = input_file
        self.output_file = output_file
        self.sample_ratio = sample_ratio
        self.desc_prompts = [
            "When does \"%s\" happen in the video?",
            "At what point in the video does \"%s\" happen?",
            "When is \"%s\" depicted in the video?",
            "At what time in the video does \"%s\" take place?",
        ]
        self.time_prompts = [
            "Answer the question only using start and end timestamps.",
            "Provide a response using only start and end timestamps.",
            "Convey your answer using start and end timestamps exclusively.",
        ]
        self.ext = ext

    def convert(self):
        converted_data = []

        # Load JSON data
        with open(self.input_file, 'r') as file:
            data = json.load(file)

        # Fix random seed for reproducibility
        random.seed(42)

        # Randomly sample entries based on the sample ratio
        vid_list = list(data.keys())
        sampled_vids = random.sample(vid_list, k=int(len(vid_list) * self.sample_ratio))

        # Iterate through sampled entries
        for vid in sampled_vids:
            details = data[vid]
            duration = details['duration']
            timestamps = details['timestamps']
            sentences = details['sentences']

            # Iterate through sentences
            for i, sentence in enumerate(sentences):
                question_id = f"{vid}_{i}"
                desc_prompt = random.choice(self.desc_prompts)
                time_prompt = random.choice(self.time_prompts)
                start_time, end_time = timestamps[i]
                answer = f"<{start_time}> <{end_time}> {sentence}"

                # Construct question
                question = (desc_prompt % sentence) + ' ' + time_prompt

                # Create entry in converted data
                converted_data.append(
                    {
                        "video": vid + self.ext,
                        "question_id": question_id,
                        "question": question,
                        "ref_answer": answer,
                        "duration": duration,
                    }
                )

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)

        # Write converted data to output file
        with open(self.output_file, 'w') as file:
            json.dump(converted_data, file, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Convert makeup QA JSON format")
    parser.add_argument("--input", help="Input DVC JSON file", required=True)
    parser.add_argument("--output_file", help="Output file", default="rtl_eval.json", required=True)
    parser.add_argument("--ratio", help="Sampling ratio between 0 and 1", type=float, default=1.0, required=False)
    parser.add_argument("--ext", help="Extension of the video files", default=".mp4", required=False)
    args = parser.parse_args()

    if args.ratio < 0 or args.ratio > 1:
        raise ValueError("Sampling ratio must be between 0 and 1")

    converter = RTLConverter(args.input, args.output_file, args.ratio, args.ext)
    converter.convert()


if __name__ == "__main__":
    main()
